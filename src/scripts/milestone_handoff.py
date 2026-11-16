"""在里程碑保存窗口请求安全停止，确认释放后启动独立短SFT；不向忙碌内核执行代码。"""
import argparse
import fcntl
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)


def training_updates(rows):
    """兼容预训练和既有SFT指标；不能把验证记录当作实际参数更新。"""
    return [row for row in rows if type(row.get('step')) is int and row['step'] > 0
            and (row.get('event') == 'train' or (
                row.get('event') is None and {'loss', 'tokens', 'lr'} <= row.keys()
                and isinstance(row['tokens'], (int, float)) and row['tokens'] > 0))]


def should_request_stop(status, milestone):
    return (status.get('stage') == 'continued_pretrain' and status.get('status') == 'backing_up'
            and isinstance(status.get('completed_steps'), int) and status['completed_steps'] >= milestone)


def validate_origin(manifest, plan):
    origin = plan['origin']
    if manifest.get('step') != origin['step'] or manifest.get('drive_folder_id') != origin['drive_folder_id']:
        raise ValueError('来源不是指定里程碑完整断点')
    files = manifest.get('files', {})
    if not {'training_state.pt', 'model.safetensors', 'tokenizer.json'} <= files.keys():
        raise ValueError('来源缺少完整预训练状态或权重')
    if files['training_state.pt']['bytes'] < 1_000_000_000:
        raise ValueError('不能使用仅评分状态的best替代完整预训练断点')
    expected = plan['input_sha256']['/content/river-valley-run/tokenizer.json']
    if files['tokenizer.json']['sha256'] != expected:
        raise ValueError('诊断tokenizer与来源不一致')


def clean_index(index, evidence, stage):
    cleaned = dict(index)
    for key in ('verified_step', 'verified_at', 'last_hourly_check', 'cloud_status_last_observed',
                'server_verified_durable_step', 'server_verified_best_step', 'server_backup_verified_at'):
        cleaned.pop(key, None)
    cleaned.update(stage=stage, evidence_dir=str(evidence.relative_to(ROOT)), status='preparing_milestone_probe',
                   cpt_evidence_dir=index['evidence_dir'], previous_cpt_session=index['session'],
                   next_action='诊断完成后复核开发回答，再从CPT最近完整成对断点续训；禁止用诊断SFT权重接CPT。')
    return cleaned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpt-evidence', required=True)
    parser.add_argument('--probe-evidence', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--probe-session', required=True)
    parser.add_argument('--wait-hours', type=float, default=3,
                        help='等待预训练里程碑的最长小时数；与诊断实例的3小时保护分开')
    args = parser.parse_args()
    if not 0 < args.wait_hours <= 48:
        parser.error('等待里程碑时间必须大于0且不超过48小时')
    cpt, probe = Path(args.cpt_evidence).resolve(), Path(args.probe_evidence).resolve()
    lock = (probe / 'handoff.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = json.loads((probe / 'plan.json').read_text()); milestone = plan['origin']['step']
    index_path = cpt.parent / 'current_attempt.json'
    from colab_cli.common import state
    from colab_cli.contents import ContentsClient
    from launch_version import reliable_contents_class, usage
    from drive_checkpoint_store import DriveStore
    cli = str(Path(sys.executable).with_name('colab'))
    session = state.get_session(args.session); endpoint = session.endpoint
    initial_index = json.loads(index_path.read_text())
    if initial_index['session'] != args.session or Path(initial_index['evidence_dir']).resolve() != cpt:
        raise ValueError('当前运行索引不匹配，拒绝操作旧会话')
    started = time.time(); requested = False

    def evidence(status, **values):
        value = {'time': time.time(), 'status': status, 'session': args.session,
                 'endpoint': endpoint, 'milestone': milestone, **values}
        (probe / 'handoff_status.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    evidence('waiting_for_checkpoint_backup')
    while time.time() - started < args.wait_hours * 3600:
        index = json.loads(index_path.read_text())
        if index['session'] != args.session:
            raise ValueError('运行索引已被其他操作切换，停止自动接续')
        try:
            assignments = state.client.list_assignments()
        except Exception as error:
            log.warning('分配查询暂时失败：类型=%s；不把网络失败当作实例释放', type(error).__name__)
            time.sleep(15)
            continue
        if not any(item.endpoint == endpoint for item in assignments):
            break
        try:
            session = state.get_session(args.session)
            response = requests.get(session.url.rstrip('/') + '/api/contents/content/river-valley-run/status.json',
                params={'authuser': '0', 'colab-runtime-proxy-token': session.token, 'format': 'text', 'content': 1}, timeout=(8, 20))
            response.raise_for_status(); status = json.loads(response.json()['content'])
            if not requested and should_request_stop(status, milestone):
                stop = probe / 'STOP_FOR_DIAGNOSTIC'
                stop.write_text(f'里程碑{milestone}已结束训练，完成当前备份后释放，接独立短SFT诊断\n')
                reliable_contents_class(ContentsClient)(session).upload(str(stop), '/content/river-valley-run/STOP')
                requested = True
                log.info('已在备份窗口请求停止：实际步数=%d 里程碑=%d', status['completed_steps'], milestone)
            evidence('waiting_for_release' if requested else 'waiting_for_checkpoint_backup', cloud_status=status)
        except requests.RequestException as error:
            log.warning('只读状态暂时失败：类型=%s，保留当前实例', type(error).__name__)
        time.sleep(15)
    else:
        raise RuntimeError('等待里程碑超时，未自动新开GPU；留给巡检调查')
    # 释放必须同时由真实分配列表和账单确认，避免短暂接口失联造成并发GPU。
    for _ in range(30):
        account = usage(cli); assignments = state.client.list_assignments()
        if account['count'] == 0 and account['rate'] == 0 and not assignments:
            break
        time.sleep(10)
    else:
        raise RuntimeError('未确认全部实例释放和计费归零，拒绝申请替代实例')
    b = json.loads((cpt / 'bootstrap.json').read_text())
    store = DriveStore(b['drive_root_id'], str(Path.home() / '.config/colab-cli/token.json'), b['quota_project'])

    def remote_json(file_id):
        return store.request('GET', 'https://www.googleapis.com/drive/v3/files/' + file_id, params={'alt': 'media'}).json()

    mid = store.child(plan['origin']['drive_folder_id'], 'BACKUP_COMPLETE.json')
    if not mid:
        raise ValueError('来源里程碑未持久化，禁止启动诊断')
    origin = remote_json(mid); validate_origin(origin, plan); store.verify(origin)
    pair_id = store.child(store.root_id, 'continued_pretrain_RESUME.json')
    if not pair_id:
        raise ValueError('预训练缺少成对完整恢复记录')
    pair = remote_json(pair_id)
    if pair['stage'] != 'continued_pretrain' or pair['full']['step'] < milestone or 'training_state.pt' not in pair['full']['files']:
        raise ValueError('最新预训练恢复状态不完整')
    store.verify(pair['full']); store.verify(pair['best'])
    receipt = {'time': time.time(), 'origin': origin, 'cpt_resume_pair': pair, 'account': account,
               'previous_session': args.session, 'endpoint': endpoint, 'release_verified': True,
               'note': '来源及预训练恢复状态均服务器验证；短SFT只初始化模型，新建Adam，不回写预训练状态'}
    (probe / 'origin_and_release_verified.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    if account['balance'] <= 20:
        raise RuntimeError('余额不足以覆盖短SFT与保存，保留来源并停止')
    new_index = clean_index(json.loads(index_path.read_text()), probe, plan['stages'][0]['id'])
    new_index['pending_milestone_probe'] = dict(new_index.get('pending_milestone_probe', {}), status='origin_verified_launching')
    index_path.write_text(json.dumps(new_index, ensure_ascii=False, indent=2) + '\n')
    evidence('launching_probe', release_account=account)
    subprocess.run([sys.executable, str(ROOT / 'src/scripts/launch_version.py'), '--evidence', str(probe),
                    '--session', args.probe_session, '--hours', '3'], cwd=ROOT, check=True, timeout=2100)
    evidence('probe_launched_waiting_for_step_growth')
    # 启动命令成功还不代表训练成功；只读实际步数确认，不向训练单元格注入exec。
    for _ in range(80):
        active = state.get_session(args.probe_session, ignore_missing_session=True)
        if active is None:
            raise RuntimeError('诊断实例消失，启动未获步数增长验证')
        try:
            response = requests.get(active.url.rstrip('/') + '/api/contents/content/river-valley-run/checkpoints/' + plan['stages'][0]['id'] + '/metrics.jsonl',
                params={'authuser': '0', 'colab-runtime-proxy-token': active.token, 'format': 'text', 'content': 1}, timeout=(8, 20))
            if response.status_code == 200:
                rows = [json.loads(line) for line in response.json()['content'].splitlines() if line.strip()]
                trained = training_updates(rows)
                if trained:
                    evidence('probe_running_verified', latest_train=trained[-1])
                    index = json.loads(index_path.read_text())
                    if index['session'] != args.probe_session:
                        raise ValueError('诊断启动后索引已切换，拒绝覆盖')
                    index.update(status='running_verified', verified_step=trained[-1]['step'], verified_at=time.time(),
                                 next_action='完成160步短SFT后读取并评分100条开发回答，核对释放，再从CPT最近完整状态续训。')
                    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
                    log.info('短SFT实际步数已增长：阶段=%s 步数=%d', plan['stages'][0]['id'], trained[-1]['step'])
                    return
        except requests.RequestException as error:
            log.warning('诊断步数读取暂时失败：类型=%s', type(error).__name__)
        time.sleep(15)
    raise RuntimeError('诊断启动尚未确认步数增长；保留云端保护和证据，交巡检核实')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        main()
    except Exception as error:
        log.error('里程碑交接未完成：类型=%s 描述=%s', type(error).__name__, str(error) if not isinstance(error, requests.RequestException) else '网络请求失败，地址已隐藏')
        raise SystemExit(1)
