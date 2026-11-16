"""在材料冻结后申请唯一实例，先部署保护，再启动真实训练单元格和只读订阅。"""
import argparse
import math
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)


def reliable_contents_class(base):
    class ReliableContents(base):
        def _request(self, method, path, params=None, json_data=None):
            url = self.base_url + '/api/contents/' + quote(path.strip('/'), safe='/')
            query = {'authuser': '0', 'colab-runtime-proxy-token': self.token, **(params or {})}
            for attempt in range(8):
                try:
                    response = requests.request(method, url, params=query, json=json_data, timeout=(10, 90))
                    if response.status_code == 404:
                        raise FileNotFoundError(path)
                    if response.status_code in (400, 409) and json_data == {'type': 'directory'}:
                        existing = self._request('GET', path, params={'content': 0})
                        if existing.get('type') == 'directory':
                            return existing
                    response.raise_for_status()
                    return None if method == 'DELETE' else response.json()
                except requests.RequestException as error:
                    status = error.response.status_code if getattr(error, 'response', None) is not None else None
                    log.warning('文件接口暂时失败：操作=%s 路径=%s HTTP=%s 类型=%s 尝试=%d',
                                method, path, status, type(error).__name__, attempt + 1)
                    if attempt == 7 or status is not None and status not in (408, 429, 500, 502, 503, 504):
                        raise
                    time.sleep(min(10, 2 ** attempt))
    return ReliableContents


def usage(cli):
    raw = subprocess.run([cli, 'usage'], capture_output=True, text=True, check=True, timeout=60).stdout.splitlines()
    return {'balance': float(next(x.split(':')[1].split()[0] for x in raw if x.startswith('Current balance:'))),
            'rate': float(next(x.split(':')[1].split('/')[0] for x in raw if x.startswith('Usage rate:'))),
            'count': int(next(x.split(':')[1] for x in raw if x.startswith('Active assignments:')))}


def startup_retry_allowed(log_text, kernels, file_statuses):
    """只重试代码提交前的连接丢失；忙碌内核、已有流水线或网络未知都不能重发。"""
    return ("Connection was lost." in log_text and "Ensure we are in /content" in log_text
            and bool(kernels) and all(item.get('execution_state') == 'idle' for item in kernels)
            and set(file_statuses) == {'pipeline.pid', 'pipeline.log', 'status.json'}
            and all(code == 404 for code in file_statuses.values()))


def confirm_pipeline_start(session, evidence, process, command, log_offset, wait_seconds=180, max_retries=2):
    """读取云端启动证据；已知提交前故障在原实例有限重试，其他不确定情况保留保护。"""
    deadline = time.monotonic() + wait_seconds
    attempts = 0
    logfile = evidence / 'training_cell.log'
    params = {'authuser': '0', 'colab-runtime-proxy-token': session.token}

    def record(status, **extra):
        result = dict(time=time.time(), status=status, retries=attempts,
                      endpoint=session.endpoint, client_pid=process.pid, **extra)
        (evidence / 'pipeline_startup.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        return result

    while time.monotonic() < deadline:
        try:
            statuses = {}
            for name in ('pipeline.pid', 'pipeline.log', 'status.json'):
                response = requests.get(session.url.rstrip('/') + '/api/contents/content/river-valley-run/' + name,
                                        params={**params, 'content': 0}, timeout=(8, 12))
                statuses[name] = response.status_code
            if any(code == 200 for code in statuses.values()):
                log.info('已观察到云端流水线启动证据：endpoint=%s 文件状态=%s；仍需实际步数确认训练',
                         session.endpoint, statuses)
                return record('pipeline_observed', files=statuses)
            response = requests.get(session.url.rstrip('/') + '/api/kernels', params=params, timeout=(8, 12))
            response.raise_for_status()
            kernels = response.json()
            log_text = logfile.read_bytes()[log_offset:].decode('utf-8', errors='replace')
            if startup_retry_allowed(log_text, kernels, statuses):
                # CLI 抛错后的网络线程可能阻止退出；只终止本函数持有的失效客户端。
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                if attempts >= max_retries:
                    record('preflight_failed_idle', files=statuses)
                    raise RuntimeError('训练提交前连接故障重试耗尽；已确认内核空闲且没有流水线')
                attempts += 1
                log.warning('提交前连接丢失，已确认内核idle且流水线不存在；原实例重试 %d/%d：endpoint=%s',
                            attempts, max_retries, session.endpoint)
                with logfile.open('a', encoding='utf-8') as output:
                    output.write(f'\n原实例启动客户端重试：第{attempts}次\n')
                    output.flush()
                    log_offset = logfile.stat().st_size
                    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                (evidence / 'training_cell.pid').write_text(str(process.pid))
                record('preflight_retry_submitted', files=statuses)
        except requests.RequestException as error:
            log.warning('启动证据读取暂时失败：类型=%s；不重复提交、不据此释放实例', type(error).__name__)
        time.sleep(5)
    log.warning('启动确认等待结束，状态仍待核实；保留实例保护和只读监控，不把启动当作训练成功')
    return record('awaiting_pipeline_evidence')


def prepare_backup_folders(store, plan):
    """申请 GPU 前准备所有计划内的备份目录，提前发现创建权限问题。"""
    from train_version import stage_segments
    for stage in plan['stages']:
        target = stage['target_steps']
        ends = set(stage_segments(0, target, stage.get('backup_interval', target))) | set(stage.get('milestone_steps', []))
        interval = stage['config'].get('eval_interval', target)
        best_steps = ends | set(range(0, target, interval))
        for name, steps in [(stage['id'], ends), (stage['id'] + '_best', best_steps)]:
            parent = store.folder(store.root_id, name)
            for step in sorted(steps):
                store.folder(parent, f'step_{step:06d}')
            log.info('备份目录准备完成：阶段=%s 目录数=%d；空目录不代表完整断点', name, len(steps))


def check_launch_balance(balance, plan):
    """默认保留原门槛；有明确成本估算的小实验按实际余额及保存余量检查。"""
    estimate = plan.get('budget_estimate')
    if estimate is None:
        minimum = 20
    else:
        required = ('setup_training_evaluation_cu', 'save_release_reserve_cu')
        if any(type(estimate.get(k)) not in (int, float) or not math.isfinite(estimate[k]) or estimate[k] <= 0 for k in required):
            raise ValueError('小实验成本估算缺失或非法')
        if estimate['save_release_reserve_cu'] < 4 or not estimate.get('basis'):
            raise ValueError('小实验缺少保存余量或估算依据')
        minimum = sum(estimate[k] for k in required)
    if not math.isfinite(balance) or balance <= minimum:
        raise RuntimeError(f'余额不足以覆盖冻结计划和保存：实际={balance} CU，需要大于{minimum} CU')
    log.info('开机余额检查通过：实际=%.2f CU 所需余量门槛=%.2f CU；保留独立截止保护', balance, minimum)


def balance_policy(plan):
    """小实验显式冻结保存阈值，历史计划继续沿用原来的 10/2 CU。"""
    policy = plan.get('balance_policy', {'soft_balance_cu': 10, 'hard_balance_cu': 2})
    soft, hard = policy.get('soft_balance_cu'), policy.get('hard_balance_cu')
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (soft, hard)):
        raise ValueError('余额保护阈值必须为有限数')
    if soft < 4 or hard < 2 or soft <= hard:
        raise ValueError('余额保护至少预留 4 CU 保存、2 CU 释放，且保存阈值须大于释放阈值')
    estimate = plan.get('budget_estimate', {})
    if soft < estimate.get('save_release_reserve_cu', 4):
        raise ValueError('余额保存阈值不能小于冻结计划的保存余量')
    return {'soft_balance_cu': soft, 'hard_balance_cu': hard}


def recalibrate_limits(limits, balance, rate, now=None):
    """分配后用真实费率收紧截止，不因准备过程或重连重新延长额度。"""
    if not math.isfinite(balance) or not math.isfinite(rate) or rate <= 0:
        raise ValueError('重新计算截止需要真实有限余额与正费率')
    now = time.time() if now is None else now
    updated = dict(limits, observed_balance_cu=balance, observed_rate_cu_per_hour=rate, recalibrated_at=now)
    for kind, default in [('soft', 10), ('hard', 2)]:
        reserve = limits.get(kind + '_balance_cu', default)
        updated[kind + '_deadline'] = min(limits[kind + '_deadline'], now + max(0, balance - reserve) / rate * 3600)
    log.info('按实际账单校准独立截止：余额=%.2f CU 费率=%.2f CU/h 保存=%s 释放=%s',
             balance, rate, updated['soft_deadline'], updated['hard_deadline'])
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--hours', type=float, default=3)
    parser.add_argument('--reuse-idle-session', action='store_true')
    args = parser.parse_args()
    evidence = Path(args.evidence).resolve()
    bootstrap = json.loads((evidence / 'bootstrap.json').read_text())
    plan = json.loads((evidence / 'plan.json').read_text())
    from colab_cli.common import state
    from colab_cli.contents import ContentsClient
    from drive_checkpoint_store import DriveStore
    cli = str(Path(sys.executable).with_name('colab'))
    before = usage(cli)
    reuse = state.get_session(args.session, ignore_missing_session=True) if args.reuse_idle_session else None
    if args.reuse_idle_session:
        if reuse is None or before['count'] != 1:
            raise RuntimeError('指定的待准备实例不存在或存在多个实例')
        response = requests.get(reuse.url.rstrip('/') + '/api/kernels',
            params={'authuser': '0', 'colab-runtime-proxy-token': reuse.token}, timeout=30)
        response.raise_for_status()
        if any(item.get('execution_state') != 'idle' for item in response.json()):
            raise RuntimeError('指定实例内核非 idle，禁止覆盖正在执行的训练')
        log.info('复用尚未开始训练的单一实例：%s', args.session)
    elif before['count'] or before['rate']:
        raise RuntimeError('已有活动或计费实例，禁止盲目创建第二台')
    check_launch_balance(before['balance'], plan)
    policy = balance_policy(plan)
    credential = Path.home() / '.config/colab-cli/token.json'
    monitor = Path('/tmp/river_valley_monitor.env')
    if not credential.is_file() or not monitor.is_file():
        raise FileNotFoundError('现有 Drive/监控凭据不完整，不能先申请 GPU 再等待')
    store = DriveStore(bootstrap['drive_root_id'], str(credential), bootstrap['quota_project'])
    # 准备材料在申请实例前已验证持久化，恢复不依赖本机临时文件仍然存在。
    persistent_inputs = {}
    for name in ('version_code.tar', 'version_materials.tar', 'bootstrap.json', 'plan.json', 'code_manifest.json'):
        persistent_inputs[name] = store.upload(evidence / name, store.root_id)
    (evidence / 'persistent_inputs.json').write_text(json.dumps(persistent_inputs, indent=2))
    store.upload(evidence / 'persistent_inputs.json', store.root_id)
    prepare_backup_folders(store, plan)
    original = json.loads((evidence.parent / 'current_attempt.json').read_text())
    started = time.time()
    rate = max(6.77, before['rate'])
    limits = {'soft_deadline': started + min(args.hours * 3600, (before['balance'] - policy['soft_balance_cu']) / rate * 3600),
              'hard_deadline': started + min((args.hours + .5) * 3600, (before['balance'] - policy['hard_balance_cu']) / rate * 3600),
              **policy,
              'stop_file': '/content/river-valley-run/STOP',
              'initial_balance_cu': original['initial_balance_cu'], 'allocation_balance_cu': before['balance'],
              'authorized_cap_cu': None, 'reason': '本次实例时限与余额保护，不是旧预算上限'}
    (evidence / 'run_limits.json').write_text(json.dumps(limits, ensure_ascii=False, indent=2))
    session = None
    try:
        if not reuse:
            subprocess.run([cli, 'new', '-s', args.session, '--gpu', 'A100', '--high-mem'], check=True, timeout=300)
        session = state.get_session(args.session)
        allocated_usage = usage(cli)
        limits = recalibrate_limits(limits, allocated_usage['balance'], max(rate, allocated_usage['rate']))
        (evidence / 'run_limits.json').write_text(json.dumps(limits, ensure_ascii=False, indent=2))
        preparing = dict(original, session=args.session, endpoint=session.endpoint, status='preparing_cloud_instance',
                         stage='setup', evidence_dir=str(evidence.relative_to(ROOT)), allocated_at=time.time())
        (evidence.parent / 'current_attempt.json').write_text(json.dumps(preparing, ensure_ascii=False, indent=2) + '\n')
        client = reliable_contents_class(ContentsClient)(session)
        # 分配 READY 不代表 Jupyter 文件接口已完成启动，先等待现有实例就绪。
        client.list_dir('/content')
        client._request('PUT', '/content/river-valley-run', json_data={'type': 'directory'})
        for name in ('run_limits.json', 'plan.json', 'bootstrap.json', 'persistent_inputs.json'):
            client.upload(str(evidence / name), '/content/river-valley-run/' + name)
        client.upload(str(ROOT / 'src/scripts/colab_deadline_guard.py'), '/content/version_guard.py')
        protect = evidence / 'protect_cell.py'
        protect.write_text("""import subprocess,sys
from pathlib import Path
run=Path('/content/river-valley-run')
with (run/'deadline_guard.log').open('a') as log:
 p=subprocess.Popen([sys.executable,'-u','/content/version_guard.py'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(run/'deadline_guard.pid').write_text(str(p.pid))
print('独立保存与释放保护已启动',p.pid)
""")
        subprocess.run([cli, 'exec', '-s', args.session, '-f', str(protect), '--timeout', '60'], check=True, timeout=90)
        client.upload(str(evidence / 'version_code.tar'), '/content/version_code.tar')
        client.upload(str(credential), '/content/river_valley_drive_credentials.json')
        client.upload(str(monitor), '/content/river_valley_monitor.env')
        client.upload(str(ROOT / 'src/scripts/version_setup.py'), '/content/version_setup.py')
        setup = evidence / 'setup_cell.py'
        setup.write_text("""import subprocess,sys
from pathlib import Path
run=Path('/content/river-valley-run')
with (run/'setup.log').open('a') as log:
 p=subprocess.run([sys.executable,'-u','/content/version_setup.py'],stdout=log,stderr=subprocess.STDOUT)
print('准备退出码',p.returncode)
if p.returncode: raise RuntimeError('准备未通过，请读取 setup.log 与 checks.log')
""")
        subprocess.run([cli, 'exec', '-s', args.session, '-f', str(setup), '--timeout', '900'], check=True, timeout=930)
        for name in ('launch_gate.json', 'setup.log', 'checks.log'):
            client.download('/content/river-valley-run/' + name, str(evidence / name))
        gate = json.loads((evidence / 'launch_gate.json').read_text())
        if gate['status'] != 'passed' or gate['plan_sha256'] != bootstrap['plan_sha256']:
            raise ValueError('云端启动检查没有对应当前冻结计划')
        training = evidence / 'training_cell.py'
        training.write_text("""import os,subprocess,sys,json
from pathlib import Path
run=Path('/content/river-valley-run')
b=json.loads((run/'bootstrap.json').read_text())
os.environ.update(RV_DRIVE_ROOT_ID=b['drive_root_id'],RV_DRIVE_CREDENTIAL_FILE='/content/river_valley_drive_credentials.json',RV_DRIVE_QUOTA_PROJECT=b['quota_project'])
subprocess.run([sys.executable,'-u','/content/river-valley/src/scripts/version_training_cell.py','--plan',str(run/'plan.json')],check=True)
""")
        commands = {
            'training_cell': [cli, 'exec', '-s', args.session, '-f', str(training), '--timeout', str((args.hours + 1) * 3600)],
            'watch': [sys.executable, '-u', str(ROOT / 'src/scripts/version_watch.py'), '--session', args.session, '--output', str(evidence)],
            'connection': [sys.executable, '-u', str(ROOT / 'src/scripts/colab_training_connection.py'), '--session', args.session,
                           '--output-dir', str(evidence / 'connection'), '--stop-at', str(limits['hard_deadline'])]}
        for name, command in commands.items():
            if name == 'training_cell':
                training_log = evidence / (name + '.log')
                training_log_offset = training_log.stat().st_size if training_log.exists() else 0
            with (evidence / (name + '.log')).open('a') as output:
                process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            (evidence / (name + '.pid')).write_text(str(process.pid))
            if name == 'training_cell':
                training_process = process
            if name == 'watch' and sys.platform == 'darwin' and Path('/usr/bin/caffeinate').exists():
                # 仅在本轮监控存活期间防止空闲睡眠，不修改系统持久设置。
                awake = subprocess.Popen(['/usr/bin/caffeinate', '-i', '-w', str(process.pid)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
                (evidence / 'idle_sleep_guard.pid').write_text(str(awake.pid))
        startup = confirm_pipeline_start(session, evidence, training_process, commands['training_cell'], training_log_offset)
        index = dict(original, session=args.session, endpoint=session.endpoint, status='launched_awaiting_step_growth',
                     stage=plan['stages'][0]['id'], evidence_dir=str(evidence.relative_to(ROOT)), launched_at=time.time(),
                     drive_root_id=bootstrap['drive_root_id'], pipeline_startup=startup,
                     target_steps=plan['stages'][0]['target_steps'],
                     next_action='读取实际步数确认运行；本计划完成后按版本记录推进后续训练与交付。')
        index.pop('pending_session', None)
        (evidence.parent / 'current_attempt.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
        log.info('训练单元格及只读监控已启动；必须以实际步数增长确认成功：%s', args.session)
    except Exception as error:
        http_status = error.response.status_code if getattr(error, 'response', None) is not None else None
        log.error('实例准备或启动失败，保存证据并释放本次实例：类型=%s HTTP=%s', type(error).__name__, http_status)
        if session:
            for name in ('setup.log', 'checks.log', 'launch_gate.json', 'status.json'):
                try:
                    client.download('/content/river-valley-run/' + name, str(evidence / name))
                except Exception:
                    pass
            state.client.unassign(session.endpoint)
            state.prune_session(args.session)
            (evidence / 'setup_failure_release.json').write_text(json.dumps({'time': time.time(), 'session': args.session, 'usage': usage(cli)}))
            failed = dict(original, session=None, status='setup_failed_released', failed_session=args.session,
                          evidence_dir=str(evidence.relative_to(ROOT)), next_action='先读取 setup/launch/checks 证据并修复，再恢复；禁止同一错误盲目重复开机。')
            (evidence.parent / 'current_attempt.json').write_text(json.dumps(failed, ensure_ascii=False, indent=2) + '\n')
        raise RuntimeError(f'启动失败：{type(error).__name__}；查看已保存的准备证据') from None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
