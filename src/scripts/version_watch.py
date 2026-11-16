"""本机只读同步版本训练及账单，不向忙碌内核提交执行请求。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

import requests

log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    from colab_cli.common import state
    from colab_cli.contents import ContentsClient
    root = Path(args.output).resolve()
    limits = json.loads((root / 'run_limits.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    from launch_version import balance_policy
    policy = balance_policy(plan)
    for key, value in policy.items():
        if limits.get(key, value) != value:
            raise ValueError('运行阈值与冻结计划不一致，拒绝使用过期保护配置：' + key)
    original = state.get_session(args.session)
    endpoint = original.endpoint
    cli = str(Path(sys.executable).with_name('colab'))
    previous_status = None
    while True:
        cycle_started = time.monotonic()
        try:
            raw = subprocess.run([cli, 'usage'], capture_output=True, text=True, timeout=45, check=True).stdout.splitlines()
            balance = float(next(x.split(':', 1)[1].split()[0] for x in raw if x.startswith('Current balance:')))
            rate = float(next(x.split(':', 1)[1].split('/')[0] for x in raw if x.startswith('Usage rate:')))
            count = int(next(x.split(':', 1)[1] for x in raw if x.startswith('Active assignments:')))
            usage = {'time': time.time(), 'balance_cu': balance, 'rate_cu_per_hour': rate, 'assignments': count,
                     'used_cu': round(max(0, limits['initial_balance_cu'] - balance), 4), 'baseline_cu': limits['initial_balance_cu']}
            (root / 'usage.json').write_text(json.dumps(usage, indent=2))
            with (root / 'usage_history.jsonl').open('a') as output:
                output.write(json.dumps(usage) + '\n')
            assignments = state.client.list_assignments() if count == 0 else None
            if count == 0 and rate == 0 and all(item.endpoint != endpoint for item in assignments):
                (root / 'release_evidence.json').write_text(json.dumps(dict(usage, session=args.session, endpoint=endpoint, evidence='实际账单无活动实例、计费为零'), ensure_ascii=False, indent=2))
                log.info('实例已释放，本次版本累计余额差 %.2f CU', usage['used_cu'])
                return
            if count == 0:
                log.info('实例账单与分配状态尚未同时归零，等待再次核实')
                if args.once:
                    return
                time.sleep(60)
                continue
            state._sessions = None
            session = state.get_session(args.session, ignore_missing_session=True)
            if session is None or session.endpoint != endpoint:
                log.info('原会话消失或更换，停止操作旧实例')
                return
            from launch_version import reliable_contents_class
            client = reliable_contents_class(ContentsClient)(session)
            client.upload(str(root / 'usage.json'), '/content/river-valley-run/budget_usage.json')
            if balance <= policy['soft_balance_cu'] or time.time() >= limits['soft_deadline']:
                (root / 'STOP').write_text('保存和停止：余额或本次运行期限保护\n')
                client.upload(str(root / 'STOP'), '/content/river-valley-run/STOP')
            if balance <= policy['hard_balance_cu'] or time.time() >= limits['hard_deadline']:
                state.client.unassign(endpoint)
                state.prune_session(args.session)
                log.warning('已按真实余额或硬截止释放指定实例')
                continue
            params = {'authuser': '0', 'colab-runtime-proxy-token': session.token}
            names = ['status.json', 'pipeline.log', 'runner_heartbeat.json', 'monitor_heartbeat.json', 'monitor_url.json',
                     'completion_receipt.json', 'server_checkpoints.json', 'setup.log', 'launch_gate.json',
                     'deadline_guard.log', 'deadline_guard_status.json', 'milestone_receipts.jsonl', 'recovery_receipt.json']
            # 阶段进程的输出也要取回，不能将下载、评估或正常验证误判为训练卡住。
            heartbeat_path = root / 'runner_heartbeat.json'
            if heartbeat_path.exists():
                child_stage = json.loads(heartbeat_path.read_text()).get('stage', '')
                if child_stage and child_stage.replace('_', '').isalnum():
                    names.append(child_stage + '.log')
            for stage in plan['stages']:
                name = stage['id']
                names.extend([f'checkpoints/{name}/metrics.jsonl', f'{name}_answers.jsonl', f'{name}_answers.summary.json'])
                if plan.get('atomic_resume_pairs'):
                    names.append(f'{name}_RESUME.json')
                if stage.get('evaluate_milestones'):
                    for step in stage.get('milestone_steps', []):
                        names.extend([f'{name}_step_{step}_answers.jsonl', f'{name}_step_{step}_evaluation_binding.json'])
                if stage.get('initial_eval_questions'):
                    names.extend([f'{name}_initial_answers.jsonl', f'{name}_initial_answers.summary.json'])
            def read_file(name):
                try:
                    response = requests.get(session.url.rstrip('/') + '/api/contents/content/river-valley-run/' + name,
                        params={**params, 'format': 'text', 'content': 1}, timeout=(8, 20))
                    if response.status_code == 200:
                        target = root / name.replace('/', '_')
                        temporary = target.with_name(target.name + '.sync-partial')
                        temporary.write_text(response.json()['content'], encoding='utf-8')
                        temporary.replace(target)
                    return {'file': name, 'http': response.status_code, 'read_at': time.time()}
                except requests.RequestException as error:
                    log.warning('单文件读取失败：%s %s', name, type(error).__name__)
                    return {'file': name, 'error': type(error).__name__, 'read_at': time.time()}
            # 有限并发只读文件，避免单个暂时失联的文件拖慢全部心跳。
            with ThreadPoolExecutor(max_workers=4) as pool:
                reads = list(pool.map(read_file, sorted(set(names))))
            (root / 'read_evidence.json').write_text(json.dumps({'time': time.time(), 'files': reads}, indent=2))
            status_path = root / 'status.json'
            if status_path.exists():
                current = json.loads(status_path.read_text())
                marker = (current.get('stage'), current.get('status'), current.get('completed_steps'))
                if marker != previous_status:
                    log.info('云端阶段状态：阶段=%s 状态=%s 最近保存步数=%s', *marker)
                    previous_status = marker
            response = requests.get(session.url.rstrip('/') + '/api/kernels', params=params, timeout=20)
            if response.status_code == 200:
                kernels = [{k: item.get(k) for k in ('id', 'execution_state', 'last_activity', 'connections')} for item in response.json()]
                (root / 'kernel_state.json').write_text(json.dumps({'time': time.time(), 'kernels': kernels}, indent=2))
        except Exception as error:
            log.warning('只读监控暂时失败：%s，不据此判断训练退出', type(error).__name__)
        if args.once:
            return
        time.sleep(max(1, 60 - (time.monotonic() - cycle_started)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
