"""独立上传版本实验的真实指标，不把不同阶段接成同一条损失曲线。"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import subprocess
import time

log = logging.getLogger(__name__)


def read(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, ValueError):
        return {} if default is None else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-root', default='/content/river-valley-run')
    parser.add_argument('--credential-file', default='/content/river_valley_monitor.env')
    parser.add_argument('--local-evidence', action='store_true', help='从本机只读同步证据接续面板，不执行远端命令')
    args = parser.parse_args()
    root = Path(args.run_root)
    plan = read(Path(args.plan))
    for line in Path(args.credential_file).read_text().splitlines():
        if line.startswith('SWANLAB_API_KEY='):
            os.environ['SWANLAB_API_KEY'] = line.split('=', 1)[1]
    import swanlab
    swanlab.login(api_key=os.environ['SWANLAB_API_KEY'], save=False)
    tracker = swanlab.init(project='river-valley-1b-training', id=plan['monitor_id'],
                          experiment_name=plan['description'], resume='allow', public=False,
                          mode='online', logdir=str(root / 'monitor_sdk'),
                          config={'version': plan['version'], 'stages': [{k: s[k] for k in ('id', 'kind', 'target_steps')} for s in plan['stages']]})
    (root / 'monitor_url.json').write_text(json.dumps({'url': tracker.url, 'id': tracker.id}))
    heartbeat_path = root / ('local_monitor_heartbeat.json' if args.local_evidence else 'monitor_heartbeat.json')
    previous = read(heartbeat_path, read(root / 'monitor_heartbeat.json'))
    positions, steps = previous.get('positions', {}), previous.get('steps', {})
    last_step = 0
    started = time.time()
    for stage in plan['stages']:
        tracker.define_metric(stage['id'] + '/*', x_axis=stage['id'] + '/step', section_name=stage['id'])
    log.info('新版监控已启动：%s', tracker.url)
    while True:
        try:
            values = {'运行/分钟': (time.time() - started) / 60, '监控/本机只读桥接': int(args.local_evidence)}
            if args.local_evidence:
                evidence = read(root / 'read_evidence.json')
                age = time.time() - evidence.get('time', 0)
                values['监控/同步延迟秒'] = age
                if age > 180:
                    log.warning('本机同步证据已过期：延迟=%.1f秒，不上传陈旧训练数值', age)
                    time.sleep(20)
                    continue
            for stage in plan['stages']:
                name = stage['id']
                path = root / (f'checkpoints_{name}_metrics.jsonl') if args.local_evidence else root / 'checkpoints' / name / 'metrics.jsonl'
                if args.local_evidence and not any(item.get('file') == f'checkpoints/{name}/metrics.jsonl' and item.get('http') == 200 for item in evidence.get('files', [])):
                    continue
                if not path.exists():
                    continue
                lines = path.read_text().splitlines()
                for line in lines[positions.get(name, 0):]:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        break
                    record = {name + '/' + key: value for key, value in row.items()
                              if isinstance(value, (int, float)) and math.isfinite(value)}
                    if record:
                        last_step = max(last_step + 1, time.time_ns() // 1000000)
                        tracker.log(record, step=last_step)
                        steps[name] = row.get('step', steps.get(name))
                    positions[name] = positions.get(name, 0) + 1
                if name in steps:
                    values[name + '/step'] = steps[name]
                    values[name + '/完成百分比'] = min(100, steps[name] / stage['target_steps'] * 100)
            if not args.local_evidence:
                try:
                    gpu = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                    if gpu.returncode == 0:
                        utilization, used, total = map(float, gpu.stdout.strip().split(','))
                        values.update({'资源/GPU利用率': utilization, '资源/显存MiB': used, '资源/总显存MiB': total})
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    log.warning('GPU资源读数失败：%s，训练指标仍独立上传', type(error).__name__)
            usage = read(root / ('usage.json' if args.local_evidence else 'budget_usage.json'))
            if usage:
                values['成本/余额CU'] = usage['balance_cu']
                values['成本/读数延迟秒'] = time.time() - usage['time']
                values['成本/本轮CU'] = usage['used_cu']
            last_step = max(last_step + 1, time.time_ns() // 1000000)
            tracker.log(values, step=last_step)
            heartbeat = {'time': time.time(), 'positions': positions, 'steps': steps, 'resources': values,
                         'status': read(root / 'status.json')}
            temporary = heartbeat_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(heartbeat, ensure_ascii=False))
            temporary.replace(heartbeat_path)
            # 故障修复期间保持监控存活，恢复后自动接续；完成或实际释放才结束。
            released = args.local_evidence and usage.get('assignments') == 0 and usage.get('rate_cu_per_hour') == 0
            if released or heartbeat['status'].get('status') in ('plan_completed', 'stopped_saved'):
                tracker.finish()
                return
        except Exception as error:
            log.warning('监控暂时失败：%s；训练独立继续', type(error).__name__)
        time.sleep(20)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
