"""持续等待真实的版本训练进程，便于 Colab 将会话识别为正在执行计算。"""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-root', default='/content/river-valley-run')
    args = parser.parse_args()
    run = Path(args.run_root)
    if (run / 'pipeline.pid').exists():
        pid = int((run / 'pipeline.pid').read_text())
        if Path(f'/proc/{pid}').exists():
            raise RuntimeError('已有实际流水线进程，拒绝重复启动')
    pid = int((run / 'deadline_guard.pid').read_text())
    if not Path(f'/proc/{pid}').exists():
        raise RuntimeError('截止保护没有运行，不能开始训练')
    if json.loads((run / 'launch_gate.json').read_text())['status'] != 'passed':
        raise RuntimeError('启动检查未通过')
    script = Path(__file__).with_name('train_version.py')
    with (run / 'pipeline.log').open('a', encoding='utf-8') as output:
        process = subprocess.Popen([sys.executable, '-u', str(script), '--plan', args.plan,
                                    '--run-root', str(run), '--release-on-finish'],
                                   stdout=output, stderr=subprocess.STDOUT)
        (run / 'pipeline.pid').write_text(str(process.pid))
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                state = json.loads((run / 'status.json').read_text())
                logging.info('实际训练进程运行中：阶段=%s 状态=%s 已完成=%s',
                             state.get('stage'), state.get('status'), state.get('completed_steps'))
        if process.returncode:
            raise RuntimeError(f'训练进程异常退出：{process.returncode}，请检查已保存日志')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
