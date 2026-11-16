"""独立于训练进程的 Colab 截止时间保险，可在本机断线后继续工作。"""
import argparse
import json
import logging
import os
from pathlib import Path
import time

import requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limits', default='/content/river-valley-run/run_limits.json')
    args = parser.parse_args()
    limits = json.loads(Path(args.limits).read_text())
    address = os.environ.get('TBE_RUNTIME_ADDR')
    if not address:
        raise RuntimeError('缺少 Colab 运行时管理地址，预算保险不可用')
    logging.info('远端预算保险启动：保存期限=%s，释放期限=%s', limits['soft_deadline'], limits['hard_deadline'])
    heartbeat = Path(args.limits).parent / 'deadline_guard_status.json'

    def record(stage):
        temporary = heartbeat.with_suffix('.tmp')
        temporary.write_text(json.dumps({'time': time.time(), 'pid': os.getpid(), 'status': stage,
            'soft_deadline': limits['soft_deadline'], 'hard_deadline': limits['hard_deadline']}), encoding='utf-8')
        temporary.replace(heartbeat)

    while time.time() < limits['soft_deadline']:
        record('watching')
        time.sleep(min(15, max(0.1, limits['soft_deadline'] - time.time())))
    Path(limits['stop_file']).write_text('接近本阶段预算，保存并停止\n', encoding='utf-8')
    logging.warning('已请求训练保存并停止')
    while time.time() < limits['hard_deadline']:
        record('save_requested')
        time.sleep(min(15, max(0.1, limits['hard_deadline'] - time.time())))
    for attempt in range(20):
        record('releasing')
        try:
            response = requests.post(f'http://{address}/unassign', timeout=15)
            logging.info('释放运行时请求：尝试=%d，HTTP=%d', attempt + 1, response.status_code)
        except requests.RequestException as error:
            logging.error('释放运行时连接异常：尝试=%d，类型=%s', attempt + 1, type(error).__name__)
        time.sleep(10)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
