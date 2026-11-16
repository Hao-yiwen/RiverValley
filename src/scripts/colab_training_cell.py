"""在持续执行的 Colab 单元格内等待真实训练流水线结束。"""
import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

from colab_train_pipeline import write_json

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default='/content/river-valley-run')
    parser.add_argument('--persistent-root', required=True)
    parser.add_argument('--resume-checkpoint', required=True)
    parser.add_argument('--resume-stage', choices=('pretrain', 'sft'), default='pretrain')
    parser.add_argument('--backup-interval', type=int, default=500)
    args = parser.parse_args()
    run = Path(args.run_root)
    if (run / 'pipeline.pid').exists():
        raise RuntimeError('本次运行已有流水线启动记录，拒绝重复启动')
    if json.loads((run / 'launch_gate.json').read_text())['status'] != 'passed':
        raise RuntimeError('恢复启动检查尚未通过')
    for name in ('deadline_guard', 'overnight_guard'):
        pid = int((run / f'{name}.pid').read_text())
        if not Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'缺少有效的云端保护进程：{name}')
    with (run / 'pipeline.log').open('a', encoding='utf-8') as log:
        command = [sys.executable, '-u', str(ROOT / 'src/scripts/colab_train_pipeline.py'),
                   '--persistent-root', args.persistent_root, '--run-root', str(run),
                   '--resume-checkpoint', args.resume_checkpoint,
                   '--resume-stage', args.resume_stage,
                   '--backup-interval', str(args.backup_interval), '--release-on-finish']
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        (run / 'pipeline.pid').write_text(str(process.pid), encoding='utf-8')
        policy = {'defer_qa_evaluation': True, 'pipeline_pid': process.pid,
                  'evaluation_output': str(run / 'bilingual_outputs.jsonl'),
                  'persistent_root': args.persistent_root, 'expected_sft_updates': 324,
                  'requested_at': time.time(), 'note': '按用户要求，保存训练结果后释放 GPU，问答验收另行进行'}
        write_json(run / 'completion_policy.json', policy)
        write_json(Path(args.persistent_root) / 'completion_policy.json', policy)
        logger.info('训练已提交：pid=%d，恢复断点=%s；单元格持续等待真实训练结束', process.pid, args.resume_checkpoint)
        # 此处等待实际工作；不以空循环或虚假交互伪装计算活动。
        while process.poll() is None:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                state = json.loads((run / 'status.json').read_text(encoding='utf-8'))
                logger.info('训练进展：阶段=%s，状态=%s，最近指标=%s', state.get('stage'),
                            state.get('status'), state.get('last_training_metric'))
        logger.info('训练流水线已退出：pid=%d，returncode=%d', process.pid, process.returncode)
        if process.returncode:
            raise RuntimeError(f'训练流水线异常退出：{process.returncode}，检查已保存日志')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
