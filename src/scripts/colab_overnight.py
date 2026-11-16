"""夜间收尾：跳过新增问答计算，并独立重试已结束流水线的实例释放。"""
import argparse
import json
import logging
import os
from pathlib import Path
import time

from colab_train_pipeline import write_json
from drive_checkpoint_store import configured_store

logger = logging.getLogger(__name__)
TERMINAL_STATES = {'completed_needs_manual_review', 'completed_saved_for_manual_test',
                   'stopped_budget', 'failed'}


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def published_checkpoint(parent):
    """发布时已经回读 SHA256；收尾再核对指针、完整清单和当前文件大小。"""
    parent = Path(parent)
    pointer = read_json(parent / 'LATEST.json')
    name = pointer.get('path', '')
    if not name or Path(name).name != name or not name.startswith('step_'):
        raise ValueError(f'无效的持久断点指针：{parent}')
    path = parent / name
    manifest = read_json(path / 'BACKUP_COMPLETE.json')
    files = manifest.get('files', {})
    required = {'model.safetensors', 'config.json', 'tokenizer.json', 'training_state.pt'}
    if manifest.get('step') != pointer.get('step') or not required <= files.keys():
        raise ValueError(f'持久断点的完成清单不完整：{path}')
    for filename, metadata in files.items():
        if Path(filename).name != filename or not filename:
            raise ValueError('备份清单包含非法路径')
        target = path / filename
        if (not target.is_file() or target.stat().st_size != metadata['bytes']
                or len(metadata.get('sha256', '')) != 64):
            raise ValueError(f'持久断点文件与已校验清单不符：{target}')
    store = configured_store()
    if store:
        store.verify(manifest)
    return {'path': str(path), 'step': pointer['step'], 'files': files,
            'drive_folder_id': manifest.get('drive_folder_id'),
            'verification': 'Drive API 服务器内容摘要通过' if store else '本地清单与大小通过，仍须等待远端刷盘'}


def defer_pipeline_evaluation(output):
    """只响应当前流水线的显式策略，不影响用户明天独立运行问答验收。"""
    output = Path(output).resolve()
    root = output.parent
    policy = read_json(root / 'completion_policy.json')
    if not policy.get('defer_qa_evaluation') or str(output) != policy.get('evaluation_output'):
        return False
    # 父进程必须是本轮已批准的流水线，普通交互评估不触发收尾逻辑。
    if os.getppid() != policy.get('pipeline_pid'):
        return False
    current = read_json(root / 'status.json')
    if current.get('stage') != 'bilingual_eval' or current.get('completed_pretrain_updates') != 7629:
        raise ValueError('尚未完成预训练和 SFT，不能发布训练完成回执')
    persistent = Path(policy['persistent_root'])
    checkpoints = {name: published_checkpoint(persistent / name)
                   for name in ('pretrain', 'pretrain_best', 'sft', 'sft_best')}
    if (checkpoints['pretrain']['step'] != 7629
            or checkpoints['sft']['step'] != current.get('completed_sft_updates')
            or checkpoints['sft']['step'] != policy['expected_sft_updates']):
        raise ValueError('持久断点步数未达到本轮训练目标')
    receipt = {'created_at': time.time(), 'status': 'completed_saved_for_manual_test',
               'qa_evaluation': 'deferred_by_user', 'qa_answers_generated': False,
               'checkpoints': checkpoints,
               'note': '预训练和 SFT 已保存；未进行问答生成或质量验收，等待用户明天测试'}
    write_json(root / 'completion_receipt.json', receipt)
    write_json(persistent / 'completion_receipt.json', receipt)
    logger.info('已按用户要求跳过问答生成；最终 SFT=%d，持久目录=%s；返回流水线执行释放',
                checkpoints['sft']['step'], persistent)
    return True


def effective_status(current, receipt):
    """兼容本轮已加载的旧流水线：完成回执比旧版固定文案更准确。"""
    if (current.get('status') in {'completed_needs_manual_review', 'completed_saved_for_manual_test'}
            and receipt.get('qa_evaluation') == 'deferred_by_user'):
        return dict(current, status='completed_saved_for_manual_test',
                    note=receipt['note'], evaluation_deferred=True)
    return current


def release_due(current, observed_terminal_at, now, grace_seconds=180):
    """只在明确结束状态持续出现后介入，给主流水线刷盘和正常释放留出时间。"""
    if current.get('status') not in TERMINAL_STATES or not current.get('release_ready'):
        return None, False
    observed_terminal_at = now if observed_terminal_at is None else observed_terminal_at
    return observed_terminal_at, now - observed_terminal_at >= grace_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default='/content/river-valley-run')
    args = parser.parse_args()
    root = Path(args.run_root)
    address = os.environ.get('TBE_RUNTIME_ADDR')
    if not address:
        raise RuntimeError('缺少 Colab 管理地址，不能启动独立收尾保险')
    logger.info('云端收尾保险已启动：仅在流水线结束超过 180 秒仍未释放时介入')
    observed = None
    while True:
        current = read_json(root / 'status.json')
        observed, due = release_due(current, observed, time.time())
        write_json(root / 'overnight_guard_heartbeat.json',
                   {'time': time.time(), 'pipeline_status': current.get('status'),
                    'terminal_observed_at': observed, 'release_due': due})
        if due:
            break
        time.sleep(15)
    logger.warning('训练已结束但实例仍在线，开始独立重试刷盘和释放：状态=%s', current.get('status'))
    # 只重试主流水线已经通过持久化门禁的释放，不绕过正在进行的最终上传。
    try:
        from google.colab import drive
        drive.flush_and_unmount(timeout_ms=60000)
        logger.info('独立收尾保险刷盘完成')
    except Exception:
        logger.exception('收尾重试刷盘失败，将保留已发布的持久备份并继续请求释放')
    import requests
    while True:
        try:
            response = requests.post(f'http://{address}/unassign', timeout=15)
            logger.info('独立收尾释放请求：HTTP %d', response.status_code)
        except requests.RequestException as error:
            logger.error('独立释放请求异常：类型=%s', type(error).__name__)
        time.sleep(15)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
