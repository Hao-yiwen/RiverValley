"""在本机下载已完成的 SFT 权重，校验摘要并附上冻结代码和测试说明。"""
import argparse
import json
import logging
from pathlib import Path
import tarfile
import time

from drive_checkpoint_store import DriveStore, fingerprints

logger = logging.getLogger(__name__)


def download(store, file_id, target, expected):
    """下载中断保留临时文件；服务器不支持断点时从头写入，最终摘要才放行。"""
    target = Path(target)
    if target.exists() and fingerprints(target)['sha256'] == expected['sha256']:
        return
    partial = target.with_suffix(target.suffix + '.partial')
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected['bytes']:
            break
        if offset > expected['bytes']:
            partial.unlink()
            offset = 0
        try:
            with store.http.get('https://www.googleapis.com/drive/v3/files/' + file_id,
                    params={'alt': 'media'}, headers={'Range': f'bytes={offset}-'} if offset else {},
                    timeout=(20, 120), stream=True) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(f'下载失败：HTTP {response.status_code}')
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                with partial.open(mode) as output:
                    for block in response.iter_content(8 * 1024**2):
                        output.write(block)
            break
        except Exception as error:
            logger.warning('权重下载中断：文件=%s，尝试=%d，类型=%s', target.name, attempt + 1, type(error).__name__)
            if attempt == 4:
                raise
            time.sleep(5)
    actual = fingerprints(partial)
    if actual['bytes'] != expected['bytes'] or actual['sha256'] != expected['sha256']:
        raise ValueError('本地交付文件摘要不符：' + target.name)
    partial.replace(target)
    logger.info('本地文件 SHA256 校验通过：%s，字节=%d', target.name, actual['bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root-id', required=True, help='使用者自己的训练交付 Drive 根目录')
    parser.add_argument('--credential-file', default=str(Path.home() / '.config/colab-cli/token.json'))
    parser.add_argument('--quota-project', required=True, help='使用者自己的 Google Cloud 计费项目')
    parser.add_argument('--output', default='checkpoints/delivery_20261018')
    parser.add_argument('--code-bundle', default='evals/results/resume4_training_20261018/resume_code_bundle.tar')
    args = parser.parse_args()
    store = DriveStore(args.drive_root_id, args.credential_file, args.quota_project)
    receipt_id = store.child(store.root_id, 'completion_receipt.json')
    if not receipt_id:
        raise RuntimeError('Drive 尚无完成回执，不能交付为训练完成')
    receipt = store.request('GET', 'https://www.googleapis.com/drive/v3/files/' + receipt_id, params={'alt': 'media'}).json()
    checkpoints = receipt['checkpoints']
    if receipt['status'] != 'completed_saved_for_manual_test' or checkpoints['pretrain']['step'] != 7629 or checkpoints['sft']['step'] != 324:
        raise ValueError('完成回执未达到预训练与 SFT 目标')
    for stage, manifest in checkpoints.items():
        store.verify(manifest)
        logger.info('交付前服务器对象校验通过：%s，step=%d', stage, manifest['step'])
    root = Path(args.output).resolve()
    best = root / 'sft_best'
    best.mkdir(parents=True, exist_ok=True)
    for name, expected in checkpoints['sft_best']['files'].items():
        if Path(name).name != name:
            raise ValueError('非法交付文件名')
        download(store, expected['id'], best / name, expected)
    code = root / 'code'
    code.mkdir(exist_ok=True)
    with tarfile.open(args.code_bundle) as archive:
        archive.extractall(code, filter='data')
    inference = code / 'river-valley/src/scripts/inference.py'
    readme = f'''# RiverValley 本地测试权重

预训练已完成 7629 步、999,948,288 token，SFT 已完成两轮 324 步。
默认权重为 SFT 验证集损失最优副本（step={checkpoints['sft_best']['step']}）。
这表示训练流程完成，尚未进行问答质量验收。

在单独的 Python 环境安装依赖：

```bash
python -m pip install torch transformers==4.57.3 tokenizers==0.22.2 safetensors
python "{inference}" --model "{best}" --prompt "你好，请介绍一下你自己。"
```

CPU 可运行，支持 CUDA 的机器会默认使用 GPU。权重采用 FP32，约 4.05 GB；运行需要额外内存。
模型使用仓库自定义结构，随包的冻结源码是加载入口，不能仅凭 model_type 假设通用 AutoModel 已注册。
完整恢复状态和最终权重保存在 Drive：
https://drive.google.com/drive/folders/{args.drive_root_id}
最终 SFT 目录：https://drive.google.com/drive/folders/{checkpoints['sft']['drive_folder_id']}
最佳 SFT 目录：https://drive.google.com/drive/folders/{checkpoints['sft_best']['drive_folder_id']}

文件摘要与服务器文件 ID 见 completion_receipt.json。请测试简单中文和英文问题，训练 loss 下降不能代替实际回答验收。
'''
    (root / 'README_本地测试.md').write_text(readme, encoding='utf-8')
    (root / 'completion_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'status': 'weights_downloaded_verified', 'time': time.time(), 'directory': str(root),
              'model': str(best / 'model.safetensors'), 'pretrain_step': 7629, 'sft_step': 324,
              'best_sft_step': checkpoints['sft_best']['step'], 'qa_evaluated': False}
    (root / 'delivery_receipt.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
