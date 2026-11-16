"""校验推理包，上传到私有 Hugging Face 模型仓库并核对服务器文件身份。"""
import argparse
import json
import logging
from pathlib import Path

from prepare_open_source_release import ROOT, MODEL_FILES, checked_file, read_json, scan_text, sha, write_json


log = logging.getLogger(__name__)


def validate_package(package, expected):
    """使用仓库固定资产摘要和包内清单双重校验，拒绝上传额外的本机文件。"""
    package = Path(package).resolve()
    checksums = {}
    for line in checked_file(package, 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, relative = line.split('  ', 1)
        if relative in checksums or len(digest) != 64:
            raise ValueError('发布文件摘要清单包含重复路径或无效摘要')
        path = checked_file(package, relative)
        if sha(path) != digest:
            raise ValueError(f'上传前文件摘要不符：{relative}')
        if relative != 'model.safetensors':
            scan_text(path, relative)
        checksums[relative] = digest
    actual = {p.relative_to(package).as_posix() for p in package.rglob('*') if p.is_file()}
    if actual != set(checksums) | {'SHA256SUMS'}:
        raise ValueError('推理包存在清单外文件或缺失文件，拒绝上传')
    for name in MODEL_FILES:
        path = checked_file(package, name)
        if checksums.get(name) != expected['assets'][name]['sha256'] or path.stat().st_size != expected['assets'][name]['bytes']:
            raise ValueError(f'上传内容偏离固定检查点：{name}')
    for relative, digest in expected['frozen_code'].items():
        if checksums.get('rivervalley_code/' + relative) != digest:
            raise ValueError(f'冻结推理源码不匹配：{relative}')
    manifest = read_json(package / 'release_manifest.json')
    if manifest.get('model_sha256') != checksums['model.safetensors']:
        raise ValueError('发布回执未绑定实际上传权重')
    checksums['SHA256SUMS'] = sha(package / 'SHA256SUMS')
    log.info('上传前校验完成：文件=%d 权重SHA256=%s', len(checksums), checksums['model.safetensors'])
    return checksums


def ensure_private_repo(api, repo_id):
    """仅创建或复用私有模型仓库，已有公开仓库不会被自动修改。"""
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.model_info(repo_id)
    except RepositoryNotFoundError:
        api.create_repo(repo_id=repo_id, repo_type='model', private=True, exist_ok=False)
        log.info('已创建私有模型仓库：%s', repo_id)
        info = api.model_info(repo_id)
    if not info.private:
        raise ValueError('目标仓库已经公开，当前许可状态只允许上传到私有仓库')
    log.info('目标仓库私有状态已确认：%s', repo_id)


def verify_remote(api, repo_id, revision, package, checksums, download):
    """大权重核对服务器 LFS SHA256，小文件下载后逐项核对 SHA256。"""
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    if not info.private:
        raise ValueError('上传后仓库私有状态发生变化')
    remote = {item.rfilename: item for item in info.siblings}
    unexpected = set(remote) - set(checksums) - {'.gitattributes'}
    if unexpected:
        raise ValueError('远端存在本次发布清单之外的文件，不能标记完成')
    for name, digest in checksums.items():
        item = remote.get(name)
        if item is None or item.size != (package / name).stat().st_size:
            raise ValueError(f'远端文件缺失或大小不符：{name}')
        if name == 'model.safetensors':
            lfs = item.lfs
            remote_digest = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
            if remote_digest != digest:
                raise ValueError('服务器报告的模型权重 SHA256 不符')
        else:
            local = Path(download(repo_id=repo_id, filename=name, revision=revision, force_download=True))
            if sha(local) != digest:
                raise ValueError(f'远端文件内容摘要不符：{name}')
        log.info('远端资产校验通过：文件=%s 字节=%d', name, item.size)
    return info


def upload_release(package, repo_id, receipt_path):
    from huggingface_hub import HfApi, hf_hub_download
    package = Path(package).resolve()
    checksums = validate_package(package, read_json(ROOT / 'release/checkpoint_assets.json'))
    api = HfApi()
    identity = api.whoami()
    log.info('开始上传模型：仓库=%s 用户=%s 文件=%d', repo_id, identity['name'], len(checksums))
    ensure_private_repo(api, repo_id)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type='model', folder_path=str(package),
        allow_patterns=sorted(checksums),
        commit_message='Upload verified RiverValley 1B experimental checkpoint',
    )
    info = verify_remote(api, repo_id, commit.oid, package, checksums, hf_hub_download)
    receipt = {
        'status': 'uploaded_private_verified', 'repo_id': repo_id,
        'url': f'https://huggingface.co/{repo_id}', 'revision': commit.oid,
        'private': info.private, 'model_sha256': checksums['model.safetensors'],
        'files_verified': len(checksums),
        'verification': '模型由服务器 LFS SHA256 和大小核验；其他文件下载后逐项 SHA256 核验',
        'weight_license': None, 'public_release_ready': False,
    }
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = receipt_path.with_suffix('.tmp')
    write_json(temporary, receipt)
    temporary.replace(receipt_path)
    log.info('模型上传及服务器核验完成：仓库=%s revision=%s 文件=%d', repo_id, commit.oid, len(checksums))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True, help='已构建并检查的 Hugging Face 推理包目录')
    parser.add_argument('--repo-id', default='yiwenX/RiverValley-1B-Experimental-20261110')
    parser.add_argument('--receipt', default='dist/huggingface_upload_receipt.json')
    args = parser.parse_args()
    try:
        receipt = upload_release(args.package, args.repo_id, args.receipt)
    except Exception as error:
        # SDK 异常可能包含临时签名地址，只记录错误类型和操作上下文。
        log.error('上传或核验失败：仓库=%s 错误类型=%s', args.repo_id, type(error).__name__)
        raise SystemExit(1) from None
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
