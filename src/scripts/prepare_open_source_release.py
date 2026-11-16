"""导出无历史源码快照和可核验的 Hub 私有暂存包，不执行上传或公开操作。"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)
SOURCE_FILES = (
    '.gitignore', 'README.md', 'LICENSE', 'NOTICE', 'CITATION.cff',
    'CONTRIBUTING.md', 'SECURITY.md', 'requirements.txt',
    'requirements-chat.txt', 'requirements-dev.txt', 'start_chat.command',
    'docs/data_sources.md', 'docs/third_party.md', 'docs/open_source_readiness.md',
    'docs/releasing.md', 'docs/quickstart.md', 'docs/cloud_training.md',
)
SOURCE_GLOBS = (
    'src/**/*.py', 'src/web/*.html', 'datasets/*.py', 'tests/*.py',
    'configs/*.json', 'evals/*.py', 'evals/*.jsonl', 'evals/*.manifest.json',
    'release/*.json', 'release/*.md', '.github/workflows/*.yml',
)
MODEL_FILES = (
    'model.safetensors', 'config.json', 'generation_config.json',
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'chat_template.jinja',
)
FROZEN_FILES = (
    'src/config.py', 'src/model/__init__.py', 'src/model/attention.py',
    'src/model/block.py', 'src/model/embeddings.py', 'src/model/feedforward.py',
    'src/model/gpt.py', 'src/model/losses.py', 'src/model/moe.py',
    'src/scripts/inference.py',
)
SECRET_PATTERNS = {
    '常见访问令牌': re.compile(r'(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|'
                           r'github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})'),
    '私钥': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    '字面量凭据': re.compile(r'''(?i)(?:password|passwd|api_key|access_token|refresh_token|client_secret)\s*[=:]\s*["'][^"'\n]{8,}["']'''),
    '登录脚本密码': re.compile(r'(?i)(?:password|密码).*send\s*["{]'),
    '带密码的连接地址': re.compile(r'(?i)\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@'),
}


def sha(path):
    """流式计算摘要，避免一次把数 GB 权重读入内存。"""
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def checked_file(root, relative):
    """白名单中的文件和父目录都不能借助符号链接逃逸。"""
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'不允许的发布路径：{relative}')
    candidate = root / relative
    for part in (candidate, *candidate.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f'拒绝发布符号链接：{relative}')
    if not candidate.is_file() or not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'发布文件缺失或越界：{relative}')
    return candidate


def scan_text(path, label):
    """只报告位置和规则，错误信息中不包含命中的凭据原文。"""
    text = Path(path).read_text(encoding='utf-8')
    for number, line in enumerate(text.splitlines(), 1):
        for rule, pattern in SECRET_PATTERNS.items():
            if pattern.search(line):
                raise ValueError(f'发布扫描拒绝：文件={label} 行={number} 规则={rule}')


def collect_source(root):
    paths = set(SOURCE_FILES)
    for pattern in SOURCE_GLOBS:
        paths.update(str(p.relative_to(root)) for p in root.glob(pattern)
                     if p.is_file() and '__pycache__' not in p.parts)
    for folder in ('src', 'datasets', 'tests', 'configs', 'evals', 'release', '.github'):
        if (root / folder).is_symlink():
            raise ValueError(f'拒绝发布符号链接目录：{folder}')
    for relative in sorted(paths):
        path = checked_file(root, relative)
        scan_text(path, relative)
        yield relative, path


def audit_tracked_files(root):
    """检查当前全部跟踪文件，避免白名单外的私有配置随主分支公开。"""
    root = Path(root).resolve()
    paths = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode('utf-8').split('\0')
    scanned, binary = 0, 0
    log.info('开始检查当前 Git 跟踪文件：文件=%d', sum(bool(p) for p in paths))
    for relative in paths:
        if not relative:
            continue
        parts = Path(relative).parts
        if '.claude' in parts or (Path(relative).name.startswith('.env') and Path(relative).name != '.env.example'):
            raise ValueError(f'拒绝跟踪本机配置或环境文件：{relative}')
        if relative == 'docs/cloud_gpu.md':
            raise ValueError('拒绝重新跟踪旧的私人云端连接文档')
        path = checked_file(root, relative)
        try:
            scan_text(path, relative)
        except UnicodeDecodeError:
            binary += 1
            continue
        scanned += 1
    log.info('当前 Git 跟踪文件检查完成：文本=%d，二进制跳过=%d', scanned, binary)
    return {'text_files': scanned, 'binary_files_skipped': binary}


def write_sums(folder):
    entries = []
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path != folder / 'SHA256SUMS':
            entries.append(f'{sha(path)}  {path.relative_to(folder).as_posix()}')
    (folder / 'SHA256SUMS').write_text('\n'.join(entries) + '\n', encoding='utf-8')
    return len(entries)


def export_source(root, target):
    log.info('开始导出源码白名单：目标=%s', target.name)
    selected = list(collect_source(root))
    target.mkdir()
    for relative, source in selected:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        scan_text(destination, relative)
    count = write_sums(target)
    log.info('源码快照完成：文件=%d，不含 Git 历史和本机实验目录', count)
    return count


def verify_checkpoint(delivery, expected):
    """发布资产必须匹配冻结的逐文件摘要，并与选优及留出证据绑定。"""
    for name in MODEL_FILES:
        path = checked_file(delivery, 'sft_best/' + name)
        record = expected['assets'][name]
        if path.stat().st_size != record['bytes'] or sha(path) != record['sha256']:
            raise ValueError(f'检查点资产摘要不符：{name}')
        log.info('检查点资产通过：文件=%s 字节=%d', name, record['bytes'])
    for name, digest in expected['evidence'].items():
        if sha(checked_file(delivery, name)) != digest:
            raise ValueError(f'评估证据摘要不符：{name}')
    receipt = read_json(delivery / 'delivery_receipt.json')
    selection = read_json(delivery / 'candidate_selection.json')
    holdout = read_json(delivery / 'holdout_binding.json')
    digest = expected['assets']['model.safetensors']['sha256']
    if any(item.get(key) != digest for item, key in (
        (receipt, 'sha256'), (selection, 'model_sha256'), (holdout, 'model_sha256')
    )):
        raise ValueError('交付、选优和留出没有绑定同一份权重')
    if not selection.get('eligible') or holdout.get('status') != 'completed':
        raise ValueError('候选尚未通过开发选择或留出生成未完成')
    if holdout.get('selection_sha256') != sha(delivery / 'candidate_selection.json'):
        raise ValueError('留出未绑定先前锁定的候选选择')
    for relative, digest in expected['frozen_code'].items():
        path = checked_file(delivery / 'code/river-valley', relative)
        if sha(path) != digest:
            raise ValueError(f'冻结推理源码摘要不符：{relative}')


def evaluation_summary(delivery):
    """仅导出白名单统计和证据摘要，不复制句对、账户或 Drive 目录信息。"""
    result = {'evaluation_type': 'internal_codex_review', 'independent_human_review': False,
              'device': 'mps', 'dtype': 'float32', 'do_sample': False,
              'repetition_penalty': 1.0, 'max_new_tokens': 160,
              'development_sentence_groups': 12, 'holdout_sentence_groups': 24,
              'holdout_has_baseline_comparison': False}
    for label, name in (('baseline', 'baseline_review.json'), ('candidate', 'candidate_review.json'),
                        ('holdout', 'holdout_review.json')):
        record = read_json(delivery / name)
        result[label] = {key: record[key] for key in (
            'model_sha256', 'answers_sha256', 'questions_sha256', 'groups'
        )}
        result[label]['review_sha256'] = sha(delivery / name)
    return result


def export_hub(root, delivery, target):
    policy = read_json(root / 'release/policy.json')
    expected = read_json(root / 'release/checkpoint_assets.json')
    if policy['model_sha256'] != expected['assets']['model.safetensors']['sha256']:
        raise ValueError('发布策略与检查点资产清单不一致')
    verify_checkpoint(delivery, expected)
    target.mkdir()
    for name in MODEL_FILES:
        shutil.copy2(delivery / 'sft_best' / name, target / name)
        if sha(target / name) != expected['assets'][name]['sha256']:
            raise ValueError(f'复制后的检查点摘要不符：{name}')
        if name != 'model.safetensors':
            scan_text(target / name, name)
    for relative in FROZEN_FILES:
        source = checked_file(delivery / 'code/river-valley', relative)
        scan_text(source, relative)
        destination = target / 'rivervalley_code' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha(destination) != expected['frozen_code'][relative]:
            raise ValueError(f'复制后的源码摘要不符：{relative}')
    for relative, destination in (
        ('release/model_card.md', 'README.md'), ('LICENSE', 'LICENSE_CODE'),
        ('NOTICE', 'NOTICE_CODE'), ('requirements-chat.txt', 'requirements-chat.txt'),
        ('docs/data_sources.md', 'DATA_SOURCES.md'),
        ('release/license_review.json', 'LICENSE_REVIEW.json'),
        ('release/tatoeba_attribution.json', 'TATOEBA_ATTRIBUTION.json'),
    ):
        shutil.copy2(checked_file(root, relative), target / destination)
        scan_text(target / destination, destination)
    summary = evaluation_summary(delivery)
    write_json(target / 'evaluation_summary.json', summary)
    write_json(target / 'release_manifest.json', {
        'release_name': policy['release_name'], 'model_sha256': policy['model_sha256'],
        'weight_license': policy['weight_license'],
        'public_release_ready': False,
        'publication_status': policy['weight_publication_status'],
        'publication_hold': policy.get('publication_hold'),
        'blockers': policy['weight_publication_blockers'],
        'assets': expected['assets'], 'frozen_code': expected['frozen_code'],
        'evidence_sha256': expected['evidence'],
        'source_license': policy['source_license'],
    })
    count = write_sums(target)
    log.info('Hub 推理包构建完成：文件=%d，权重许可待核实；本次构建不执行上传', count)
    return count


def build(root, output, delivery=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root.is_relative_to(output):
        raise ValueError('发布目录不能覆盖项目或项目父目录')
    if output.is_relative_to(root) and output.relative_to(root).parts[0] != 'dist':
        raise ValueError('仓库内的发布产物只能写入 dist/，避免污染源码白名单')
    if output.exists():
        raise FileExistsError('拒绝覆盖已存在的发布目录')
    output.parent.mkdir(parents=True, exist_ok=True)
    # 在同一文件系统暂存，所有校验成功后才原子发布目录。
    staging = Path(tempfile.mkdtemp(prefix='.release-', dir=output.parent))
    try:
        source_count = export_source(root, staging / 'source')
        hub_count = export_hub(root, Path(delivery).resolve(), staging / 'huggingface') if delivery else 0
        with tarfile.open(staging / 'source.tar.gz', 'w:gz') as archive:
            for path in sorted((staging / 'source').rglob('*')):
                if path.is_file():
                    archive.add(path, arcname='river-valley/' + str(path.relative_to(staging / 'source')), recursive=False)
        write_json(staging / 'build_report.json', {
            'source_files': source_count, 'hub_files': hub_count,
            'source_archive_sha256': sha(staging / 'source.tar.gz'),
            'history_included': False, 'uploaded': False,
            'public_release_ready': False,
            'reason': '发布前仍需处理 release/policy.json 中列明的历史凭据及权重许可事项。',
        })
        staging.rename(output)
    except Exception as error:
        shutil.rmtree(staging)
        log.error('发布准备失败：类型=%s；暂存目录已清理', type(error).__name__)
        raise
    log.info('发布准备完成：源码文件=%d Hub文件=%d 目标=%s', source_count, hub_count, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dist/open-source-20261116')
    parser.add_argument('--checkpoint', default='checkpoints/delivery_curriculum_20261110')
    parser.add_argument('--source-only', action='store_true', help='仅导出源码，不读取本地检查点')
    parser.add_argument('--audit-tracked', action='store_true', help='仅检查当前全部 Git 跟踪文件')
    args = parser.parse_args()
    if args.audit_tracked:
        audit_tracked_files(ROOT)
        return
    build(ROOT, Path(args.output), None if args.source_only else Path(args.checkpoint))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
