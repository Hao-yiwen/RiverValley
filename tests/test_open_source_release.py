"""发布包必须隔离私有文件，并绑定真实权重及冻结源码。"""
import json
from pathlib import Path
import sys
import subprocess
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/scripts'))
import prepare_open_source_release as release


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    for name in (*release.SOURCE_FILES, 'release/model_card.md'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('公开的示例文件\n', encoding='utf-8')
    for name in ('src/example.py', 'tests/test_example.py', 'evals/bilingual_acceptance_v1.jsonl'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('公开的示例文件\n', encoding='utf-8')
    return root


def test_source_export_excludes_history_credentials_and_raw_evidence(source, tmp_path):
    secret = 'hf_' + 'A' * 30
    for name in ('.env', '.env.production', '.claude/settings.local.json', '.git/config',
                 'docs/cloud_gpu.md', 'evals/results/private.json', 'checkpoints/raw/data.jsonl'):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secret, encoding='utf-8')
    target = tmp_path / 'output'
    release.build(source, target)
    assert (target / 'source/src/example.py').is_file()
    assert (target / 'source/evals/bilingual_acceptance_v1.jsonl').is_file()
    exported = [p for p in (target / 'source').rglob('*') if p.is_file()]
    assert all(secret not in p.read_text(encoding='utf-8') for p in exported)
    with tarfile.open(target / 'source.tar.gz') as archive:
        assert all(name.startswith('river-valley/') and '..' not in Path(name).parts
                   for name in archive.getnames())
        assert not any('/.git/' in name or '/checkpoints/' in name for name in archive.getnames())


@pytest.mark.parametrize('value', [
    'hf_' + 'A' * 30,
    '-----BEGIN ' + 'PRIVATE KEY-----',
    'pass' + 'word = ' + repr('a-real-looking-password'),
    'https://name:' + 'private-password@example.com',
])
def test_included_secret_fails_without_disclosing_value(source, tmp_path, value):
    (source / 'src/example.py').write_text(value, encoding='utf-8')
    with pytest.raises(ValueError) as error:
        release.build(source, tmp_path / 'output')
    assert value not in str(error.value)
    assert not (tmp_path / 'output').exists()
    assert not list(tmp_path.glob('.release-*'))


def test_symlink_cannot_escape_source_allowlist(source, tmp_path):
    outside = tmp_path / 'private.py'
    outside.write_text('私有内容', encoding='utf-8')
    (source / 'src/linked.py').symlink_to(outside)
    with pytest.raises(ValueError, match='符号链接'):
        list(release.collect_source(source))
    with pytest.raises(ValueError, match='不允许'):
        release.checked_file(source, '../private.py')


def test_release_refuses_to_overwrite_or_write_into_source_tree(source, tmp_path):
    target = tmp_path / 'existing'
    target.mkdir()
    marker = target / 'preserve.txt'
    marker.write_text('保留', encoding='utf-8')
    with pytest.raises(FileExistsError):
        release.build(source, target)
    assert marker.read_text(encoding='utf-8') == '保留'
    with pytest.raises(ValueError):
        release.build(source, source / 'src/release')


@pytest.fixture
def delivery(tmp_path):
    folder = tmp_path / 'delivery'
    (folder / 'sft_best').mkdir(parents=True)
    for name in release.MODEL_FILES:
        (folder / 'sft_best' / name).write_bytes(b'original-artifact')
    digest = release.sha(folder / 'sft_best/model.safetensors')
    release.write_json(folder / 'candidate_selection.json', {'eligible': True, 'model_sha256': digest})
    release.write_json(folder / 'delivery_receipt.json', {'sha256': digest})
    release.write_json(folder / 'holdout_binding.json', {
        'status': 'completed', 'model_sha256': digest,
        'selection_sha256': release.sha(folder / 'candidate_selection.json'),
    })
    frozen = folder / 'code/river-valley/src/model/gpt.py'
    frozen.parent.mkdir(parents=True)
    frozen.write_text('# 冻结源码\n', encoding='utf-8')
    expected = {
        'assets': {name: {'bytes': (folder / 'sft_best' / name).stat().st_size,
                         'sha256': release.sha(folder / 'sft_best' / name)} for name in release.MODEL_FILES},
        'frozen_code': {'src/model/gpt.py': release.sha(frozen)},
        'evidence': {name: release.sha(folder / name) for name in (
            'candidate_selection.json', 'delivery_receipt.json', 'holdout_binding.json')},
    }
    return folder, expected


@pytest.mark.parametrize('name', ['model.safetensors', 'tokenizer.json', 'config.json'])
def test_reject_changed_weight_tokenizer_or_config_even_at_same_size(delivery, name):
    folder, expected = delivery
    release.verify_checkpoint(folder, expected)
    (folder / 'sft_best' / name).write_bytes(b'changed!-artifact')
    with pytest.raises(ValueError, match='资产摘要'):
        release.verify_checkpoint(folder, expected)


def test_reject_changed_frozen_code(delivery):
    folder, expected = delivery
    (folder / 'code/river-valley/src/model/gpt.py').write_text('# 修改源码\n', encoding='utf-8')
    with pytest.raises(ValueError, match='冻结推理源码'):
        release.verify_checkpoint(folder, expected)


def test_holdout_must_bind_selected_model_not_just_locally_consistent_hashes(delivery):
    folder, expected = delivery
    release.write_json(folder / 'holdout_binding.json', {'status': 'completed', 'model_sha256': 'another-model'})
    expected['evidence']['holdout_binding.json'] = release.sha(folder / 'holdout_binding.json')
    with pytest.raises(ValueError, match='同一份权重'):
        release.verify_checkpoint(folder, expected)


def test_evaluation_export_uses_summary_allowlist(tmp_path):
    row = {'model_sha256': 'model', 'answers_sha256': 'answers', 'questions_sha256': 'questions',
           'groups': {'translation': {'rows': 48, 'points': 40, 'correct': 12}},
           'drive_folder_id': 'private-id', 'account': 'private-account', 'items': ['private sentence']}
    for name in ('baseline_review.json', 'candidate_review.json', 'holdout_review.json'):
        release.write_json(tmp_path / name, row)
    output = release.evaluation_summary(tmp_path)
    assert output['holdout']['groups']['translation']['correct'] == 12
    assert 'private' not in json.dumps(output)
    assert output['independent_human_review'] is False
    assert output['holdout_has_baseline_comparison'] is False


def test_tracked_audit_preserves_ignored_local_environment(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / '.gitignore').write_text('.env\n', encoding='utf-8')
    (tmp_path / '.env').write_text('HF_TOKEN=' + 'hf_' + 'A' * 30, encoding='utf-8')
    (tmp_path / 'README.md').write_text('公开说明', encoding='utf-8')
    subprocess.run(['git', 'add', '.gitignore', 'README.md'], cwd=tmp_path, check=True)
    assert release.audit_tracked_files(tmp_path)['text_files'] == 2
    assert (tmp_path / '.env').exists()


def test_tracked_audit_rejects_private_settings(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    private = tmp_path / '.claude/settings.local.json'
    private.parent.mkdir()
    private.write_text('{}', encoding='utf-8')
    subprocess.run(['git', 'add', '-f', '.claude/settings.local.json'], cwd=tmp_path, check=True)
    with pytest.raises(ValueError, match='本机配置'):
        release.audit_tracked_files(tmp_path)
