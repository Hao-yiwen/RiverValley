"""私有模型上传须拒绝错权重、额外文件及服务器摘要不一致。"""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/scripts'))
from prepare_open_source_release import MODEL_FILES, sha, write_json, write_sums
from upload_huggingface_release import validate_package, ensure_private_repo, verify_remote


@pytest.fixture
def package(tmp_path):
    for name in MODEL_FILES:
        (tmp_path / name).write_bytes(b'original-artifact')
    expected = {'assets': {name: {'bytes': (tmp_path / name).stat().st_size,
                                 'sha256': sha(tmp_path / name)} for name in MODEL_FILES},
                'frozen_code': {}}
    write_json(tmp_path / 'release_manifest.json', {'model_sha256': sha(tmp_path / 'model.safetensors')})
    write_sums(tmp_path)
    return tmp_path, expected


def test_upload_refuses_package_modified_after_build(package):
    root, expected = package
    validate_package(root, expected)
    (root / 'model.safetensors').write_bytes(b'changed!-artifact')
    with pytest.raises(ValueError, match='摘要不符'):
        validate_package(root, expected)


def test_upload_refuses_extra_private_file_even_if_not_listed(package):
    root, expected = package
    (root / '.env').write_text('本机私有信息', encoding='utf-8')
    with pytest.raises(ValueError, match='清单外文件'):
        validate_package(root, expected)


def test_existing_public_repository_cannot_receive_pending_license_model():
    api = SimpleNamespace(model_info=lambda _: SimpleNamespace(private=False))
    with pytest.raises(ValueError, match='已经公开'):
        ensure_private_repo(api, 'owner/model')


def test_remote_weight_must_match_server_sha256(package):
    root, expected = package
    info = SimpleNamespace(private=True, siblings=[SimpleNamespace(
        rfilename='model.safetensors', size=(root / 'model.safetensors').stat().st_size,
        lfs=SimpleNamespace(sha256='wrong-weight'))])
    api = SimpleNamespace(model_info=lambda *a, **k: info)
    with pytest.raises(ValueError, match='权重 SHA256 不符'):
        verify_remote(api, 'owner/model', 'frozen-revision', root,
                      {'model.safetensors': expected['assets']['model.safetensors']['sha256']}, None)
