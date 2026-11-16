"""先启动截止保护，再校验冻结材料和依赖；准备失败时保留诊断并及时释放。"""
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

RUN = Path('/content/river-valley-run')
REPO = Path('/content/river-valley')
log = logging.getLogger(__name__)


def unpack(path, expected, target):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('上传压缩包摘要不匹配：' + path.name)
    with tarfile.open(path) as source:
        source.extractall(target, filter='data')
    log.info('压缩包摘要与解包通过：%s', path.name)


def main():
    RUN.mkdir(exist_ok=True)
    bootstrap = json.loads((RUN / 'bootstrap.json').read_text())
    limits = json.loads((RUN / 'run_limits.json').read_text())
    if limits['soft_deadline'] <= time.time() or not os.environ.get('TBE_RUNTIME_ADDR'):
        raise RuntimeError('运行期限已过或缺少运行时释放接口')
    unpack(Path('/content/version_code.tar'), bootstrap['code_sha256'], Path('/content'))
    guard_file = RUN / 'deadline_guard.pid'
    guard_pid = int(guard_file.read_text()) if guard_file.exists() else 0
    if not guard_pid or not Path(f'/proc/{guard_pid}').exists():
        with (RUN / 'deadline_guard.log').open('a') as output:
            guard = subprocess.Popen([sys.executable, '-u', str(REPO / 'src/scripts/colab_deadline_guard.py')],
                                     stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        guard_pid = guard.pid
        guard_file.write_text(str(guard_pid))
    log.info('独立截止保护已启动：pid=%d', guard_pid)
    for name in ('river_valley_drive_credentials.json', 'river_valley_monitor.env'):
        Path('/content', name).chmod(0o600)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.3',
                    'tokenizers==0.22.2', 'pytest==9.1.1', 'langid==1.1.6', 'swanlab==0.10.2'], check=True)
    sys.path[:0] = [str(REPO / 'src'), str(REPO / 'src/scripts')]
    from train_version import verify_plan
    from colab_train_pipeline import sha256_file, write_json
    from deliver_training_weights import download
    from drive_checkpoint_store import DriveStore
    from config import SFTConfig, PretrainConfig
    import torch
    import transformers
    import tokenizers
    # 多 GB 语料走 Drive 流式传输，避免本机 Contents API 的整体 Base64 内存开销。
    store = DriveStore(bootstrap['drive_root_id'], '/content/river_valley_drive_credentials.json', bootstrap['quota_project'])
    inputs = json.loads((RUN / 'persistent_inputs.json').read_text())
    expected = inputs['version_materials.tar']
    if expected['sha256'] != bootstrap['materials_sha256']:
        raise ValueError('服务器材料清单与冻结摘要不一致')
    download(store, expected['id'], Path('/content/version_materials.tar'), expected)
    unpack(Path('/content/version_materials.tar'), bootstrap['materials_sha256'], RUN)
    plan = verify_plan(json.loads((RUN / 'plan.json').read_text()))
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('实例不具备 CUDA BF16 能力')
    if torch.cuda.get_device_properties(0).total_memory < 75 * 1024 ** 3:
        raise RuntimeError('实例显存不足 75 GiB，不能按 A100 80GB 配置开始')
    for path, expected in plan['input_sha256'].items():
        if sha256_file(path) != expected:
            raise ValueError('冻结输入摘要不匹配：' + path)
    for stage in plan['stages']:
        (SFTConfig if stage['kind'] == 'sft' else PretrainConfig)(**stage['config'])
    with (RUN / 'checks.log').open('w') as output:
        subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_train_version.py',
                        'tests/test_checkpoint_pair.py',
                        'tests/test_sft_quality_v2.py', 'tests/test_drive_checkpoint_store.py',
                        'tests/test_colab_training_connection.py', 'tests/test_training_correctness.py', '-q'], cwd=REPO,
                       stdout=output, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, RV_TEST_TOKENIZER_PATH=str(RUN / 'tokenizer.json')))
    write_json(RUN / 'launch_gate.json', {'status': 'passed', 'time': time.time(),
               'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
               'transformers': transformers.__version__, 'tokenizers': tokenizers.__version__,
               'plan_sha256': sha256_file(RUN / 'plan.json'), 'code_sha256': bootstrap['code_sha256'],
               'material_sha256': bootstrap['materials_sha256'], 'tests': 'checks.log'})
    with (RUN / 'version_monitor.log').open('a') as output:
        monitor = subprocess.Popen([sys.executable, '-u', str(REPO / 'src/scripts/version_monitor.py'),
                                    '--plan', str(RUN / 'plan.json')], stdin=subprocess.DEVNULL,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    (RUN / 'version_monitor.pid').write_text(str(monitor.pid))
    log.info('环境与输入校验完成，监控进程已启动：pid=%d', monitor.pid)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
