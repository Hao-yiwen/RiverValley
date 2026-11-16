"""Colab 第一阶段训练：固定学习率计划、完整断点备份、自动 SFT 和预算停机。"""
import argparse
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
from drive_checkpoint_store import configured_store

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def write_json(path, value):
    """先写临时文件，再发布状态，避免读取到半份 JSON。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path):
    """持久副本必须有完整清单，且每个文件的字节数和摘要一致。"""
    path = Path(path)
    manifest = json.loads((path / 'BACKUP_COMPLETE.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        target = path / name
        if not target.is_relative_to(path) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('备份清单包含非法路径')
        if target.stat().st_size != expected['bytes'] or sha256_file(target) != expected['sha256']:
            raise ValueError(f'持久断点摘要不匹配：{name}')
    return manifest


def publish_snapshot(source, parent, step, keep=2):
    """仅在训练子进程退出后复制不可变断点；完整验证后才更新持久指针。"""
    source, parent = Path(source), Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f'step_{step:06d}'
    stage = parent / f'.step_{step:06d}.staging'
    if target.exists():
        saved = verify_snapshot(target)
        source_names = {p.name for p in source.iterdir() if p.is_file()}
        if source_names != set(saved['files']):
            raise ValueError('同一步数的备份文件集合不同，拒绝覆盖')
        for name, metadata in saved['files'].items():
            if sha256_file(source / name) != metadata['sha256']:
                raise ValueError('同一步数的备份内容不同，拒绝覆盖')
        store = configured_store()
        if store:
            if saved.get('drive_folder_id'):
                store.verify(saved)
            else:
                saved = store.publish(source, parent.name, step)
                write_json(target / 'BACKUP_COMPLETE.json', saved)
        return target
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    files = {}
    started = time.monotonic()
    for item in sorted(source.iterdir()):
        if not item.is_file() or item.is_symlink():
            raise ValueError(f'断点含不支持的文件类型：{item.name}')
        destination = stage / item.name
        digest = hashlib.sha256()
        with item.open('rb') as reader, destination.open('wb') as writer:
            for chunk in iter(lambda: reader.read(8 * 1024 * 1024), b''):
                writer.write(chunk)
                digest.update(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        files[item.name] = {'bytes': item.stat().st_size, 'sha256': digest.hexdigest()}
        logger.info('备份文件已写入：step=%d，文件=%s，字节数=%d', step, item.name, item.stat().st_size)
    store = configured_store()
    manifest = store.publish(stage, parent.name, step) if store else {'step': step, 'files': files, 'created_at': time.time()}
    write_json(stage / 'BACKUP_COMPLETE.json', manifest)
    verify_snapshot(stage)
    os.replace(stage, target)
    write_json(parent / 'LATEST.json', {'step': step, 'path': target.name})
    candidates = sorted((p for p in parent.iterdir() if p.is_dir() and not p.is_symlink()
                         and re.fullmatch(r'step_\d+', p.name)
                         and (p / 'BACKUP_COMPLETE.json').exists()), key=lambda p: int(p.name[5:]), reverse=True)
    for old in candidates[max(1, keep):]:
        logger.info('清理本轮已被新副本替代的持久断点：%s', old)
        shutil.rmtree(old)
    logger.info('持久断点提交完成：%s，用时 %.1f 秒', target, time.monotonic() - started)
    return target


def inspect_training_state(path, require_optimizer=True):
    """只读取本次训练自行保存的状态；验证模型、优化器、RNG 和游标均存在。"""
    import torch
    path = Path(path)
    if not (path / 'model.safetensors').is_file():
        raise ValueError('缺失模型权重')
    state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
    if not state.get('rng_states') or (require_optimizer and (not state.get('has_optimizer_state') or not state.get('optimizer'))):
        raise ValueError('断点缺少优化器或随机数状态')
    step = state.get('next_iter')
    if not isinstance(step, int) or step < int(require_optimizer) or not state.get('resume_signature'):
        raise ValueError('断点缺少有效游标或恢复签名')
    del state
    gc.collect()
    return step


def remaining_targets(completed, interval=1000, total=7629):
    """恢复后只安排未完成的更新，学习率总计划仍由训练配置固定。"""
    if not 0 <= completed <= total or interval <= 0:
        raise ValueError('恢复步数或备份间隔无效')
    targets = set(range((completed // interval + 1) * interval, total, interval))
    if completed == 0:
        targets.update((3, 250))
    if completed < total:
        targets.add(total)
    return sorted(step for step in targets if completed < step <= total)


def restore_stage(source, run, persistent, stage='pretrain'):
    """校验完整恢复状态并保留已有最佳权重，避免从随机参数重新开始。"""
    source, run, persistent = Path(source), Path(run), Path(persistent)
    manifest = verify_snapshot(source)
    completed = inspect_training_state(source)
    if stage not in ('pretrain', 'sft'):
        raise ValueError('不支持的恢复阶段')
    total = 7629 if stage == 'pretrain' else 324
    if manifest['step'] != completed or not 0 < completed <= total:
        raise ValueError('恢复断点的清单步数与实际训练游标不一致')
    pointer = json.loads((persistent / f'{stage}_best/LATEST.json').read_text(encoding='utf-8'))
    if not re.fullmatch(r'step_\d+', pointer['path']):
        raise ValueError('最佳断点指针非法')
    best_source = persistent / f'{stage}_best' / pointer['path']
    best_manifest = verify_snapshot(best_source)
    if best_manifest['step'] != pointer['step'] or pointer['step'] > completed:
        raise ValueError('最佳权重来自恢复游标之后，不能混入后续训练')
    best_target = run / 'checkpoints' / stage / 'best'
    if best_target.exists():
        raise FileExistsError('恢复目标已有 best，请使用干净的本轮工作目录')
    best_target.mkdir(parents=True)
    for name in best_manifest['files']:
        shutil.copy2(best_source / name, best_target / name)
    logger.info('完整断点校验通过：恢复 step=%d，最佳权重 step=%d；优化器、RNG 和数据游标将由训练入口恢复',
                completed, pointer['step'])
    return completed


def restore_pretrain(source, run, persistent):
    return restore_stage(source, run, persistent, 'pretrain')


def runtime_diagnostics(run, label, returncode=None):
    """记录退出码、磁盘与 cgroup 内存事件，便于区分进程错误、OOM 和实例整体消失。"""
    record = {'time': time.time(), 'stage': label, 'returncode': returncode,
              'disk_free_bytes': shutil.disk_usage(run).free}
    for name in ('memory.events', 'memory.current', 'memory.max'):
        path = Path('/sys/fs/cgroup') / name
        if path.is_file():
            record[name] = path.read_text(encoding='utf-8').strip()
    with (Path(run) / 'runtime_events.jsonl').open('a', encoding='utf-8') as output:
        output.write(json.dumps(record, ensure_ascii=False) + '\n')
    logger.info('运行诊断：%s', json.dumps(record, ensure_ascii=False))
    return record


def sync_run_logs(run, persistent):
    """尽力同步运行证据；临时文件与单项同步失败不能终止正在训练的子进程。"""
    run, persistent = Path(run), Path(persistent)
    try:
        sources = [(path, path.name) for path in run.iterdir()
                   if not path.name.startswith('.') and path.suffix in ('.log', '.json', '.jsonl')]
    except OSError as error:
        logger.warning('运行证据目录暂不可读，下轮重试：目录=%s，错误=%s', run, error)
        return
    for stage_name in ('pretrain', 'sft'):
        metrics = run / 'checkpoints' / stage_name / 'metrics.jsonl'
        if metrics.is_file():
            sources.append((metrics, f'{stage_name}_metrics.jsonl'))
    for source, name in sources:
        target = persistent / name
        temporary = persistent / f'.{name}.sync-{os.getpid()}.tmp'
        try:
            if source.is_symlink() or not source.is_file():
                continue
            # 先复制到独立临时文件，再原子发布，失败时保留上一份有效证据。
            shutil.copy2(source, temporary)
            os.replace(temporary, target)
        except OSError as error:
            logger.warning('运行证据同步失败，下轮重试且继续训练：源=%s，目标=%s，错误=%s',
                           source, target, error)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                logger.warning('运行证据临时文件清理失败：路径=%s，错误=%s', temporary, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', default='/content/river-valley-run')
    parser.add_argument('--persistent-root', required=True)
    parser.add_argument('--release-on-finish', action='store_true')
    parser.add_argument('--resume-checkpoint', help='已发布且包含 BACKUP_COMPLETE.json 的完整预训练断点')
    parser.add_argument('--resume-stage', choices=('pretrain', 'sft'), default='pretrain')
    parser.add_argument('--backup-interval', type=int, default=1000)
    args = parser.parse_args()
    run, persistent = Path(args.run_root), Path(args.persistent_root)
    limits = json.loads((run / 'run_limits.json').read_text())
    env = dict(os.environ, RV_TRAIN_STOP_FILE=str(run / 'STOP'), RV_TRAIN_DEADLINE=str(limits['soft_deadline']),
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    current = {'status': 'starting', 'started_at': limits.get('original_started_at', time.time()),
               'attempt_started_at': time.time(), 'authorized_cap_cu': limits['authorized_cap_cu'],
               'planned_pretrain_updates': 7629, 'planned_pretrain_tokens': 999948288,
               'soft_deadline': limits['soft_deadline'], 'hard_deadline': limits['hard_deadline']}
    persistent.mkdir(parents=True, exist_ok=True)

    def stopped():
        return (run / 'STOP').exists() or time.time() >= limits['soft_deadline']

    def sync_logs():
        sync_run_logs(run, persistent)

    def status(**values):
        current.update(values, updated_at=time.time())
        write_json(run / 'status.json', current)
        write_json(persistent / 'status.json', current)
        logger.info('流水线状态：%s', json.dumps(values, ensure_ascii=False))

    def execute(label, command):
        status(status='running', stage=label)
        with (run / f'{label}.log').open('a', encoding='utf-8') as output:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT)
            status(child_pid=process.pid)
            runtime_diagnostics(run, label + ':start')
            synced = 0
            while process.poll() is None:
                if time.monotonic() - synced >= 60:
                    stage_name = 'pretrain' if label.startswith('pretrain') else 'sft'
                    metrics = run / 'checkpoints' / stage_name / 'metrics.jsonl'
                    if metrics.exists():
                        records = metrics.read_text(encoding='utf-8').splitlines()
                        if records:
                            try:
                                status(last_training_metric=json.loads(records[-1]))
                            except json.JSONDecodeError:
                                logger.warning('指标末行仍在写入，等待下一次同步：%s', metrics)
                    sync_logs()
                    synced = time.monotonic()
                time.sleep(5)
        runtime_diagnostics(run, label + ':exit', process.returncode)
        sync_logs()
        if process.returncode:
            raise RuntimeError(f'{label} 退出码 {process.returncode}，详情见对应日志')

    def config_file(name, value):
        path = run / f'{name}.json'
        write_json(path, value)
        return str(path)

    safe_to_release = False
    def request_stop(signum, frame):
        reason = signal.Signals(signum).name
        (run / 'STOP').write_text(f'收到 {reason}，请求保存后停止\n', encoding='utf-8')
        logger.warning('收到终止信号：%s；等待完整更新和断点保存', reason)
        status(stop_reason=reason)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    try:
        if stopped():
            raise RuntimeError('启动前已达到预算停止条件')
        # 材料、CUDA 测试和短对照必须由启动检查完成，不能悄悄跳过失败项。
        gate = json.loads((run / 'launch_gate.json').read_text())
        if gate.get('status') != 'passed':
            raise RuntimeError('GPU 开训检查尚未通过')
        config = json.loads((ROOT / 'configs/colab_a100_1b_pretrain.json').read_text())
        config['loss_chunk_size'] = gate['selected_loss_chunk_size']
        config['eval_interval'] = 200
        config['save_interval'] = 250
        resume_path = None
        completed = 0
        if args.resume_checkpoint and args.resume_stage == 'pretrain':
            resume_path = Path(args.resume_checkpoint)
            completed = restore_pretrain(resume_path, run, persistent)
            safe_to_release = True
            status(status='restored', stage='pretrain', resumed_from_step=completed,
                   completed_pretrain_updates=completed, completed_pretrain_tokens=completed * 131072,
                   persistent_checkpoint=str(persistent / 'pretrain' / f'step_{completed:06d}'))
        sft_resume_path = None
        sft_completed = 0
        if args.resume_stage == 'sft':
            if not args.resume_checkpoint:
                raise ValueError('SFT 恢复必须提供完整断点')
            pretrain_pointer = json.loads((persistent / 'pretrain/LATEST.json').read_text(encoding='utf-8'))
            if pretrain_pointer['step'] != 7629:
                raise ValueError('预训练未完成，不能跳到 SFT')
            verify_snapshot(persistent / 'pretrain' / pretrain_pointer['path'])
            sft_resume_path = Path(args.resume_checkpoint)
            sft_completed = restore_stage(sft_resume_path, run, persistent, 'sft')
            completed = 7629
            safe_to_release = True
            status(status='restored', stage='sft', completed_pretrain_updates=completed,
                   completed_sft_updates=sft_completed, resumed_from_step=sft_completed)
        # 恢复时跳过已完成分段，只调整备份频率，不重置学习率计划或数据游标。
        for target in remaining_targets(completed, args.backup_interval):
            if stopped():
                break
            segment = dict(config, max_steps=target, resume=resume_path is not None)
            if resume_path:
                segment['resume_path'] = str(resume_path)
            path = config_file(f'pretrain_to_{target}', segment)
            execute(f'pretrain_to_{target}', [sys.executable, '-u', 'src/scripts/pretrain.py', '--config', path])
            latest = run / 'checkpoints/pretrain/latest'
            completed = inspect_training_state(latest)
            status(status='backing_up', stage='pretrain', completed_pretrain_updates=completed,
                   completed_pretrain_tokens=completed * 131072)
            backup = publish_snapshot(latest, persistent / 'pretrain', completed)
            safe_to_release = True
            best = run / 'checkpoints/pretrain/best'
            if best.exists():
                publish_snapshot(best, persistent / 'pretrain_best', inspect_training_state(best, False), keep=1)
            if completed == 3:
                # 下一进程直接读取持久盘；成功继续更新才能认定恢复演练完成。
                if inspect_training_state(backup) != completed:
                    raise RuntimeError('持久盘恢复游标不一致')
                resume_path = backup
            else:
                resume_path = latest
                if not current.get('persistent_resume_verified'):
                    status(persistent_resume_verified=True)
            status(status='segment_complete', stage='pretrain', persistent_checkpoint=str(backup))
            sync_logs()
            if completed != target:
                break
        if completed != 7629 or stopped():
            status(status='stopped_budget', stage='pretrain', completed_pretrain_updates=completed)
        else:
            sft = json.loads((ROOT / 'configs/colab_a100_1b_sft.json').read_text())
            best = run / 'checkpoints/sft/best'
            for target in remaining_targets(sft_completed, 100, 324):
                if stopped():
                    break
                segment = dict(sft, max_steps=target, resume=sft_resume_path is not None)
                if sft_resume_path:
                    segment['resume_path'] = str(sft_resume_path)
                execute(f'sft_to_{target}', [sys.executable, '-u', 'src/scripts/sft.py', '--config', config_file(f'sft_to_{target}', segment)])
                latest = run / 'checkpoints/sft/latest'
                sft_completed = inspect_training_state(latest)
                status(status='backing_up', stage='sft', completed_sft_updates=sft_completed)
                publish_snapshot(latest, persistent / 'sft', sft_completed)
                publish_snapshot(best, persistent / 'sft_best', inspect_training_state(best, False), keep=1)
                sft_resume_path = latest
                safe_to_release = True
                if sft_completed != target:
                    break
            if stopped() or sft_completed != 324:
                status(status='stopped_budget', stage='sft')
            else:
                output = run / 'bilingual_outputs.jsonl'
                execute('bilingual_eval', [sys.executable, '-u', 'src/scripts/eval_bilingual.py',
                        '--model', str(best), '--output', str(output)])
                for path in run.glob('bilingual_outputs*'):
                    shutil.copy2(path, persistent / path.name)
                receipt_path = run / 'completion_receipt.json'
                receipt = json.loads(receipt_path.read_text(encoding='utf-8')) if receipt_path.exists() else {}
                if receipt.get('qa_evaluation') == 'deferred_by_user':
                    status(status='completed_saved_for_manual_test', stage='done',
                           note=receipt['note'], evaluation_deferred=True)
                else:
                    status(status='completed_needs_manual_review', stage='done',
                           note='模型已生成冻结题答案，人工语义题仍须评分，不能自动宣称问答验收通过')
    except Exception as error:
        logger.exception('训练流水线失败：%s', error)
        status(status='failed', error_type=type(error).__name__, error=str(error))
        # 若错误发生在新一段中，保留上一份已经核验的持久断点，避免发布不完整目录。
        safe_to_release = safe_to_release or not any((run / 'checkpoints').glob('*/latest/model.safetensors'))
        raise
    finally:
        try:
            sync_logs()
        except Exception:
            logger.exception('结束时日志同步失败，继续执行独立的实例释放流程')
        # 直接 API 上传完成后才允许收尾；挂载盘缓存不作为放行依据。
        store = configured_store()
        durable_final = False
        if store:
            try:
                for path in persistent.iterdir():
                    if path.is_file() and not path.name.startswith('.') and path.suffix in ('.json', '.jsonl', '.log'):
                        store.upload(path, store.root_id)
                durable_final = True
            except Exception:
                logger.exception('最终证据尚未通过 Drive 服务器校验，保留实例等待恢复处理')
        elif safe_to_release:
            try:
                from google.colab import drive
                drive.flush_and_unmount(timeout_ms=1200000)
                durable_final = True
            except Exception:
                logger.exception('Drive 刷盘失败，禁止按成功完成释放')
        if args.release_on_finish and safe_to_release and durable_final:
            status(releasing_runtime=True, release_ready=True)
            try:
                import requests
                address = os.environ['TBE_RUNTIME_ADDR']
                response = requests.post(f'http://{address}/unassign', timeout=30)
                response.raise_for_status()
                logger.info('流水线完成后释放运行时：HTTP %d', response.status_code)
            except Exception:
                logger.exception('立即释放未成功，保留独立截止时间保险和本机余额监控')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
