"""按冻结阶段计划训练、服务器验证备份及恢复；不把新阶段伪装成旧断点续训。"""
import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'src/scripts')]
from colab_train_pipeline import inspect_training_state, write_json, sha256_file
from deliver_training_weights import download
from drive_checkpoint_store import configured_store

log = logging.getLogger(__name__)


def read_remote_json(store, file_id):
    return store.request('GET', 'https://www.googleapis.com/drive/v3/files/' + file_id,
                         params={'alt': 'media'}).json()


def restore_folder(store, folder_id, target, require_optimizer=False, weights_only=False):
    """先校验服务器对象，再下载逐个 SHA256 验证；清单本身不是成功凭证。"""
    manifest_id = store.child(folder_id, 'BACKUP_COMPLETE.json')
    if not manifest_id:
        raise ValueError('来源没有完整断点清单')
    manifest = read_remote_json(store, manifest_id)
    store.verify(manifest)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name, expected in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError('清单存在非法文件路径')
        # 新 SFT 仅加载初始化权重；阶段内恢复必须完整取回优化器和随机状态。
        if name == 'training_state.pt' and weights_only:
            continue
        download(store, expected['id'], target / name, expected)
    if require_optimizer and inspect_training_state(target) != manifest['step']:
        raise ValueError('恢复游标与服务器清单不一致')
    write_json(target / 'SOURCE_MANIFEST.json', manifest)
    return manifest


def latest_remote(store, stage):
    folder = store.child(store.root_id, stage)
    if not folder:
        return None
    pointer = store.child(folder, 'LATEST.json')
    return read_remote_json(store, pointer) if pointer else None


def committed_resume_pair(store, stage, required=False):
    """最新模型和最佳模型成对发布；中途单个LATEST变化不能代表恢复事务完成。"""
    marker = store.child(store.root_id, stage + '_RESUME.json')
    if marker:
        pair = read_remote_json(store, marker)
        full, best = pair.get('full', {}), pair.get('best', {})
        if (pair.get('stage') != stage or 'training_state.pt' not in full.get('files', {})
                or type(full.get('step')) is not int or type(best.get('step')) is not int
                or not 0 <= best['step'] <= full['step']):
            raise ValueError('成对恢复记录的阶段、游标或完整状态无效')
        store.verify(full)
        store.verify(best)
        return pair
    if required:
        if latest_remote(store, stage) or latest_remote(store, stage + '_best'):
            raise ValueError('存在未成对提交的备份；先核对原实例文件或保留完整初始来源，禁止拼接新旧恢复状态')
        return None
    # 旧计划只用于既有实例兼容；新计划显式开启成对恢复，不能默默降级。
    latest, best = latest_remote(store, stage), latest_remote(store, stage + '_best')
    if latest and (not best or best['step'] > latest['step']):
        raise ValueError('旧计划最佳权重缺失或来自游标之后')
    return {'stage': stage, 'full': latest, 'best': best} if latest else None


def commit_resume_pair(store, stage, full, best, workspace):
    """先验证两份服务器副本，再用单个小对象发布可恢复的一致快照。"""
    if 'training_state.pt' not in full.get('files', {}) or not 0 <= best['step'] <= full['step']:
        raise ValueError('不能发布缺少完整状态或最佳游标异常的恢复记录')
    store.verify(full)
    store.verify(best)
    pair = {'stage': stage, 'full': full, 'best': best, 'time': time.time(), 'status': 'pair_verified'}
    store.json_file(pair, store.root_id, stage + '_RESUME.json', workspace)
    write_json(Path(workspace) / (stage + '_RESUME.json'), pair)
    log.info('成对恢复记录已提交：阶段=%s，完整步数=%d，最佳步数=%d', stage, full['step'], best['step'])
    return pair


def stage_segments(completed, target, interval):
    if not 0 <= completed <= target or interval <= 0:
        raise ValueError('阶段游标或备份间隔非法')
    return list(range((completed // interval + 1) * interval, target, interval)) + ([target] if completed < target else [])


def verify_plan(plan):
    ids = set()
    if not plan.get('stages'):
        raise ValueError('缺少训练阶段')
    for stage in plan['stages']:
        name = stage['id']
        if not name.replace('_', '').isalnum() or name in ids:
            raise ValueError('重复或非法阶段标识')
        ids.add(name)
        if stage['kind'] not in ('sft', 'pretrain') or stage['target_steps'] <= 0:
            raise ValueError('非法训练目标')
        config = stage['config']
        if config.get('schedule_steps') != stage['target_steps']:
            raise ValueError('学习率总计划必须与本阶段目标一致')
        if config.get('resume'):
            raise ValueError('冻结基础配置不应包含临时恢复位置')
        if not stage.get('init_folder_id') and not stage.get('init_stage'):
            raise ValueError('缺少显式初始化来源')
    return plan


def upload_evidence_snapshot(store, source, snapshot_root, remote_name=None):
    """先冻结仍在写入的日志，再上传；避免上传日志本身产生的新日志改变待校验字节。"""
    source, snapshot_root = Path(source), Path(snapshot_root)
    snapshot_root.mkdir(exist_ok=True)
    target = snapshot_root / (remote_name or source.name)
    for attempt in range(4):
        shutil.copyfile(source, target)
        try:
            if source.suffix == '.json':
                json.loads(target.read_text(encoding='utf-8'))
            return store.upload(target, store.root_id, remote_name or source.name)
        except json.JSONDecodeError:
            if attempt == 3:
                raise ValueError('无法取得完整的 JSON 证据快照：' + source.name) from None
            time.sleep(.2)


def evaluate_checkpoint(run, name, checkpoint, questions, execute):
    """问答必须绑定实际权重与题集摘要，已有文件只有完整一致时才能复用。"""
    from eval_bilingual import validate_evaluation_rows
    run, checkpoint, questions = Path(run), Path(checkpoint), Path(questions)
    output = run / (name + '_answers.jsonl')
    summary = output.with_suffix('.summary.json')
    binding = run / (name + '_evaluation_binding.json')
    identity = {'model_sha256': sha256_file(checkpoint / 'model.safetensors'),
                'tokenizer_sha256': sha256_file(checkpoint / 'tokenizer.json'),
                'config_sha256': sha256_file(checkpoint / 'config.json'),
                'question_set_sha256': sha256_file(questions)}
    question_rows = [json.loads(line) for line in questions.read_text().splitlines()]

    def validate():
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        validate_evaluation_rows(rows, question_rows, identity['question_set_sha256'])
        json.loads(summary.read_text())
        return rows

    if output.exists() and summary.exists() and binding.exists():
        previous = json.loads(binding.read_text())
        if all(previous.get(key) == value for key, value in identity.items()) and previous.get('answers_sha256') == sha256_file(output):
            validate()
            log.info('复用完整且权重摘要一致的开发评估：%s', name)
            return previous
    for path in (output, summary, binding):
        if path.exists():
            path.rename(path.with_name(path.name + f'.prior_{time.time_ns()}'))
    execute(name + '_eval', [sys.executable, '-u', str(ROOT / 'src/scripts/eval_bilingual.py'),
        '--model', str(checkpoint), '--questions', str(questions), '--output', str(output), '--device', 'cuda'])
    rows = validate()
    result = dict(identity, answers_sha256=sha256_file(output), questions=len(rows), time=time.time(),
                  status='generated_pending_quality_review')
    write_json(binding, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--run-root', default='/content/river-valley-run')
    parser.add_argument('--release-on-finish', action='store_true')
    args = parser.parse_args()
    plan_path, run = Path(args.plan), Path(args.run_root)
    plan = verify_plan(json.loads(plan_path.read_text(encoding='utf-8')))
    run.mkdir(parents=True, exist_ok=True)
    store = configured_store()
    if store is None:
        raise RuntimeError('训练必须配置直接 Drive 服务器备份')
    limits = json.loads((run / 'run_limits.json').read_text())
    os.environ.update(RV_TRAIN_STOP_FILE=str(run / 'STOP'), RV_TRAIN_DEADLINE=str(limits['soft_deadline']))
    current = {'version': plan['version'], 'status': 'starting', 'stage': 'setup', 'time': time.time(),
               'plan_sha256': sha256_file(plan_path)}
    snapshots = {}
    durable = True

    def status(**values):
        current.update(values, time=time.time())
        write_json(run / 'status.json', current)

    def upload_logs():
        snapshot_root = run / '.evidence_upload'
        for path in run.glob('*'):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
                if path.name.endswith('_RESUME.json'):
                    # 恢复指针只能由成对提交发布，通用日志快照不能回滚服务器上的新记录。
                    continue
                # 实例里的启动元数据只作为运行证据，不能覆盖已修复的冻结材料根清单。
                remote_name = 'runtime_' + path.name if path.name in ('bootstrap.json', 'persistent_inputs.json', 'code_manifest.json') else path.name
                upload_evidence_snapshot(store, path, snapshot_root, remote_name)
        for stage in plan['stages']:
            for path in (run / 'checkpoints' / stage['id']).glob('metrics.jsonl'):
                upload_evidence_snapshot(store, path, snapshot_root, stage['id'] + '_metrics.jsonl')

    def execute(name, command):
        log.info('阶段命令开始：%s', name)
        with (run / (name + '.log')).open('a', encoding='utf-8') as output:
            process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            status(child_pid=process.pid, child_started_at=time.time())
            while process.poll() is None:
                write_json(run / 'runner_heartbeat.json', {'time': time.time(), 'stage': name,
                           'pid': os.getpid(), 'child_pid': process.pid})
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
        log.info('阶段命令退出：%s，returncode=%s', name, process.returncode)
        if process.returncode:
            raise RuntimeError(f'{name} 退出码 {process.returncode}，详见阶段日志')

    try:
        for path, expected in plan['input_sha256'].items():
            if sha256_file(path) != expected:
                raise ValueError('冻结输入摘要不一致：' + path)
        existing_plan = store.child(store.root_id, 'plan.json')
        if existing_plan and read_remote_json(store, existing_plan) != plan:
            raise ValueError('服务器已有不同的冻结计划，请使用新的实验目录')
        store.upload(plan_path, store.root_id, 'plan.json')
        for stage in plan['stages']:
            name, kind, target = stage['id'], stage['kind'], stage['target_steps']
            folder = run / 'checkpoints' / name
            folder.mkdir(parents=True, exist_ok=True)
            pair = committed_resume_pair(store, name, plan.get('atomic_resume_pairs', False))
            latest = pair['full'] if pair else None
            completed = latest['step'] if latest else 0
            if completed > target:
                raise ValueError('服务器断点已超过冻结目标，禁止混用计划')
            config = dict(stage['config'], checkpoint_dir=str(folder), use_swanlab=False)
            status(stage=name, kind=kind, target_steps=target, completed_steps=completed, status='restoring')
            resume = None
            if latest:
                resume = folder / 'latest'
                snapshots[name] = restore_folder(store, latest['drive_folder_id'], resume, True)
                best = pair['best']
                if not best or best['step'] > completed:
                    raise ValueError('最佳权重缺失或来自游标之后')
                snapshots[name + '_best'] = restore_folder(store, best['drive_folder_id'], folder / 'best')
            else:
                init_id = stage.get('init_folder_id')
                if not init_id:
                    source = latest_remote(store, stage['init_stage'])
                    if not source:
                        raise ValueError('前置阶段尚无完整权重')
                    init_id = source['drive_folder_id']
                initial = run / 'initial' / init_id
                restore_folder(store, init_id, initial, kind == 'pretrain', weights_only=kind == 'sft')
                if kind == 'sft':
                    config.update(pretrain_checkpoint=str(initial), resume=False)
                else:
                    # 新预训练阶段的初始化模式必须由配置显式选择，不能偷偷重置优化器。
                    config.update(resume=True, resume_path=str(initial))
                    if not config.get('new_stage_from_complete'):
                        raise ValueError('继续预训练需要 new_stage_from_complete 初始化入口')
                if stage.get('initial_eval_questions'):
                    output = run / (name + '_initial_answers.jsonl')
                    status(status='evaluating_initial', stage=name)
                    if output.exists():
                        output.rename(output.with_name(output.stem + f'_partial_{int(time.time())}.jsonl'))
                    execute(name + '_initial_eval', [sys.executable, '-u', str(ROOT / 'src/scripts/eval_bilingual.py'),
                            '--model', str(initial), '--questions', stage['initial_eval_questions'],
                            '--output', str(output), '--device', 'cuda'])
                    upload_logs()
            if stage.get('evaluate_milestones') and completed and completed in stage.get('milestone_steps', []):
                status(status='evaluating_milestone', completed_steps=completed)
                evaluate_checkpoint(run, f'{name}_step_{completed}', folder / 'latest', stage['eval_questions'], execute)
                upload_logs()
            targets = stage_segments(completed, target, stage.get('backup_interval', target))
            milestones = stage.get('milestone_steps', [])
            if any(type(step) is not int or not 0 < step <= target for step in milestones):
                raise ValueError('检查里程碑必须位于本阶段计划内')
            targets = sorted(set(targets) | {step for step in milestones if step > completed})
            for step in targets:
                if (run / 'STOP').exists() or time.time() >= limits['soft_deadline']:
                    break
                segment = dict(config, max_steps=step)
                if resume:
                    segment.update(resume=True, resume_path=str(resume), resume_weights_only=False)
                    segment.pop('new_stage_from_complete', None)
                path = run / f'{name}_to_{step}.json'
                write_json(path, segment)
                status(status='training', stage=name, completed_steps=completed, next_backup=step)
                execute(f'{name}_to_{step}', [sys.executable, '-u', str(ROOT / 'src/scripts' / (kind + '.py')), '--config', str(path)])
                resume = folder / 'latest'
                completed = inspect_training_state(resume)
                status(status='backing_up', completed_steps=completed)
                durable = False
                snapshots[name] = store.publish(resume, name, completed)
                # 最佳权重包含评分状态，但无需额外下载其 optimizer。
                best_path = folder / 'best'
                best_step = inspect_training_state(best_path, False)
                snapshots[name + '_best'] = store.publish(best_path, name + '_best', best_step)
                if plan.get('atomic_resume_pairs'):
                    commit_resume_pair(store, name, snapshots[name], snapshots[name + '_best'], run)
                durable = True
                write_json(run / 'server_checkpoints.json', snapshots)
                if completed in milestones:
                    with (run / 'milestone_receipts.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps({'stage': name, 'step': completed, 'time': time.time(),
                            'checkpoint': snapshots[name], 'note': '断点已通过服务器校验；由巡检读取固定验证集指标判断变化'},
                            ensure_ascii=False) + '\n')
                if completed in milestones and stage.get('evaluate_milestones'):
                    status(status='evaluating_milestone', completed_steps=completed)
                    evaluate_checkpoint(run, f'{name}_step_{completed}', resume, stage['eval_questions'], execute)
                upload_logs()
                if completed != step:
                    break
            if completed != target:
                status(status='stopped_saved', stage=name, completed_steps=completed)
                break
            questions = stage.get('eval_questions')
            if questions:
                status(status='evaluating', stage=name)
                evaluate_checkpoint(run, name, folder / 'best', questions, execute)
                upload_logs()
            status(status='stage_completed', stage=name, completed_steps=completed)
        else:
            receipt = {'status': 'plan_completed', 'version': plan['version'], 'plan_sha256': sha256_file(plan_path),
                       'stages': {s['id']: s['target_steps'] for s in plan['stages']},
                       'checkpoints': snapshots, 'time': time.time(),
                       'quality_status': '问答已生成时仍需按证据评分；计划完成不等于最终验收通过'}
            write_json(run / 'completion_receipt.json', receipt)
            # 只有回执也通过服务器摘要验证后，才能对外报告阶段计划完成。
            store.upload(run / 'completion_receipt.json', store.root_id)
            status(status='plan_completed', stage='done')
    except Exception as error:
        log.exception('版本训练失败：%s', error)
        status(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        # 失败时保留此前通过服务器验证的完整断点；日志上传失败不能伪装为交付完成。
        try:
            upload_logs()
        except Exception:
            log.exception('结束证据上传失败，保留先前完整断点和独立截止保护')
        if args.release_on_finish and durable:
            status(releasing_runtime=True)
            import requests
            address = os.environ.get('TBE_RUNTIME_ADDR')
            if address:
                for attempt in range(3):
                    try:
                        response = requests.post(f'http://{address}/unassign', timeout=20)
                        response.raise_for_status()
                        log.info('本轮实例释放请求成功：HTTP %d', response.status_code)
                        break
                    except requests.RequestException as error:
                        log.warning('释放重试：%s', type(error).__name__)
                        time.sleep(5)
        elif not durable:
            log.error('新断点尚未通过服务器校验，禁止按完成释放；保留实例供巡检修复及独立截止保护')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    main()
