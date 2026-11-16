"""锁定正式SFT候选，准备本地一次性留出验收；不申请任何云端实例。"""
import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from colab_train_pipeline import sha256_file
from deliver_training_weights import download
from deliver_version import load_completed, selected_checkpoint
from eval_bilingual import validate_evaluation_rows

ROOT = Path(__file__).resolve().parents[2]
HOLDOUT_SHA = '07befc8f9c23e70eb9eac45323fa12f83745aee6fd3d723d3443d5527bdd38e7'
log = logging.getLogger(__name__)


def validate_selection(selection, development_review):
    """开发报告必须先存在且摘要一致，留出回答不能反过来决定候选。"""
    report = json.loads(Path(development_review).read_text(encoding='utf-8'))
    if (selection.get('basis') != 'development' or not selection.get('reason')
            or not (report.get('complete') is True or report.get('all_reviewed') is True)
            or selection.get('development_review_sha256') != sha256_file(development_review)):
        raise ValueError('候选没有绑定已完成的开发复核报告')


def freeze_identity(path, identity):
    """已锁定的权重、题集、选择和执行参数不能在重复调用时变化。"""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        if old['identity'] != identity:
            raise ValueError('已有留出验收锁，禁止更换候选、题集或执行参数')
        return old
    result = {'time': time.time(), 'identity': identity, 'status': 'locked_before_holdout'}
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def validate_answers(output, questions, identity):
    """完整回答与选定权重路径、题集逐题一致后才可以进入质量复核。"""
    rows = [json.loads(x) for x in Path(output).read_text(encoding='utf-8').splitlines()]
    items = [json.loads(x) for x in Path(questions).read_text(encoding='utf-8').splitlines()]
    validate_evaluation_rows(rows, items, identity['question_set_sha256'])
    if any(row.get('model') != identity['model_directory'] for row in rows):
        raise ValueError('原始回答不是锁定候选生成的结果')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-evidence', required=True)
    parser.add_argument('--pretrain-evidence', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--development-review', required=True)
    parser.add_argument('--output', default='checkpoints/delivery_v2_20261019')
    parser.add_argument('--python', default=str(ROOT / '.venv/bin/python'))
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--run', action='store_true', help='准备并实际运行；省略时只准备，不使用留出题生成回答')
    args = parser.parse_args()
    evidence = Path(args.sft_evidence).resolve()
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    validate_selection(selection, args.development_review)
    questions = ROOT / 'evals/bilingual_acceptance_v2_holdout.jsonl'
    if sha256_file(questions) != HOLDOUT_SHA:
        raise ValueError('独立留出题偏离最初冻结版本')
    credential = str(Path.home() / '.config/colab-cli/token.json')
    store, plan, receipt = load_completed(evidence, credential)
    _, _, pretrain = load_completed(Path(args.pretrain_evidence), credential)
    stage = next(s for s in plan['stages'] if s['id'] == 'sft')
    if (plan.get('purpose') != 'final_sft_candidate'
            or stage['init_folder_id'] not in {m['drive_folder_id'] for m in pretrain['checkpoints'].values()}):
        raise ValueError('正式SFT未绑定已完成的本轮预训练')
    candidate = selected_checkpoint(store, plan, receipt, 'sft', selection)
    bootstrap = json.loads((evidence / 'bootstrap.json').read_text())
    bundle = evidence / 'version_code.tar'
    if sha256_file(bundle) != bootstrap['code_sha256']:
        raise ValueError('冻结源码压缩包摘要不一致')

    target = Path(args.output).resolve()
    if (target / 'delivery_receipt.json').exists():
        delivered = json.loads((target / 'delivery_receipt.json').read_text(encoding='utf-8'))
        if delivered.get('sha256') != candidate['files']['model.safetensors']['sha256']:
            raise ValueError('目标目录已有其他已交付权重，禁止覆盖；请使用新的候选目录')
    work = evidence / 'acceptance'
    work.mkdir(exist_ok=True)
    model = target / 'sft_best'
    identity = {key: candidate['files'][name]['sha256'] for key, name in [
        ('model_sha256', 'model.safetensors'), ('tokenizer_sha256', 'tokenizer.json'), ('config_sha256', 'config.json')]}
    identity.update(selection_sha256=sha256_file(selection_path), question_set_sha256=HOLDOUT_SHA,
                    code_sha256=bootstrap['code_sha256'], model_directory=str(model),
                    device=args.device, python=str(Path(args.python).absolute()), max_new_tokens=160,
                    do_sample=False, repetition_penalty=1.0)
    freeze_identity(work / 'acceptance_lock.json', identity)
    log.info('最终候选已锁定：SFT步数=%d，设备=%s，选择摘要=%s',
             candidate['step'], args.device, identity['selection_sha256'])
    model.mkdir(parents=True, exist_ok=True)
    for name, expected in candidate['files'].items():
        if Path(name).name != name:
            raise ValueError('服务器清单包含非法文件名')
        if name != 'training_state.pt':
            download(store, expected['id'], model / name, expected)
    code = target / 'code'
    code.mkdir(exist_ok=True)
    with tarfile.open(bundle) as archive:
        archive.extractall(code, filter='data')
    frozen = code / 'river-valley'
    for item in json.loads((evidence / 'code_manifest.json').read_text()):
        if sha256_file(frozen / item['path']) != item['sha256']:
            raise ValueError('解包后的源码摘要不一致：' + item['path'])
    output = work / 'holdout_answers.jsonl'
    command = [identity['python'], '-u', str(frozen / 'src/scripts/eval_bilingual.py'),
               '--model', str(model), '--questions', str(questions), '--output', str(output), '--device', args.device]
    (work / 'command.json').write_text(json.dumps(command, ensure_ascii=False, indent=2) + '\n')
    if not args.run:
        log.info('本地权重及冻结源码准备完成，未生成任何留出回答：%s', model)
        return

    # 原始文件已存在时只核验复用，绝不覆盖或静默重复整套留出题。
    if output.exists():
        binding_path = work / 'acceptance_binding.json'
        if binding_path.exists():
            previous = json.loads(binding_path.read_text(encoding='utf-8'))
            if (any(previous.get(key) != value for key, value in identity.items())
                    or previous.get('answers_sha256') != sha256_file(output)):
                raise ValueError('已绑定的原始回答或候选身份被修改，拒绝重新盖章')
        rows = validate_answers(output, questions, identity)
        log.info('已有完整同候选留出结果，复用%d题；没有重复生成', len(rows))
    else:
        environment = os.environ.copy()
        environment.pop('PYTHONPATH', None)
        environment['PYTHONNOUSERSITE'] = '1'
        probe = "import json,sys,torch,transformers,tokenizers; print(json.dumps(dict(python=sys.version,torch=torch.__version__,transformers=transformers.__version__,tokenizers=tokenizers.__version__,mps_available=torch.backends.mps.is_available())))"
        runtime = json.loads(subprocess.run([identity['python'], '-c', probe], cwd=frozen,
                             env=environment, text=True, capture_output=True, check=True).stdout)
        if args.device == 'mps' and not runtime['mps_available']:
            raise RuntimeError('本机MPS不可用，保留候选，不能静默更换执行设备')
        (work / 'local_environment.json').write_text(json.dumps(runtime, ensure_ascii=False, indent=2) + '\n')
        log.info('开始一次本地留出生成：240题，设备=%s，原始回答=%s', args.device, output)
        with (work / 'generation.log').open('a', encoding='utf-8') as stream:
            subprocess.run(command, cwd=frozen, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        rows = validate_answers(output, questions, identity)
    if sha256_file(selection_path) != identity['selection_sha256'] or sha256_file(questions) != HOLDOUT_SHA:
        raise ValueError('生成期间候选选择或题集被修改，不能发布有效验收绑定')
    binding = dict(identity, time=time.time(), answers_sha256=sha256_file(output), questions=len(rows),
                   status='generated_pending_quality_review',
                   environment_sha256=sha256_file(work / 'local_environment.json'),
                   precision_note='本地MPS/CPU为FP32；开发生成使用CUDA/BF16，分数差不能仅归因于训练。')
    (work / 'acceptance_binding.json').write_text(json.dumps(binding, ensure_ascii=False, indent=2) + '\n')
    log.info('留出原始回答及摘要绑定完成，尚需逐题质量复核：题数=%d', len(rows))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        main()
    except Exception as error:
        log.error('本地验收准备或生成未完成：类型=%s，原因=%s；保留所有已生成原始证据', type(error).__name__, error)
        raise
