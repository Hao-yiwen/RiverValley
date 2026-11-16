"""用冻结权重区分监督链路、训练集拟合和跨样本生成问题，不修改模型。"""
import argparse
from collections import defaultdict
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import time

log = logging.getLogger(__name__)


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def select_samples(rows, per_group):
    """按固定内容摘要抽样，不读取旧留出题，不根据模型回答挑题。"""
    groups = defaultdict(list)
    for row in rows:
        if row['task'] in {'general', 'translation', 'format', 'arithmetic', 'extraction', 'multi_turn'}:
            groups[(row['task'], row['language'])].append(row)
    return [row for key in sorted(groups) for row in sorted(groups[key], key=lambda item: item['id'])[:per_group]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--delivery', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--per-group', type=int, default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    delivery, data, output = map(lambda x: Path(x).resolve(), (args.delivery, args.data, args.output))
    if output.exists():
        raise FileExistsError('诊断输出已存在，拒绝覆盖原始证据')
    if args.per_group < 1 or args.max_new_tokens < 1:
        raise ValueError('抽样数和生成长度必须为正')
    code = delivery / 'code/river-valley'
    sys.path[:0] = [str(code / 'src'), str(code / 'src/scripts')]
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    from model import RiverValleyModel
    from data.dataset import SFTDataset

    torch.set_num_threads(4)
    receipt = json.loads((delivery / 'delivery_receipt.json').read_text(encoding='utf-8'))
    model_dir = delivery / 'sft_best'
    model_hash = digest(model_dir / 'model.safetensors')
    if model_hash != receipt['sha256']:
        raise ValueError('待测权重与交付摘要不一致')
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    samples = []
    for split in ('train', 'val'):
        filename = f'sft_{split}.jsonl'
        if digest(data / filename) != manifest['hashes'][filename]:
            raise ValueError('原训练材料摘要不一致：' + filename)
        rows = [json.loads(line) for line in (data / filename).read_text(encoding='utf-8').splitlines()]
        samples.extend(dict(row, audit_split=split) for row in select_samples(rows, args.per_group))
    output.mkdir(parents=True)
    frozen = {'time': time.time(), 'model_sha256': model_hash, 'data_hashes': manifest['hashes'],
              'script_sha256': digest(__file__), 'device': args.device, 'dtype': 'float32',
              'per_group': args.per_group, 'max_new_tokens': args.max_new_tokens,
              'sample_ids': [row['id'] for row in samples], 'status': 'running',
              'scope': '训练/原验证样本诊断，不是独立验收；不读取旧240题留出，不训练权重。'}
    (output / 'binding.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('开始冻结权重诊断：模型=%s 样本=%d 设备=%s', model_hash, len(samples), args.device)
    tok = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    hf = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = RiverValleyModel.from_pretrained(model_dir, local_files_only=True, dtype=torch.float32).eval().to(args.device)
    ds = SFTDataset(samples, tok, model.config.max_seq_len, hf.pad_token_id)
    if len(ds) != len(samples):
        raise ValueError('诊断样本被数据集过滤，不能继续比较')
    results = []
    with torch.inference_mode(), (output / 'samples.jsonl').open('w', encoding='utf-8') as sink:
        for index, row in enumerate(samples):
            started = time.monotonic()
            x, y, mask, attention = ds[index]
            length = int(attention.sum())
            x, y, mask, attention = [v[:length].unsqueeze(0).to(args.device) for v in (x, y, mask, attention)]
            logits = model(x, attention_mask=attention, use_cache=False).logits
            selected = mask.bool()
            losses = F.cross_entropy(logits[selected].float(), y[selected], reduction='none')
            predictions = logits.argmax(-1)
            content = selected & (y != hf.convert_tokens_to_ids('<|im_end|>')) & (y != tok.token_to_id('Ċ'))
            turns = row['conversations']
            last = max(i for i, turn in enumerate(turns) if turn['role'] == 'assistant')
            messages, reference = turns[:last], turns[last]['content']
            ids = hf.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt').to(args.device)
            # 只比原始前缀未被长样本截断的情况，避免把窗口差异误判为注意力错误。
            prefix_matches = ids.shape[1] < x.shape[1] and torch.equal(ids, x[:, :ids.shape[1]])
            first = model(ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0, -1]
            item = {'id': row['id'], 'split': row['audit_split'], 'task': row['task'], 'language': row['language'],
                    'source': row['source'], 'messages': messages, 'reference': reference,
                    'teacher_forced_loss': float(losses.mean()), 'supervised_tokens': int(selected.sum()),
                    'teacher_forced_correct': int((predictions[selected] == y[selected]).sum()),
                    'content_tokens': int(content.sum()), 'content_correct': int((predictions[content] == y[content]).sum()),
                    'training_prefix_matches_generation': bool(prefix_matches)}
            if prefix_matches:
                target = y[0, ids.shape[1] - 1]
                item.update(first_target=hf.decode([target.item()]), first_prediction=hf.decode([first.argmax().item()]),
                            first_target_probability=float(first.softmax(-1)[target]),
                            teacher_vs_prefix_max_logit_difference=float((first - logits[0, ids.shape[1] - 1]).abs().max()))
            del logits, predictions, losses, selected, content, x, y, mask, attention, first
            allowed = min(args.max_new_tokens, model.config.max_seq_len - ids.shape[1])
            if allowed <= 0:
                item.update(answer=None, skipped='上下文没有生成空间')
            else:
                generated = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False,
                    repetition_penalty=1.0, max_new_tokens=allowed, eos_token_id=[1, 3], pad_token_id=hf.pad_token_id)
                new = generated[0, ids.shape[1]:]
                answer = hf.decode(new, skip_special_tokens=True)
                item.update(answer=answer, exact_match=answer.strip() == reference.strip(),
                            generated_tokens=len(new), finish_reason='stop' if new[-1].item() in (1, 3) else 'length')
                del generated, new
            item['elapsed_seconds'] = time.monotonic() - started
            sink.write(json.dumps(item, ensure_ascii=False) + '\n'); sink.flush()
            results.append(item)
            log.info('诊断完成：%d/%d 分组=%s/%s/%s loss=%.4f 首token概率=%s 完全匹配=%s',
                     index + 1, len(samples), item['split'], item['task'], item['language'],
                     item['teacher_forced_loss'], item.get('first_target_probability'), item.get('exact_match'))
    groups = defaultdict(list)
    for row in results:
        groups[(row['split'], row['task'])].append(row)
    summary = {'binding': frozen, 'time': time.time(), 'complete': True, 'groups': {}}
    for key, rows in groups.items():
        tokens = sum(r['supervised_tokens'] for r in rows)
        content_tokens = sum(r['content_tokens'] for r in rows)
        summary['groups']['/'.join(key)] = {'rows': len(rows), 'teacher_forced_loss': sum(r['teacher_forced_loss'] * r['supervised_tokens'] for r in rows) / tokens,
            'token_accuracy': sum(r['teacher_forced_correct'] for r in rows) / tokens,
            'content_accuracy': sum(r['content_correct'] for r in rows) / content_tokens if content_tokens else None,
            'free_generation_exact': sum(r.get('exact_match', False) for r in rows),
            'prefix_mismatches': sum(not r['training_prefix_matches_generation'] for r in rows)}
    summary['samples_sha256'] = digest(output / 'samples.jsonl')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.info('冻结权重诊断全部完成：输出=%s，权重未修改', output)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        main()
    except Exception:
        log.exception('行为诊断退出，保留已有原始输出，不自动重跑')
        raise
