"""在CPU按真实SFT去重顺序和恢复采样器统计监督量，不额外调用模型。"""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from data.dataset import SFTDataset
from tokenizer.bpe import load_tokenizer, get_pad_id
from training_utils import ResumableBatchSampler
from colab_train_pipeline import sha256_file

log=logging.getLogger(__name__)


def selected_token_count(lengths, steps, batch_size, accumulation_steps, seed):
    sampler=ResumableBatchSampler(len(lengths),batch_size,accumulation_steps,seed)
    completed=tokens=examples=0
    while completed < steps:
        sampler.set_step(completed)
        iterator=iter(sampler)
        remaining=min(steps-completed,sampler.steps_per_epoch-completed%sampler.steps_per_epoch)
        for _ in range(remaining):
            for _ in range(accumulation_steps):
                indexes=next(iterator)
                tokens+=sum(lengths[i] for i in indexes);examples+=len(indexes)
            completed+=1
    return {'updates':completed,'sample_presentations':examples,'supervised_tokens':tokens,
            'steps_per_epoch':sampler.steps_per_epoch}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True)
    parser.add_argument('--materials',required=True,help='本机包含data和tokenizer.json的冻结材料根目录')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();plan=json.loads(Path(args.plan).read_text());root=Path(args.materials)
    if sha256_file(root/'tokenizer.json')!=plan['input_sha256']['/content/river-valley-run/tokenizer.json']:
        raise ValueError('tokenizer摘要不符')
    tokenizer=load_tokenizer(str(root/'tokenizer.json'));results={}
    for stage in plan['stages']:
        if stage['kind']!='sft':continue
        config=stage['config'];remote=config['data_path'];relative=remote.removeprefix('/content/river-valley-run/')
        if relative==remote:raise ValueError('未知冻结材料路径')
        path=root/relative
        if sha256_file(path)!=plan['input_sha256'][remote]:raise ValueError('数据摘要不符')
        dataset=SFTDataset(str(path),tokenizer,config['max_seq_len'],get_pad_id(tokenizer))
        unique={}
        for sample in dataset.samples:
            key=json.dumps(sample['conversations'],ensure_ascii=False,sort_keys=True)
            unique.setdefault(key,sample)
        dataset=SFTDataset(list(unique.values()),tokenizer,config['max_seq_len'],get_pad_id(tokenizer))
        lengths=[int(dataset[i][2].sum()) for i in range(len(dataset))]
        result=selected_token_count(lengths,stage['target_steps'],config['batch_size'],config['accumulation_steps'],config['random_seed'])
        result.update(unique_examples=len(dataset),data_sha256=plan['input_sha256'][remote],mode='相同CPU采样器精确重放索引；无模型推理',tokens_per_single_pass=sum(lengths))
        results[stage['id']]=result;log.info('监督量核算：阶段=%s %s',stage['id'],result)
    Path(args.output).write_text(json.dumps({'plan_sha256':sha256_file(Path(args.plan)),'stages':results},ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    main()
