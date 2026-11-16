"""锁定并下载增量预训练语料，不分配 GPU、不改变第一版材料。"""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from pathlib import Path
import random

from download_bilingual_sources import download, session

log = logging.getLogger(__name__)
ROOT = Path('checkpoints/bilingual_v2_20261019')
LOCK = Path('configs/v2_pretrain_sources.lock.json')


def main():
    client = session()
    if not LOCK.exists():
        repo = 'opencsg/Fineweb-Edu-Chinese-V2.1'
        response = client.get(f'https://huggingface.co/api/datasets/{repo}', timeout=90)
        response.raise_for_status()
        meta = response.json()
        revision = meta['sha']
        url = f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/4_5?limit=1000'
        files = []
        while url:
            response = client.get(url, timeout=90)
            response.raise_for_status()
            files.extend(x for x in response.json() if x['type'] == 'file' and x['path'].endswith('.parquet'))
            url = response.links.get('next', {}).get('url')
        random.Random(20261019).shuffle(files)
        # 从全部高分分片确定性采样，避免按磁盘首段只取某一来源。
        zh = [{'filename': f"pretrain_zh_{x['path'].split('/')[-1]}", 'size_bytes': x['size'],
               'source_path': x['path'], 'url': f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{x['path']}"}
              for x in files[:600]]
        old = json.loads(Path('configs/bilingual_sources.lock.json').read_text())
        english = next(x for x in old if x['name'] == 'pretrain_en')
        response = client.get('https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/' + english['parquet_revision'] + '/sample-10BT/train', params={'limit': 1000}, timeout=90)
        response.raise_for_status()
        item = next(x for x in response.json() if x['path'].endswith('/0001.parquet'))
        en = [{'filename': 'pretrain_en_0001.parquet', 'size_bytes': item['size'], 'source_path': item['path'],
               'url': f"https://huggingface.co/datasets/{english['dataset']}/resolve/{english['parquet_revision']}/{item['path']}"}]
        records = [{'name': 'pretrain_zh', 'dataset': repo, 'source_revision': revision,
                    'selection': '4_5 分数目录，随机种子20261019抽取600分片', 'card_license': meta.get('cardData', {}).get('license'), 'files': zh},
                   {**english, 'files': en, 'selection': '第一版未使用的0001分片；清洗时再做跨版本去重'}]
        LOCK.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    records = json.loads(LOCK.read_text(encoding='utf-8'))
    target = ROOT / 'pretrain_raw'
    target.mkdir(parents=True, exist_ok=True)
    items = [item for source in records for item in source['files']]
    results = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for result in pool.map(lambda item: download(item, target), items):
            results.append(result)
            if len(results) % 25 == 0:
                log.warning('增量材料下载进度：%d/%d', len(results), len(items))
    (ROOT / 'pretrain_download_manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    log.warning('增量预训练材料全部下载完成，共 %d 个文件', len(items))


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(message)s')
    main()
