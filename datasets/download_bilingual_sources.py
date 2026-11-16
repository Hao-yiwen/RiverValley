"""锁定公开数据的真实 Parquet revision，下载并记录内容摘要；不申请 GPU。"""
import argparse
import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)
SOURCES = [
    ("pretrain_zh", "opencsg/chinese-fineweb-edu", "default", [0, 1, 2]),
    ("pretrain_en", "HuggingFaceFW/fineweb-edu", "sample-10BT", [0]),
    ("sft_zh", "opencsg/smoltalk-chinese", "default", [0]),
    ("sft_en", "HuggingFaceTB/smol-smoltalk", "default", [0]),
]
SFT_SUPPLEMENT = [(f"sft_zh_cqia_{name}", "m-a-p/COIG-CQIA", name, [0])
                  for name in ("wiki", "coig_pc", "chinese_traditional", "wikihow", "zhihu")]
SFT_SUPPLEMENT.append(("sft_zh_alpaca_gpt4", "FreedomIntelligence/alpaca-gpt4-chinese", "default", [0]))


def session():
    """只重试幂等读取，避免短暂网络错误导致整个下载任务重来。"""
    client = requests.Session()
    client.mount("https://", HTTPAdapter(max_retries=Retry(
        total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
    return client


def sha256_file(path):
    """流式计算摘要，避免把数 GB 文件全部加载进内存。"""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lock_sources(path, sources=SOURCES):
    """记录原仓库与转换分支两个 revision，避免查看器数据被误当成原仓库快照。"""
    client = session()
    records = []
    for name, repo, config, indices in sources:
        def get(url, **kwargs):
            result = client.get(url, timeout=90, **kwargs)
            result.raise_for_status()
            return result.json()
        metadata = get(f"https://huggingface.co/api/datasets/{repo}")
        converted = get(f"https://huggingface.co/api/datasets/{repo}/revision/refs%2Fconvert%2Fparquet")
        files = get("https://datasets-server.huggingface.co/parquet", params={"dataset": repo})["parquet_files"]
        files = sorted((f for f in files if f["config"] == config and f["split"] == "train"), key=lambda f: f["filename"])
        selected = []
        for index in indices:
            item = files[index]
            tail = unquote(item["url"].split("/resolve/", 1)[1]).removeprefix("refs/convert/parquet/")
            selected.append({"url": f"https://huggingface.co/datasets/{repo}/resolve/{converted['sha']}/{tail}",
                             "size_bytes": item["size"], "filename": f"{name}_{index:04d}.parquet"})
        records.append({"name": name, "dataset": repo, "config": config, "split": "train",
                        "source_revision": metadata["sha"], "parquet_revision": converted["sha"],
                        "card_license": (metadata.get("cardData") or {}).get("license"), "files": selected})
        logger.info("锁定数据源: %s，文件数=%d，转换 revision=%s", name, len(selected), converted["sha"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def download(item, output):
    """使用 Range 继续未完成的文件，服务端不支持续传时重新下载该文件。"""
    target = output / item["filename"]
    temporary = target.with_suffix(".partial")
    if not target.exists():
        failures = 0
        while True:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == item["size_bytes"]:
                break
            end = min(item["size_bytes"] - 1, offset + 32 * 1024 * 1024 - 1)
            logger.info("开始下载: 文件=%s，已有=%d，总字节=%d，连续失败=%d", target.name, offset, item["size_bytes"], failures)
            try:
                # 有界请求避免慢速 CDN 在超大响应中长期占用连接；分块 URL 防止旧重定向缓存。
                with session().get(item["url"], params={"download": "true", "range_start": offset},
                                   headers={"Range": f"bytes={offset}-{end}"},
                                   stream=True, timeout=(30, 45)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {offset}-{end}/{item['size_bytes']}":
                        raise ValueError(f"续传起点不匹配: {target.name}")
                    with temporary.open("ab") as sink:
                        for block in response.iter_content(256 * 1024):
                            sink.write(block)
                if temporary.stat().st_size != end + 1:
                    raise ValueError(f"下载长度不匹配: {target.name}")
                failures = 0
            except (requests.RequestException, ValueError):
                logger.exception("下载中断: %s，保留临时文件用于续传", target.name)
                failures += 1
                if failures >= 20:
                    raise
        temporary.replace(target)
    if target.stat().st_size != item["size_bytes"]:
        raise ValueError(f"本地文件长度不匹配，请检查文件: {target}")
    digest = sha256_file(target)
    logger.info("下载完成: %s，字节=%d，SHA256=%s", target.name, target.stat().st_size, digest)
    return {**item, "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description="下载锁定版本的中英文训练材料")
    parser.add_argument("--lock", default="configs/bilingual_sources.lock.json")
    parser.add_argument("--output", default="checkpoints/bilingual_20261011/raw")
    parser.add_argument("--lock-only", action="store_true")
    parser.add_argument("--sft-supplement", action="store_true", help="锁定和下载中文知识与指令补充数据")
    args = parser.parse_args()
    lock = Path(args.lock)
    if args.sft_supplement and args.lock == "configs/bilingual_sources.lock.json":
        lock = Path("configs/bilingual_sft_sources.lock.json")
    if not lock.exists():
        lock_sources(lock, SFT_SUPPLEMENT if args.sft_supplement else SOURCES)
    if args.lock_only:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = json.loads(lock.read_text(encoding="utf-8"))
    files = [item for source in records for item in source["files"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        downloaded = list(pool.map(lambda item: download(item, output), files))
    receipt = "sft_supplement_download_manifest.json" if args.sft_supplement else "download_manifest.json"
    (output / receipt).write_text(json.dumps(downloaded, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
