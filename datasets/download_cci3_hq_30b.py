# -*- coding: utf-8 -*-
"""
CCI3-HQ 30B Tokens 数据下载脚本

使用方法:
    python download_cci3_hq_30b.py
    python download_cci3_hq_30b.py --output_dir /root/autodl-fs/cci3_hq_30b
    python download_cci3_hq_30b.py --num_proc 16
    python download_cci3_hq_30b.py --no_mirror  # 不使用镜像源

注意: 需要先在 HuggingFace 同意数据集协议并登录
    huggingface-cli login
"""

import os
import argparse


def setup_mirror():
    """设置 HuggingFace 镜像源加速下载"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("已启用 HuggingFace 镜像源: https://hf-mirror.com")


def download_data(output_dir: str = "/root/autodl-fs/cci3_hq_30b", num_proc: int = 8):
    """
    下载 CCI3-HQ 30B tokens 数据

    Args:
        output_dir: 保存目录
        num_proc: 并行下载进程数
    """
    from datasets import load_dataset

    print("=" * 60)
    print("下载 CCI3-HQ 30B tokens 数据")
    print("=" * 60)

    print(f"正在从 HuggingFace 下载数据集...")
    print(f"数据集: BAAI/CCI3-HQ (~500GB)")
    print(f"分片: train[:24%] (~30B tokens)")
    print(f"并行进程数: {num_proc}")
    print(f"保存目录: {output_dir}")
    print()

    # 下载 24%（约 30B tokens）
    dataset = load_dataset(
        "BAAI/CCI3-HQ",
        split="train[:24%]",
        num_proc=num_proc,
    )

    # 保存到磁盘
    print(f"\n正在保存到 {output_dir}...")
    dataset.save_to_disk(output_dir)

    print(f"\n下载完成!")
    print(f"  - 文档数: {len(dataset):,}")
    print(f"  - 保存路径: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CCI3-HQ 30B Tokens 数据下载")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-fs/cci3_hq_30b",
        help="数据保存目录",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="并行下载进程数",
    )
    parser.add_argument(
        "--no_mirror",
        action="store_true",
        help="不使用镜像源（直接从 HuggingFace 官方下载）",
    )

    args = parser.parse_args()

    # 默认启用镜像加速
    if not args.no_mirror:
        setup_mirror()

    download_data(output_dir=args.output_dir, num_proc=args.num_proc)


if __name__ == "__main__":
    main()
