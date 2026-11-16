# -*- coding: utf-8 -*-
"""
FineWeb-Edu 30B Tokens 数据下载脚本

使用方法:
    python download_fineweb_30b.py
    python download_fineweb_30b.py --output_dir /root/autodl-fs/fineweb_edu_30b
    python download_fineweb_30b.py --num_proc 16
    python download_fineweb_30b.py --no_mirror  # 不使用镜像源
"""

import os
import argparse


def setup_mirror():
    """设置 HuggingFace 镜像源加速下载"""
    # hf-mirror.com 是国内常用的 HuggingFace 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("已启用 HuggingFace 镜像源: https://hf-mirror.com")


def download_data(output_dir: str = "/root/autodl-fs/fineweb_edu_30b", num_proc: int = 8):
    """
    下载 FineWeb-Edu 30B tokens 原始数据

    Args:
        output_dir: 保存目录
        num_proc: 并行下载进程数
    """
    from datasets import load_dataset

    print("=" * 60)
    print("下载 FineWeb-Edu 30B tokens 原始数据")
    print("=" * 60)

    print(f"正在从 HuggingFace 下载数据集...")
    print(f"数据集: HuggingFaceFW/fineweb-edu (sample-100BT)")
    print(f"分片: train[:30%]")
    print(f"并行进程数: {num_proc}")
    print(f"保存目录: {output_dir}")
    print()

    # 下载 sample-100BT 的前 30%（约 30B tokens）
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-100BT",
        split="train[:30%]",
        num_proc=num_proc,
    )

    # 保存到磁盘
    print(f"\n正在保存到 {output_dir}...")
    dataset.save_to_disk(output_dir)

    print(f"\n下载完成!")
    print(f"  - 文档数: {len(dataset):,}")
    print(f"  - 保存路径: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="FineWeb-Edu 30B Tokens 数据下载")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-fs/fineweb_edu_30b",
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
