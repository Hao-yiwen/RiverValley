# 云端训练通用指南

本页只提供可替换的部署示例。服务器连接地址、登录凭据、个人云项目、Drive 目录和账户状态由使用者在本机提供。

## 准备环境

在自己的 GPU 主机安装 Python 3.12 及与驱动兼容的 PyTorch，然后安装项目依赖。根据实际资源准备 tokenizer、memmap 或 SFT JSONL、输出目录及配置文件，先运行 CPU 测试和材料检查。

```bash
python -m pip install -r requirements.txt
python src/scripts/pretrain.py --config /absolute/path/to/pretrain.json
python src/scripts/sft.py --config /absolute/path/to/sft.json
```

长任务可放在 tmux 会话中运行，并将输出写入日志：

```bash
mkdir -p logs
tmux new -s train
torchrun --nproc_per_node=2 src/scripts/pretrain.py \
  --config /absolute/path/to/pretrain.json 2>&1 | tee logs/train.log
```

GPU 数量、显存、磁盘容量和路径都应以当前机器实际情况为准。`/root/autodl-tmp/` 与 `/content/river-valley-run/` 是历史配置示例，需要按部署环境覆盖。

## 恢复与交付

完整中断恢复依赖模型、优化器、RNG、数据游标、配置和材料签名。新优化阶段与阶段内恢复采用不同的入口，不应绕过已有的签名检查。保留模型和最佳模型的成对恢复关系；远端持久化完成后按服务器内容摘要核验。

历史云端编排需要使用者自己的认证、配额项目和存储目录。例如相关准备脚本要求显式传入 `--quota-project`，部分入口还要求 `--drive-root-id`、`--init-folder-id` 或 `--drive-parent-id`，具体见各脚本的 `--help`。

最新模型已完成训练，公开或私有模型分发只使用推理所需的权重、分词器、配置和冻结源码。不要为上传模型重新启动训练实例。最新训练链、质量结果及来源核查见 [模型卡](../release/model_card.md)、[数据来源](data_sources.md)和[发布说明](releasing.md)。
