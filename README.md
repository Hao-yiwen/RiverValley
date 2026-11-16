# RiverValley

RiverValley 是一个从零训练中英文语言模型的实验项目，包含模型实现、语料准备、预训练与 SFT、断点恢复、评估、检查点交付和本地聊天工具。

**源码已按 Apache-2.0 开源，首发仓库为 [Hao-yiwen/RiverValley](https://github.com/Hao-yiwen/RiverValley)。** 本仓库从经过检查的源码快照建立，不继承旧开发仓库的 Git 历史，包含训练、评估和本地推理工具，不包含训练数据或模型权重。

最新检查点 **2026-11-10 翻译修复实验版**已在 [Hugging Face](https://huggingface.co/yiwenX/RiverValley-1B-Experimental-20261110) 公开下载，文件身份见 [上传回执](release/huggingface_upload.json)。**权重许可证、训练数据条款和署名仍在核查，公开下载不代表已取得开放许可或可自由商用。**当前模型在常识、算术、翻译和开放问答上仍有明显错误，适合学习与研究训练流程，尚未达到可靠通用助手的质量。

[快速开始](docs/quickstart.md) · [模型卡草稿](release/model_card.md) · [数据来源与许可](docs/data_sources.md) · [开源审查](docs/open_source_readiness.md) · [发布操作](docs/releasing.md)

## 最新模型

| 配置项 | 值 |
| --- | --- |
| 参数量 | 1,012,488,192（约 1.01B） |
| 结构 | 稠密 decoder-only，21 层，隐藏维度 2048 |
| 注意力 | GQA，16 个 Query 头 / 4 个 KV 头，head_dim=128 |
| FFN | SwiGLU，intermediate_size=5632 |
| 位置编码 / 归一化 | RoPE，theta=1,000,000 / RMSNorm |
| 词表 / 上下文 | 32,000 / 1024 token（输入与输出合计） |
| 输出头 | 与 Token Embedding 共享权重 |
| 交付权重 | FP32 safetensors，约 4.05 GB |
| 预训练输入处理量 | 累计 2,999,844,864 token |

实际训练链为：约 10 亿 token 预训练 → 新增约 20 亿 token 继续预训练 → 正式 SFT 864 步 → 行为修复选取 250 步候选 → 翻译修复 160 步。保留了旧版本、冻结源码及摘要绑定。代码还提供 MoE、YaRN、DPO、思维链蒸馏和 GRPO 研究路径，这些路径没有全部用于最新检查点。

## 当前评估

以下为交付时已有的内部评估，使用 MPS FP32、贪心解码、160 个新 token、重复惩罚 1.0，由 Codex 按 0/1/2 分规则逐题复核；尚无独立人工验收或标准公开 benchmark 成绩。

| 题集 | 修复前 | 最新候选 |
| --- | --- | --- |
| 翻译开发集，48 题 / 12 句组 | 12/96 分，3 题完整正确 | 32/96 分，9 题完整正确 |
| 旧开发回归集，100 题 | 60/200 分，23 题完整正确 | 73/200 分，29 题完整正确 |
| 新翻译留出，48 题 / 24 句组 | 未测同题起点 | 40/96 分，12 题完整正确 |

句子、方向及措辞之间存在相关性，不能把题目数量当作独立样本量。最新候选旧开发常识只有 3/20 分、算术 4/40 分；训练 loss 降低和局部任务改善均不能代表通用能力达标。留出源句质量问题、分区边界和评分限制见[模型卡](release/model_card.md)。

## 安装与推理

推荐 Python 3.12，固定推理依赖如下：

```bash
git clone https://github.com/Hao-yiwen/RiverValley.git
cd RiverValley
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-chat.txt
python src/scripts/inference.py \
  --model /absolute/path/to/model \
  --prompt '请提取数字：这里有12本书。' \
  --max_tokens 160 --greedy --repetition_penalty 1.0
```

检查点可从 Hugging Face 下载；使用前应查看模型卡中的许可缺口和能力限制。架构类是 `RiverValleyModel`，尚未注册为原生 Transformers Auto 模型；检查点包提供与权重绑定的冻结推理源码。[快速开始](docs/quickstart.md) 包含独立模型包运行方式和本地网页的 CUDA / MPS / CPU 配置。

## 训练和测试

```bash
python -m pip install -r requirements.txt
python src/scripts/pretrain.py --config /absolute/path/to/pretrain.json
python src/scripts/sft.py --config /absolute/path/to/sft.json

# 多卡训练按实际 GPU 数量设置。
torchrun --nproc_per_node=2 src/scripts/pretrain.py --config /absolute/path/to/pretrain.json

# 完整 CPU 测试；CUDA 专用检查需要相应硬件。
python -m pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

训练参数统一由 `src/config.py` 的配置类定义，预训练读取 memmap，SFT 读取 JSONL 并仅监督助手内容。使用者需要准备自己的数据、tokenizer 和输出路径；默认值及 A100 示例包含历史云端路径，应先按[快速开始](docs/quickstart.md)覆盖配置。训练数据不随本仓库分发，其来源和使用条件见[数据来源](docs/data_sources.md)。

## 仓库结构

```text
src/model/          模型、注意力、位置编码、损失与可选 MoE
src/data/           预训练、SFT 和偏好数据集
src/tokenizer/      BPE、ChatML 及分词器工具
src/scripts/        训练、恢复、评估、本地聊天和发布准备
src/web/            本地聊天页面
src/training_utils.py    训练更新、保存与恢复辅助
src/training_runtime.py  运行状态与运行边界
configs/            示例配置与数据来源锁文件
datasets/           下载、清洗、去重、采样和任务生成器
evals/              基础评估题与生成器
tests/              数学正确性、数据、恢复、交付和发布校验
docs/               使用、来源、审查与发布说明
release/            模型卡、发布策略和检查点摘要
```

本地开发目录还可能保留私有运行记录和交付证据；公开源码以发布工具生成的白名单快照为准。[发布操作](docs/releasing.md)解释了历史凭据处理、GitHub 源码与 Hugging Face 权重的分工。

## 许可证与贡献

原创源码使用 [Apache-2.0](LICENSE)，第三方说明见 [NOTICE](NOTICE) 和[依赖来源](docs/third_party.md)。源码许可不自动授权第三方数据或最新权重，权重许可待核实。欢迎参考[贡献指南](CONTRIBUTING.md)提交改进；安全问题见 [SECURITY.md](SECURITY.md)。
