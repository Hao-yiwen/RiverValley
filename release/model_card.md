---
language:
  - zh
  - en
tags:
  - text-generation
  - pytorch
  - custom-architecture
  - experimental
inference: false
datasets:
  - opencsg/chinese-fineweb-edu
  - opencsg/Fineweb-Edu-Chinese-V2.1
  - HuggingFaceFW/fineweb-edu
  - m-a-p/COIG-CQIA
  - Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese
  - HuggingFaceTB/smol-smoltalk
  - bdx33/tatoeba-hsk-cmn-eng-fra
---

# RiverValley 1B — 2026-11-10 实验检查点

**本检查点已公开下载，但训练数据条款、署名及权重许可证仍未核实完成。公开下载不代表已取得开放许可证，也不代表可以自由商用。** 源码采用 Apache-2.0，具体范围见 `LICENSE_CODE` 与 `NOTICE_CODE`；这不构成对模型权重或第三方训练数据的 Apache-2.0 授权。

2026-11-16，所有者在公开独立源码仓库后，明确要求公开最新实验权重。许可询问仍待上游答复，本模型卡保留全部已知缺口，不将该请求视为第三方授权或许可核查完成。原开发仓库继续私有。

这是从零预训练实验继续迭代得到的中英文稠密语言模型，适合研究训练流程、数据清洗、指令修复和评估，不是可靠的通用聊天助手。常识、算术、翻译和开放问答仍会出现明显错误。项目内评分由 Codex 逐题复核，尚无独立人工验收或标准公开基准成绩。

## 模型和来源

| 项目 | 内容 |
| --- | --- |
| 架构 | 自定义 `rivervalley`，LLaMA 风格 decoder-only，GQA / RoPE / SwiGLU / RMSNorm |
| 参数量 | 1,012,488,192，共享 embedding 与输出头 |
| 层数 / 隐藏维度 | 21 / 2048 |
| 注意力头 / KV 头 | 16 / 4，head_dim=128 |
| FFN / 词表 | 5632 / 32,000 |
| 上下文 | 1024 token，包含输入和生成输出 |
| 权重格式 | FP32 safetensors，4,049,971,704 字节 |
| 版本 | `translation_curriculum_20261110`，最后阶段 160 步 |
| 预训练输入处理量 | 累计 2,999,844,864 token，不等同于独立语料量 |
| SHA256 | `1e8c6b939ae4cf17aa7fb0e94d8d6f0da0628c47a64b25da4ebec032820877e4` |

训练链为原约 10 亿 token 预训练 → 新增约 20 亿 token 继续预训练 → 正式 SFT 864 步 → 行为修复选取 250 步候选 → 翻译修复 160 步。它没有执行 DPO、GRPO 或思维链蒸馏。最终短修复使用新 AdamW、有效 batch 64、上下文 1024、学习率 1e-5 到 1e-6；10,240 条材料中翻译占 70%，其余为旧任务回放。

## 评估及限制

下表是本次交付保存的既有结果，本次发布准备没有重新训练或重跑完整题集。统一 MPS FP32、ChatML、贪心解码、最多 160 个新 token、repetition_penalty=1.0。分数为内部 0/1/2 语义规则，2 分表示完整正确，不能当作外部标准 benchmark。

| 题集 | 起点 | 本候选 |
| --- | --- | --- |
| 新翻译开发集，48 题 / 12 句组 | 12/96 分，3/48 完整正确 | 32/96 分，9/48 完整正确 |
| 旧开发回归集，100 题 | 60/200 分，23/100 完整正确 | 73/200 分，29/100 完整正确 |
| 新翻译留出，48 题 / 24 句组 | 未评估同题起点 | 40/96 分，12/48 完整正确 |

开发集每组包含两个方向和两种措辞，48 道题不代表 48 个独立问题。候选先按开发结果锁定，再使用新留出；没有同题起点留出对照，不能声称留出相对提升。分区排除了当前 SFT 的句组交叉，但无法确认公开句子是否出现在更早预训练中。

留出中两个源句组有不完整或错字。主要结果保留全部 48 题；事后排除对应 4 题的敏感性统计为 35/88 分、10/44 完整正确，未参与选优。原始回答与题集的摘要见 `evaluation_summary.json`，本包没有分发包含第三方句子的原始题目或回答。

候选仍会混淆词义、施害者与受害者、数字和否定；旧开发常识仅 3/20 分、算术 4/40 分。局部修复不能解释为任意措辞或通用聊天可靠，也没有完成系统性安全、偏见、隐私记忆或对抗评估。

## 使用

需要 Python 3.12 和包内固定依赖。可匿名下载检查点；自定义架构尚未注册到原生 Transformers Auto 类，需使用随检查点保存的冻结源码：

```bash
hf download yiwenX/RiverValley-1B-Experimental-20261110 \
  --revision translation-curriculum-20261110 \
  --local-dir /absolute/path/to/downloaded-model
cd /absolute/path/to/downloaded-model
python -m pip install -r requirements-chat.txt
python rivervalley_code/src/scripts/inference.py \
  --model "$PWD" --prompt '请提取数字：这里有12本书。' \
  --max_tokens 160 --greedy --repetition_penalty 1.0
```

FP32 权重约 4.05 GB，实际推理需要额外内存。默认使用 CUDA 或 CPU；其他设备需单独验证。输入与生成预算合计不能超过 1024 token。此命令用于功能验证，不保证回答正确。

## 数据与许可待办

完整来源、固定版本和排除关系见 `DATA_SOURCES.md`。OpenCSG 社区协议与商用授权条件、COIG-CQIA 和 Magpie 的授权，以及 Tatoeba 作者署名仍需核实。所有者选择在这些事项未完成时先公开下载，不能据此确定权重许可证或认为上游已经授权。数据卡的顶层标签不能覆盖正文的附加条款。当前不填写未经确认的权重许可证，也不声明可自由商用。

已核对固定训练版本的许可正文，并发送 OpenCSG 邮件以及 COIG、Magpie 现有许可讨论留言，尚待明确答复。Tatoeba 已整理训练材料并集的 23,052 个句子 ID 与当前所有者索引，其中 582 个孤儿句、1 个未匹配句及完整历史作者的署名仍需确认。相关索引和证据随包附在 `TATOEBA_ATTRIBUTION.json` 与 `LICENSE_REVIEW.json`，不包含原始句子正文；对应源码文件为 `release/tatoeba_attribution.json` 与 `release/license_review.json`。

包内只包含推理所需权重、分词器、配置、冻结推理源码、依赖、统计和文件摘要，不含训练集、账户记录、云端目录 ID、优化器、RNG 或训练游标，不能直接作为完整中断恢复包。

## 发布关联

公开源码位于 [Hao-yiwen/RiverValley 的 main 分支](https://github.com/Hao-yiwen/RiverValley/tree/main)。模型包内包含与本检查点摘要绑定的冻结推理源码；开发分支之后的文档或部署修改不会改变这份冻结源码。实际 Hub revision 由上传完成后对服务器文件进行核验的回执记录，权重公开版本 tag 待许可核查完成后确定。
