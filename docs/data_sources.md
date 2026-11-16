# 数据来源与权重许可核查

核查日期：2026-11-16。源码按 Apache-2.0 在独立的 [RiverValley 仓库](https://github.com/Hao-yiwen/RiverValley) 发布；最新检查点的许可证尚未确定，权重继续私有。以下数据卡标签和正文存在差异的条目，不能只读取 `card_license` 字段就作出可商用结论。这里记录已找到的证据及未解决的问题，不把许可不明直接等同于禁止训练。固定版本、README 摘要和对外询问状态见源码仓库的 `release/license_review.json`。

## 最新检查点的训练链

本次候选为 `translation_curriculum_20261110` 的 160 步检查点。它继承正式 V2 SFT 864 步后的行为修复 250 步候选；V2 SFT 从累计预训练 2,999,844,864 输入 token 的基座初始化。新增约 20 亿 token 的继续预训练接续原约 10 亿 token 的基座，不能将它描述为重新从零训练的新基座。正式 SFT、行为修复和翻译修复均为明确的新优化阶段。

预训练输入 token 是训练处理量，不能据此声称同等数量的独立、唯一语料。最初的分词器由该训练流程沿用，需把其训练语料来源一并纳入许可核查。

## 来源清单

| 数据来源 | 本次追溯关系 | 已核实说明 | 发布前事项 |
| --- | --- | --- | --- |
| [opencsg/chinese-fineweb-edu](https://huggingface.co/datasets/opencsg/chinese-fineweb-edu) | 第一阶段中文预训练；锁文件 `bilingual_sources.lock.json` | 固定版本标签 Apache-2.0，正文同时要求社区协议及商用取得许可 | 已向卡片指定联系人发信，等待完整社区条款和下游权重分发条件 |
| [opencsg/Fineweb-Edu-Chinese-V2.1](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1) | 第二版新增中文预训练；锁文件 `v2_pretrain_sources.lock.json` | 固定训练版本已经包含 OpenCSG 社区协议及商用授权表述，并非近期网页新增 | 与第一阶段数据一并询问，不能默认无附加限制 |
| [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | 两阶段英文预训练 | ODC-By 1.0；卡片同时要求遵守 Common Crawl 条款 | 保留署名、来源和相应条款链接；数据库许可不代表所有网页文本采用同一许可 |
| [m-a-p/COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) 的 `coig_pc` / `wiki` | 正式 V2 SFT 与后续自然任务回放 | 固定卡片许可信息仍待补充；实际 `copyright` 字段多为空泛说明或作者简介 | 已在现有许可讨论中追问子集授权、署名及下游权重条件 |
| [Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese) | 正式 V2 SFT 与后续回放 | 固定卡片没有数据集许可证；框架代码的 MIT 不能自动覆盖本数据 | 已在现有许可讨论中询问数据许可、生成模型条件与下游分发要求 |
| [HuggingFaceTB/smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) | 正式 SFT 使用 openhermes、rewrite、summary、everyday-conversations 子来源，后续部分回放 | 卡片 Apache-2.0；混合数据需保留子来源信息 | 整理实际子来源及版本，不能抹去来源链 |
| [bdx33/tatoeba-hsk-cmn-eng-fra](https://huggingface.co/datasets/bdx33/tatoeba-hsk-cmn-eng-fra) / [Tatoeba](https://tatoeba.org/en/terms_of_use) | 正式 SFT、翻译修复和相应评估句对 | 整理版卡片 CC-BY-2.0；原站文本条款 CC-BY-2.0-FR，要求作者署名 | 已整理 23,052 个训练材料句子 ID；582 个孤儿句及 1 个缺失句的署名仍需确认，不分发原句对 |
| RiverValley 程序生成任务 | 格式、抽取、排序、算术、复制、多轮等任务 | 仓库包含生成器和规则校验 | 区分内部合成任务与自然语言通用能力，检查与外部文本的混合关系 |

固定 revision、分片选择和原下载地址保留在 `configs/*sources.lock.json` 与 `v2_tatoeba_source.lock.json`。已重新下载六项中文/指令/翻译来源的固定版本 README，记录 SHA256，确认核查当日的 main 与这些固定版本相同；英文预训练另核对其固定 README。没有修改旧锁文件以补造历史授权。

## 本轮许可核实与询问

两个 OpenCSG 数据卡在各自固定 revision 的正文中都要求社区协议，并提及商用须联系取得许可。已于 2026-11-16 向卡片指定联系人发送询问，要求明确从零训练的自有模型是否可以公开分发权重、完整社区条款版本、可采用的权重许可证与商用条件；尚未收到授权答复。不能仅凭顶层 Apache-2.0 标签排除这些要求。

COIG-CQIA 的 [许可讨论 #32](https://huggingface.co/datasets/m-a-p/COIG-CQIA/discussions/32#6afa7279de7795c7c04fc407) 和 Magpie 的 [许可讨论 #4](https://huggingface.co/datasets/Magpie-Align/Magpie-Qwen2-Pro-200K-Chinese/discussions/4#6afa72823deee809df7c5aa4) 已发布针对本次固定版本及下游权重的询问。COIG-PC 的 [上游数据卡](https://huggingface.co/datasets/BAAI/COIG-PC-core) 对没有单独声明的子集给出默认 Apache-2.0，同时保留子集许可优先；这不能直接证明 CQIA 的全部 `coig_pc` 和 `wiki` 已获同样授权。`wiki` 只是子集名称，不能推断其全部来自 Wikipedia 或统一采用某一许可。Magpie 框架代码的 MIT 与论文的许可也不能替代数据集授权。

感谢 [Tatoeba](https://tatoeba.org) 的文本贡献者及整理者 bdx33。本项目筛选、去重中英句对并添加翻译或修复指令，没有使用音频。原文本许可链接为 [CC-BY-2.0-FR](https://creativecommons.org/licenses/by/2.0/fr/)，不代表作者为本模型背书。源码仓库 `release/tatoeba_attribution.json` 提供训练材料并集的逐句链接模板、当前 owner 索引、三个阶段材料摘要和官方导出摘要；不含句子正文。23,052 个句子 ID 中，22,469 个能在 2026-11-14 官方导出中匹配到具名当前所有者，582 个为孤儿句，另 1 个未匹配。当前所有者未必是原作者或全部历史作者；旧版站点条款虽曾允许链接署名，不能据此声称现行署名要求已全部满足。该索引覆盖材料池，不表示短阶段训练读取过其中每条材料。

## 实际使用与排除

正式 SFT 的 `formal_candidate_05` 有 27,667 条训练样本。其中自生成 14,110 条、Tatoeba 8,793 条、COIG-CQIA coig_pc 1,430 条及 wiki 99 条、Magpie 519 条、smol-smoltalk rewrite 636 条 / openhermes 1,786 条 / summary 290 条 / everyday-conversations 4 条。它从预训练基座初始化，没有继承第一版 SFT 权重。

中间行为修复的材料池有 32,000 条：自生成 24,000 条、Tatoeba 5,128 条、COIG-CQIA coig_pc 752 条及 wiki 59 条、Magpie 288 条、smol-smoltalk openhermes 1,148 条 / rewrite 424 条 / summary 201 条。最新权重继承其中 250 步候选，不能忽略这一阶段，也不能把材料池行数等同于实际处理的独立样本数。

最新翻译修复的 10,240 条材料包括 Tatoeba 7,168 条、自生成修复任务 2,560 条、COIG-CQIA coig_pc 176 条和 wiki 11 条、Magpie 69 条、smol-smoltalk/openhermes 256 条。这只是最后一阶段的配比，不能替代完整训练链。

OPUS-100 和 Alpaca-GPT4 中文来源曾被下载或用于早期实验，但正式 V2 SFT 已排除它们。OpenCSG smoltalk-chinese 在候选草案中出现，正式候选 05 也已排除；本次不把这些来源计入最新权重的正式 SFT 来源。历史脚本、锁文件和旧版模型仍需保留各自的来源说明。

## 首发边界

GitHub 源码快照包含生成/下载代码、锁文件及项目自编的基础评估题，不包含原始训练分片、清洗后训练集、第三方翻译句对、账户记录或完整运行日志。Hub 暂存包包含权重、分词器、冻结推理源码及汇总统计，不包含训练集、原始翻译评估句子或优化器恢复文件。

在上述许可事项完成前，权重包保持本地或私有草稿状态，模型卡不填写 `license: apache-2.0`，也不作“可自由商用”的承诺。数据许可核实后，再确定权重许可、补充署名并记录授权依据。
