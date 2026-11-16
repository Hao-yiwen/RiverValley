# 本地运行与训练入口

推荐 Python 3.12。已在本机验证 PyTorch 2.9.0、Transformers 4.57.3、Tokenizers 0.22.2、Safetensors 0.8.0。平台和 GPU 驱动组合需各自验证。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-chat.txt
```

本次首发只公开源码，最新检查点尚未开放下载。你可以使用自行训练得到的兼容检查点运行下方推理入口。维护者已经私有上传并核验模型包；以下下载命令仅适用于已经获得 [Hugging Face 私有模型仓库](https://huggingface.co/yiwenX/RiverValley-1B-Experimental-20261110) 访问权限的账号，普通访问者无法下载：

```bash
hf download yiwenX/RiverValley-1B-Experimental-20261110 \
  --revision 172441396568aef1230586da5993eac069526e34 \
  --local-dir checkpoints/huggingface_20261110
```

取得发布包后，将下面路径替换为包含 `config.json`、`model.safetensors` 和分词器文件的绝对目录：

```bash
python src/scripts/inference.py \
  --model /absolute/path/to/model \
  --prompt '请提取数字：这里有12本书。' \
  --max_tokens 160 --greedy --repetition_penalty 1.0
```

这是自定义 `rivervalley` 架构。使用本仓库的 `RiverValleyModel`，不要假定原生 `AutoModelForCausalLM` 已注册该架构。Hub 暂存包附带与交付检查点绑定的冻结源码，可以不依赖开发分支运行：

```bash
cd /absolute/path/to/hub-package
python -m pip install -r requirements-chat.txt
python rivervalley_code/src/scripts/inference.py \
  --model "$PWD" --prompt '请提取数字：这里有12本书。' \
  --max_tokens 160 --greedy --repetition_penalty 1.0
```

默认推理入口选择 CUDA（可用时）或 CPU。需要本地网页及 Apple MPS 时，在源码仓库中执行：

```bash
python src/scripts/local_chat.py \
  --model /absolute/path/to/hub-package \
  --code-root /absolute/path/to/hub-package/rivervalley_code \
  --device auto
```

打开终端显示的本机地址，默认 `http://127.0.0.1:8765`。没有本地历史交付目录时，需显式传入上面的自定义路径，直接双击旧启动脚本不会自动下载模型。FP32 权重本身约 4.05 GB，运行还需要额外内存。

## 训练

```bash
python -m pip install -r requirements.txt
python src/scripts/pretrain.py --config /absolute/path/to/pretrain.json
python src/scripts/sft.py --config /absolute/path/to/sft.json
torchrun --nproc_per_node=2 src/scripts/pretrain.py --config /absolute/path/to/pretrain.json
```

JSON 中的字段对应 `src/config.py` 的 `PretrainConfig` / `SFTConfig`。必须提供自己的 tokenizer、数据、输出与初始化检查点路径。`configs/colab_a100_1b_*.json` 是历史 A100 配方示例，路径指向 `/content/river-valley-run`；配置默认值还包含历史 AutoDL 路径，不能在新机器上无准备直接运行。数据生成及清洗入口位于 `datasets/`，预训练实际读取 memmap 的 `train.bin` 和 `meta.json`。

DPO、思维链蒸馏、GRPO 和 MoE 保留为研究实现；本次最新 1B 权重使用稠密模型，实际训练链为预训练、继续预训练、SFT 和两次短修复。不要把代码支持的所有阶段写成该检查点已执行的阶段。

历史云端自动编排需要额外的 Colab CLI、Google 认证、Drive 目录及用户自己的云项目；普通本地推理和 CPU 单元测试不需要这些账户或服务。
