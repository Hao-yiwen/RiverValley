# 第三方代码和依赖

截至 2026-11-16，本次检查确认 `train_reason.py` 与 `train_grpo.py` 的文件说明引用了 [MiniMind](https://github.com/jingyaogong/minimind)。其当前[许可证](https://github.com/jingyaogong/minimind/blob/master/LICENSE)为 Apache-2.0。根目录 NOTICE 保留引用关系；本次没有将完整上游代码树或依赖源码打包。

| 组件 | 用途 | 上游许可证入口 |
| --- | --- | --- |
| PyTorch | 张量计算、训练、推理、DDP | [LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| Transformers | 模型保存、加载、生成接口 | [LICENSE](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| Tokenizers | BPE 分词器 | [LICENSE](https://github.com/huggingface/tokenizers/blob/main/LICENSE) |
| Safetensors | 权重格式 | [LICENSE](https://github.com/huggingface/safetensors/blob/main/LICENSE) |
| NumPy | memmap 和数值处理 | [LICENSE](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| PyArrow | Parquet 语料处理 | [LICENSE](https://github.com/apache/arrow/blob/main/LICENSE.txt) |
| Requests | 下载及服务请求 | [LICENSE](https://github.com/psf/requests/blob/main/LICENSE) |
| langid.py | 语言识别 | [LICENSE](https://github.com/saffsd/langid.py/blob/master/LICENSE) |
| SwanLab | 可选训练监控 | [LICENSE](https://github.com/SwanHubX/SwanLab/blob/main/LICENSE) |
| pytest | 测试 | [LICENSE](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |

安装依赖时以上游发行包所携带的具体版本条款为准。`requirements-chat.txt` 固定本次已验证的推理环境；`requirements.txt` 为训练版本范围，尚不是所有平台的完整锁文件。未把“测试通过”当作完整供应链或依赖漏洞审计结论。

Google Drive / Colab 的账户登录、CLI 与可选依赖不属于普通本地推理的必要条件。历史云端编排脚本需要使用者自己的项目、目录、认证和额外工具；不随公开包提供原作者的云端配置、账户状态或训练恢复文件。
