# 参与贡献

建议使用 Python 3.12，在项目根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-chat.txt -r requirements-dev.txt
python -m pytest tests/ -q
```

源码统一使用 UTF-8。代码注释和关键运行日志使用中文。仓库内部导入以 `src/` 为根，例如 `from config import RiverValleyConfig`，不要增加 `src.` 前缀。

提交训练逻辑变更时，说明数据签名、标签对齐、损失归一化、优化器或恢复状态是否变化。对错误分支、状态转换、网络操作和保存入口保留有上下文的日志，日志中不输出凭据。数学逻辑、断点恢复和发布校验的变更需要相应测试。

模型质量结论须绑定权重、题集、源码和解码参数摘要。区分训练完成、内部开发集结果、留出结果和用户验收；不要根据留出反复选择候选。无需为了文档或简单样式修改增加重复测试。

请在 PR 中说明问题、最终行为、验证方式及已知限制。新增第三方代码或数据来源时补充出处、版本与许可说明。贡献的原创源码沿用本仓库 Apache-2.0；第三方材料仍按其原有许可证处理。
