# 发布操作

采用双仓布局：GitHub 管理源码、文档、测试与数据来源锁文件；Hugging Face Hub 管理权重、分词器、模型卡与统计。模型已上传到 [yiwenX/RiverValley-1B-Experimental-20261110](https://huggingface.co/yiwenX/RiverValley-1B-Experimental-20261110)，当前为私有仓库。上传文件通过服务器核验，实际版本和权重身份见 [上传回执](../release/huggingface_upload.json)。

所有者在 2026-11-16 授权先公开源码：从已经扫描的无历史快照独立建立 [Hao-yiwen/RiverValley](https://github.com/Hao-yiwen/RiverValley)，原开发仓库和 Hugging Face 权重继续私有。`release/policy.json` 分别记录源码发布范围及原仓库、权重的私有要求；源码公开不会解除其余范围的限制。

## 生成可审阅的发布产物

在项目根目录执行，输出目录必须不存在：

```bash
python src/scripts/prepare_open_source_release.py \
  --output dist/open-source-20261116
```

没有本地检查点时，可运行 `--source-only --output dist/source-only`。脚本只写本机，既不申请 GPU，也不上传任何内容。它读取当前工作区，因此能包含尚未提交的新版本实现。源码白名单排除 `.git`、本机助手配置、云端私有运行文档、实验原始证据、日志与检查点目录，保留测试所需的项目基础题集。匹配到的符号链接或常见凭据会使构建失败，失败时清理暂存目录。

完整产物结构：

```text
dist/open-source-20261116/
  source/                    # 可作为新公开仓库起点的源码快照
  source.tar.gz              # 同一快照的压缩包，不含旧历史
  huggingface/               # 仅供私有暂存的检查点包
  build_report.json          # 构建数量、压缩包摘要、未上传状态
```

Hub 包逐项校验 `release/checkpoint_assets.json` 固定的权重、配置、分词器、冻结推理源码和评估证据摘要。只提取统计白名单，不带出 Drive ID、账单、云端会话、训练句对和优化器状态。Hub 根目录的 `LICENSE_CODE` 仅授权附带源码，权重许可仍待解决。

许可跟进版本另附 `LICENSE_REVIEW.json` 和 `TATOEBA_ATTRIBUTION.json`，将证据索引、已发询问和署名缺口与权重一起保留；发布清单同时记录所有者要求继续私有的状态。

## 检查产物

```bash
cd dist/open-source-20261116/source
python -m pip install -r requirements-chat.txt -r requirements-dev.txt
python -m pytest tests/ -q
```

分别在源码与 Hub 包根目录使用 `sha256sum -c SHA256SUMS`；macOS 可用 `shasum -a 256 -c SHA256SUMS`。按 [快速开始](quickstart.md) 从 Hub 包的冻结源码进行一次真实加载和生成。摘要证明文件身份，不证明答案质量或许可已经满足。

## GitHub 源码首发

公开仓库从发布工具生成的 `source/` 目录初始化，只创建新的初始提交；不用旧仓库的 `.git`，不使用 GitHub fork 或模板复制历史，也不推送旧 refs。初始内容需重新核对摘要、已知敏感值、全部跟踪文件及测试，随后推送并核验 CI。引用元数据与模型卡使用新的公开源码地址。

以下清理记录只对应原私有开发仓库，原仓库不会随源码快照公开：

本轮已将最新 1B 版本合入 main，并使用 git-filter-repo 删除各历史中的私人云端连接文档和本机助手设置、替换已确认的敏感值。5 个远端可写分支全部核验，4 个发生历史更新；使用逐分支精确版本租约与原子推送，main 当前文件内容保持不变。仓库可见性仍为私有。

GitHub 旧 MR #1 / #3 的只读引用与缓存清理申请已经通过官方支持门户提交，提交成功页面已保存为本地受限证据，当前等待人工答复；尚未确认缓存清除。旧本地克隆、未提交工作和受限备份不在公开分发包中。旧克隆不能直接把原分支重新推回远端，否则会重新引入旧历史。清理 Git 文件不会使外部登录密码自动失效，账户所有者目前不能确认其失效，因此继续保持私有。平台完整清理步骤见 [GitHub 官方说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)。

原仓库若未来需要改为公开，仍须复核平台缓存及外部凭据处理结果。本次源码首发使用独立快照，不改变原仓库可见性，也不宣称旧缓存已清除。

## Hugging Face

先核对 `release/policy.json` 和 [数据来源说明](data_sources.md) 中的权重许可事项。当前可将包保存为私有草稿；公开发布必须等权重许可与署名完成。

固定版本数据卡的摘要与询问状态记录在 `release/license_review.json`。已发送 OpenCSG 邮件及 COIG、Magpie 许可讨论留言；等待答复期间不把缺失许可改写成 Apache-2.0。`release/tatoeba_attribution.json` 仅提供不含句子正文的来源和当前所有者索引，其覆盖范围及未匹配项不能省略。

本轮使用官方 SDK 上传并完成服务器核验。后续重新构建或上传新版本时，推荐使用同一入口，它会先检查本地包、固定资产和目标仓库私有状态，再上传并核对服务器内容：

```bash
hf auth whoami
HF_HUB_DISABLE_XET=1 python src/scripts/upload_huggingface_release.py \
  --package dist/open-source-20261116/huggingface \
  --repo-id yiwenX/RiverValley-1B-Experimental-20261110 \
  --receipt dist/huggingface_upload_receipt.json
```

本次 Xet 通道连续重试后切换为标准 HTTP/LFS，以上环境变量用于选择该通道。大权重按服务器报告的 LFS SHA256 和大小核验，其余文件下载后逐项 SHA256 核验；全部通过才原子写入完成回执。已有公开仓库会被入口拒绝，避免在许可待定时意外公开权重。

权重许可确认后再补充模型卡许可证和公开版本 tag，并重新构建校验。下载时用上传回执中的实际 Hub commit 锁定版本。不要把内部 `delivery_*` 整个目录上传到 Hub。
