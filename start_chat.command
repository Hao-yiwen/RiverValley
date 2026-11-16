#!/bin/zsh
# 从脚本位置进入项目，确保双击启动和终端启动使用同一套权重。
set -eu
cd "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print '缺少 .venv 环境，请先按 docs/local_chat.md 安装依赖。'
  exit 1
fi
if curl --silent --fail --max-time 2 http://127.0.0.1:8765/api/status >/dev/null; then
  print '本地聊天服务已在运行：http://127.0.0.1:8765'
  open http://127.0.0.1:8765
  exit 0
fi
mkdir -p logs
print '正在启动 RiverValley 本地聊天；模型首次导入可能需要约一分钟。'
print '页面：http://127.0.0.1:8765；在此终端按 Ctrl+C 停止服务并释放模型内存。'
print "$$" > logs/local_chat.pid
open http://127.0.0.1:8765
exec .venv/bin/python -u src/scripts/local_chat.py
