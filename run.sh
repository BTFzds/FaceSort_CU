#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "  FaceSort 本地人脸照片整理工具"
echo "  100% 本地运行，数据不会上传"
echo "========================================"

if [ ! -d ".venv" ]; then
  echo "[1/3] 创建虚拟环境..."
  python3 -m venv .venv
fi

echo "[2/3] 安装依赖..."
.venv/bin/pip install -r requirements.txt -q

echo "[3/3] 启动界面 → http://127.0.0.1:7860"
.venv/bin/python main.py
