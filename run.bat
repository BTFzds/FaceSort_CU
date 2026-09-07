@echo off
chcp 65001 >nul
echo ========================================
echo   FaceSort 本地人脸整理（桌面版）
echo   100%% 本地运行，不经过浏览器
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] 创建虚拟环境...
    python -m venv .venv
    if errorlevel 1 (
        echo 错误：请先安装 Python 3.10 或更高版本
        pause
        exit /b 1
    )
)

echo [2/3] 安装依赖...
.venv\Scripts\pip install -r requirements.txt -q
.venv\Scripts\pip uninstall onnxruntime -y >nul 2>&1
.venv\Scripts\pip install "onnxruntime-gpu==1.20.2" nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12 -q
if errorlevel 1 (
    echo GPU 包安装失败，尝试 CPU 版本...
    .venv\Scripts\pip uninstall onnxruntime-gpu onnxruntime -y >nul 2>&1
    .venv\Scripts\pip install -r requirements-cpu.txt -q
)

echo [3/3] 启动桌面窗口...
.venv\Scripts\python main.py
pause
