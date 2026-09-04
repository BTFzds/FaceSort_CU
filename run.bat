@echo off
chcp 65001 >nul
echo ========================================
echo   FaceSort 本地人脸照片整理工具
echo   100%% 本地运行，数据不会上传
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

echo [2/3] 安装依赖（含 GPU 加速包，首次运行较慢）...
.venv\Scripts\pip install -r requirements.txt -q
.venv\Scripts\pip uninstall onnxruntime -y >nul 2>&1
.venv\Scripts\pip install onnxruntime-gpu>=1.18.0 -q
if errorlevel 1 (
    echo GPU 包安装失败，尝试 CPU 版本...
    .venv\Scripts\pip uninstall onnxruntime-gpu onnxruntime -y >nul 2>&1
    .venv\Scripts\pip install -r requirements-cpu.txt -q
)

echo [3/3] 启动界面，浏览器将自动打开...
echo 若未自动打开，请访问 http://127.0.0.1:7860
echo 按 Ctrl+C 可停止
echo.
.venv\Scripts\python main.py
pause
