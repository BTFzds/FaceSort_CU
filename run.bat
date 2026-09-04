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

echo [2/3] 安装依赖（首次运行较慢，需下载 AI 模型）...
.venv\Scripts\pip install -r requirements.txt -q

echo [3/3] 启动界面，浏览器将自动打开...
echo 若未自动打开，请访问 http://127.0.0.1:7860
echo 按 Ctrl+C 可停止
echo.
.venv\Scripts\python main.py
pause
