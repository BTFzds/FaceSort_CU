@echo off
chcp 65001 >nul
echo ========================================
echo   推送 FaceSort 到 GitHub
echo ========================================
echo.

where gh >nul 2>&1
if errorlevel 1 (
    echo 未找到 GitHub CLI，请先安装：
    echo   winget install GitHub.cli
    pause
    exit /b 1
)

gh auth status >nul 2>&1
if errorlevel 1 (
    echo 请先登录 GitHub（会打开浏览器）：
    gh auth login --hostname github.com --git-protocol https --web
    if errorlevel 1 (
        echo 登录失败，请检查网络后重试
        pause
        exit /b 1
    )
)

echo 正在创建 GitHub 仓库并推送...
gh repo create FaceSort_CU --public --source=. --remote=origin --push --description "本地人脸照片检索与分类工具 - 100%% offline face photo search"

if errorlevel 1 (
    echo.
    echo 若仓库名已存在，可手动指定：
    echo   gh repo create FaceSort_CU-你的用户名 --public --source=. --remote=origin --push
) else (
    echo.
    echo 完成！仓库地址：
    gh repo view --web
)

pause
