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

echo 正在推送到已有远程仓库...
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    git remote add origin git@github.com:BTFzds/FaceSort_CU.git
)
git push -u origin main

if errorlevel 1 (
    echo.
    echo 推送失败。请确认已配置 SSH 密钥，或改用 HTTPS：
    echo   git remote set-url origin https://github.com/BTFzds/FaceSort_CU.git
    echo   git push -u origin main
) else (
    echo.
    echo 完成！仓库地址：https://github.com/BTFzds/FaceSort_CU
)

pause
