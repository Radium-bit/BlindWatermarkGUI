@echo off
chcp 65001 >nul
echo ========================================
echo 项目初始化脚本 - Windows Ver
echo ========================================

:: 检查Python版本
echo 正在检查Python版本...
python --version 2>nul
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请确保Python已安装并添加到PATH环境变量中
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo 当前Python版本: %PYTHON_VERSION%

:: 简单的版本检查（检查是否为3.x）
echo %PYTHON_VERSION% | findstr /R "^3\." >nul
if %errorlevel% neq 0 (
    echo 警告: 建议使用Python 3.7-3.11版本
)

echo.
echo ========================================
echo 步骤1: 安装Python依赖包
echo ========================================
echo 正在安装依赖包...
pip install blind-watermark pillow tkinterdnd2-universal qrcode pyzbar qreader numpy python-dotenv noise py7zr

if %errorlevel% neq 0 (
    echo 错误: 依赖包安装失败
    pause
    exit /b 1
)
echo 依赖包安装完成！

echo.
echo ========================================
echo 步骤2: 创建DEV.ENV配置文件
echo ========================================
if exist "DEV.ENV" (
    echo DEV.ENV文件已存在，跳过创建步骤
) else (
    if exist "DEV.ENV_SAMPLE" (
        copy "DEV.ENV_SAMPLE" "DEV.ENV" >nul
        echo DEV.ENV文件已创建
        echo 请手动编辑DEV.ENV文件，设置正确的SITE_PACKAGE_PATH路径
    ) else (
        echo 警告: 未找到DEV.ENV_SAMPLE文件，跳过此步骤
    )
)

echo.
echo ========================================
echo 步骤3: 创建BUILD.ENV配置文件
echo ========================================
if exist "BUILD.ENV" (
    echo BUILD.ENV文件已存在，跳过创建步骤
) else (
    if exist "BUILD.ENV_SAMPLE" (
        copy "BUILD.ENV_SAMPLE" "BUILD.ENV" >nul
        echo BUILD.ENV文件已创建
    ) else (
        echo 警告: 未找到BUILD.ENV_SAMPLE文件，跳过此步骤
    )
)

echo.
echo ========================================
echo 步骤4: 创建APP.ENV配置文件
echo ========================================
if exist "APP.ENV" (
    echo APP.ENV文件已存在，跳过创建步骤
) else (
    if exist "APP.ENV_SAMPLE" (
        copy "APP.ENV_SAMPLE" "APP.ENV" >nul
        echo APP.ENV文件已创建
    ) else (
        echo 警告: 未找到APP.ENV_SAMPLE文件，跳过此步骤
    )
)

echo.
echo ========================================
echo 初始化完成！
echo ========================================
echo 提醒事项:
echo 1. 请检查并编辑DEV.ENV文件中的SITE_PACKAGE_PATH路径
echo 2. 如需要，可以修改BUILD.ENV中的配置
echo 3. 检查APP.ENV配置是否符合需求
echo.
echo 打包运行前准备工作已完成！
echo ========================================
pause