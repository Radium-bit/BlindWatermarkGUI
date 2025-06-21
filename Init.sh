#!/bin/bash

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}项目初始化脚本 - Linux Ver ${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查Python版本
echo -e "${YELLOW}正在检查Python版本...${NC}"
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python，请确保Python已安装${NC}"
    exit 1
fi

# 优先使用python3
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
else
    PYTHON_CMD="python"
    PIP_CMD="pip"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "${GREEN}当前Python版本: $PYTHON_VERSION${NC}"

# 检查Python版本是否符合要求
if [[ $PYTHON_VERSION =~ ^3\.([7-9]|1[01])\. ]]; then
    echo -e "${GREEN}Python版本符合要求 (3.7-3.11)${NC}"
elif [[ $PYTHON_VERSION =~ ^3\. ]]; then
    echo -e "${YELLOW}警告: 建议使用Python 3.7-3.11版本${NC}"
else
    echo -e "${RED}警告: 建议使用Python 3.7-3.11版本${NC}"
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}步骤1: 安装Python依赖包${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}正在安装依赖包...${NC}"

$PIP_CMD install blind-watermark pillow tkinterdnd2-universal qrcode pyzbar qreader numpy python-dotenv noise

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 依赖包安装失败${NC}"
    exit 1
fi
echo -e "${GREEN}依赖包安装完成！${NC}"

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}步骤2: 创建DEV.ENV配置文件${NC}"
echo -e "${BLUE}========================================${NC}"
if [ -f "DEV.ENV" ]; then
    echo -e "${YELLOW}DEV.ENV文件已存在，跳过创建步骤${NC}"
else
    if [ -f "DEV.ENV_SAMPLE" ]; then
        cp "DEV.ENV_SAMPLE" "DEV.ENV"
        echo -e "${GREEN}DEV.ENV文件已创建${NC}"
        echo -e "${YELLOW}请手动编辑DEV.ENV文件，设置正确的SITE_PACKAGE_PATH路径${NC}"
    else
        echo -e "${YELLOW}警告: 未找到DEV.ENV_SAMPLE文件，跳过此步骤${NC}"
    fi
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}步骤3: 创建BUILD.ENV配置文件${NC}"
echo -e "${BLUE}========================================${NC}"
if [ -f "BUILD.ENV" ]; then
    echo -e "${YELLOW}BUILD.ENV文件已存在，跳过创建步骤${NC}"
else
    if [ -f "BUILD.ENV_SAMPLE" ]; then
        cp "BUILD.ENV_SAMPLE" "BUILD.ENV"
        echo -e "${GREEN}BUILD.ENV文件已创建${NC}"
    else
        echo -e "${YELLOW}警告: 未找到BUILD.ENV_SAMPLE文件，跳过此步骤${NC}"
    fi
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}步骤4: 创建APP.ENV配置文件${NC}"
echo -e "${BLUE}========================================${NC}"
if [ -f "APP.ENV" ]; then
    echo -e "${YELLOW}APP.ENV文件已存在，跳过创建步骤${NC}"
else
    if [ -f "APP.ENV_SAMPLE" ]; then
        cp "APP.ENV_SAMPLE" "APP.ENV"
        echo -e "${GREEN}APP.ENV文件已创建${NC}"
    else
        echo -e "${YELLOW}警告: 未找到APP.ENV_SAMPLE文件，跳过此步骤${NC}"
    fi
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}初始化完成！${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}提醒事项:${NC}"
echo -e "${YELLOW}1. 请检查并编辑DEV.ENV文件中的SITE_PACKAGE_PATH路径${NC}"
echo -e "${YELLOW}2. 如需要，可以修改BUILD.ENV中的配置${NC}"
echo -e "${YELLOW}3. 检查APP.ENV配置是否符合需求${NC}"
echo
echo -e "${GREEN}打包运行前准备工作已完成！${NC}"
echo -e "${BLUE}========================================${NC}"

# 设置脚本执行权限提醒
if [ ! -x "$0" ]; then
    echo -e "${YELLOW}提示: 如果脚本无法执行，请运行: chmod +x init.sh${NC}"
fi