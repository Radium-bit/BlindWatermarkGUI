# Blind-Watermark GUI

---

## 还在为图片版权烦恼？这款盲水印工具帮你轻松维权！

你是否曾因原创图片被盗用而束手无策，甚至反被恶意索赔？为了解决图片版权的困扰，我们为你带来了这款基于 [blind_watermark](https://github.com/guofei9987/blind_watermark) 的**图形界面工具**。

现已整合来自实验版V3-rc1的先进水印算法以及BlindWatermarkCore的图像空间探测方法。这项技术优化了二进制文件的嵌入流程，显著增强了水印的纠错能力，比0.2.7版本单纯依赖阈值计算具备更强的**文件误码修复能力**。更多算法详情可见[TECH.md](https://github.com/Radium-bit/BlindWatermarkGUI/blob/main/TECH.md)。

### 什么是盲水印？

盲水印技术能将你的专属水印悄无声息地嵌入图片之中。处理后的图片在视觉上与原图几乎无异，但凭借你的密钥，我们就能轻松提取出隐藏的水印，为你提供有力的**版权证据辅助**！

### 本工具的优势

这款工具专为**非开发者**设计，操作非常简单。你只需通过**拖拽和点击**，即可快速实现图片水印的**嵌入与提取**。告别复杂的代码，轻松保护你的图片版权！

## ✨ 功能特性

- ✅ 支持嵌入和提取盲水印（Blind Watermark）
- ✅ 图形界面操作（TkinterDnD2），支持拖拽图片
- ✅ 自动记录与识别水印长度与原图尺寸
- ✅ 自动根据原图尺寸缩放，使用高质量 `LANCZOS` 算法
- ✅ 自动识别用户临时目录
- ✅ 自动清理临时文件
- ✅ 预留 `Copyright@`和 `Author@`标识，方便填写，也可以自行修改内容
- ✅ 更强的抗干扰编码方式，使用QR Code和YOLO进行水印编码识别
- ✅ 自定义导出格式 JPG/PNG
- ✅ 自定义图像文件的嵌入/提取
- ✅ 支持任意文件的嵌入/提取（在允许嵌入长度内）

---

## ⚠️ 注意事项

1. **由于程序使用了较多运行库，双击后没反应请稍后，一般在30秒内会响应，请勿同时运行多个本程序。** 目前仍在持续优化中...

2. **如启动时失败，程序不断重启，请安装 [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) 点击即可下载**

3. **原始文件保存**：
   
   - 始终建议在打水印后，不要删除您的原始图片

4. **水印信息记录（对于v0.1.3及以下版本）**：
   
   - 保持打水印后图像的完整文件名，如`xxx-Watermark-ws{长度}-size{宽}x{高}.png`
   - 请记录使用的水印文本长度 及 原始图片尺寸（宽高）
   - 新版本**除文件嵌入外**无需保存水印文本长度，文件名不再是重点

5. 图像水印嵌入：
   
   - 嵌入的图像需约为128x128的内容，建议使用对比明显的黑白水印图
   
   - 若不符合程序会强制转换分辨率并转为灰度图像！（受限于基础库）

【WARNING！】

**自v2.0版本开始，由于抗干扰模式更新，新版水印将不再向下兼容**

**v0.2.4已经具有向下兼容功能，可兼容解析v0.1.3的旧版水印，但不推荐再使用！**

---

## 📥 如何下载

目前程序提供三个版本供选择：

### 1. `_Portable` 版本（推荐开发者/懂电脑的用户）

- 后缀为 `_Portable.7z`，解压即可运行  
- 无需安装，便于携带和自定义配置

### 2. `onefile` 版本（推荐轻度用户）

- 无后缀，双击即可运行  
- 最简单直接，但首次启动较慢（因需解压到内存）

### 3. `_Installer` 安装版（推荐大多数用户）

- 后缀为 `_Installer.exe`推荐体验方式，启动快捷，兼容性最好  
- 安装流程：
  1. 双击安装程序启动
  2. 点击 `Next`
  3. 选择安装路径
  4. 点击 `Install` 开始安装

---

## 📦 安装依赖/编译准备

**Python版本**: ≥3.7, ≤3.10

0. 直接运行初始化脚本 `Init.bat` 或 `Init.sh`

1. 检查并编辑 `DEV.ENV` 和 `BUILD.ENV` 以符合您的开发和打包环境

### 依赖安装

```bash
pip install blind-watermark pillow tkinterdnd2-universal qrcode pyzbar qreader numpy python-dotenv noise py7zr
```

### 安装 NSIS（用于生成安装版）

1. 访问 NSIS 官网：[https://nsis.sourceforge.io/Download](https://nsis.sourceforge.io/Download)

2. 下载并安装最新版本（建议安装到默认路径）

3. 安装后请将 NSIS 安装路径加入系统 `PATH` 环境变量，例如：
   
   ```
   C:\Program Files (x86)\NSIS\
   ```

---

## 🛠️ 打包 Windows 可执行文件（.exe）

使用 [PyInstaller](https://www.pyinstaller.org/) 创建单文件 `.exe`

### 1. 检查 `DEV.ENV` 和 `BUILD.ENV` 确保打包参数正确

### 2. 确保安装依赖后在程序根目录执行以下命令：

```bash
pyinstaller main.spec --clean --noconfirm
```

---

## 🚀 使用方法

### 0. 程序说明

1. **增强水印模式：** 通过给输入图像添加人眼低可识别噪声来增加水印的附着面积，增强抗干扰能力。
2. **启用兼容模式：** 调用 *v0.1.3* 的旧版水印处理算法，抗干扰差，已被弃用，仅作备份。
3. **提取显示原图：** 如果程序经过算法分析后仍然无法分析水印内容，那么显示提取后的原始数据，而不是程序处理后的图像。

### 1. 运行程序

双击运行 `BlindWatermarkGUI.exe`，稍等片刻使其完成启动

或下载源代码后，通过以下命令运行：

```bash
python main.py
```

### 2. 嵌入水印（Embed）

1. 启动后选择 “嵌入水印” 模式

2. 输入水印内容，可换行

3. 拖拽图片进入中间区域

4. 自动输出到原图目录：例如：
   
   ```
   example-Watermark-ws256-size1920x1080.png
   ```

### 3. 提取水印（Extract）

1. 选择 “提取水印” 模式

2. 拖拽已加水印的图片

3. 自动识别 `ws长度` 与 `sizeWxH`
   
   * 若未识别，可手动输入原始尺寸，新版只需输入原始宽高

4. 输出结果会弹窗展示水印内容

---

## 📁 输出说明

嵌入水印后会在图片同目录生成一张新图，文件名中包括：

* `wsXXX`：嵌入的水印位数
* `sizeWxH`：原始图片尺寸
* 输出格式视用户指定

---

## 🌱 贡献分支说明

- **main**: 此分支用于**最新的技术开发和测试**，在稳定前均使用 **Pre-release** 形式发布。

- **2.x-dev**: 此分支用于对现有稳定版本 **2.x** 添加**增强功能**。

- **2.x-fixes**: 此分支用于**小范围错误修复**和**发行安装包的可用性**。


---

## 📄 许可证 License

本项目采用 [Apache License 2.0](LICENSE.txt)。

```text
Copyright 2025 Radium-bit

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## 🙏 致谢

本项目基于开源项目 [blind\_watermark](https://github.com/guofei9987/blind_watermark) 构建，感谢作者 @guofei9987 提供优秀的盲水印核心功能。

---

## 💡 后续计划（TODO）

* [x] 对旧版本水印的解码支持
* [x] 增加启动画面
* [x] 增加对自定义图像文件的嵌入/提取
* [x] 对任意文件嵌入的支持
* [ ] 基于PGP的签名校验和机制
* [x] 优化性能
* [ ] 减少打包体积
* [ ] 美化图形界面
* [ ] 进一步简化用户流程，实现操作向导
