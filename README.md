# Blind-Watermark GUI 工具

本项目是一个基于 [blind_watermark](https://github.com/guofei9987/blind_watermark) 的图形界面工具，支持图片水印的 **嵌入与提取** 操作，旨在方便非开发者通过拖拽和点击快速使用盲水印功能。

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

---

## ⚠️ 注意事项

1. **原始文件保存**：
   - 始终建议在打水印后，不要删除您的原始图片
2. **水印信息记录（对于v0.1.3及以下版本）**：
   - 保持打水印后图像的完整文件名，如`xxx-Watermark-ws{长度}-size{宽}x{高}.png`
   - 请记录使用的水印文本长度 及 原始图片尺寸（宽高）
   - 新版本无需保存这些信息，文件名不再是重点

**！这些信息对后续水印提取非常重要 ！**

【WARNING！】

**自v2.0版本开始，由于抗干扰模式更新，新版水印将不再向下兼容**

**向下兼容的提取正在制作中，如果您的水印使用旧版本打上，请使用v0.1.3版本**

---

## 📦 安装依赖/编译准备

 **Python版本**: ≥3.7, <3.12

0. 直接运行初始化脚本`Init.bat`或`Init.sh`

1. 检查并编辑`DEV.ENV`和`BUILD.ENV`以符合您的开发和打包环境

---

1. 安装依赖

```bash
pip install blind-watermark pillow tkinterdnd2-universal qrcode pyzbar qreader numpy python-dotenv noise
```

2. 复制程序目录下的`DEV.ENV_SAMPLE`文件并重命名为`DEV.ENV`

3. 编辑`DEV.ENV`的内容以符合您当前的Python开发环境

```python
# Site-Packages目录示例
SITE_PACKAGE_PATH='C:\Python310\Lib\site-packages\qrdet\.model'
```

4. 复制程序目录下的`BUILD.ENV_SAMPLE`文件并重命名为`BUILD.ENV`

5. 编辑`BUILD.ENV`的内容以符合您当前的编译结果，如修改版本号

```python
## 【注意】 此文件会覆盖DEV.ENV的同名内容，敬请留意

#构建时输出文件名是否包括 Git Hash 
INCLUDE_GIT_HASH=true
# 程序构建版本号
BUILD_VERSION='0.2.2'
```

6. 复制程序目录下的`APP.ENV_SAMPLE`文件并重命名为`APP.ENV`

至此打包运行前准备工作已完成

---

## 🛠️ 打包 Windows 可执行文件（.exe）

推荐使用 [PyInstaller](https://www.pyinstaller.org/) 创建单文件 `.exe`

### 1. 确保安装依赖后在目录执行以下命令

```bash
pyinstaller --clean main.spec
```

---

## 🚀 使用方法

### 1. 运行程序

如果你没有exe或打算手动运行源代码，执行下列命令

```bash
python main.py
```

### 2. 嵌入水印（Embed）

1. 启动后选择 “嵌入水印” 模式

2. 输入水印内容，可换行

3. 拖拽图片进入中间区域

4. 自动输出：`原名-Watermark-ws{长度}-size{宽}x{高}.png`，例如：
   
   ```
   example-Watermark-ws256-size1920x1080.png
   ```

### 3. 提取水印（Extract）

1. 选择 “提取水印” 模式

2. 拖拽已加水印的图片

3. 自动从文件名中识别 `ws长度` 与 `sizeWxH`：
   
   * 若未识别，可手动输入水印长度与尺寸

4. 输出结果会弹窗展示水印内容

---

## 📁 输出说明

嵌入水印后会在图片同目录生成一张新图，文件名中包括：

* `wsXXX`：嵌入的水印位数
* `sizeWxH`：原始图片尺寸
* 后缀和原始格式相同

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

- [x] 对旧版本水印的解码支持

- [ ] 增加对自定义图像文件的嵌入/提取

- [ ] 对任意文件嵌入的支持

- [ ] 基于PGP的签名校验和机制

- [ ] 优化性能

- [ ] 减少打包体积

- [ ] 美化图形界面

- [ ] 进一步简化用户流程，实现操作向导
