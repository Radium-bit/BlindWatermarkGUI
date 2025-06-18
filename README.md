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

---

## ⚠️ 注意事项

1. **原始文件保存**：
   - 始终建议在打水印后，不要删除您的原始图片
2. **水印信息记录**：
   - 保持打水印后图像的完整文件名，如`xxx-Watermark-ws{长度}-size{宽}x{高}.png`
   - 请记录使用的水印文本长度 及 原始图片尺寸（宽高）

**！这些信息对后续水印提取非常重要 ！**

---

## 📦 安装依赖

需使用 Python 3.7 及以上版本。

```bash
pip install blind-watermark pillow tkinterdnd2-universal
```

---

## 🛠️ 打包 Windows 可执行文件（.exe）

推荐使用 [PyInstaller](https://www.pyinstaller.org/) 创建单文件 `.exe`

### 1. 确保安装依赖后在目录执行以下命令

```bash
pyinstaller main.spec
```

*如无法成功编译，可以尝试使用原始命令（不推荐，会导致main.spec被覆盖）*

```bash
pyinstaller --additional-hooks-dir=hooks --onefile --windowed main.py
```

---

## 🚀 使用方法

### 1. 运行程序

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

咕咕咕咕咕咕......
