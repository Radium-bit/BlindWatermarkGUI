# 第三方许可声明（Third Party Notices）

本项目主许可证为 Apache-2.0（见 `LICENSE` / `LICENSE.txt`），以下为重构阶段新增或重点关注的第三方组件许可说明：

## PySide6 (Qt for Python)

- 组件：`PySide6`
- 来源：Qt Company / Qt for Python 项目
- 许可证：LGPLv3（以及 Qt 对应组件的其他许可条款）

分发可执行程序时，请按 PySide6/Qt 官方要求一并提供其许可证文本与声明。

## Microsoft Visual C++ 运行库（随包分发）

- 组件：`thirdParty/vc_runtime/x64/` 下的 12 个 DLL（`vcruntime140.dll`、
  `vcruntime140_1.dll`、`vcruntime140_threads.dll`、`msvcp140.dll`、
  `msvcp140_1.dll`、`msvcp140_2.dll`、`msvcp140_atomic_wait.dll`、
  `msvcp140_codecvt_ids.dll`、`concrt140.dll`、`vccorlib140.dll`、
  `vcamp140.dll`、`vcomp140.dll`）
- 来源：Microsoft Visual C++ 2015-2022 Redistributable (x64) 安装载荷
  （版本 14.51.36247.0，采集与校验方式见 `thirdParty/vc_runtime/x64/README.md`）
- 许可证：Microsoft 可再分发运行库条款。这些文件随本程序一并分发，仅用于
  满足程序运行所需的 C++ 运行库依赖，未做任何修改，也不单独分发。
- 说明：`ucrtbase.dll` 属于 Windows 操作系统组件（UCRT），不随本程序分发。

## 说明

1. 本文件用于记录项目分发时应附带的第三方许可信息，不替代各上游项目的原始法律文本。
2. 若后续新增依赖，请同步补充本文件，避免遗漏许可义务。

