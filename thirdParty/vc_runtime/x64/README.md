# 随包分发的 Microsoft C++ 运行库（x64）

本目录下的 DLL 随程序一起分发，目的是让程序**不依赖目标机是否安装
VC++ Redistributable**。程序启动时由 `main.py` 的
`preload_bundled_vc_runtime()` 在任何三方库导入之前，用绝对路径优先加载
这里的副本；只有在随包副本缺失或加载失败时才回退到 `System32`。

## 文件清单（共 12 个，约 2.54 MB）

`vcruntime140.dll`、`vcruntime140_1.dll`、`vcruntime140_threads.dll`、
`msvcp140.dll`、`msvcp140_1.dll`、`msvcp140_2.dll`、
`msvcp140_atomic_wait.dll`、`msvcp140_codecvt_ids.dll`、
`concrt140.dll`、`vccorlib140.dll`、`vcamp140.dll`、`vcomp140.dll`

## 来源与校验

- 来源：官方安装包 **Microsoft Visual C++ 2015-2022 Redistributable (x64)**
  的安装载荷。在本机可通过 `C:\ProgramData\Package Cache\` 下的
  `vcRuntimeMinimum_amd64\cab1.cab` 用系统 `expand.exe` 解出（文件在 cab 内
  名为 `xxx.dll_amd64`，解出后重命名为 `xxx.dll`）。
- 版本：`14.51.36247.0`（随官方 Redist 更新，重新采集即可）。
- 校验：每个文件的 Authenticode 签名状态为 `Valid`，签名主体为
  `Microsoft Corporation` / `Microsoft Windows Software Compatibility Publisher`。
  重新采集后请再次校验，签名无效的文件不得进入本目录。

## 为什么不再附带 ucrtbase.dll

`ucrtbase.dll` 属于 UCRT（Universal C Runtime）。从 Windows 10 起 UCRT 是
**操作系统组件**，并以 api-set 形式激活，应用程序本地副本无法覆盖它，
硬塞只会增加风险。因此本目录只放 C++ 部分（vcruntime / msvcp / concrt 等），
UCRT 交给系统提供。

## 相关开关（排查用）

- `BW_DISABLE_BUNDLED_CRT=1`：强制使用系统副本，忽略本目录（用于对比验证）。
- `BW_FORCE_BUNDLED_CRT=1`：禁止回退系统副本，用于验证"只靠随包运行库"能否启动。

## 许可

这些文件是 Microsoft 的可再分发运行库（随 VC++ Redistributable 一并分发），
仅作为本程序运行所需的运行库随附，不修改、不单独分发。详见
`THIRD_PARTY_NOTICES.md`。
