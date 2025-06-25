## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

import os
import sys

# 在 Torch 导入前设置环境变量
os.environ['TORCH_DISABLE_NUMPY'] = '1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# Windows 特定的 DLL 加载设置
if sys.platform.startswith('win'):
    try:
        import ctypes
        # 添加 RTLD_GLOBAL 标志
        sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    except:
        pass