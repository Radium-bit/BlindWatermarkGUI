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

# 注意：Windows 下不要强行使用 RTLD_GLOBAL！
# 它会污染进程 DLL 空间，导致 numpy 出现 
# "CPU dispatcher tracer already initlized" 的 RuntimeError