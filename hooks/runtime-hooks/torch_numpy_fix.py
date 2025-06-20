# hooks/runtime-hooks/torch_numpy_fix.py
"""
在Torch导入前应用的运行时修复
"""
import os
import sys

# 在Torch导入前设置环境变量
os.environ['TORCH_DISABLE_NUMPY'] = '1'
os.environ['PYTORCH_JIT'] = '0'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

# 防止Torch尝试加载Numpy集成
os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'

# 设置Deepbind标志以解决DLL加载问题
if sys.platform.startswith('win'):
    try:
        import ctypes
        sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
    except:
        pass