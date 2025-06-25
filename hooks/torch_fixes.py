## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

# hooks/torch_fixes.py
"""
修复Torch Numpy初始化问题的补丁
"""

def apply_torch_numpy_fix():
    """修复torch._numpy._ufuncs中的name未定义错误"""
    try:
        # 延迟导入，确保在正确时机执行
        import torch._numpy as tnp
        
        # 覆盖有问题的代码段
        if hasattr(tnp, '_ufuncs') and hasattr(tnp._ufuncs, '_ufunc_defs'):
            for name, ufunc in tnp._ufuncs._ufunc_defs.items():
                if not hasattr(tnp._ufuncs, name):
                    setattr(tnp._ufuncs, name, tnp._ufuncs.deco_binary_ufunc(ufunc))
                    
        print("成功应用Torch Numpy修复")
                
    except Exception as e:
        print(f"Torch Numpy修复应用失败: {e}")

def disable_problematic_features():
    """禁用可能导致问题的Torch特性"""
    import os
    os.environ['TORCH_DISABLE_NUMPY'] = '1'
    os.environ['PYTORCH_JIT'] = '0'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    
    try:
        import torch
        # 禁用JIT相关特性
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_nvfuser_enabled(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        
        # 重置动态模块状态
        if hasattr(torch, '_dynamo'):
            torch._dynamo.reset()
        
        print("已禁用Torch问题特性")
    except:
        pass

# 应用修复 - 将在导入时自动执行
disable_problematic_features()
apply_torch_numpy_fix()