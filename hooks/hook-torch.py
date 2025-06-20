# hooks/hook-torch.py
from PyInstaller.utils.hooks import collect_data_files

# 基本 Torch 数据收集
datas = collect_data_files('torch', include_py_files=True)

# 隐藏导入
hiddenimports = [
    'torch._numpy',
    'torch._numpy._ufuncs',
    'torch._numpy._ndarray',
    'torch._numpy._dtypes',
    'torch._numpy._funcs',
    'torch._numpy._util',
    'torch._dynamo',
    'torch.fx'
]

# 二进制文件由 PyInstaller 自动处理
binaries = []