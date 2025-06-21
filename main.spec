# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
import os
import json
import lzma
import subprocess
from dotenv import load_dotenv

load_dotenv('DEV.ENV')
load_dotenv('BUILD.ENV', override=True)

env_path = os.getenv('SITE_PACKAGE_PATH')
VERSION = os.getenv('VERSION')

# 控制选项：是否在版本号后添加 Git hash
INCLUDE_GIT_HASH = os.getenv('INCLUDE_GIT_HASH', 'false').lower() == 'true'

def get_git_hash():
    """获取当前 Git commit 的短hash"""
    try:
        # 获取短 hash (7位)
        result = subprocess.run(['git', 'rev-parse', '--short=7', 'HEAD'], 
                            capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 如果 git 命令失败或找不到，返回默认值
        print("警告: 无法获取 Git hash，使用默认值")
        return "unknown"

# 构建最终版本号
if INCLUDE_GIT_HASH:
    git_hash = get_git_hash()
    FINAL_VERSION = f"{VERSION}_build.{git_hash}"
    print(f"Build Version: {FINAL_VERSION}")
else:
    FINAL_VERSION = VERSION
    print(f"Build Version: {FINAL_VERSION}")

qrdet_model_path = os.path.join(env_path,'qrdet','.model')

# 定义 block_cipher
block_cipher = None

# 获取 hooks 目录的路径
hooks_dir = 'hooks'

# 定义去重列表，可显式导入的部分
REQUIRED_IMPORTS = [
    'qreader',
    'qrcode',
    'ultralytics',
    'torch._numpy',
    'torch._numpy._ufuncs',
    'torch._numpy._ndarray',
    'torch._numpy._dtypes',
    'torch._numpy._funcs',
    'torch._numpy._util',
    'torchvision.ops',
    'torchvision.models',
    'torchvision.transforms',
    'torchvision.io',
    'torch._dynamo',
    'torch.fx',
    'scipy._lib.array_api_compat.common._fft',
    'scipy._lib.array_api_compat.common',
    'scipy._lib.array_api_compat.numpy.fft',
    'quadrilateral_fitter',
    'quadrilateral_fitter.quadrilateral_fitter',
    # 添加 watermark 模块
    'watermark',
    'watermark.embed',
    'watermark.extract',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        # qr模型
        (qrdet_model_path, 'qrdet/.model'),
        # 构建环境
        ('BUILD.ENV', '.'),
        # 包含修复文件
        (os.path.join(hooks_dir, 'torch_fixes.py'), '.'),
        (os.path.join(hooks_dir, 'torch_numpy_fix.py'), '.'),
        (os.path.join(env_path, 'scipy/_lib/array_api_compat/numpy'), 'scipy/_lib/array_api_compat/numpy'),
        ('hidden_imports.json', '.'),
        *collect_data_files('ultralytics'),
        ## 拆分后的模块
        # watermark 模块
        ('watermark', 'watermark'),
    ],
    hiddenimports = REQUIRED_IMPORTS + [
    imp.strip('"') for imp in json.load(open('hidden_imports.json'))
    if imp not in REQUIRED_IMPORTS],
    hookspath=[hooks_dir],
    hooksconfig={},
    runtime_hooks=[os.path.join(hooks_dir, 'torch_numpy_fix.py')],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

# pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher, compression=lzma, compression_level=6)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=f'BlindWatermarkGUI_v{FINAL_VERSION}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, ## 测试时为True
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
    optimize=2 ## Debug需要改回0，生产2
)