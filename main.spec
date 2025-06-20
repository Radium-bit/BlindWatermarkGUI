# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
import os
import json
from dotenv import load_dotenv

load_dotenv('DEV.ENV')
env_path = os.getenv('SITE_PACKAGE_PATH')
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
    'scipy._lib.array_api_compat.common._fft'
]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        (qrdet_model_path, 'qrdet/.model'),
        # 包含修复文件
        (os.path.join(hooks_dir, 'torch_fixes.py'), '.'),
        (os.path.join(hooks_dir, 'torch_numpy_fix.py'), '.'),
        (os.path.join(env_path, 'scipy/_lib/array_api_compat/numpy'), 'scipy/_lib/array_api_compat/numpy'),
        ('hidden_imports.json', '.'),
        *collect_data_files('ultralytics')
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

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BlindWatermarkGUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False, ## 测试时为true
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
    optimize=2 ## Debug需要改回0
)