# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
import os
import json
import lzma
import subprocess
from dotenv import load_dotenv
import re

# ========================================
# 环境配置加载
# ========================================

# 加载环境配置文件（不加载BUILD.ENV，避免解析错误）
load_dotenv('DEV.ENV')
load_dotenv('APP.ENV')

# 从环境变量获取基础配置
env_path = os.getenv('SITE_PACKAGE_PATH')

# 编译时显示更多细节
MORE_DETAILS = False

# ========================================
# BUILD.ENV 配置解析（不使用python-dotenv）
# ========================================

def parse_build_config():
    """直接解析BUILD.ENV文件，避免python-dotenv的解析问题"""
    # 默认配置
    config = {
        'include_git_hash': True,
        'build_version': '1.0.0',
        'enable_console_debug': False,
        'enable_optimize': True,
        'enable_compress': True,
        'compress_format': 'lzma',
        'compress_level': 9,
        'one_file_mode': True,
        'required_imports': [],
        'datas': [],
        'hooks': ['hooks'],
        'runtime_hooks': [],
        'exclude_imports': []
    }
    
    if not os.path.exists('BUILD.ENV'):
        print("BUILD.ENV文件不存在，使用默认配置")
        return config
    
    try:
        with open('BUILD.ENV', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 解析简单的键值对配置
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            
            # 跳过注释、空行和复杂配置行
            if (not line or line.startswith('#') or 
                line.startswith(('REQUIRED_IMPORTS', 'DATAS', 'HOOKS', 'RUNTIME_HOOKS', 'EXCLUDE_IMPORTS'))):
                continue
            
            # 只处理简单的键值对
            if '=' in line and not any(char in line for char in ['[', ']', '(', ')']):
                try:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 处理行内注释
                    if '#' in value:
                        value = value.split('#')[0].strip()
                    
                    # 移除引号
                    value = value.strip("'\"")
                    
                    if key == 'INCLUDE_GIT_HASH':
                        config['include_git_hash'] = value.lower() == 'true'
                    elif key == 'BUILD_VERSION':
                        config['build_version'] = value
                    elif key == 'ENABLE_CONSOLE_DEBUG':
                        config['enable_console_debug'] = value.lower() == 'true'
                    elif key == 'ENABLE_OPTIMIZE':
                        config['enable_optimize'] = value.lower() == 'true'
                    elif key == 'ENABLE_COMPRESS':
                        config['enable_compress'] = value.lower() == 'true'
                    elif key == 'COMPRESS_FORMAT':
                        config['compress_format'] = value
                    elif key == 'COMPRESS_LEVEL':
                        config['compress_level'] = int(value)
                    elif key == 'ONE_FILE_MODE':
                        config['one_file_mode'] = value.lower() == 'true'
                except Exception as e:
                    print(f"解析配置行时出错: {line} - {e}")
                    continue
        
        # 解析复杂配置
        config['required_imports'] = parse_python_list_from_content(content, 'REQUIRED_IMPORTS')
        config['datas'] = parse_python_list_from_content(content, 'DATAS')  # 注意这里使用大写 DATAS
        config['hooks'] = parse_python_list_from_content(content, 'HOOKS')
        config['runtime_hooks'] = parse_python_list_from_content(content, 'RUNTIME_HOOKS')
        config['exclude_imports'] = parse_python_list_from_content(content, 'EXCLUDE_IMPORTS')
        
        # 调试信息
        print(f"🔍 解析结果:")
        print(f"   - datas配置: {config['datas']}")
        if 'DATAS=[' in content or 'DATAS =' in content:  # 检查大写 DATAS
            print(f"   - 找到DATAS配置段")
        else:
            print(f"   - 未找到DATAS配置段")
        
        print("✅ 成功解析BUILD.ENV配置")
        
    except Exception as e:
        print(f"⚠️ 解析BUILD.ENV时出错，使用默认配置: {e}")
    
    return config

def parse_python_list_from_content(content, var_name):
    """从内容中解析Python列表"""
    try:
        # 尝试两种模式：带空格和不带空格
        start_patterns = [
            f'{var_name} = [',
            f'{var_name}=['
        ]
        
        start_pos = -1
        for pattern in start_patterns:
            pos = content.find(pattern)
            if pos != -1:
                start_pos = pos
                start_pattern = pattern
                break
        
        if start_pos == -1:
            return []
        
        # 找到列表的开始位置
        list_start = start_pos + len(start_pattern) - 1  # -1 是为了包含 '['
        
        # 找到匹配的右括号
        bracket_count = 0
        pos = list_start
        list_end = -1
        
        while pos < len(content):
            char = content[pos]
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    list_end = pos + 1
                    break
            pos += 1
        
        if list_end == -1:
            print(f"警告: 未找到{var_name}的结束括号")
            return []
        
        # 提取列表内容
        list_content = content[list_start:list_end]
        
        # 手动解析列表项
        return parse_list_items(list_content)
        
    except Exception as e:
        print(f"解析{var_name}时出错: {e}")
        return []

def parse_list_items(list_content):
    """解析列表项内容"""
    items = []
    
    # 移除外层括号
    content = list_content.strip()
    if content.startswith('['):
        content = content[1:]
    if content.endswith(']'):
        content = content[:-1]
    
    # 分割项目（按行分割）
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # 跳过注释和空行
        if not line or line.startswith('#'):
            continue
        
        # 移除末尾的逗号
        if line.endswith(','):
            line = line[:-1].strip()
        
        # 处理行内注释 - 找到注释位置但要小心引号内的#
        comment_pos = -1
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(line):
            if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            elif char == '#' and not in_quotes:
                comment_pos = i
                break
        
        if comment_pos != -1:
            line = line[:comment_pos].strip()
            if line.endswith(','):
                line = line[:-1].strip()
        
        # 处理COLLECT()函数调用
        if 'COLLECT(' in line:
            items.append(line)
            continue
        
        # 处理包含=>的数据映射
        if '=>' in line:
            # 先移除外层引号（如果有的话）
            line = line.strip("'\"")
            items.append(line)
            continue
        
        # 处理普通字符串项
        if line.startswith("'") and line.endswith("'"):
            items.append(line[1:-1])
        elif line.startswith('"') and line.endswith('"'):
            items.append(line[1:-1])
        elif line:
            # 移除引号
            items.append(line.strip("'\""))
    
    return items

def resolve_env_path_expression(expression, env_path):
    """解析包含DEV_ENV.SITE_PACKAGE_PATH的表达式"""
    if not env_path:
        return expression
    
    # 处理字符串拼接语法: DEV_ENV.SITE_PACKAGE_PATH+'path'
    if 'DEV_ENV.SITE_PACKAGE_PATH+' in expression:
        # 提取拼接的部分
        match = re.search(r"DEV_ENV\.SITE_PACKAGE_PATH\+(['\"])([^'\"]+)\1", expression)
        if match:
            additional_path = match.group(2)
            # 规范化路径
            full_path = os.path.join(env_path, additional_path)
            return os.path.normpath(full_path)
    
    # 处理直接替换
    if 'DEV_ENV.SITE_PACKAGE_PATH' in expression:
        result = expression.replace('DEV_ENV.SITE_PACKAGE_PATH', env_path)
        return os.path.normpath(result)
    
    return expression

# ========================================
# 版本管理
# ========================================

def get_git_hash():
    """获取当前 Git commit 的短 hash"""
    try:
        result = subprocess.run(['git', 'rev-parse', '--short=7', 'HEAD'], 
                            capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: 无法获取 Git hash，使用默认值")
        return "unknown"

def update_ver_env(version):
    """更新 APP.ENV 文件中的版本号"""
    try:
        ver_env_content = ""
        if os.path.exists('APP.ENV'):
            with open('APP.ENV', 'r', encoding='utf-8') as f:
                ver_env_content = f.read()
        
        lines = ver_env_content.split('\n')
        version_updated = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('VERSION='):
                lines[i] = f"VERSION='{version}'"
                version_updated = True
                break
        
        if not version_updated:
            if lines and lines[-1].strip():
                lines.append('')
            lines.append(f"VERSION='{version}'")
        
        with open('APP.ENV', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"已更新 APP.ENV 中的版本号为: {version}")
        
    except Exception as e:
        print(f"更新 APP.ENV 失败: {e}")

# ========================================
# 数据文件处理（修复版）
# ========================================

def expand_data_paths(datas_config):
    """展开数据文件路径（修复版）"""
    expanded_datas = []
    collected_packages = []  # 记录collect_data_files收集的包
    
    for item in datas_config:
        if isinstance(item, str):
            if MORE_DETAILS:
                print(f"🔍 处理配置项: {repr(item)}")
            
            # 处理COLLECT()格式
            if item.startswith("COLLECT("):
                match = re.search(r"COLLECT\((['\"]?)([^'\"]+)\1\)", item)
                if match:
                    package_name = match.group(2)
                    print(f"📦 使用collect_data_files收集: {package_name}")
                    try:
                        collected = collect_data_files(package_name)
                        if collected:
                            expanded_datas.extend(collected)
                            collected_packages.append(package_name)
                            print(f"✅ 成功收集 {package_name} 的 {len(collected)} 个数据文件")
                        else:
                            print(f"⚠️ {package_name} 没有发现数据文件")
                    except Exception as e:
                        print(f"❌ 收集包数据失败 {package_name}: {e}")
                continue
        
        # 处理字符串格式 "src=>dst"
        if isinstance(item, str) and '=>' in item:
            src, dst = item.split('=>', 1)
            src = src.strip()
            dst = dst.strip()
            
            # 移除目标路径的引号和注释
            dst = dst.strip("'\"")
            # 处理行内注释
            if '#' in dst:
                dst = dst.split('#')[0].strip()
            if MORE_DETAILS:
                print(f"🔍 解析路径映射: {repr(src)} => {repr(dst)}")
            
            # 处理源路径中的DEV_ENV.SITE_PACKAGE_PATH引用
            original_src = src
            src = resolve_env_path_expression(src, env_path)
            src = src.strip("'\"")  # 添加这一行，移除源路径的引号
            if MORE_DETAILS:
                print(f"🔍 路径解析: {repr(original_src)} -> {repr(src)}")
            
            # 检查源文件/目录是否存在
            if os.path.exists(src):
                expanded_datas.append((src, dst))
                if MORE_DETAILS:
                    print(f"✅ 添加数据文件: {src} => {dst}")
            else:
                print(f"⚠️ 数据文件不存在: {src}")
        
        # 处理元组格式
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            src, dst = item
            if os.path.exists(src):
                expanded_datas.append((src, dst))
                if MORE_DETAILS:
                    print(f"✅ 添加数据文件: {src} => {dst}")
            else:
                print(f"⚠️ 数据文件不存在: {src}")
    
    return expanded_datas, collected_packages

# ========================================
# 主要配置
# ========================================

# 解析构建配置（不使用python-dotenv加载BUILD.ENV）
print("📋 开始解析BUILD.ENV配置...")
build_config = parse_build_config()

# 打印解析的配置信息
print(f"📝 解析的配置:")
print(f"   - 包含Git Hash: {build_config['include_git_hash']}")
print(f"   - 构建版本: {build_config['build_version']}")
print(f"   - 控制台调试: {build_config['enable_console_debug']}")
print(f"   - 启用优化: {build_config['enable_optimize']}")
print(f"   - 启用压缩: {build_config['enable_compress']}")
print(f"   - 压缩格式: {build_config['compress_format']}")
print(f"   - 压缩级别: {build_config['compress_level']}")
print(f"   - 单文件模式: {build_config['one_file_mode']}")
print(f"   - 导入模块数: {len(build_config.get('required_imports', []))}")
print(f"   - 排除模块数: {len(build_config.get('exclude_imports', []))}")
print(f"   - 数据文件数: {len(build_config.get('datas', []))}")

# 构建最终版本号
BUILD_VERSION = build_config['build_version']

if build_config['include_git_hash']:
    git_hash = get_git_hash()
    FINAL_VERSION = f"{BUILD_VERSION}.build.{git_hash}"
    FILENAME_VERSION = f"{BUILD_VERSION}_build.{git_hash}"
    print(f"构建版本: {FINAL_VERSION}")
    update_ver_env(FINAL_VERSION)
else:
    FINAL_VERSION = BUILD_VERSION
    FILENAME_VERSION = BUILD_VERSION
    print(f"构建版本: {FINAL_VERSION}")
    update_ver_env(FINAL_VERSION)

# QR模型路径
qrdet_model_path = os.path.join(env_path, 'qrdet', '.model') if env_path else None

# 定义 block_cipher
block_cipher = None

# 获取 hooks 目录
hooks_dir = build_config.get('hooks', ['hooks'])
if isinstance(hooks_dir, list):
    hooks_dir = hooks_dir[0] if hooks_dir else 'hooks'

# 构建导入列表
REQUIRED_IMPORTS = build_config.get('required_imports', [])
EXCLUDE_IMPORTS = build_config.get('exclude_imports', [])
print(f"📦 必需导入模块数量: {len(REQUIRED_IMPORTS)}")
print(f"🚫 排除导入模块数量: {len(EXCLUDE_IMPORTS)}")

# 从hidden_imports.json添加额外导入
if os.path.exists('hidden_imports.json'):
    try:
        with open('hidden_imports.json', 'r') as f:
            json_imports = json.load(f)
            additional_imports = [imp.strip('"') for imp in json_imports 
                                if imp not in REQUIRED_IMPORTS]
            REQUIRED_IMPORTS.extend(additional_imports)
            print(f"📦 从hidden_imports.json添加了 {len(additional_imports)} 个额外导入")
    except Exception as e:
        print(f"读取hidden_imports.json失败: {e}")

# 构建数据文件列表
base_datas = []

# 添加基础数据文件
if qrdet_model_path and os.path.exists(qrdet_model_path):
    base_datas.append((qrdet_model_path, 'qrdet/.model'))
    print(f"✅ 添加QR模型: {qrdet_model_path}")

base_datas.extend([
    ('APP.ENV', '.'),
    ('hidden_imports.json', '.'),
])

# 添加hooks文件
if os.path.exists(hooks_dir):
    for hook_file in ['torch_fixes.py', 'torch_numpy_fix.py']:
        hook_path = os.path.join(hooks_dir, hook_file)
        if os.path.exists(hook_path):
            base_datas.append((hook_path, '.'))
            print(f"✅ 添加Hook文件: {hook_path}")

# 添加scipy数据
if env_path:
    scipy_path = os.path.join(env_path, 'scipy/_lib/array_api_compat/numpy')
    if os.path.exists(scipy_path):
        base_datas.append((scipy_path, 'scipy/_lib/array_api_compat/numpy'))
        print(f"✅ 添加Scipy兼容数据: {scipy_path}")

# 添加watermark模块
if os.path.exists('watermark'):
    base_datas.append(('watermark', 'watermark'))
    print(f"✅ 添加Watermark模块: watermark")

# 处理BUILD.ENV中的datas配置
print("📂 开始处理数据文件配置...")
config_datas, collected_packages = expand_data_paths(build_config.get('datas', []))

# 合并所有数据文件
all_datas = base_datas + config_datas

print(f"📦 数据文件总数: {len(all_datas)}")

# 创建简洁的显示列表
display_datas = base_datas.copy()
for package in collected_packages:
    display_datas.append(f"*collect_data_files('{package}')")
if MORE_DETAILS:
    print(f"📦 数据文件: {display_datas}")

# ========================================
# PyInstaller 配置
# ========================================

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=all_datas,
    hiddenimports=REQUIRED_IMPORTS,
    hookspath=[hooks_dir] if os.path.exists(hooks_dir) else [],
    hooksconfig={},
    runtime_hooks=[os.path.join(hooks_dir, hook) for hook in build_config.get('runtime_hooks', []) 
                if os.path.exists(os.path.join(hooks_dir, hook))],
    excludes=EXCLUDE_IMPORTS,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

# 配置PYZ压缩
if build_config['enable_compress']:
    compress_format = build_config['compress_format'].lower()
    compress_level = build_config['compress_level']
    
    if compress_format == 'lzma':
        pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher, 
                compression=lzma, compression_level=compress_level)
        print(f"🗜️ 使用LZMA压缩，级别: {compress_level}")
    elif compress_format == 'zip':
        import zipfile
        pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher, 
                compression=zipfile.ZIP_DEFLATED, compression_level=compress_level)
        print(f"🗜️ 使用ZIP压缩，级别: {compress_level}")
    else:
        pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
        print("📦 无压缩")
else:
    pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
    print("📦 无压缩")

# 配置EXE
exe_config = {
    'name': f'BlindWatermarkGUI_v{FILENAME_VERSION}',
    'debug': build_config['enable_console_debug'],
    'bootloader_ignore_signals': False,
    'strip': False,
    'upx': False,  # 默认关闭UPX，因为可能导致问题
    'upx_exclude': [],
    'runtime_tmpdir': None,
    'console': build_config['enable_console_debug'],
    'disable_windowed_traceback': False,
    'argv_emulation': False,
    'target_arch': None,
    'codesign_identity': None,
    'entitlements_file': None,
    'onefile': build_config['one_file_mode'],
    'optimize': 2 if build_config['enable_optimize'] else 0
}

print(f"🔧 EXE配置:")
print(f"   - 调试模式: {exe_config['debug']}")
print(f"   - 控制台: {exe_config['console']}")
print(f"   - 单文件: {exe_config['onefile']}")
print(f"   - 优化级别: {exe_config['optimize']}")

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    **exe_config
)

print("✅ 程序构建完毕")