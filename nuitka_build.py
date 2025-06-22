#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nuitka打包脚本 - 从BUILD.ENV自动读取配置
使用方法: python build_nuitka.py
"""

import os
import sys
import ast
import json
import subprocess
import re
from pathlib import Path
from dotenv import load_dotenv

class BuildConfigParser:
    """构建配置解析器"""
    
    def __init__(self):
        self.build_config = {}
        self.dev_config = {}
        
    def load_simple_env_configs(self):
        """加载简单的环境配置文件(仅KEY=VALUE格式)"""
        # 只加载APP.ENV，因为它通常是简单的KEY=VALUE格式
        if os.path.exists('APP.ENV'):
            load_dotenv('APP.ENV')
            print("✅ 已加载 APP.ENV")
    
    def parse_env_file(self, filepath):
        """解析环境配置文件，支持Python语法"""
        config = {}
        if not os.path.exists(filepath):
            return config
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析简单的KEY=VALUE行
            for line_num, line in enumerate(content.split('\n'), 1):
                original_line = line
                line = line.strip()
                
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                
                # 处理简单的KEY=VALUE格式
                if '=' in line and not any(line.startswith(x) for x in ['REQUIRED_IMPORTS', 'datas', 'HOOKS', 'RUNTIME_HOOKS']):
                    try:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # 处理布尔值
                        if value.lower() in ['true', 'false']:
                            config[key] = value.lower() == 'true'
                        # 处理数字
                        elif value.isdigit():
                            config[key] = int(value)
                        # 处理带引号的字符串
                        elif (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                            config[key] = value[1:-1]
                        else:
                            config[key] = value
                    except Exception as e:
                        print(f"警告: 第{line_num}行解析失败: {original_line}")
                        
            print(f"✅ 已解析 {filepath}")
            
        except Exception as e:
            print(f"解析{filepath}时出错: {e}")
            
        return config
    
    def parse_build_env(self):
        """解析BUILD.ENV文件中的Python配置"""
        if not os.path.exists('BUILD.ENV'):
            raise FileNotFoundError("BUILD.ENV 文件不存在")
            
        print("正在解析 BUILD.ENV...")
        
        # 先解析DEV.ENV获取基础配置
        dev_config = self.parse_env_file('DEV.ENV')
        
        # 解析BUILD.ENV中的简单配置
        build_config = self.parse_env_file('BUILD.ENV')
        
        # 手动解析复杂的Python配置
        self._parse_python_configs()
        
        # 合并配置
        self.build_config = {
            'include_git_hash': build_config.get('INCLUDE_GIT_HASH', True),
            'build_version': build_config.get('BUILD_VERSION', '1.0.0'),
            'enable_console_debug': build_config.get('ENABLE_CONSOLE_DEBUG', False),
            'enable_optimize': build_config.get('ENABLE_OPTIMIZE', True),
            'enable_compress': build_config.get('ENABLE_COMPRESS', True),
            'compress_format': build_config.get('COMPRESS_FORMAT', 'lzma'),
            'compress_level': build_config.get('COMPRESS_LEVEL', 9),
            'one_file_mode': build_config.get('ONE_FILE_MODE', True),
            'required_imports': getattr(self, 'required_imports', []),
            'datas': getattr(self, 'datas', []),
            'hooks': getattr(self, 'hooks', []),
            'runtime_hooks': getattr(self, 'runtime_hooks', []),
            'dev_config': dev_config,  # 保存DEV配置用于后续引用
        }
        
        print("✅ BUILD.ENV 解析完成")
        return self.build_config
    
    def _parse_python_configs(self):
        """手动解析BUILD.ENV中的Python配置块"""
        try:
            with open('BUILD.ENV', 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析REQUIRED_IMPORTS
            self._extract_python_list('REQUIRED_IMPORTS', content, 'required_imports')
            
            # 解析datas
            self._extract_python_list('datas', content, 'datas')
            
            # 解析HOOKS
            self._extract_python_list('HOOKS', content, 'hooks')
            
            # 解析RUNTIME_HOOKS
            self._extract_python_list('RUNTIME_HOOKS', content, 'runtime_hooks')
            
        except Exception as e:
            print(f"解析Python配置时出错: {e}")
            # 设置默认值
            self.required_imports = []
            self.datas = []
            self.hooks = []
            self.runtime_hooks = []
    
    def _extract_python_list(self, var_name, content, attr_name):
        """提取Python列表配置"""
        try:
            # 查找变量定义的开始
            start_pattern = f'{var_name} = ['
            start_pos = content.find(start_pattern)
            
            if start_pos == -1:
                setattr(self, attr_name, [])
                return
            
            # 找到列表的开始位置
            list_start = start_pos + len(f'{var_name} = ')
            
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
                setattr(self, attr_name, [])
                return
            
            # 提取列表内容
            list_content = content[list_start:list_end]
            
            # 手动解析列表项
            items = self._parse_list_items(list_content)
            setattr(self, attr_name, items)
            
            print(f"✅ 解析{var_name}: {len(items)}项")
            
        except Exception as e:
            print(f"解析{var_name}时出错: {e}")
            setattr(self, attr_name, [])
    
    def _parse_list_items(self, list_content):
        """解析列表项内容"""
        items = []
        
        # 移除外层括号
        content = list_content.strip()
        if content.startswith('['):
            content = content[1:]
        if content.endswith(']'):
            content = content[:-1]
        
        # 分割项目（简化处理，按行分割）
        lines = content.split('\n')
        current_item = ""
        
        for line in lines:
            line = line.strip()
            
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            
            # 移除末尾的逗号
            if line.endswith(','):
                line = line[:-1].strip()
            
            # 处理字符串项
            if line.startswith("'") and line.endswith("'"):
                items.append(line[1:-1])
            elif line.startswith('"') and line.endswith('"'):
                items.append(line[1:-1])
            # 处理特殊函数调用
            elif 'COLLECT(' in line:
                items.append(line)
            # 处理包含=>的数据映射
            elif '=>' in line:
                items.append(line.strip("'\""))
            # 处理其他格式
            elif line:
                items.append(line.strip("'\""))
        
        return items


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

def prepare_environment(build_config):
    """准备构建环境"""
    build_version = build_config['build_version']
    include_git_hash = build_config['include_git_hash']
    
    # 构建版本号
    if include_git_hash:
        git_hash = get_git_hash()
        final_version = f"{build_version}.build.{git_hash}"
        filename_version = f"{build_version}_build.{git_hash}"
        print(f"构建版本: {final_version}")
        update_ver_env(final_version)
    else:
        final_version = build_version
        filename_version = build_version
        print(f"构建版本: {final_version}")
        update_ver_env(final_version)
    
    return final_version, filename_version

def expand_data_paths(datas, dev_config):
    """展开数据文件路径"""
    expanded_datas = []
    
    for item in datas:
        if isinstance(item, str):
            # 处理COLLECT()格式
            if item.startswith("COLLECT("):
                # 提取包名
                match = re.search(r"COLLECT\('([^']+)'\)", item)
                if match:
                    package_name = match.group(1)
                    print(f"收集包数据: {package_name}")
                    try:
                        # 尝试导入包并获取其数据文件
                        import importlib
                        module = importlib.import_module(package_name)
                        module_path = Path(module.__file__).parent
                        
                        # 添加整个包目录
                        expanded_datas.append((str(module_path), package_name))
                    except ImportError:
                        print(f"警告: 无法导入包 {package_name}")
                continue
        
        # 处理字符串格式 "src=>dst"
        if isinstance(item, str) and '=>' in item:
            src, dst = item.split('=>', 1)
            src = src.strip()
            dst = dst.strip()
            
            # 处理DEV_ENV.SITE_PACKAGE_PATH引用
            if 'DEV_ENV.SITE_PACKAGE_PATH' in src:
                site_package_path = dev_config.get('SITE_PACKAGE_PATH', '')
                if site_package_path:
                    # 处理Windows路径中的反斜杠
                    site_package_path = site_package_path.replace('\\', '/')
                    src = src.replace('DEV_ENV.SITE_PACKAGE_PATH', site_package_path)
                    src = src.replace('/', os.sep)  # 转换为系统路径分隔符
                else:
                    print(f"警告: SITE_PACKAGE_PATH 未定义，跳过: {src}")
                    continue
            
            # 检查源文件/目录是否存在
            if os.path.exists(src):
                expanded_datas.append((src, dst))
            else:
                print(f"警告: 数据文件不存在: {src}")
        
        # 处理元组格式
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            src, dst = item
            if os.path.exists(src):
                expanded_datas.append((src, dst))
            else:
                print(f"警告: 数据文件不存在: {src}")
    
    return expanded_datas

def build_nuitka_command(build_config):
    """根据构建配置构建Nuitka命令"""
    final_version, filename_version = prepare_environment(build_config)
    
    # 基础命令
    cmd = ['python', '-m', 'nuitka']
    
    # 单文件模式
    if build_config['one_file_mode']:
        cmd.append('--onefile')
    else:
        cmd.append('--standalone')
    
    # 控制台调试
    if build_config['enable_console_debug']:
        cmd.append('--windows-console-mode=force')  # 强制显示控制台
    else:
        cmd.append('--windows-console-mode=disable')
    # 优化级别
    if build_config['enable_optimize']:
        cmd.append('--python-flag=-OO') # O2优化
        cmd.append('--lto=yes')  # 链接时优化

    # 反膨胀配置 - 排除测试相关模块（实际上去了就跑不起来，先琢磨琢磨再说）
    # anti_bloat_options = [
    #     '--noinclude-pytest-mode=nofollow',
    #     '--noinclude-unittest-mode=nofollow',
    #     '--noinclude-IPython-mode=nofollow',
    #     '--noinclude-setuptools-mode=nofollow',
    #     '--nofollow-import-to=pytest',
    #     '--nofollow-import-to=nose',
    #     '--nofollow-import-to=mock',
    #     '--nofollow-import-to=unittest',
    #     '--nofollow-import-to=doctest',
    #     '--nofollow-import-to=*.tests.*',
    #     '--nofollow-import-to=*test*',
    #     # 开发和构建工具
    #     '--nofollow-import-to=setuptools',
    #     '--nofollow-import-to=pip',
    #     '--nofollow-import-to=wheel',
    #     '--nofollow-import-to=distutils',
    #     '--nofollow-import-to=pkg_resources',
    #     # 文档和交互工具
    #     '--nofollow-import-to=pydoc',
    #     '--nofollow-import-to=sphinx',
    #     '--nofollow-import-to=docutils',
    #     '--nofollow-import-to=IPython',
    #     '--nofollow-import-to=jupyter',
    #     '--nofollow-import-to=notebook',
    #     # 库特定测试模块
    #     '--nofollow-import-to=numpy.testing',
    #     '--nofollow-import-to=scipy.testing',
    #     '--nofollow-import-to=PIL.tests',
    #     '--nofollow-import-to=cv2.tests',
    #     # 通用测试目录和文件
    #     '--nofollow-import-to=tests',
    #     '--nofollow-import-to=test',
    # ]
    # cmd.extend(anti_bloat_options)
    
    # 压缩设置
    if build_config['enable_compress'] and build_config['one_file_mode']:
        compress_format = build_config['compress_format'].lower()
        compress_level = build_config['compress_level']
        
        if compress_format == 'lzma':
            cmd.append('--onefile-tempdir-spec={CACHE_DIR}/{COMPANY}/{PRODUCT}/{VERSION}')
            # Nuitka 默认使用压缩，可以通过环境变量控制
            os.environ['NUITKA_ONEFILE_COMPRESSION'] = 'lzma'
        elif compress_format == 'zip':
            os.environ['NUITKA_ONEFILE_COMPRESSION'] = 'zlib'
        elif compress_format == 'none':
            cmd.append('--onefile-no-compression')
    
    # 其他基础选项
    cmd.extend([
        '--assume-yes-for-downloads',
        '--show-progress',
        # '--show-memory', # 编译后显示内存信息，一般没必要
        '--enable-plugin=tk-inter',
        '--enable-plugin=numpy',
        '--jobs=4',  # 并行编译
    ])
    # 确保输出目录存在
    os.makedirs('dist', exist_ok=True)
    # 输出文件名
    output_name = f'dist/BlindWatermarkGUI_v{filename_version}_Nuitka.exe'
    cmd.append(f'--output-filename={output_name}')
    
    # 添加隐藏导入
    required_imports = build_config.get('required_imports', [])
    
    # 从hidden_imports.json添加额外导入
    if os.path.exists('hidden_imports.json'):
        try:
            with open('hidden_imports.json', 'r') as f:
                json_imports = json.load(f)
                for imp in json_imports:
                    if isinstance(imp, str):
                        clean_imp = imp.strip('"\'')
                        if clean_imp not in required_imports:
                            required_imports.append(clean_imp)
        except Exception as e:
            print(f"读取hidden_imports.json失败: {e}")
    
    for module in required_imports:
        cmd.append(f'--include-module={module}')
    
    # 添加数据文件
    datas = expand_data_paths(build_config.get('datas', []), build_config.get('dev_config', {}))
    
    for src, dst in datas:
        if os.path.exists(src):
            if os.path.isdir(src):
                cmd.append(f'--include-data-dir={src}={dst}')
            else:
                cmd.append(f'--include-data-file={src}={dst}')
        else:
            print(f"警告: 数据文件不存在: {src}")
    
    # 特殊处理ultralytics数据文件
    try:
        import ultralytics
        ultralytics_path = Path(ultralytics.__file__).parent
        ultralytics_data = ultralytics_path / 'cfg'
        if ultralytics_data.exists():
            cmd.append(f'--include-data-dir={ultralytics_data}=ultralytics/cfg')
    except ImportError:
        print("警告: ultralytics模块未找到")
    
    # 添加包含整个包
    packages_to_include = [
        'torch', 'torchvision', 'ultralytics', 'qreader', 
        'qrcode', 'scipy', 'numpy', 'PIL', 'cv2'
    ]
    
    for package in packages_to_include:
        cmd.append(f'--include-package={package}')
    
    # 添加Hooks
    hooks = build_config.get('hooks', [])
    for hook_dir in hooks:
        if os.path.exists(hook_dir):
            cmd.append(f'--include-data-dir={hook_dir}={hook_dir}')
    
    # 添加Runtime Hooks
    runtime_hooks = build_config.get('runtime_hooks', [])
    for hook_file in runtime_hooks:
        if os.path.exists(hook_file):
            cmd.append(f'--include-data-file={hook_file}=.')
    
    # 禁用一些不需要的插件以加快构建
    cmd.extend([
        '--disable-plugin=multiprocessing',
    ])
    
    # 主脚本
    cmd.append('main.py')
    
    return cmd

def run_build():
    """运行构建过程"""
    print("=" * 60)
    print("🚀 Nuitka 自动构建脚本 (BUILD.ENV版)")
    print("=" * 60)
    
    # 检查必要文件
    required_files = ['main.py', 'BUILD.ENV']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ 错误: 缺少必要文件: {', '.join(missing_files)}")
        return False
    
    # 检查Nuitka是否安装
    try:
        result = subprocess.run(['python', '-m', 'nuitka', '--version'], 
                            capture_output=True, text=True, check=True)
        print(f"✅ Nuitka版本: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: Nuitka未安装或无法运行")
        print("请先安装: pip install nuitka")
        return False
    
    try:
        # 解析构建配置
        parser = BuildConfigParser()
        parser.load_simple_env_configs()  # 只加载APP.ENV等简单配置
        build_config = parser.parse_build_env()  # 解析BUILD.ENV的复杂配置
        
        print("✅ 构建配置解析完成")
        print(f"📋 配置概览:")
        print(f"   - 版本: {build_config['build_version']}")
        print(f"   - Git Hash: {'包含' if build_config['include_git_hash'] else '不包含'}")
        print(f"   - 单文件模式: {'是' if build_config['one_file_mode'] else '否'}")
        print(f"   - 控制台调试: {'开启' if build_config['enable_console_debug'] else '关闭'}")
        print(f"   - 优化: {'开启' if build_config['enable_optimize'] else '关闭'}")
        print(f"   - 压缩: {'开启' if build_config['enable_compress'] else '关闭'}")
        if build_config['enable_compress']:
            print(f"   - 压缩格式: {build_config['compress_format']}")
            print(f"   - 压缩级别: {build_config['compress_level']}")
        
        # 构建命令
        cmd = build_nuitka_command(build_config)
        
        print(f"\n📋 执行命令:")
        print(' '.join(cmd))
        print(f"\n" + "=" * 60)
        print("🔨 开始构建...")
        print("=" * 60)
        
        # 执行构建
        result = subprocess.run(cmd, check=True)
        
        print("=" * 60)
        print("🎉 构建完成!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ 构建失败: {e}")
        print("=" * 60)
        return False
    except KeyboardInterrupt:
        print("\n⚠️  用户取消构建")
        return False

if __name__ == '__main__':
    success = run_build()
    
    if success:
        print("🎯 提示: 可执行文件已生成在当前目录")
    else:
        print("💡 提示: 请检查错误信息并重试")
    
    # 在Windows下暂停，方便查看结果
    if os.name == 'nt':
        input("\n按Enter键退出...")
    
    sys.exit(0 if success else 1)