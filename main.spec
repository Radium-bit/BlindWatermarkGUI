# -*- mode: python ; coding: utf-8 -*-
## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms
from PyInstaller.utils.hooks import collect_data_files
import os
import json
import lzma
import subprocess
import shutil
import py7zr
from dotenv import load_dotenv

load_dotenv('DEV.ENV')
load_dotenv('BUILD.ENV', override=True)
load_dotenv('APP.ENV')

env_path = os.getenv('SITE_PACKAGE_PATH')
BUILD_VERSION = os.getenv('BUILD_VERSION')
COMPRESS_LEVEL = os.getenv('COMPRESS_LEVEL')
OPTIMIZE = os.getenv('OPTIMIZE')
PROGRAM_GUID = os.getenv('PROGRAM_GUID')
INCLUDE_ONEFILE = os.getenv('INCLUDE_ONEFILE', 'false').lower() == 'true'
ENABLE_CONSOLE = os.getenv('ENABLE_CONSOLE_DEBUG', 'false').lower() == 'true'

# 控制选项：是否在版本号后添加 Git hash
INCLUDE_GIT_HASH = os.getenv('INCLUDE_GIT_HASH', 'false').lower() == 'true'

# 控制选项：是否额外打包Protable模式
INCLUDE_PROTABLE = os.getenv('INCLUDE_PROTABLE', 'false').lower() == 'true'

# 控制选项：是否额外打包安装程序模式
INCLUDE_MSI = os.getenv('INCLUDE_INSTALLER', 'false').lower() == 'true'

def get_git_hash():
    """获取当前 Git commit 的短 hash"""
    try:
        # 获取短 hash (7位)
        result = subprocess.run(['git', 'rev-parse', '--short=7', 'HEAD'], 
                            capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 如果 git 命令失败或找不到，返回默认值
        print(f"🔧 获取Git hash失败，使用默认值: {result.stderr if 'result' in locals() else 'Git未安装或不在PATH中'}")
        return "unknown"

def update_ver_env(version):
    """更新 APP.ENV 文件中的版本号"""
    try:
        # 读取现有的 APP.ENV 内容
        ver_env_content = ""
        if os.path.exists('APP.ENV'):
            with open('APP.ENV', 'r', encoding='utf-8') as f:
                ver_env_content = f.read()
        
        # 更新或添加 VERSION 字段
        lines = ver_env_content.split('\n')
        version_updated = False
        
        for i, line in enumerate(lines):
            if line.strip().startswith('VERSION='):
                lines[i] = f"VERSION='{version}'"
                version_updated = True
                break
        
        # 如果没有找到 VERSION 字段，添加它
        if not version_updated:
            if lines and lines[-1].strip():  # 如果最后一行不为空，添加新行
                lines.append('')
            lines.append(f"VERSION='{version}'")
        
        # 写回文件
        with open('APP.ENV', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 已更新APP.ENV中的版本号为: {version}")
        
    except Exception as e:
        print(f"❌ 更新APP.ENV失败: {e}")

def create_7z_archive(source_dir, output_file):
    """创建7z压缩包"""
    try:
        print(f"📦 正在创建Portable压缩包...")
        print(f"   源目录：{source_dir}")
        print(f"   输出：{output_file}")
        filters = [
            {
                "id": py7zr.FILTER_LZMA2,  # -m0=LZMA2
                "preset": 9,               # -mx9 (压缩级别)
                "dict_size": 64 * 1024 * 1024,  # 64MB字典
            }]
        with py7zr.SevenZipFile(output_file, 'w', filters=filters) as archive:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    archive.write(file_path, arcname)
        print(f"✅ Portable压缩包创建成功: {output_file}")
        return True
    except Exception as e:
        print(f"❌ 创建Portable压缩包失败: {e}")
        return False

def create_NSIS_installer(dist_dir, main_program_name, program_guid, installer_file, version):
    """
    使用 NSIS 创建 Windows 安装程序
   
    Args:
        dist_dir: 要打包的目录路径
        main_program_name: 主程序名称（在dist_dir内）
        program_guid: 程序的GUID，用于卸载识别
        installer_file: 输出的安装程序文件路径
        version: 程序版本号
   
    Returns:
        bool: 创建成功返回True，失败返回False
    """
    try:
        # 检查输入参数
        if not os.path.exists(dist_dir):
            print(f"❌ 错误：源目录不存在: {dist_dir}")
            return False
       
        main_program_path = os.path.join(dist_dir, main_program_name)
        if not os.path.exists(main_program_path):
            print(f"❌ 错误：主程序不存在: {main_program_path}")
            return False
       
        # 检查 NSIS 是否安装
        nsis_path = find_nsis()
        if not nsis_path:
            print("❌ 错误：未找到 NSIS，请确保已安装 NSIS")
            print("   下载地址: https://nsis.sourceforge.io/Download")
            return False
       
        print(f"📦 开始创建 NSIS 安装程序...")
        print(f"   源目录: {dist_dir}")
        print(f"   主程序: {main_program_name}")
        print(f"   输出: {installer_file}")
       
        # 确保输出目录存在
        os.makedirs(os.path.dirname(installer_file), exist_ok=True)
       
        # 创建临时的 NSIS 脚本
        script_content = generate_nsis_script(
            dist_dir, main_program_name, program_guid, installer_file, version
        )
       
        # 写入到 build.nsi 中
        script_path = os.path.join("build.nsi")
        with open(script_path, 'w', encoding='utf-8-sig') as f:  # 使用 UTF-8 BOM 编码
            f.write(script_content)
       
        try:
            # 编译 NSIS 脚本
            cmd = [nsis_path, script_path]
            
            # 设置环境变量以支持UTF-8
            env = os.environ.copy()
            env['NSIS_UNICODE'] = '1'
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',  # 更改为 replace 以避免编码错误
                env=env
            )
           
            if result.returncode == 0:
                if os.path.exists(installer_file):
                    file_size = os.path.getsize(installer_file) / (1024 * 1024)
                    print(f"✅ NSIS 安装程序创建成功!")
                    print(f"   文件: {installer_file}")
                    print(f"   大小: {file_size:.2f} MB")
                    return True
                else:
                    print(f"❌ 编译成功但未找到输出文件: {installer_file}")
                    return False
            else:
                print(f"❌ NSIS 编译失败:")
                print(f"   返回码: {result.returncode}")
                if result.stdout:
                    print(f"   输出: {result.stdout}")
                if result.stderr:
                    print(f"   错误: {result.stderr}")
                return False
               
        finally:
            # 清理临时脚本文件
            try:
                os.unlink(script_path)
            except:
                pass
               
    except Exception as e:
        print(f"❌ 创建 NSIS 安装程序时发生错误: {e}")
        return False

def generate_nsis_script(dist_dir, main_program_name, program_guid, installer_file, version, license_file="LICENSE.rtf"):
    """
    生成NSIS脚本内容，支持版本检查和覆盖安装提示
    
    参数:
    - dist_dir: 发布目录
    - main_program_name: 主程序名称
    - program_guid: 程序GUID
    - installer_file: 安装程序文件名
    - version: 版本号
    - license_file: 许可证文件路径（默认为LICENSE.rtf）
    """
    # 提取程序名称（去掉扩展名和版本信息）
    program_name = main_program_name.split('_')[0] if '_' in main_program_name else main_program_name.replace('.exe', '')
    
    # 检查许可证文件是否存在
    license_path = os.path.abspath(license_file) if os.path.exists(license_file) else ""
    
    # 获取dist目录的绝对路径并转换为Windows路径格式
    dist_dir_abs = os.path.abspath(dist_dir)
    
    # 使用纯ASCII字符串，避免中文编码问题
    script_content = f'''
; NSIS Script for {program_name}
; Generated automatically with version check support

!define PRODUCT_NAME "{program_name}"
!define PRODUCT_VERSION "{version}"
!define MAIN_PROGRAM_NAME "{main_program_name}"
!define PRODUCT_PUBLISHER "Radiumbit"
!define PRODUCT_WEB_SITE "https://github.com/Radium-bit/BlindWatermarkGUI"
!define PRODUCT_DIR_REGKEY "Software\\Microsoft\\Windows\\CurrentVersion\\App Paths\\BlindWatermarkGUI"
!define PRODUCT_UNINST_KEY "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{program_guid}"
!define PRODUCT_UNINST_ROOT_KEY "HKLM"

; Modern UI
!include "MUI2.nsh"
!include "Sections.nsh"
!include "LogicLib.nsh"
!include "FileFunc.nsh"
!include "WinMessages.nsh"

; General
Name "${{PRODUCT_NAME}} ${{PRODUCT_VERSION}}"
OutFile "{installer_file}"
InstallDir "$PROGRAMFILES\\${{PRODUCT_NAME}}"
InstallDirRegKey HKLM "${{PRODUCT_DIR_REGKEY}}" ""
ShowInstDetails show
ShowUnInstDetails show
SetCompressor /SOLID lzma
SetCompressorDictSize 64

; Variables
Var ExistingPath
Var IsUpgrade
Var OldMainProgram

; Interface Settings
!define MUI_ABORTWARNING

; Pages
!insertmacro MUI_PAGE_WELCOME
'''
    
    # 如果许可证文件存在，添加许可证页面
    if license_path:
        script_content += f'''
; License page
!insertmacro MUI_PAGE_LICENSE "{license_path}"
'''
    
    script_content += '''
!insertmacro MUI_PAGE_DIRECTORY

; Components page for optional features
!insertmacro MUI_PAGE_COMPONENTS

!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

; Uninstaller pages
!insertmacro MUI_UNPAGE_INSTFILES

; Language files
!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "SimpChinese"

; Reserve files
ReserveFile /plugin InstallOptions.dll

; Component descriptions
LangString DESC_MainProgram ${LANG_ENGLISH} "Main program files (required)"
LangString DESC_MainProgram ${LANG_SIMPCHINESE} "主程序文件（必需）"
LangString DESC_DesktopShortcut ${LANG_ENGLISH} "Create desktop shortcut"
LangString DESC_DesktopShortcut ${LANG_SIMPCHINESE} "创建桌面快捷方式"

; Installation overwrite messages
LangString MSG_ProgramExists ${LANG_ENGLISH} "A program already exists in the target directory.$\\r$\\n$\\r$\\nDo you want to overwrite the existing installation?"
LangString MSG_ProgramExists ${LANG_SIMPCHINESE} "目标目录已存在程序。$\\r$\\n$\\r$\\n是否要覆盖现有安装？"

; Function to check existing installation
Function .onInit
  ; Initialize variables
  StrCpy $IsUpgrade "false"
  
  ; Check if program is already installed
  ReadRegStr $ExistingPath HKLM "${PRODUCT_DIR_REGKEY}" ""
  ; Check if main program has recorded
  ReadRegStr $OldMainProgram HKLM "${PRODUCT_DIR_REGKEY}" "MainProgramName"
  
  ; If existing installation found
  ${If} $ExistingPath != ""
    ; Set install directory to existing path (remove executable name)
    Push $ExistingPath
    ;Call GetParent
    Pop $INSTDIR
    
    StrCpy $IsUpgrade "true"
    ; Show confirmation dialog
    MessageBox MB_YESNO|MB_ICONQUESTION "$(MSG_ProgramExists)" /SD IDYES IDYES +2
    Abort
  ${EndIf}
  
  ; Desktop shortcut is selected by default
FunctionEnd

; Function to get parent directory from full file path
Function GetParent
  Exch $R0
  Push $R1
  
  ; Use NSIS Function
  ${GetParent} $R0 $R1
  StrCpy $R0 $R1
  
  Pop $R1
  Exch $R0
FunctionEnd

; Install sections
Section "!${PRODUCT_NAME}" SEC_Main
  ; This section is required
  SectionIn RO
  
  SetOutPath "$INSTDIR"
  SetOverwrite on
  
  ; If this is an upgrade, show details
  ${If} $IsUpgrade == "true"
    ${AndIf} $OldMainProgram != ""
      DetailPrint "Removing old main program: $OldMainProgram"
      Delete "$INSTDIR\$OldMainProgram"
    DetailPrint "Overwriting existing installation"
  ${EndIf}
  
  ; Install all files from dist directory
  File /r "''' + dist_dir_abs + '''\\*.*"
  
  ; Create start menu shortcuts (always)
  CreateDirectory "$SMPROGRAMS\\${PRODUCT_NAME}"
  CreateShortCut "$SMPROGRAMS\\${PRODUCT_NAME}\\${PRODUCT_NAME}.lnk" "$INSTDIR\\${MAIN_PROGRAM_NAME}"
  CreateShortCut "$SMPROGRAMS\\${PRODUCT_NAME}\\Uninstall.lnk" "$INSTDIR\\uninst.exe"
  
  ; Register installation
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "" "$INSTDIR"
  WriteRegStr HKLM "${PRODUCT_DIR_REGKEY}" "MainProgramName" "${MAIN_PROGRAM_NAME}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\\uninst.exe"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\\${MAIN_PROGRAM_NAME}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteUninstaller "$INSTDIR\\uninst.exe"
  
  ; Update version in registry
  WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
SectionEnd

Section "Desktop Shortcut" SEC_Desktop
  ; This section is optional
  CreateShortCut "$DESKTOP\\${PRODUCT_NAME}.lnk" "$INSTDIR\\${MAIN_PROGRAM_NAME}"
SectionEnd

; Section descriptions
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_Main} $(DESC_MainProgram)
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_Desktop} $(DESC_DesktopShortcut)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; Uninstaller section
Section Uninstall
  ; Remove shortcuts
  Delete "$SMPROGRAMS\\${PRODUCT_NAME}\\${PRODUCT_NAME}.lnk"
  Delete "$SMPROGRAMS\\${PRODUCT_NAME}\\Uninstall.lnk"
  Delete "$DESKTOP\\${PRODUCT_NAME}.lnk"
  RMDir "$SMPROGRAMS\\${PRODUCT_NAME}"
  
  ; Remove installation directory
  RMDir /r "$INSTDIR"
  
  ; Remove registry keys
  DeleteRegKey HKLM "${PRODUCT_UNINST_KEY}"
  DeleteRegKey HKLM "${PRODUCT_DIR_REGKEY}"
  
  SetAutoClose true
SectionEnd
'''
    
    return script_content

def find_nsis():
    """
    查找NSIS安装路径
    """
    possible_paths = [
        "C:\\Program Files (x86)\\NSIS\\makensis.exe",
        "C:\\Program Files\\NSIS\\makensis.exe",
        "makensis.exe"  # 如果在PATH中
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 尝试在PATH中查找
    try:
        result = subprocess.run(['where', 'makensis'], 
                            capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    
    return None

def create_msi_with_legacy_wix(source_dir, output_file, version):
    raise Exception("不再支持旧版构建。")

# 构建最终版本号
if INCLUDE_GIT_HASH:
    git_hash = get_git_hash()
    FINAL_VERSION = f"{BUILD_VERSION}.build.{git_hash}"  # 用于 APP.ENV
    FILENAME_VERSION = f"{BUILD_VERSION}_build.{git_hash}"  # 用于文件名
    print(f"🏗️  构建版本: {FINAL_VERSION}")
    
    # 更新 APP.ENV 文件
    update_ver_env(FINAL_VERSION)
else:
    FINAL_VERSION = BUILD_VERSION
    FILENAME_VERSION = BUILD_VERSION
    print(f"🏗️  构建版本: {FINAL_VERSION}")
    
    # 更新 APP.ENV 文件
    update_ver_env(FINAL_VERSION)

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
        ('APP.ENV', '.'),
        # 包含修复文件
        (os.path.join(hooks_dir, 'torch_fixes.py'), '.'),
        (os.path.join(hooks_dir, 'torch_numpy_fix.py'), '.'),
        (os.path.join(env_path, 'scipy/_lib/array_api_compat/numpy'), 'scipy/_lib/array_api_compat/numpy'),
        ('hidden_imports.json', '.'),
        *collect_data_files('ultralytics'),
        ## 拆分后的模块
        # watermark 模块
        ('watermark', 'watermark'),
        ## Microsoft Visual C++ Redistributable (x64)
        ('thirdParty/VC_redist.x64.exe','.')
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
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher, compression=lzma, compression_level=COMPRESS_LEVEL)

# 根据配置决定打包方式
if INCLUDE_PROTABLE or INCLUDE_MSI:
    # 需要额外打包onedir模式
    print("📦 检测到需要额外打包，生成PrePackage中...")
    
    # 先创建onedir版本
    exe_dir = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=f'BlindWatermarkGUI_v{FILENAME_VERSION}_d',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        optimize=OPTIMIZE
    )
    
    coll = COLLECT(
        exe_dir,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=f'BlindWatermarkGUI_v{FILENAME_VERSION}_d'
    )
    if INCLUDE_ONEFILE:
        print("📦 检测到需要额外打包，生成Onefile中...")
        # 创建onefile版本（原有逻辑）
        exe = EXE(
            pyz,
            a.scripts,
            a.binaries,
            a.datas,
            [],
            name=f'BlindWatermarkGUI_v{FILENAME_VERSION}',
            debug=False,
            bootloader_ignore_signals=False,
            strip=False,
            upx=True,
            upx_exclude=[],
            runtime_tmpdir=None,
            console=ENABLE_CONSOLE,
            disable_windowed_traceback=False,
            argv_emulation=False,
            target_arch=None,
            codesign_identity=None,
            entitlements_file=None,
            onefile=True,
            optimize=OPTIMIZE
        )
    
    # 后处理：创建7z包和MSI安装包
    import atexit
    
    def post_build():
        dist_dir = os.path.join('dist', f'BlindWatermarkGUI_v{FILENAME_VERSION}_d')
        main_program_name = f'BlindWatermarkGUI_v{FILENAME_VERSION}_d.exe'
        
        # 记录需要执行的任务
        portable_success = True
        msi_success = True
        
        # 打包前先把LICENSE.rtf复制到dist_dir（若有）
        license_file_path = 'LICENSE.rtf'
        if license_file_path and os.path.exists(license_file_path):
            license_dest = os.path.join(dist_dir, os.path.basename(license_file_path))
            shutil.copy2(license_file_path, license_dest)
            print(f"📄 已复制许可证文件: {os.path.basename(license_file_path)}")
        elif license_file_path:
            print(f"⚠️  许可证文件不存在: {license_file_path}")
        
        if INCLUDE_PROTABLE and os.path.exists(dist_dir):
            # 创建Portable 7z包
            portable_7z = os.path.join('dist',f'BlindWatermarkGUI_v{FILENAME_VERSION}_Portable.7z')
            portable_success = create_7z_archive(dist_dir, portable_7z)
        
        if INCLUDE_MSI and os.path.exists(dist_dir):
            # 创建安装包
            installer_file = os.path.join('dist',f'BlindWatermarkGUI_v{FILENAME_VERSION}_Installer.exe')
            # msi_success = create_msi_installer(dist_dir, msi_file, FINAL_VERSION)
            msi_success = create_NSIS_installer(dist_dir,main_program_name,PROGRAM_GUID, installer_file, FINAL_VERSION)
        
        # 只有在所有任务都成功完成后才清理onedir目录
        if (not INCLUDE_PROTABLE or portable_success) and (not INCLUDE_MSI or msi_success):
            if os.path.exists(dist_dir):
                try:
                    print(f"🧹 正在清理临时目录: {dist_dir}")
                    shutil.rmtree(dist_dir)
                    print(f"✅ 临时目录清理完成")
                except Exception as e:
                    print(f"⚠️  清理临时目录失败: {e}")
            
            print("🎉 所有构建任务完成!")
        else:
            print("⚠️  部分构建任务失败，保留临时目录以供调试")
    
    atexit.register(post_build)
    
else:
    # 只打包onefile版本（原有逻辑）
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=f'BlindWatermarkGUI_v{FILENAME_VERSION}',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=ENABLE_CONSOLE,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        onefile=True,
        optimize=OPTIMIZE
    )