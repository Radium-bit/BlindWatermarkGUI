## Copyright (c) 2025 Radium-bit
## SPDX-License-Identifier: Apache-2.0
## See LICENSE file for full terms

#!/usr/bin/env python3
"""
tkinterdnd2 版本属性修复脚本
在Nuitka构建之前运行此脚本
"""

import os
import sys
import site

def fix_tkinterdnd2_version():
    """修复tkinterdnd2模块的__version__属性"""
    try:
        import tkinterdnd2
        module_path = tkinterdnd2.__file__
        module_dir = os.path.dirname(module_path)
        
        # 检查是否已有__version__属性
        if hasattr(tkinterdnd2, '__version__'):
            print(f"✅ tkinterdnd2已有版本属性: {tkinterdnd2.__version__}")
            return True
        
        # 查找__init__.py文件
        init_file = os.path.join(module_dir, '__init__.py')
        
        if not os.path.exists(init_file):
            print(f"❌ 找不到__init__.py文件: {init_file}")
            return False
        
        # 读取现有内容
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有__version__定义
        if '__version__' in content:
            print("⚠️  __init__.py中已有__version__定义，但模块中不可访问")
            return False
        
        # 添加版本定义
        version_line = '__version__ = "0.3.0"\n'
        
        # 在文件开头添加版本信息（在导入之后）
        lines = content.split('\n')
        insert_pos = 0
        
        # 找到合适的插入位置（在文档字符串和导入之后）
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                insert_pos = i + 1
                break
        
        lines.insert(insert_pos, version_line.rstrip())
        
        # 写回文件
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"✅ 已添加__version__属性到: {init_file}")
        
        # 验证修复
        # 重新导入模块
        if 'tkinterdnd2' in sys.modules:
            del sys.modules['tkinterdnd2']
        
        import tkinterdnd2
        if hasattr(tkinterdnd2, '__version__'):
            print(f"✅ 验证成功，版本: {tkinterdnd2.__version__}")
            return True
        else:
            print("❌ 验证失败，版本属性仍不可访问")
            return False
            
    except ImportError:
        print("❌ tkinterdnd2模块未安装")
        return False
    except Exception as e:
        print(f"❌ 修复过程中出错: {e}")
        return False

def backup_and_restore_info():
    """显示备份和恢复信息"""
    try:
        import tkinterdnd2
        module_path = tkinterdnd2.__file__
        module_dir = os.path.dirname(module_path)
        init_file = os.path.join(module_dir, '__init__.py')
        backup_file = init_file + '.backup'
        
        print(f"\n📁 模块位置: {module_dir}")
        print(f"📄 __init__.py: {init_file}")
        print(f"💾 备份文件: {backup_file}")
        print(f"\n💡 如需恢复原始文件，请运行:")
        print(f"   copy \"{backup_file}\" \"{init_file}\"")
        
    except ImportError:
        pass

def create_backup():
    """创建备份文件"""
    try:
        import tkinterdnd2
        module_path = tkinterdnd2.__file__
        module_dir = os.path.dirname(module_path)
        init_file = os.path.join(module_dir, '__init__.py')
        backup_file = init_file + '.backup'
        
        if not os.path.exists(backup_file):
            with open(init_file, 'r', encoding='utf-8') as f:
                content = f.read()
            with open(backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 已创建备份文件: {backup_file}")
        else:
            print(f"ℹ️  备份文件已存在: {backup_file}")
            
    except Exception as e:
        print(f"⚠️  创建备份失败: {e}")

if __name__ == '__main__':
    print("=" * 50)
    print("🔧 tkinterdnd2版本属性修复工具")
    print("=" * 50)
    
    # 创建备份
    create_backup()
    
    # 执行修复
    success = fix_tkinterdnd2_version()
    
    if success:
        print("\n🎉 修复完成！现在可以运行Nuitka构建了。")
    else:
        print("\n❌ 修复失败，请尝试其他解决方案。")
    
    # 显示相关信息
    backup_and_restore_info()
    
    print("\n" + "=" * 50)
    
    # Windows下暂停
    if os.name == 'nt':
        input("按Enter键退出...")
    
    sys.exit(0 if success else 1)