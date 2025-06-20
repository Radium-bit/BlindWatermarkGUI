import sys
import builtins
import json

# 原始导入函数
_original_import = builtins.__import__

# 跟踪的模块集合
imported_modules = set()

def custom_import(name, globals=None, locals=None, fromlist=(), level=0):
    """自定义导入函数，记录所有导入的模块"""
    module = _original_import(name, globals, locals, fromlist, level)
    
    # 记录qreader及其依赖的所有包
    if name == 'qreader' or (fromlist and any('QReader' in item for item in fromlist)) or \
       any(m in name for m in ['numpy', 'torch', 'ultralytics']):
        module_name = name.split('.')[0]
        if module_name not in sys.builtin_module_names:
            imported_modules.add(module_name)
        
        if fromlist:
            for item in fromlist:
                if item != '*':
                    submodule_name = f"{name}.{item}"
                    imported_modules.add(submodule_name)
    
    return module

def start_tracking():
    """开始跟踪模块导入"""
    builtins.__import__ = custom_import
    print("模块跟踪已启动...")

def stop_and_save_tracking():
    """停止跟踪并保存结果"""
    builtins.__import__ = _original_import
    
    # 过滤掉不需要的模块
    filtered_modules = {m for m in imported_modules 
                        if not m.startswith(('_', 'pywin', 'win32', 'pkg_resources'))}
    
    # 保存结果到文件
    with open('hidden_imports.json', 'w') as f:
        json.dump(sorted(filtered_modules), f, indent=2)
    
    print(f"已保存 {len(filtered_modules)} 个隐藏导入到 hidden_imports.json")
    return filtered_modules

if __name__ == '__main__':
    start_tracking()
    # 在这里导入您的应用程序
    import sys
    import importlib

    def track_main_imports():
        """通过动态导入避免循环依赖"""
        if 'main' in sys.modules:
            return sys.modules['main']
        return importlib.import_module('main')
    stop_and_save_tracking()