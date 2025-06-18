from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = collect_data_files('tkinterdnd2')  
# 添加动态库
binaries = collect_dynamic_libs('tkinterdnd2')  