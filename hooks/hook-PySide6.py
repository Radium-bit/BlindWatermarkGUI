# 迁移阶段先显式收集核心模块，避免条件导入导致漏包
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]
