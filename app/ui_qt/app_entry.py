from __future__ import annotations

from pathlib import Path
import sys

from PySide6.QtWidgets import QApplication

from app.config.settings import load_settings
from app.ui_qt.main_window import MainWindow


def _try_apply_qss(app: QApplication) -> None:
    qss_path = Path(__file__).resolve().parent / "resources" / "style.qss"
    if not qss_path.exists():
        return
    try:
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))
    except Exception:
        pass


def _update_native_splash(text: str) -> None:
    """更新原生启动画面状态；不可用时静默跳过。"""
    try:
        from app.ui_native import splash

        splash.set_status(text)
    except Exception:
        pass


def _close_native_splash() -> None:
    """关闭原生启动画面，并打印显示时长便于对比启动耗时。"""
    try:
        from app.ui_native import splash

        ms = splash.close()
        if ms:
            print(f"原生启动画面已关闭，显示时长 {ms} ms")
    except Exception:
        pass


def run_qt_app(argv: list[str] | None = None) -> int:
    _update_native_splash("正在加载界面框架...")
    app = QApplication(argv or sys.argv)
    app.setApplicationName("BlindWatermarkGUI")
    app.setOrganizationName("Radium-bit")

    _update_native_splash("正在准备界面资源...")
    _try_apply_qss(app)

    _update_native_splash("正在读取配置...")
    settings = load_settings()

    _update_native_splash("正在构建用户界面...")
    window = MainWindow(settings)
    window.show()

    # 主窗口已显示后再关闭启动画面，避免出现"既没有窗口也没有提示"的空档。
    _close_native_splash()
    return app.exec()
