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


def run_qt_app(argv: list[str] | None = None) -> int:
    app = QApplication(argv or sys.argv)
    app.setApplicationName("BlindWatermarkGUI")
    app.setOrganizationName("Radium-bit")

    _try_apply_qss(app)
    settings = load_settings()
    window = MainWindow(settings)
    window.show()
    return app.exec()

