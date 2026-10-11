from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout, QWidget


class ProgressDialog(QDialog):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("处理中")
        self.setModal(True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setMinimumWidth(360)

        self._label = QLabel(title, self)
        self._label.setWordWrap(True)
        progress = QProgressBar(self)
        progress.setRange(0, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(progress)

    def set_message(self, text: str) -> None:
        self._label.setText(text)
