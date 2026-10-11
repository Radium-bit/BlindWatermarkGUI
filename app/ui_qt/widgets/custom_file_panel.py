from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class CustomFilePanel(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("自定义文件", parent)

        self.file_path_edit = QLineEdit(self)
        self.file_path_edit.setPlaceholderText("请选择要嵌入的自定义文件（文本/图片/二进制）")

        browse_button = QPushButton("选择文件", self)
        browse_button.clicked.connect(self._pick_file)

        row = QHBoxLayout()
        row.addWidget(self.file_path_edit)
        row.addWidget(browse_button)

        tip = QLabel("说明：文件模式默认支持 RC1 路径，提取时会自动识别输出扩展名。", self)
        tip.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(tip)

    def payload_path(self) -> Path:
        return Path(self.file_path_edit.text().strip())

    def set_visible_for_mode(self, visible: bool) -> None:
        self.setVisible(visible)

    def _pick_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择自定义文件")
        if file_path:
            self.file_path_edit.setText(file_path)
