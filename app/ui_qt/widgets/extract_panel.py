from __future__ import annotations

import re

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class ExtractPanel(QWidget):
    def __init__(self, default_sizes: tuple[int, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.input_image_edit = QLineEdit(self)
        self.input_image_edit.setPlaceholderText("请选择待提取的图片文件")

        self.mode_combo = QComboBox(self)
        self.mode_combo.addItem("文本水印", "text")
        self.mode_combo.addItem("文件水印", "file")

        self.size_edit = QLineEdit(self)
        self.size_edit.setText(",".join(str(value) for value in default_sizes))
        self.size_edit.setPlaceholderText("文本提取尺寸，示例：256,128,96,64")

        self.compat_ws_edit = QLineEdit(self)
        self.compat_ws_edit.setPlaceholderText("兼容模式(v1)可选：手动填写 ws 长度")

        self.target_size_edit = QLineEdit(self)
        self.target_size_edit.setPlaceholderText("可选：原图尺寸，示例 1920x1080")

        self.result_text_edit = QPlainTextEdit(self)
        self.result_text_edit.setReadOnly(True)
        self.result_text_edit.setPlaceholderText("提取结果将在这里显示")

        self.run_button = QPushButton("开始提取", self)

        browse_button = QPushButton("选择图片", self)
        browse_button.clicked.connect(self._pick_image)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_image_edit)
        input_layout.addWidget(browse_button)

        form = QFormLayout()
        form.addRow("输入图片", input_layout)
        form.addRow("提取模式", self.mode_combo)
        form.addRow("提取尺寸", self.size_edit)
        form.addRow("兼容长度(ws)", self.compat_ws_edit)
        form.addRow("原图尺寸", self.target_size_edit)
        form.addRow("提取结果", self.result_text_edit)

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(form)
        root_layout.addWidget(self.run_button, alignment=Qt.AlignRight)

    def _pick_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if file_path:
            self.input_image_edit.setText(file_path)

    def parse_sizes(self) -> tuple[int, ...]:
        items = [chunk.strip() for chunk in self.size_edit.text().split(",")]
        values: list[int] = []
        for item in items:
            if item.isdigit():
                values.append(int(item))
        return tuple(values)

    def parse_compat_ws(self) -> int | None:
        text = self.compat_ws_edit.text().strip()
        if text.isdigit():
            return int(text)
        return None

    def parse_target_size(self) -> tuple[int, int] | None:
        text = self.target_size_edit.text().strip()
        match = re.match(r"^(\d+)x(\d+)$", text, flags=re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    def is_file_mode(self) -> bool:
        return str(self.mode_combo.currentData()) == "file"

    def force_text_mode(self) -> None:
        index = self.mode_combo.findData("text")
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)

    def set_result_text(self, text: str) -> None:
        self.result_text_edit.setPlainText(text)
