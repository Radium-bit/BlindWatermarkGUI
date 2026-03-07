from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)


class EmbedPanel(QWidget):
    def __init__(self, default_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.input_image_edit = QLineEdit(self)
        self.input_image_edit.setPlaceholderText("请选择待嵌入的图片文件")
        self.output_name_edit = QLineEdit(self)
        self.output_name_edit.setPlaceholderText("可选：自定义输出文件名（不含扩展名）")
        self.watermark_text_edit = QPlainTextEdit(self)
        self.watermark_text_edit.setPlainText(default_text)
        self.run_button = QPushButton("开始嵌入文本水印", self)

        browse_button = QPushButton("选择图片", self)
        browse_button.clicked.connect(self._pick_image)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_image_edit)
        input_layout.addWidget(browse_button)

        form = QFormLayout()
        form.addRow("输入图片", input_layout)
        form.addRow("输出命名", self.output_name_edit)
        form.addRow("水印文本", self.watermark_text_edit)

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(form)
        root_layout.addWidget(self.run_button)

    def _pick_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if file_path:
            self.input_image_edit.setText(file_path)

    def build_output_filename(self, input_image: Path) -> str:
        custom_name = self.output_name_edit.text().strip()
        suffix = input_image.suffix or ".png"
        if custom_name:
            return f"{custom_name}{suffix}"
        return f"{input_image.stem}-QtWatermark{suffix}"

