from __future__ import annotations

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


class ExtractPanel(QWidget):
    def __init__(self, default_sizes: tuple[int, ...], parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.input_image_edit = QLineEdit(self)
        self.input_image_edit.setPlaceholderText("请选择待提取的图片文件")
        self.size_edit = QLineEdit(self)
        self.size_edit.setText(",".join(str(value) for value in default_sizes))
        self.result_text_edit = QPlainTextEdit(self)
        self.result_text_edit.setReadOnly(True)
        self.result_text_edit.setPlaceholderText("提取结果将在这里显示")
        self.run_button = QPushButton("开始提取文本水印", self)

        browse_button = QPushButton("选择图片", self)
        browse_button.clicked.connect(self._pick_image)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_image_edit)
        input_layout.addWidget(browse_button)

        form = QFormLayout()
        form.addRow("输入图片", input_layout)
        form.addRow("提取尺寸", self.size_edit)
        form.addRow("提取结果", self.result_text_edit)

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

    def parse_sizes(self) -> tuple[int, ...]:
        items = [chunk.strip() for chunk in self.size_edit.text().split(",")]
        values: list[int] = []
        for item in items:
            if item.isdigit():
                values.append(int(item))
        return tuple(values)

    def set_result_text(self, text: str) -> None:
        self.result_text_edit.setPlainText(text)

