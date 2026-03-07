from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class OptionsPanel(QGroupBox):
    def __init__(
        self,
        default_password: str,
        default_output_dir: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("全局参数", parent)

        self.password_edit = QLineEdit(self)
        self.password_edit.setText(default_password)
        self.password_edit.setPlaceholderText("请输入数字密码")

        self.output_dir_edit = QLineEdit(self)
        self.output_dir_edit.setText(default_output_dir)
        self.output_dir_edit.setPlaceholderText("请选择输出目录")

        browse_button = QPushButton("选择目录", self)
        browse_button.clicked.connect(self._pick_output_dir)
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(browse_button)

        self.compatibility_mode = QCheckBox("兼容模式（v1）", self)
        self.enhanced_mode = QCheckBox("增强模式", self)
        self.custom_file_mode = QCheckBox("自定义文件嵌入", self)
        self.rc1_mode = QCheckBox("RC1 路径", self)

        form = QFormLayout()
        form.addRow("密码", self.password_edit)
        form.addRow("输出目录", output_layout)

        toggles = QVBoxLayout()
        toggles.addWidget(self.compatibility_mode)
        toggles.addWidget(self.enhanced_mode)
        toggles.addWidget(self.custom_file_mode)
        toggles.addWidget(self.rc1_mode)

        root_layout = QVBoxLayout(self)
        root_layout.addLayout(form)
        root_layout.addLayout(toggles)

    def _pick_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir_edit.text().strip())
        if directory:
            self.output_dir_edit.setText(directory)

    def values(self) -> dict[str, object]:
        return {
            "password": self.password_edit.text().strip(),
            "output_dir": self.output_dir_edit.text().strip(),
            "compatibility_mode": self.compatibility_mode.isChecked(),
            "enhanced_mode": self.enhanced_mode.isChecked(),
            "custom_file_mode": self.custom_file_mode.isChecked(),
            "rc1_mode": self.rc1_mode.isChecked(),
        }

