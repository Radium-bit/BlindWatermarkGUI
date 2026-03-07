from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config.settings import AppSettings
from app.models.job_models import EmbedTextJob, ExtractTextJob, JobResult
from app.services.task_runner import TaskRunner
from app.services.watermark_service import WatermarkService
from app.ui_qt.widgets.embed_panel import EmbedPanel
from app.ui_qt.widgets.extract_panel import ExtractPanel
from app.ui_qt.widgets.options_panel import OptionsPanel
from app.ui_qt.widgets.progress_dialog import ProgressDialog


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.service = WatermarkService()
        self.task_runner = TaskRunner(max_workers=1)

        self._pending_future: Future | None = None
        self._pending_handler = None
        self._progress_dialog: ProgressDialog | None = None

        self._future_timer = QTimer(self)
        self._future_timer.setInterval(140)
        self._future_timer.timeout.connect(self._poll_future)

        self.setWindowTitle("BlindWatermarkGUI - Qt 预览版")
        self.resize(980, 760)

        self.options_panel = OptionsPanel(
            default_password=self.settings.default_password,
            default_output_dir=str(self.settings.default_output_dir),
            parent=self,
        )
        self.embed_panel = EmbedPanel(default_text=self.settings.default_wm_text, parent=self)
        self.extract_panel = ExtractPanel(
            default_sizes=self.settings.extract_candidate_sizes,
            parent=self,
        )
        self.log_box = QPlainTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("运行日志（仅本次会话）")

        self._build_ui()
        self._wire_events()
        self._append_log("Qt 预览版已启动，当前阶段优先覆盖文本水印流程。")

    def _build_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)

        tip_label = QLabel(
            "当前为 Qt 迁移第一阶段：优先支持文本水印嵌入/提取；复杂路径仍建议使用经典界面。",
            container,
        )
        tip_label.setWordWrap(True)

        tabs = QTabWidget(container)
        tabs.addTab(self.embed_panel, "嵌入")
        tabs.addTab(self.extract_panel, "提取")

        layout.addWidget(tip_label)
        layout.addWidget(self.options_panel)
        layout.addWidget(tabs)
        layout.addWidget(self.log_box)

        self.setCentralWidget(container)

    def _wire_events(self) -> None:
        self.embed_panel.run_button.clicked.connect(self._on_embed_clicked)
        self.extract_panel.run_button.clicked.connect(self._on_extract_clicked)

    def _on_embed_clicked(self) -> None:
        if self._has_running_task():
            return

        options = self.options_panel.values()
        self._warn_if_not_supported(options)

        input_path = Path(self.embed_panel.input_image_edit.text().strip())
        output_dir = Path(str(options["output_dir"]).strip() or str(self.settings.default_output_dir))
        output_name = self.embed_panel.build_output_filename(input_path)
        watermark_text = self.embed_panel.watermark_text_edit.toPlainText().strip()

        if not input_path.exists():
            QMessageBox.warning(self, "路径错误", "输入图片不存在，请检查路径。")
            return
        if not watermark_text:
            QMessageBox.warning(self, "参数错误", "水印文本不能为空。")
            return

        job = EmbedTextJob(
            input_image=input_path,
            output_image=output_dir / output_name,
            watermark_text=watermark_text,
            password=str(options["password"]),
        )
        self._submit_task("正在嵌入文本水印，请稍候...", lambda: self.service.embed_text(job), self._handle_embed_result)

    def _on_extract_clicked(self) -> None:
        if self._has_running_task():
            return

        options = self.options_panel.values()
        self._warn_if_not_supported(options)

        input_path = Path(self.extract_panel.input_image_edit.text().strip())
        if not input_path.exists():
            QMessageBox.warning(self, "路径错误", "输入图片不存在，请检查路径。")
            return

        sizes = self.extract_panel.parse_sizes()
        if not sizes:
            QMessageBox.warning(self, "参数错误", "提取尺寸请输入数字列表，例如：256,128,96,64")
            return

        output_dir = Path(str(options["output_dir"]).strip() or str(self.settings.default_output_dir))
        job = ExtractTextJob(
            input_image=input_path,
            password=str(options["password"]),
            candidate_sizes=sizes,
            debug_output_dir=output_dir,
        )
        self._submit_task("正在提取文本水印，请稍候...", lambda: self.service.extract_text(job), self._handle_extract_result)

    def _submit_task(self, text: str, task_func, done_handler) -> None:
        self._progress_dialog = ProgressDialog(text, self)
        self._progress_dialog.show()
        self._pending_future = self.task_runner.submit(task_func)
        self._pending_handler = done_handler
        self._future_timer.start()

    def _poll_future(self) -> None:
        if not self._pending_future:
            self._future_timer.stop()
            return
        if not self._pending_future.done():
            return

        self._future_timer.stop()
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None

        future = self._pending_future
        handler = self._pending_handler
        self._pending_future = None
        self._pending_handler = None

        try:
            result: JobResult = future.result()
        except Exception as exc:
            QMessageBox.critical(self, "任务失败", f"任务执行失败：{exc}")
            self._append_log(f"任务执行失败：{exc}")
            return

        if handler:
            handler(result)

    def _handle_embed_result(self, result: JobResult) -> None:
        if result.success:
            self._append_log(result.message)
            QMessageBox.information(self, "嵌入完成", result.message)
        else:
            self._append_log(result.message)
            QMessageBox.warning(self, "嵌入失败", result.message)

    def _handle_extract_result(self, result: JobResult) -> None:
        if result.success:
            final_text = result.extracted_text or ""
            self.extract_panel.set_result_text(final_text)
            self._append_log(result.message)
            QMessageBox.information(self, "提取完成", f"{result.message}\n\n{final_text}")
        else:
            self.extract_panel.set_result_text("")
            self._append_log(result.message)
            QMessageBox.warning(self, "提取失败", result.message)

    def _has_running_task(self) -> bool:
        if self._pending_future and not self._pending_future.done():
            QMessageBox.information(self, "提示", "已有任务在执行，请稍候。")
            return True
        return False

    def _warn_if_not_supported(self, options: dict[str, object]) -> None:
        unsupported: list[str] = []
        if options.get("compatibility_mode"):
            unsupported.append("兼容模式（v1）")
        if options.get("enhanced_mode"):
            unsupported.append("增强模式")
        if options.get("custom_file_mode"):
            unsupported.append("自定义文件嵌入")
        if options.get("rc1_mode"):
            unsupported.append("RC1 路径")

        if unsupported:
            text = "、".join(unsupported)
            QMessageBox.information(
                self,
                "阶段说明",
                f"以下选项在 Qt 第一阶段暂未接入：{text}\n将按默认文本路径执行。",
            )

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{timestamp}] {text}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.task_runner.shutdown()
        super().closeEvent(event)

