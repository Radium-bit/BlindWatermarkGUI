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
from app.models.job_models import (
    EmbedFileJob,
    EmbedTextJob,
    ExtractFileJob,
    ExtractTextJob,
    JobResult,
)
from app.services.qt_mode_guard import normalize_modes
from app.services.task_runner import TaskRunner
from app.services.watermark_service import WatermarkService
from app.ui_qt.widgets.custom_file_panel import CustomFilePanel
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
        self.resize(980, 780)

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
        self.custom_file_panel = CustomFilePanel(self)
        self.custom_file_panel.setVisible(False)

        self.log_box = QPlainTextEdit(self)
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("运行日志（仅本次会话）")

        self._build_ui()
        self._wire_events()
        self._refresh_mode_ui()
        self._append_log("Qt 阶段二模式已启用：支持文本与自定义文件链路。")

    def _build_ui(self) -> None:
        container = QWidget(self)
        layout = QVBoxLayout(container)

        tip_label = QLabel(
            "当前为 Qt 迁移阶段二：支持文本水印与自定义文件链路；默认入口仍为经典界面。",
            container,
        )
        tip_label.setWordWrap(True)

        tabs = QTabWidget(container)
        tabs.addTab(self.embed_panel, "嵌入")
        tabs.addTab(self.extract_panel, "提取")

        layout.addWidget(tip_label)
        layout.addWidget(self.options_panel)
        layout.addWidget(self.custom_file_panel)
        layout.addWidget(tabs)
        layout.addWidget(self.log_box)

        self.setCentralWidget(container)

    def _wire_events(self) -> None:
        self.embed_panel.run_button.clicked.connect(self._on_embed_clicked)
        self.extract_panel.run_button.clicked.connect(self._on_extract_clicked)
        self.options_panel.compatibility_mode.toggled.connect(self._on_compatibility_toggled)
        self.options_panel.enhanced_mode.toggled.connect(self._on_enhanced_toggled)
        self.options_panel.custom_file_mode.toggled.connect(self._on_custom_file_toggled)
        self.options_panel.rc1_mode.toggled.connect(self._on_rc1_toggled)

    def _on_embed_clicked(self) -> None:
        if self._has_running_task():
            return

        options = self._normalize_and_sync_options(show_notices=True)
        input_path = Path(self.embed_panel.input_image_edit.text().strip())
        if not input_path.exists():
            QMessageBox.warning(self, "路径错误", "输入图片不存在，请检查路径。")
            return

        output_dir = Path(str(options["output_dir"]).strip() or str(self.settings.default_output_dir))
        output_name = self.embed_panel.build_output_filename(input_path)
        output_path = output_dir / output_name
        password = str(options["password"])
        custom_output_name = self.embed_panel.output_name_edit.text().strip()

        if bool(options.get("compatibility_mode")) and not custom_output_name:
            output_path = output_dir / f"{input_path.stem}{(input_path.suffix or '.png')}"

        if bool(options.get("custom_file_mode")):
            payload_path = self.custom_file_panel.payload_path()
            if not payload_path.exists():
                QMessageBox.warning(self, "参数错误", "请选择有效的自定义文件后再执行嵌入。")
                return

            job = EmbedFileJob(
                input_image=input_path,
                output_dir=output_dir,
                payload_file=payload_path,
                password=password,
                output_basename=output_path.stem,
                use_rc1=bool(options.get("rc1_mode")),
                enhanced_mode=bool(options.get("enhanced_mode")),
            )
            mode_text = "RC1" if job.use_rc1 else "RAW"
            self._submit_task(
                f"正在嵌入自定义文件（{mode_text}），请稍候...",
                lambda: self.service.embed_file(job),
                self._handle_embed_result,
            )
            return

        watermark_text = self.embed_panel.watermark_text_edit.toPlainText().strip()
        if not watermark_text:
            QMessageBox.warning(self, "参数错误", "水印文本不能为空。")
            return

        job = EmbedTextJob(
            input_image=input_path,
            output_image=output_path,
            watermark_text=watermark_text,
            password=password,
            compatibility_mode=bool(options.get("compatibility_mode")),
            enhanced_mode=bool(options.get("enhanced_mode")),
        )
        if job.compatibility_mode:
            task_text = "正在按兼容模式（v1）嵌入文本水印，请稍候..."
        else:
            task_text = "正在嵌入文本水印，请稍候..."
        self._submit_task(task_text, lambda: self.service.embed_text(job), self._handle_embed_result)

    def _on_extract_clicked(self) -> None:
        if self._has_running_task():
            return

        options = self._normalize_and_sync_options(show_notices=True)
        input_path = Path(self.extract_panel.input_image_edit.text().strip())
        if not input_path.exists():
            QMessageBox.warning(self, "路径错误", "输入图片不存在，请检查路径。")
            return

        password = str(options["password"])
        output_dir = Path(str(options["output_dir"]).strip() or str(self.settings.default_output_dir))
        target_size = self.extract_panel.parse_target_size()
        compat_ws = self.extract_panel.parse_compat_ws()

        if bool(options.get("compatibility_mode")):
            job = ExtractTextJob(
                input_image=input_path,
                password=password,
                candidate_sizes=tuple(self.settings.extract_candidate_sizes),
                debug_output_dir=output_dir,
                compatibility_mode=True,
                expected_wm_length=compat_ws,
                target_size=target_size,
            )
            self._submit_task(
                "正在按兼容模式（v1）提取文本水印，请稍候...",
                lambda: self.service.extract_text(job),
                self._handle_extract_result,
            )
            return

        if self.extract_panel.is_file_mode():
            job = ExtractFileJob(
                input_image=input_path,
                password=password,
                output_dir=output_dir,
                use_rc1=bool(options.get("rc1_mode")),
                expected_wm_length=compat_ws,
                target_size=target_size,
            )
            mode_text = "RC1" if job.use_rc1 else "RAW"
            self._submit_task(
                f"正在提取文件水印（{mode_text}），请稍候...",
                lambda: self.service.extract_file(job),
                self._handle_extract_result,
            )
            return

        sizes = self.extract_panel.parse_sizes()
        if not sizes:
            QMessageBox.warning(self, "参数错误", "提取尺寸请输入数字列表，例如：256,128,96,64")
            return

        job = ExtractTextJob(
            input_image=input_path,
            password=password,
            candidate_sizes=sizes,
            debug_output_dir=output_dir,
            target_size=target_size,
        )
        self._submit_task(
            "正在提取文本水印，请稍候...",
            lambda: self.service.extract_text(job),
            self._handle_extract_result,
        )

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
        self._append_log(result.message)
        if result.success:
            QMessageBox.information(self, "嵌入完成", result.message)
        else:
            QMessageBox.warning(self, "嵌入失败", result.message)

    def _handle_extract_result(self, result: JobResult) -> None:
        if result.success:
            if result.extracted_text:
                self.extract_panel.set_result_text(result.extracted_text)
            elif result.output_path:
                self.extract_panel.set_result_text(f"输出文件：{result.output_path}")
            else:
                self.extract_panel.set_result_text("")
            self._append_log(result.message)
            QMessageBox.information(self, "提取完成", result.message)
            return

        self.extract_panel.set_result_text("")
        self._append_log(result.message)
        QMessageBox.warning(self, "提取失败", result.message)

    def _has_running_task(self) -> bool:
        if self._pending_future and not self._pending_future.done():
            QMessageBox.information(self, "提示", "已有任务在执行，请稍候。")
            return True
        return False

    def _on_compatibility_toggled(self, checked: bool) -> None:
        if checked:
            QMessageBox.warning(
                self,
                "兼容模式",
                "已启用 v0.1.3 兼容模式。\n仅支持文本水印，增强模式和文件模式将自动关闭。",
            )
        self._normalize_and_sync_options(show_notices=False)

    def _on_enhanced_toggled(self, checked: bool) -> None:
        if checked and not self.options_panel.compatibility_mode.isChecked():
            QMessageBox.information(
                self,
                "增强模式",
                "增强模式会轻微降低图像质量，但通常可提升抗干扰能力。",
            )
        self._normalize_and_sync_options(show_notices=False)

    def _on_custom_file_toggled(self, checked: bool) -> None:
        if checked and not self.options_panel.compatibility_mode.isChecked():
            QMessageBox.information(
                self,
                "自定义文件模式",
                "已启用自定义文件嵌入，支持文本/图片/二进制文件。",
            )
        self._normalize_and_sync_options(show_notices=False)

    def _on_rc1_toggled(self, _: bool) -> None:
        self._normalize_and_sync_options(show_notices=False)

    def _normalize_and_sync_options(self, show_notices: bool) -> dict[str, object]:
        raw_options = self.options_panel.values()
        guard = normalize_modes(raw_options)
        normalized = guard.options

        for key in ("compatibility_mode", "enhanced_mode", "custom_file_mode", "rc1_mode"):
            self.options_panel.set_mode_checked(key, bool(normalized.get(key)))

        self._refresh_mode_ui()

        if show_notices and guard.notices:
            QMessageBox.warning(self, "模式限制", "\n".join(guard.notices))

        return normalized

    def _refresh_mode_ui(self) -> None:
        values = self.options_panel.values()
        compatibility_mode = bool(values.get("compatibility_mode"))
        custom_file_mode = bool(values.get("custom_file_mode")) and not compatibility_mode

        if compatibility_mode:
            self.extract_panel.force_text_mode()

        self.custom_file_panel.set_visible_for_mode(custom_file_mode)
        self.embed_panel.set_text_mode_enabled(not custom_file_mode)

    def _append_log(self, text: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{timestamp}] {text}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.task_runner.shutdown()
        super().closeEvent(event)
