from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile
from typing import Iterable

from blind_watermark import WaterMark
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import qrcode

from app.models.job_models import EmbedTextJob, ExtractTextJob, JobResult


class WatermarkService:
    def __init__(self) -> None:
        pass

    def embed_text(self, job: EmbedTextJob) -> JobResult:
        try:
            if not job.input_image.exists():
                return JobResult(False, "输入图片不存在，请检查路径。")
            if not job.watermark_text.strip():
                return JobResult(False, "水印文本不能为空。")

            output_dir = job.output_image.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            password = self._parse_password(job.password)

            tmp_qr = Path(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
            tmp_input = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            try:
                self._prepare_rgb_image(job.input_image, tmp_input)
                candidate_sizes = [job.qr_size, 96, 64, 48, 32]
                unique_sizes: list[int] = []
                for size in candidate_sizes:
                    if size > 0 and size not in unique_sizes:
                        unique_sizes.append(size)

                last_error = ""
                used_size = None
                for current_size in unique_sizes:
                    try:
                        self._create_qr_image(job.watermark_text, tmp_qr, current_size)
                        bwm = WaterMark(password_wm=password, password_img=password)
                        bwm.read_img(str(tmp_input))
                        bwm.read_wm(str(tmp_qr))
                        bwm.embed(str(job.output_image))
                        used_size = current_size
                        break
                    except Exception as inner_exc:
                        last_error = str(inner_exc)

                if used_size is None:
                    raise RuntimeError(last_error or "嵌入失败，未找到可用二维码尺寸。")
            finally:
                self._safe_unlink(tmp_input)
                self._safe_unlink(tmp_qr)

            return JobResult(
                success=True,
                message=f"嵌入完成（二维码尺寸 {used_size}），输出文件：{job.output_image}",
                output_path=job.output_image,
            )
        except Exception as exc:
            return JobResult(False, f"嵌入失败：{exc}")

    def extract_text(self, job: ExtractTextJob) -> JobResult:
        try:
            if not job.input_image.exists():
                return JobResult(False, "输入图片不存在，请检查路径。")
            password = self._parse_password(job.password)
            sizes = self._sanitize_sizes(job.candidate_sizes)
            if not sizes:
                return JobResult(False, "提取尺寸列表无效，请输入数字尺寸。")

            tmp_qr = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            tmp_input = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            last_error = ""
            try:
                self._prepare_rgb_image(job.input_image, tmp_input)
                for size in sizes:
                    try:
                        bwm = WaterMark(password_wm=password, password_img=password)
                        bwm.extract(
                            filename=str(tmp_input),
                            wm_shape=(size, size),
                            out_wm_name=str(tmp_qr),
                        )
                        decoded_text = self._decode_qr(tmp_qr)
                        if decoded_text:
                            preview_path = None
                            if job.debug_output_dir:
                                job.debug_output_dir.mkdir(parents=True, exist_ok=True)
                                preview_path = job.debug_output_dir / (
                                    f"{job.input_image.stem}-qt-extracted-preview.png"
                                )
                                shutil.copyfile(tmp_qr, preview_path)
                            return JobResult(
                                success=True,
                                message=f"提取成功（尺寸 {size}x{size}）。",
                                extracted_text=decoded_text,
                                output_path=preview_path,
                                debug_details={"size": f"{size}x{size}"},
                            )
                    except Exception as inner_exc:
                        last_error = str(inner_exc)

                debug_path = None
                if job.debug_output_dir:
                    job.debug_output_dir.mkdir(parents=True, exist_ok=True)
                    debug_path = job.debug_output_dir / (
                        f"{job.input_image.stem}-qt-extracted-preview.png"
                    )
                    shutil.copyfile(tmp_qr, debug_path)

                message = "未能识别出二维码文本。"
                if last_error:
                    message = f"{message} 最后一次错误：{last_error}"
                if debug_path:
                    message = f"{message} 已导出提取预览：{debug_path}"
                return JobResult(False, message, output_path=debug_path)
            finally:
                self._safe_unlink(tmp_input)
                self._safe_unlink(tmp_qr)
        except Exception as exc:
            return JobResult(False, f"提取失败：{exc}")

    @staticmethod
    def _parse_password(password: str) -> int:
        value = (password or "").strip()
        if not value or not value.isdigit():
            raise ValueError("密码必须是数字。")
        return int(value)

    @staticmethod
    def _sanitize_sizes(sizes: Iterable[int]) -> tuple[int, ...]:
        result: list[int] = []
        for size in sizes:
            if isinstance(size, int) and size > 0:
                result.append(size)
        return tuple(result)

    @staticmethod
    def _create_qr_image(text: str, output_file: Path, size: int) -> None:
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=1,
        )
        qr.add_data(text)
        qr.make(fit=True)
        image = qr.make_image(fill_color="white", back_color="black").convert("RGB")
        image = image.resize((size, size), Image.Resampling.LANCZOS)
        image.save(output_file, format="JPEG", quality=100)

    @staticmethod
    def _prepare_rgb_image(input_file: Path, output_file: Path) -> None:
        image = Image.open(input_file)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(output_file, format="PNG")

    def _decode_qr(self, qr_image_path: Path) -> str | None:
        image = Image.open(qr_image_path).convert("RGB")
        variants = self._build_decode_variants(image)
        for variant in variants:
            text = self._decode_qr_with_pyzbar(variant)
            if text:
                return text
            text = self._decode_qr_with_opencv(variant)
            if text:
                return text
        return None

    @staticmethod
    def _build_decode_variants(image: Image.Image) -> list[Image.Image]:
        gray = ImageOps.grayscale(image)
        return [
            image.convert("RGB"),
            gray.convert("RGB"),
            ImageOps.invert(gray).convert("RGB"),
            ImageEnhance.Contrast(image).enhance(2.0).convert("RGB"),
            gray.point(lambda p: 255 if p > 128 else 0).convert("RGB"),
        ]

    @staticmethod
    def _decode_qr_with_pyzbar(image: Image.Image) -> str | None:
        try:
            from pyzbar.pyzbar import decode

            result = decode(image)
            if result:
                return result[0].data.decode("utf-8", errors="ignore")
        except Exception:
            return None
        return None

    @staticmethod
    def _decode_qr_with_opencv(image: Image.Image) -> str | None:
        try:
            image_array = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            detector = cv2.QRCodeDetector()
            result_text, _, _ = detector.detectAndDecode(image_array)
            if result_text:
                return result_text
        except Exception:
            return None
        return None

    @staticmethod
    def _safe_unlink(file_path: Path) -> None:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass
