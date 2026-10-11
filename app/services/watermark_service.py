from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import tempfile
from typing import Iterable

from blind_watermark import WaterMark
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import qrcode

from app.models.job_models import (
    EmbedFileJob,
    EmbedTextJob,
    ExtractFileJob,
    ExtractTextJob,
    JobResult,
)


class WatermarkService:
    def embed_text(self, job: EmbedTextJob) -> JobResult:
        if job.compatibility_mode:
            return self._embed_text_v1(job)
        return self._embed_text_v2(job)

    def extract_text(self, job: ExtractTextJob) -> JobResult:
        if job.compatibility_mode:
            return self._extract_text_v1(job)
        return self._extract_text_v2(job)

    def embed_file(self, job: EmbedFileJob) -> JobResult:
        try:
            if not job.input_image.exists():
                return JobResult(False, "输入图片不存在，请检查路径。")
            if not job.payload_file.exists():
                return JobResult(False, "自定义文件不存在，请检查路径。")

            binary_data = job.payload_file.read_bytes()
            if not binary_data:
                return JobResult(False, "自定义文件为空，无法嵌入。")

            job.output_dir.mkdir(parents=True, exist_ok=True)
            password = self._parse_password(job.password)

            tmp_input = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            try:
                width, height = self._prepare_rgb_image(job.input_image, tmp_input)
                if job.enhanced_mode:
                    self._apply_enhanced_noise(tmp_input)

                available_capacity = self._estimate_watermark_capacity(tmp_input, password, width, height)

                if job.use_rc1:
                    from watermark.dataShielder import adapt_to_watermark_capacity_binary

                    bit_array = adapt_to_watermark_capacity_binary(
                        binary_data,
                        available_capacity,
                        safety_margin=0.90,
                    )
                    encoding_tag = "RC1"
                else:
                    bit_array = self._bytes_to_bits(binary_data)
                    if len(bit_array) > available_capacity:
                        return JobResult(
                            False,
                            f"原始文件过大，当前图像最多可容纳约 {available_capacity} 比特，请改用 RC1 或更大图片。",
                        )
                    encoding_tag = "RAW"

                bwm = WaterMark(password_wm=password, password_img=password)
                bwm.read_img(str(tmp_input))
                bwm.read_wm(bit_array, mode="bit")
                wm_len = len(bit_array)
                output_path = self._build_file_output_path(
                    output_dir=job.output_dir,
                    base_name=(job.output_basename or job.input_image.stem),
                    source_suffix=(job.input_image.suffix or ".png"),
                    encoding_tag=encoding_tag,
                    wm_len=wm_len,
                    width=width,
                    height=height,
                )
                bwm.embed(str(output_path))
            finally:
                self._safe_unlink(tmp_input)

            return JobResult(
                success=True,
                message=(
                    f"嵌入完成（{encoding_tag}）\n"
                    f"输出文件：{output_path}\n"
                    f"原始文件大小：{len(binary_data)} 字节\n"
                    f"嵌入长度：{wm_len} 比特"
                ),
                output_path=output_path,
                debug_details={"encoding": encoding_tag, "wm_len": wm_len},
            )
        except Exception as exc:
            return JobResult(False, f"文件嵌入失败：{exc}")

    def extract_file(self, job: ExtractFileJob) -> JobResult:
        try:
            if not job.input_image.exists():
                return JobResult(False, "输入图片不存在，请检查路径。")

            job.output_dir.mkdir(parents=True, exist_ok=True)
            password = self._parse_password(job.password)

            wm_len = job.expected_wm_length or self._parse_ws_from_filename(job.input_image.name)
            if wm_len is None:
                return JobResult(False, "未提供水印长度，且文件名中未找到 ws 信息（例如 ws4096）。")

            target_size = job.target_size or self._parse_size_from_filename(job.input_image.name)

            tmp_input = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            try:
                self._prepare_rgb_image(job.input_image, tmp_input, target_size=target_size)
                bwm = WaterMark(password_wm=password, password_img=password)
                wm_bits = bwm.extract(filename=str(tmp_input), wm_shape=wm_len, mode="bit")
                bit_array = [bool(value) for value in wm_bits]
            finally:
                self._safe_unlink(tmp_input)

            if job.use_rc1:
                from watermark.dataShielder import decode_watermark_to_binary

                binary_data, stats = decode_watermark_to_binary(bit_array)
                if binary_data is None:
                    binary_data, stats = decode_watermark_to_binary(bit_array, interleave_depth=1)
                if binary_data is None:
                    return JobResult(
                        False,
                        (
                            "RC1 解码失败，未能恢复有效数据。\n"
                            f"找到数据包：{stats.get('total_packets_found', 0)}\n"
                            f"有效数据包：{stats.get('valid_packets', 0)}"
                        ),
                    )
                encoding_tag = "RC1"
                details = {
                    "total_packets_found": stats.get("total_packets_found", 0),
                    "valid_packets": stats.get("valid_packets", 0),
                    "errors_corrected": stats.get("total_errors_corrected", 0),
                }
            else:
                binary_data = self._bits_to_bytes(bit_array)
                encoding_tag = "RAW"
                details = {}

            file_ext = self._detect_file_type(binary_data)
            output_path = job.output_dir / f"{job.input_image.stem}-Extracted-{encoding_tag}{file_ext}"
            output_path.write_bytes(binary_data)

            message = (
                f"提取完成（{encoding_tag}）\n"
                f"输出文件：{output_path}\n"
                f"提取数据大小：{len(binary_data)} 字节\n"
                f"检测类型：{file_ext}"
            )
            if details:
                message = (
                    f"{message}\n"
                    f"有效数据包：{details['valid_packets']}/{details['total_packets_found']}\n"
                    f"纠错字节：{details['errors_corrected']}"
                )

            return JobResult(
                success=True,
                message=message,
                output_path=output_path,
                debug_details=details,
            )
        except Exception as exc:
            return JobResult(False, f"文件提取失败：{exc}")

    def _embed_text_v2(self, job: EmbedTextJob) -> JobResult:
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
                if job.enhanced_mode:
                    self._apply_enhanced_noise(tmp_input)

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

            mode_text = "增强模式" if job.enhanced_mode else "默认模式"
            return JobResult(
                success=True,
                message=f"嵌入完成（二维码尺寸 {used_size}，{mode_text}），输出文件：{job.output_image}",
                output_path=job.output_image,
            )
        except Exception as exc:
            return JobResult(False, f"嵌入失败：{exc}")

    def _extract_text_v2(self, job: ExtractTextJob) -> JobResult:
        try:
            if not job.input_image.exists():
                return JobResult(False, "输入图片不存在，请检查路径。")
            password = self._parse_password(job.password)
            sizes = self._sanitize_sizes(job.candidate_sizes)
            if not sizes:
                return JobResult(False, "提取尺寸列表无效，请输入数字尺寸。")

            target_size = job.target_size or self._parse_size_from_filename(job.input_image.name)

            tmp_qr = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            tmp_input = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            last_error = ""
            try:
                self._prepare_rgb_image(job.input_image, tmp_input, target_size=target_size)
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

    def _embed_text_v1(self, job: EmbedTextJob) -> JobResult:
        try:
            if not job.input_image.exists():
                return JobResult(False, "输入图片不存在，请检查路径。")
            if not job.watermark_text.strip():
                return JobResult(False, "水印文本不能为空。")

            password = self._parse_password(job.password)
            output_dir = job.output_image.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            tmp_input = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            try:
                width, height = self._prepare_rgb_image(job.input_image, tmp_input)
                bwm = WaterMark(password_wm=password, password_img=password)
                bwm.read_img(str(tmp_input))
                bwm.read_wm(job.watermark_text, mode="str")
                wm_len = len(bwm.wm_bit)
                output_path = output_dir / (
                    f"{job.output_image.stem}-Watermark-ws{wm_len}-size{width}x{height}"
                    f"{(job.output_image.suffix or job.input_image.suffix or '.png')}"
                )
                bwm.embed(str(output_path))
            finally:
                self._safe_unlink(tmp_input)

            return JobResult(
                success=True,
                message=(
                    "兼容模式（v1）嵌入完成。\n"
                    f"输出文件：{output_path}\n"
                    f"水印长度：{wm_len} 尺寸：{width}x{height}"
                ),
                output_path=output_path,
            )
        except Exception as exc:
            return JobResult(False, f"兼容模式嵌入失败：{exc}")

    def _extract_text_v1(self, job: ExtractTextJob) -> JobResult:
        try:
            if not job.input_image.exists():
                return JobResult(False, "输入图片不存在，请检查路径。")
            password = self._parse_password(job.password)

            wm_len = job.expected_wm_length or self._parse_ws_from_filename(job.input_image.name)
            if wm_len is None:
                return JobResult(False, "兼容模式提取需要水印长度（ws），请在输入框填写或使用含 ws 的文件名。")

            target_size = job.target_size or self._parse_size_from_filename(job.input_image.name)
            if target_size is None:
                with Image.open(job.input_image) as image:
                    target_size = image.size

            tmp_input = Path(tempfile.NamedTemporaryFile(suffix=".png", delete=False).name)
            try:
                self._prepare_rgb_image(job.input_image, tmp_input, target_size=target_size)
                bwm = WaterMark(password_wm=password, password_img=password)
                wm_extract = bwm.extract(str(tmp_input), wm_shape=wm_len, mode="str")
                wm_extract = wm_extract.replace("\\n", "\n")
            finally:
                self._safe_unlink(tmp_input)

            return JobResult(
                success=True,
                message=f"兼容模式（v1）提取成功，长度：{wm_len}。",
                extracted_text=wm_extract,
                debug_details={"wm_len": wm_len},
            )
        except Exception as exc:
            return JobResult(False, f"兼容模式提取失败：{exc}")

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
            if isinstance(size, int) and size > 0 and size not in result:
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
    def _prepare_rgb_image(
        input_file: Path,
        output_file: Path,
        target_size: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        with Image.open(input_file) as image:
            if target_size and image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            if image.mode != "RGB":
                image = image.convert("RGB")
            width, height = image.size
            image.save(output_file, format="PNG")
            return width, height

    @staticmethod
    def _apply_enhanced_noise(image_file: Path) -> None:
        with Image.open(image_file) as image:
            array = np.array(image.convert("RGB"), dtype=np.float32)
        noise = np.random.normal(loc=0.0, scale=8.0, size=array.shape)
        enhanced = np.clip(array + noise, 0, 255).astype(np.uint8)
        Image.fromarray(enhanced, mode="RGB").save(image_file, format="PNG")

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
    def _parse_ws_from_filename(filename: str) -> int | None:
        match = re.search(r"ws(\d+)", filename, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _parse_size_from_filename(filename: str) -> tuple[int, int] | None:
        match = re.search(r"size(\d+)x(\d+)", filename, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1)), int(match.group(2))
        except ValueError:
            return None

    @staticmethod
    def _estimate_watermark_capacity(
        image_file: Path,
        password: int,
        width: int,
        height: int,
    ) -> int:
        try:
            bwm = WaterMark(password_wm=password, password_img=password)
            bwm.read_img(str(image_file))
            bwm.bwm_core.init_block_index()
            capacity = int(bwm.bwm_core.block_num)
            if capacity > 0:
                return capacity
        except Exception:
            pass
        return max(int(width * height * 0.25), 1024)

    @staticmethod
    def _build_file_output_path(
        output_dir: Path,
        base_name: str,
        source_suffix: str,
        encoding_tag: str,
        wm_len: int,
        width: int,
        height: int,
    ) -> Path:
        suffix = source_suffix or ".png"
        safe_base = base_name.strip() or "output"
        filename = f"{safe_base}-Watermark-{encoding_tag}-ws{wm_len}-size{width}x{height}{suffix}"
        return output_dir / filename

    @staticmethod
    def _bytes_to_bits(data: bytes) -> list[bool]:
        bits: list[bool] = []
        for byte_value in data:
            for bit in range(8):
                bits.append(bool((byte_value >> (7 - bit)) & 1))
        return bits

    @staticmethod
    def _bits_to_bytes(bits: Iterable[bool]) -> bytes:
        bit_list = list(bits)
        result = bytearray()
        for index in range(0, len(bit_list), 8):
            chunk = bit_list[index : index + 8]
            if len(chunk) < 8:
                break
            byte_value = 0
            for bit_index, bit in enumerate(chunk):
                if bit:
                    byte_value |= 1 << (7 - bit_index)
            result.append(byte_value)
        return bytes(result)

    @staticmethod
    def _detect_file_type(data: bytes) -> str:
        if len(data) < 4:
            return ".bin"

        signatures = {
            b"\xFF\xD8\xFF": ".jpg",
            b"\x89PNG": ".png",
            b"GIF87a": ".gif",
            b"GIF89a": ".gif",
            b"%PDF": ".pdf",
            b"PK\x03\x04": ".zip",
            b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1": ".doc",
            b"MZ": ".exe",
            b"\x7FELF": ".elf",
            b"\x1F\x8B": ".gz",
            b"\x37\x7A\xBC\xAF\x27\x1C": ".7z",
            b"Rar!": ".rar",
            b"ID3": ".mp3",
            b"RIFF": ".wav",
            b"OggS": ".ogg",
        }
        for magic, ext in signatures.items():
            if data.startswith(magic):
                return ext

        try:
            probe = data[:1000].decode("utf-8")
            if probe:
                printable = sum(
                    1 for char in probe[:100] if char.isprintable() or char in "\r\n\t"
                )
                if printable >= max(1, int(len(probe[:100]) * 0.9)):
                    return ".txt"
        except Exception:
            pass

        return ".bin"

    @staticmethod
    def _safe_unlink(file_path: Path) -> None:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception:
            pass
