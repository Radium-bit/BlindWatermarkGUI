from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EmbedTextJob:
    input_image: Path
    output_image: Path
    watermark_text: str
    password: str
    qr_size: int = 128
    compatibility_mode: bool = False
    enhanced_mode: bool = False


@dataclass(slots=True)
class ExtractTextJob:
    input_image: Path
    password: str
    candidate_sizes: tuple[int, ...]
    debug_output_dir: Path | None = None
    compatibility_mode: bool = False
    expected_wm_length: int | None = None
    target_size: tuple[int, int] | None = None


@dataclass(slots=True)
class EmbedFileJob:
    input_image: Path
    output_dir: Path
    payload_file: Path
    password: str
    output_basename: str | None = None
    use_rc1: bool = True
    enhanced_mode: bool = False


@dataclass(slots=True)
class ExtractFileJob:
    input_image: Path
    password: str
    output_dir: Path
    use_rc1: bool = True
    expected_wm_length: int | None = None
    target_size: tuple[int, int] | None = None


@dataclass(slots=True)
class JobResult:
    success: bool
    message: str
    output_path: Path | None = None
    extracted_text: str | None = None
    debug_details: dict[str, Any] = field(default_factory=dict)
