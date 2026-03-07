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


@dataclass(slots=True)
class ExtractTextJob:
    input_image: Path
    password: str
    candidate_sizes: tuple[int, ...]
    debug_output_dir: Path | None = None


@dataclass(slots=True)
class JobResult:
    success: bool
    message: str
    output_path: Path | None = None
    extracted_text: str | None = None
    debug_details: dict[str, Any] = field(default_factory=dict)

