from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile


@dataclass(frozen=True)
class AppSettings:
    default_password: str
    default_output_dir: Path
    default_wm_text: str
    extract_candidate_sizes: tuple[int, ...]


def load_settings() -> AppSettings:
    default_output_dir = Path(
        os.getenv("BW_GUI_OUTPUT_DIR", tempfile.gettempdir())
    ).expanduser()
    default_password = os.getenv("BW_GUI_DEFAULT_PWD", "123456")

    raw_sizes = os.getenv("BW_GUI_EXTRACT_SIZES", "256,128,96,64,48,32")
    parsed_sizes: list[int] = []
    for chunk in raw_sizes.split(","):
        value = chunk.strip()
        if value.isdigit():
            parsed_sizes.append(int(value))
    if not parsed_sizes:
        parsed_sizes = [256, 128, 96, 64, 48, 32]

    return AppSettings(
        default_password=default_password,
        default_output_dir=default_output_dir,
        default_wm_text="Copyright@\nAuthor@",
        extract_candidate_sizes=tuple(parsed_sizes),
    )
