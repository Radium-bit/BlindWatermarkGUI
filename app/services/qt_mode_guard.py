from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GuardResult:
    options: dict[str, object]
    notices: list[str] = field(default_factory=list)


def normalize_modes(options: dict[str, object]) -> GuardResult:
    normalized = dict(options)
    notices: list[str] = []

    compatibility_mode = bool(normalized.get("compatibility_mode"))
    enhanced_mode = bool(normalized.get("enhanced_mode"))
    custom_file_mode = bool(normalized.get("custom_file_mode"))
    rc1_mode = bool(normalized.get("rc1_mode"))

    if compatibility_mode:
        if enhanced_mode:
            normalized["enhanced_mode"] = False
            notices.append("v0.1.3兼容模式下无法增强处理，已自动关闭增强模式。")
        if custom_file_mode:
            normalized["custom_file_mode"] = False
            notices.append("v0.1.3兼容模式下无法嵌入自定义文件，已自动关闭该选项。")
        if rc1_mode:
            normalized["rc1_mode"] = False
            notices.append("v0.1.3兼容模式下不支持 RC1 文件路径，已自动关闭 RC1。")

    return GuardResult(options=normalized, notices=notices)
