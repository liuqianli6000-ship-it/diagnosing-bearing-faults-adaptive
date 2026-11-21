"""Route a diagnosis request to stable or adaptive mode.

Default is stable. Explicit adaptive wording enables the adaptive planner.
"""

from __future__ import annotations

ADAPTIVE_KEYWORDS = (
    "精细诊断",
    "自适应诊断",
    "自适应",
    "精细",
    "深度诊断",
    "详细诊断",
    "精确诊断",
    "迭代诊断",
    "多轮诊断",
)


def is_adaptive_request(text: str | None) -> bool:
    if not text:
        return False
    normalized = "".join(str(text).split()).lower()
    return any(keyword.lower() in normalized for keyword in ADAPTIVE_KEYWORDS)


def resolve_mode(request_text: str | None, adaptive: bool | None) -> bool:
    """Explicit ``adaptive`` wins; otherwise infer from request text."""

    if adaptive is not None:
        return bool(adaptive)
    return is_adaptive_request(request_text)
