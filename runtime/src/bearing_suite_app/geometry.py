"""轴承几何 → 故障特征频率（通用，不绑定具体数据集）。

支持两种输入方式：
1. ``feature_frequency_ratios``：直接给出 BPFI/BPFO/BSF/FTF 相对转频的阶次比。
2. 物理几何：滚动体直径 d、节圆直径 D、滚动体数量 n、接触角 α，按经典公式计算。

FTF（保持架频率）是滚动体故障 BSF 边带检测所需的调制频率；若未显式给出，
由 BPFI/BPFO 阶次比推导：c=(BPFI-BPFO)/(BPFI+BPFO)，FTF=(1-c)/2。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

FAMILY_ORDER = ("BPFO", "BPFI", "BSF")


def _pick(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) not in (None, ""):
            return mapping.get(key)
    return None


def fault_frequency_ratios_from_geometry(
    rolling_element_diameter: float,
    pitch_diameter: float,
    rolling_element_count: int,
    contact_angle_degrees: float = 0.0,
) -> dict[str, float]:
    """经典轴承故障特征频率阶次比（相对转频）。"""

    d = float(rolling_element_diameter)
    diameter = float(pitch_diameter)
    count = int(rolling_element_count)
    angle = float(contact_angle_degrees)
    if not all(math.isfinite(item) for item in (d, diameter, angle)):
        raise ValueError("轴承几何必须为有限数值")
    if d <= 0 or diameter <= 0 or d >= diameter:
        raise ValueError("要求 0 < 滚动体直径 < 节圆直径")
    if count < 2:
        raise ValueError("滚动体数量必须 >= 2")

    cosine_term = d / diameter * math.cos(math.radians(angle))
    contact_factor = 1.0 - cosine_term * cosine_term
    return {
        "BPFI": float(count / 2.0 * (1.0 + cosine_term)),
        "BPFO": float(count / 2.0 * (1.0 - cosine_term)),
        "BSF": float(diameter / (2.0 * d) * contact_factor),
        "FTF": float(0.5 * (1.0 - cosine_term)),
    }


def ftf_ratio_from(bpfi_ratio: float, bpfo_ratio: float) -> float:
    """由 BPFI/BPFO 阶次比推导 FTF 阶次比。"""

    bpfi = float(bpfi_ratio)
    bpfo = float(bpfo_ratio)
    denominator = bpfi + bpfo
    if denominator <= 0:
        return 0.0
    contact = (bpfi - bpfo) / denominator
    return float((1.0 - contact) / 2.0)


def fault_frequency_ratios(geometry: Mapping[str, Any] | None) -> dict[str, float]:
    """把任意“几何描述”归一化为故障频率阶次比字典（含 FTF）。"""

    if not isinstance(geometry, Mapping):
        raise ValueError("必须提供 bearing_geometry（feature_frequency_ratios 或物理几何）")

    ratios: dict[str, float] = {}

    explicit = geometry.get("feature_frequency_ratios")
    if isinstance(explicit, Mapping):
        for family in ("BPFI", "BPFO", "BSF", "FTF"):
            value = _pick(explicit, family, family.lower())
            if value is not None:
                try:
                    ratio = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{family} 阶次比必须为正有限数值") from exc
                if not math.isfinite(ratio) or ratio <= 0:
                    raise ValueError(f"{family} 阶次比必须为正有限数值")
                ratios[family] = ratio
        if "BPFI" in ratios and "BPFO" in ratios and "BSF" in ratios:
            if "FTF" not in ratios:
                ratios["FTF"] = ftf_ratio_from(ratios["BPFI"], ratios["BPFO"])
            return ratios

    d = _pick(geometry, "rolling_element_diameter", "ball_diameter", "roller_diameter", "d")
    diameter = _pick(geometry, "pitch_diameter", "bearing_pitch_diameter", "D")
    count = _pick(geometry, "rolling_element_count", "ball_count", "roller_count", "number_of_rolling_elements", "n")
    alpha = _pick(geometry, "contact_angle_degrees", "contact_angle", "alpha")
    if d is None or diameter is None or count is None:
        raise ValueError(
            "geometry 必须提供 feature_frequency_ratios "
            "或 (rolling_element_diameter, pitch_diameter, rolling_element_count)"
        )
    return fault_frequency_ratios_from_geometry(
        float(d), float(diameter), int(count), 0.0 if alpha is None else float(alpha)
    )


def fault_frequencies_hz(rpm: float, ratios: Mapping[str, float]) -> dict[str, float]:
    """由转速(rpm)和阶次比计算各故障特征频率(Hz)。"""

    speed_hz = float(rpm) / 60.0
    if not math.isfinite(speed_hz) or speed_hz <= 0:
        raise ValueError("rpm 必须为正有限数值")
    result = {}
    for family, ratio in ratios.items():
        value = float(ratio)
        frequency = speed_hz * value
        if not math.isfinite(value) or value <= 0 or not math.isfinite(frequency):
            raise ValueError(f"{family} 阶次比及特征频率必须为正有限数值")
        result[family] = frequency
    return result
