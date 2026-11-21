"""确定性裁决（鲁棒版）。

证据等级按支持带数划分。主链的支持带内须有基频及其他合格倍频，且至少一带通过同带拟合。
支持带数达到相应门槛且基频不与其他族碰撞时，记为 strong 或 moderate，其余为 none。

裁决：唯一 strong/moderate → 该故障；多个 → unknown（跨族竞争）；无 → normal。
置信度是确定性档位，不做加分式堆叠。
"""

from __future__ import annotations

import math
from typing import Any, Mapping

FAMILY_TO_FAULT = {
    "BPFO": ("outer_race_fault", "外圈故障"),
    "BPFI": ("inner_race_fault", "内圈故障"),
    "BSF": ("rolling_element_fault", "滚动体故障"),
}


def fundamental_collisions(
    frequencies_hz: Mapping[str, float],
    tolerance_hz_map: Mapping[str, float],
) -> set[str]:
    """返回基频与其它族基频发生碰撞（过于接近）的故障族集合。"""

    families = list(frequencies_hz.keys())
    colliding: set[str] = set()
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            a, b = families[i], families[j]
            if abs(frequencies_hz[a] - frequencies_hz[b]) <= tolerance_hz_map[a] + tolerance_hz_map[b]:
                colliding.add(a)
                colliding.add(b)
    return colliding


def assess_families(
    fundamental_bands: Mapping[str, int],
    colliding_families: set[str],
    *,
    strong_min_bands: int = 3,
    moderate_min_bands: int = 2,
) -> dict[str, str]:
    """由跨频带一致性的带数评定证据等级。

    ``fundamental_bands`` 接收各族的支持带数。主链另行记录原始基频命中带数。
    """

    levels: dict[str, str] = {}
    for family, bands in fundamental_bands.items():
        specific = family not in colliding_families
        if bands >= strong_min_bands and specific:
            levels[family] = "strong"
        elif bands >= moderate_min_bands and specific:
            levels[family] = "moderate"
        else:
            levels[family] = "none"
    return levels


def _dominant_family(
    candidates: list[str],
    fundamental_bands: Mapping[str, int],
    harmonic_counts: Mapping[str, int],
    min_margin: int,
    residuals: Mapping[str, float] | None = None,
) -> str | None:
    """从多个同级候选族里选出主导的那个；无法区分时返回 None。

    主导性先比「跨带一致带数 → 谐波数」。带数不足以区分（领先 < min_margin）时，
    改用**判据自身的标尺**：各族命中带里**最差**的拟合残差（单位 bin，越小越好）。
    残差更小者意味着它在同样数量的频带上都更好地解释了谱列；残差相等才判 unknown。

    这样不需要为"主导"另设经验边距——比较用的量已经是同一套阈值下的校准量。
    """

    scored = sorted(
        candidates,
        key=lambda f: (fundamental_bands.get(f, 0), harmonic_counts.get(f, 0)),
        reverse=True,
    )
    top = scored[0]
    runner = scored[1] if len(scored) > 1 else None
    if runner is None:
        return top
    band_margin = fundamental_bands.get(top, 0) - fundamental_bands.get(runner, 0)
    if band_margin >= min_margin:
        return top
    if not residuals:
        return None
    # 带数不足以区分时，在**同级候选中按拟合残差重排**：残差更小者意味着它在
    # 同样数量的频带上更彻底地解释了谱列。残差相等则仍判 unknown。
    by_residual = sorted(scored, key=lambda f: float(residuals.get(f, math.inf)))
    best, second = by_residual[0], by_residual[1]
    best_residual = float(residuals.get(best, math.inf))
    second_residual = float(residuals.get(second, math.inf))
    return best if best_residual < second_residual else None


def _fault_verdict(family: str, confidence: float, reason: str) -> dict[str, Any]:
    fault_type, fault_zh = FAMILY_TO_FAULT[family]
    return {
        "fault_type": fault_type,
        "fault_type_zh": fault_zh,
        "leading_family": family,
        "confidence": float(confidence),
        "status": "diagnosed",
        "reason": reason,
    }


def _unknown_verdict(reason: str) -> dict[str, Any]:
    return {
        "fault_type": "unknown",
        "fault_type_zh": "未知",
        "leading_family": "uncertain",
        "confidence": 0.0,
        "status": "unknown",
        "reason": reason,
    }


def decide(
    levels: Mapping[str, str],
    *,
    fundamental_bands: Mapping[str, int] | None = None,
    harmonic_counts: Mapping[str, int] | None = None,
    residuals: Mapping[str, float] | None = None,
    confidence: Mapping[str, float] | None = None,
    dominance_margin: int = 2,
) -> dict[str, Any]:
    """由各族证据等级给出结论：具体故障 / normal / unknown。

    多个族同等级时先比跨带一致带数；领先不足 ``dominance_margin`` 时改用各族
    最差拟合残差（bin）比较（见 ``_dominant_family``）。两者都无法区分才 unknown。
    """

    conf = dict(confidence or {})
    fb = dict(fundamental_bands or {})
    hc = dict(harmonic_counts or {})
    rs = dict(residuals or {})
    strongs = [f for f, level in levels.items() if level == "strong"]
    moderates = [f for f, level in levels.items() if level == "moderate"]

    if len(strongs) == 1:
        return _fault_verdict(
            strongs[0], conf.get("strong_unique", 0.85),
            f"唯一强证据指向 {strongs[0]}（同一谱列在多个谐振频带一致出现）",
        )
    if len(strongs) >= 2:
        top = _dominant_family(strongs, fb, hc, dominance_margin, rs)
        if top is not None:
            return _fault_verdict(
                top, conf.get("strong_dominant", 0.8),
                f"多族强证据中 {top} 主导（跨带一致带数或拟合残差更优）",
            )
        return _unknown_verdict("多个故障族同时达到强证据且无明显主导，无法唯一判定")
    if len(moderates) == 1:
        return _fault_verdict(
            moderates[0], conf.get("moderate_unique", 0.65),
            f"唯一中等证据指向 {moderates[0]}（多个谐振频带内有基频及其他合格倍频）",
        )
    if len(moderates) >= 2:
        top = _dominant_family(moderates, fb, hc, dominance_margin, rs)
        if top is not None:
            return _fault_verdict(
                top, conf.get("moderate_dominant", 0.6),
                f"多族中等证据中 {top} 主导（跨带一致带数或拟合残差更优）",
            )
        return _unknown_verdict("多个故障族达到中等证据且无明显主导，无法唯一判定")

    return {
        "fault_type": "normal",
        "fault_type_zh": "正常",
        "leading_family": "NONE",
        "confidence": float(conf.get("normal", 0.75)),
        "status": "normal",
        "reason": "成组故障倍频与同带拟合证据未达到数值故障判据",
    }
