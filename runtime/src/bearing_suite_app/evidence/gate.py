"""Local evidence gate.

The LLM may suggest stopping, but only this gate can declare evidence
"clear".  This keeps a hallucinated "I am confident" from ending a diagnosis
before the measured evidence supports it.
"""

from __future__ import annotations

from typing import Any, Mapping

from bearing_suite_app.diagnosis_schemas import EvidenceBundle


def _level(assessments: Mapping[str, Any], family: str) -> str:
    item = assessments.get(family)
    if not isinstance(item, Mapping):
        return "none"
    return str(item.get("evidence_level") or "none")


_LEVEL_SCORE = {"strong": 3, "moderate": 2, "weak": 1, "none": 0}


def evidence_rank(
    evidence: EvidenceBundle,
    target_family: str | None = None,
) -> dict[str, Any]:
    """Rank evidence for one target family against the runner-up.

    The rank is lexicographic: level first, then band count, then level
    margin.  This is the signal used to decide whether an L3-enhanced signal
    actually improved the diagnosis, not just the absolute envelope energy.
    """

    assessments = evidence.assessments or {}
    levels = {family: _level(assessments, family) for family in ("BPFO", "BPFI", "BSF")}
    bands = {family: _bands(assessments, family) for family in levels}
    if target_family not in levels:
        target_family = max(levels, key=lambda family: (_LEVEL_SCORE[levels[family]], bands[family]))
    others = [family for family in levels if family != target_family]
    runner = max(others, key=lambda family: (_LEVEL_SCORE[levels[family]], bands[family])) if others else None
    level_score = _LEVEL_SCORE.get(levels.get(target_family, "none"), 0)
    runner_level = _LEVEL_SCORE.get(levels.get(runner, "none"), 0) if runner else 0
    target_bands = bands.get(target_family, 0)
    runner_bands = bands.get(runner, 0) if runner else 0
    return {
        "target_family": target_family,
        "runner_up": runner,
        "level": levels.get(target_family, "none"),
        "level_score": level_score,
        "bands": target_bands,
        "runner_level": levels.get(runner, "none") if runner else "none",
        "runner_bands": runner_bands,
        "level_margin": level_score - runner_level,
        "band_margin": target_bands - runner_bands,
        "key": (level_score, target_bands, level_score - runner_level, target_bands - runner_bands),
    }



def _bands(assessments: Mapping[str, Any], family: str) -> int:
    item = assessments.get(family)
    if not isinstance(item, Mapping):
        return 0
    try:
        return int(item.get("evidence_bands") or 0)
    except (TypeError, ValueError):
        return 0


def evaluate(
    evidence: EvidenceBundle,
    baseline: Any,
) -> dict[str, Any]:
    """Return verdict, target family/frequency and a human-readable reason."""

    if not evidence.analyzable:
        return {
            "verdict": "limited",
            "reason": evidence.limitation_reason or "证据不可诊断",
            "target_family": None,
            "target_frequency_hz": None,
            "levels": {},
            "bands": {},
        }

    assessments = evidence.assessments or {}
    levels = {family: _level(assessments, family) for family in ("BPFO", "BPFI", "BSF")}
    bands = {family: _bands(assessments, family) for family in levels}
    status = str(getattr(baseline, "status", "unknown"))
    if status == "normal" and not any(level == "strong" for level in levels.values()):
        return {
            "verdict": "clear",
            "reason": "确定性基线为正常，且没有强故障族证据",
            "target_family": "NONE",
            "target_frequency_hz": None,
            "levels": levels,
            "bands": bands,
        }
    active = [
        family
        for family, level in levels.items()
        if level in {"strong", "moderate"}
    ]
    target_family = getattr(baseline, "leading_family", None)
    if target_family not in levels:
        target_family = max(levels, key=lambda family: levels[family]) if levels else None

    if len(active) >= 2:
        return {
            "verdict": "conflict",
            "reason": "多个故障族同时存在中等及以上证据：" + ", ".join(active),
            "target_family": target_family,
            "target_frequency_hz": None,
            "levels": levels,
            "bands": bands,
        }

    if target_family and levels.get(target_family) == "strong":
        return {
            "verdict": "clear",
            "reason": f"{target_family} 达到强证据，跨频带一致",
            "target_family": target_family,
            "target_frequency_hz": _frequency(evidence, target_family),
            "levels": levels,
            "bands": bands,
        }

    if (
        target_family
        and levels.get(target_family) == "moderate"
        and bands.get(target_family, 0) >= 2
    ):
        return {
            "verdict": "clear",
            "reason": f"{target_family} 中等证据且至少两个有效频带",
            "target_family": target_family,
            "target_frequency_hz": _frequency(evidence, target_family),
            "levels": levels,
            "bands": bands,
        }

    return {
        "verdict": "uncertain",
        "reason": "初诊证据不足以明确故障族，需要精密分析",
        "target_family": target_family,
        "target_frequency_hz": _frequency(evidence, target_family) if target_family else None,
        "levels": levels,
        "bands": bands,
    }


def after_precision(
    gate: dict[str, Any],
    baseline: Any,
    precision_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Re-evaluate after one L3 round.

    The gate does not accept an arbitrary LLM confidence.  It requires a
    measurable improvement over the previous precision round while the target
    family stays consistent with the initial evidence.
    """

    if gate.get("verdict") in {"clear", "limited"}:
        return gate
    target = gate.get("target_family")
    improvements = [
        float(item.get("metrics", {}).get("improvement", 0.0) or 0.0)
        for item in precision_results
        if item.get("status") == "ok"
    ]
    best = max(improvements) if improvements else 0.0
    if target and best >= 0.30:
        return {
            **gate,
            "verdict": "clear",
            "reason": (
                f"精密分析使 {target} 目标故障能量提升 {best:.0%}，"
                "且基线故障族未改变"
            ),
            "precision_improvement": best,
        }
    return {
        **gate,
        "verdict": gate.get("verdict", "uncertain"),
        "reason": (
            "精密分析仍未形成足够清晰的目标故障证据"
            + (f"（{gate.get('verdict')} 未解除）" if gate.get("verdict") == "conflict" else "")
        ),
        "precision_improvement": best,
    }


def _frequency(evidence: EvidenceBundle, family: str) -> float | None:
    assessments = evidence.assessments or {}
    item = assessments.get(family)
    if not isinstance(item, Mapping):
        return None
    for key in ("refined_frequency_hz", "theoretical_frequency_hz"):
        value = item.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    frequencies = evidence.fault_frequencies_hz or {}
    value = frequencies.get(family)
    return float(value) if value else None
