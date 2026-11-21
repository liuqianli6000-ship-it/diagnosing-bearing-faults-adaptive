"""Fixed envelope matching on explicit bands."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from bearing_suite_app.agent import BearingDiagnosisAgent
from bearing_suite_app.assessment.band_banks import BAND_METHODS
from bearing_suite_app.decision import FAMILY_TO_FAULT


def _concrete(fault_type: str | None) -> bool:
    return bool(fault_type) and fault_type not in {"unknown", "uncertain"}


def route_band_evidence(results: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """Choose which computed strategies the model should examine."""
    usable = []
    sufficient = []
    for method in BAND_METHODS:
        result = results.get(method, {})
        evidence = result.get("evidence")
        if evidence is None:
            continue
        gate = evidence.diagnosability
        if not evidence.analyzable and not (
            gate.get("invalidating_flags") == ["insufficient_valid_demodulation_bands"]
            and gate.get("stages", {}).get("band_diagnosability", {}).get("valid_band_count", 0) > 0
        ):
            continue
        usable.append(method)
        supported = [
            family for family, assessment in evidence.assessments.items()
            if family in FAMILY_TO_FAULT and assessment.get("evidence_level") in {"moderate", "strong"}
            and assessment.get("fit_supported_bands", 0) > 0
        ]
        if (evidence.analyzable and len(supported) == 1
                and result.get("fault_type") == FAMILY_TO_FAULT[supported[0]][0]):
            sufficient.append(method)

    primary = BAND_METHODS[0]
    if primary in sufficient:
        return "primary", [primary]
    if sufficient and len({results[method]["fault_type"] for method in sufficient}) == 1:
        return "candidate", sufficient
    return ("combined", usable) if usable else ("unavailable", [])


def _whole_record_result(
    agent: BearingDiagnosisAgent,
    signal: np.ndarray,
    sampling_rate_hz: float,
    rpm: float,
    geometry: dict[str, Any],
    bands: Sequence[Sequence[float]],
    on_envelope_spectra: Callable[[list[dict[str, Any]]], None] | None = None,
    band_method: str | None = None,
) -> dict[str, Any]:
    evidence, analysis = agent.collect_envelope_evidence(
        signal, sampling_rate_hz, rpm, geometry, bands,
        **({"retain_spectra": True} if on_envelope_spectra else {}),
        **({"band_method": band_method} if band_method else {}),
    )
    if on_envelope_spectra:
        on_envelope_spectra(analysis.get("spectrum_artifacts", []))
    baseline = agent._baseline_from_evidence(evidence, include_reference_cases=False)
    return {
        "evidence": evidence,
        "baseline": baseline,
        "fault_type": baseline.fault_type,
        "confidence": float(baseline.confidence),
        "status": baseline.status,
        "leading_family": baseline.leading_family,
        "reason": baseline.reason,
    }



def _segment_consensus(
    agent: BearingDiagnosisAgent,
    signal: np.ndarray,
    sampling_rate_hz: float,
    rpm: float,
    geometry: dict[str, Any],
    bands: Sequence[Sequence[float]],
    full_fault: str,
) -> dict[str, Any]:
    """Check the fixed diagnosis on independent time segments.

    This is a stability check, not a parameter search.  It is skipped for short
    records where a segment would not contain enough revolutions.
    """

    values = np.asarray(signal, dtype=float).reshape(-1)
    duration = values.size / max(float(sampling_rate_hz), 1e-9)
    revolutions = duration * float(rpm) / 60.0
    if duration < 4.5 or revolutions < 8.0:
        return {"checked": False, "reason": "record_too_short_for_segments"}
    segment_size = values.size // 3
    if segment_size < max(1024, int(sampling_rate_hz * 1.0)):
        return {"checked": False, "reason": "segment_too_short"}
    segment_results = []
    for index in range(3):
        segment = values[index * segment_size : (index + 1) * segment_size]
        try:
            evidence, _ = agent.collect_envelope_evidence(
                segment, sampling_rate_hz, rpm, geometry, bands
            )
            baseline = agent._baseline_from_evidence(
                evidence, include_reference_cases=False
            )
            segment_results.append(
                {
                    "index": index,
                    "fault_type": baseline.fault_type,
                    "status": baseline.status,
                    "leading_family": baseline.leading_family,
                    "analyzable": bool(evidence.analyzable),
                    "assessments": {
                        family: {key: item.get(key) for key in ("evidence_level", "evidence_bands",
                                                               "fit_supported_bands", "qualified_orders",
                                                               "max_residual_bins")}
                        for family, item in getattr(evidence, "assessments", {}).items()
                    },
                }
            )
        except Exception as exc:  # noqa: BLE001
            segment_results.append(
                {"index": index, "fault_type": "unknown", "error": f"{type(exc).__name__}: {exc}"}
            )
    analyzable = [item for item in segment_results if item.get("analyzable")]
    fault_types = [item.get("fault_type") for item in analyzable]
    support = sum(item == full_fault for item in fault_types)
    normal_count = sum(item == "normal" for item in fault_types)
    conflict = any(_concrete(item) and item not in {"normal", full_fault} for item in fault_types)
    consistent = len(analyzable) == 3 and all(item == full_fault for item in fault_types)
    if not analyzable:
        reason = "no analyzable segments"
    elif conflict:
        reason = "segment conflict"
    elif consistent:
        reason = "segments agree"
    else:
        reason = "segment support insufficient"
    return {
        "checked": True,
        "consensus": consistent,
        "conflict": conflict,
        "support": support,
        "analyzable_count": len(analyzable),
        "normal_count": normal_count,
        "full_fault": full_fault,
        "segments": segment_results,
        "reason": reason,
    }


def diagnose_numeric(
    agent: BearingDiagnosisAgent,
    signal: np.ndarray,
    sampling_rate_hz: float,
    rpm: float,
    geometry: dict[str, Any],
    bands: Sequence[Sequence[float]],
    *,
    on_envelope_spectra: Callable[[list[dict[str, Any]]], None] | None = None,
    band_method: str | None = None,
) -> dict[str, Any]:
    primary = _whole_record_result(agent, signal, sampling_rate_hz, rpm, geometry, bands, on_envelope_spectra, band_method)
    segment = _segment_consensus(
        agent,
        signal,
        sampling_rate_hz,
        rpm,
        geometry,
        bands,
        primary["fault_type"],
    )

    verdict = {key: primary[key] for key in ("fault_type", "confidence", "status", "reason")}
    if segment.get("conflict"):
        verdict.update(fault_type="unknown", confidence=0.0, status="unknown",
                       reason="分段证据冲突：" + segment["reason"])
    return {
        "evidence": primary["evidence"],
        "baseline": primary["baseline"],
        **verdict,
        "segment_consensus": segment,
        "segment_instability": bool(_concrete(primary["fault_type"]) and segment.get("checked")
                                    and not segment.get("consensus") and not segment.get("conflict")),
        "leading_family": primary["leading_family"],
        "bands": [[float(b[0]), float(b[1])] for b in bands],
    }
