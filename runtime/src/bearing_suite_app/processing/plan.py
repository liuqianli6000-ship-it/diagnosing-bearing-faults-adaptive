"""Initial processing plan (round 1) from the signal assessment."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _bands(assessment: Mapping[str, Any]) -> list[list[float]]:
    return [
        [float(band[0]), float(band[1])]
        for band in (assessment.get("spectral") or {}).get("spectral_kurtosis_bands_hz", [])
        if isinstance(band, (list, tuple)) and len(band) >= 2
    ]


def normalize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Map LLM-friendly aliases to the tool contract without changing bounds."""

    normalized = dict(plan)
    rounds = []
    for round_plan in plan.get("rounds") or []:
        item = dict(round_plan)
        filters = []
        for step in item.get("filters") or []:
            step = dict(step)
            params = dict(step.get("params") or {})
            if step.get("tool") == "bandpass":
                params["low_hz"] = params.pop("low_hz", params.pop("low", 5.0))
                params["high_hz"] = params.pop("high_hz", params.pop("high", 3000.0))
            step["params"] = params
            filters.append(step)
        processing = []
        for step in item.get("processing") or []:
            step = dict(step)
            params = dict(step.get("params") or {})
            if step.get("tool") == "med":
                params["filter_length"] = max(
                    10, min(100, int(params.get("filter_length", params.pop("window_size", 30))))
                )
                params["max_iter"] = max(50, min(500, int(params.get("max_iter", 200))))
            step["params"] = params
            processing.append(step)
        item["filters"] = filters
        item["processing"] = processing
        rounds.append(item)
    normalized["rounds"] = rounds
    return normalized


def build_plan(assessment: Mapping[str, Any], *, adaptive: bool = True) -> dict[str, Any]:
    """Build only the first guided processing round.

    Baseline is a round-0 state computed by the round engine, never a round.
    Rounds 2/3 are produced by ``refine_plan`` from the previous result.
    """

    quality = assessment.get("quality") or {}
    if not quality.get("valid_for_analysis", False) or not adaptive:
        return {
            "reason": "quality gate failed" if not quality.get("valid_for_analysis") else "adaptive disabled",
            "rounds": [],
            "max_rounds": 3,
            "source": "local",
        }
    bands = _bands(assessment)
    noise = assessment.get("noise") or {}
    impulsive = assessment.get("impulsiveness") or {}
    narrow = int(noise.get("narrowband_peak_count") or 0)
    kurtosis = float(impulsive.get("kurtosis") or 0.0)

    filters: list[dict[str, Any]] = []
    processing: list[dict[str, Any]] = []
    method = "bandpass"
    direction = {"envelope_snr": "up", "out_of_band": "down", "modulation": "keep"}
    if narrow >= 4:
        filters.append({"tool": "adaptive_notch", "params": {"max_notches": 2, "q_factor": 20.0}})
        method = "adaptive_notch"
    if kurtosis >= 6.0:
        filters.append(
            {
                "tool": "bandpass",
                "params": {
                    "low_hz": bands[0][0] if bands else 500.0,
                    "high_hz": bands[-1][1] if bands else 3000.0,
                },
            }
        )
        processing.append({"tool": "med", "params": {"filter_length": 30, "max_iter": 200}})
        method += "+bandpass+med"
    elif bands:
        filters.append(
            {
                "tool": "bandpass",
                "params": {"low_hz": bands[0][0], "high_hz": bands[0][1]},
            }
        )
        method += "+bandpass"
    if not filters and not processing:
        return {"reason": "no processing needed", "rounds": [], "max_rounds": 3, "source": "local"}
    return {
        "reason": "assessment-driven initial plan",
        "max_rounds": 3,
        "source": "local",
        "rounds": [
            {
                "round": 1,
                "parent": 0,
                "method": method,
                "filters": filters,
                "processing": processing,
                "bands": bands,
                "direction": direction,
                "reason": "initial plan from signal characteristics",
            }
        ],
    }


def valid_processing_plan(plan: Any, sampling_rate_hz: float | None = None) -> bool:
    if not isinstance(plan, dict):
        return False
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        return False
    for round_plan in rounds:
        if not isinstance(round_plan, dict):
            return False
        filters = round_plan.get("filters") or []
        processing = round_plan.get("processing") or []
        if not isinstance(filters, list) or not isinstance(processing, list):
            return False
        if len(filters) > 2 or len(processing) > 1:
            return False
        for steps, allowed in ((filters, {"bandpass", "adaptive_notch"}), (processing, {"med"})):
            for step in steps:
                if not isinstance(step, dict) or step.get("tool") not in allowed:
                    return False
                if not isinstance(step.get("params", {}), dict):
                    return False
                if step["tool"] == "bandpass":
                    params = step.get("params", {})
                    if not _valid_band([params.get("low_hz", params.get("low", 5.0)), params.get("high_hz", params.get("high", 3000.0))], sampling_rate_hz):
                        return False
        if round_plan.get("round", 1) != 1 or round_plan.get("parent", 0) != 0:
            return False
        bands = round_plan.get("bands") or []
        if not isinstance(bands, list) or any(not _valid_band(band, sampling_rate_hz) for band in bands):
            return False
    return True


def _valid_band(band: Any, sampling_rate_hz: float | None) -> bool:
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        return False
    try:
        low, high = map(float, band)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite([low, high]).all() and 0 < low < high and (sampling_rate_hz is None or high <= sampling_rate_hz / 2.0))
