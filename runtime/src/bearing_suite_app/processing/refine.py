"""Refine rounds 2/3 from the previous round's actual deltas."""

from __future__ import annotations

from typing import Any, Mapping


def _adjust_band(band: list[float], wider: bool, fs: float) -> list[float]:
    low, high = float(band[0]), float(band[1])
    width = max(high - low, 1.0)
    delta = width * 0.10
    if wider:
        low = max(0.0, low - delta)
        high = min(fs / 2.0 * 0.98, high + delta)
    else:
        low = min(low + delta, high - max(1.0, width * 0.3))
        high = max(high - delta, low + 1.0)
    return [round(low, 3), round(high, 3)]


def decide_mode(result: Mapping[str, Any], baseline: Mapping[str, Any] | None = None) -> str:
    """Choose refine/retry from actual deltas, without performance thresholds."""

    if not (result.get("hard_gate") or {}).get("passed", True):
        return "retry"
    deltas = {
        key: float(value)
        for key, value in (result.get("deltas") or {}).items()
        if key in {"envelope_snr_db", "modulation_index", "band_energy_ratio", "kurtosis", "crest_factor"}
    }
    if not deltas:
        return "retry"
    positive = sum(value > 0 for value in deltas.values())
    if positive == 0:
        return "retry"

    # If R1 is not better than the baseline in the actual metric ranking, retry.
    if baseline is not None:
        try:
            from . import metrics as metric_mod

            ranking = metric_mod.rank_candidates(
                [
                    {
                        "round": 0,
                        "hard_gate": baseline.get("hard_gate") or {"passed": True},
                        "after": baseline.get("after") or {},
                    },
                    {
                        "round": result.get("round", 1),
                        "hard_gate": result.get("hard_gate") or {"passed": True},
                        "after": result.get("after") or {},
                    },
                ]
            )
            best = ranking.get("best") or {}
            if int(best.get("round", 0)) == 0:
                return "retry"
        except Exception:  # noqa: BLE001
            pass

    # Sign-based diagnostic preference: if both envelope quality and band
    # energy got worse, a different path is more promising than more tuning.
    snr = float(deltas.get("envelope_snr_db", 0.0))
    # Envelope SNR is the primary bearing-diagnosis metric; if it fell, try a
    # different path rather than only tuning the same filter.
    if snr < 0:
        return "retry"
    return "refine"


def retry_plan(
    previous_plan: Mapping[str, Any],
    result: Mapping[str, Any],
    assessment: Mapping[str, Any],
    round_no: int,
    fs: float,
) -> dict[str, Any] | None:
    """Switch to a different mechanism when the previous path did not help."""

    prev_filters = [step.get("tool") for step in previous_plan.get("filters", [])]
    prev_processing = [step.get("tool") for step in previous_plan.get("processing", [])]
    bands = [
        [float(band[0]), float(band[1])]
        for band in (previous_plan.get("bands") or [])
        if len(band) >= 2
    ]
    if not bands:
        bands = [
            [float(band[0]), float(band[1])]
            for band in (assessment.get("spectral") or {}).get("spectral_kurtosis_bands_hz", [])
            if len(band) >= 2
        ]
    noise = assessment.get("noise") or {}
    impulsive = assessment.get("impulsiveness") or {}
    narrow = int(noise.get("narrowband_peak_count") or 0)
    kurtosis = float(impulsive.get("kurtosis") or 0.0)

    # Prefer a mechanism that was not attempted in R1.
    if "adaptive_notch" not in prev_filters and narrow >= 4:
        filters = [{"tool": "adaptive_notch", "params": {"max_notches": 3, "q_factor": 30.0}}]
        processing: list[dict[str, Any]] = []
        method = "adaptive_notch"
    elif "med" not in prev_processing and kurtosis > 0:
        filters = (
            [{"tool": "bandpass", "params": {"low_hz": bands[0][0], "high_hz": bands[0][1]}}]
            if bands
            else []
        )
        processing = [{"tool": "med", "params": {"filter_length": 30, "max_iter": 200}}]
        method = "bandpass+med"
    else:
        filters = []
        processing = [{"tool": "subband_analysis", "params": {"band_count": 12, "overlap": 0.5}}]
        method = "subband_analysis"
    if not filters and not processing:
        return None
    return {
        "round": round_no,
        "parent": int(result.get("round", round_no - 1)),
        "mode": "retry",
        "method": method,
        "filters": filters,
        "processing": processing,
        "bands": bands,
        "direction": {"envelope_snr": "up", "out_of_band": "down", "modulation": "keep"},
        "change": {"from": prev_filters + prev_processing, "to": method},
        "reason": "retry a different mechanism because the previous path did not improve",
    }


def refine_plan(
    previous_round: Mapping[str, Any],
    result: Mapping[str, Any],
    assessment: Mapping[str, Any],
    round_no: int,
    fs: float,
) -> dict[str, Any]:
    """Return one refined round plan, based on actual deltas (no thresholds)."""

    previous_plan = result.get("plan") or previous_round.get("plan") or {}
    filters = [dict(step) for step in previous_plan.get("filters", [])]
    processing = [dict(step) for step in previous_plan.get("processing", [])]
    bands = [list(map(float, band)) for band in (previous_plan.get("bands") or []) if len(band) >= 2]
    deltas = result.get("deltas") or {}
    snr = float(deltas.get("envelope_snr_db", 0.0) or 0.0)
    modulation = float(deltas.get("modulation_index", 0.0) or 0.0)
    kurtosis = float(deltas.get("kurtosis", 0.0) or 0.0)
    change: dict[str, Any] = {}

    wider = snr < 0 or modulation < 0
    if bands:
        bands = [_adjust_band(band, wider=wider, fs=fs) for band in bands]
        change["bands"] = bands

    for step in filters:
        if step.get("tool") == "adaptive_notch":
            params = dict(step.get("params") or {})
            q = float(params.get("q_factor", 20.0))
            params["q_factor"] = round(max(5.0, min(60.0, q + (5.0 if snr < 0 else -2.0))), 2)
            if snr < 0:
                params["max_notches"] = min(4, int(params.get("max_notches", 2)) + 1)
            step["params"] = params
            change["adaptive_notch"] = params
        elif step.get("tool") == "bandpass" and bands:
            step["params"] = {"low_hz": bands[0][0], "high_hz": bands[-1][1]}
            change["bandpass"] = step["params"]

    for step in processing:
        if step.get("tool") == "med":
            params = dict(step.get("params") or {})
            length = int(params.get("filter_length", 30))
            if kurtosis < 0 or snr > 0:
                length = min(100, length + 10)
            else:
                length = max(10, length - 5)
            params["filter_length"] = length
            params["max_iter"] = min(500, int(params.get("max_iter", 200)) + 50)
            step["params"] = params
            change["med"] = params
        elif step.get("tool") == "subband_analysis":
            params = dict(step.get("params") or {})
            params["band_count"] = min(20, int(params.get("band_count", 10)) + 2)
            step["params"] = params
            change["subband_analysis"] = params

    return {
        "round": round_no,
        "parent": int(result.get("round", round_no - 1)),
        "mode": "refine",
        "method": previous_plan.get("method", "refine"),
        "filters": filters,
        "processing": processing,
        "bands": bands,
        "direction": previous_plan.get("direction", {}),
        "change": change,
        "reason": "refine from previous actual deltas",
    }
