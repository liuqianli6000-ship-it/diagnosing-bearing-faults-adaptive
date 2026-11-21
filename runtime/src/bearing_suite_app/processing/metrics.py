"""Industrial metrics, hard gates and qualitative comparison (no thresholds)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal as scipy_signal

from bearing_suite_app import signal_tools


def _time_metrics(x: np.ndarray) -> dict[str, float]:
    mean = float(np.mean(x))
    std = float(np.std(x))
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    abs_mean = float(np.mean(np.abs(x)))
    return {
        "rms": rms,
        "peak": peak,
        "p2p": float(np.ptp(x)),
        "std": std,
        "kurtosis": float(np.mean((x - mean) ** 4) / (std**4)) if std > 0 else 0.0,
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "impulse_factor": peak / abs_mean if abs_mean > 0 else 0.0,
        "clearance_factor": peak / (float(np.mean(np.sqrt(np.abs(x)))) ** 2) if abs_mean > 0 else 0.0,
    }


def _envelope_metrics(
    x: np.ndarray, fs: float, bands: Sequence[Sequence[float]]
) -> dict[str, float]:
    snrs = []
    modulations = []
    band_ratios = []
    input_rms = float(np.sqrt(np.mean((x - float(np.mean(x))) ** 2)))
    for band in bands:
        if len(band) < 2 or float(band[1]) <= float(band[0]):
            continue
        result = signal_tools.envelope_spectrum(x, fs, band_hz=band)
        snrs.append(float(result["envelope_snr_db"]))
        filtered = signal_tools.bandpass_filter(x, fs, float(band[0]), float(band[1]))
        analytic = np.abs(scipy_signal.hilbert(filtered))
        mean = float(np.mean(analytic))
        modulations.append(float(np.std(analytic) / mean) if mean > 0 else 0.0)
        filtered_rms = float(result.get("filtered_rms", 0.0))
        band_ratios.append(float((filtered_rms / input_rms) ** 2) if input_rms > 0 else 0.0)
    return {
        "envelope_snr_db": float(max(snrs)) if snrs else 0.0,
        "modulation_index": float(max(modulations)) if modulations else 0.0,
        "band_energy_ratio": float(max(band_ratios)) if band_ratios else 0.0,
    }


def metrics(x: np.ndarray, fs: float, bands: Sequence[Sequence[float]]) -> dict[str, float]:
    values = np.asarray(x, dtype=float).reshape(-1)
    result = _time_metrics(values)
    result.update(_envelope_metrics(values, fs, bands))
    return result


def hard_gate(
    before: np.ndarray,
    after: np.ndarray,
    fs: float,
    bands: Sequence[Sequence[float]],
) -> dict[str, Any]:
    before = np.asarray(before, dtype=float).reshape(-1)
    after = np.asarray(after, dtype=float).reshape(-1)
    reasons = []
    if not np.isfinite(fs) or fs <= 0:
        reasons.append("invalid_sampling_rate")
    if not after.size:
        return {"passed": False, "reasons": ["empty_signal", "length_changed"]}
    if before.size != after.size:
        reasons.append("length_changed")
    if not np.all(np.isfinite(after)):
        reasons.append("non_finite")
    if float(np.std(after)) < 1e-6 * max(float(np.std(before)), 1e-12):
        reasons.append("amplitude_collapsed")
    if float(np.ptp(after)) > 2.0 * max(float(np.ptp(before)), 1e-12):
        reasons.append("amplitude_exploded")
    edge = max(1, int(0.05 * after.size))
    if edge * 2 < after.size:
        edge_ratio = float(np.sum(after[:edge] ** 2) + np.sum(after[-edge:] ** 2))
        edge_base = float(np.sum(before[:edge] ** 2) + np.sum(before[-edge:] ** 2))
        if edge_ratio > 2.0 * max(edge_base, 1e-12) + 1e-9:
            reasons.append("edge_artifact")
    nyq = fs / 2.0
    for band in bands:
        if len(band) != 2:
            reasons.append("illegal_band")
            break
        low, high = float(band[0]), float(band[1])
        if not np.isfinite([low, high]).all() or low < 0 or high <= low or high > nyq:
            reasons.append("illegal_band")
            break
    return {"passed": not reasons, "reasons": reasons}


def compare(before: Mapping[str, float], after: Mapping[str, float]) -> dict[str, Any]:
    keys = [
        "envelope_snr_db",
        "modulation_index",
        "band_energy_ratio",
        "kurtosis",
        "crest_factor",
    ]
    deltas = {}
    relative = {}
    parts = []
    for key in keys:
        if key not in before or key not in after:
            continue
        delta = float(after[key]) - float(before[key])
        deltas[key] = round(delta, 8)
        relative[key] = round(delta / abs(float(before[key])), 8) if before[key] else None
        if abs(delta) < 1e-12:
            parts.append(f"{key} 基本持平")
        elif delta > 0:
            parts.append(f"{key} 提升 {delta:+.3f}")
        else:
            parts.append(f"{key} 下降 {delta:+.3f}")
    if not parts:
        overall = "无可比较指标"
    elif all(value >= -1e-12 for value in deltas.values()):
        overall = "整体提升或持平"
    elif all(value <= 1e-12 for value in deltas.values()):
        overall = "整体下降或持平"
    else:
        overall = "有升有降"
    return {
        "deltas": deltas,
        "relative": relative,
        "qualitative": {
            "parts": parts,
            "overall": overall,
        },
    }


def rank_candidates(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Sum ordinal metric ranks with equal ranks for equal metric values."""

    keys = ("envelope_snr_db", "modulation_index", "band_energy_ratio")
    valid = [
        item for item in candidates
        if (item.get("hard_gate") or {}).get("passed", True)
        and all(np.isfinite(float((item.get("after") or {}).get(key, 0.0))) for key in keys)
    ]
    if not valid:
        return {"best": None, "ranking": []}
    ranks = {id(item): 0 for item in valid}
    for key in keys:
        ordered = sorted(
            valid,
            key=lambda item: float((item.get("after") or {}).get(key, 0.0)),
            reverse=True,
        )
        previous_value = None
        metric_rank = 0
        for position, item in enumerate(ordered, start=1):
            value = float((item.get("after") or {}).get(key, 0.0))
            if value != previous_value:
                metric_rank = position
            ranks[id(item)] += metric_rank
            previous_value = value
    ordered_all = sorted(valid, key=lambda item: ranks[id(item)])
    for index, item in enumerate(ordered_all, start=1):
        item["rank"] = index
    return {"best": ordered_all[0], "ranking": ordered_all}
