"""State-driven preprocessing: conservative denoise + band-pass.

Parameters come from the signal assessment (narrowband/impulsiveness/spectral
kurtosis bands), never from fault-family matching or labels.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from bearing_suite_app import geometry as geom
from bearing_suite_app.toolbox.contracts import ToolContext

from . import metrics as metric_mod


def build_plan(
    assessment: Mapping[str, Any],
    sampling_rate_hz: float,
    rpm: float,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    state = assessment.get("state") or {}
    tf = assessment.get("time_frequency") or {}
    freq = assessment.get("frequency_domain") or {}
    fault_freqs = (assessment.get("fault_observability") or {}).get("fault_frequencies_hz") or {}
    try:
        max_fault = max(float(value) for value in fault_freqs.values() if float(value) > 0)
    except ValueError:
        max_fault = 0.0
    if max_fault <= 0:
        ratios = geom.fault_frequency_ratios(geometry)
        freqs = geom.fault_frequencies_hz(float(rpm), ratios)
        max_fault = max((float(value) for value in freqs.values()), default=1.0)
    low_floor = max(50.0, 5.0 * max_fault)
    nyq = float(sampling_rate_hz) / 2.0
    bands = [
        [float(band[0]), float(band[1])]
        for band in (tf.get("spectral_kurtosis_bands_hz") or [])
        if len(band) >= 2 and float(band[1]) > float(band[0])
    ]
    if bands:
        low = max(low_floor, min(band[0] for band in bands[:2]))
        high = min(0.95 * nyq, max(band[1] for band in bands[:2]))
    else:
        low = low_floor
        high = 0.95 * nyq
    if high <= low:
        low, high = low_floor, 0.95 * nyq
    steps = []
    narrow = int(freq.get("narrowband_peak_count") or 0)
    interference = str(state.get("narrowband_interference") or "none")
    if interference in {"mild", "strong"} or narrow >= 4:
        steps.append(
            {
                "tool": "adaptive_notch",
                "params": {"max_notches": min(4, max(1, narrow)), "auto": True, "max_trials": 2},
            }
        )
    selected = (assessment.get("spectral") or {}).get("envelope_band_selection", {}).get("selected")
    if not selected:
        steps.append(
            {
                "tool": "bandpass",
                "params": {"low_hz": round(low, 3), "high_hz": round(high, 3), "order": 4},
            }
        )
    return {
        "steps": steps,
        "low_hz": round(low, 3),
        "high_hz": round(high, 3),
        "reason": [
            "narrowband interference" if steps and steps[0]["tool"] == "adaptive_notch" else "no notch",
            "band-pass applied within each selector" if selected else "band-pass derived from spectral-kurtosis bands",
        ],
    }


def run_preprocess(
    signal: Sequence[float] | np.ndarray,
    fs: float,
    plan: Mapping[str, Any],
    registry: Any,
    context: ToolContext,
    bands: Sequence[Sequence[float]],
    *,
    max_trials: int = 2,
) -> dict[str, Any]:
    context.notes["processing_objective"] = "signal_state"
    original = np.asarray(signal, dtype=float).reshape(-1)
    current = original.copy()
    applied = []
    for step in plan.get("steps") or []:
        tool = str(step.get("tool") or "")
        params = dict(step.get("params") or {})
        auto = bool(params.pop("auto", False))
        requested_trials = int(params.pop("max_trials", max_trials))
        trial_limit = max(1, min(max_trials, requested_trials))
        try:
            result = registry.execute(tool, current, context, params, auto=auto, max_trials=trial_limit)
        except Exception as exc:  # noqa: BLE001
            applied.append({"tool": tool, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            continue
        if result.status == "ok" and result.arrays.get("signal") is not None:
            current = np.asarray(result.arrays["signal"], dtype=float).reshape(-1)
        applied.append({"tool": tool, "status": result.status, "params": result.params, "error": result.error})
    gate = metric_mod.hard_gate(original, current, fs, bands)
    before = metric_mod.metrics(original, fs, bands)
    after = metric_mod.metrics(current, fs, bands) if gate["passed"] else {}
    keys = ("envelope_snr_db", "modulation_index", "band_energy_ratio")
    comparable = [key for key in keys if key in before and key in after]
    dominated = bool(comparable) and all(after[key] <= before[key] for key in comparable) and any(
        after[key] < before[key] for key in comparable
    )
    fallback_reason = None
    if not gate["passed"]:
        fallback_reason = "hard_gate"
    elif dominated:
        fallback_reason = "metrics_dominated"
    fallback = fallback_reason is not None
    return {
        "plan": plan,
        "applied": applied,
        "before_metrics": before,
        "after_metrics": after,
        "hard_gate": gate,
        "fallback": fallback,
        "fallback_reason": fallback_reason,
        "signal": original if fallback else current,
    }
