"""Layer 3: adaptive precision analysis.

Only called when the evidence gate marks the initial diagnosis uncertain or
conflicting.  Every tool returns an enhanced signal plus local metrics; the
adaptive tuner tries a bounded parameter set and records all trials.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from bearing_suite_app import signal_tools
from bearing_suite_app.precision.errors import MethodNotImplementedError
from bearing_suite_app.precision.methods.med import med
from bearing_suite_app.precision.methods.mckd import mckd
from bearing_suite_app.precision.methods.momeda import momeda

from .contracts import ToolContext, ToolResult, ToolSpec, timed
from .l1 import _result, _stats
from .l2 import _envelope, _fault_energy
from .registry import ToolRegistry, parameter


def _target_frequency(context: ToolContext) -> float:
    target = context.notes.get("target_fault_frequency_hz")
    if target:
        try:
            value = float(target)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    values = [float(value) for value in context.fault_frequencies_hz.values() if value > 0]
    return max(values) if values else 1.0


def _quality(signal: np.ndarray, context: ToolContext) -> dict[str, float]:
    envelope = _envelope(signal, context)
    snr = float(envelope.get("envelope_snr_db", 0.0))
    stats = _stats(signal)
    if context.notes.get("processing_objective") == "signal_state":
        # Guided preprocessing ranks a dimensionless signal indicator.  It must
        # not use a candidate fault family or its harmonics as its objective.
        return {
            "objective": max(0.0, snr),
            "envelope_snr_db": snr,
            "kurtosis": stats["kurtosis"],
        }
    energy, _, by_family = _fault_energy(
        envelope["frequencies_hz"], envelope["amplitudes"], context
    )
    target = str(context.notes.get("target_family") or "")
    if target not in by_family:
        target = max(by_family, key=by_family.get) if by_family else ""
    target_energy = float(by_family.get(target, 0.0))
    others = [value for family, value in by_family.items() if family != target]
    runner_up = float(max(others)) if others else 0.0
    margin = (target_energy - runner_up) / (target_energy + runner_up + 1e-12)
    objective = target_energy * (1.0 + max(0.0, snr) / 20.0) * (1.0 + 0.5 * max(0.0, margin))
    return {
        "objective": objective,
        "fault_energy": float(energy),
        "target_family_energy": target_energy,
        "runner_up_energy": runner_up,
        "target_margin": float(margin),
        "envelope_snr_db": snr,
        "kurtosis": stats["kurtosis"],
        "best_family_energy": float(max(by_family.values())) if by_family else 0.0,
    }


def _improvement(before: dict[str, float], after: dict[str, float]) -> float:
    key = "fault_energy" if "fault_energy" in before else "objective"
    base = float(before.get(key, 0.0))
    return (float(after.get(key, 0.0)) - base) / max(abs(base), 1e-12)


@timed
def adaptive_notch_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["adaptive_notch"]
    values = np.asarray(x, dtype=float).reshape(-1)
    fs = context.sampling_rate_hz
    max_notches = int(params.get("max_notches", 2))
    q_factor = float(params.get("q_factor", 20.0))

    taper = scipy_signal.windows.hann(values.size, sym=False)
    amps = np.abs(np.fft.rfft(values * taper))
    freqs = np.fft.rfftfreq(values.size, d=1.0 / fs)
    fault_values = (
        [] if context.notes.get("processing_objective") == "signal_state"
        else [float(value) for value in context.fault_frequencies_hz.values() if value > 0]
    )
    candidates = []
    for index in np.argsort(amps)[::-1]:
        frequency = float(freqs[index])
        if frequency < 20.0 or frequency >= fs / 2.0:
            continue
        if any(abs(frequency - target) <= max(2.0, 0.05 * target) for target in fault_values):
            continue
        if any(abs(frequency - existing) < 5.0 for existing in candidates):
            continue
        candidates.append(frequency)
        if len(candidates) >= max_notches:
            break
    if not candidates:
        return _result(
            spec,
            params=params,
            data={"notches_hz": [], "reason": "no interference peaks"},
            metrics={"objective": 0.0},
            signal=values,
            evidence_id="L3.notch",
        )

    cleaned = values
    for frequency in candidates:
        try:
            b, a = scipy_signal.iirnotch(frequency, q_factor, fs)
            cleaned = scipy_signal.filtfilt(b, a, cleaned)
        except ValueError:
            continue
    before = _quality(values, context)
    after = _quality(cleaned, context)
    return _result(
        spec,
        params=params,
        data={
            "notches_hz": [round(value, 3) for value in candidates],
            "q_factor": q_factor,
            "improvement": round(_improvement(before, after), 6),
        },
        metrics={**after, "improvement": _improvement(before, after)},
        signal=cleaned,
        evidence_id="L3.notch",
    )


def _deconvolution(
    spec: ToolSpec,
    function: Any,
    x: np.ndarray,
    context: ToolContext,
    params: dict[str, Any],
) -> ToolResult:
    values = np.asarray(x, dtype=float).reshape(-1)
    try:
        enhanced = function(
            values,
            context.sampling_rate_hz,
            None if spec.name == "med" else _target_frequency(context),
            **params,
        )
    except (MethodNotImplementedError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return _result(
            spec,
            status="error",
            params=params,
            error=str(exc),
            error_code="method_failed",
        )
    enhanced = np.asarray(enhanced, dtype=float)
    if enhanced.ndim != 1 or enhanced.size != values.size or not np.all(np.isfinite(enhanced)):
        return _result(
            spec, status="error", params=params,
            error="deconvolution must return finite signal with unchanged shape",
            error_code="invalid_output",
        )
    before = _quality(values, context)
    after = _quality(enhanced, context)
    return _result(
        spec,
        params=params,
        data={"improvement": round(_improvement(before, after), 6)},
        metrics={**after, "improvement": _improvement(before, after)},
        signal=enhanced,
        evidence_id=f"L3.{spec.name}",
    )


@timed
def med_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    return _deconvolution(_SPECS["med"], med, x, context, params)


@timed
def mckd_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    return _deconvolution(_SPECS["mckd"], mckd, x, context, params)


@timed
def momeda_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    return _deconvolution(_SPECS["momeda"], momeda, x, context, params)


@timed
def iterative_sk_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["iterative_sk"]
    values = np.asarray(x, dtype=float).reshape(-1)
    fs = context.sampling_rate_hz
    band_count = int(params.get("band_count", 6))
    nyq = fs / 2.0
    edges = np.geomspace(max(30.0, context.max_fault_frequency_hz * 0.5), nyq * 0.95, band_count + 1)
    scored = []
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        if high - low < 20.0:
            continue
        filtered = signal_tools.bandpass_filter(values, fs, low, high)
        envelope = _envelope(filtered, context)
        energy, _, _ = _fault_energy(
            envelope["frequencies_hz"], envelope["amplitudes"], context
        )
        snr = float(envelope.get("envelope_snr_db", 0.0))
        scored.append(
            {
                "low_hz": low,
                "high_hz": high,
                "objective": float(energy) * (1.0 + max(0.0, snr) / 20.0),
                "fault_energy": float(energy),
                "envelope_snr_db": snr,
            }
        )
    if not scored:
        return _result(spec, status="error", params=params, error="no valid band", error_code="no_band")
    best = max(scored, key=lambda item: item["objective"])
    # 第二轮：在最优带内再细分为两条子带，保留得分更高者
    filtered = signal_tools.bandpass_filter(values, fs, best["low_hz"], best["high_hz"])
    mid = float(np.sqrt(best["low_hz"] * best["high_hz"]))
    for low, high in ((best["low_hz"], mid), (mid, best["high_hz"])):
        if high - low < 10.0:
            continue
        candidate = signal_tools.bandpass_filter(values, fs, low, high)
        envelope = _envelope(candidate, context)
        energy, _, _ = _fault_energy(
            envelope["frequencies_hz"], envelope["amplitudes"], context
        )
        snr = float(envelope.get("envelope_snr_db", 0.0))
        objective = float(energy) * (1.0 + max(0.0, snr) / 20.0)
        if objective > best["objective"]:
            best = {
                "low_hz": low,
                "high_hz": high,
                "objective": objective,
                "fault_energy": float(energy),
                "envelope_snr_db": snr,
            }
            filtered = candidate
    before = _quality(values, context)
    return _result(
        spec,
        params=params,
        data={
            "selected_band_hz": [round(best["low_hz"], 3), round(best["high_hz"], 3)],
            "candidate_bands": [
                {
                    "low_hz": round(item["low_hz"], 3),
                    "high_hz": round(item["high_hz"], 3),
                    "objective": round(item["objective"], 8),
                }
                for item in scored
            ],
            "improvement": round(_improvement(before, best), 6),
        },
        metrics={**best, "improvement": _improvement(before, best)},
        signal=filtered,
        evidence_id="L3.iterative_sk",
    )


@timed
def subband_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["subband_analysis"]
    values = np.asarray(x, dtype=float).reshape(-1)
    fs = context.sampling_rate_hz
    band_count = int(params.get("band_count", 10))
    overlap = float(params.get("overlap", 0.25))
    nyq = fs / 2.0
    edges = np.geomspace(max(20.0, context.max_fault_frequency_hz * 0.3), nyq * 0.95, band_count + 1)
    candidates = []
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        width = high - low
        step = width * (1.0 - overlap)
        if step < 10.0:
            continue
        filtered = signal_tools.bandpass_filter(values, fs, low, high)
        envelope = _envelope(filtered, context)
        energy, _, _ = _fault_energy(
            envelope["frequencies_hz"], envelope["amplitudes"], context
        )
        snr = float(envelope.get("envelope_snr_db", 0.0))
        candidates.append(
            {
                "low_hz": low,
                "high_hz": high,
                "objective": float(energy) * (1.0 + max(0.0, snr) / 20.0),
                "fault_energy": float(energy),
                "envelope_snr_db": snr,
            }
        )
    if not candidates:
        return _result(spec, status="error", params=params, error="no subband", error_code="no_band")
    best = max(candidates, key=lambda item: item["objective"])
    enhanced = signal_tools.bandpass_filter(values, fs, best["low_hz"], best["high_hz"])
    before = _quality(values, context)
    return _result(
        spec,
        params=params,
        data={
            "selected_band_hz": [round(best["low_hz"], 3), round(best["high_hz"], 3)],
            "candidate_count": len(candidates),
            "improvement": round(_improvement(before, best), 6),
        },
        metrics={**best, "improvement": _improvement(before, best)},
        signal=enhanced,
        evidence_id="L3.subband",
    )


_SPECS: dict[str, ToolSpec] = {
    "adaptive_notch": ToolSpec(
        name="adaptive_notch",
        version="1.0.0",
        layer="L3",
        family="interference",
        description="Detect narrow-band interference peaks and apply adaptive IIR notches.",
        parameters=(
            parameter("max_notches", "int", "maximum notches", default=2, minimum=1, maximum=4, adaptive=True),
            parameter("q_factor", "float", "notch quality factor", default=20.0, minimum=5.0, maximum=60.0, adaptive=True),
        ),
        output_fields=("notches_hz", "improvement"),
        candidates_fn=lambda context: [
            {"max_notches": 1, "q_factor": 15.0},
            {"max_notches": 2, "q_factor": 20.0},
            {"max_notches": 3, "q_factor": 30.0},
        ],
    ),
    "med": ToolSpec(
        name="med", version="1.0.0", layer="L3", family="deconvolution",
        description="Minimum entropy deconvolution; no fault-period prior.",
        parameters=(
            parameter("filter_length", "int", "FIR length", default=30, minimum=10, maximum=100, adaptive=True),
            parameter("max_iter", "int", "optimizer iterations", default=200, minimum=50, maximum=500),
        ),
        output_fields=("signal", "improvement"),
        candidates_fn=lambda context: [
            {"filter_length": 20, "max_iter": 150},
            {"filter_length": 30, "max_iter": 200},
            {"filter_length": 40, "max_iter": 250},
        ],
    ),
    "mckd": ToolSpec(
        name="mckd", version="1.0.0", layer="L3", family="deconvolution",
        description="Maximum correlated kurtosis deconvolution using target fault period.",
        parameters=(
            parameter("filter_length", "int", "FIR length", default=30, minimum=10, maximum=100, adaptive=True),
            parameter("m", "int", "correlation order", default=1, minimum=1, maximum=3, adaptive=True),
            parameter("max_iter", "int", "optimizer iterations", default=100, minimum=30, maximum=300),
        ),
        output_fields=("signal", "improvement"),
        candidates_fn=lambda context: [
            {"filter_length": 20, "m": 1, "max_iter": 80},
            {"filter_length": 30, "m": 1, "max_iter": 100},
            {"filter_length": 30, "m": 2, "max_iter": 120},
        ],
    ),
    "momeda": ToolSpec(
        name="momeda", version="1.0.0", layer="L3", family="deconvolution",
        description="Multipoint optimal minimum entropy deconvolution (closed form).",
        parameters=(
            parameter("filter_length", "int", "FIR length", default=200, minimum=50, maximum=600, adaptive=True),
            parameter("window", "int", "target window", default=3, minimum=1, maximum=7, adaptive=True),
        ),
        output_fields=("signal", "improvement"),
        candidates_fn=lambda context: [
            {"filter_length": 100, "window": 3},
            {"filter_length": 200, "window": 3},
            {"filter_length": 300, "window": 5},
        ],
    ),
    "iterative_sk": ToolSpec(
        name="iterative_sk", version="1.0.0", layer="L3", family="resonance",
        description="Iterative spectral-kurtosis band selection and demodulation.",
        parameters=(
            parameter("band_count", "int", "initial subbands", default=6, minimum=4, maximum=12, adaptive=True),
        ),
        output_fields=("selected_band_hz", "candidate_bands", "improvement"),
        candidates_fn=lambda context: [{"band_count": 4}, {"band_count": 6}, {"band_count": 8}],
    ),
    "subband_analysis": ToolSpec(
        name="subband_analysis", version="1.0.0", layer="L3", family="subband",
        description="Overlapping subband filterbank scored by fault energy and envelope SNR.",
        parameters=(
            parameter("band_count", "int", "number of subbands", default=10, minimum=4, maximum=20, adaptive=True),
            parameter("overlap", "float", "fractional overlap", default=0.25, minimum=0.0, maximum=0.75, adaptive=True),
        ),
        output_fields=("selected_band_hz", "candidate_count", "improvement"),
        candidates_fn=lambda context: [
            {"band_count": 8, "overlap": 0.25},
            {"band_count": 10, "overlap": 0.25},
            {"band_count": 12, "overlap": 0.5},
        ],
    ),
}


def register_l3(registry: ToolRegistry) -> None:
    registry.register(_SPECS["adaptive_notch"], adaptive_notch_tool)
    registry.register(_SPECS["med"], med_tool)
    registry.register(_SPECS["mckd"], mckd_tool)
    registry.register(_SPECS["momeda"], momeda_tool)
    registry.register(_SPECS["iterative_sk"], iterative_sk_tool)
    registry.register(_SPECS["subband_analysis"], subband_tool)
