"""Length-independent shaft-speed estimation from a vibration record.

Only the measured signal and the nominal shaft speed (search centre) are used.
Fault labels, fault-family frequencies and the fault comb are not inputs.  The
returned rpm is an estimated condition parameter, not a measured phase or
tachometer reference.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal


SETTINGS = {
    "harmonic_count": 5,
    "search_relative": 0.20,
    "peak_window_relative": 0.02,
    "minimum_harmonics": 2,
    "maximum_relative_residual": 0.01,
    "zero_pad": 4,
    "grid_step_hz": 0.005,
    "grid_candidates": 5,
}


def _validate(x, fs, nominal_rpm):
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size < 64:
        raise ValueError("signal must contain at least 64 samples")
    if not np.isfinite(values).all():
        raise ValueError("signal must be finite")
    sampling_rate = float(fs)
    if not math.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    nominal = float(nominal_rpm)
    if not math.isfinite(nominal) or nominal <= 0:
        raise ValueError("nominal_rpm must be positive")
    return values, sampling_rate, nominal


def _spectrum(values, fs, zero_pad):
    window = signal.windows.hann(values.size, sym=False)
    nfft = max(256, int(values.size * int(zero_pad)))
    amplitude = 2.0 * np.abs(np.fft.rfft((values - values.mean()) * window, n=nfft)) / window.sum()
    frequency = np.fft.rfftfreq(nfft, 1.0 / fs)
    return frequency, amplitude


def _peak(frequency, amplitude, target, half):
    if target <= 0 or target >= frequency[-1]:
        return None
    low = max(1, int(np.searchsorted(frequency, target - half, side="left")))
    high = min(amplitude.size - 1, int(np.searchsorted(frequency, target + half, side="right")))
    if high - low < 2:
        return None
    index = low + int(np.argmax(amplitude[low:high]))
    if index <= 0 or index >= amplitude.size - 1:
        return None
    y0, y1, y2 = np.log(np.maximum(amplitude[index - 1:index + 2], 1e-30))
    denominator = y0 - 2.0 * y1 + y2
    offset = 0.5 * (y0 - y2) / denominator if abs(denominator) > 1e-12 else 0.0
    return {"observed_hz": float(frequency[index] + offset * (frequency[1] - frequency[0])),
            "amplitude": float(amplitude[index])}


def _fit(frequency, amplitude, start_hz, settings):
    df = frequency[1] - frequency[0]
    harmonics = int(settings["harmonic_count"])
    minimum = int(settings["minimum_harmonics"])
    f0 = float(start_hz)
    observations = []
    for _ in range(2):
        observations = []
        for order in range(1, harmonics + 1):
            target = order * f0
            half = max(3.0 * df, float(settings["peak_window_relative"]) * target)
            peak = _peak(frequency, amplitude, target, half)
            if peak is not None:
                observations.append((order, peak["observed_hz"], peak["amplitude"]))
        if len(observations) < minimum:
            return None
        orders = np.array([item[0] for item in observations], dtype=float)
        peaks = np.array([item[1] for item in observations], dtype=float)
        weights = np.array([item[2] for item in observations], dtype=float)
        f0 = float(np.sum(weights * orders * peaks) / np.sum(weights * orders * orders))
    residual = float(np.sqrt(np.sum(weights * (peaks - orders * f0) ** 2) / np.sum(weights)))
    return {"shaft_hz": f0, "rpm": f0 * 60.0, "relative_residual": residual / f0 if f0 else None,
            "residual_hz": residual, "harmonics_used": len(observations),
            "harmonic_amplitude_sum": float(np.sum(weights)),
            "harmonics": [{"order": order, "observed_hz": peak, "amplitude": weight}
                          for order, peak, weight in observations]}


def _grid_candidates(frequency, amplitude, nominal_hz, settings):
    df = frequency[1] - frequency[0]
    harmonics = int(settings["harmonic_count"])
    search = float(settings["search_relative"])
    step = float(settings["grid_step_hz"])
    low, high = nominal_hz * (1.0 - search), nominal_hz * (1.0 + search)
    grid = np.arange(low, high + step / 2.0, step)
    scores = np.zeros_like(grid)
    for position, f0 in enumerate(grid):
        total = 0.0
        for order in range(1, harmonics + 1):
            target = order * f0
            half = max(3.0 * df, float(settings["peak_window_relative"]) * target)
            peak = _peak(frequency, amplitude, target, half)
            if peak is not None:
                total += peak["amplitude"] / order
        scores[position] = total
    order = np.argsort(scores)[-int(settings["grid_candidates"]):][::-1]
    selected = []
    for index in order:
        value = float(grid[index])
        if any(abs(value - other) <= 0.005 * nominal_hz for other in selected):
            continue
        selected.append(value)
    return selected


def estimate_shaft_speed(signal_values, sampling_rate_hz, nominal_rpm, **overrides) -> dict[str, Any]:
    """Estimate shaft speed from any-length vibration signal.

    Parameters are the same as ``SETTINGS``.  The result contains the used rpm,
    fit quality and the candidate list; ``method`` is ``nominal_fallback`` when
    no candidate met the residual/harmonic guard.
    """
    settings = dict(SETTINGS)
    unknown = set(overrides) - set(settings)
    if unknown:
        raise TypeError("unknown settings: " + ", ".join(sorted(unknown)))
    settings.update(overrides)
    values, fs, nominal = _validate(signal_values, sampling_rate_hz, nominal_rpm)
    nominal_hz = nominal / 60.0
    frequency, amplitude = _spectrum(values, fs, settings["zero_pad"])
    candidates = []
    nominal_fit = _fit(frequency, amplitude, nominal_hz, settings)
    if nominal_fit is not None:
        candidates.append({"origin": "nominal", **nominal_fit})
    for start in _grid_candidates(frequency, amplitude, nominal_hz, settings):
        fit = _fit(frequency, amplitude, start, settings)
        if fit is not None:
            candidates.append({"origin": "grid", **fit})
    if not candidates:
        return {"rpm": nominal, "shaft_hz": nominal_hz, "nominal_rpm": nominal,
                "method": "nominal_fallback", "fallback_reason": "no_harmonic_candidate",
                "relative_residual": None, "residual_hz": None, "harmonics_used": 0,
                "candidates": [], "sampling_rate_hz": fs, "samples": int(values.size)}
    candidates.sort(key=lambda item: (item["relative_residual"], -item["harmonics_used"],
                                      -item["harmonic_amplitude_sum"]))
    best = candidates[0]
    maximum = float(settings["maximum_relative_residual"])
    if best["relative_residual"] > maximum:
        return {"rpm": nominal, "shaft_hz": nominal_hz, "nominal_rpm": nominal,
                "method": "nominal_fallback", "fallback_reason": "residual_above_limit",
                "relative_residual": best["relative_residual"], "residual_hz": best["residual_hz"],
                "harmonics_used": best["harmonics_used"], "unused_estimate": best,
                "candidates": candidates[:8], "sampling_rate_hz": fs, "samples": int(values.size)}
    return {"rpm": best["rpm"], "shaft_hz": best["shaft_hz"], "nominal_rpm": nominal,
            "method": "harmonic_fit", "fallback_reason": None,
            "relative_residual": best["relative_residual"], "residual_hz": best["residual_hz"],
            "harmonics_used": best["harmonics_used"], "candidates": candidates[:8],
            "sampling_rate_hz": fs, "samples": int(values.size)}
