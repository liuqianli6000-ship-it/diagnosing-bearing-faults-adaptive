"""Descriptive time/frequency/time-frequency indicators.

This module measures the state of the signal. It does not compute fault
families, fault confidence or envelope matching.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import signal as scipy_signal

from bearing_suite_app import signal_tools


def _finite(x: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(x, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    return values if values.size else np.zeros(1, dtype=float)


def time_domain(x: Sequence[float] | np.ndarray, fs: float) -> dict[str, Any]:
    values = _finite(x)
    mean = float(np.mean(values))
    std = float(np.std(values))
    rms = float(np.sqrt(np.mean(values**2)))
    peak = float(np.max(np.abs(values)))
    abs_mean = float(np.mean(np.abs(values)))
    p2p = float(np.ptp(values))
    kurtosis = float(np.mean((values - mean) ** 4) / (std**4)) if std > 0 else 0.0
    skewness = float(np.mean((values - mean) ** 3) / (std**3)) if std > 0 else 0.0
    envelope = np.abs(scipy_signal.hilbert(values))
    env_mean = float(np.mean(envelope))
    modulation = float(np.std(envelope) / env_mean) if env_mean > 0 else 0.0
    zero_cross = float(np.mean(np.diff(np.signbit(values)) != 0)) if values.size > 1 else 0.0
    segments = np.array_split(values, min(4, values.size)) if values.size else []
    segment_rms = [float(np.sqrt(np.mean(seg**2))) for seg in segments if seg.size]
    segment_mean = float(np.mean(segment_rms)) if segment_rms else 0.0
    rms_cv = float(np.std(segment_rms) / segment_mean) if segment_mean > 0 else 0.0
    if rms_cv > 0.5:
        stationarity_label = "transient"
    elif rms_cv > 0.15:
        stationarity_label = "drifting"
    else:
        stationarity_label = "stable"
    t = np.arange(values.size, dtype=float)
    slope = float(np.polyfit(t, values, 1)[0]) if values.size > 2 else 0.0
    trend = "none"
    if std > 0 and abs(slope) * values.size > 0.1 * std:
        trend = "increasing" if slope > 0 else "decreasing"
    return {
        "mean": round(mean, 8),
        "std": round(std, 8),
        "rms": round(rms, 8),
        "peak": round(peak, 8),
        "peak_to_peak": round(p2p, 8),
        "crest_factor": round(peak / rms, 6) if rms > 0 else 0.0,
        "impulse_factor": round(peak / abs_mean, 6) if abs_mean > 0 else 0.0,
        "clearance_factor": round(peak / (float(np.mean(np.sqrt(np.abs(values)))) ** 2), 6)
        if abs_mean > 0
        else 0.0,
        "skewness": round(skewness, 6),
        "kurtosis": round(kurtosis, 6),
        "zero_crossing_rate": round(zero_cross, 8),
        "modulation_index": round(modulation, 6),
        "stationarity": {
            "label": stationarity_label,
            "segment_rms_cv": round(rms_cv, 6),
            "trend": trend,
        },
    }


def amplitude_spectrum(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    taper = np.hanning(values.size)
    amplitudes = np.abs(np.fft.rfft(values * taper)) / (values.size * max(float(np.mean(taper)), 1e-12))
    amplitudes[1 : -1 if values.size % 2 == 0 else None] *= 2.0
    freqs = np.fft.rfftfreq(values.size, d=1.0 / fs)
    return freqs, amplitudes


def frequency_domain(
    x: Sequence[float] | np.ndarray,
    fs: float,
    *,
    n_peaks: int = 8,
) -> dict[str, Any]:
    values = _finite(x)
    freqs, amps = amplitude_spectrum(values, fs)
    total_energy = float(np.sum(amps**2))
    if amps.size:
        indices = np.argsort(amps)[-max(1, int(n_peaks)) :][::-1]
    else:
        indices = np.array([], dtype=int)
    peak_level = float(amps[indices[0]]) if indices.size else 0.0
    peaks = []
    for index in indices:
        peaks.append(
            {
                "frequency_hz": round(float(freqs[index]), 4),
                "amplitude": round(float(amps[index]), 8),
                "relative_amplitude": round(float(amps[index] / peak_level), 6) if peak_level > 0 else 0.0,
            }
        )
    noise_floor = float(np.median(amps)) if amps.size else 0.0
    snr_db = 20.0 * np.log10((peak_level + 1e-12) / (noise_floor + 1e-12)) if peak_level > 0 else 0.0
    local_noise = noise_floor
    local_snr = snr_db
    if indices.size:
        peak_index = int(indices[0])
        half = max(5, min(50, amps.size // 20))
        low = max(0, peak_index - half)
        high = min(amps.size, peak_index + half + 1)
        neighborhood = np.concatenate(
            [amps[low : max(low, peak_index - 3)], amps[min(high, peak_index + 4) : high]]
        )
        if neighborhood.size:
            local_noise = float(np.median(neighborhood))
            local_snr = (
                20.0 * np.log10((peak_level + 1e-12) / (local_noise + 1e-12))
                if peak_level > 0
                else 0.0
            )
    energy = amps**2
    centroid = float(np.sum(freqs * energy) / total_energy) if total_energy > 0 else 0.0
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * energy) / total_energy)) if total_energy > 0 else 0.0
    probabilities = energy / total_energy if total_energy > 0 else np.zeros_like(energy)
    entropy = float(-np.sum(probabilities * np.log2(probabilities + 1e-12)) / np.log2(max(2, probabilities.size)))
    nyq = fs / 2.0
    edges = np.linspace(0.0, nyq, num=9)
    band_energy = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        ratio = float(np.sum(energy[mask]) / total_energy) if total_energy > 0 else 0.0
        band_energy.append(
            {"low_hz": round(float(low), 3), "high_hz": round(float(high), 3), "ratio": round(ratio, 6)}
        )
    narrow = [item for item in peaks if item["frequency_hz"] >= 20.0]
    return {
        "dominant_peaks": peaks,
        "spectral_centroid_hz": round(centroid, 4),
        "spectral_spread_hz": round(spread, 4),
        "spectral_entropy": round(entropy, 6),
        "spectral_peak_snr_db": round(float(snr_db), 3),
        "local_peak_snr_db": round(float(local_snr), 3),
        "noise_floor": noise_floor,
        "local_noise_floor": local_noise,
        "narrowband_peak_count": len(narrow),
        "band_energy": band_energy,
        "harmonic_series": [],
    }


def stft_spectrum(values: np.ndarray, fs: float):
    nperseg = min(1024, max(64, values.size // 16))
    return scipy_signal.stft(values, fs=fs, nperseg=nperseg)


def time_frequency(
    x: Sequence[float] | np.ndarray,
    fs: float,
    *,
    min_band_hz: float = 50.0,
    n_bands: int = 4,
) -> dict[str, Any]:
    values = _finite(x)
    stft_status = "ok"
    try:
        _, _, stft = stft_spectrum(values, fs)
        frame_rms = np.sqrt(np.mean(np.abs(stft) ** 2, axis=0))
    except Exception:  # noqa: BLE001
        stft_status = "fallback"
        frame_rms = np.array([float(np.sqrt(np.mean(values**2)))])
    frame_mean = float(np.mean(frame_rms)) if frame_rms.size else 0.0
    nonstationarity = float(np.std(frame_rms) / frame_mean) if frame_mean > 0 else 0.0
    threshold = frame_mean + 2.0 * float(np.std(frame_rms)) if frame_rms.size else 0.0
    transient_count = int(np.sum(frame_rms > threshold)) if threshold > 0 else 0
    scores = signal_tools.kurtosis_band_scores(values, fs, min_band_hz=min_band_hz, n_edges=10) if n_bands else []
    selected = scores[: max(1, int(n_bands))]
    total_energy = float(np.sum(values**2))
    time_band_energy = []
    for item in selected:
        low, high = item["band_hz"]
        filtered = signal_tools.bandpass_filter(values, fs, low, high)
        ratio = float(np.sum(filtered**2) / total_energy) if total_energy > 0 else 0.0
        time_band_energy.append({"band_hz": [round(low, 3), round(high, 3)], "energy_ratio": round(ratio, 6)})
    concentration = max((item["energy_ratio"] for item in time_band_energy), default=0.0)
    return {
        "method": "stft+kurtogram" if n_bands else "stft",
        "stft_status": stft_status,
        "nonstationarity_index": round(nonstationarity, 6),
        "transient_count": transient_count,
        "energy_concentration": round(float(concentration), 6),
        "spectral_kurtosis_bands_hz": [item["band_hz"] for item in selected],
        "spectral_kurtosis_scores": [item["kurtosis"] for item in selected],
        "time_band_energy": time_band_energy,
    }
