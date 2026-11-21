"""S1: descriptive signal-state assessment (not a fault diagnosis)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from bearing_suite_app import diagnosability as diag_gate
from bearing_suite_app import geometry as geom

from .band_banks import select_envelope_bands
from .domain_indicators import frequency_domain, time_domain, time_frequency


def _state(
    quality: Mapping[str, Any],
    time_stats: Mapping[str, Any],
    freq_stats: Mapping[str, Any],
    tf_stats: Mapping[str, Any],
    frequency_gate: Mapping[str, Any],
    sk_band_count: int,
) -> dict[str, Any]:
    snr = float(freq_stats.get("local_peak_snr_db", freq_stats.get("spectral_peak_snr_db")) or 0.0)
    narrow = int(freq_stats.get("narrowband_peak_count") or 0)
    kurtosis = float(time_stats.get("kurtosis") or 0.0)
    modulation = float(time_stats.get("modulation_index") or 0.0)
    nonstationarity = float(tf_stats.get("nonstationarity_index") or 0.0)
    if not quality.get("valid_for_analysis", False):
        signal_type = "low_information"
    elif narrow >= 6 and snr < 10.0:
        signal_type = "interference_dominated"
    elif kurtosis >= 6.0 or nonstationarity > 0.5:
        signal_type = "transient"
    elif nonstationarity > 0.3:
        signal_type = "nonstationary"
    else:
        signal_type = "stationary"
    return {
        "signal_type": signal_type,
        "noise_level": "high" if snr < 3.0 else "medium" if snr < 10.0 else "low",
        "impulsiveness": "high" if kurtosis >= 6.0 else "medium" if kurtosis >= 4.0 else "low",
        "narrowband_interference": "strong"
        if narrow >= 6 and snr < 10.0
        else "mild"
        if narrow >= 4
        else "none",
        "modulation": "clear" if modulation >= 0.3 else "weak" if modulation >= 0.1 else "none",
        "speed_stability": (quality.get("speed_stability") or {}).get("label", "unknown")
        if isinstance(quality.get("speed_stability"), Mapping)
        else "unknown",
        "diagnosability": {
            "fault_frequencies_observable": bool(frequency_gate.get("valid", False)),
            "valid_band_count": int(sk_band_count),
            "limitations": list(frequency_gate.get("flags") or []),
        },
    }


def assess_signal(
    signal: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    rpm: float,
    geometry: Mapping[str, Any] | None,
    quality_cfg: Mapping[str, Any],
    observability_cfg: Mapping[str, Any],
    *,
    select_bands: bool = True,
) -> dict[str, Any]:
    """Return a fixed signal-state profile; never a fault family/evidence level."""

    x, normalization = diag_gate.normalize_real_series(signal)
    fs = float(sampling_rate_hz)
    rpm_value = float(rpm)
    quality = diag_gate.assess_input_quality(
        x, fs, rpm_value, quality_cfg, input_normalization=normalization
    )
    duration = x.size / fs if fs > 0 else 0.0
    revolutions = duration * rpm_value / 60.0 if rpm_value > 0 else 0.0
    min_revolutions = float(quality_cfg.get("min_revolutions", 5.0))
    if revolutions < min_revolutions:
        quality["valid_for_analysis"] = False
        flags = list(quality.get("invalidating_flags", []))
        flags.append("insufficient_revolutions")
        quality["invalidating_flags"] = flags
    quality["revolutions"] = round(revolutions, 4)

    ratios = geom.fault_frequency_ratios(geometry)
    frequencies = geom.fault_frequencies_hz(rpm_value, ratios)
    frequency_gate = diag_gate.assess_frequency_observability(
        frequencies, fs, int(x.size), observability_cfg, max_harmonics=5
    )
    max_fault = max((float(v) for v in frequencies.values() if float(v) > 0), default=1.0)
    min_band = max(50.0, 5.0 * max_fault)

    time_stats = time_domain(x, fs)
    freq_stats = frequency_domain(x, fs)
    tf_stats = time_frequency(x, fs, min_band_hz=min_band)
    sk_bands = tf_stats["spectral_kurtosis_bands_hz"]
    spectral = {"spectral_kurtosis_bands_hz": sk_bands}
    band_count = len(sk_bands)
    if select_bands and quality["valid_for_analysis"] and frequency_gate["valid"]:
        selection = select_envelope_bands(x, fs, min_band)
        spectral["envelope_band_selection"] = selection
        band_count = len(selection["selected"])
    state = _state(quality, time_stats, freq_stats, tf_stats, frequency_gate, band_count)

    reasons = []
    preprocess = []
    if not quality.get("valid_for_analysis", False):
        preprocess.append("stop_quality_gate")
        reasons.append("quality gate failed")
    if state["narrowband_interference"] in {"mild", "strong"}:
        preprocess.append("adaptive_notch")
        reasons.append("narrowband peaks present")
    if state["impulsiveness"] in {"medium", "high"}:
        preprocess.append("bandpass/med")
        reasons.append("impulsive content present")
    if not frequency_gate.get("valid", False):
        preprocess.append("limited")
        reasons.append("fault frequency not observable")
    if not preprocess:
        preprocess.append("fixed envelope baseline")
        reasons.append("no strong preprocessing need detected")

    status = "invalid" if not quality.get("valid_for_analysis", False) else "ok"
    if status == "ok" and not frequency_gate.get("valid", False):
        status = "limited"
    limitations = list(quality.get("invalidating_flags") or [])
    if not frequency_gate.get("valid", False):
        limitations.extend(frequency_gate.get("flags") or [])
    return {
        "schema_version": "signal-assessment/v1",
        "status": status,
        "meta": {
            "sampling_rate_hz": fs,
            "duration_seconds": round(duration, 6),
            "sample_count": int(x.size),
            "nyquist_hz": round(fs / 2.0, 6),
            "frequency_resolution_hz": round(fs / max(1, x.size), 6),
            "rpm": rpm_value,
            "revolutions": round(revolutions, 4),
        },
        "quality": quality,
        "time_domain": time_stats,
        "frequency_domain": freq_stats,
        "time_frequency": tf_stats,
        "state": state,
        "recommendations": {
            "preprocess": preprocess,
            "avoid": [],
            "reason": reasons,
        },
        "limitations": list(dict.fromkeys(limitations)),
        # Backward-compatible aliases used by the planner/processing plan.
        "noise": {
            "spectral_peak_snr_estimate_db": freq_stats.get("local_peak_snr_db", freq_stats["spectral_peak_snr_db"]),
            "spectral_peak_snr_full_band_db": freq_stats["spectral_peak_snr_db"],
            "noise_floor": freq_stats.get("local_noise_floor", freq_stats["noise_floor"]),
            "narrowband_peak_count": freq_stats["narrowband_peak_count"],
            "dominant_peaks": freq_stats["dominant_peaks"][:5],
        },
        "impulsiveness": {
            "rms": time_stats["rms"],
            "peak": time_stats["peak"],
            "kurtosis": time_stats["kurtosis"],
            "crest_factor": time_stats["crest_factor"],
            "impulse_factor": time_stats["impulse_factor"],
        },
        "spectral": spectral,
        "fault_observability": {
            "fault_frequencies_hz": {k: round(float(v), 6) for k, v in frequencies.items()},
            "gate": frequency_gate,
        },
        "recommended_processing": preprocess,
    }
