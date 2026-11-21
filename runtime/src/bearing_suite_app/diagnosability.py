"""诊断前硬门控：先确认信号与分析链足以支持“正常/故障”判定。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

FAMILIES = ("BPFO", "BPFI", "BSF")


def normalize_real_series(
    values: Sequence[float] | np.ndarray,
    *,
    field_name: str = "signal",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize one real-valued channel without silently merging channels."""

    report: dict[str, Any] = {
        "field": field_name,
        "valid": False,
        "original_shape": None,
        "normalized_shape": [0],
        "original_dtype": None,
        "action": "rejected",
        "reason": None,
    }
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError):
        report["reason"] = "unreadable_array"
        return np.asarray([], dtype=float), report

    report["original_shape"] = [int(size) for size in raw.shape]
    report["original_dtype"] = str(raw.dtype)
    if raw.ndim == 1:
        normalized = raw
        action = "none"
    elif raw.ndim == 2 and 1 in raw.shape:
        normalized = raw.reshape(-1)
        action = "squeezed_singleton_axis"
    else:
        report["reason"] = "not_single_channel"
        return np.asarray([], dtype=float), report

    if (
        np.issubdtype(normalized.dtype, np.bool_)
        or np.issubdtype(normalized.dtype, np.complexfloating)
        or not np.issubdtype(normalized.dtype, np.number)
    ):
        report["reason"] = "not_real_numeric"
        return np.asarray([], dtype=float), report

    result = normalized.astype(float, copy=False)
    report.update(
        {
            "valid": True,
            "normalized_shape": [int(result.size)],
            "action": action,
        }
    )
    return result, report


def _longest_repeated_run_ratio(values: np.ndarray, std_value: float) -> float:
    if values.size < 2:
        return 1.0 if values.size else 0.0
    tolerance = max(1e-12, std_value * 1e-9)
    repeated = np.abs(np.diff(values)) <= tolerance
    longest = current = 1
    for same in repeated:
        current = current + 1 if bool(same) else 1
        longest = max(longest, current)
    return float(longest / values.size)


def assess_input_quality(
    values: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    rpm: float,
    config: Mapping[str, Any],
    *,
    speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
    require_rpm: bool = True,
    input_normalization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """工业常用输入检查：有效性、长度、恒定、削顶、漂移与脱落。"""

    if input_normalization is None:
        x, normalization = normalize_real_series(values)
    else:
        x = np.asarray(values, dtype=float).reshape(-1)
        normalization = dict(input_normalization)
    try:
        fs = float(sampling_rate_hz)
    except (TypeError, ValueError):
        fs = math.nan
    try:
        rpm_value = float(rpm)
    except (TypeError, ValueError):
        rpm_value = math.nan

    finite_mask = np.isfinite(x)
    finite_ratio = float(np.mean(finite_mask)) if x.size else 0.0
    usable = x[finite_mask]
    sample_count = int(x.size)
    finite_sample_count = int(usable.size)
    duration = float(sample_count / fs) if math.isfinite(fs) and fs > 0 else 0.0
    revolutions = (
        float(duration * rpm_value / 60.0)
        if math.isfinite(rpm_value) and rpm_value > 0
        else 0.0
    )

    normalization_valid = bool(normalization.get("valid", False))
    min_samples = int(config.get("min_samples", 1024))
    invalid: list[str] = []
    if not normalization_valid:
        invalid.append(
            "invalid_signal_shape"
            if normalization.get("reason") == "not_single_channel"
            else "invalid_signal_dtype"
        )
    if not math.isfinite(fs) or fs <= 0:
        invalid.append("invalid_sampling_rate")
    if require_rpm and (not math.isfinite(rpm_value) or rpm_value <= 0):
        invalid.append("invalid_rpm")
    if normalization_valid and x.size < min_samples:
        invalid.append("record_too_short")
    if normalization_valid and finite_sample_count != sample_count:
        invalid.append("non_finite_samples")

    if usable.size:
        min_value = float(np.min(usable))
        max_value = float(np.max(usable))
        mean_value = float(np.mean(usable))
        std_value = float(np.std(usable))
        peak_to_peak = float(max_value - min_value)
        unique_value_count = int(np.unique(usable).size)
        centered = usable - mean_value
        rms = float(np.sqrt(np.mean(np.square(centered))))
        clipping_ratio = float(
            np.mean((usable == min_value) | (usable == max_value))
        )

        if usable.size >= 16 and std_value > 0:
            normalized_index = np.linspace(-1.0, 1.0, usable.size)
            slope, intercept = np.polyfit(normalized_index, usable, 1)
            trend = slope * normalized_index + intercept
            trend_ratio = float(np.std(trend) / (std_value + np.finfo(float).eps))
            dropout_run_ratio = _longest_repeated_run_ratio(usable, std_value)
        else:
            trend_ratio = 0.0
            dropout_run_ratio = 0.0
    else:
        min_value = max_value = mean_value = std_value = rms = 0.0
        peak_to_peak = 0.0
        unique_value_count = 0
        clipping_ratio = trend_ratio = dropout_run_ratio = 0.0

    constant_signal = bool(
        normalization_valid
        and finite_sample_count == sample_count
        and sample_count > 0
        and min_value == max_value
    )
    if constant_signal:
        invalid.append("constant_signal")
    if clipping_ratio > float(config.get("max_clipping_ratio", 0.05)):
        invalid.append("possible_clipping_plateau")
    if trend_ratio > float(config.get("max_trend_ratio", 0.35)):
        invalid.append("large_trend_or_sensor_drift")
    if dropout_run_ratio > float(config.get("max_dropout_run_ratio", 0.05)):
        invalid.append("possible_sensor_dropout_or_repeated_samples")

    speed_stability: dict[str, Any] = {"available": speed_trace_rpm is not None}
    if speed_trace_rpm is not None:
        speed, speed_normalization = normalize_real_series(
            speed_trace_rpm, field_name="speed_trace_rpm"
        )
        speed_stability["input_normalization"] = speed_normalization
        valid_speed = np.isfinite(speed) & (speed > 0)
        speed_stability["sample_count"] = int(speed.size)
        speed_stability["finite_positive_ratio"] = (
            float(np.mean(valid_speed)) if speed.size else 0.0
        )
        if (
            not speed_normalization["valid"]
            or speed.size < 2
            or not np.all(valid_speed)
        ):
            invalid.append("invalid_speed_trace")
        else:
            median_speed = float(np.median(speed))
            relative_span = float(
                (np.percentile(speed, 95) - np.percentile(speed, 5))
                / max(median_speed, np.finfo(float).eps)
            )
            speed_stability.update(
                {
                    "median_rpm": median_speed,
                    "relative_span_p05_p95": relative_span,
                }
            )
            reference_error = float(
                abs(median_speed - rpm_value)
                / max(median_speed, np.finfo(float).eps)
            )
            speed_stability["rpm_reference_relative_error"] = reference_error
            if relative_span > float(config.get("max_speed_relative_span", 0.02)):
                invalid.append("time_varying_speed_requires_order_tracking")
            if reference_error > float(
                config.get("max_speed_reference_relative_error", 0.02)
            ):
                invalid.append("rpm_reference_mismatch")

    invalid = list(dict.fromkeys(invalid))
    return {
        "evidence_id": "quality.signal",
        "ok": not invalid,
        "valid_for_analysis": not invalid,
        "finite": finite_sample_count == sample_count and sample_count > 0,
        "finite_ratio": finite_ratio,
        "sample_count": sample_count,
        "finite_sample_count": finite_sample_count,
        "min_samples_ok": x.size >= min_samples,
        "duration_seconds": duration,
        "revolutions": revolutions,
        "mean": mean_value,
        "std": std_value,
        "constant_signal": constant_signal,
        "peak_to_peak": peak_to_peak,
        "unique_value_count": unique_value_count,
        "rms": rms,
        "min": min_value,
        "max": max_value,
        "clipping_ratio": clipping_ratio,
        "trend_ratio": trend_ratio,
        "dropout_run_ratio": dropout_run_ratio,
        "speed_stability": speed_stability,
        "input_normalization": normalization,
        "invalidating_flags": invalid,
        "warning_flags": [],
    }


def assess_frequency_observability(
    fault_frequencies_hz: Mapping[str, float],
    sampling_rate_hz: float,
    sample_count: int,
    config: Mapping[str, Any],
    *,
    max_harmonics: int,
) -> dict[str, Any]:
    """确认三类目标频率、若干谐波与频率分辨率均可观测。"""

    fs = float(sampling_rate_hz)
    nyquist = fs / 2.0
    reliable_max = nyquist * float(config.get("reliable_nyquist_fraction", 0.95))
    df = float(fs / sample_count) if sample_count > 0 else math.inf
    required_harmonics = min(
        int(max_harmonics), int(config.get("min_observable_harmonics", 3))
    )
    max_relative_df = float(config.get("max_df_relative_to_fault", 0.05))
    families: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    duration = float(sample_count / fs) if fs > 0 else 0.0
    for family in FAMILIES:
        frequency = float(fault_frequencies_hz.get(family, 0.0) or 0.0)
        observable_harmonics = (
            min(int(max_harmonics), int(math.floor(reliable_max / frequency)))
            if frequency > 0
            else 0
        )
        relative_df = float(df / frequency) if frequency > 0 else math.inf
        reliable = (
            frequency > 0
            and observable_harmonics >= required_harmonics
            and relative_df <= max_relative_df
        )
        flags: list[str] = []
        if frequency <= 0 or frequency >= reliable_max:
            flags.append("fundamental_outside_reliable_range")
        if observable_harmonics < required_harmonics:
            flags.append("insufficient_observable_harmonics")
        if relative_df > max_relative_df:
            flags.append("coarse_frequency_resolution")
        if not reliable:
            invalid.append(f"{family}_not_reliably_observable")
        families[family] = {
            "theoretical_frequency_hz": frequency,
            "cycles_in_record": float(frequency * duration),
            "frequency_resolution_hz": df,
            "relative_resolution": relative_df,
            "observable_harmonics": observable_harmonics,
            "required_harmonics": required_harmonics,
            "reliable": reliable,
            "flags": flags,
        }

    highest_fault = max(
        (float(fault_frequencies_hz.get(family, 0.0) or 0.0) for family in FAMILIES),
        default=0.0,
    )
    resonance_search_floor = highest_fault * int(max_harmonics)
    carrier_search_available = resonance_search_floor < reliable_max
    if not carrier_search_available:
        invalid.append("no_resonance_carrier_search_range")
    return {
        "evidence_id": "diagnosability.frequency",
        "valid": not invalid,
        "frequency_resolution_hz": df,
        "analysis_range_hz": [0.0, reliable_max],
        "resonance_search_floor_hz": resonance_search_floor,
        "carrier_search_available": carrier_search_available,
        "families": families,
        "invalidating_flags": invalid,
    }


def assess_band_diagnosability(
    band_evidence: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """检查有效解调频带数量、能量和包络峰噪比。"""

    min_energy = float(config.get("min_band_energy_ratio", 1e-4))
    min_snr_db = float(config.get("min_envelope_snr_db", 8.0))
    min_valid_bands = int(config.get("min_valid_bands", 2))
    valid_ids: list[str] = []

    for band in band_evidence:
        energy_ratio = float(band.get("band_energy_ratio", 0.0) or 0.0)
        snr_db = float(band.get("envelope_snr_db", -math.inf) or -math.inf)
        enough_energy = energy_ratio >= min_energy
        enough_snr = snr_db >= min_snr_db
        valid = enough_energy and enough_snr
        band_flags: list[str] = []
        if not enough_energy:
            band_flags.append("band_energy_below_threshold")
        if not enough_snr:
            band_flags.append("envelope_snr_below_threshold")
        band["diagnosability"] = {
            "valid": valid,
            "enough_energy": enough_energy,
            "enough_envelope_snr": enough_snr,
            "invalidating_flags": band_flags,
        }
        if valid:
            valid_ids.append(str(band.get("evidence_id", "")))

    invalid: list[str] = []
    if len(valid_ids) < min_valid_bands:
        invalid.append("insufficient_valid_demodulation_bands")
    return {
        "evidence_id": "diagnosability.bands",
        "valid": not invalid,
        "valid_band_count": len(valid_ids),
        "required_valid_band_count": min_valid_bands,
        "valid_band_evidence_ids": valid_ids,
        "thresholds": {
            "min_band_energy_ratio": min_energy,
            "min_envelope_snr_db": min_snr_db,
        },
        "invalidating_flags": invalid,
    }


def gate_summary(
    *,
    input_quality: Mapping[str, Any],
    frequency_observability: Mapping[str, Any] | None = None,
    band_diagnosability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """合并三阶段门控，提供稳定的 unknown_quality_limited 原因。"""

    stages = {
        "input_quality": dict(input_quality),
        "frequency_observability": dict(frequency_observability or {}),
        "band_diagnosability": dict(band_diagnosability or {}),
    }
    invalid: list[str] = []
    invalid.extend(input_quality.get("invalidating_flags", []))
    if frequency_observability:
        invalid.extend(frequency_observability.get("invalidating_flags", []))
    if band_diagnosability:
        invalid.extend(band_diagnosability.get("invalidating_flags", []))
    invalid = list(dict.fromkeys(str(flag) for flag in invalid))
    return {
        "evidence_id": "diagnosability.gate",
        "passed": not invalid,
        "diagnosis_code_on_failure": "unknown_quality_limited",
        "invalidating_flags": invalid,
        "stages": stages,
    }
