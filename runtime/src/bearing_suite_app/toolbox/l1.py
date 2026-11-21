"""Layer 1: raw-signal cleaning tools.

Only cleaning happens here.  No fault family, no matching and no diagnosis is
computed in this layer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from bearing_suite_app import signal_tools

from .contracts import ToolContext, ToolResult, ToolSpec, timed
from .registry import ToolRegistry, parameter


def _stats(x: np.ndarray) -> dict[str, float]:
    values = np.asarray(x, dtype=float).reshape(-1)
    std = float(np.std(values))
    mean = float(np.mean(values))
    kurtosis = (
        float(np.mean((values - mean) ** 4) / (std**4)) if std > 0 else 0.0
    )
    return {
        "mean": mean,
        "std": std,
        "kurtosis": kurtosis,
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
    }


def _result(
    spec: ToolSpec,
    *,
    status: str = "ok",
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    signal: np.ndarray | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
    error_code: str | None = None,
    evidence_id: str | None = None,
) -> ToolResult:
    return ToolResult(
        tool=spec.name,
        version=spec.version,
        layer=spec.layer,
        status=status,
        params=params or {},
        data=data or {},
        metrics=metrics or {},
        arrays={"signal": signal} if signal is not None else {},
        warnings=warnings or [],
        error=error,
        error_code=error_code,
        evidence_id=evidence_id,
    )


@timed
def detrend_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["detrend"]
    order = int(params.get("order", 1))
    values = np.asarray(x, dtype=float).reshape(-1)
    if values.size < max(8, order + 2):
        return _result(spec, status="rejected", params=params, error="signal too short")
    if order <= 0:
        cleaned = values - float(np.mean(values))
    elif order == 1:
        cleaned = scipy_signal.detrend(values, type="linear")
    else:
        # scipy.signal.detrend only supports constant/linear; use least squares
        # for higher-order polynomial trends.
        t = np.arange(values.size, dtype=float)
        coefficients = np.polyfit(t, values, order)
        cleaned = values - np.polyval(coefficients, t)
    stats = _stats(cleaned)
    residual = abs(float(np.mean(cleaned)))
    return _result(
        spec,
        params=params,
        data={"order": order, **stats},
        metrics={"objective": -residual, "residual_mean": residual, **stats},
        signal=cleaned,
        evidence_id="L1.detrend",
    )


@timed
def outlier_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["outlier_remove"]
    threshold = float(params.get("threshold", 6.0))
    values = np.asarray(x, dtype=float).reshape(-1)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(values))
    if scale <= 0:
        return _result(spec, params=params, data={"clipped": 0}, metrics={"objective": 0.0}, signal=values)
    z = np.abs(values - median) / scale
    mask = z > threshold
    cleaned = values.copy()
    cleaned[mask] = median
    ratio = float(np.mean(mask))
    stats = _stats(cleaned)
    return _result(
        spec,
        params=params,
        data={"clipped_count": int(np.sum(mask)), "clipped_ratio": ratio, **stats},
        metrics={"objective": -ratio, "outlier_ratio": ratio, **stats},
        signal=cleaned,
        evidence_id="L1.outlier",
    )


@timed
def normalize_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["normalize"]
    mode = str(params.get("mode", "zscore"))
    values = np.asarray(x, dtype=float).reshape(-1)
    if mode == "zscore":
        std = float(np.std(values))
        cleaned = (values - float(np.mean(values))) / std if std > 0 else values - float(np.mean(values))
    elif mode == "rms":
        rms = float(np.sqrt(np.mean(values**2)))
        cleaned = values / rms if rms > 0 else values
    else:
        cleaned = values
    stats = _stats(cleaned)
    return _result(
        spec,
        params=params,
        data={"mode": mode, **stats},
        metrics={"objective": stats["std"], **stats},
        signal=cleaned,
        evidence_id="L1.normalize",
    )


@timed
def bandpass_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["bandpass"]
    nyq = context.sampling_rate_hz / 2.0
    low = float(params.get("low_hz", 5.0))
    high = float(params.get("high_hz", nyq * 0.95))
    order = int(params.get("order", 4))
    if not 0 < low < high <= nyq:
        return _result(spec, status="rejected", params=params,
                       error=f"带通频带须满足 0 < low_hz < high_hz <= Nyquist ({nyq:g} Hz)")
    cleaned = signal_tools.bandpass_filter(x, context.sampling_rate_hz, low, high, order=order)
    stats = _stats(cleaned)
    return _result(
        spec,
        params=params,
        data={"low_hz": low, "high_hz": high, "order": order, **stats},
        metrics={"objective": stats["kurtosis"], **stats},
        signal=cleaned,
        evidence_id="L1.bandpass",
    )


def _bandpass_candidates(context: ToolContext) -> list[dict[str, Any]]:
    nyq = context.sampling_rate_hz / 2.0
    max_fault = max(1.0, context.max_fault_frequency_hz)
    return [
        {"low_hz": 5.0, "high_hz": min(nyq * 0.95, max(2000.0, 10 * max_fault)), "order": 4},
        {"low_hz": 5.0, "high_hz": nyq * 0.95, "order": 4},
        {
            "low_hz": max(5.0, 0.5 * max_fault),
            "high_hz": min(nyq * 0.95, max(3000.0, 20 * max_fault)),
            "order": 4,
        },
    ]


_SPECS: dict[str, ToolSpec] = {
    "detrend": ToolSpec(
        name="detrend",
        version="1.0.0",
        layer="L1",
        family="preprocess",
        description="Remove DC/linear/polynomial trend; no fault features.",
        parameters=(
            parameter("order", "int", "0=mean, 1=linear, 2=polynomial", default=1, minimum=0, maximum=3, adaptive=True),
        ),
        output_fields=("signal", "mean", "std", "kurtosis"),
        candidates_fn=lambda context: [{"order": 0}, {"order": 1}, {"order": 2}],
    ),
    "outlier_remove": ToolSpec(
        name="outlier_remove",
        version="1.0.0",
        layer="L1",
        family="preprocess",
        description="Clip robust outliers using median absolute deviation.",
        parameters=(
            parameter("threshold", "float", "robust z threshold", default=6.0, minimum=2.0, maximum=12.0, adaptive=True),
        ),
        output_fields=("signal", "clipped_count", "clipped_ratio"),
        candidates_fn=lambda context: [{"threshold": 4.5}, {"threshold": 6.0}, {"threshold": 8.0}],
    ),
    "normalize": ToolSpec(
        name="normalize",
        version="1.0.0",
        layer="L1",
        family="preprocess",
        description="Amplitude normalization; does not change spectral shape.",
        parameters=(
            parameter("mode", "enum", "zscore/rms/none", default="zscore", candidates=("zscore", "rms", "none"), adaptive=True),
        ),
        output_fields=("signal", "mode", "std"),
        candidates_fn=lambda context: [{"mode": "zscore"}, {"mode": "rms"}, {"mode": "none"}],
    ),
    "bandpass": ToolSpec(
        name="bandpass",
        version="1.0.0",
        layer="L1",
        family="preprocess",
        description="Zero-phase Butterworth band-pass for a broad analysis band.",
        parameters=(
            parameter("low_hz", "float", "low cutoff below Nyquist", default=5.0, unit="Hz", adaptive=True),
            parameter("high_hz", "float", "high cutoff at or below Nyquist", default=3000.0, unit="Hz", adaptive=True),
            parameter("order", "int", "filter order", default=4, minimum=1, maximum=8),
        ),
        output_fields=("signal", "low_hz", "high_hz", "kurtosis"),
        candidates_fn=_bandpass_candidates,
    ),
}


def register_l1(registry: ToolRegistry) -> None:
    registry.register(_SPECS["detrend"], detrend_tool)
    registry.register(_SPECS["outlier_remove"], outlier_tool)
    registry.register(_SPECS["normalize"], normalize_tool)
    registry.register(_SPECS["bandpass"], bandpass_tool)
