"""Layer 1 assessment tools: descriptive time/frequency/time-frequency state.

These tools do not compute fault families, fault confidence or envelope matches.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from bearing_suite_app.assessment.domain_indicators import (
    frequency_domain,
    time_domain,
    time_frequency,
)

from .contracts import ToolContext, ToolResult, ToolSpec, timed
from .registry import ToolRegistry, parameter


def _result(spec: ToolSpec, data: dict[str, Any], params: dict[str, Any], metrics: dict[str, float]) -> ToolResult:
    return ToolResult(
        tool=spec.name,
        version=spec.version,
        layer=spec.layer,
        status="ok",
        params=params,
        data=data,
        metrics=metrics,
        evidence_id=f"L1.{spec.name}",
    )


@timed
def time_domain_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    spec = _SPECS["assess_time_domain"]
    values = np.asarray(x, dtype=float).reshape(-1)
    data = time_domain(values, context.sampling_rate_hz)
    return _result(
        spec,
        data,
        params,
        {"kurtosis": float(data["kurtosis"]), "crest_factor": float(data["crest_factor"])},
    )


@timed
def frequency_domain_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    spec = _SPECS["assess_frequency_domain"]
    values = np.asarray(x, dtype=float).reshape(-1)
    data = frequency_domain(values, context.sampling_rate_hz, n_peaks=int(params.get("n_peaks", 8)))
    return _result(
        spec,
        data,
        params,
        {
            "spectral_peak_snr_db": float(data["spectral_peak_snr_db"]),
            "spectral_entropy": float(data["spectral_entropy"]),
        },
    )


@timed
def time_frequency_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    spec = _SPECS["assess_time_frequency"]
    values = np.asarray(x, dtype=float).reshape(-1)
    min_band = max(50.0, 5.0 * context.max_fault_frequency_hz)
    data = time_frequency(
        values,
        context.sampling_rate_hz,
        min_band_hz=min_band,
        n_bands=int(params.get("n_bands", 4)),
    )
    return _result(
        spec,
        data,
        params,
        {"nonstationarity_index": float(data["nonstationarity_index"])},
    )


_SPECS: dict[str, ToolSpec] = {
    "assess_time_domain": ToolSpec(
        name="assess_time_domain",
        version="1.0.0",
        layer="L1",
        family="assessment",
        description="Descriptive time-domain state: amplitude, impulsiveness, stationarity, modulation.",
        output_fields=("time_domain",),
    ),
    "assess_frequency_domain": ToolSpec(
        name="assess_frequency_domain",
        version="1.0.0",
        layer="L1",
        family="assessment",
        description="Descriptive frequency-domain state: peaks, SNR, entropy, band energy.",
        parameters=(
            parameter("n_peaks", "integer", "number of dominant spectral peaks", default=8, minimum=3, maximum=20),
        ),
        output_fields=("frequency_domain",),
    ),
    "assess_time_frequency": ToolSpec(
        name="assess_time_frequency",
        version="1.0.0",
        layer="L1",
        family="assessment",
        description="Descriptive time-frequency state: nonstationarity, transients, spectral-kurtosis bands.",
        parameters=(
            parameter("n_bands", "integer", "number of spectral-kurtosis bands", default=4, minimum=1, maximum=8),
        ),
        output_fields=("time_frequency",),
    ),
}


def register_assessment(registry: ToolRegistry) -> None:
    registry.register(_SPECS["assess_time_domain"], time_domain_tool)
    registry.register(_SPECS["assess_frequency_domain"], frequency_domain_tool)
    registry.register(_SPECS["assess_time_frequency"], time_frequency_tool)
