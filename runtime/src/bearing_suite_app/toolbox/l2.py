"""Layer 2: mechanism features.

FFT, envelope spectrum, spectral kurtosis, time-domain indicators, theoretical
fault frequencies, harmonic matching, EMD and wavelet decomposition.  Results
are evidence, not a final diagnosis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as scipy_signal

from bearing_suite_app import geometry as geom
from bearing_suite_app import signal_tools

from .contracts import ToolContext, ToolResult, ToolSpec, timed
from .l1 import _result, _stats
from .registry import ToolRegistry, parameter


def _top_peaks(
    frequencies: np.ndarray, amplitudes: np.ndarray, *, count: int = 8
) -> list[dict[str, float]]:
    values = np.asarray(amplitudes, dtype=float)
    if values.size == 0:
        return []
    idx = np.argsort(values)[-count:][::-1]
    return [
        {
            "frequency_hz": round(float(frequencies[i]), 4),
            "amplitude": round(float(values[i]), 8),
        }
        for i in idx
        if np.isfinite(values[i])
    ]


def _fault_energy(
    freqs: np.ndarray,
    amps: np.ndarray,
    context: ToolContext,
    *,
    harmonics: int = 3,
    tolerance_ratio: float = 0.03,
) -> tuple[float, list[float], dict[str, float]]:
    total = 0.0
    matched: list[float] = []
    by_family: dict[str, float] = {}
    freqs = np.asarray(freqs, dtype=float)
    amps = np.asarray(amps, dtype=float)
    for family, base in context.fault_frequencies_hz.items():
        base = float(base)
        if base <= 0:
            continue
        family_total = 0.0
        for order in range(1, harmonics + 1):
            target = base * order
            tolerance = max(
                context.sampling_rate_hz / 4096.0,
                target * tolerance_ratio,
            )
            mask = (freqs >= target - tolerance) & (freqs <= target + tolerance)
            if np.any(mask):
                value = float(np.max(amps[mask]))
                family_total += value
                if value > 0:
                    matched.append(round(float(freqs[mask][np.argmax(amps[mask])]), 3))
        by_family[family] = family_total
        total += family_total
    return total, matched, by_family


def _envelope(
    x: np.ndarray,
    context: ToolContext,
    *,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    mode: str = "squared",
) -> dict[str, Any]:
    band = None
    if band_low_hz is not None and band_high_hz is not None and band_high_hz > band_low_hz:
        band = [float(band_low_hz), float(band_high_hz)]
    return signal_tools.envelope_spectrum(
        x,
        context.sampling_rate_hz,
        band_hz=band,
        mode=mode,
        window="hann",
    )


@timed
def fft_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    spec = _SPECS["fft_spectrum"]
    values = np.asarray(x, dtype=float).reshape(-1)
    fs = context.sampling_rate_hz
    taper = scipy_signal.windows.hann(values.size, sym=False)
    amps = np.abs(np.fft.rfft(values * taper)) / (values.size * float(np.mean(taper)))
    amps[1 : -1 if values.size % 2 == 0 else None] *= 2.0
    freqs = np.fft.rfftfreq(values.size, d=1.0 / fs)
    top = _top_peaks(freqs, amps)
    return ToolResult(
        tool=spec.name,
        version=spec.version,
        layer=spec.layer,
        status="ok",
        params=params,
        data={"frequency_resolution_hz": round(fs / values.size, 6), "top_peaks": top},
        metrics={
            "objective": float(np.max(amps)) if amps.size else 0.0,
            "peak_amplitude": float(np.max(amps)) if amps.size else 0.0,
        },
        arrays={"frequencies_hz": freqs, "amplitudes": amps},
        evidence_id="L2.fft",
    )


@timed
def envelope_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    spec = _SPECS["envelope_spectrum"]
    result = _envelope(
        x,
        context,
        band_low_hz=params.get("band_low_hz"),
        band_high_hz=params.get("band_high_hz"),
        mode=str(params.get("mode", "squared")),
    )
    freqs = result["frequencies_hz"]
    amps = result["amplitudes"]
    energy, matched, by_family = _fault_energy(freqs, amps, context)
    return ToolResult(
        tool=spec.name,
        version=spec.version,
        layer=spec.layer,
        status="ok",
        params=params,
        data={
            "band_hz": result.get("band_hz"),
            "envelope_snr_db": round(float(result.get("envelope_snr_db", 0.0)), 3),
            "top_peaks": _top_peaks(freqs, amps),
            "matched_fault_frequencies_hz": matched[:12],
        },
        metrics={
            "objective": float(energy),
            "fault_energy": float(energy),
            "envelope_snr_db": float(result.get("envelope_snr_db", 0.0)),
            "frequency_resolution_hz": float(result.get("frequency_resolution_hz", 0.0)),
        },
        arrays={"frequencies_hz": freqs, "amplitudes": amps},
        evidence_id="L2.envelope",
    )


@timed
def spectral_kurtosis_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["spectral_kurtosis"]
    bands = signal_tools.select_bands_by_kurtosis(
        x,
        context.sampling_rate_hz,
        min_band_hz=float(params.get("min_band_hz", 50.0)),
        n_bands=int(params.get("n_bands", 4)),
        n_edges=int(params.get("n_edges", 10)),
    )
    scores = [
        {
            "low_hz": round(float(low), 3),
            "high_hz": round(float(high), 3),
            "bandwidth_hz": round(float(high - low), 3),
        }
        for low, high in bands
    ]
    return _result(
        spec,
        params=params,
        data={"bands": scores},
        metrics={
            "objective": float(len(bands)),
            "band_count": float(len(bands)),
            "best_bandwidth_hz": float(scores[0]["bandwidth_hz"]) if scores else 0.0,
        },
        evidence_id="L2.spectral_kurtosis",
    )


@timed
def time_stats_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["time_stats"]
    values = np.asarray(x, dtype=float).reshape(-1)
    stats = _stats(values)
    mean = stats["mean"]
    std = stats["std"]
    peak = stats["peak"]
    rms = float(np.sqrt(np.mean(values**2)))
    abs_mean = float(np.mean(np.abs(values)))
    crest = peak / rms if rms > 0 else 0.0
    impulse = peak / abs_mean if abs_mean > 0 else 0.0
    clearance = peak / float(np.mean(np.sqrt(np.abs(values)))) ** 2 if abs_mean > 0 else 0.0
    return _result(
        spec,
        params=params,
        data={
            "rms": round(rms, 8),
            "peak": round(peak, 8),
            "kurtosis": round(stats["kurtosis"], 6),
            "crest_factor": round(crest, 6),
            "impulse_factor": round(impulse, 6),
            "clearance_factor": round(clearance, 6),
        },
        metrics={
            "objective": stats["kurtosis"],
            "rms": rms,
            "kurtosis": stats["kurtosis"],
            "crest_factor": crest,
            "impulse_factor": impulse,
        },
        evidence_id="L2.time_stats",
    )


@timed
def fault_frequency_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["fault_frequency"]
    ratios = geom.fault_frequency_ratios(context.geometry)
    hz = geom.fault_frequencies_hz(context.rpm, ratios)
    return _result(
        spec,
        params=params,
        data={
            "ratios": {key: round(float(value), 6) for key, value in ratios.items()},
            "fault_frequencies_hz": {key: round(float(value), 6) for key, value in hz.items()},
        },
        metrics={"objective": float(max(hz.values())) if hz else 0.0},
        evidence_id="L2.fault_frequency",
    )


@timed
def harmonic_match_tool(
    x: np.ndarray, context: ToolContext, params: dict[str, Any]
) -> ToolResult:
    spec = _SPECS["harmonic_match"]
    envelope = _envelope(x, context)
    freqs = envelope["frequencies_hz"]
    amps = envelope["amplitudes"]
    harmonics = int(params.get("max_harmonics", 5))
    tolerance_ratio = float(params.get("tolerance_ratio", 0.03))
    energy, matched, by_family = _fault_energy(
        freqs, amps, context, harmonics=harmonics, tolerance_ratio=tolerance_ratio
    )
    best = max(by_family, key=by_family.get) if by_family else None
    return _result(
        spec,
        params=params,
        data={
            "best_family": best,
            "family_energy": {key: round(float(value), 8) for key, value in by_family.items()},
            "matched_frequencies_hz": matched[:16],
        },
        metrics={
            "objective": float(energy),
            "fault_energy": float(energy),
            "best_family_energy": float(by_family.get(best, 0.0)) if best else 0.0,
        },
        evidence_id="L2.harmonic_match",
    )


@timed
def emd_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    spec = _SPECS["emd_decompose"]
    try:
        from PyEMD import EMD
    except ImportError:
        return _result(spec, status="unavailable", params=params, error="EMD-signal is not installed", error_code="dependency_missing")
    values = np.asarray(x, dtype=float).reshape(-1)
    max_samples = int(params.get("max_samples", 8192))
    segment = values[: max(256, min(values.size, max_samples))]
    max_imf = int(params.get("max_imf", 6))
    decomposer = EMD()
    imfs = decomposer.emd(segment, max_imf=max_imf)
    if imfs.ndim == 1:
        imfs = imfs.reshape(1, -1)
    energies = []
    for index in range(len(imfs)):
        envelope = _envelope(imfs[index], context)
        score, _, _ = _fault_energy(
            envelope["frequencies_hz"], envelope["amplitudes"], context
        )
        energies.append(float(score))
    if not energies:
        return _result(spec, status="error", params=params, error="EMD returned no IMF", error_code="empty_decomposition")
    threshold = float(np.mean(energies)) + 0.5 * float(np.std(energies))
    selected = [i for i, score in enumerate(energies) if score >= threshold]
    if not selected:
        selected = [int(np.argmax(energies))]
    enhanced = np.sum(imfs[selected], axis=0)[: segment.size]
    return _result(
        spec,
        params=params,
        data={
            "algorithm_parameters": {name: getattr(decomposer, name) for name in (
                "spline_kind", "nbsym", "extrema_detection", "std_thr", "svar_thr", "total_power_thr",
                "range_thr", "energy_ratio_thr", "MAX_ITERATION", "FIXE", "FIXE_H")},
            "imf_count": len(energies),
            "selected_imfs": selected,
            "imf_energies": [round(value, 8) for value in energies],
            "processed_samples": int(segment.size),
            "input_samples": int(values.size),
            "truncated": bool(segment.size < values.size),
            "selection_threshold": threshold,
        },
        metrics={
            "objective": float(sum(energies[i] for i in selected)),
            "fault_energy": float(sum(energies[i] for i in selected)),
            "imf_count": float(len(energies)),
        },
        signal=enhanced,
        evidence_id="L2.emd",
    )


@timed
def wavelet_tool(x: np.ndarray, context: ToolContext, params: dict[str, Any]) -> ToolResult:
    spec = _SPECS["wavelet_decompose"]
    try:
        import pywt
    except ImportError:
        return _result(spec, status="unavailable", params=params, error="PyWavelets is not installed", error_code="dependency_missing")
    values = np.asarray(x, dtype=float).reshape(-1)
    max_samples = int(params.get("max_samples", 65536))
    segment = values[: max(512, min(values.size, max_samples))]
    wavelet = str(params.get("wavelet", "db4"))
    level = int(params.get("level", 4))
    coefficients = pywt.wavedec(segment, wavelet, mode="symmetric", level=level)
    energies = []
    for part in coefficients:
        component = pywt.waverec(
            [np.zeros_like(coefficient) if coefficient is not part else coefficient for coefficient in coefficients],
            wavelet,
            mode="symmetric",
        )[: segment.size]
        envelope = _envelope(component, context)
        score, _, _ = _fault_energy(
            envelope["frequencies_hz"], envelope["amplitudes"], context
        )
        energies.append(float(score))
    threshold = float(np.mean(energies)) + 0.5 * float(np.std(energies))
    selected = [i for i, score in enumerate(energies) if score >= threshold]
    if not selected:
        selected = [int(np.argmax(energies))]
    reconstruction = [
        coefficient if index in selected else np.zeros_like(coefficient)
        for index, coefficient in enumerate(coefficients)
    ]
    enhanced = pywt.waverec(reconstruction, wavelet, mode="symmetric")[: segment.size]
    return _result(
        spec,
        params=params,
        data={
            "algorithm_parameters": {"wavelet": wavelet, "level": level, "extension_mode": "symmetric"},
            "wavelet": wavelet,
            "level": level,
            "selected_coefficients": selected,
            "coefficient_energies": [round(value, 8) for value in energies],
            "processed_samples": int(segment.size),
            "input_samples": int(values.size),
            "truncated": bool(segment.size < values.size),
            "selection_threshold": threshold,
        },
        metrics={
            "objective": float(sum(energies[i] for i in selected)),
            "fault_energy": float(sum(energies[i] for i in selected)),
        },
        signal=enhanced,
        evidence_id="L2.wavelet",
    )


def _envelope_candidates(context: ToolContext) -> list[dict[str, Any]]:
    nyq = context.sampling_rate_hz / 2.0
    fallback_low = max(50.0, context.max_fault_frequency_hz * 2)
    fallback_high = min(nyq * 0.95, max(2000.0, context.max_fault_frequency_hz * 15))
    return [
        {"band_low_hz": None, "band_high_hz": None, "mode": "squared"},
        {"band_low_hz": fallback_low, "band_high_hz": fallback_high, "mode": "squared"},
        {
            "band_low_hz": max(50.0, context.max_fault_frequency_hz * 3),
            "band_high_hz": min(nyq * 0.95, max(3000.0, context.max_fault_frequency_hz * 20)),
            "mode": "squared",
        },
    ]


_SPECS: dict[str, ToolSpec] = {
    "fft_spectrum": ToolSpec(
        name="fft_spectrum", version="1.0.0", layer="L2", family="feature",
        description="Hann-windowed FFT amplitude spectrum and top peaks.",
        parameters=(),
        output_fields=("frequencies_hz", "amplitudes", "top_peaks"),
    ),
    "envelope_spectrum": ToolSpec(
        name="envelope_spectrum", version="1.0.0", layer="L2", family="feature",
        description="Hilbert envelope spectrum for one demodulation band.",
        parameters=(
            parameter("band_low_hz", "float", "demodulation low cutoff", default=None, unit="Hz", minimum=0.0, maximum=100000.0, adaptive=True),
            parameter("band_high_hz", "float", "demodulation high cutoff", default=None, unit="Hz", minimum=1.0, maximum=100000.0, adaptive=True),
            parameter("mode", "enum", "ordinary/squared", default="squared", candidates=("ordinary", "squared")),
        ),
        output_fields=("frequencies_hz", "amplitudes", "top_peaks", "matched_fault_frequencies_hz"),
        candidates_fn=lambda context: [
            {"band_low_hz": None, "band_high_hz": None, "mode": "squared"},
            {"band_low_hz": max(50.0, context.max_fault_frequency_hz * 2), "band_high_hz": context.sampling_rate_hz / 2 * 0.6, "mode": "squared"},
            {"band_low_hz": max(50.0, context.max_fault_frequency_hz * 3), "band_high_hz": context.sampling_rate_hz / 2 * 0.8, "mode": "squared"},
        ],
    ),
    "spectral_kurtosis": ToolSpec(
        name="spectral_kurtosis", version="1.0.0", layer="L2", family="feature",
        description="Spectral-kurtosis band search.",
        parameters=(
            parameter("min_band_hz", "float", "minimum band width", default=50.0, unit="Hz", minimum=1.0, maximum=10000.0),
            parameter("n_bands", "int", "number of returned bands", default=4, minimum=1, maximum=8),
            parameter("n_edges", "int", "log-frequency edges", default=10, minimum=4, maximum=20),
        ),
        output_fields=("bands",),
    ),
    "time_stats": ToolSpec(
        name="time_stats", version="1.0.0", layer="L2", family="feature",
        description="Time-domain RMS, peak, kurtosis, crest/impulse/clearance.",
        parameters=(),
        output_fields=("rms", "peak", "kurtosis", "crest_factor", "impulse_factor"),
    ),
    "fault_frequency": ToolSpec(
        name="fault_frequency", version="1.0.0", layer="L2", family="feature",
        description="Theoretical fault frequencies from geometry and rpm.",
        parameters=(),
        output_fields=("ratios", "fault_frequencies_hz"),
    ),
    "harmonic_match": ToolSpec(
        name="harmonic_match", version="1.0.0", layer="L2", family="feature",
        description="Match envelope peaks against fault-frequency harmonics.",
        parameters=(
            parameter("max_harmonics", "int", "harmonic orders", default=5, minimum=1, maximum=10, adaptive=True),
            parameter("tolerance_ratio", "float", "relative tolerance", default=0.03, minimum=0.005, maximum=0.1, adaptive=True),
        ),
        output_fields=("best_family", "family_energy", "matched_frequencies_hz"),
        candidates_fn=lambda context: [
            {"max_harmonics": 3, "tolerance_ratio": 0.02},
            {"max_harmonics": 5, "tolerance_ratio": 0.03},
            {"max_harmonics": 6, "tolerance_ratio": 0.05},
        ],
    ),
    "emd_decompose": ToolSpec(
        name="emd_decompose", version="1.0.0", layer="L2", family="decomposition",
        description="EMD decomposition with fault-energy IMF selection.",
        parameters=(
            parameter("max_imf", "int", "maximum IMFs", default=6, minimum=2, maximum=12, adaptive=True),
            parameter("max_samples", "int", "analysis window", default=8192, minimum=512, maximum=65536),
        ),
        output_fields=("imf_count", "selected_imfs", "imf_energies"),
        candidates_fn=lambda context: [{"max_imf": 4}, {"max_imf": 6}, {"max_imf": 8}],
        error_codes=("dependency_missing", "empty_decomposition"),
    ),
    "wavelet_decompose": ToolSpec(
        name="wavelet_decompose", version="1.0.0", layer="L2", family="decomposition",
        description="Wavelet decomposition with fault-energy coefficient selection.",
        parameters=(
            parameter("wavelet", "enum", "wavelet basis", default="db4", candidates=("db4", "db8", "sym8")),
            parameter("level", "int", "decomposition level", default=4, minimum=2, maximum=8, adaptive=True),
            parameter("max_samples", "int", "analysis window", default=65536, minimum=1024, maximum=262144),
        ),
        output_fields=("wavelet", "level", "selected_coefficients"),
        candidates_fn=lambda context: [{"wavelet": "db4", "level": 3}, {"wavelet": "db4", "level": 4}, {"wavelet": "db8", "level": 4}],
        error_codes=("dependency_missing",),
    ),
}


def register_l2(registry: ToolRegistry) -> None:
    registry.register(_SPECS["fft_spectrum"], fft_tool)
    registry.register(_SPECS["envelope_spectrum"], envelope_tool)
    registry.register(_SPECS["spectral_kurtosis"], spectral_kurtosis_tool)
    registry.register(_SPECS["time_stats"], time_stats_tool)
    registry.register(_SPECS["fault_frequency"], fault_frequency_tool)
    registry.register(_SPECS["harmonic_match"], harmonic_match_tool)
    registry.register(_SPECS["emd_decompose"], emd_tool)
    registry.register(_SPECS["wavelet_decompose"], wavelet_tool)
