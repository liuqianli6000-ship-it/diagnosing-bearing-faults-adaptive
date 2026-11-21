"""诊断主链的可视化证据输出。

本模块只消费 ``chain.analyze_envelope_chain`` 本次已经用于谐波匹配的谱数组，
不重新滤波、不重新计算包络谱，也不产生诊断标签。PNG 是可审计派生证据；数值
证据仍是裁决依据。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bearing_suite_app.assessment.band_banks import BAND_METHOD_NAMES

FAMILY_COLORS = {
    "BPFO": "#D55E00",
    "BPFI": "#0072B2",
    "BSF": "#009E73",
}
GENERATOR_VERSION = "bearing-visual/v2"


def _configure_chinese_font() -> None:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    preferred = next(
        (
            name
            for name in ("Noto Sans SC", "Microsoft YaHei", "SimHei", "SimSun")
            if name in installed
        ),
        "DejaVu Sans",
    )
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [preferred, "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _band_label(band_id: str) -> str:
    method, separator, suffix = band_id.partition(".band.")
    if separator and method in BAND_METHOD_NAMES:
        return f"{BAND_METHOD_NAMES[method]} · 频带 {suffix}"
    match = re.fullmatch(r"band\.(\d+)", band_id)
    return f"频带 {match.group(1)}" if match else band_id


def _safe_run_id(value: str | None) -> str:
    if value is None:
        return uuid.uuid4().hex[:16]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe[:80] or uuid.uuid4().hex[:16]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"不是有效 PNG：{path}")
    return struct.unpack(">II", header[16:24])


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _qualified_peak_frequencies(spectrum: Mapping[str, Any]) -> list[float]:
    frequencies: list[float] = []
    families = spectrum.get("families", {})
    if not isinstance(families, Mapping):
        return frequencies
    for family_data in families.values():
        if not isinstance(family_data, Mapping):
            continue
        for peak in family_data.get("harmonic_peaks", []):
            if isinstance(peak, Mapping) and peak.get("qualified"):
                value = peak.get("observed_frequency_hz")
                if value is not None and np.isfinite(float(value)):
                    frequencies.append(float(value))
    return frequencies


def _downsample_minmax(
    frequencies_hz: np.ndarray,
    amplitudes: np.ndarray,
    *,
    end_hz: float,
    max_points: int,
    preserve_frequencies_hz: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """按横轴桶保留局部极小/极大值，并强制保留已合格峰所在点。"""

    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    values = np.asarray(amplitudes, dtype=float).reshape(-1)
    if frequencies.size != values.size:
        raise ValueError("可视化谱的频率与幅值数组长度不一致")
    mask = (
        np.isfinite(frequencies)
        & np.isfinite(values)
        & (frequencies >= 0.0)
        & (frequencies <= float(end_hz))
    )
    frequencies = frequencies[mask]
    values = values[mask]
    if frequencies.size <= max(16, int(max_points)):
        return frequencies, values

    bucket_count = max(1, int(max_points) // 2)
    edges = np.linspace(0, frequencies.size, bucket_count + 1, dtype=int)
    retained: set[int] = {0, frequencies.size - 1}
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        local = values[start:stop]
        retained.add(start + int(np.argmin(local)))
        retained.add(start + int(np.argmax(local)))
    for target in preserve_frequencies_hz:
        if 0.0 <= float(target) <= float(end_hz):
            retained.add(int(np.argmin(np.abs(frequencies - float(target)))))
    indices = np.asarray(sorted(retained), dtype=int)
    return frequencies[indices], values[indices]


def save_signal_analysis_plot(
    values: np.ndarray, fs: float, domain: str, path: Path, *, unit: str,
) -> None:
    """Render the same signal and spectral transforms used by the descriptive tools."""
    from .assessment.domain_indicators import amplitude_spectrum, stft_spectrum

    _configure_chinese_font()
    figure, axis = plt.subplots(figsize=(8, 3.8), layout="constrained")
    amplitude_label = f"幅值（{unit}）" if unit else "幅值"
    try:
        if domain == "time":
            x = np.arange(values.size) / fs
            y = values
            axis.set(title="时域波形", xlabel="时间（s）", ylabel=amplitude_label)
        elif domain == "frequency":
            x, y = amplitude_spectrum(values, fs)
            axis.set(title="幅值频谱", xlabel="频率（Hz）", ylabel=amplitude_label)
        elif domain == "time_frequency":
            frequencies, times, coefficients = stft_spectrum(values, fs)
            magnitude = np.abs(coefficients)
            relative_db = 20 * np.log10(np.maximum(magnitude / max(float(magnitude.max()), np.finfo(float).tiny), 1e-4))
            mesh = axis.pcolormesh(times, frequencies, relative_db, shading="auto", cmap="magma", vmin=-80, vmax=0)
            axis.set(title="STFT 时频图", xlabel="时间（s）", ylabel="频率（Hz）",
                     xlim=(0, values.size / fs), ylim=(0, fs / 2))
            figure.colorbar(mesh, ax=axis, label="相对幅值（dB，以图中峰值为 0 dB）")
        else:
            raise ValueError(f"Unsupported analysis plot: {domain}")
        if domain in ("time", "frequency"):
            x, y = _downsample_minmax(x, y, end_hz=float(x[-1]), max_points=2500, preserve_frequencies_hz=[])
            axis.plot(x, y, color="#4268d5", linewidth=0.8)
            axis.set_xlim(0, x[-1])
            axis.grid(alpha=0.2)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, facecolor="white")
    finally:
        plt.close(figure)


def _plot_end_hz(
    spectra: Sequence[Mapping[str, Any]],
    fault_frequencies_hz: Mapping[str, float],
    max_harmonics: int,
) -> float:
    available = min(
        float(np.asarray(spectrum["frequencies_hz"], dtype=float).reshape(-1)[-1])
        for spectrum in spectra
    )
    targets = [
        float(value) * max(1, int(max_harmonics))
        for value in fault_frequencies_hz.values()
        if np.isfinite(float(value)) and float(value) > 0.0
    ]
    if not targets:
        return available
    return min(available, max(targets) * 1.15)


def _source_evidence_ids(spectra: Sequence[Mapping[str, Any]]) -> list[str]:
    ids: set[str] = set()
    for spectrum in spectra:
        if spectrum.get("band_id"):
            ids.add(str(spectrum["band_id"]))
        families = spectrum.get("families", {})
        if not isinstance(families, Mapping):
            continue
        for family_data in families.values():
            if not isinstance(family_data, Mapping):
                continue
            if family_data.get("evidence_id"):
                ids.add(str(family_data["evidence_id"]))
            for peak in family_data.get("harmonic_peaks", []):
                if (
                    isinstance(peak, Mapping)
                    and peak.get("qualified")
                    and peak.get("evidence_id")
                ):
                    ids.add(str(peak["evidence_id"]))
    return sorted(ids)


def save_multiband_envelope_evidence(
    spectra: Sequence[Mapping[str, Any]],
    fault_frequencies_hz: Mapping[str, float],
    output_root: str | Path,
    *,
    run_id: str | None = None,
    max_harmonics: int = 5,
    max_plot_points: int = 2000,
    dpi: int = 160,
    metadata: Mapping[str, Any] | None = None,
    independent_view: bool = False,
) -> tuple[list[dict[str, Any]], str | None]:
    """保存多频带包络谱 PNG 及 JSON 清单，返回轻量证据描述。"""

    if not spectra:
        return [], None

    resolved_root = Path(output_root).expanduser().resolve()
    resolved_run_id = _safe_run_id(run_id)
    run_dir = (resolved_root / resolved_run_id).resolve()
    if resolved_root != run_dir and resolved_root not in run_dir.parents:
        raise ValueError("可视化输出目录越界")
    run_dir.mkdir(parents=True, exist_ok=False)

    plot_end_hz = _plot_end_hz(spectra, fault_frequencies_hz, max_harmonics)
    if not np.isfinite(plot_end_hz) or plot_end_hz <= 0.0:
        raise ValueError("包络谱没有可绘制的正频率范围")

    _configure_chinese_font()
    figure, axes = plt.subplots(
        len(spectra),
        1,
        sharex=True,
        figsize=(12.0, max(3.2, 2.35 * len(spectra))),
        squeeze=False,
    )
    displayed_points: dict[str, int] = {}
    temporary_image: Path | None = None
    try:
        for axis, spectrum in zip(axes[:, 0], spectra):
            preserve = [] if independent_view else _qualified_peak_frequencies(spectrum)
            frequencies, amplitudes = _downsample_minmax(
                np.asarray(spectrum["frequencies_hz"], dtype=float),
                np.asarray(spectrum["amplitudes"], dtype=float),
                end_hz=plot_end_hz,
                max_points=max_plot_points,
                preserve_frequencies_hz=preserve,
            )
            if frequencies.size == 0:
                raise ValueError("包络谱在目标范围内没有有限数据")
            absolute = np.abs(amplitudes)
            scale = float(np.percentile(absolute, 99.5)) if absolute.size else 0.0
            if scale <= np.finfo(float).eps:
                scale = max(float(np.max(absolute)), np.finfo(float).eps)
            normalized = amplitudes / scale
            band_id = str(spectrum.get("band_id", "band"))
            displayed_points[band_id] = int(frequencies.size)
            axis.plot(frequencies, normalized, color="#252525", linewidth=0.8)
            axis.axhline(0.0, color="#888888", linewidth=0.55, alpha=0.6)

            families = spectrum.get("families", {})
            for family, color in FAMILY_COLORS.items():
                fundamental = float(fault_frequencies_hz.get(family, 0.0))
                if fundamental <= 0.0:
                    continue
                for harmonic in range(1, max(1, int(max_harmonics)) + 1):
                    target = harmonic * fundamental
                    if target > plot_end_hz:
                        break
                    axis.axvline(
                        target,
                        color=color,
                        linestyle="--",
                        linewidth=0.65,
                        alpha=0.34,
                    )
                if independent_view:
                    continue
                family_data = (
                    families.get(family, {})
                    if isinstance(families, Mapping)
                    else {}
                )
                if not isinstance(family_data, Mapping):
                    continue
                for peak in family_data.get("harmonic_peaks", []):
                    if not isinstance(peak, Mapping) or not peak.get("qualified"):
                        continue
                    observed = float(peak["observed_frequency_hz"])
                    if observed > plot_end_hz:
                        continue
                    index = int(np.argmin(np.abs(frequencies - observed)))
                    axis.scatter(
                        [frequencies[index]],
                        [normalized[index]],
                        s=24,
                        marker="o",
                        color=color,
                        edgecolors="white",
                        linewidths=0.55,
                        zorder=4,
                    )

            band = spectrum.get("band_hz", [0.0, 0.0])
            title = (
                f"{_band_label(band_id)}｜解调载频带 "
                f"{float(band[0]):.0f}–{float(band[1]):.0f} Hz"
            )
            axis.set_title(
                title if independent_view else
                f"{title}｜包络信噪比 "
                f"{float(spectrum.get('envelope_snr_db', 0.0)):.1f} dB｜"
                f"能量比 {float(spectrum.get('band_energy_ratio', 0.0)):.3g}",
                loc="left",
                fontsize=9,
            )
            axis.set_ylabel("幅值")
            axis.grid(True, color="#D9D9D9", linewidth=0.45, alpha=0.55)
            axis.set_xlim(0.0, plot_end_hz)

        legend = [
            Line2D([0], [0], color="#252525", linewidth=1.0, label="包络残差")
        ]
        for family, color in FAMILY_COLORS.items():
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle="--",
                    label=f"{family} 理论频率",
                )
            )
        if not independent_view:
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color="#555555",
                    marker="o",
                    linestyle="None",
                    markeredgecolor="white",
                    label="合格实测峰",
                )
            )
        axes[0, 0].legend(handles=legend, loc="upper right", fontsize=8, ncol=3)
        axes[-1, 0].set_xlabel("包络频率（Hz）")
        figure.suptitle(
            "多频带包络谱（各频带独立归一化）",
            fontsize=12,
            y=0.997,
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))

        image_path = run_dir / "envelope_multiband.png"
        temporary_image = run_dir / f".{image_path.name}.{uuid.uuid4().hex}.tmp"
        figure.savefig(
            temporary_image,
            format="png",
            dpi=max(72, int(dpi)),
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary_image, image_path)
    finally:
        plt.close(figure)
        if temporary_image is not None and temporary_image.exists():
            temporary_image.unlink()

    width, height = _png_size(image_path)
    artifact = {
        "evidence_id": "visual.envelope.multiband",
        "artifact_type": "multiband_envelope_spectrum",
        "mime_type": "image/png",
        "path": str(image_path.resolve()),
        "sha256": _sha256(image_path),
        "width": width,
        "height": height,
        "source_evidence_ids": _source_evidence_ids(spectra),
        "processing": {
            "generator_version": GENERATOR_VERSION,
            "display_language": "zh-CN",
            "independent_visual_view": independent_view,
            "spectrum_source": "current_diagnosis_chain_arrays",
            "envelope_mode": str(spectra[0].get("mode", "squared")),
            "window": str(spectra[0].get("window", "hann")),
            "baseline_removed": True,
            "plot_downsampling": "min_max" if independent_view else "min_max_with_qualified_peak_preservation",
            "max_plot_points_per_band": max(16, int(max_plot_points)),
            "displayed_points_by_band": displayed_points,
            "independent_band_normalization": True,
            "displayed_frequency_range_hz": [0.0, float(plot_end_hz)],
            "max_harmonics": max(1, int(max_harmonics)),
        },
        "role": "audit_only_not_independent_diagnostic_evidence",
    }
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "schema_version": "bearing-visual-evidence/v1",
        "run_id": resolved_run_id,
        "metadata": dict(metadata or {}),
        "artifacts": [artifact],
    }
    _atomic_json(manifest_path, manifest)
    return [artifact], str(manifest_path.resolve())
