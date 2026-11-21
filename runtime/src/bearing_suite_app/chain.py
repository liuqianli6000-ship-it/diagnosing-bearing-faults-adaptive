"""稳态主链：多频带包络谱 → 逐族谐波匹配 → 汇总带内成组倍频。

``agent`` 与 ``features`` 共用此实现，保证测量口径一致：
   - 频带由调用方传入（谱峭度选带，见 signal_tools.select_resonance_bands）；
- 每个频带做平方 Hilbert 包络谱，对 BPFO/BPFI/BSF 逐族做谐波匹配；
- 基频命中、成组匹配与拟合支持分别记录，证据等级须有成组匹配和同带拟合支持；
- 基频碰撞按理论频率间距判定（两族基频差 ≤ 容差和 → 碰撞，不得升级）；
- 提供转频时再做轴频谐波退化判定：基频与轴频整数倍谐波潜在重合的族，
  按配置要求更高的跨带一致带数（见 decision.shaft_degenerate_families）。
"""

from __future__ import annotations

import collections
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import decision
from .diagnosability import assess_band_diagnosability
from . import matcher
from . import signal_tools as sig

FAMILY_ORDER = matcher.FAMILY_ORDER


def family_tolerance_fn(
    family: str, fundamental_hz: float, resolution_hz: float, matching: Mapping[str, Any]
) -> Callable[[int], float]:
    """某个故障族的谐波容差函数（matching 参数来自 mfd_method.json）。"""

    slip = matching["slip_relative_uncertainty_by_family"].get(family, 0.01)

    def tolerance(harmonic: int) -> float:
        return matcher.harmonic_tolerance_hz(
            fundamental_hz,
            harmonic,
            resolution_hz,
            speed_relative_uncertainty=matching["speed_relative_uncertainty"],
            geometry_relative_uncertainty=matching["geometry_relative_uncertainty"],
            slip_relative_uncertainty=slip,
            resolution_multiplier=matching["tolerance_resolution_multiplier"],
            minimum_tolerance_hz=matching["minimum_tolerance_hz"],
        )

    return tolerance


def _lattice_for_family(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    family: str,
    fundamental_hz: float,
    frequencies_hz: Mapping[str, float],
    matching: Mapping[str, Any],
    shaft_hz: float | None,
    decision_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """某族在某频带上的栅格模型选择结果（主判据）。

    调制频率按物理取：内圈用轴频（载荷区调制），滚动体用 FTF（保持架调制），
    外圈无调制。轴频参考缺失时不做干扰模型比较。
    """

    modulation = None
    if family == "BPFI":
        modulation = shaft_hz
    elif family == "BSF":
        modulation = float(frequencies_hz.get("FTF", 0.0) or 0.0) or None

    evidence = matcher.lattice_evidence(
        frequencies,
        amplitudes,
        fundamental_hz,
        max_harmonics=int(matching["max_harmonics"]),
        shaft_hz=shaft_hz,
        modulation_hz=modulation,
    )

    # 细化后的频率回代逐峰判据：数据内估计的频率吸收参考误差（实测载荷工况偏低
    # 1.5%~1.7%），而逐峰局部显著性用窄窗保住特异性。两者都不需要按数据集标定的容差。
    fitted_hz = float(evidence["lattice_frequency_hz"] or 0.0)
    refined_orders: list[int] = []
    refined_used = False
    # 细化后的频率必须**在同一批阶次上明显更显著**才值得替代理论值：否则细化只是
    # 在搜索范围内挑到一个偶然峰（实测 1200rpm 正常样本的 BSF 会锁到 −5% 的偶然位置）。
    # 判据用局部 z 的均值差，单位为 z，与轴承/数据集无关。
    gain_min = float(decision_cfg.get("refined_gain_min", 3.0))
    refined_gain = 0.0
    if (
        fitted_hz > 0
        and math.isfinite(fitted_hz)
        and 0.95 * fundamental_hz <= fitted_hz <= 1.05 * fundamental_hz
        and not evidence["fitted_is_interference"]
        and abs(fitted_hz / fundamental_hz - 1.0) > 1e-3
    ):
        theory_z = matcher.comb_local_z(
            frequencies,
            amplitudes,
            fundamental_hz,
            max_harmonics=int(matching["max_harmonics"]),
            background_radius_bins=int(matching["background_radius_bins"]),
        )["mean_z"]
        refined_z = matcher.comb_local_z(
            frequencies,
            amplitudes,
            fitted_hz,
            max_harmonics=int(matching["max_harmonics"]),
            background_radius_bins=int(matching["background_radius_bins"]),
        )["mean_z"]
        refined_gain = float(refined_z - theory_z)
        if refined_gain >= gain_min:
            refined = matcher.match_harmonics(
                frequencies,
                amplitudes,
                fitted_hz,
                max_harmonics=int(matching["max_harmonics"]),
                tolerance_hz_fn=family_tolerance_fn(
                    family, fitted_hz, float(np.median(np.diff(frequencies))), matching
                ),
                min_relative_amplitude=float(matching["min_relative_amplitude"]),
                min_robust_z=float(matching["min_robust_z"]),
                background_radius_bins=int(matching["background_radius_bins"]),
            )
            refined_orders = list(refined.get("qualified_orders", []))
            refined_used = True

    return {
        "best_model": evidence["best_model"],
        "sideband_order": evidence["sideband_order"],
        "fault_supported": bool(evidence["fault_supported"]),
        "refined_frequency_used": refined_used,
        "refined_gain": refined_gain,
        "refined_qualified_orders": refined_orders,
        "refined_fundamental_hit": bool(1 in refined_orders),
        "fitted_is_interference": bool(evidence["fitted_is_interference"]),
        "fitted_frequency_hz": evidence["lattice_frequency_hz"],
        "frequency_shift_ratio": (
            float(evidence["lattice_frequency_hz"] / fundamental_hz - 1.0)
            if fundamental_hz > 0 and evidence["lattice_frequency_hz"] == evidence["lattice_frequency_hz"]
            else 0.0
        ),
        "residual_rms_bins": evidence["residual_rms_bins"],
        "null_rms_bins": evidence["null_rms_bins"],
        "accept_threshold_bins": evidence["accept_threshold_bins"],
        "effective_line_count": evidence["effective_line_count"],
    }


def _empty_match() -> dict[str, Any]:
    return {
        "harmonic_peaks": [],
        "qualified_orders": [],
        "qualified_match_count": 0,
        "has_low_order_anchor": False,
    }


def analyze_envelope_chain(
    x: np.ndarray,
    fs: float,
    frequencies_hz: Mapping[str, float],
    bands: Sequence[Sequence[float]],
    *,
    matching: Mapping[str, Any],
    decision_cfg: Mapping[str, Any],
    diagnosability_cfg: Mapping[str, Any] | None = None,
    envelope_mode: str = "squared",
    envelope_window: str | None = "hann",
    retain_spectra: bool = False,
    shaft_hz: float | None = None,
    band_method: str | None = None,
) -> dict[str, Any]:
    """跑一遍稳态主链，返回跨频带证据测量（不裁决、不输出标签）。

    ``retain_spectra`` 仅供可视化审计层复用本次匹配所用的原始数组；默认关闭，
    避免在普通诊断路径中长期持有大数组。保留的数组不会写入 EvidenceBundle 或
    LLM 文本载荷。

    ``shaft_hz`` 提供转频时启用"轴频谐波退化"判定：基频与轴频整数倍谐波潜在重合的
    族，按 ``decision_cfg`` 的 ``strict_min_bands_by_family`` 要求更高的跨带一致性。
    为 None 时不做该判定，行为与既有路径一致。
    """

    prefix = f"{band_method}." if band_method else ""
    resolution = 0.0
    input_rms = float(np.sqrt(np.mean(np.square(x - float(np.mean(x))))))
    band_matches: list[tuple[tuple[float, float], dict[str, dict]]] = []
    band_evidence: list[dict[str, Any]] = []
    spectrum_artifacts: list[dict[str, Any]] = []
    for band_index, band in enumerate(bands, start=1):
        band_id = f"{prefix}band.{band_index}"
        env = sig.envelope_spectrum(
            x, fs, band_hz=list(band), mode=envelope_mode, window=envelope_window
        )
        if resolution <= 0:
            resolution = env["frequency_resolution_hz"]
        matches: dict[str, dict] = {}
        for family in FAMILY_ORDER:
            fundamental = frequencies_hz.get(family, 0.0)
            if fundamental <= 0:
                matches[family] = _empty_match()
                matches[family]["evidence_id"] = f"{band_id}.{family}"
                continue
            matches[family] = matcher.match_harmonics(
                env["frequencies_hz"],
                env["amplitudes"],
                fundamental,
                max_harmonics=matching["max_harmonics"],
                tolerance_hz_fn=family_tolerance_fn(
                    family, fundamental, env["frequency_resolution_hz"], matching
                ),
                min_relative_amplitude=matching["min_relative_amplitude"],
                min_robust_z=matching["min_robust_z"],
                background_radius_bins=matching["background_radius_bins"],
            )
            matches[family]["evidence_id"] = f"{band_id}.{family}"
            for peak in matches[family].get("harmonic_peaks", []):
                peak["evidence_id"] = (
                    f"{band_id}.{family}.harmonic.{int(peak.get('harmonic', 0))}"
                )
            matches[family]["lattice"] = _lattice_for_family(
                env["frequencies_hz"],
                env["amplitudes"],
                family,
                fundamental,
                frequencies_hz,
                matching,
                shaft_hz,
                decision_cfg,
            )
        band_matches.append((tuple(band), matches))
        band_energy_ratio = float(
            (float(env["filtered_rms"]) / max(input_rms, np.finfo(float).eps)) ** 2
        )
        band_evidence.append(
            {
                "evidence_id": band_id,
                "band_hz": [float(band[0]), float(band[1])],
                "frequency_resolution_hz": float(env["frequency_resolution_hz"]),
                "filtered_rms": float(env["filtered_rms"]),
                "band_energy_ratio": band_energy_ratio,
                "envelope_snr_db": float(env["envelope_snr_db"]),
                "families": matches,
            }
        )
        if retain_spectra:
            spectrum_artifacts.append(
                {
                    "band_id": band_id,
                    "band_hz": [float(band[0]), float(band[1])],
                    "frequencies_hz": env["frequencies_hz"],
                    "amplitudes": env["amplitudes"],
                    "frequency_resolution_hz": float(
                        env["frequency_resolution_hz"]
                    ),
                    "filtered_rms": float(env["filtered_rms"]),
                    "band_energy_ratio": band_energy_ratio,
                    "envelope_snr_db": float(env["envelope_snr_db"]),
                    "families": matches,
                    "mode": str(env["mode"]),
                    "window": str(env["window"]),
                }
            )

    band_gate = None
    if diagnosability_cfg is not None:
        band_gate = assess_band_diagnosability(band_evidence, diagnosability_cfg)
        band_matches = [
            entry for entry, evidence in zip(band_matches, band_evidence, strict=True)
            if evidence["diagnosability"]["valid"]
        ]

    fundamental_bands: dict[str, int] = {family: 0 for family in FAMILY_ORDER}
    evidence_bands: dict[str, int] = {family: 0 for family in FAMILY_ORDER}
    fit_supported_bands: dict[str, int] = {family: 0 for family in FAMILY_ORDER}
    consensus_hz: dict[str, float | None] = {family: None for family in FAMILY_ORDER}
    spread_bins: dict[str, float | None] = {family: None for family in FAMILY_ORDER}
    union_orders: dict[str, set[int]] = {family: set() for family in FAMILY_ORDER}
    for _, matches in band_matches:
        for family in FAMILY_ORDER:
            orders = matches[family].get("qualified_orders", [])
            union_orders[family] |= set(orders)
            if 1 in orders:
                fundamental_bands[family] += 1

    # 细化匹配可补充已有多峰基础的频带。只有孤立峰的频带保留细化结果供复核，
    # 避免围绕同一峰重新拟合后将重复出现的孤立峰升级为跨带强证据。
    consistency_bins = 2.0
    for family in FAMILY_ORDER:
        refined_values: list[float] = []
        for _, matches in band_matches:
            lattice = matches[family].get("lattice") or {}
            if (len(matches[family].get("qualified_orders", [])) >= 2
                    and lattice.get("refined_frequency_used") and lattice.get("refined_fundamental_hit")
                    and len(lattice.get("refined_qualified_orders", [])) >= 2):
                refined_values.append(float(lattice["fitted_frequency_hz"]))
        consensus: float | None = None
        spread: float | None = None
        if refined_values:
            arr = np.asarray(refined_values, dtype=float)
            consensus = float(np.median(arr))
            if resolution > 0:
                spread = float(np.median(np.abs(arr - consensus)) / resolution)
        consensus_hz[family] = consensus
        spread_bins[family] = spread

        window = consistency_bins * resolution if resolution > 0 else 0.0
        for _, matches in band_matches:
            lattice = matches[family].get("lattice") or {}
            orders = matches[family].get("qualified_orders", [])
            theory_ok = (
                1 in orders and len(orders) >= 2
                and lattice.get("best_model") != "shaft"
            )
            refined_ok = False
            if (
                len(orders) >= 2
                and lattice.get("refined_frequency_used")
                and lattice.get("refined_fundamental_hit")
                and len(lattice.get("refined_qualified_orders", [])) >= 2
                and consensus is not None
            ):
                refined_ok = (
                    abs(float(lattice["fitted_frequency_hz"]) - consensus) <= window
                )
            grouped = theory_ok or refined_ok
            fitted = grouped and bool(lattice.get("fault_supported")) and not lattice.get("fitted_is_interference")
            matches[family]["grouped_peak_match"] = grouped
            matches[family]["grouped_fit_supported"] = fitted
            if grouped:
                evidence_bands[family] += 1
            if fitted:
                fit_supported_bands[family] += 1

    tolerance1: dict[str, float] = {}
    for family in FAMILY_ORDER:
        fundamental = frequencies_hz.get(family, 0.0)
        tolerance1[family] = (
            family_tolerance_fn(family, fundamental, resolution, matching)(1)
            if fundamental > 0 and resolution > 0
            else 0.0
        )
    colliding = decision.fundamental_collisions(
        {family: frequencies_hz.get(family, 0.0) for family in FAMILY_ORDER},
        tolerance1,
    )
    # 保留实测峰列和匹配带数。至少一个匹配带也通过拟合，才评为中等或强支持。
    levels = decision.assess_families(
        {family: count if fit_supported_bands[family] else 0 for family, count in evidence_bands.items()},
        colliding,
        strong_min_bands=decision_cfg["strong_min_bands"],
        moderate_min_bands=decision_cfg["moderate_min_bands"],
    )

    assessments: dict[str, dict] = {}
    for family in FAMILY_ORDER:
        supported = [
            matches[family].get("lattice") or {}
            for _, matches in band_matches
            if (matches[family].get("lattice") or {}).get("fault_supported")
        ]
        residuals = sorted(
            float(item.get("residual_rms_bins", float("inf"))) for item in supported
        )
        models = collections.Counter(
            str((matches[family].get("lattice") or {}).get("best_model"))
            for _, matches in band_matches
        )
        assessments[family] = {
            "evidence_id": f"{prefix}assessment.{family}",
            "evidence_level": levels[family],
            "evidence_bands": int(evidence_bands[family]),
            "fit_supported_bands": fit_supported_bands[family],
            "refined_frequency_hz": consensus_hz[family],
            "refined_consistency_bins": spread_bins[family],
            "fundamental_bands": int(fundamental_bands[family]),
            "lattice_models": dict(models),
            "median_residual_bins": (
                residuals[len(residuals) // 2] if residuals else None
            ),
            "max_residual_bins": (residuals[-1] if residuals else None),
            "qualified_orders": sorted(union_orders[family]),
            "fundamental_collides": family in colliding,
        }

    return {
        "resolution": resolution,
        "evidence_bands": evidence_bands,
        "fundamental_bands": fundamental_bands,
        "colliding": colliding,
        "levels": levels,
        "assessments": assessments,
        "band_evidence": band_evidence,
        "band_gate": band_gate,
        "spectrum_artifacts": spectrum_artifacts,
    }
