"""故障特征频率谐波匹配（确定性，精简版）。

对每个故障族（BPFO/BPFI/BSF），在包络谱上按理论频率整数倍找谐波峰，
用「容差内 + 相对幅值 + 局部稳健 z 分数」三条件判定谐波是否合格。
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np

FAMILY_ORDER = ("BPFO", "BPFI", "BSF")


def harmonic_tolerance_hz(
    fundamental_hz: float,
    harmonic_order: int,
    frequency_resolution_hz: float,
    *,
    speed_relative_uncertainty: float = 0.0,
    geometry_relative_uncertainty: float = 0.0,
    slip_relative_uncertainty: float = 0.0,
    resolution_multiplier: float = 2.5,
    minimum_tolerance_hz: float = 0.5,
) -> float:
    """单个故障谐波的物理匹配容差。"""

    fundamental = float(fundamental_hz)
    order = int(harmonic_order)
    resolution = float(frequency_resolution_hz)
    if not math.isfinite(fundamental) or fundamental <= 0:
        raise ValueError("fundamental_hz 必须为正")
    if order <= 0:
        raise ValueError("harmonic_order 必须为正")
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("frequency_resolution_hz 必须为正")

    uncertainties = (
        max(0.0, float(speed_relative_uncertainty)),
        max(0.0, float(geometry_relative_uncertainty)),
        max(0.0, float(slip_relative_uncertainty)),
    )
    combined = math.sqrt(sum(u * u for u in uncertainties))
    return max(
        float(resolution_multiplier) * resolution,
        float(minimum_tolerance_hz),
        (order * fundamental) * combined,
    )


def _spectrum_arrays(
    frequencies_hz: Sequence[float] | np.ndarray,
    amplitudes: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    frequencies = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    spectrum = np.asarray(amplitudes, dtype=float).reshape(-1)
    if frequencies.size != spectrum.size or frequencies.size < 4:
        raise ValueError("频率与幅值必须等长且 >= 4 点")
    if not np.all(np.isfinite(frequencies)) or not np.all(np.isfinite(spectrum)):
        raise ValueError("频率与幅值必须有限")
    if np.any(np.diff(frequencies) <= 0):
        raise ValueError("频率必须严格递增")
    resolution = float(np.median(np.diff(frequencies)))
    if resolution <= 0:
        raise ValueError("频率分辨率必须为正")
    # 不截零：包络谱已去除谱底，残留围绕 0 波动；负值不是峰，由判定条件自然排除，
    # 同时保持局部背景的稳健统计（中位数/MAD）有效。
    return frequencies, spectrum, resolution


def match_harmonics(
    frequencies_hz: Sequence[float] | np.ndarray,
    amplitudes: Sequence[float] | np.ndarray,
    theoretical_fundamental_hz: float,
    *,
    max_harmonics: int = 5,
    tolerance_hz_fn: Callable[[int], float],
    min_relative_amplitude: float = 0.10,
    min_robust_z: float = 4.5,
    background_radius_bins: int = 12,
    analysis_end_hz: float | None = None,
) -> dict[str, Any]:
    """在包络谱上匹配某个故障族的谐波族，返回合格谐波明细。"""

    frequencies, spectrum, resolution = _spectrum_arrays(frequencies_hz, amplitudes)
    theoretical = float(theoretical_fundamental_hz)
    if not math.isfinite(theoretical) or theoretical <= 0:
        raise ValueError("theoretical_fundamental_hz 必须为正")

    start = 0
    end = (
        spectrum.size - 1
        if analysis_end_hz is None
        else min(
            spectrum.size - 1,
            int(np.searchsorted(frequencies, float(analysis_end_hz), side="right") - 1),
        )
    )
    relative_threshold = max(0.0, float(min_relative_amplitude))
    robust_z_threshold = max(0.0, float(min_robust_z))

    # 两阶段：先取各阶窗峰与局部背景，再以**本族自身最强谐波**为相对幅值参考。
    # 用该带全谱最大峰作参考会跨族偏置：强族或结构峰把弱族的全部阶次压到阈值以下
    # （实测 BJTU 上 BPFI 因此被拒最多，1163/4000 次带观测）。
    collected: list[dict[str, Any]] = []
    for harmonic in range(1, int(max_harmonics) + 1):
        target = float(harmonic * theoretical)
        if end < start or target < frequencies[start] or target > frequencies[end]:
            continue
        tolerance_hz = float(tolerance_hz_fn(harmonic))
        if not math.isfinite(tolerance_hz) or tolerance_hz < 0:
            raise ValueError("谐波容差必须为非负有限数值")
        tolerance_bins = int(math.ceil(tolerance_hz / resolution))
        lo = max(start, int(np.searchsorted(frequencies, target - tolerance_hz, side="left")))
        hi = min(end + 1, int(np.searchsorted(frequencies, target + tolerance_hz, side="right")))
        if hi <= lo:
            continue
        index = int(lo + np.argmax(spectrum[lo:hi]))
        observed = float(frequencies[index])
        amplitude = float(spectrum[index])

        # 谐波必须对应谱上的局部极大值：平滑衰减斜坡上的点不是“峰”，
        # 否则包络谱低频富余（带通噪声/健康信号的固有形状）会被误判为谐波命中。
        is_local_peak = bool(
            index > 0
            and index < spectrum.size - 1
            and amplitude >= spectrum[index - 1]
            and amplitude >= spectrum[index + 1]
        )
        background_median, robust_sigma = _local_background_stats(
            spectrum, index, tolerance_bins, start, end, background_radius_bins
        )
        robust_z = float(
            (amplitude - background_median) / max(robust_sigma, np.finfo(float).eps)
        )
        collected.append(
            {
                "harmonic": harmonic,
                "target_frequency_hz": target,
                "observed_frequency_hz": observed,
                "error_hz": float(abs(observed - target)),
                "tolerance_hz": tolerance_hz,
                "amplitude": amplitude,
                "local_robust_z": robust_z,
                "is_local_peak": is_local_peak,
            }
        )

    family_peak = max((item["amplitude"] for item in collected), default=0.0)
    harmonic_peaks: list[dict[str, Any]] = []
    qualified_orders: list[int] = []
    for item in collected:
        relative_amplitude = (
            float(item["amplitude"] / family_peak)
            if family_peak > np.finfo(float).tiny
            else 0.0
        )
        qualified = bool(
            item["error_hz"] <= item["tolerance_hz"] + np.finfo(float).eps
            and item["is_local_peak"]
            and relative_amplitude >= relative_threshold
            and item["local_robust_z"] >= robust_z_threshold
        )
        peak = {
            "harmonic": item["harmonic"],
            "target_frequency_hz": item["target_frequency_hz"],
            "observed_frequency_hz": item["observed_frequency_hz"],
            "error_hz": item["error_hz"],
            "tolerance_hz": item["tolerance_hz"],
            "relative_amplitude": relative_amplitude,
            "family_peak_amplitude": family_peak,
            "local_robust_z": item["local_robust_z"],
            "amplitude": item["amplitude"],
            "is_local_peak": item["is_local_peak"],
            "qualified": qualified,
        }
        harmonic_peaks.append(peak)
        if qualified:
            qualified_orders.append(int(item["harmonic"]))

    qualified_orders = sorted(qualified_orders)
    return {
        "theoretical_fundamental_hz": theoretical,
        "frequency_resolution_hz": resolution,
        "harmonic_peaks": harmonic_peaks,
        "qualified_orders": qualified_orders,
        "qualified_match_count": len(qualified_orders),
        "has_low_order_anchor": bool({1, 2} & set(qualified_orders)),
    }


def comb_local_z(
    frequencies_hz: Sequence[float] | np.ndarray,
    amplitudes: Sequence[float] | np.ndarray,
    fundamental_hz: float,
    *,
    max_harmonics: int = 5,
    peak_half_bins: int = 2,
    background_radius_bins: int = 12,
    analysis_end_hz: float | None = None,
) -> dict[str, Any]:
    """某基频下各阶局部稳健 z 的均值（梳的"局部显著性"总量）。

    用于比较两个竞争基频假设（理论值 vs 数据细化值）在同一批阶次上的解释力：
    细化值必须在同一批阶次上明显更显著，才值得替代理论值。
    """

    frequencies, spectrum, resolution = _spectrum_arrays(frequencies_hz, amplitudes)
    fundamental = float(fundamental_hz)
    if not math.isfinite(fundamental) or fundamental <= 0:
        raise ValueError("fundamental_hz 必须为正")

    half = max(1, int(peak_half_bins))
    start = 0
    end = (
        spectrum.size - 1
        if analysis_end_hz is None
        else min(
            spectrum.size - 1,
            int(np.searchsorted(frequencies, float(analysis_end_hz), side="right") - 1),
        )
    )
    z_values: list[float] = []
    orders: list[int] = []
    for order in range(1, max(1, int(max_harmonics)) + 1):
        target = order * fundamental
        if target < frequencies[start] or target > frequencies[end]:
            continue
        _, amplitude = _window_peak(frequencies, spectrum, target, half, start, end)
        index = min(max(int(np.searchsorted(frequencies, target)), start), end)
        median, sigma = _local_background_stats(
            spectrum, index, half, start, end, background_radius_bins
        )
        z_values.append((amplitude - median) / max(sigma, np.finfo(float).eps))
        orders.append(order)
    return {
        "fundamental_hz": fundamental,
        "mean_z": float(np.mean(z_values)) if z_values else 0.0,
        "orders": orders,
        "z_by_order": z_values,
        "frequency_resolution_hz": resolution,
    }


def _window_peak(
    frequencies: np.ndarray,
    spectrum: np.ndarray,
    target_hz: float,
    half: int,
    start: int,
    end: int,
) -> tuple[float, float]:
    """目标频率 ±half bin 内的最大幅值点。"""

    index = int(np.searchsorted(frequencies, target_hz))
    index = min(max(index, start), end)
    if index > start and abs(frequencies[index - 1] - target_hz) < abs(
        frequencies[index] - target_hz
    ):
        index -= 1
    low = max(start, index - half)
    high = min(end + 1, index + half + 1)
    if high <= low:
        return float(frequencies[index]), float(spectrum[index])
    best = int(low + np.argmax(spectrum[low:high]))
    return float(frequencies[best]), float(spectrum[best])


def _fit_lattice_scale(
    observations: Sequence[float],
    coefficients: Sequence[float],
    offsets: Sequence[float],
    weights: Sequence[float] | None = None,
) -> tuple[float, float, float, int]:
    """幅值加权的过原点最小二乘拟合，返回 (F, 加权残差RMS, n_eff, 线数)。

    权重取谱线幅值：栅格证据应来自**显著**的谱线，缺失/微弱的分量不提供信息。
    ``n_eff = (Σw)²/Σw²`` 是加权统计的有效样本数，用于防止"单根强峰"冒充栅格。
    """

    obs = np.asarray(observations, dtype=float)
    k = np.asarray(coefficients, dtype=float)
    off = np.asarray(offsets, dtype=float)
    if obs.size < 2:
        return float("nan"), float("inf"), 0.0, int(obs.size)
    if weights is None:
        w = np.ones(obs.size, dtype=float)
    else:
        w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
        if not np.any(w > 0):
            w = np.ones(obs.size, dtype=float)
    denominator = float(np.sum(w * k * k))
    if denominator <= 0:
        return float("nan"), float("inf"), 0.0, int(obs.size)
    scale = float(np.sum(w * k * (obs - off)) / denominator)
    residual = obs - (k * scale + off)
    total = float(np.sum(w))
    weighted_rms = float(np.sqrt(np.sum(w * residual ** 2) / total))
    effective = float(total ** 2 / max(float(np.sum(w * w)), np.finfo(float).eps))
    return scale, weighted_rms, effective, int(obs.size)


def _shaft_commensurate(candidate_hz: float, shaft_hz: float, tolerance_hz: float,
                        max_harmonics: int) -> bool:
    """候选间距的谱列是否与轴频整数倍谱列在紧容差内重合 ≥2 阶。

    只在**绝对**容差（bin）下判定；用相对容差会随阶次退化——实测 k 大到 8 时
    几乎任何 F 都会落在某个轴频谐波附近。
    """

    hits = 0
    for order in range(1, int(max_harmonics) + 1):
        target = order * candidate_hz
        multiple = round(target / shaft_hz)
        if multiple >= 1 and abs(multiple * shaft_hz - target) <= tolerance_hz:
            hits += 1
    return hits >= 2


def lattice_evidence(
    frequencies_hz: Sequence[float] | np.ndarray,
    amplitudes: Sequence[float] | np.ndarray,
    theoretical_fundamental_hz: float,
    *,
    max_harmonics: int = 5,
    relative_span: float = 0.05,
    refine_steps: int = 401,
    peak_half_bins: int = 2,
    shaft_hz: float | None = None,
    modulation_hz: float | None = None,
    max_sideband_order: int = 2,
    min_orders_for_fit: int = 3,
    visibility_sigma: float = 2.0,
    background_radius_bins: int = 12,
    accept_ratio: float = 0.6,
    analysis_end_hz: float | None = None,
) -> dict[str, Any]:
    """对峰列做模型选择：纯栅格 / 调制栅格 / 干扰栅格 / 无。

    与 ``match_harmonics`` 的逐峰局部显著性不同，本判据问的是"这些峰是否构成一个
    可用的周期栅格"，用**过原点最小二乘残差（单位 bin）**度量。

    零假设分布由搜索窗宽决定：若各阶的取峰位置在 ±h bin 内均匀随机，残差 RMS 期望为
    ``h/√3``。因此接受阈值是"残差/null"这个无量纲比值——阈值只依赖频率分辨率，
    不依赖轴承型号、转速或数据集，可直接跨设备比较。

    干扰栅格用**同一批阶次位置上的轴频谐波**来拟合，从而与故障模型构成同参数个数的
    公平比较：若轴频模型拟合更好，则这些峰应归因于轴频而非轴承故障。

    返回值是测量与模型比较结果，不含故障族标签，也不做裁决。
    """

    frequencies, spectrum, resolution = _spectrum_arrays(frequencies_hz, amplitudes)
    theoretical = float(theoretical_fundamental_hz)
    if not math.isfinite(theoretical) or theoretical <= 0:
        raise ValueError("theoretical_fundamental_hz 必须为正")

    half = max(1, int(peak_half_bins))
    orders = max(2, int(max_harmonics))
    start = 0
    end = (
        spectrum.size - 1
        if analysis_end_hz is None
        else min(
            spectrum.size - 1,
            int(np.searchsorted(frequencies, float(analysis_end_hz), side="right") - 1),
        )
    )
    tolerance_hz = half * resolution
    # 零假设：各阶取峰位置在 ±half bin 内均匀随机 → 残差 RMS 期望 half/√3（单位 bin，
    # 与频率分辨率无关）。残差同样以 bin 计，因此接受阈值是跨设备可比的无量纲比值。
    null_rms_bins = half / math.sqrt(3.0)
    threshold_bins = float(accept_ratio) * null_rms_bins
    candidates = np.linspace(
        theoretical * (1.0 - float(relative_span)),
        theoretical * (1.0 + float(relative_span)),
        # 搜索网格只需落到一个 bin 以内，之后由最小二乘细化；据此自适应控制开销
        max(
            41,
            min(
                max(3, int(refine_steps)),
                int(2.0 * float(relative_span) * theoretical / resolution) + 1,
            ),
        ),
    )

    shaft = float(shaft_hz) if shaft_hz else None
    modulation = float(modulation_hz) if modulation_hz else None

    def build_targets(
        base_hz: float, sideband_order: int
    ) -> list[tuple[int, float, float]]:
        """返回 (系数 k, 偏移 j·fm, 目标频率) 列表；sideband_order=0 即纯栅格。"""

        rows: list[tuple[int, float, float]] = []
        if modulation is None or sideband_order <= 0:
            for order in range(1, orders + 1):
                rows.append((order, 0.0, order * base_hz))
            return rows
        for order in range(1, orders + 1):
            for side in range(-int(sideband_order), int(sideband_order) + 1):
                offset = side * modulation
                target = order * base_hz + offset
                if target > 0:
                    rows.append((order, offset, target))
        return rows

    def score(base_hz: float, sideband_order: int) -> float:
        total = 0.0
        for _, _, target in build_targets(base_hz, sideband_order):
            if target < frequencies[start] or target > frequencies[end]:
                continue
            _, amplitude = _window_peak(frequencies, spectrum, target, half, start, end)
            total += amplitude
        return total

    def evaluate(base_hz: float, sideband_order: int) -> dict[str, Any]:
        rows = build_targets(base_hz, sideband_order)
        observations, coefficients, offsets, lines = [], [], [], []
        for order, offset, target in rows:
            if target < frequencies[start] or target > frequencies[end]:
                continue
            observed, amplitude = _window_peak(
                frequencies, spectrum, target, half, start, end
            )
            index = min(max(int(np.searchsorted(frequencies, observed)), start), end)
            median, sigma = _local_background_stats(
                spectrum, index, half, start, end, background_radius_bins
            )
            local_z = (amplitude - median) / max(sigma, np.finfo(float).eps)
            observations.append(observed)
            coefficients.append(order)
            offsets.append(offset)
            lines.append(
                {
                    "order": order,
                    "sideband": offset,
                    "target_frequency_hz": target,
                    "observed_frequency_hz": observed,
                    "amplitude": amplitude,
                    "local_robust_z": float(local_z),
                    "visible": bool(local_z >= float(visibility_sigma)),
                }
            )
        visible_line_count = sum(1 for line in lines if line["visible"])
        scale, residual_hz, effective, count = _fit_lattice_scale(
            observations, coefficients, offsets, [line["amplitude"] for line in lines]
        )
        dropped_order: int | None = None
        # 稳健化：弱谐波在窗口内可能被噪声峰顶替，单个错误赋值会把残差整体顶高。
        # 允许剔除单根最差线后重拟合（只剔一根，且必须确有改善），这是标准做法。
        if (
            count >= int(min_orders_for_fit) + 1
            and residual_hz / resolution > threshold_bins
            and len(observations) >= 3
        ):
            weights = np.clip(
                np.asarray([line["amplitude"] for line in lines], dtype=float), 0.0, None
            )
            coefficients_array = np.asarray(coefficients, dtype=float)
            residual_vector = np.asarray(observations, dtype=float) - (
                coefficients_array * scale + np.asarray(offsets, dtype=float)
            )
            worst = int(np.argmax(np.abs(residual_vector)))
            keep = [i for i in range(count) if i != worst]
            if len(keep) >= int(min_orders_for_fit):
                trimmed_scale, trimmed_hz, trimmed_effective, trimmed_count = (
                    _fit_lattice_scale(
                        [observations[i] for i in keep],
                        [coefficients[i] for i in keep],
                        [offsets[i] for i in keep],
                        [weights[i] for i in keep],
                    )
                )
                if trimmed_hz < residual_hz:
                    scale, residual_hz = trimmed_scale, trimmed_hz
                    effective, count = trimmed_effective, trimmed_count
                    dropped_order = int(lines[worst]["order"])
        residual_rms_bins = residual_hz / resolution
        # BIC：n_eff·ln(RSS/n_eff) + p·ln(n_eff)，p = 1 个尺度参数 + 边带跨度
        if effective >= 1.0 and math.isfinite(residual_hz):
            rss = residual_hz ** 2 * effective
            bic = effective * math.log(
                max(rss / effective, np.finfo(float).tiny)
            ) + (1 + int(sideband_order)) * math.log(effective)
        else:
            bic = math.inf
        return {
            "model": "lattice" if sideband_order <= 0 else "modulated",
            "sideband_order": int(sideband_order),
            "fitted_frequency_hz": scale,
            "residual_rms_hz": residual_hz,
            "residual_rms_bins": residual_rms_bins,
            "line_count": count,
            "effective_line_count": effective,
            "visible_line_count": visible_line_count,
            "dropped_order": dropped_order,
            "bic": bic,
            "lines": lines,
        }

    def search(sideband_order: int) -> dict[str, Any]:
        best: dict[str, Any] | None = None
        best_score = -math.inf
        for candidate in candidates:
            if shaft is not None and _shaft_commensurate(
                candidate, shaft, tolerance_hz, orders
            ):
                continue
            value = score(float(candidate), sideband_order)
            if value > best_score:
                best_score = value
                best = evaluate(float(candidate), sideband_order)
        if best is None:
            best = evaluate(theoretical, sideband_order)
        return best

    lattice_model = search(0)

    models: dict[str, Any] = {"lattice": lattice_model, "sideband_0": lattice_model}
    best_span = 0
    if modulation is not None and modulation > 0:
        fitted: dict[int, dict[str, Any]] = {0: lattice_model}
        for span in range(1, int(max_sideband_order) + 1):
            fitted[span] = search(span)
            models[f"sideband_{span}"] = fitted[span]
        # 最简充分模型：取满足接受阈值的**最小**边带跨度（纯栅格优先）
        eligible = [
            s
            for s, m in fitted.items()
            if m["effective_line_count"] >= float(min_orders_for_fit)
        ]
        adequate = [s for s in eligible if fitted[s]["residual_rms_bins"] <= threshold_bins]
        if adequate:
            best_span = min(adequate)
        elif eligible:
            best_span = min(eligible, key=lambda s: fitted[s]["residual_rms_bins"])
        else:
            best_span = 0
        lattice_model = fitted[best_span]
        models["modulated"] = fitted[best_span] if best_span > 0 else None

    if shaft is not None:
        # 干扰模型：在与故障模型相同的阶次位置上取轴频谐波，做同参数个数拟合
        anchors = [line["target_frequency_hz"] for line in lattice_model["lines"]]
        anchors = anchors or [theoretical * k for k in range(1, orders + 1)]
        observations, coefficients, lines = [], [], []
        for index, anchor in enumerate(anchors, start=1):
            multiple = max(1, round(anchor / shaft))
            target = multiple * shaft
            if target < frequencies[start] or target > frequencies[end]:
                continue
            observed, amplitude = _window_peak(
                frequencies, spectrum, target, half, start, end
            )
            index = min(max(int(np.searchsorted(frequencies, observed)), start), end)
            median, sigma = _local_background_stats(
                spectrum, index, half, start, end, background_radius_bins
            )
            local_z = (amplitude - median) / max(sigma, np.finfo(float).eps)
            observations.append(observed)
            coefficients.append(float(multiple))
            lines.append(
                {
                    "order": index,
                    "shaft_multiple": multiple,
                    "target_frequency_hz": target,
                    "observed_frequency_hz": observed,
                    "amplitude": amplitude,
                    "local_robust_z": float(local_z),
                    "visible": bool(local_z >= float(visibility_sigma)),
                }
            )
        shaft_scale, shaft_residual_hz, shaft_effective, shaft_count = _fit_lattice_scale(
            observations,
            coefficients,
            [0.0] * len(observations),
            [line["amplitude"] for line in lines],
        )
        models["shaft"] = {
            "model": "shaft",
            "fitted_shaft_hz": shaft_scale,
            "residual_rms_hz": shaft_residual_hz,
            "residual_rms_bins": shaft_residual_hz / resolution,
            "line_count": shaft_count,
            "effective_line_count": shaft_effective,
            "visible_line_count": sum(1 for line in lines if line["visible"]),
            "lines": lines,
        }

    def valid(model_result: Mapping[str, Any] | None) -> bool:
        # 守卫用"有多少根线高于局部背景"，而不是幅值分布的集中度（n_eff）：
        # 轴承包络谱的谐波幅值本就随阶次强烈衰减，用 n_eff 会把真故障误拒
        # （实测 CRRC 92rpm 外圈：残差 0.296 bin、5 线全中，n_eff 仅 2.06）。
        return bool(
            model_result is not None
            and model_result["visible_line_count"] >= int(min_orders_for_fit)
            and model_result["residual_rms_bins"] <= threshold_bins
        )

    fault_model = lattice_model
    fault_valid = valid(fault_model)
    # 最小二乘重拟合可能把基频拉到轴频通约的位置（实测 1200rpm 正常样本 F*≈99.65Hz，
    # 5 次轴频谐波=100Hz）。候选搜索已排除这类候选，拟合结果必须再判一次。
    fitted_frequency = float(fault_model["fitted_frequency_hz"] or 0.0)
    fitted_is_interference = bool(
        shaft is not None
        and math.isfinite(fitted_frequency)
        and fitted_frequency > 0
        and _shaft_commensurate(fitted_frequency, shaft, tolerance_hz, orders)
    )
    shaft_model = models.get("shaft")
    shaft_wins = bool(
        valid(shaft_model)
        and (not fault_valid or shaft_model["residual_rms_bins"] < fault_model["residual_rms_bins"])
    )

    if shaft_wins:
        best_model = "shaft"
    elif fault_valid:
        best_model = "lattice" if best_span <= 0 else "modulated"
    else:
        best_model = "none"

    return {
        "theoretical_fundamental_hz": theoretical,
        "frequency_resolution_hz": resolution,
        "peak_half_bins": half,
        "null_rms_bins": null_rms_bins,
        "accept_threshold_bins": threshold_bins,
        "best_model": best_model,
        "sideband_order": int(best_span),
        "fault_supported": best_model in {"lattice", "modulated"},
        "fitted_is_interference": fitted_is_interference,
        "lattice_frequency_hz": fault_model["fitted_frequency_hz"],
        "residual_rms_hz": fault_model["residual_rms_hz"],
        "residual_rms_bins": fault_model["residual_rms_bins"],
        "line_count": fault_model["line_count"],
        "effective_line_count": fault_model["effective_line_count"],
        "visible_line_count": fault_model["visible_line_count"],
        "bic": fault_model["bic"],
        "models": models,
    }


def _local_background_stats(
    spectrum: np.ndarray,
    index: int,
    exclusion_bins: int,
    start: int,
    end: int,
    radius_bins: int,
) -> tuple[float, float]:
    """峰两侧环形区域的稳健背景估计（中位数 / 1.4826*MAD）。"""

    radius = max(int(exclusion_bins) + 1, int(radius_bins))
    local_start = max(start, index - radius)
    local_stop = min(end + 1, index + radius + 1)
    left = spectrum[local_start : max(local_start, index - exclusion_bins)]
    right = spectrum[min(local_stop, index + exclusion_bins + 1) : local_stop]
    background = np.concatenate((left, right))
    if background.size:
        median = float(np.median(background))
        mad = float(np.median(np.abs(background - median)))
    else:
        median, mad = 0.0, 0.0
    return median, 1.4826 * mad


def comb_evidence(
    frequencies_hz: Sequence[float] | np.ndarray,
    amplitudes: Sequence[float] | np.ndarray,
    theoretical_fundamental_hz: float,
    *,
    max_harmonics: int = 5,
    relative_span: float = 0.015,
    refine_steps: int = 61,
    peak_half_bins: int = 2,
    background_radius_bins: int = 12,
    min_order_z: float = 4.5,
    plateau_fraction: float = 0.999,
    analysis_end_hz: float | None = None,
) -> dict[str, Any]:
    """在包络谱上评估「整条谐波梳」的相干性，并细化基频。

    与 ``match_harmonics`` 的差别（也是本判据要解决的问题）：

    1. **频率细化**：在理论基频 ±``relative_span`` 内搜索使谐波梳总幅值最大的
       基频 ``comb_frequency_hz``，吸收转速/几何误差造成的系统频率偏差；
    2. **多阶同时成立**：``significant_count`` 统计超过局部噪声底的阶次数，
       一根偶然落在容差窗内的孤立谱峰只能得到 1 阶，不能冒充故障证据；
    3. **跨频带可比**：``comb_geomean_z`` 是各阶局部稳健 z 的几何均值，对缺阶
       强惩罚，且与整体幅值尺度无关，可用于不同解调频带之间的比较。

    返回值是测量结果，不含任何故障族标签或裁决。
    """

    frequencies, spectrum, resolution = _spectrum_arrays(frequencies_hz, amplitudes)
    theoretical = float(theoretical_fundamental_hz)
    if not math.isfinite(theoretical) or theoretical <= 0:
        raise ValueError("theoretical_fundamental_hz 必须为正")

    span = max(0.0, float(relative_span))
    steps = max(3, int(refine_steps))
    half = max(1, int(peak_half_bins))
    order_count = max(1, int(max_harmonics))

    start = 0
    end = (
        spectrum.size - 1
        if analysis_end_hz is None
        else min(
            spectrum.size - 1,
            int(np.searchsorted(frequencies, float(analysis_end_hz), side="right") - 1),
        )
    )

    candidates = np.linspace(
        theoretical * (1.0 - span), theoretical * (1.0 + span), steps
    )
    order_scores = np.zeros((order_count, candidates.size), dtype=float)
    usable = np.zeros(order_count, dtype=bool)
    for order in range(1, order_count + 1):
        targets = order * candidates
        inside = (targets >= frequencies[start]) & (targets <= frequencies[end])
        if not inside.any():
            continue
        usable[order - 1] = True
        centers = np.clip(
            np.searchsorted(frequencies, targets), 1, frequencies.size - 1
        )
        nearest = np.where(
            np.abs(frequencies[centers] - targets)
            <= np.abs(frequencies[centers - 1] - targets),
            centers,
            centers - 1,
        )
        for slot in np.nonzero(inside)[0]:
            center = int(nearest[slot])
            low = max(start, center - half)
            high = min(end + 1, center + half + 1)
            if high > low:
                order_scores[order - 1, slot] = float(spectrum[low:high].max())

    if not usable.any():
        return {
            "theoretical_fundamental_hz": theoretical,
            "comb_frequency_hz": theoretical,
            "frequency_shift_ratio": 0.0,
            "frequency_resolution_hz": resolution,
            "comb_amplitude": 0.0,
            "comb_geomean_z": 0.0,
            "significant_count": 0,
            "significant_orders": [],
            "harmonics": [],
        }

    totals = order_scores.sum(axis=0)
    # S(F) 在一个区间内是平台（所有谐波窗都套住同一批峰），argmax 只会返回平台最低
    # 索引，使 F* 系统性偏低（实测在 ±1.5% 网格上偏低约 0.25%）。报平台中心。
    top_total = float(totals.max()) if totals.size else 0.0
    plateau = candidates[totals >= top_total * float(plateau_fraction)]
    if plateau.size:
        comb_frequency = float((plateau.min() + plateau.max()) / 2.0)
        plateau_width = float(plateau.max() - plateau.min())
    else:  # pragma: no cover - candidates 非空时 plateau 至少含 argmax
        comb_frequency = float(candidates[int(np.argmax(totals))])
        plateau_width = 0.0

    harmonics: list[dict[str, Any]] = []
    significant_orders: list[int] = []
    log_terms: list[float] = []
    for order in range(1, order_count + 1):
        target = order * comb_frequency
        if target < frequencies[start] or target > frequencies[end]:
            continue
        center = int(np.argmin(np.abs(frequencies - target)))
        low = max(start, center - half)
        high = min(end + 1, center + half + 1)
        index = int(low + np.argmax(spectrum[low:high]))
        observed = float(frequencies[index])
        amplitude = float(spectrum[index])
        background_median, robust_sigma = _local_background_stats(
            spectrum, index, half, start, end, background_radius_bins
        )
        robust_z = float(
            (amplitude - background_median) / max(robust_sigma, np.finfo(float).eps)
        )
        significant = bool(robust_z >= float(min_order_z))
        if significant:
            significant_orders.append(order)
        log_terms.append(math.log(max(robust_z, 1.0)))
        harmonics.append(
            {
                "harmonic": order,
                "target_frequency_hz": target,
                "observed_frequency_hz": observed,
                "error_hz": float(abs(observed - target)),
                "amplitude": amplitude,
                "local_background_median": background_median,
                "local_robust_sigma": robust_sigma,
                "local_robust_z": robust_z,
                "significant": significant,
            }
        )

    geomean = float(math.exp(sum(log_terms) / len(log_terms))) if log_terms else 0.0
    return {
        "theoretical_fundamental_hz": theoretical,
        "comb_frequency_hz": comb_frequency,
        "frequency_shift_ratio": float(comb_frequency / theoretical - 1.0),
        "frequency_resolution_hz": resolution,
        "comb_amplitude": top_total,
        "plateau_width_hz": plateau_width,
        "comb_geomean_z": geomean,
        "significant_count": len(significant_orders),
        "significant_orders": significant_orders,
        "harmonics": harmonics,
    }
