"""确定性信号处理工具（精简版）。

诊断必需的三个能力：
1. 带通滤波（谐振带解调）。
2. Hilbert 包络谱（普通/平方包络）。
3. 多频带分析（谱峭度选带，用于跨频带基频一致性）。
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import signal


def bandpass_filter(
    values: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> np.ndarray:
    """零相位带通滤波。"""

    x = np.asarray(values, dtype=float)
    fs = float(sampling_rate_hz)
    low, high = float(low_hz), float(high_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz 必须为正有限数值")
    nyq = fs / 2.0
    if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high <= nyq:
        raise ValueError("频带必须满足 0 <= low_hz < high_hz <= Nyquist")
    if x.ndim != 1 or x.size < 8 or not np.all(np.isfinite(x)):
        raise ValueError("信号必须是一维且至少包含 8 个有限样本")
    if isinstance(order, (bool, np.bool_)) or int(order) != order or order < 1:
        raise ValueError("order 必须为正整数")
    if low == 0 and high == nyq:
        return x.copy()
    if low == 0:
        sos = signal.butter(order, high / nyq, btype="lowpass", output="sos")
    elif high == nyq:
        sos = signal.butter(order, low / nyq, btype="highpass", output="sos")
    else:
        sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def envelope_spectrum(
    values: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    *,
    band_hz: Sequence[float] | None = None,
    mode: str = "squared",
    window: str | None = "hann",
    return_envelope: bool = False,
) -> dict:
    """Hilbert 包络谱（普通或平方包络 + 单边幅值谱）。"""

    if type(return_envelope) is not bool:
        raise ValueError("return_envelope must be a boolean")
    x = np.asarray(values, dtype=float).reshape(-1)
    fs = float(sampling_rate_hz)
    if x.size < 8 or not np.all(np.isfinite(x)):
        raise ValueError("信号至少需要 8 个有限样本")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("sampling_rate_hz 必须为正有限数值")
    if mode not in {"ordinary", "squared"}:
        raise ValueError("mode 只支持 ordinary / squared")

    if band_hz is not None:
        band = [float(b) for b in band_hz]
        if len(band) != 2:
            raise ValueError("band_hz 必须包含两个频带边界")
        x = bandpass_filter(x, fs, band[0], band[1])

    filtered_rms = float(np.sqrt(np.mean(np.square(x - float(np.mean(x))))))

    analytic_envelope = np.abs(signal.hilbert(x))
    spectrum_input = analytic_envelope if mode == "ordinary" else np.square(analytic_envelope)
    spectrum_input = spectrum_input - float(np.mean(spectrum_input))

    if window is None or str(window).strip().lower() in {"none", "rectangular"}:
        taper = np.ones(spectrum_input.size, dtype=float)
        window_name = "rectangular"
    elif str(window).strip().lower() == "hann":
        taper = signal.windows.hann(spectrum_input.size, sym=False)
        window_name = "hann"
    else:
        raise ValueError("window 只支持 hann / rectangular / None")

    coherent_gain = float(np.mean(taper))
    amplitude = np.abs(np.fft.rfft(spectrum_input * taper)) / (
        spectrum_input.size * max(coherent_gain, np.finfo(float).eps)
    )
    stop = -1 if spectrum_input.size % 2 == 0 else None
    amplitude[1:stop] *= 2.0
    positive_amplitude = amplitude[1:] if amplitude.size > 1 else amplitude
    envelope_background = (
        float(np.percentile(positive_amplitude, 95)) if positive_amplitude.size else 0.0
    )
    envelope_peak = float(np.max(positive_amplitude)) if positive_amplitude.size else 0.0
    envelope_snr_db = float(
        20.0
        * np.log10(
            (envelope_peak + np.finfo(float).eps)
            / (envelope_background + np.finfo(float).eps)
        )
    )

    # 去除包络谱的低频谱底（带通噪声/健康信号平方包络的固有鼓包，宽约滤波带宽），
    # 只保留尖锐谐波峰：中值滤波估计谱底并减去。保留负值——去底后的残留围绕 0
    # 波动，负值由匹配器忽略（不是峰），同时让局部稳健 z 分数保持有效尺度。
    # 窗口上限 101 bin：远小于故障谐波间隔（≥ ~12 Hz），远小于谱底鼓包宽度，
    # 保证只去谱底、不伤谐波梳。
    baseline_bins = max(31, min(2 * (spectrum_input.size // 2048) + 1, 101) | 1)
    padded = np.pad(amplitude, baseline_bins // 2, mode="reflect")
    baseline = signal.medfilt(padded, baseline_bins)[
        baseline_bins // 2 : baseline_bins // 2 + amplitude.size
    ]
    amplitude = amplitude - baseline
    frequencies = np.fft.rfftfreq(spectrum_input.size, d=1.0 / fs)
    return {
        "frequencies_hz": frequencies,
        "amplitudes": amplitude,
        "frequency_resolution_hz": float(fs / spectrum_input.size),
        "filtered_rms": filtered_rms,
        "envelope_snr_db": envelope_snr_db,
        "mode": mode,
        "window": window_name,
        "band_hz": list(band_hz) if band_hz is not None else None,
        **({"envelope_values": spectrum_input} if return_envelope else {}),
    }


def envelope_excess_kurtosis(
    values: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    band_hz: Sequence[float],
) -> float:
    """某频带解调包络的超额峰度（谱峭度代理指标）。"""

    filtered = bandpass_filter(values, sampling_rate_hz, band_hz[0], band_hz[1])
    envelope = np.abs(signal.hilbert(filtered))
    envelope = envelope - float(np.mean(envelope))
    m2 = float(np.mean(np.square(envelope)))
    m4 = float(np.mean(np.square(np.square(envelope))))
    if m2 < 1e-30:
        return 0.0
    return m4 / (m2 * m2) - 3.0


def select_bands_by_kurtosis(
    values: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    *,
    min_band_hz: float = 50.0,
    n_bands: int = 4,
    n_edges: int = 10,
) -> list[tuple[float, float]]:
    """用谱峭度（解调包络峰度）自适应选出最优谐振频带。

    这是通用性的关键：频带由信号自身决定，而不是固定倍频带——
    高速轴承的共振可能在 1–32 kHz，低速轴承可能在数百 Hz，两者都能被找到。
    """

    x = np.asarray(values, dtype=float)
    nyq = sampling_rate_hz / 2.0
    start_hz = max(float(min_band_hz), 1.0)
    if not np.isfinite(nyq) or nyq <= 0 or start_hz >= nyq * 0.999:
        return []
    edges = np.geomspace(start_hz, nyq, num=max(2, int(n_edges)))
    scored: list[tuple[float, tuple[float, float]]] = []
    for i in range(edges.size - 1):
        low = float(edges[i])
        high = float(edges[i + 1])
        if high - low < float(min_band_hz):
            continue
        score = envelope_excess_kurtosis(x, sampling_rate_hz, (low, high))
        scored.append((score, (low, high)))

    scored.sort(key=lambda item: item[0], reverse=True)
    bands = [band for score, band in scored if score > 0.0][: int(n_bands)]
    if not bands:
        bands = [scored[0][1]] if scored else [(0.0, nyq)]
    return bands


def select_resonance_bands(
    values: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    *,
    min_band_hz: float = 50.0,
    n_bands: int = 5,
    n_edges: int = 10,
) -> list[tuple[float, float]]:
    """稳态主链的统一选带：谱峭度 top-N 频带（实测最优策略）。

    - 数据驱动：谱峭度（解调包络峰度）高的频带最可能承载故障冲击共振，
      频带由信号自身激活，无需人工指定共振范围；
    - 若所有候选频带峰度 ≤ 0（无可激活频带，如纯噪声），只取峰度最高的
      单个频带；后续可诊断性门会因有效频带不足输出 unknown_quality_limited；
    - 频带下限不低于 min_band_hz（调用方传入 max(50, 最高故障谐波)），
      避免频带直接覆盖故障频率本身。
    """

    return select_bands_by_kurtosis(
        values,
        sampling_rate_hz,
        min_band_hz=min_band_hz,
        n_bands=int(n_bands),
        n_edges=int(n_edges),
    )


def kurtosis_band_scores(
    values: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    *,
    min_band_hz: float = 50.0,
    n_edges: int = 10,
) -> list[dict]:
    """对数频带上的谱峭度（解调包络峰度）分数表，供「AI 确定参数」读取。"""

    x = np.asarray(values, dtype=float)
    nyq = sampling_rate_hz / 2.0
    if not np.isfinite(nyq) or nyq <= 0 or float(min_band_hz) >= nyq:
        return []
    edges = np.geomspace(max(float(min_band_hz), 1.0), nyq, num=max(2, int(n_edges)))
    scored: list[dict] = []
    for i in range(edges.size - 1):
        low = float(edges[i])
        high = float(edges[i + 1])
        if high - low < float(min_band_hz):
            continue
        kurtosis = envelope_excess_kurtosis(x, sampling_rate_hz, (low, high))
        scored.append({"band_hz": [low, high], "kurtosis": round(kurtosis, 3)})
    scored.sort(key=lambda item: item["kurtosis"], reverse=True)
    return scored
