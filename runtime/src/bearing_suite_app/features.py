"""时频域特征提取 + 特征描述文本生成（P0）。

从振动信号提取三类特征，产出两样东西：
1. ``feature_vector``：固定长度的数值向量（归一化后），供案例库检索算相似度。
2. ``description``：结构化中文文本，供 LLM 直接读取。

三类特征：
- 时域：RMS、峰值、波峰因子、峭度、偏度、冲击因子。
- 频域：谱峭度（kurtogram）top 频带。
- 包络域：各族故障频率谐波匹配（基频跨频带命中数、谐波阶次）。

包络域证据复用稳态主链（``chain.analyze_envelope_chain``），与诊断主链同口径；
MFD 参数默认来自 mfd_method.json（经 ``config.load_config`` 统一加载）。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal as scipy_signal

from . import chain
from . import geometry as geom
from . import signal_tools as sig
from .config import load_config

FAMILY_ORDER = ("BPFO", "BPFI", "BSF")
FEATURE_SCHEMA_VERSION = "bearing-features/v2-psd"

# 固定物理频带便于同一设备/采样链的健康基线比较。超出 Nyquist 的频带保留为 0，
# 从而保证案例向量维度稳定；PSD 只作正常性/异常性辅助，不直接定位故障族。
PSD_BANDS_HZ: tuple[tuple[float, float | None], ...] = (
    (0.0, 200.0),
    (200.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 8000.0),
    (8000.0, 12000.0),
    (12000.0, 18000.0),
    (18000.0, 24000.0),
    (24000.0, None),
)


# ------------------------------------------------------------------ 时域
def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _kurtosis(x: np.ndarray) -> float:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < np.finfo(float).eps:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)


def _skewness(x: np.ndarray) -> float:
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < np.finfo(float).eps:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def time_domain_features(x: np.ndarray) -> dict[str, float]:
    """时域统计特征。"""

    rms = _rms(x)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    abs_mean = float(np.mean(np.abs(x))) if x.size else 0.0
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": float(peak / rms) if rms > 0 else 0.0,
        "kurtosis": _kurtosis(x),
        "skewness": _skewness(x),
        "impulse_factor": float(peak / abs_mean) if abs_mean > 0 else 0.0,
        "shape_factor": float(rms / abs_mean) if abs_mean > 0 else 0.0,
    }


# ------------------------------------------------------------------ 频域
def frequency_domain_features(x: np.ndarray, fs: float) -> dict[str, Any]:
    """谱峭度频带 + Welch PSD 频带能量画像。"""

    bands = sig.kurtosis_band_scores(x, fs)[:3]
    return {"kurtosis_bands": bands, "psd": welch_psd_features(x, fs)}


def welch_psd_features(x: np.ndarray, fs: float) -> dict[str, Any]:
    """生成固定维度的 Welch PSD 频带能量画像。

    返回相对总功率的频带占比和 dB 画像。绝对振动量仍由 RMS 表达，避免 PSD
    同时混入量纲和传感器增益；跨案例比较前必须先按型号、测点和工况过滤。
    """

    values = np.asarray(x, dtype=float).reshape(-1)
    sampling_rate = float(fs)
    if values.size < 4 or not np.isfinite(sampling_rate) or sampling_rate <= 0:
        return {
            "evidence_id": "signal.psd",
            "method": "welch",
            "profile_version": FEATURE_SCHEMA_VERSION,
            "sampling_rate_hz": sampling_rate,
            "nperseg": 0,
            "band_power_ratios": [],
            "log_band_power_ratio_db": [0.0] * len(PSD_BANDS_HZ),
        }

    nperseg = min(16384, int(values.size))
    frequencies, density = scipy_signal.welch(
        values,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    reliable_high = sampling_rate * 0.5 * 0.95
    df = float(np.median(np.diff(frequencies))) if frequencies.size > 1 else 0.0
    reliable = frequencies <= reliable_high
    total_power = float(np.sum(density[reliable]) * df) if df > 0 else 0.0
    floor = np.finfo(float).tiny

    band_rows: list[dict[str, Any]] = []
    profile_db: list[float] = []
    for low, configured_high in PSD_BANDS_HZ:
        high = reliable_high if configured_high is None else min(float(configured_high), reliable_high)
        valid = bool(low < reliable_high and high > low and df > 0 and total_power > floor)
        if valid:
            mask = (frequencies >= low) & (frequencies < high) & reliable
            band_power = float(np.sum(density[mask]) * df)
            ratio = max(0.0, min(1.0, band_power / total_power))
        else:
            ratio = 0.0
        relative_db = float(10.0 * np.log10(max(ratio, 1e-12)))
        band_rows.append(
            {
                "band_hz": [float(low), float(high)],
                "power_ratio": ratio,
                "relative_db": relative_db,
                "observable": valid,
            }
        )
        profile_db.append(relative_db)

    return {
        "evidence_id": "signal.psd",
        "method": "welch",
        "profile_version": FEATURE_SCHEMA_VERSION,
        "sampling_rate_hz": sampling_rate,
        "nperseg": int(nperseg),
        "frequency_resolution_hz": df,
        "reliable_range_hz": [0.0, float(reliable_high)],
        "band_power_ratios": band_rows,
        "log_band_power_ratio_db": profile_db,
    }


# ------------------------------------------------------------------ 包络域
def envelope_domain_features(
    x: np.ndarray, fs: float, rpm: float, geometry: Mapping[str, Any] | None, m: Mapping[str, Any]
) -> dict[str, Any]:
    """包络域：各族故障频率谐波匹配（复用稳态主链，基频跨频带命中数 + 谐波阶次）。"""

    ratios = geom.fault_frequency_ratios(geometry)
    frequencies_hz = geom.fault_frequencies_hz(rpm, ratios)

    max_fault = max((v for v in frequencies_hz.values() if v > 0), default=0.0)
    min_band = max(50.0, max_fault * m["max_harmonics"])
    bands = sig.select_resonance_bands(x, fs, min_band_hz=min_band, n_bands=5)

    decision_cfg = load_config()["decision"]
    analysis = chain.analyze_envelope_chain(
        x, fs, frequencies_hz, bands, matching=m, decision_cfg=decision_cfg,
        shaft_hz=(float(rpm) / 60.0) if rpm else None,
    )

    family_result: dict[str, dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        a = analysis["assessments"].get(family, {})
        family_result[family] = {
            "fundamental_hz": round(float(frequencies_hz.get(family, 0.0)), 2),
            "fundamental_bands": int(a.get("fundamental_bands", 0)),
            "qualified_orders": list(a.get("qualified_orders", [])),
        }

    return {"fault_frequencies_hz": {k: round(v, 2) for k, v in frequencies_hz.items()}, "families": family_result}


# ------------------------------------------------------------ 特征向量 + 描述
def feature_vector(features: Mapping[str, Any]) -> list[float]:
    """固定长度特征向量（用于检索相似度）。"""

    t = features.get("time", {})
    f = features.get("frequency", {})
    e = features.get("envelope", {})
    families = e.get("families", {})

    kurt_bands = f.get("kurtosis_bands", [])
    top1 = kurt_bands[0] if len(kurt_bands) > 0 else {}
    top2 = kurt_bands[1] if len(kurt_bands) > 1 else {}

    def _center(band: dict) -> float:
        b = band.get("band_hz", [0, 0])
        return float((b[0] + b[1]) / 2.0) if isinstance(b, (list, tuple)) and len(b) == 2 else 0.0

    vec = [
        float(t.get("crest_factor", 0.0)),
        float(t.get("kurtosis", 0.0)),
        float(t.get("skewness", 0.0)),
        float(t.get("rms", 0.0)),
        float(t.get("impulse_factor", 0.0)),
        _center(top1),
        float(top1.get("kurtosis", 0.0)),
        _center(top2),
        float(top2.get("kurtosis", 0.0)),
    ]
    for family in FAMILY_ORDER:
        fam = families.get(family, {})
        vec.append(float(fam.get("fundamental_bands", 0)))
    for family in FAMILY_ORDER:
        fam = families.get(family, {})
        vec.append(float(len(fam.get("qualified_orders", []))))
    psd_profile = f.get("psd", {}).get("log_band_power_ratio_db", [])
    vec.extend(float(value) for value in psd_profile)
    return vec


def describe(features: Mapping[str, Any]) -> str:
    """生成结构化中文特征描述文本，供 LLM 读取。"""

    t = features.get("time", {})
    f = features.get("frequency", {})
    e = features.get("envelope", {})
    freqs = e.get("fault_frequencies_hz", {})
    families = e.get("families", {})

    lines: list[str] = []
    lines.append(
        f"【工况】转速 {features.get('rpm')} rpm，采样率 {features.get('fs')} Hz，"
        f"时长 {features.get('duration_s')} s"
    )
    lines.append(
        f"【时域】波峰因子 {t.get('crest_factor', 0):.2f}，峭度 {t.get('kurtosis', 0):.2f}，"
        f"偏度 {t.get('skewness', 0):.2f}，RMS {t.get('rms', 0):.4f}，冲击因子 {t.get('impulse_factor', 0):.2f}"
    )

    kurt_bands = f.get("kurtosis_bands", [])
    if kurt_bands:
        parts = [f"{b['band_hz'][0]:.0f}–{b['band_hz'][1]:.0f} Hz({b['kurtosis']:.2f})" for b in kurt_bands]
        lines.append(f"【频域】谱峭度最高频带：" + "、".join(parts))

    psd = f.get("psd", {})
    psd_bands = [
        band
        for band in psd.get("band_power_ratios", [])
        if isinstance(band, dict) and band.get("observable")
    ]
    if psd_bands:
        strongest = sorted(
            psd_bands, key=lambda band: float(band.get("power_ratio", 0.0)), reverse=True
        )[:3]
        parts = [
            f"{band['band_hz'][0]:.0f}–{band['band_hz'][1]:.0f} Hz "
            f"{100.0 * float(band.get('power_ratio', 0.0)):.1f}%"
            for band in strongest
        ]
        lines.append("【PSD】Welch 频带能量占比最高：" + "、".join(parts))

    lines.append(
        f"【包络域】理论故障频率 BPFO={freqs.get('BPFO', 0)} Hz、"
        f"BPFI={freqs.get('BPFI', 0)} Hz、BSF={freqs.get('BSF', 0)} Hz"
    )
    for family in FAMILY_ORDER:
        fam = families.get(family, {})
        lines.append(
            f"  {family}：基频在 {fam.get('fundamental_bands', 0)} 个频带命中，"
            f"谐波阶次 {fam.get('qualified_orders', [])}"
        )
    return "\n".join(lines)


# ------------------------------------------------------------ 总入口
def extract_features(
    signal: Sequence[float] | np.ndarray,
    sampling_rate_hz: float,
    rpm: float,
    geometry: Mapping[str, Any] | None,
    *,
    matching: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """提取时频域特征，返回特征字典 + 特征向量 + 描述文本。"""

    x = np.asarray(signal, dtype=float).reshape(-1)
    fs = float(sampling_rate_hz)
    m = {**load_config()["matching"], **(matching or {})}

    time_feats = time_domain_features(x)
    freq_feats = frequency_domain_features(x, fs)
    env_feats = envelope_domain_features(x, fs, rpm, geometry, m)

    features = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fs": fs,
        "rpm": float(rpm),
        "duration_s": round(x.size / fs, 3),
        "time": time_feats,
        "frequency": freq_feats,
        "envelope": env_feats,
    }
    features["feature_vector"] = feature_vector(features)
    features["description"] = describe(features)
    return features


def retrieval_features_from_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """从主链 EvidenceBundle 快照构造检索特征，不重复计算信号处理。

    接受 ``EvidenceBundle.model_dump()`` 或等价 Mapping。检索向量和描述仍复用
    本模块的统一实现，因此案例库接口不需要改变。
    """

    signal_features = evidence.get("signal_features", {})
    if not isinstance(signal_features, Mapping):
        signal_features = {}
    time_features = signal_features.get("time_domain", {})
    if not isinstance(time_features, Mapping):
        time_features = {}
    psd = signal_features.get("psd", {})
    if not isinstance(psd, Mapping):
        psd = {}
    kurtosis_bands = signal_features.get("kurtosis_bands", [])
    if not isinstance(kurtosis_bands, list):
        kurtosis_bands = []

    assessments = evidence.get("assessments", {})
    if not isinstance(assessments, Mapping):
        assessments = {}
    families: dict[str, dict[str, Any]] = {}
    for family in FAMILY_ORDER:
        assessment = assessments.get(family, {})
        if not isinstance(assessment, Mapping):
            assessment = {}
        families[family] = {
            "fundamental_hz": round(
                float(evidence.get("fault_frequencies_hz", {}).get(family, 0.0)),
                2,
            ),
            "fundamental_bands": int(assessment.get("fundamental_bands", 0)),
            "qualified_orders": list(assessment.get("qualified_orders", [])),
        }

    context = evidence.get("operating_context", {})
    if not isinstance(context, Mapping):
        context = {}
    features: dict[str, Any] = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fs": float(context.get("sampling_rate_hz") or 0.0),
        "rpm": float(context.get("rpm") or 0.0),
        "duration_s": round(float(context.get("duration_s") or 0.0), 3),
        "time": dict(time_features),
        "frequency": {
            "kurtosis_bands": list(kurtosis_bands),
            "psd": dict(psd),
        },
        "envelope": {
            "fault_frequencies_hz": {
                str(key): round(float(value), 2)
                for key, value in evidence.get("fault_frequencies_hz", {}).items()
            },
            "families": families,
        },
    }
    features["feature_vector"] = feature_vector(features)
    features["description"] = describe(features)
    return features
