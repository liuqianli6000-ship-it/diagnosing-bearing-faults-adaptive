"""Render the four-paragraph diagnosis from the fixed verdict and current evidence."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

from bearing_suite_app.assessment.band_banks import BAND_METHOD_NAMES
from bearing_suite_app.diagnosis_schemas import FAULT_TO_FAMILY


FAULT_NAMES = {
    "normal": "正常",
    "inner_race_fault": "内圈故障",
    "outer_race_fault": "外圈故障",
    "rolling_element_fault": "滚动体故障",
    "unknown": "无法唯一判定",
}
SPECIFIC_FAULTS = {key for key in FAULT_NAMES if key not in {"normal", "unknown"}}


def _conflict_candidate(numeric: Mapping[str, Any]) -> tuple[str | None, str]:
    """Use an existing branch verdict to prioritize review after a conflict."""
    segment = numeric.get("segment_consensus") or {}
    if not segment.get("conflict"):
        return None, ""
    baseline = numeric.get("baseline")
    primary = baseline.get("fault_type") if isinstance(baseline, Mapping) else getattr(baseline, "fault_type", None)
    primary = primary or segment.get("full_fault")
    if primary in SPECIFIC_FAULTS:
        return primary, "整段包络谱"
    counts = Counter(item.get("fault_type") for item in segment.get("segments", []) if item.get("analyzable"))
    candidates = [fault for fault in SPECIFIC_FAULTS if counts[fault] > 0]
    if candidates:
        highest = max(counts[fault] for fault in candidates)
        leading = [fault for fault in candidates if counts[fault] == highest]
        if len(leading) == 1:
            return leading[0], "可分析分段"
    return None, ""


def _measurement(label: str, value: Any, unit: str = "") -> str:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return f"{label}{value:.2f}".rstrip("0").rstrip(".") + unit
    return ""


def _matched_peaks(evidence: Mapping[str, Any], family: str) -> str:
    for band in evidence.get("band_evidence") or []:
        bounds = band.get("band_hz") or []
        if len(bounds) != 2 or (band.get("diagnosability") or {}).get("valid") is False:
            continue
        matches = []
        for peak in ((band.get("families") or {}).get(family) or {}).get("harmonic_peaks", []):
            target = _measurement("理论", peak.get("target_frequency_hz"), " Hz")
            observed = _measurement("实测", peak.get("observed_frequency_hz"), " Hz")
            if peak.get("qualified") and target and observed and peak.get("harmonic"):
                matches.append(f"第{peak['harmonic']}阶{target}、{observed}")
        if matches:
            return f"在{bounds[0]:.2f}–{bounds[1]:.2f} Hz解调带内，通过峰值筛选的例子有" + "和".join(matches[:2]) + "。"
    return ""


def build_narrative(state: Mapping[str, Any]) -> dict[str, Any]:
    assessment = state.get("assessment") or {}
    numeric = state.get("numeric_result") or {}
    final = state.get("final") or {}
    quality = assessment.get("quality") or {}
    meta = assessment.get("meta") or {}
    speed = state.get("speed_reference") or {}
    fault_type = final.get("fault_type") or numeric.get("fault_type") or "unknown"
    source = final.get("source")
    category = FAULT_NAMES.get(fault_type, FAULT_NAMES["unknown"])
    evidence = numeric.get("evidence") or {}
    if hasattr(evidence, "model_dump"):
        evidence = evidence.model_dump()
    limited = (not numeric and state.get("status") in {"limited", "rejected"}) or quality.get("valid_for_analysis") is False
    likely, candidate_source = (
        _conflict_candidate(numeric)
        if source not in {"skill_llm", "evidence_only"} and fault_type == "unknown" and not limited else (None, "")
    )

    conditions = ["本次采用自适应诊断" if state.get("adaptive") else "本次采用稳定诊断"]
    if speed.get("method") == "harmonic_fit":
        conditions.append(f"输入转速{speed['nominal_rpm']:g}转/分，按波形估速采用{speed['rpm']:.2f}转/分")
    elif speed.get("method") == "nominal_fallback":
        conditions.append(f"估速未通过采用条件，沿用输入转速{speed['rpm']:g}转/分")
    else:
        conditions.append(_measurement("输入转速", meta.get("rpm"), " 转/分"))
    conditions.extend(filter(None, (
        _measurement("采样率", meta.get("sampling_rate_hz"), " Hz"),
        _measurement("记录时长", meta.get("duration_seconds"), " 秒"),
    )))
    condition = "，".join(filter(None, conditions)) + "。"
    if limited:
        condition += "本次输入质量或可诊断条件受限，具体限制见本次运行记录。"
    elif quality.get("valid_for_analysis") is True:
        condition += "信号通过输入质量检查，诊断范围以当前记录的可观测频率为限。"

    time = assessment.get("time_domain") or {}
    frequency = assessment.get("frequency_domain") or {}
    tf = assessment.get("time_frequency") or {}
    peaks = frequency.get("dominant_peaks") or []
    metrics = list(filter(None, (
        _measurement("时域峭度为", time.get("kurtosis")),
        _measurement("频谱主峰约为", peaks[0].get("frequency_hz") if peaks else None, " Hz"),
        _measurement("时频分析的高能量帧数为", tf.get("transient_count"), " 帧"),
    )))
    if metrics:
        overall = "，".join(metrics) + "。这些指标用于描述当前信号状态，故障归属还需结合包络谱的倍频匹配与一致性结果。"
    else:
        overall = "本次未取得可写入说明的时域、频域或时频域指标，信号的幅值分布、谱峰和短时能量变化暂无法描述。"

    bands = numeric.get("bands") or []
    primary_candidate = likely if candidate_source == "整段包络谱" else None
    family = FAULT_TO_FAMILY.get(primary_candidate or fault_type, "")
    support = (evidence.get("assessments") or {}).get(family) or {}
    features = []
    used = numeric.get("used_methods") or []
    if used:
        names = "、".join(BAND_METHOD_NAMES[method] for method in used)
        route = numeric["evidence_route"]
        features.append({
            "primary": f"本次采用{names}的包络结果。",
            "candidate": f"快速谱峭度证据不足，本次采用{names}的候选包络结果。",
            "combined": f"单种方法的证据不足以确定类别，本次对照{names}的包络结果。",
        }[route])
    if bands:
        features.append(f"对{len(bands)}个解调频带进行平方包络匹配，比较内圈、外圈和滚动体的理论特征频率及其倍频。")
        if support.get("evidence_bands"):
            features.append(f"有{support['evidence_bands']}个解调带检出成组倍频。")
            if support.get("fit_supported_bands") is not None:
                features.append(f"其中{support['fit_supported_bands']}个带同时获得同带谱列拟合支持。")
        features.append(_matched_peaks(evidence, family))
        for method in used[1:]:
            candidate = numeric["band_methods"][method]["evidence"]
            if hasattr(candidate, "model_dump"):
                candidate = candidate.model_dump()
            if sample := _matched_peaks(candidate, family):
                features.append(BAND_METHOD_NAMES[method] + "：" + sample)
    else:
        features.append("本次未取得可用的解调频带结果，缺少可引用的包络谱峰与故障倍频匹配记录。")
    segment = numeric.get("segment_consensus") or {}
    if segment:
        if not segment.get("checked"):
            features.append("记录长度或转数未达到分段检查条件。")
        elif segment.get("conflict"):
            features.append("分段结果出现冲突。")
        elif segment.get("consensus"):
            features.append("分段检查与整段结果一致。")
        else:
            features.append("分段结果对整段结论的支持不足。")

    follow_up = ""
    if segment.get("conflict") or numeric.get("segment_instability"):
        follow_up = "需进一步分析各分段的故障倍频及其重复性。"
    headline = f"判断为：{category}。"
    judgment = headline + {
        "normal": "在本次可观测范围内，数值证据未达到故障定位条件，结论限于本次测量。",
        "unknown": "现有输入与证据不足以给出唯一结论，复核应优先处理上文列出的限制。",
    }.get(fault_type, "故障部位依据本次固定数值裁决给出，判断限于当前工况及频率覆盖范围。")
    if likely:
        headline = f"初步判断：最可能为{FAULT_NAMES[likely]}，需进一步分析。"
        judgment = f"初步判断：最可能为{FAULT_NAMES[likely]}（依据{candidate_source}的数值结果）。{follow_up}当前数值裁决仍待确认。"
    elif follow_up:
        judgment += follow_up
    if source == "skill_llm":
        follow_up = final.get("follow_up") or ""
        if fault_type in SPECIFIC_FAULTS and follow_up:
            headline = f"初步判断：最可能为{category}，需进一步分析。"
            likely = fault_type
        elif fault_type == "normal":
            headline = "当前工况下轴承振动表现正常。"
        judgment = headline + final["reasoning"] + (" " + follow_up if follow_up else "")
    elif source == "evidence_only":
        headline = "本次已完成数值取证。"
        judgment = final["reasoning"]

    def render():
        return "\n\n".join(("【分析条件】" + condition, "【整体表现】" + overall,
                            "【故障特征证据】" + "".join(features), "【综合判断】" + judgment))

    # Sparse runs can report their limits without filling the text with invented measurements.
    for explanation in (
        "转频及其整数倍也可能产生局部谱峰，部位判断需要综合同一频带的峰位关系及各频带的支持情况。",
        "是否能够识别特征频率取决于记录长度、频率分辨率和解调带宽，应按本次工具记录解释覆盖范围。",
        "复核时应把理论特征频率与本次实测峰逐项对应，并检查分段与整段结果是否指向同一部位。",
        "若现有记录不足，后续可在确认采样率、转速和轴承几何后补充可分析的振动信号。",
    ):
        if len("".join(render().split())) >= 300:
            break
        features.append(explanation)
    reason = render()
    return {
        "headline": headline,
        "text": reason,
        "diagnosis": {"fault_type": fault_type, "reason": reason},
        **({"likely_fault_type": likely} if likely else {}),
    }
