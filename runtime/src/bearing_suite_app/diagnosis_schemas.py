"""诊断智能体的结构化证据与 LLM 输出契约。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FaultType = Literal[
    "normal",
    "inner_race_fault",
    "outer_race_fault",
    "rolling_element_fault",
    "unknown",
]
LeadingFamily = Literal["BPFI", "BPFO", "BSF", "NONE", "uncertain"]
DiagnosisStatus = Literal["diagnosed", "normal", "unknown"]
ConfidenceLevel = Literal["high", "medium", "low", "unknown"]
VisualFamily = Literal["BPFI", "BPFO", "BSF", "NONE", "uncertain"]

FAULT_TO_FAMILY: dict[str, str] = {
    "inner_race_fault": "BPFI",
    "outer_race_fault": "BPFO",
    "rolling_element_fault": "BSF",
    "normal": "NONE",
    "unknown": "uncertain",
}

DEFAULT_CONFIDENCE_MAPPING: dict[str, float] = {
    "high": 0.85,
    "medium": 0.65,
    "low": 0.40,
    "unknown": 0.0,
}


class EvidenceBundle(BaseModel):
    """工具链产出的可审计证据；本结构本身不包含最终故障标签。"""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "bearing-evidence/v1"
    analyzable: bool = True
    limitation_reason: str | None = None
    quality: dict[str, Any] = Field(default_factory=dict)
    diagnosability: dict[str, Any] = Field(default_factory=dict)
    operating_context: dict[str, Any] = Field(default_factory=dict)
    fault_frequencies_hz: dict[str, float] = Field(default_factory=dict)
    selected_bands_hz: list[list[float]] = Field(default_factory=list)
    band_evidence: list[dict[str, Any]] = Field(default_factory=list)
    assessments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    visual_evidence: list[dict[str, Any]] = Field(default_factory=list)
    signal_features: dict[str, Any] = Field(default_factory=dict)
    similar_cases: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: dict[str, Any] = Field(default_factory=dict)

    def evidence_ids(self) -> set[str]:
        """返回当前证据包内允许被 LLM 引用的证据 ID。"""

        ids = {"quality.signal", "diagnosability.gate"}
        for stage in self.diagnosability.get("stages", {}).values():
            if isinstance(stage, dict) and stage.get("evidence_id"):
                ids.add(str(stage["evidence_id"]))
        for family, assessment in self.assessments.items():
            ids.add(str(assessment.get("evidence_id") or f"assessment.{family}"))
        for band in self.band_evidence:
            if band.get("evidence_id"):
                ids.add(str(band["evidence_id"]))
            families = band.get("families", {})
            if not isinstance(families, dict):
                continue
            for family_data in families.values():
                if not isinstance(family_data, dict):
                    continue
                if family_data.get("evidence_id"):
                    ids.add(str(family_data["evidence_id"]))
                for peak in family_data.get("harmonic_peaks", []):
                    if isinstance(peak, dict) and peak.get("evidence_id"):
                        ids.add(str(peak["evidence_id"]))
        for case in self.similar_cases:
            if isinstance(case, dict) and case.get("evidence_id"):
                ids.add(str(case["evidence_id"]))
        for key in ("psd", "psd_normality"):
            feature = self.signal_features.get(key)
            if isinstance(feature, dict) and feature.get("evidence_id"):
                ids.add(str(feature["evidence_id"]))
        return ids

    def to_llm_payload(self, *, include_weak_peaks: bool = False,
                       include_shared_peaks: bool = False) -> dict[str, Any]:
        """生成供 LLM 判读的精简证据；完整 EvidenceBundle 仍用于输出审计。

        ``visual_evidence`` 不进入核心文本载荷；启用多模态时只能由受约束的
        图像观察工具按需读取，也不能作为独立诊断依据。
        """

        quality_keys = (
            "evidence_id",
            "valid_for_analysis",
            "sample_count",
            "duration_seconds",
            "rms",
            "clipping_ratio",
            "trend_ratio",
            "dropout_run_ratio",
            "invalidating_flags",
        )
        quality_summary = {
            key: self.quality[key] for key in quality_keys if key in self.quality
        }

        gate_keys = ("evidence_id", "passed", "invalidating_flags")
        diagnosability_summary = {
            key: self.diagnosability[key]
            for key in gate_keys
            if key in self.diagnosability
        }
        frequency_stage = self.diagnosability.get("stages", {}).get(
            "frequency_observability", {}
        )
        if isinstance(frequency_stage, dict):
            diagnosability_summary["frequency_resolution_hz"] = frequency_stage.get(
                "frequency_resolution_hz"
            )
            diagnosability_summary["analysis_range_hz"] = frequency_stage.get(
                "analysis_range_hz"
            )

        assessment_summary: dict[str, dict[str, Any]] = {}
        supported_families: set[str] = set()
        for family in ("BPFO", "BPFI", "BSF"):
            assessment = self.assessments.get(family, {})
            level = str(assessment.get("evidence_level", "none"))
            assessment_summary[family] = {
                "evidence_id": str(
                    assessment.get("evidence_id") or f"assessment.{family}"
                ),
                "evidence_level": level,
                # 同一带内有基频及其他合格倍频的支持带数。
                "evidence_bands": int(assessment.get("evidence_bands", 0)),
                "fit_supported_bands": assessment.get("fit_supported_bands"),
                "lattice_models": dict(assessment.get("lattice_models") or {}),
                "median_residual_bins": assessment.get("median_residual_bins"),
                # 基频命中带数包含孤立基频。
                "fundamental_bands": int(assessment.get("fundamental_bands", 0)),
                "qualified_orders": list(assessment.get("qualified_orders", [])),
                "fundamental_collides": bool(
                    assessment.get("fundamental_collides", False)
                ),
            }
            if include_weak_peaks or level in {"moderate", "strong"}:
                supported_families.add(family)

        band_summaries: list[dict[str, Any]] = []
        qualified_peak_candidates: dict[str, list[dict[str, Any]]] = {
            family: [] for family in supported_families
        }
        for band in self.band_evidence:
            family_summaries: dict[str, dict[str, Any]] = {}
            families = band.get("families", {})
            band_quality = band.get("diagnosability", {})
            valid_band = isinstance(band_quality, dict) and band_quality.get("valid") is True
            if isinstance(families, dict):
                for family in ("BPFO", "BPFI", "BSF"):
                    family_data = families.get(family, {})
                    if not isinstance(family_data, dict):
                        continue
                    orders = [int(v) for v in family_data.get("qualified_orders", [])]
                    family_summaries[family] = {
                        "evidence_id": str(
                            family_data.get("evidence_id")
                            or f"{band.get('evidence_id', 'band')}.{family}"
                        ),
                        "fundamental_hit": 1 in orders,
                        "qualified_orders": orders,
                    }
                    if include_weak_peaks:
                        lattice = family_data.get("lattice") or {}
                        family_summaries[family]["lattice"] = {
                            key: lattice[key] for key in (
                                "best_model", "fault_supported", "fitted_is_interference",
                                "refined_qualified_orders", "fitted_frequency_hz",
                                "residual_rms_bins", "accept_threshold_bins",
                            ) if key in lattice
                        }
                        if valid_band:
                            family_summaries[family]["grouped_peak_match"] = family_data.get("grouped_peak_match")
                            family_summaries[family]["grouped_fit_supported"] = family_data.get("grouped_fit_supported")
                            qualified_peaks = [
                                peak for peak in family_data.get("harmonic_peaks", [])
                                if peak.get("qualified") and peak.get("observed_frequency_hz") is not None
                            ]
                            family_summaries[family]["matched_peaks_hz"] = {
                                str(peak["harmonic"]): peak["observed_frequency_hz"]
                                for peak in qualified_peaks
                            }
                            family_summaries[family]["matched_peak_evidence"] = [
                                {key: peak[key] for key in (
                                    "evidence_id", "harmonic", "target_frequency_hz", "observed_frequency_hz",
                                    "error_hz", "tolerance_hz", "local_robust_z",
                                ) if key in peak}
                                for peak in qualified_peaks if peak.get("evidence_id")
                            ]
                        else:
                            family_summaries[family].update(fundamental_hit=False, qualified_orders=[])
                            continue
                    if family not in supported_families:
                        continue
                    for peak in family_data.get("harmonic_peaks", []):
                        if not isinstance(peak, dict) or not peak.get("qualified"):
                            continue
                        qualified_peak_candidates[family].append(
                            {
                                "evidence_id": peak.get("evidence_id"),
                                "band_evidence_id": band.get("evidence_id"),
                                "family": family,
                                "harmonic": int(peak.get("harmonic", 0)),
                                "target_frequency_hz": peak.get("target_frequency_hz"),
                                "observed_frequency_hz": peak.get("observed_frequency_hz"),
                                "relative_amplitude": peak.get("relative_amplitude"),
                                "local_robust_z": peak.get("local_robust_z"),
                            }
                        )

            if include_weak_peaks and include_shared_peaks and valid_band:
                peaks_by_frequency: dict[float, list[tuple[str, dict[str, Any]]]] = {}
                for family, summary in family_summaries.items():
                    for peak in summary.get("matched_peak_evidence", []):
                        # Each match uses a bin from this band's envelope spectrum.
                        peaks_by_frequency.setdefault(peak["observed_frequency_hz"], []).append((family, peak))
                for matches in peaks_by_frequency.values():
                    for family, peak in matches:
                        shared = [other["evidence_id"] for other_family, other in matches if other_family != family]
                        if shared:
                            peak["shared_with"] = shared
                for summary in family_summaries.values():
                    summary["unshared_orders"] = sorted({
                        peak["harmonic"] for peak in summary.get("matched_peak_evidence", [])
                        if not peak.get("shared_with")
                    })

            band_summaries.append(
                {
                    "evidence_id": band.get("evidence_id"),
                    "band_hz": band.get("band_hz"),
                    "band_energy_ratio": band.get("band_energy_ratio"),
                    "envelope_snr_db": band.get("envelope_snr_db"),
                    "valid_for_demodulation": (
                        band_quality.get("valid")
                        if isinstance(band_quality, dict)
                        else None
                    ),
                    "families": family_summaries,
                }
            )

        key_peaks: list[dict[str, Any]] = []
        for family in ("BPFO", "BPFI", "BSF"):
            candidates = qualified_peak_candidates.get(family, [])
            candidates.sort(
                key=lambda peak: (
                    0 if peak["harmonic"] == 1 else 1,
                    peak["harmonic"],
                    -float(peak.get("local_robust_z") or 0.0),
                )
            )
            key_peaks.extend(candidates[:3])

        feature_summary: dict[str, Any] = {}
        if isinstance(self.signal_features.get("time_domain"), dict):
            feature_summary["time_domain"] = self.signal_features["time_domain"]
        psd = self.signal_features.get("psd")
        if isinstance(psd, dict):
            observable_bands = [
                band
                for band in psd.get("band_power_ratios", [])
                if isinstance(band, dict) and band.get("observable")
            ]
            strongest_bands = sorted(
                observable_bands,
                key=lambda band: float(band.get("power_ratio", 0.0)),
                reverse=True,
            )[:4]
            feature_summary["psd"] = {
                key: psd[key]
                for key in (
                    "evidence_id",
                    "method",
                    "profile_version",
                    "sampling_rate_hz",
                    "frequency_resolution_hz",
                    "reliable_range_hz",
                )
                if key in psd
            }
            feature_summary["psd"]["strongest_band_power_ratios"] = [
                {
                    "band_hz": band.get("band_hz"),
                    "power_ratio": band.get("power_ratio"),
                }
                for band in strongest_bands
            ]
        psd_normality = self.signal_features.get("psd_normality")
        if isinstance(psd_normality, dict):
            feature_summary["psd_normality"] = {
                key: psd_normality[key]
                for key in (
                    "evidence_id",
                    "status",
                    "reason",
                    "reference_count",
                    "reference_source_count",
                    "robust_distance",
                    "similarity_score",
                    "abnormal_band_count",
                    "abnormal_bands",
                    "role",
                    "thresholds",
                    "condition_scope",
                )
                if key in psd_normality
            }
        if self.signal_features.get("feature_description"):
            feature_summary["feature_description"] = self.signal_features[
                "feature_description"
            ]

        similar_case_summary = []
        for case in self.similar_cases:
            if not isinstance(case, dict):
                continue
            similar_case_summary.append(
                {
                    key: case[key]
                    for key in (
                        "evidence_id",
                        "case_id",
                        "fault_family",
                        "fault_type",
                        "score",
                        "psd_similarity",
                        "similarity_basis",
                        "trust_level",
                        "use_boundary",
                    )
                    if key in case
                }
            )

        return {
            "schema_version": "bearing-evidence-llm/v1",
            "analyzable": self.analyzable,
            "limitation_reason": self.limitation_reason,
            "quality": quality_summary,
            "diagnosability": diagnosability_summary,
            "operating_context": self.operating_context,
            "fault_frequencies_hz": self.fault_frequencies_hz,
            "selected_bands_hz": self.selected_bands_hz,
            "assessments": assessment_summary,
            "band_summaries": band_summaries,
            "key_peaks": key_peaks,
            "signal_features": feature_summary,
            "similar_cases": similar_case_summary,
            "analysis_method": {
                key: self.tool_trace[key]
                for key in (
                    "band_selection",
                    "envelope_mode",
                    "envelope_resolution_hz",
                    "case_retrieval",
                    "psd_normal_reference",
                )
                if key in self.tool_trace
            },
        }


class LLMDiagnosis(BaseModel):
    """LLM 基于 EvidenceBundle 给出的最终结构化诊断。"""

    model_config = ConfigDict(extra="forbid")

    fault_type: FaultType
    leading_family: LeadingFamily
    status: DiagnosisStatus
    confidence_level: ConfidenceLevel
    reasoning: str = Field(min_length=1)
    evidence_refs: list[str] = Field(min_length=1)
    conflicts: list[str]

    @model_validator(mode="after")
    def validate_consistency(self) -> "LLMDiagnosis":
        expected_family = FAULT_TO_FAMILY[self.fault_type]
        if self.leading_family != expected_family:
            raise ValueError(
                f"fault_type={self.fault_type} 时 leading_family 必须为 {expected_family}"
            )
        expected_status = (
            "normal"
            if self.fault_type == "normal"
            else "unknown"
            if self.fault_type == "unknown"
            else "diagnosed"
        )
        if self.status != expected_status:
            raise ValueError(f"fault_type={self.fault_type} 时 status 必须为 {expected_status}")
        if self.fault_type == "unknown" and self.confidence_level != "unknown":
            raise ValueError("unknown 诊断的 confidence_level 必须为 unknown")
        if self.fault_type != "unknown" and self.confidence_level == "unknown":
            raise ValueError("非 unknown 诊断不能使用 unknown confidence_level")
        return self

    def confidence(self, mapping: dict[str, float] | None = None) -> float:
        values = dict(DEFAULT_CONFIDENCE_MAPPING)
        if mapping:
            values.update({str(k): float(v) for k, v in mapping.items()})
        return min(max(float(values[self.confidence_level]), 0.0), 1.0)


class VisualFinding(BaseModel):
    """多模态观察器的一条受约束观察，不是独立诊断。"""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(min_length=1)
    family: VisualFamily
    orders: list[int]
    supporting_band_ids: list[str]
    confidence_level: ConfidenceLevel
    description: str = Field(min_length=1)
    numeric_evidence_refs: list[str] = Field(min_length=1)


class VisualObservation(BaseModel):
    """包络谱图像观察契约。"""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    image_sha256: str
    observable: bool
    observations: list[VisualFinding]
    conflicts: list[str]
    limitations: list[str]

    @model_validator(mode="after")
    def validate_observability(self) -> "VisualObservation":
        if not self.observable and self.observations:
            raise ValueError("observable=false 时 observations 必须为空")
        return self


def validate_evidence_references(
    diagnosis: LLMDiagnosis,
    evidence: EvidenceBundle,
    additional_evidence_ids: set[str] | None = None,
) -> list[str]:
    """校验 LLM 是否引用了真实存在的工具证据。"""

    if not diagnosis.evidence_refs:
        return ["LLM 诊断必须至少引用一条工具证据"]
    available = evidence.evidence_ids() | set(additional_evidence_ids or set())
    missing = [ref for ref in diagnosis.evidence_refs if ref not in available]
    errors = [f"不存在的证据引用: {ref}" for ref in missing]
    if missing:
        return errors
    if diagnosis.status == "diagnosed":
        if not evidence.analyzable or evidence.diagnosability.get("passed") is not True:
            errors.append("具体故障要求通过可诊断性门")
        assessment = evidence.assessments.get(diagnosis.leading_family, {})
        if assessment.get("evidence_level") not in {"moderate", "strong"}:
            errors.append(f"具体故障要求 {diagnosis.leading_family} 达到 moderate/strong")
        relevant = [
            ref
            for ref in diagnosis.evidence_refs
            if ref == f"assessment.{diagnosis.leading_family}"
            or f".{diagnosis.leading_family}" in ref
        ]
        if not relevant:
            errors.append(f"具体故障必须引用 {diagnosis.leading_family} 的工具证据")
    elif diagnosis.status == "normal":
        if not evidence.analyzable or evidence.diagnosability.get("passed") is not True:
            errors.append("normal 要求通过可诊断性门")
        if any(evidence.assessments.get(family, {}).get("evidence_level") != "none" for family in ("BPFO", "BPFI", "BSF")):
            errors.append("normal 要求三族 evidence_level 均为 none")
        required = {"assessment.BPFO", "assessment.BPFI", "assessment.BSF"}
        if not required.issubset(set(diagnosis.evidence_refs)):
            errors.append("normal 诊断必须引用 BPFO/BPFI/BSF 三族 assessment 证据")
    return errors
