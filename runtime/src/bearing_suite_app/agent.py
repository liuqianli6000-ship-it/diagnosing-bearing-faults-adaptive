"""通用轴承诊断智能体（工具取证 + LLM 诊断解释）。

工具链负责信号质量检查、故障频率计算、选带、包络谱与谐波匹配，形成不含当前样本
最终标签的结构化证据包。LLM 基于证据给出最终诊断和解释；确定性规则仅作为可审计
基线以及 LLM 不可用、超时或输出无效时的回退。

路径：
1. ``diagnose()`` / ``diagnose_deterministic()``：确定性基线兼容入口。
2. ``diagnose_agentic()``：工具取证 + LLM 最终诊断，失败时回退基线。
3. ``diagnose_rag()``：工具取证 + 去锚定案例检索 + LLM 最终诊断，失败时回退基线。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import case_kb as kb
from . import agent_runtime
from . import chain
from . import decision
from . import diagnosability as diag_gate
from . import features as features_mod
from . import geometry as geom
from . import matcher
from . import run_store as audit_store
from . import sample_case
from . import signal_tools as sig
from .agent_state import AgentState
from .skill_loader import DiagnosisSkillLoader
from .dual_channel import diagnose_image, summarize_channels
from .visual_observer import MultimodalEvidenceObserver
from .config import load_config
from .diagnosis_schemas import (
    DEFAULT_CONFIDENCE_MAPPING,
    EvidenceBundle,
    LLMDiagnosis,
    validate_evidence_references,
)

FAULT_TO_FAMILY = {
    "outer_race_fault": "BPFO",
    "inner_race_fault": "BPFI",
    "rolling_element_fault": "BSF",
    "normal": "NONE",
    "unknown": "uncertain",
}
VALID_FAULT_TYPES = set(FAULT_TO_FAMILY)
FAULT_TYPE_ZH = {
    "outer_race_fault": "外圈故障",
    "inner_race_fault": "内圈故障",
    "rolling_element_fault": "滚动体故障",
    "normal": "正常",
    "unknown": "未知",
}
EVIDENCE_LEVEL_ZH = {"strong": "强", "moderate": "中等", "none": "不足"}
CONFIDENCE_LEVEL_ZH = {
    "high": "高",
    "medium": "中",
    "low": "低",
    "unknown": "未知",
}


@dataclass
class DiagnosisResult:
    fault_type: str
    fault_type_zh: str
    leading_family: str
    confidence: float
    status: str  # diagnosed | normal | unknown
    reason: str
    fault_frequencies_hz: dict[str, float]
    assessments: dict[str, dict]
    similar_cases: list[dict]
    explanation: str
    resonance_band_hz: list[list[float]] = field(default_factory=list)
    envelope_resolution_hz: float = 0.0
    llm_used: bool = False
    diagnosis_code: str = "unknown"
    diagnosis_source: str = "deterministic"
    baseline_verdict: dict[str, Any] = field(default_factory=dict)
    llm_diagnosis: dict[str, Any] = field(default_factory=dict)
    fallback_reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    visual_evidence: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None
    audit_path: str | None = None
    decision_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BearingDiagnosisAgent:
    """可解释轴承诊断智能体（稳态专用，时变链预留）。"""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        config_path: str | None = None,
        case_kb_path: str | None = None,
        llm: Any | None = None,
        sample_store: Any | None = None,
        artifact_dir: str | None = None,
        trace_dir: str | None = None,
        multimodal: bool | None = None,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.config = config if config is not None else load_config(config_path)
        self.case_kb_path = case_kb_path
        self.llm = llm
        self.sample_store = sample_store
        self.artifact_dir = artifact_dir
        self.trace_dir = trace_dir
        self.multimodal = multimodal
        self.progress_callback = progress_callback
        self._cases: list[dict[str, Any]] | None = None

    def _emit_progress(self, event: str, **details: Any) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event, details)
        except Exception:
            # 进度展示属于观察层，不得改变诊断结果。
            return

    def _start_audit_run(
        self,
        mode: str,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
    ) -> tuple[audit_store.DiagnosticRunStore | None, int]:
        configured = self.config.get("audit", {})
        if not isinstance(configured, Mapping):
            configured = {}
        root = self.trace_dir or configured.get("trace_dir", "outputs/runs")
        enabled = bool(self.trace_dir or configured.get("enabled", False))
        exchange_start = len(getattr(self.llm, "exchanges", []) or [])
        if not enabled:
            return None, exchange_start
        try:
            values, normalization = diag_gate.normalize_real_series(signal)
            store = audit_store.DiagnosticRunStore(
                str(root),
                mode=mode,
                config=self.config,
                input_metadata={
                    "sampling_rate_hz": float(sampling_rate_hz),
                    "rpm": float(rpm),
                    "sample_count": int(values.size),
                    "signal_sha256": (
                        audit_store.signal_sha256(values)
                        if normalization["valid"]
                        else None
                    ),
                    "signal_input_normalization": normalization,
                    "bearing_geometry": dict(geometry or {}),
                },
            )
            return store, exchange_start
        except Exception:
            # 审计层不可阻断诊断；具体失败会在最终 decision_trace 中体现。
            return None, exchange_start

    def _finish_audit_run(
        self,
        store: audit_store.DiagnosticRunStore | None,
        exchange_start: int,
        evidence: EvidenceBundle,
        result: DiagnosisResult,
    ) -> DiagnosisResult:
        if store is None:
            return result
        try:
            store.record_evidence(
                evidence.model_dump(mode="json"),
                evidence.to_llm_payload(),
            )
            store.append_event(
                "baseline_recorded",
                {"baseline_verdict": result.baseline_verdict},
            )
            exchanges = list(getattr(self.llm, "exchanges", []) or [])[exchange_start:]
            for attempt, exchange in enumerate(exchanges, start=1):
                if isinstance(exchange, Mapping):
                    store.record_exchange(exchange, attempt=attempt)
            runtime_trace = result.decision_trace.get("agent_runtime")
            if isinstance(runtime_trace, Mapping):
                store.append_event("agent_runtime_recorded", dict(runtime_trace))
            result.run_id = store.run_id
            result.audit_path = str(store.run_dir)
            result.decision_trace["audit"] = {
                "run_id": store.run_id,
                "path": str(store.run_dir),
                "llm_exchange_count": len(exchanges),
            }
            store.finalize(result.to_dict())
        except Exception as exc:  # noqa: BLE001
            result.decision_trace["audit_error"] = f"{type(exc).__name__}: {exc}"
        return result

    # ------------------------------------------------------------------ 工具
    def _cases_loaded(self) -> list[dict[str, Any]]:
        if self._cases is None:
            self._cases = kb.load_cases(self.case_kb_path) if self.case_kb_path else []
        return self._cases

    def signal_quality(
        self,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float | None = None,
        rpm: float | None = None,
        *,
        speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
    ) -> dict[str, Any]:
        """输入质量工具；缺少采样率或转速时返回不可诊断，而不是抛异常。"""

        values, normalization = diag_gate.normalize_real_series(signal)
        return diag_gate.assess_input_quality(
            values,
            float("nan") if sampling_rate_hz is None else sampling_rate_hz,
            float("nan") if rpm is None else rpm,
            self.config["signal"]["quality"],
            speed_trace_rpm=speed_trace_rpm,
            input_normalization=normalization,
        )

    def _unknown_result(
        self,
        reason: str,
        trace: dict[str, Any] | None = None,
        *,
        evidence: EvidenceBundle | None = None,
        diagnosis_code: str = "unknown_quality_limited",
    ) -> DiagnosisResult:
        """质量门失败/主链受阻/输入非法时的 unknown 结论。

        assessments 恒含 BPFI/BPFO/BSF 三族（evidence_level=none），
        保证任何路径的输出都能通过 diagnosis-output-contract 校验。
        """

        empty = {
            "evidence_level": "none",
            "fundamental_bands": 0,
            "qualified_orders": [],
            "fundamental_collides": False,
        }
        evidence_assessments = evidence.assessments if evidence and evidence.assessments else {}
        assessments = {
            family: {"evidence_id": f"assessment.{family}", **dict(evidence_assessments.get(family, empty))}
            for family in matcher.FAMILY_ORDER
        }
        if evidence is not None:
            evidence.assessments = assessments
        return DiagnosisResult(
            fault_type="unknown",
            fault_type_zh="未知",
            leading_family="uncertain",
            confidence=0.0,
            status="unknown",
            reason=reason,
            fault_frequencies_hz=dict(evidence.fault_frequencies_hz) if evidence else {},
            assessments=assessments,
            similar_cases=[],
            explanation=f"无法分析：{reason}",
            resonance_band_hz=list(evidence.selected_bands_hz) if evidence else [],
            envelope_resolution_hz=(
                float(evidence.tool_trace.get("envelope_resolution_hz", 0.0))
                if evidence
                else 0.0
            ),
            diagnosis_code=diagnosis_code,
            diagnosis_source="hard_gate_unknown",
            evidence=evidence.model_dump(mode="json") if evidence is not None else {},
            visual_evidence=list(evidence.visual_evidence) if evidence else [],
            decision_trace=trace or {},
        )

    def _attach_gate_state(
        self,
        result: DiagnosisResult,
        evidence: EvidenceBundle,
        *,
        run_id: str | None = None,
    ) -> DiagnosisResult:
        """让缺少必要输入的硬门结果也具备可继续处理的任务状态。"""

        flags = [
            str(flag)
            for source in (
                evidence.quality.get("invalidating_flags", []),
                evidence.diagnosability.get("invalidating_flags", []),
            )
            for flag in (source if isinstance(source, list) else [])
        ]
        required: list[str] = []
        flag_text = " ".join(flags).lower()
        if "rpm" in flag_text or "speed" in flag_text:
            required.append("rpm")
        if "bearing" in flag_text or "geometry" in flag_text or "reference" in flag_text:
            required.append("bearing geometry")
        if "sampling" in flag_text or "sample" in flag_text:
            required.append("sampling_rate_hz")
        state = AgentState(
            run_id=run_id or result.run_id or AgentState().run_id,
            phase="awaiting_input" if required else "limited",
            termination_reason="missing_required_input" if required else "quality_limited",
            pending_input=(
                {
                    "reason": "missing_required_input",
                    "required_fields": required,
                    "request_revision": 1,
                }
                if required
                else None
            ),
        )
        result.decision_trace["agent_state"] = state.model_dump(mode="json")
        return result

    def _resonance_min_band(self, frequencies_hz: Mapping[str, float]) -> float:
        """谐振频带下限：必须高于最高故障谐波，避免频带直接覆盖故障频率本身。

        若频带下限过低（如 50 Hz），会与故障特征频率范围重叠，包络谱里
        直接出现故障基频，而不是通过谐振解调得到，导致正常轴承被误报。
        """

        max_harmonics = self.config["matching"].get("max_harmonics", 5)
        max_fault = max((v for v in frequencies_hz.values() if v > 0), default=0.0)
        return max(50.0, max_fault * max_harmonics)

    def _select_bands(
        self, x: np.ndarray, fs: float, frequencies_hz: Mapping[str, float]
    ) -> list[tuple[float, float]]:
        """谱峭度选带（不足 3 带时固定倍频带兜底），稳态主链的统一选带入口。"""

        min_band = self._resonance_min_band(frequencies_hz)
        return sig.select_resonance_bands(
            x,
            fs,
            min_band_hz=min_band,
            n_bands=self.config["signal"].get("band_count", 5),
        )

    def _visualization_settings(self) -> dict[str, Any]:
        configured = self.config.get("visualization", {})
        if not isinstance(configured, Mapping):
            configured = {}
        output_root = self.artifact_dir or configured.get(
            "artifact_dir", "outputs/evidence"
        )
        multimodal = self._multimodal_settings()
        return {
            "enabled": bool(
                (
                    self.artifact_dir
                    or configured.get("enabled", False)
                    or multimodal["requested"]
                )
                and configured.get("save_multiband_envelope", True)
            ),
            "artifact_dir": str(output_root),
            "dpi": int(configured.get("dpi", 160)),
            "max_plot_points": int(configured.get("max_plot_points", 2000)),
        }

    def _multimodal_settings(self) -> dict[str, Any]:
        configured = self.config.get("multimodal", {})
        if not isinstance(configured, Mapping):
            configured = {}
        requested = (
            bool(self.multimodal)
            if self.multimodal is not None
            else bool(configured.get("enabled", False))
        )
        return {
            "requested": requested,
            "max_image_bytes": int(
                configured.get("max_image_bytes", 8 * 1024 * 1024)
            ),
        }

    def _save_visual_evidence(
        self,
        analysis: Mapping[str, Any],
        frequencies_hz: Mapping[str, float],
        context: Mapping[str, Any],
        settings: Mapping[str, Any],
        signal: np.ndarray,
        run_id: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """保存审计图；失败只进入 trace，不得阻断或改变诊断。"""

        if not settings.get("enabled"):
            return [], {"visualization_status": "disabled"}
        try:
            import hashlib

            from .visualization import save_multiband_envelope_evidence

            canonical_signal = np.asarray(signal, dtype="<f8").reshape(-1)
            artifacts, manifest_path = save_multiband_envelope_evidence(
                analysis.get("spectrum_artifacts", []),
                frequencies_hz,
                str(settings["artifact_dir"]),
                run_id=run_id,
                max_harmonics=int(self.config["matching"].get("max_harmonics", 5)),
                max_plot_points=int(settings["max_plot_points"]),
                dpi=int(settings["dpi"]),
                independent_view=self._multimodal_settings()["requested"],
                metadata={
                    "operating_context": dict(context),
                    "signal_fingerprint": {
                        "algorithm": "sha256",
                        "canonical_dtype": "float64_le",
                        "sha256": hashlib.sha256(
                            canonical_signal.tobytes(order="C")
                        ).hexdigest(),
                    },
                },
            )
            return artifacts, {
                "visualization_status": "saved" if artifacts else "not_available",
                "visual_manifest_path": manifest_path,
            }
        except Exception as exc:  # noqa: BLE001 - 可视化不能破坏诊断主链
            return [], {
                "visualization_status": "failed",
                "visualization_error": f"{type(exc).__name__}: {exc}",
            }

    # ------------------------------------------------------------ 证据采集
    def collect_evidence(
        self,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
        *,
        speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
        run_id: str | None = None,
    ) -> EvidenceBundle:
        """先过可诊断性硬门，再收集故障证据；本函数不输出最终标签。"""

        x, input_normalization = diag_gate.normalize_real_series(signal)
        try:
            fs = float(sampling_rate_hz)
        except (TypeError, ValueError):
            fs = float("nan")
        try:
            rpm_value = float(rpm)
        except (TypeError, ValueError):
            rpm_value = float("nan")

        quality = diag_gate.assess_input_quality(
            x,
            fs,
            rpm_value,
            self.config["signal"]["quality"],
            speed_trace_rpm=speed_trace_rpm,
            input_normalization=input_normalization,
        )
        bearing_type = str(geometry.get("type", "generic")) if isinstance(geometry, Mapping) else "generic"
        context = {
            "sampling_rate_hz": fs if np.isfinite(fs) else None,
            "rpm": rpm_value if np.isfinite(rpm_value) else None,
            "sample_count": int(x.size),
            "duration_s": float(x.size / fs) if np.isfinite(fs) and fs > 0 else 0.0,
            "bearing_type": bearing_type,
            "speed_trace_provided": speed_trace_rpm is not None,
        }
        gate = diag_gate.gate_summary(input_quality=quality)
        self._emit_progress(
            "signal_quality",
            passed=bool(quality.get("valid_for_analysis")),
            sample_count=int(x.size),
            duration_seconds=context["duration_s"],
            rms=quality.get("rms"),
            clipping_ratio=quality.get("clipping_ratio"),
            trend_ratio=quality.get("trend_ratio"),
            invalidating_flags=list(quality.get("invalidating_flags", [])),
        )
        if not quality["valid_for_analysis"]:
            return EvidenceBundle(
                analyzable=False,
                limitation_reason=(
                    "可诊断性门未通过：" + ", ".join(gate["invalidating_flags"])
                ),
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                tool_trace={"gate": "diagnosability_input_quality"},
            )

        try:
            ratios = geom.fault_frequency_ratios(geometry)
            frequencies_hz = geom.fault_frequencies_hz(rpm_value, ratios)
        except (ValueError, TypeError) as exc:
            frequency_gate = {
                "evidence_id": "diagnosability.frequency",
                "valid": False,
                "invalidating_flags": ["missing_or_invalid_bearing_reference"],
            }
            gate = diag_gate.gate_summary(
                input_quality=quality, frequency_observability=frequency_gate
            )
            return EvidenceBundle(
                analyzable=False,
                limitation_reason=f"缺少可信转速或轴承几何，无法计算故障特征频率：{exc}",
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                tool_trace={"gate": "bearing_reference"},
            )

        diag_cfg = self.config["signal"]["diagnosability"]
        max_harmonics = int(self.config["matching"].get("max_harmonics", 5))
        frequency_gate = diag_gate.assess_frequency_observability(
            frequencies_hz,
            fs,
            int(x.size),
            diag_cfg,
            max_harmonics=max_harmonics,
        )
        gate = diag_gate.gate_summary(
            input_quality=quality, frequency_observability=frequency_gate
        )
        self._emit_progress(
            "frequency_observability",
            passed=bool(frequency_gate.get("valid")),
            fault_frequencies_hz={
                str(key): float(value) for key, value in frequencies_hz.items()
            },
            frequency_resolution_hz=frequency_gate.get("frequency_resolution_hz"),
            analysis_range_hz=frequency_gate.get("analysis_range_hz"),
            invalidating_flags=list(frequency_gate.get("invalidating_flags", [])),
        )
        if not frequency_gate["valid"]:
            return EvidenceBundle(
                analyzable=False,
                limitation_reason=(
                    "故障特征频率不可可靠观测："
                    + ", ".join(gate["invalidating_flags"])
                ),
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                fault_frequencies_hz={str(k): float(v) for k, v in frequencies_hz.items()},
                tool_trace={"gate": "frequency_observability"},
            )

        d = self.config["decision"]
        bands = self._select_bands(x, fs, frequencies_hz)
        self._emit_progress(
            "resonance_bands",
            selected_bands_hz=[[float(low), float(high)] for low, high in bands],
        )
        if not bands:
            band_gate = {
                "evidence_id": "diagnosability.bands",
                "valid": False,
                "valid_band_count": 0,
                "required_valid_band_count": int(diag_cfg.get("min_valid_bands", 2)),
                "invalidating_flags": ["resonance_band_selection_failed"],
            }
            gate = diag_gate.gate_summary(
                input_quality=quality,
                frequency_observability=frequency_gate,
                band_diagnosability=band_gate,
            )
            return EvidenceBundle(
                analyzable=False,
                limitation_reason="可诊断性门未通过：共振频带选择失败",
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                fault_frequencies_hz=frequencies_hz,
                tool_trace={"gate": "band_selection"},
            )
        visualization_settings = self._visualization_settings()
        analysis = chain.analyze_envelope_chain(
            x,
            fs,
            frequencies_hz,
            bands,
            matching=self.config["matching"],
            decision_cfg=d,
            diagnosability_cfg=diag_cfg,
            envelope_mode=self.config["signal"]["envelope"].get("mode", "squared"),
            envelope_window=self.config["signal"]["envelope"].get("window", "hann"),
            retain_spectra=bool(visualization_settings["enabled"]),
            shaft_hz=float(rpm_value) / 60.0,
        )
        band_gate = analysis["band_gate"]
        gate = diag_gate.gate_summary(
            input_quality=quality,
            frequency_observability=frequency_gate,
            band_diagnosability=band_gate,
        )
        self._emit_progress(
            "envelope_analysis",
            diagnosability_passed=bool(gate.get("passed")),
            valid_band_count=band_gate.get("valid_band_count"),
            required_valid_band_count=band_gate.get("required_valid_band_count"),
            invalidating_flags=list(gate.get("invalidating_flags", [])),
            assessments={
                family: {
                    "evidence_level": assessment.get("evidence_level", "none"),
                    "fundamental_bands": assessment.get("fundamental_bands", 0),
                    "qualified_orders": list(assessment.get("qualified_orders", [])),
                }
                for family, assessment in analysis["assessments"].items()
            },
        )
        visual_evidence, visual_trace = self._save_visual_evidence(
            analysis,
            frequencies_hz,
            context,
            visualization_settings,
            x,
            run_id,
        )
        self._emit_progress(
            "visual_evidence",
            status=visual_trace.get("visualization_status"),
            artifact_paths=[item.get("path") for item in visual_evidence],
        )
        common_payload = {
            "quality": quality,
            "diagnosability": gate,
            "operating_context": context,
            "fault_frequencies_hz": {str(k): float(v) for k, v in frequencies_hz.items()},
            "selected_bands_hz": [[float(b[0]), float(b[1])] for b in bands],
            "band_evidence": analysis["band_evidence"],
            "assessments": analysis["assessments"],
            "visual_evidence": visual_evidence,
            "tool_trace": {
                "band_selection": "spectral_kurtosis",
                "envelope_mode": self.config["signal"]["envelope"].get("mode", "squared"),
                "envelope_resolution_hz": float(analysis["resolution"]),
                "colliding_families": sorted(analysis["colliding"]),
                "diagnosability_gate": "passed" if gate["passed"] else "failed",
                **visual_trace,
            },
        }
        if not gate["passed"]:
            return EvidenceBundle(
                analyzable=False,
                limitation_reason=(
                    "可诊断性门未通过：" + ", ".join(gate["invalidating_flags"])
                ),
                **common_payload,
            )

        frequency_features = features_mod.frequency_domain_features(x, fs)
        signal_features = {
            "feature_schema_version": features_mod.FEATURE_SCHEMA_VERSION,
            "time_domain": features_mod.time_domain_features(x),
            "kurtosis_bands": frequency_features.get("kurtosis_bands", [])[:6],
            "psd": frequency_features.get("psd", {}),
        }
        psd = signal_features.get("psd", {})
        strongest_psd_bands = sorted(
            [
                band
                for band in psd.get("band_power_ratios", [])
                if isinstance(band, Mapping) and band.get("observable")
            ],
            key=lambda band: float(band.get("power_ratio", 0.0)),
            reverse=True,
        )[:3]
        self._emit_progress(
            "signal_features",
            time_domain=signal_features["time_domain"],
            psd_frequency_resolution_hz=psd.get("frequency_resolution_hz"),
            strongest_psd_bands=[dict(band) for band in strongest_psd_bands],
        )
        return EvidenceBundle(
            signal_features=signal_features,
            **common_payload,
        )

    def collect_envelope_evidence(
        self,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
        bands: Sequence[Sequence[float]],
        *,
        speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
        run_id: str | None = None,
        retain_spectra: bool = False,
        band_method: str | None = None,
    ) -> tuple[EvidenceBundle, dict[str, Any]]:
        """显式频带的固定包络证据。

        与 ``collect_evidence`` 的区别：不自动选带、不做 PSD、不做可视化。
        质量门和频率可观测性仍然执行；包络谱与谐波匹配复用固定实现。
        """

        x, input_normalization = diag_gate.normalize_real_series(signal)
        try:
            fs = float(sampling_rate_hz)
        except (TypeError, ValueError):
            fs = float("nan")
        try:
            rpm_value = float(rpm)
        except (TypeError, ValueError):
            rpm_value = float("nan")
        quality = diag_gate.assess_input_quality(
            x,
            fs,
            rpm_value,
            self.config["signal"]["quality"],
            speed_trace_rpm=speed_trace_rpm,
            input_normalization=input_normalization,
        )
        context = {
            "sampling_rate_hz": fs if np.isfinite(fs) else None,
            "rpm": rpm_value if np.isfinite(rpm_value) else None,
            "sample_count": int(x.size),
            "duration_s": float(x.size / fs) if np.isfinite(fs) and fs > 0 else 0.0,
            "bearing_type": str(geometry.get("type", "generic")) if isinstance(geometry, Mapping) else "generic",
            "speed_trace_provided": speed_trace_rpm is not None,
        }
        gate = diag_gate.gate_summary(input_quality=quality)
        if not quality["valid_for_analysis"]:
            return EvidenceBundle(
                analyzable=False,
                limitation_reason="可诊断性门未通过：" + ", ".join(gate["invalidating_flags"]),
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                tool_trace={"band_selection": "explicit", "gate": "input_quality"},
            ), {}
        try:
            ratios = geom.fault_frequency_ratios(geometry)
            frequencies_hz = geom.fault_frequencies_hz(rpm_value, ratios)
        except (ValueError, TypeError) as exc:
            return EvidenceBundle(
                analyzable=False,
                limitation_reason=f"缺少可信转速或轴承几何：{exc}",
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                tool_trace={"band_selection": "explicit", "gate": "bearing_reference"},
            ), {}
        diag_cfg = self.config["signal"]["diagnosability"]
        max_harmonics = int(self.config["matching"].get("max_harmonics", 5))
        frequency_gate = diag_gate.assess_frequency_observability(
            frequencies_hz, fs, int(x.size), diag_cfg, max_harmonics=max_harmonics
        )
        gate = diag_gate.gate_summary(
            input_quality=quality, frequency_observability=frequency_gate
        )
        if not frequency_gate["valid"]:
            return EvidenceBundle(
                analyzable=False,
                limitation_reason="故障特征频率不可可靠观测：" + ", ".join(gate["invalidating_flags"]),
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                fault_frequencies_hz={str(k): float(v) for k, v in frequencies_hz.items()},
                tool_trace={"band_selection": "explicit", "gate": "frequency_observability"},
            ), {}
        normalized_bands = [
            [float(band[0]), float(band[1])] for band in bands if len(band) >= 2
        ]
        if not normalized_bands:
            return EvidenceBundle(
                analyzable=False,
                limitation_reason="未提供显式包络频带",
                quality=quality,
                diagnosability=gate,
                operating_context=context,
                fault_frequencies_hz={str(k): float(v) for k, v in frequencies_hz.items()},
                tool_trace={"band_selection": "explicit", "gate": "no_bands"},
            ), {}
        analysis = chain.analyze_envelope_chain(
            x,
            fs,
            frequencies_hz,
            normalized_bands,
            matching=self.config["matching"],
            decision_cfg=self.config["decision"],
            diagnosability_cfg=diag_cfg,
            envelope_mode=self.config["signal"]["envelope"].get("mode", "squared"),
            envelope_window=self.config["signal"]["envelope"].get("window", "hann"),
            retain_spectra=retain_spectra,
            shaft_hz=float(rpm_value) / 60.0,
            band_method=band_method,
        )
        gate = diag_gate.gate_summary(
            input_quality=quality,
            frequency_observability=frequency_gate,
            band_diagnosability=analysis["band_gate"],
        )
        evidence = EvidenceBundle(
            analyzable=bool(gate["passed"]),
            limitation_reason=None
            if gate["passed"]
            else "可诊断性门未通过：" + ", ".join(gate["invalidating_flags"]),
            quality=quality,
            diagnosability=gate,
            operating_context=context,
            fault_frequencies_hz={str(k): float(v) for k, v in frequencies_hz.items()},
            selected_bands_hz=normalized_bands,
            band_evidence=analysis["band_evidence"],
            assessments={
                family: {
                    "evidence_id": assessment.get("evidence_id", f"assessment.{family}"),
                    "evidence_level": assessment.get("evidence_level", "none"),
                    "evidence_bands": assessment.get("evidence_bands", 0),
                    "fit_supported_bands": assessment.get("fit_supported_bands", 0),
                    "fundamental_bands": assessment.get("fundamental_bands", 0),
                    "fundamental_collides": assessment.get("fundamental_collides", False),
                    "qualified_orders": list(assessment.get("qualified_orders", [])),
                }
                for family, assessment in analysis["assessments"].items()
            },
            tool_trace={
                "band_selection": band_method or "explicit",
                "band_source": "processing_plan",
                "envelope_mode": self.config["signal"]["envelope"].get("mode", "squared"),
                "envelope_resolution_hz": float(analysis["resolution"]),
                "colliding_families": sorted(analysis["colliding"]),
                "diagnosability_gate": "passed" if gate["passed"] else "failed",
            },
        )
        return evidence, analysis

    # -------------------------------------------------------------- 确定性基线
    def _baseline_from_evidence(
        self,
        evidence: EvidenceBundle,
        *,
        include_reference_cases: bool,
    ) -> DiagnosisResult:
        """从唯一 EvidenceBundle 计算盲基线，不重新运行信号处理。"""

        if not evidence.analyzable:
            result = self._unknown_result(
                evidence.limitation_reason or "工具证据不足",
                {
                    "quality": evidence.quality,
                    "diagnosability": evidence.diagnosability,
                    "tool_trace": evidence.tool_trace,
                },
                evidence=evidence,
            )
            self._emit_progress(
                "deterministic_baseline",
                fault_type=result.fault_type,
                leading_family=result.leading_family,
                status=result.status,
                reason=result.reason,
            )
            return result

        d = self.config["decision"]
        harmonic_counts = {
            family: len(assessment.get("qualified_orders", []))
            for family, assessment in evidence.assessments.items()
        }
        residuals = {
            family: float(assessment.get("max_residual_bins") or 0.0)
            for family, assessment in evidence.assessments.items()
            if assessment.get("max_residual_bins") is not None
        }
        verdict = decision.decide(
            {family: assessment.get("evidence_level", "none") for family, assessment in evidence.assessments.items()},
            fundamental_bands={family: int(assessment.get("evidence_bands", assessment.get("fundamental_bands", 0))) for family, assessment in evidence.assessments.items()},
            harmonic_counts=harmonic_counts,
            residuals=residuals,
            confidence=d["confidence"],
            dominance_margin=d.get("dominance_margin", 2),
        )

        similar_cases = (
            self._reference_cases(verdict["leading_family"])
            if include_reference_cases
            else []
        )
        explanation = self._explain(
            verdict,
            evidence.assessments,
            evidence.fault_frequencies_hz,
            evidence.band_evidence,
        )

        result = DiagnosisResult(
            fault_type=verdict["fault_type"],
            fault_type_zh=verdict["fault_type_zh"],
            leading_family=verdict["leading_family"],
            confidence=verdict["confidence"],
            status=verdict["status"],
            reason=verdict["reason"],
            fault_frequencies_hz=evidence.fault_frequencies_hz,
            assessments=evidence.assessments,
            similar_cases=similar_cases,
            explanation=explanation,
            resonance_band_hz=evidence.selected_bands_hz,
            envelope_resolution_hz=float(evidence.tool_trace.get("envelope_resolution_hz", 0.0)),
            llm_used=False,
            diagnosis_code=verdict["fault_type"],
            diagnosis_source="deterministic",
            baseline_verdict=dict(verdict),
            evidence=evidence.model_dump(mode="json"),
            visual_evidence=list(evidence.visual_evidence),
            decision_trace={
                "quality": evidence.quality,
                "diagnosability": evidence.diagnosability,
                "tool_trace": evidence.tool_trace,
                "deterministic_baseline": verdict,
            },
        )
        self._emit_progress(
            "deterministic_baseline",
            fault_type=result.fault_type,
            leading_family=result.leading_family,
            status=result.status,
            confidence=result.confidence,
            reason=result.reason,
        )
        return result

    def diagnose_deterministic(
        self,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
        *,
        speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
    ) -> DiagnosisResult:
        store, exchange_start = self._start_audit_run(
            "deterministic", signal, sampling_rate_hz, rpm, geometry
        )
        evidence = self.collect_evidence(
            signal,
            sampling_rate_hz,
            rpm,
            geometry,
            speed_trace_rpm=speed_trace_rpm,
            run_id=store.run_id if store else None,
        )
        result = self._baseline_from_evidence(
            evidence,
            include_reference_cases=True,
        )
        return self._finish_audit_run(store, exchange_start, evidence, result)

    def diagnose(
        self,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
        *,
        speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
    ) -> DiagnosisResult:
        """兼容入口：返回确定性基线；智能体最终诊断使用 diagnose_agentic。"""

        return self.diagnose_deterministic(
            signal,
            sampling_rate_hz,
            rpm,
            geometry,
            speed_trace_rpm=speed_trace_rpm,
        )

    # ---------------------------------------------- LLM 最终诊断与回退
    def _llm_confidence_mapping(self) -> dict[str, float]:
        configured = self.config.get("llm", {}).get("confidence_mapping", {})
        values = dict(DEFAULT_CONFIDENCE_MAPPING)
        if isinstance(configured, Mapping):
            values.update({str(key): float(value) for key, value in configured.items()})
        return values

    def _fallback_to_baseline(self, result: DiagnosisResult, reason: str) -> DiagnosisResult:
        result.diagnosis_source = "deterministic_fallback"
        result.fallback_reason = reason
        result.decision_trace["fallback_reason"] = reason
        result.llm_used = False
        return result

    def _apply_llm_diagnosis(
        self,
        result: DiagnosisResult,
        evidence: EvidenceBundle,
        diagnosis: LLMDiagnosis,
        *,
        source: str,
    ) -> DiagnosisResult:
        if self.config.get("llm", {}).get("require_evidence_refs", True):
            reference_errors = validate_evidence_references(
                diagnosis,
                evidence,
                {str(item.get("evidence_id")) for item in result.visual_evidence},
            )
            if reference_errors:
                raise ValueError("；".join(reference_errors))

        baseline_fault = result.baseline_verdict.get("fault_type", result.fault_type)
        numeric_confidence = diagnosis.confidence(self._llm_confidence_mapping())
        diagnosis_payload = diagnosis.model_dump(mode="json")
        diagnosis_payload["confidence"] = numeric_confidence

        result.fault_type = diagnosis.fault_type
        result.fault_type_zh = FAULT_TYPE_ZH[diagnosis.fault_type]
        result.leading_family = diagnosis.leading_family
        result.confidence = numeric_confidence
        result.status = diagnosis.status
        result.diagnosis_code = diagnosis.fault_type
        explanation = self._engineering_explanation(
            diagnosis, evidence, result.decision_trace
        )
        result.reason = explanation
        result.explanation = explanation
        result.llm_used = True
        result.diagnosis_source = source
        result.llm_diagnosis = diagnosis_payload
        result.fallback_reason = None
        result.evidence = evidence.model_dump(mode="json")
        result.similar_cases = [dict(case) for case in evidence.similar_cases]
        result.visual_evidence = [dict(item) for item in evidence.visual_evidence]
        result.decision_trace["llm_diagnosis"] = diagnosis_payload
        result.decision_trace["llm_overrode_baseline"] = diagnosis.fault_type != baseline_fault
        return result

    def _run_llm_with_repair(
        self,
        result: DiagnosisResult,
        evidence: EvidenceBundle,
        *,
        source: str,
    ) -> DiagnosisResult:
        """执行最终诊断；首次结构/引用失败时最多修复一次。"""

        try:
            self._emit_progress("llm_evidence_request", attempt=1)
            diagnosis = self.llm.diagnose_evidence(evidence)
            return self._apply_llm_diagnosis(
                result, evidence, diagnosis, source=source
            )
        except Exception as first_error:  # noqa: BLE001
            repair = getattr(self.llm, "repair_diagnosis", None)
            max_repairs = int(
                self.config.get("llm", {}).get("max_semantic_repair_attempts", 1)
            )
            if callable(repair) and max_repairs > 0:
                result.decision_trace["llm_repair_attempted"] = True
                result.decision_trace["llm_first_error"] = (
                    f"{type(first_error).__name__}: {first_error}"
                )
                try:
                    self._emit_progress(
                        "llm_repair_request", error=f"{type(first_error).__name__}: {first_error}"
                    )
                    diagnosis = repair(evidence, str(first_error))
                    result = self._apply_llm_diagnosis(
                        result, evidence, diagnosis, source=source
                    )
                    result.decision_trace["llm_repair_succeeded"] = True
                    return result
                except Exception as repair_error:  # noqa: BLE001
                    result.decision_trace["llm_repair_succeeded"] = False
                    return self._fallback_to_baseline(
                        result,
                        "llm_diagnosis_failed_after_repair:"
                        f"{type(repair_error).__name__}: {repair_error}",
                    )
            return self._fallback_to_baseline(
                result,
                f"llm_diagnosis_failed:{type(first_error).__name__}: {first_error}",
            )

    def _tool_runtime_enabled(self) -> bool:
        configured = self.config.get("agent_runtime", {})
        return bool(isinstance(configured, Mapping) and configured.get("enabled", True)
                    and getattr(self.llm, "supports_tool_calling", False)
                    and callable(getattr(self.llm, "complete_messages", None)))

    def _run_constrained_agent(
        self,
        result: DiagnosisResult,
        evidence: EvidenceBundle,
        *,
        source: str,
        psd_provider: Callable[[], Mapping[str, Any]] | None = None,
        case_provider: Callable[[], Mapping[str, Any]] | None = None,
        run_id: str | None = None,
    ) -> DiagnosisResult:
        configured = self.config.get("agent_runtime", {})
        enabled = bool(
            isinstance(configured, Mapping)
            and configured.get("enabled", True)
            and getattr(self.llm, "supports_tool_calling", False)
            and callable(getattr(self.llm, "complete_messages", None))
        )
        if not enabled:
            self._emit_progress(
                "agent_runtime_unavailable",
                reason="provider_or_configuration_without_tool_calling",
            )
            return self._run_llm_with_repair(
                result,
                evidence,
                source=("llm_rag" if source.endswith("rag") else "llm_evidence"),
            )
        multimodal = self._multimodal_settings()
        observer = None
        multimodal_reason = "not_requested"
        if multimodal["requested"]:
            if not evidence.visual_evidence:
                multimodal_reason = "visual_artifact_unavailable"
            elif not getattr(self.llm, "supports_image_input", False):
                multimodal_reason = "provider_image_input_not_enabled"
            else:
                observer = MultimodalEvidenceObserver(
                    self.llm,
                    max_image_bytes=multimodal["max_image_bytes"],
                )
                multimodal_reason = "available_as_optional_tool"
        result.decision_trace["multimodal"] = {
            "requested": multimodal["requested"],
            "enabled": observer is not None,
            "reason": multimodal_reason,
        }
        runtime = agent_runtime.ConstrainedDiagnosisRuntime(
            self.llm,
            evidence,
            max_tool_calls=int(configured.get("max_tool_calls", 4)),
            max_turns=int(configured.get("max_turns", 3)),
            visual_observer=observer.observe if observer is not None else None,
            progress_callback=self.progress_callback,
            compare_normal_psd=psd_provider,
            retrieve_balanced_cases=case_provider,
            run_id=run_id,
            max_no_progress_actions=int(configured.get("max_no_progress_actions", 2)),
            tool_timeout_seconds=float(configured.get("tool_timeout_seconds", 30.0)),
        )
        try:
            runtime_result = runtime.run()
            result.decision_trace["agent_runtime"] = {
                "status": "completed",
                "events": runtime_result.events,
                "state": runtime_result.state,
            }
            return self._apply_llm_diagnosis(
                result,
                evidence,
                runtime_result.diagnosis,
                source=source,
            )
        except Exception as exc:  # noqa: BLE001
            result.decision_trace["agent_runtime"] = {
                "status": "fallback_to_evidence_mode",
                "error": f"{type(exc).__name__}: {exc}",
                "events": runtime.events,
                "state": runtime.state.model_dump(mode="json"),
            }
            if isinstance(exc, agent_runtime.AgentRuntimeStopped):
                result.decision_trace["agent_runtime"]["status"] = "limited"
                return self._fallback_to_baseline(result, f"agent_stopped:{runtime.state.termination_reason}")
            invalid_output = runtime.last_output
            repair = getattr(self.llm, "repair_diagnosis", None)
            max_repairs = int(
                self.config.get("llm", {}).get(
                    "max_semantic_repair_attempts", 1
                )
            )
            if invalid_output and callable(repair) and max_repairs > 0:
                result.decision_trace["llm_repair_attempted"] = True
                result.decision_trace["llm_first_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                try:
                    diagnosis = repair(
                        evidence,
                        str(exc),
                        invalid_output=invalid_output,
                    )
                    runtime.validate_final(diagnosis)
                    result = self._apply_llm_diagnosis(
                        result,
                        evidence,
                        diagnosis,
                        source=source,
                    )
                    result.decision_trace["llm_repair_succeeded"] = True
                    result.decision_trace["agent_runtime"]["status"] = (
                        "completed_after_repair"
                    )
                    result.decision_trace["agent_runtime"]["state"].update(
                        phase="limited" if diagnosis.status == "unknown" else "completed",
                        termination_reason="completed_after_repair",
                    )
                    return result
                except Exception as repair_error:  # noqa: BLE001
                    result.decision_trace["llm_repair_succeeded"] = False
                    result.decision_trace["agent_runtime"]["status"] = (
                        "failed_after_repair"
                    )
                    return self._fallback_to_baseline(
                        result,
                        "llm_tool_diagnosis_failed_after_repair:"
                        f"{type(repair_error).__name__}: {repair_error}",
                    )
            return self._fallback_to_baseline(
                result, f"llm_tool_diagnosis_failed:{type(exc).__name__}: {exc}"
            )

    def _compare_psd_on_demand(self, evidence, geometry, rpm) -> dict[str, Any]:
        self._attach_psd_normality(evidence, geometry, rpm, suppress_errors=False)
        value = evidence.signal_features.get("psd_normality")
        return dict(value) if isinstance(value, Mapping) else {
            "evidence_id": "signal.psd_normality", "status": "unavailable",
            "reason": "当前存储未提供可比健康PSD参考；不能据此判断正常或故障",
        }

    def _skill_driven_agent(
        self,
        *,
        mode: str,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
        speed_trace_rpm: Sequence[float] | np.ndarray | None,
        case_enabled: bool = False,
    ) -> DiagnosisResult:
        """延迟运行主链：模型先读取 Skill，再请求核心证据工具。"""

        store, exchange_start = self._start_audit_run(
            mode, signal, sampling_rate_hz, rpm, geometry
        )
        holder: dict[str, EvidenceBundle] = {}
        multimodal = self._multimodal_settings()
        pool = ThreadPoolExecutor(max_workers=1) if multimodal["requested"] else None
        visual_future = None

        def finish(evidence, result):
            if pool is None:
                return self._finish_audit_run(store, exchange_start, evidence, result)
            try:
                visual = visual_future.result() if visual_future is not None else {
                    "status": "unavailable", "reason": "no_analyzable_image",
                }
            except Exception as exc:
                visual = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
            finally:
                pool.shutdown(wait=True)
            numeric = {
                "status": result.decision_trace.get("agent_runtime", {}).get("status"),
                "diagnosis": {key: deepcopy(result.llm_diagnosis[key]) for key in LLMDiagnosis.model_fields}
                if result.llm_used and result.llm_diagnosis else None,
                "error": result.decision_trace.get("agent_runtime", {}).get("error"),
            }
            trace = {"numeric": numeric, "visual": visual}
            result.decision_trace["dual_channel"] = trace
            result.decision_trace["multimodal"] = {
                "requested": True, "enabled": visual.get("status") == "completed",
                "reason": visual.get("reason", "independent_visual_diagnosis"),
            }
            self._emit_progress("visual_diagnosis_result", **visual)
            if evidence.analyzable:
                self._emit_progress("fusion_request")
                try:
                    merged = summarize_channels(self.llm, evidence, numeric, visual)
                    result = self._apply_llm_diagnosis(result, evidence, merged.diagnosis, source="llm_dual_channel")
                    trace["fusion"] = {"status": "completed", **merged.model_dump(mode="json")}
                    self._emit_progress("fusion_result", **trace["fusion"])
                except Exception as exc:
                    trace["fusion"] = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
                    result.fallback_reason = "fusion_failed_retained_numeric_result"
                    self._emit_progress("fusion_result", **trace["fusion"])
            else:
                trace["fusion"] = {"status": "skipped", "reason": "quality_gate_failed"}
            return self._finish_audit_run(store, exchange_start, evidence, result)

        def provide_evidence() -> EvidenceBundle:
            nonlocal visual_future
            evidence = self.collect_evidence(
                signal,
                sampling_rate_hz,
                rpm,
                geometry,
                speed_trace_rpm=speed_trace_rpm,
                run_id=store.run_id if store else None,
            )
            holder["evidence"] = evidence
            if pool is not None and evidence.analyzable and evidence.visual_evidence:
                self._emit_progress("visual_diagnosis_request")
                visual_future = pool.submit(
                    diagnose_image, self.llm, deepcopy(evidence.visual_evidence[0]),
                    dict(evidence.fault_frequencies_hz), multimodal["max_image_bytes"],
                )
            return evidence

        configured = self.config.get("agent_runtime", {})
        def current_evidence() -> EvidenceBundle:
            evidence = holder.get("evidence")
            if evidence is None:
                raise RuntimeError("core_evidence_not_collected")
            return evidence

        runtime = agent_runtime.ConstrainedDiagnosisRuntime(
            self.llm,
            None,
            evidence_provider=provide_evidence,
            skill_loader=DiagnosisSkillLoader(),
            require_skill_first=True,
            numeric_only=multimodal["requested"],
            max_tool_calls=int(configured.get("max_tool_calls", 4)),
            max_turns=int(configured.get("max_turns", 6)),
            visual_observer=None,
            compare_normal_psd=lambda: self._compare_psd_on_demand(
                current_evidence(), geometry, rpm
            ),
            retrieve_balanced_cases=(
                (lambda: self._retrieve_cases(current_evidence(), geometry, rpm))
                if case_enabled
                else None
            ),
            progress_callback=self.progress_callback,
            run_id=store.run_id if store else None,
            max_no_progress_actions=int(configured.get("max_no_progress_actions", 2)),
            tool_timeout_seconds=float(configured.get("tool_timeout_seconds", 30.0)),
        )
        try:
            runtime_result = runtime.run()
            evidence = runtime.evidence
            if evidence is None:
                raise RuntimeError("runtime_did_not_return_core_evidence")
            result = self._baseline_from_evidence(evidence, include_reference_cases=False)
            if result.diagnosis_source == "hard_gate_unknown":
                self._attach_gate_state(result, evidence, run_id=store.run_id if store else None)
            else:
                result = self._apply_llm_diagnosis(
                    result, evidence, runtime_result.diagnosis,
                    source="llm_tool_rag" if case_enabled else "llm_tool_agent",
                )
            result.decision_trace["agent_runtime"] = {
                "status": "completed",
                "events": runtime_result.events,
                "state": runtime_result.state,
            }
            return finish(evidence, result)
        except Exception as exc:
            evidence = runtime.evidence or holder.get("evidence")
            if evidence is None:
                # 模型在 Skill 阶段失败时仍生成可审计的受限基线；此调用发生在
                # 智能体已进入之后，不再出现入口前自动分析。
                evidence = provide_evidence()
            result = self._baseline_from_evidence(evidence, include_reference_cases=False)
            if result.diagnosis_source == "hard_gate_unknown":
                self._attach_gate_state(result, evidence, run_id=store.run_id if store else None)
            result.decision_trace["agent_runtime"] = {
                "status": "limited" if isinstance(exc, agent_runtime.AgentRuntimeStopped) else "fallback",
                "error": f"{type(exc).__name__}: {exc}",
                "events": runtime.events,
                "state": runtime.state.model_dump(mode="json"),
            }
            invalid_output = runtime.last_output
            repair = getattr(self.llm, "repair_diagnosis", None)
            if invalid_output and callable(repair):
                try:
                    diagnosis = repair(evidence, str(exc), invalid_output=invalid_output)
                    runtime.validate_final(diagnosis)
                    result = self._apply_llm_diagnosis(
                        result, evidence, diagnosis,
                        source="llm_tool_rag" if case_enabled else "llm_tool_agent",
                    )
                    result.decision_trace["llm_repair_succeeded"] = True
                    result.decision_trace["agent_runtime"]["status"] = "completed_after_repair"
                except Exception:
                    result.decision_trace["llm_repair_succeeded"] = False
            return finish(evidence, result)

    def _attach_psd_normality(
        self,
        evidence: EvidenceBundle,
        geometry: Mapping[str, Any] | None,
        rpm: float,
        *,
        suppress_errors: bool = True,
    ) -> None:
        """从案例库附加健康 PSD 偏离证据；失败不改变主诊断链。"""

        comparator = getattr(
            self.sample_store, "compare_normal_psd_by_condition", None
        )
        psd = evidence.signal_features.get("psd")
        if not callable(comparator) or not isinstance(psd, Mapping):
            return
        bearing_type = (
            str(geometry.get("type", "generic"))
            if isinstance(geometry, Mapping)
            else "generic"
        )
        try:
            normality = comparator(bearing_type, rpm, psd)
        except Exception as exc:  # noqa: BLE001  # 辅助案例证据不可阻断主链
            if not suppress_errors:
                raise
            evidence.tool_trace["psd_normal_reference"] = (
                f"unavailable:{type(exc).__name__}"
            )
            return
        if isinstance(normality, Mapping):
            evidence.signal_features["psd_normality"] = dict(normality)
            evidence.tool_trace["psd_normal_reference"] = str(
                normality.get("status", "unavailable")
            )

    def diagnose_agentic(
        self,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
        *,
        speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
    ) -> DiagnosisResult:
        """工具收集证据，LLM 给出最终诊断；失败时返回确定性基线。"""

        runtime_config = self.config.get("agent_runtime", {})
        if (
            self._tool_runtime_enabled()
            and isinstance(runtime_config, Mapping)
            and runtime_config.get("skill_driven", True)
        ):
            return self._skill_driven_agent(
                mode="agentic",
                signal=signal,
                sampling_rate_hz=sampling_rate_hz,
                rpm=rpm,
                geometry=geometry,
                speed_trace_rpm=speed_trace_rpm,
            )

        store, exchange_start = self._start_audit_run(
            "agentic", signal, sampling_rate_hz, rpm, geometry
        )
        evidence = self.collect_evidence(
            signal,
            sampling_rate_hz,
            rpm,
            geometry,
            speed_trace_rpm=speed_trace_rpm,
            run_id=store.run_id if store else None,
        )
        result = self._baseline_from_evidence(
            evidence,
            include_reference_cases=False,
        )
        if result.diagnosis_source == "hard_gate_unknown":
            self._attach_gate_state(result, evidence, run_id=store.run_id if store else None)
            return self._finish_audit_run(store, exchange_start, evidence, result)

        if not self._tool_runtime_enabled():
            self._attach_psd_normality(evidence, geometry, rpm)
        result.evidence = evidence.model_dump(mode="json")
        result.similar_cases = list(evidence.similar_cases)

        if self.llm is None or not getattr(self.llm, "available", False):
            result = self._fallback_to_baseline(result, "llm_unavailable")
            return self._finish_audit_run(store, exchange_start, evidence, result)

        result = self._run_constrained_agent(
            result,
            evidence,
            source="llm_tool_agent",
            psd_provider=lambda: self._compare_psd_on_demand(evidence, geometry, rpm),
            run_id=store.run_id if store else None,
        )
        return self._finish_audit_run(store, exchange_start, evidence, result)

    # ------------------------------------------------------------ RAG 诊断
    def diagnose_rag(
        self,
        signal: Sequence[float] | np.ndarray,
        sampling_rate_hz: float,
        rpm: float,
        geometry: Mapping[str, Any] | None,
        *,
        speed_trace_rpm: Sequence[float] | np.ndarray | None = None,
    ) -> DiagnosisResult:
        """工具取证 + 去锚定案例检索 + LLM 最终诊断，失败时回退基线。"""

        runtime_config = self.config.get("agent_runtime", {})
        if (
            self._tool_runtime_enabled()
            and isinstance(runtime_config, Mapping)
            and runtime_config.get("skill_driven", True)
        ):
            return self._skill_driven_agent(
                mode="rag",
                signal=signal,
                sampling_rate_hz=sampling_rate_hz,
                rpm=rpm,
                geometry=geometry,
                speed_trace_rpm=speed_trace_rpm,
                case_enabled=True,
            )

        store, exchange_start = self._start_audit_run(
            "rag", signal, sampling_rate_hz, rpm, geometry
        )
        evidence = self.collect_evidence(
            signal,
            sampling_rate_hz,
            rpm,
            geometry,
            speed_trace_rpm=speed_trace_rpm,
            run_id=store.run_id if store else None,
        )
        result = self._baseline_from_evidence(
            evidence,
            include_reference_cases=False,
        )
        if result.diagnosis_source == "hard_gate_unknown":
            self._attach_gate_state(result, evidence, run_id=store.run_id if store else None)
            return self._finish_audit_run(store, exchange_start, evidence, result)

        if not self._tool_runtime_enabled():
            self._retrieve_cases(evidence, geometry, rpm)
            self._attach_psd_normality(evidence, geometry, rpm)
            result.evidence = evidence.model_dump(mode="json")
            result.similar_cases = list(evidence.similar_cases)

        if self.llm is None or not getattr(self.llm, "available", False):
            result = self._fallback_to_baseline(result, "llm_unavailable")
            return self._finish_audit_run(store, exchange_start, evidence, result)
        result = self._run_constrained_agent(
            result,
            evidence,
            source="llm_tool_rag",
            psd_provider=lambda: self._compare_psd_on_demand(evidence, geometry, rpm),
            case_provider=lambda: self._retrieve_cases(evidence, geometry, rpm),
            run_id=store.run_id if store else None,
        )
        return self._finish_audit_run(store, exchange_start, evidence, result)

    def _retrieve_cases(self, evidence: EvidenceBundle, geometry, rpm) -> dict[str, Any]:
        """按调用时的当前证据检索；工具模式不预先运行本函数。"""
        similar_cases: list[dict[str, Any]] = []
        retrieval_scope = ""
        try:
            features = features_mod.retrieval_features_from_evidence(
                evidence.model_dump(mode="python")
            )
        except Exception:  # noqa: BLE001  # 特征提取失败不影响诊断
            features = None

        if self.sample_store is not None and features is not None:
            try:
                bearing_type = str(geometry.get("type", "generic")) if isinstance(geometry, Mapping) else "generic"
                matched, filtered_count = self.sample_store.retrieve_balanced_by_condition(
                    bearing_type, rpm, features, top_k=4
                )
                similar_cases = [
                    {
                        "evidence_id": f"case.{index}",
                        "case_id": item.case_id,
                        "fault_family": item.fault_family,
                        "fault_type": item.fault_type,
                        "score": item.score,
                        "psd_similarity": getattr(item, "score_components", {}).get(
                            "psd_profile_similarity"
                        ),
                        "similarity_basis": list(
                            getattr(item, "similarity_basis", [])
                        )[:4],
                    }
                    for index, item in enumerate(matched, start=1)
                ]
                retrieval_scope = (
                    f"已在同型号({bearing_type})、转速±50% 的 {filtered_count} 条案例中按故障族平衡检索"
                )
            except Exception:  # noqa: BLE001  # 检索失败不影响诊断
                similar_cases = []

        if not similar_cases:
            similar_cases = self._balanced_reference_cases()

        self._emit_progress(
            "case_retrieval",
            source=retrieval_scope or "local_balanced_reference",
            case_count=len(similar_cases),
            cases=[
                {
                    "case_id": case.get("case_id"),
                    "fault_family": case.get("fault_family"),
                    "score": case.get("score"),
                }
                for case in similar_cases
            ],
        )

        evidence.similar_cases = [dict(c) for c in similar_cases]
        if features is not None:
            evidence.signal_features["feature_description"] = features.get("description", "")
            evidence.signal_features["feature_vector"] = features.get("feature_vector", [])
        evidence.tool_trace["case_retrieval"] = retrieval_scope or "local_balanced_reference"

        return {"similar_cases": list(evidence.similar_cases), "scope": retrieval_scope or "local_balanced_reference"}

    # ------------------------------------------------------------- 案例参考
    def _reference_cases(self, family: str) -> list[dict[str, Any]]:
        return [
            kb.reference_view(c)
            for c in kb.retrieve_similar(
                self._cases_loaded(),
                family=family,
                top_k=self.config["case_kb"].get("top_k", 3),
            )
        ]

    def _balanced_reference_cases(self) -> list[dict[str, Any]]:
        """每个故障族最多取一条，避免用基线标签预筛案例导致 LLM 被锚定。"""

        selected: list[dict[str, Any]] = []
        for family in ("BPFO", "BPFI", "BSF", "NONE"):
            cases = kb.retrieve_similar(self._cases_loaded(), family=family, top_k=1)
            if not cases:
                continue
            view = kb.reference_view(cases[0])
            view["evidence_id"] = f"case.{len(selected) + 1}"
            selected.append(view)
        return selected

    # ---------------------------------------------------------------- 解释
    def _explain(
        self,
        verdict: dict[str, Any],
        assessments: Mapping[str, Mapping[str, Any]],
        frequencies_hz: Mapping[str, float],
        band_evidence: Sequence[Mapping[str, Any]],
    ) -> str:
        status = str(verdict.get("status"))
        family = str(verdict.get("leading_family", "uncertain"))
        conclusion = (
            "当前未发现明确轴承故障"
            if status == "normal"
            else "暂无法可靠判断故障类型"
            if status == "unknown"
            else str(verdict["fault_type_zh"])
        )
        lines = [f"结论：{conclusion}（置信度：{verdict['confidence']:.2f}）。"]
        if status == "diagnosed":
            lines.append(self._key_evidence_summary(assessments, frequencies_hz, family))
            locations = self._band_location_summary(band_evidence, family)
            if locations:
                lines.append(locations)
        elif status == "normal":
            lines.append("关键依据：BPFO、BPFI、BSF 均未形成可归类的跨频带一致证据。")
        else:
            lines.append(f"限制：{verdict['reason']}")
        return "\n".join(lines)

    @staticmethod
    def _order_summary(values: Sequence[Any]) -> str:
        orders = sorted({int(value) for value in values if int(value) > 0})
        if not orders:
            return ""
        if len(orders) > 1 and orders == list(range(orders[0], orders[-1] + 1)):
            return f"{orders[0]}X–{orders[-1]}X"
        return "、".join(f"{order}X" for order in orders)

    @classmethod
    def _key_evidence_summary(
        cls,
        assessments: Mapping[str, Mapping[str, Any]],
        frequencies_hz: Mapping[str, float],
        family: str,
    ) -> str:
        assessment = assessments.get(family, {})
        level = EVIDENCE_LEVEL_ZH.get(
            str(assessment.get("evidence_level", "none")), "不足"
        )
        band_count = int(
            assessment.get(
                "evidence_bands", assessment.get("fundamental_bands", 0)
            )
        )
        orders = cls._order_summary(assessment.get("qualified_orders", []))
        order_text = f"，整体命中 {orders}" if orders else ""
        competitors = [
            candidate
            for candidate in matcher.FAMILY_ORDER
            if candidate != family
            and assessments.get(candidate, {}).get("evidence_level")
            in {"moderate", "strong"}
        ]
        competition = (
            f"；另有 {'、'.join(competitors)} 形成支持，但弱于主导频族"
            if competitors
            else "；其余频族未形成同等级竞争证据"
        )
        return (
            f"关键依据：{family} 理论基频 {frequencies_hz.get(family, 0.0):.2f} Hz，"
            f"证据等级为{level}，在 {band_count} 个频带形成一致谐波{order_text}"
            f"{competition}。"
        )

    @staticmethod
    def _band_location_summary(
        band_evidence: Sequence[Mapping[str, Any]], family: str
    ) -> str:
        if family not in matcher.FAMILY_ORDER:
            return ""
        locations: list[str] = []
        for position, band in enumerate(band_evidence, start=1):
            bounds = band.get("band_hz")
            families = band.get("families", {})
            family_data = families.get(family, {}) if isinstance(families, Mapping) else {}
            if not (
                isinstance(bounds, (list, tuple))
                and len(bounds) == 2
                and isinstance(family_data, Mapping)
            ):
                continue
            peaks = []
            for peak in family_data.get("harmonic_peaks", []):
                if not isinstance(peak, Mapping) or not peak.get("qualified"):
                    continue
                order = int(peak.get("harmonic", 0))
                observed = peak.get("observed_frequency_hz")
                if order > 0 and observed is not None:
                    peaks.append((order, float(observed)))
            if peaks:
                peak_text = "、".join(
                    f"{order}X={frequency:.2f} Hz"
                    for order, frequency in sorted(peaks)
                )
                locations.append(
                    f"频带 {position}（{float(bounds[0]):.0f}–"
                    f"{float(bounds[1]):.0f} Hz 解调载频带）：{peak_text}。"
                )
        if not locations:
            return ""
        return "频带定位：\n- " + "\n- ".join(locations)

    @classmethod
    def _engineering_explanation(
        cls,
        diagnosis: LLMDiagnosis,
        evidence: EvidenceBundle,
        decision_trace: Mapping[str, Any],
    ) -> str:
        conclusion = (
            "当前未发现明确轴承故障"
            if diagnosis.status == "normal"
            else "暂无法可靠判断故障类型"
            if diagnosis.status == "unknown"
            else FAULT_TYPE_ZH[diagnosis.fault_type]
        )
        confidence = CONFIDENCE_LEVEL_ZH[diagnosis.confidence_level]
        lines = [f"结论：{conclusion}（置信度：{confidence}）。"]

        if diagnosis.status == "diagnosed":
            lines.append(
                cls._key_evidence_summary(
                    evidence.assessments,
                    evidence.fault_frequencies_hz,
                    diagnosis.leading_family,
                )
            )
            locations = cls._band_location_summary(
                evidence.band_evidence, diagnosis.leading_family
            )
            if locations:
                lines.append(locations)
        elif diagnosis.status == "normal":
            lines.append("关键依据：BPFO、BPFI、BSF 均未形成可归类的跨频带一致证据。")

        limitations = list(diagnosis.conflicts)
        dual_channel = decision_trace.get("dual_channel", {})
        visual = dual_channel.get("visual", {}) if isinstance(dual_channel, Mapping) else {}
        if isinstance(visual, Mapping) and visual.get("status") not in {None, "completed"}:
            limitations.append("视觉通道未完成，本结论仅依据数值证据")
        if diagnosis.status == "unknown" and not limitations:
            limitations.append(evidence.limitation_reason or diagnosis.reasoning)
        if limitations:
            lines.append("限制：" + "；".join(dict.fromkeys(limitations)) + "。")
        return "\n".join(lines)
