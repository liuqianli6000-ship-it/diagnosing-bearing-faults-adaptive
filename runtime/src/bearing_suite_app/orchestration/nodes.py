"""LangGraph nodes for the three-layer adaptive diagnosis."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bearing_suite_app.assessment.band_banks import BAND_METHODS
from bearing_suite_app.assessment.signal_assessment import assess_signal
from bearing_suite_app.diagnosability import assess_input_quality, normalize_real_series
from bearing_suite_app import geometry as geom
from bearing_suite_app.evidence import gate as evidence_gate
from bearing_suite_app.execution import observe_call
from bearing_suite_app.final.narrative import build_narrative, SPECIFIC_FAULTS
from bearing_suite_app.final.numeric import diagnose_numeric, route_band_evidence
from bearing_suite_app.processing.plan import build_plan, normalize_plan
from bearing_suite_app.processing.preprocess import build_plan as build_preprocess_plan
from bearing_suite_app.processing.preprocess import run_preprocess
from bearing_suite_app.processing.rounds import run_rounds
from bearing_suite_app.speed_estimation import estimate_shaft_speed

from .state import GraphState, Runtime


def _baseline_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if hasattr(result, "model_dump"):
        data = result.model_dump(mode="json")
    else:
        data = {
            key: getattr(result, key)
            for key in (
                "fault_type",
                "confidence",
                "status",
                "diagnosis_source",
                "leading_family",
                "reason",
            )
            if hasattr(result, key)
        }
    return {
        key: data.get(key)
        for key in (
            "fault_type",
            "confidence",
            "status",
            "diagnosis_source",
            "leading_family",
            "reason",
        )
    }


def _evidence_summary(evidence: Any) -> dict[str, Any]:
    if evidence is None:
        return {}
    assessments = evidence.assessments or {}
    levels = {}
    bands = {}
    for family in ("BPFO", "BPFI", "BSF"):
        item = assessments.get(family) or {}
        levels[family] = item.get("evidence_level", "none") if isinstance(item, dict) else "none"
        bands[family] = item.get("evidence_bands", 0) if isinstance(item, dict) else 0
    quality = evidence.quality or {}
    return {
        "analyzable": bool(evidence.analyzable),
        "limitation_reason": evidence.limitation_reason,
        "quality_ok": bool(quality.get("valid_for_analysis", quality.get("ok", False))),
        "quality_flags": list(quality.get("warning_flags", []))
        + list(quality.get("invalidating_flags", [])),
        "diagnosability_passed": bool((evidence.diagnosability or {}).get("passed", False)),
        "fault_frequencies_hz": dict(evidence.fault_frequencies_hz or {}),
        "assessment_levels": levels,
        "assessment_bands": bands,
        "valid_band_count": (evidence.diagnosability or {})
        .get("stages", {})
        .get("band_diagnosability", {})
        .get("valid_band_count", 0),
    }


def _round(state: GraphState, runtime: Runtime) -> int:
    gate = state.get("gate") or {}
    skill = runtime.skill or {}
    skill_text = str(skill.get("skill_text") or "")
    return {
        "case_id": state.get("case_id", runtime.case_id),
        "round": int(state.get("round", 0)) + 1,
        "max_rounds": runtime.max_rounds,
        "baseline": state.get("baseline", {}),
        "gate": gate,
        "evidence": state.get("evidence_summary", {}),
        "signal_profile": state.get("signal_profile", {}),
        "analysis_plan": state.get("analysis_plan", []),
        "step_observations": (state.get("step_observations") or [])[-4:],
        "precision_results": state.get("precision_results", [])[-3:],
        "last_evaluator": (state.get("evaluator_decisions") or [{}])[-1],
        "allowed_methods": [
            "adaptive_notch",
            "med",
            "mckd",
            "momeda",
            "iterative_sk",
            "subband_analysis",
            "stop",
        ],
        "max_trials": runtime.max_trials,
        "skill": {
            "id": skill.get("skill_id"),
            "workflow_steps": skill.get("workflow_steps"),
            "contract": skill.get("workflow_contract"),
            "text": skill_text[:6000],
        },
    }


def intake_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    signal, normalization = normalize_real_series(runtime.raw_signal)
    errors = []
    if not normalization["valid"]:
        errors.append("invalid_signal_shape" if normalization["reason"] == "not_single_channel" else "invalid_signal_dtype")
    if signal.size < 1024:
        errors.append("signal_too_short")
    if signal.size and float(np.ptp(signal)) == 0.0:
        errors.append("constant_signal")
    if not np.isfinite(signal).all():
        errors.append("signal_non_finite")
    if not np.isfinite(runtime.sampling_rate_hz) or runtime.sampling_rate_hz <= 0:
        errors.append("invalid_sampling_rate")
    if not np.isfinite(runtime.rpm) or runtime.rpm <= 0:
        errors.append("invalid_rpm")
    try:
        ratios = geom.fault_frequency_ratios(runtime.geometry)
    except (TypeError, ValueError, AttributeError, OverflowError):
        errors.append("invalid_geometry")
    if errors:
        return {
            "case_id": runtime.case_id,
            "status": "rejected",
            "round": 0,
            "max_rounds": runtime.max_rounds,
            "errors": errors,
            "final": {
                "fault_type": "unknown",
                "confidence": 0.0,
                "reasoning": "输入检查失败：" + ", ".join(errors),
                "source": "input_validation",
            },
        }
    quality = assess_input_quality(signal, runtime.sampling_rate_hz, runtime.rpm,
                                   runtime.agent.config["signal"]["quality"], input_normalization=normalization)
    speed_reference = None
    if quality["valid_for_analysis"]:
        speed_reference = observe_call(
            runtime.on_event, "estimate_shaft_speed",
            {"nominal_rpm": runtime.rpm, "sampling_rate_hz": runtime.sampling_rate_hz, "samples": signal.size},
            lambda: estimate_shaft_speed(signal, runtime.sampling_rate_hz, runtime.rpm),
        )
        candidates = speed_reference["candidates"]
        speed_reference["estimated_rpm"] = candidates[0]["rpm"] if candidates else None
        runtime.rpm = speed_reference["rpm"]
    runtime.fault_frequencies_hz = geom.fault_frequencies_hz(runtime.rpm, ratios)
    runtime.signal = signal
    runtime.base_signal = signal
    return {
        "case_id": runtime.case_id,
        "status": "running",
        "speed_reference": speed_reference,
        "round": 0,
        "max_rounds": runtime.max_rounds,
        "errors": [],
        "precision_results": [],
        "planner_decisions": [],
        "evaluator_decisions": [],
    }


def l1_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    current = np.asarray(runtime.signal, dtype=float).reshape(-1)
    results = []
    # Outlier clipping is recorded as a candidate, but is not applied by
    # default: hard clipping can remove genuine fault impacts and change the
    # deterministic baseline.  Normalization is scale-only and safe.
    for tool in ("detrend", "outlier_remove", "normalize"):
        try:
            result = runtime.registry.execute(
                tool, current, runtime.tool_context(), auto=True, max_trials=2
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"tool": tool, "status": "error", "error": str(exc)})
            continue
        results.append(result.to_dict())
        if tool in {"detrend", "normalize"} and result.status == "ok" and result.arrays.get("signal") is not None:
            current = np.asarray(result.arrays["signal"], dtype=float).reshape(-1)
    runtime.signal = current
    runtime.base_signal = current
    return {"preprocess_results": results, "status": "running"}


def l2_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    results = []
    for tool in ("fft_spectrum", "envelope_spectrum", "time_stats", "fault_frequency"):
        try:
            result = runtime.registry.execute(
                tool, runtime.signal, runtime.tool_context(), auto=True, max_trials=2
            )
        except Exception as exc:  # noqa: BLE001
            results.append({"tool": tool, "status": "error", "error": str(exc)})
            continue
        results.append(result.to_dict())
    return {"l2_results": results, "status": "running"}


def profile_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    """Lightweight signal profile; deliberately no EvidenceBundle/fault matching."""

    results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for tool in ("time_stats", "fft_spectrum", "spectral_kurtosis", "fault_frequency"):
        try:
            result = runtime.registry.execute(
                tool, runtime.base_signal, runtime.tool_context(), auto=False
            )
        except Exception as exc:  # noqa: BLE001
            results[tool] = {"status": "error", "error": str(exc)}
            continue
        results[tool] = result.to_dict()
        arrays.update(result.arrays)
    stats = results.get("time_stats", {})
    fft = results.get("fft_spectrum", {})
    top_peaks = (fft.get("data") or {}).get("top_peaks", [])
    kurtosis = float((stats.get("metrics") or {}).get("kurtosis") or 0.0)
    crest = float((stats.get("metrics") or {}).get("crest_factor") or 0.0)
    peak_amplitudes = [float(item.get("amplitude", 0.0)) for item in top_peaks]
    noise_floor = float(np.median(peak_amplitudes)) if peak_amplitudes else 0.0
    peak_level = float(max(peak_amplitudes)) if peak_amplitudes else 0.0
    snr_db = 20.0 * np.log10((peak_level + 1e-12) / (noise_floor + 1e-12)) if peak_level > 0 else 0.0
    narrow_peaks = [
        item for item in top_peaks
        if float(item.get("frequency_hz", 0.0)) >= 20.0
    ]
    profile = {
        "sampling_rate_hz": runtime.sampling_rate_hz,
        "rpm": runtime.rpm,
        "signal_samples": int(runtime.base_signal.size),
        "kurtosis": kurtosis,
        "crest_factor": crest,
        "spectral_peak_snr_estimate_db": round(float(snr_db), 3),
        "dominant_peaks_hz": [item.get("frequency_hz") for item in top_peaks[:5]],
        "narrowband_peak_count": len(narrow_peaks),
        "spectral_kurtosis_bands": (results.get("spectral_kurtosis", {}).get("data") or {}).get("bands", []),
        "fault_frequencies_hz": (results.get("fault_frequency", {}).get("data") or {}).get("fault_frequencies_hz", {}),
        "profile_tools": results,
    }
    runtime.notes["signal_profile"] = profile
    return {"signal_profile": profile, "status": "profiling"}


def _valid_plan(plan: Any, adaptive: bool) -> tuple[bool, str]:
    if not isinstance(plan, list) or not plan:
        return False, "plan_empty"
    if len(plan) > 3:
        return False, "plan_too_long"
    evidence_tools = {
        "envelope_spectrum",
        "harmonic_match",
        "emd_decompose",
        "wavelet_decompose",
    }
    l3_tools = {
        "adaptive_notch",
        "med",
        "mckd",
        "momeda",
        "iterative_sk",
        "subband_analysis",
    }
    seen_evidence = False
    l3_count = 0
    for step in plan:
        tool = str(step.get("tool") or "")
        if tool in evidence_tools:
            seen_evidence = True
        if tool in l3_tools:
            l3_count += 1
        if tool in {"collect_evidence", "diagnose"}:
            return False, "evidence_tool_in_plan"
    if not seen_evidence:
        return False, "missing_l2_evidence_step"
    if not adaptive and l3_count:
        return False, "l3_disabled"
    if l3_count > 2:
        return False, "too_many_l3_steps"
    return True, "ok"


def analysis_plan_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    plan_round = int(state.get("plan_round", 0)) + 1
    if plan_round > runtime.max_plan_rounds:
        return {
            "next_route": "evidence",
            "plan_round": plan_round,
            "status": "planning",
        }
    summary = {
        "case_id": runtime.case_id,
        "round": plan_round,
        "phase": "initial" if not state.get("gate") else "replan",
        "adaptive": runtime.adaptive,
        "max_steps": 3,
        "signal_profile": state.get("signal_profile") or runtime.notes.get("signal_profile") or {},
        "gate": state.get("gate") or {},
        "evidence": state.get("evidence_summary") or {},
        "step_observations": (state.get("step_observations") or [])[-3:],
        "max_trials": runtime.max_trials,
        "skill": {
            "id": (runtime.skill or {}).get("skill_id"),
            "workflow_steps": (runtime.skill or {}).get("workflow_steps"),
            "text": str((runtime.skill or {}).get("skill_text") or "")[:6000],
        },
    }
    decision = runtime.controller.plan_analysis(summary)
    plan = decision.get("plan") or []
    valid, reason = _valid_plan(plan, runtime.adaptive)
    if not valid:
        fallback = runtime.controller._fallback_plan_analysis(
            {**summary, "phase": "fallback"}, reason
        )
        plan = fallback.get("plan") or []
        valid, reason = _valid_plan(plan, runtime.adaptive)
    if not valid:
        return {"next_route": "evidence", "status": "planning", "errors": [f"invalid_plan:{reason}"]}
    history = list(state.get("plan_history") or [])
    history.append({"round": plan_round, "reason": decision.get("reason"), "plan": plan, "source": decision.get("source")})
    return {
        "analysis_plan": plan,
        "plan_index": 0,
        "plan_round": plan_round,
        "plan_history": history,
        "next_route": "execute",
        "status": "planning",
    }


def execute_step_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    plan = list(state.get("analysis_plan") or [])
    index = int(state.get("plan_index", 0))
    if index >= len(plan):
        return {"next_route": "evidence", "status": "executing"}
    step = plan[index]
    tool = str(step.get("tool") or "")
    try:
        result = runtime.registry.execute(
            tool,
            runtime.base_signal,
            runtime.tool_context(),
            auto=True,
            max_trials=runtime.max_trials,
        )
    except Exception as exc:  # noqa: BLE001
        observation = {
            "step": index,
            "tool": tool,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "purpose": step.get("purpose"),
        }
        return {
            "step_observations": list(state.get("step_observations") or []) + [observation],
            "precision_results": list(state.get("precision_results") or []) + [observation],
            "plan_index": index + 1,
            "next_route": "execute",
            "status": "executing",
        }
    observation = result.to_dict()
    observation.update(
        {"step": index, "tool": tool, "purpose": step.get("purpose")}
    )
    runtime.precision_results.append(observation)
    if result.arrays.get("signal") is not None:
        runtime.step_arrays.append(
            {
                "step": index,
                "tool": tool,
                "signal": np.asarray(result.arrays["signal"], dtype=float).reshape(-1),
                "metrics": dict(result.metrics),
            }
        )
    return {
        "step_observations": list(state.get("step_observations") or []) + [observation],
        "precision_results": list(state.get("precision_results") or []) + [observation],
        "plan_index": index + 1,
        "next_route": "execute",
        "status": "executing",
    }


def step_evaluate_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    plan = list(state.get("analysis_plan") or [])
    index = int(state.get("plan_index", 0))
    if index >= len(plan):
        return {"next_route": "evidence", "status": "evaluating"}
    summary = {
        "case_id": runtime.case_id,
        "round": int(state.get("plan_round", 1)),
        "plan_index": index,
        "analysis_plan": plan,
        "signal_profile": state.get("signal_profile") or {},
        "step_observations": (state.get("step_observations") or [])[-3:],
        "max_steps": len(plan),
    }
    decision = runtime.controller.evaluate_step(summary)
    decisions = list(state.get("step_decisions") or [])
    decisions.append({"round": int(state.get("plan_round", 1)), **decision})
    if decision.get("decision") == "switch":
        route = "plan"
    elif decision.get("decision") == "stop":
        route = "evidence"
    else:
        route = "execute" if index < len(plan) else "evidence"
    return {
        "step_decisions": decisions,
        "next_route": route,
        "status": "evaluating",
    }


def evidence_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    evidence = runtime.agent.collect_evidence(
        runtime.base_signal,
        runtime.sampling_rate_hz,
        runtime.rpm,
        runtime.geometry,
    )
    baseline = runtime.agent._baseline_from_evidence(
        evidence, include_reference_cases=False
    )
    adopted = False
    target = getattr(baseline, "leading_family", None)
    rank_before = evidence_gate.evidence_rank(evidence, target)
    if runtime.step_arrays:
        best = max(
            runtime.step_arrays,
            key=lambda item: float(item.get("metrics", {}).get("objective", 0.0)),
        )
        try:
            enhanced = runtime.agent.collect_evidence(
                best["signal"],
                runtime.sampling_rate_hz,
                runtime.rpm,
                runtime.geometry,
            )
            rank_after = evidence_gate.evidence_rank(enhanced, target)
            if (
                rank_after.get("target_family") == rank_before.get("target_family")
                and rank_after["key"] > rank_before["key"]
            ):
                evidence = enhanced
                baseline = runtime.agent._baseline_from_evidence(
                    evidence, include_reference_cases=False
                )
                adopted = True
                runtime.notes["evidence_adopted_tool"] = best.get("tool")
        except Exception as exc:  # noqa: BLE001
            runtime.notes["evidence_adoption_error"] = f"{type(exc).__name__}: {exc}"
    runtime.evidence = evidence
    runtime.baseline = baseline
    return {
        "baseline": _baseline_dict(baseline),
        "evidence_summary": _evidence_summary(evidence),
        "evidence_adopted": adopted,
        "status": "evidence",
    }


def gate_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    result = evidence_gate.evaluate(runtime.evidence, runtime.baseline)
    if result.get("target_frequency_hz"):
        runtime.notes["target_fault_frequency_hz"] = result["target_frequency_hz"]
    if result.get("target_family"):
        runtime.notes["target_family"] = result["target_family"]
    status = {
        "clear": "diagnosed",
        "limited": "limited",
        "uncertain": "precision",
        "conflict": "precision",
    }.get(result["verdict"], "precision")
    return {"gate": result, "status": status}


def plan_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    round_index = int(state.get("round", 0)) + 1
    if round_index > runtime.max_rounds:
        return {
            "next_action": "stop",
            "planner_decisions": list(state.get("planner_decisions", []))
            + [{"round": round_index, "action": "stop", "source": "budget"}],
        }
    summary = _round(state, runtime)
    decision = runtime.controller.plan(summary)
    action = decision.get("action", "stop")
    gate = state.get("gate") or {}
    # A planner cannot stop while the local gate still reports conflict or
    # uncertainty.  This is the guard that forces at least one L3 attempt.
    if (
        action == "stop"
        and gate.get("verdict") not in {"clear", "limited"}
        and round_index <= runtime.max_rounds
    ):
        last_eval = (state.get("evaluator_decisions") or [{}])[-1]
        suggested = last_eval.get("next_action")
        if suggested in {
            "adaptive_notch",
            "med",
            "mckd",
            "momeda",
            "iterative_sk",
            "subband_analysis",
        }:
            action = suggested
            decision = {
                **decision,
                "action": action,
                "source": "local_guard:evaluator_switch",
                "reason": f"本地门控为 {gate.get('verdict')}，采用 evaluator 建议的 {action}；"
                + str(decision.get("reason") or ""),
            }
        else:
            fallback = runtime.controller._fallback_plan(summary, "gate_not_clear")
            action = fallback.get("action", "iterative_sk")
            decision = {
                **decision,
                "action": action,
                "source": "local_guard",
                "reason": f"本地门控为 {gate.get('verdict')}，不允许直接停止；"
                + str(decision.get("reason") or ""),
            }
    decisions = list(state.get("planner_decisions", []))
    decisions.append({"round": round_index, **decision})
    return {
        "round": round_index,
        "next_action": action,
        "planner_decisions": decisions,
        "status": "precision",
    }


def precision_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    action = state.get("next_action") or "stop"
    if action == "stop":
        return {"status": "precision"}
    try:
        result = runtime.registry.execute(
            action,
            runtime.base_signal,
            runtime.tool_context(),
            auto=True,
            max_trials=runtime.max_trials,
        )
    except Exception as exc:  # noqa: BLE001
        record = {
            "tool": action,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "params": {},
            "metrics": {},
        }
        return {
            "precision_results": list(state.get("precision_results", [])) + [record],
            "status": "precision",
        }
    summary = result.to_dict()
    summary["round"] = int(state.get("round", 1))
    enhanced = result.arrays.get("signal")
    if enhanced is not None:
        enhanced = np.asarray(enhanced, dtype=float).reshape(-1)
        runtime.precision_arrays.append(
            {"round": int(state.get("round", 1)), "tool": action, "signal": enhanced}
        )
        target = runtime.notes.get("target_family")
        rank_before = evidence_gate.evidence_rank(runtime.evidence, target)
        try:
            enhanced_evidence = runtime.agent.collect_evidence(
                enhanced,
                runtime.sampling_rate_hz,
                runtime.rpm,
                runtime.geometry,
            )
            enhanced_baseline = runtime.agent._baseline_from_evidence(
                enhanced_evidence, include_reference_cases=False
            )
            rank_after = evidence_gate.evidence_rank(enhanced_evidence, target)
            same_target = rank_after.get("target_family") == rank_before.get("target_family")
            adopted = bool(rank_after["key"] > rank_before["key"] and same_target)
            summary["evidence_before"] = rank_before
            summary["evidence_after"] = rank_after
            summary["evidence_adopted"] = adopted
            if adopted:
                runtime.evidence = enhanced_evidence
                runtime.baseline = enhanced_baseline
                runtime.evidence_history.append(
                    {
                        "round": int(state.get("round", 1)),
                        "tool": action,
                        "rank_before": rank_before,
                        "rank_after": rank_after,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            summary["evidence_after"] = None
            summary["evidence_adopted"] = False
            summary["evidence_error"] = f"{type(exc).__name__}: {exc}"
    runtime.precision_results.append(summary)
    update: dict[str, Any] = {
        "precision_results": list(state.get("precision_results", [])) + [summary],
        "status": "precision",
    }
    if summary.get("evidence_adopted"):
        update["baseline"] = _baseline_dict(runtime.baseline)
        update["evidence_summary"] = _evidence_summary(runtime.evidence)
    return update


def evaluate_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    previous_gate = dict(state.get("gate") or {})
    recomputed = evidence_gate.evaluate(runtime.evidence, runtime.baseline)
    adopted = any(
        item.get("evidence_adopted")
        for item in state.get("precision_results", [])
        if isinstance(item, dict)
    )
    if adopted and recomputed.get("verdict") == "clear":
        updated_gate = recomputed
    else:
        updated_gate = evidence_gate.after_precision(
            previous_gate,
            runtime.baseline,
            list(state.get("precision_results", [])),
        )
    summary = _round({**state, "gate": updated_gate}, runtime)
    summary["round"] = int(state.get("round", 1))
    decision = runtime.controller.evaluate(summary)
    if (
        decision.get("decision") == "stop"
        and updated_gate.get("verdict") not in {"clear", "limited"}
        and int(state.get("round", 1)) < runtime.max_rounds
    ):
        decision = {
            **decision,
            "decision": "continue",
            "source": "local_guard",
            "reason": "本地门控仍未清晰且仍有轮次预算，继续尝试另一组精密分析。",
        }
    evaluations = list(state.get("evaluator_decisions", []))
    evaluations.append({"round": int(state.get("round", 1)), **decision})
    status = "diagnosed" if updated_gate.get("verdict") == "clear" else "precision"
    return {
        "gate": updated_gate,
        "evaluator_decisions": evaluations,
        "status": status,
    }


def synthesize_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    summary = _round(state, runtime)
    summary["gate"] = state.get("gate") or {}
    result = runtime.controller.synthesize(summary)
    baseline = state.get("baseline") or {}
    if result.get("fault_type") not in {
        "outer_race_fault",
        "inner_race_fault",
        "rolling_element_fault",
        "normal",
        "unknown",
    }:
        result = {
            **result,
            "fault_type": baseline.get("fault_type", "unknown"),
            "confidence": baseline.get("confidence", 0.0),
            "source": "local_fallback",
        }
    gate = state.get("gate") or {}
    result["gate"] = gate
    if gate.get("verdict") not in {"clear", None}:
        # The LLM may still choose a family, but the local gate has not cleared
        # the evidence; do not present high confidence or let the model claim
        # that a conflict was resolved.
        try:
            result["confidence"] = min(float(result.get("confidence", 0.0)), 0.5)
        except (TypeError, ValueError):
            result["confidence"] = 0.0
        result["reasoning"] = (
            f"【本地证据门：{gate.get('verdict')}，未达到 clear；"
            "以下模型解释仅供复核】"
            + str(result.get("reasoning") or "")
        )
        note = f"本地证据门为 {gate.get('verdict')}，置信度已封顶 0.5。"
        result["uncertainty"] = (str(result.get("uncertainty") or "") + " " + note).strip()
        levels = gate.get("levels") or {}
        target = gate.get("target_family")
        baseline_fault = baseline.get("fault_type")
        if (
            gate.get("verdict") == "conflict"
            and target
            and levels.get(target) == "strong"
            and baseline_fault in {
                "outer_race_fault",
                "inner_race_fault",
                "rolling_element_fault",
            }
        ):
            # A strong family with a moderate runner-up is not a unique
            # diagnosis, but it is still the dominant measured family.  Keep
            # it at low confidence instead of discarding all evidence.
            result["fault_type"] = baseline_fault
            result["confidence"] = 0.5
            result["severity"] = "无法判断"
            result["uncertainty"] = (
                str(result.get("uncertainty") or "")
                + " 跨族冲突未解除，按主导强证据给出低置信结论，需人工复核。"
            ).strip()
    result["rounds"] = state.get("plan_round", state.get("round", 0))
    result["llm_calls"] = len(runtime.controller.calls)
    result["baseline"] = baseline
    return {"final": result, "status": "completed"}


def limited_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    gate = state.get("gate") or {}
    baseline = state.get("baseline") or {}
    final = {
        "fault_type": "unknown",
        "confidence": 0.0,
        "severity": "无法判断",
        "reasoning": f"证据不足，未进入精密诊断：{gate.get('reason', '')}",
        "key_evidence": [f"gate:{gate.get('verdict')}"],
        "uncertainty": gate.get("reason"),
        "source": "limited_gate",
        "gate": gate,
        "baseline": baseline,
        "rounds": 0,
        "llm_calls": 0,
    }
    return {"final": final, "status": "limited"}


def guided_assessment_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    assessment = assess_signal(
        runtime.raw_signal,
        runtime.sampling_rate_hz,
        runtime.rpm,
        runtime.geometry,
        runtime.agent.config["signal"]["quality"],
        runtime.agent.config["signal"]["diagnosability"],
    )
    return {"assessment": assessment, "status": "assessing"}


def guided_preprocess_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    assessment = state.get("assessment") or {}
    plan = build_preprocess_plan(
        assessment, runtime.sampling_rate_hz, runtime.rpm, runtime.geometry
    )
    spectral = assessment.get("spectral") or {}
    selected = (spectral.get("envelope_band_selection") or {}).get("selected", [])
    bands = [
        [float(band[0]), float(band[1])]
        for band in ([item["band_hz"] for item in selected] or spectral.get("spectral_kurtosis_bands_hz", []))
        if len(band) >= 2
    ]
    if not bands:
        bands = [
            [float(band[0]), float(band[1])]
            for band in (assessment.get("time_frequency") or {}).get("spectral_kurtosis_bands_hz", [])
            if len(band) >= 2
        ]
    context = runtime.tool_context()
    context.notes["processing_objective"] = "signal_state"
    result = run_preprocess(
        runtime.base_signal,
        runtime.sampling_rate_hz,
        plan,
        runtime.registry,
        context,
        bands,
        max_trials=runtime.max_trials,
    )
    signal = result.pop("signal")
    import hashlib

    result["signal_sha256"] = hashlib.sha256(np.asarray(signal, dtype=float).tobytes()).hexdigest()
    return {
        "preprocessing_plan": plan,
        "preprocess_result": result,
        "preprocessed_signal": np.asarray(signal, dtype=float).reshape(-1),
        "status": "preprocessing",
    }


def _valid_processing_plan(plan: Any, sampling_rate_hz: float | None = None) -> bool:
    if not isinstance(plan, dict):
        return False
    rounds = plan.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 1:
        return False
    for round_plan in rounds:
        if not isinstance(round_plan, dict):
            return False
        filters = round_plan.get("filters") or []
        processing = round_plan.get("processing") or []
        if not isinstance(filters, list) or not isinstance(processing, list):
            return False
        if len(filters) > 2 or len(processing) > 1:
            return False
        for steps, allowed in ((filters, {"bandpass", "adaptive_notch"}), (processing, {"med"})):
            for step in steps:
                if not isinstance(step, dict) or step.get("tool") not in allowed:
                    return False
                if not isinstance(step.get("params", {}), dict):
                    return False
                if step["tool"] == "bandpass":
                    params = step.get("params", {})
                    if not _valid_band([params.get("low_hz", params.get("low", 5.0)), params.get("high_hz", params.get("high", 3000.0))], sampling_rate_hz):
                        return False
        if round_plan.get("round", 1) != 1 or round_plan.get("parent", 0) != 0:
            return False
        bands = round_plan.get("bands") or []
        if not isinstance(bands, list) or any(not _valid_band(band, sampling_rate_hz) for band in bands):
            return False
    return True


def _valid_band(band: Any, sampling_rate_hz: float | None) -> bool:
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        return False
    try:
        low, high = map(float, band)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(np.isfinite([low, high]).all() and 0 < low < high and (sampling_rate_hz is None or high <= sampling_rate_hz / 2.0))


def guided_plan_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    assessment = state.get("assessment") or {}
    if not runtime.adaptive or runtime.max_rounds <= 0:
        return {
            "processing_plan": {
                "reason": "adaptive disabled" if not runtime.adaptive else "processing budget exhausted",
                "rounds": [],
                "max_rounds": runtime.max_rounds,
                "source": "baseline",
            },
            "status": "planning",
        }
    plan = runtime.controller.plan_processing(assessment, adaptive=runtime.adaptive, reserve_calls=3)
    try:
        if not _valid_processing_plan(plan, runtime.sampling_rate_hz):
            raise ValueError("invalid_processing_plan")
        plan = normalize_plan(plan)
        for round_plan in plan["rounds"]:
            for step in round_plan["filters"] + round_plan["processing"]:
                params = dict(step["params"])
                auto = params.pop("auto", False)
                max_trials = params.pop("max_trials", runtime.max_trials)
                if not isinstance(auto, bool) or type(max_trials) is not int or max_trials < 1:
                    raise ValueError("auto must be boolean and max_trials must be a positive integer")
                if any(value is None for value in params.values()):
                    raise ValueError(f"{step['tool']} parameters cannot be null")
                runtime.registry.validate(runtime.registry.get(step["tool"]).spec, params)
    except (TypeError, ValueError, OverflowError) as exc:
        plan = build_plan(assessment, adaptive=runtime.adaptive)
        plan["source"] = "local_fallback"
        plan["fallback_reason"] = str(exc)
        plan = normalize_plan(plan)
    return {"processing_plan": plan, "status": "planning"}


def guided_processing_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    plan = state.get("processing_plan") or {}
    context = runtime.tool_context()
    context.notes["processing_objective"] = "signal_state"
    base = state.get("preprocessed_signal")
    if base is None:
        base = runtime.base_signal
    result = run_rounds(
        base,
        runtime.sampling_rate_hz,
        plan,
        runtime.registry,
        context,
        state.get("assessment") or {},
        max_rounds=runtime.max_rounds,
        max_trials=runtime.max_trials,
    )
    return {
        "round_history": result["rounds"],
        "best_round": result["best"],
        "best_bands": result["best_bands"],
        "best_signal": result["best_signal"],
        "baseline": result["baseline"],
        "status": "processing",
    }


def guided_numeric_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    spectral = (state.get("assessment") or {}).get("spectral") or {}
    selected = (spectral.get("envelope_band_selection") or {}).get("selected", [])
    signal = state.get("best_signal")
    if signal is None:
        signal = state.get("preprocessed_signal")
    if signal is None:
        signal = runtime.base_signal

    results = {}
    spectra = {method: [] for method in BAND_METHODS}
    for method in BAND_METHODS:
        bands = [item["band_hz"] for item in selected if item["method"] == method]
        if not bands:
            results[method] = {"status": "limited", "fault_type": "unknown", "bands": [],
                               "reason": "no eligible bands"}
            continue
        try:
            results[method] = diagnose_numeric(
                runtime.agent, signal, runtime.sampling_rate_hz, runtime.rpm, runtime.geometry, bands,
                band_method=method,
                on_envelope_spectra=spectra[method].extend if runtime.on_envelope_spectra else None,
            )
        except Exception as exc:  # A failed selector must leave the other evidence available.
            results[method] = {"status": "limited", "fault_type": "unknown", "bands": bands,
                               "error": f"{type(exc).__name__}: {exc}"}

    route, methods = route_band_evidence(results)
    if runtime.on_envelope_spectra and methods:
        runtime.on_envelope_spectra([item for method in methods for item in spectra[method]])
    numeric = {
        **results[methods[0] if methods else BAND_METHODS[0]],
        "band_methods": results, "evidence_route": route, "used_methods": methods,
    }
    return {
        "numeric_result": numeric,
        "baseline": _baseline_dict(numeric.get("baseline")),
        "evidence_summary": _evidence_summary(numeric.get("evidence")),
        "best_bands": numeric["bands"],
        "status": "numeric" if methods else "limited",
    }


def guided_numeric_final_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    numeric = state.get("numeric_result") or {}
    assessment = state.get("assessment") or {}
    primary = numeric.get("evidence")
    segment = numeric.get("segment_consensus") or {}
    used = numeric.get("used_methods") or []
    method_results = numeric.get("band_methods") or {}

    def segment_payload(check, *, review=False):
        keys = ("index", "analyzable", "assessments", "fault_type") if review else ("index", "analyzable", "assessments")
        return {
            "analyzable_count": check["analyzable_count"],
            "conflict": check["conflict"],
            "windows": [{key: item[key] for key in keys if key in item} for item in check["segments"]],
        }

    def model_evidence(signal, *, review=False):
        evidence = {"signal": signal, "envelope": primary.to_llm_payload(
            include_weak_peaks=True, include_shared_peaks=review,
        )}
        if segment.get("checked"):
            evidence["segments"] = segment_payload(segment, review=review)
        if used:
            evidence["band_selection"] = {
                "route": numeric["evidence_route"], "primary_method": BAND_METHODS[0], "used_methods": used,
            }
        if len(used) > 1:
            evidence["candidate_envelopes"] = {}
            for method in used[1:]:
                item = method_results[method]
                payload = item["evidence"].to_llm_payload(include_weak_peaks=True, include_shared_peaks=review)
                if (check := item["segment_consensus"]).get("checked"):
                    payload["segments"] = segment_payload(check, review=review)
                evidence["candidate_envelopes"][method] = payload
        return evidence

    result = {"error": "insufficient_numeric_evidence"}
    if primary is not None and (used or primary.analyzable):
        signal = {
            "conditions": {key: (assessment.get("meta") or {}).get(key)
                           for key in ("sampling_rate_hz", "duration_seconds", "rpm")},
            "time": {key: (assessment.get("time_domain") or {}).get(key)
                     for key in ("rms", "peak", "kurtosis", "crest_factor", "modulation_index", "stationarity")},
            "dominant_peaks": (assessment.get("frequency_domain") or {}).get("dominant_peaks", [])[:5],
            "transient_frames": (assessment.get("time_frequency") or {}).get("transient_count"),
            "shaft_frequency_hz": (state.get("speed_reference") or {}).get("shaft_hz"),
        }
        rules = (runtime.skill or {}).get("diagnosis_criteria")
        result = runtime.controller.judge_diagnosis(model_evidence(signal), rules) if rules else {"error": "missing_skill_rules"}
    fault_type = result.get("fault_type", "unknown")
    reason = result.get("reasoning") or {
        "insufficient_numeric_evidence": numeric.get("reason") or "本次未取得可用于判断的数值证据。",
        "llm_unavailable": "本次数值证据已保存。连接专家模型后可完成故障判断。",
        "llm_budget_exhausted": "本次数值证据已保存。本轮模型调用预算已用完。",
    }.get(result.get("error"), "本次数值证据已保存，模型未返回可校验的诊断。")
    follow_up = result.get("follow_up", "")
    if fault_type in SPECIFIC_FAULTS and segment.get("conflict"):
        follow_up = follow_up or "需进一步核对存在分歧的分段的特征频率及其重复性。"
    final = {
        "fault_type": fault_type,
        "confidence": result.get("confidence", 0.0),
        "severity": "无法判断",
        "reasoning": reason,
        "source": result.get("source", "evidence_only"),
        "evidence_refs": result.get("evidence_refs", []),
        "follow_up": follow_up,
        **({"model_error": result["error"]} if "error" in result else {}),
        "evidence_summary": state.get("evidence_summary") or {},
        "numeric": {
            "fault_type": numeric.get("fault_type"),
            "confidence": numeric.get("confidence"),
            "leading_family": numeric.get("leading_family"),
            "bands_hz": numeric.get("bands"),
            "evidence_route": numeric.get("evidence_route"),
            "used_methods": used,
        },
        "assessment": state.get("assessment"),
        "processing_plan": state.get("processing_plan"),
        "rounds": state.get("round_history"),
        "best_round": state.get("best_round"),
    }
    status = "limited" if fault_type == "unknown" else "completed"
    review_rules = (runtime.skill or {}).get("diagnosis_review")
    if final["source"] == "skill_llm" and review_rules:
        report = build_narrative({**state, "final": final, "status": status})["text"]
        draft = {**result, "follow_up": follow_up}
        review = runtime.controller.review_diagnosis(model_evidence(signal, review=True), draft, report, review_rules)
        final["reason_review"] = {**review, "draft": draft}
    final["llm_calls"] = len(runtime.controller.calls)
    return {"final": final, "status": status}


def route_assessment(state: GraphState, runtime: Runtime) -> str:
    assessment = state.get("assessment") or {}
    quality = assessment.get("quality") or {}
    if assessment.get("status") in {"invalid", "limited"}:
        return "limited"
    if not quality.get("valid_for_analysis", False):
        return "limited"
    return "preprocess"


def limited_final_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    assessment = state.get("assessment") or {}
    prior_final = state.get("final") or {}
    final = {
        "fault_type": "unknown",
        "confidence": 0.0,
        "reasoning": prior_final.get("reasoning") or "初步信号评估未通过："
        + ", ".join((assessment.get("quality") or {}).get("invalidating_flags", [])),
        "source": prior_final.get("source", "signal_assessment"),
        "assessment": assessment,
        "numeric": None,
    }
    return {"final": final, "status": "limited"}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return f"<array shape={value.shape}>"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def finalize_node(state: GraphState, runtime: Runtime) -> dict[str, Any]:
    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    llm_calls = [asdict(call) for call in runtime.controller.calls]
    numeric = dict(state.get("numeric_result") or {})
    for key in ("evidence", "baseline"):
        value = numeric.get(key)
        if hasattr(value, "model_dump"):
            numeric[key] = value.model_dump(mode="json")
    audit = {
        "case_id": runtime.case_id,
        "status": state.get("status"),
        "adaptive": runtime.adaptive,
        "rounds": len(state.get("round_history") or []),
        "assessment": state.get("assessment", {}),
        "speed_reference": state.get("speed_reference"),
        "preprocessing_plan": state.get("preprocessing_plan", {}),
        "preprocess_result": state.get("preprocess_result", {}),
        "processing_plan": state.get("processing_plan", {}),
        "round_history": state.get("round_history", []),
        "best_round": state.get("best_round", {}),
        "best_bands": state.get("best_bands", []),
        "numeric_result": numeric,
        "plan_round": state.get("plan_round", 0),
        "max_rounds": runtime.max_rounds,
        "signal_profile": state.get("signal_profile", {}),
        "analysis_plan": state.get("analysis_plan", []),
        "plan_history": state.get("plan_history", []),
        "step_observations": state.get("step_observations", []),
        "step_decisions": state.get("step_decisions", []),
        "evidence_adopted": state.get("evidence_adopted", False),
        "gate": state.get("gate", {}),
        "baseline": state.get("baseline", {}),
        "preprocess_results": state.get("preprocess_results", []),
        "l2_results": state.get("l2_results", []),
        "precision_results": state.get("precision_results", []),
        "planner_decisions": state.get("planner_decisions", []),
        "evaluator_decisions": state.get("evaluator_decisions", []),
        "llm_calls": llm_calls,
        "final": state.get("final"),
        "narrative": build_narrative(state),
        "errors": state.get("errors", []),
        "tool_specs": runtime.registry.specs(),
        "fault_frequencies_hz": runtime.fault_frequencies_hz,
    }
    (runtime.output_dir / "result.json").write_text(
        json.dumps(_jsonable(audit), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"audit": {"result_path": str(runtime.output_dir / "result.json")}}


def route_profile(state: GraphState, runtime: Runtime) -> str:
    return "plan" if runtime.adaptive else "evidence"


def route_gate_after_evidence(state: GraphState, runtime: Runtime) -> str:
    gate = state.get("gate") or {}
    if gate.get("verdict") == "clear":
        return "synthesize"
    if gate.get("verdict") == "limited":
        return "limited"
    if runtime.adaptive and int(state.get("plan_round", 0)) < runtime.max_plan_rounds:
        return "plan"
    return "synthesize"


def route_plan_next(state: GraphState) -> str:
    return "evidence" if state.get("next_route") == "evidence" else "execute"
    gate = state.get("gate") or {}
    if state.get("status") == "rejected":
        return "limited"
    if runtime is not None and not runtime.adaptive:
        return "synthesize"
    return {"clear": "synthesize", "limited": "limited"}.get(
        gate.get("verdict"), "plan"
    )


def route_plan(state: GraphState) -> str:
    return "synthesize" if state.get("next_action") == "stop" else "precision"


def route_evaluate(state: GraphState) -> str:
    gate = state.get("gate") or {}
    if gate.get("verdict") == "clear":
        return "synthesize"
    if state.get("status") == "diagnosed":
        return "synthesize"
    decisions = state.get("evaluator_decisions") or []
    last = decisions[-1] if decisions else {}
    if last.get("decision") == "stop":
        return "synthesize"
    if int(state.get("round", 0)) >= int(state.get("max_rounds", 3)):
        return "synthesize"
    return "plan"
