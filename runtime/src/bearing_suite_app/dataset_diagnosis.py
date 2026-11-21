"""Run and present the dataset-construction diagnosis without dataset writeback."""
from __future__ import annotations

import logging
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from . import dataset_evidence as evidence
from . import evidence_expert as expert
from .config import load_config
from .execution import observe_call, public_value, record_call
from .visualization import save_multiband_envelope_evidence, save_signal_analysis_plot


@dataclass
class DatasetResult:
    payload: dict
    audit_path: str
    state: dict

    def to_dict(self):
        return self.payload

    @property
    def final(self):
        return self.payload["final"]

    @property
    def narrative(self):
        return self.payload["narrative"]

    @property
    def status(self):
        return self.payload["status"]

    @property
    def rounds(self):
        return self.payload["rounds"]

    @property
    def llm_calls(self):
        return self.payload["llm_calls"]


def public_case(signal, *, sampling_rate_hz, rpm, geometry, diagnosis_input=None, unit="g"):
    if diagnosis_input is None:
        diagnosis_input = {
            "signal": {"sampling_rate_hz": sampling_rate_hz, "segment_samples": len(signal), "unit": unit},
            "equipment": {"bearing_geometry": geometry},
            "operating_condition": {"rpm": rpm},
        }
    return evidence.public_input(diagnosis_input)


ANALYSIS_PLOTS = {"time": "时域波形图", "frequency": "幅值频谱图", "time_frequency": "STFT 时频图"}
_LOG = logging.getLogger(__name__)


class DiagnosisState(TypedDict, total=False):
    chain: dict
    features: dict
    speed: dict | None
    spectra: list
    frequencies: dict
    selected_signal: object
    rounds: list
    selected_round: int
    processing_error: str | None
    processed_signal: object
    review: dict
    trace: list
    raw_report: dict | None
    analysis: dict | None
    issues: list
    source: str
    message: str
    model_error: str | None
    plots: dict
    plot_errors: dict
    graph_steps: list[str]
    result: DatasetResult


def _record_extraction(raw, on_event, round_number):
    by_step = defaultdict(dict)
    matches = defaultdict(list)
    for item in raw["extracted_features"]:
        source, feature = item["name"].split(".", 1)
        by_step[int(source.removeprefix("step_"))][feature] = item["value"]
    for row in raw["frequency_matches"]:
        matches[row["source_step"]].append(row)
    for step in raw["tool_call_chain"]:
        order = step["order"]
        measured = {"features": by_step[order]} if by_step[order] else {}
        if matches[order]:
            measured["frequency_matches"] = matches[order]
        record_call(on_event, step["tool"], {"trace_step": order, "round": round_number, **step["parameters"]},
                    measured or {"message": "此步骤已完成，处理结果传给下一步。"})


def run_dataset_case(case, signal, *, controller, case_id, output_dir, adaptive, prepare=None,
                     on_event=None, on_envelope_spectra=None, on_analysis_signal=None,
                     plot=False, plot_lock=None, config_path=None):
    output = Path(output_dir)
    source_unit = case["signal"].get("unit", "g")

    def extract(values, speed=None, round_number=0):
        spectra, frequencies = [], {}

        def collect(items, fundamentals):
            spectra.extend(items)
            frequencies.update(fundamentals)

        raw = evidence.extract_evidence(
            case, values, unit=source_unit, speed_reference=speed, config_path=config_path,
            on_envelope_spectra=collect if plot or on_envelope_spectra else None,
        )
        if on_event is not None:
            _record_extraction(raw, on_event, round_number)
        chain = evidence.compact_evidence(raw)
        features = {item["name"].split(".", 1)[1]: item["value"] for item in chain["extracted_features"]}
        return {"chain": chain, "features": features, "spectra": spectra, "frequencies": frequencies}

    def extract_raw(_: DiagnosisState):
        extracted = observe_call(on_event, "extract_dataset_evidence", {"unit": source_unit},
                                 lambda: extract(signal), kind="overview", output_view=lambda _: None)
        return {**extracted, "speed": extracted["features"].get("shaft.speed_reference"),
                "selected_signal": signal, "selected_round": 0, "rounds": [],
                "review": {}, "trace": [], "issues": [], "analysis": None, "model_error": None}

    def process_candidate(state: DiagnosisState):
        try:
            processed = prepare(state["chain"])
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return {"processing_error": str(exc)}
        selected_round = int(processed.get("best", {}).get("round", 0))
        return {"rounds": processed["rounds"], "selected_round": selected_round,
                **({"processed_signal": processed["best_signal"]} if selected_round else {})}

    def extract_processed(state: DiagnosisState):
        selected_round = state["selected_round"]
        values = state["processed_signal"]
        try:
            candidate = observe_call(on_event, "extract_processed_evidence", {"round": selected_round},
                                     lambda: extract(values, state["speed"], selected_round),
                                     kind="overview", output_view=lambda _: None)
        except (ValueError, RuntimeError) as exc:
            return {"selected_round": 0, "processing_error": str(exc)}
        if not candidate["chain"]["frequency_matches"]:
            return {"selected_round": 0, "processing_error": "处理波形的频率匹配不足，采用原始波形"}
        return {**candidate, "selected_signal": values}

    def check_diagnosability(state: DiagnosisState):
        if not state["chain"]["frequency_matches"]:
            return {"source": "insufficient_numeric_evidence",
                    "message": "信号质量或频率覆盖不足，已保存本次可用的证据。"}
        if not controller.available or controller.max_calls == 0:
            return {"source": "evidence_only", "message": "本次数值证据已保存。连接专家模型后可完成故障判断。"}
        if controller.max_calls - len(controller.calls) < expert.DIAGNOSIS_CALLS:
            return {"source": "llm_budget_exhausted", "model_error": "llm_budget_exhausted",
                    "message": "本轮剩余模型调用预算不足，已保存数值证据。"}
        return {"source": "skill_llm"}

    def expert_diagnosis(state: DiagnosisState):
        review, trace = {}, []

        def diagnose():
            client = controller.budgeted_client(controller.client, role="dataset_experts")
            return expert.diagnose(case, state["chain"], client, trace, review_log=review, on_event=on_event)

        try:
            raw = observe_call(on_event, "diagnose_from_evidence", {}, diagnose,
                               kind="overview", output_view=lambda _: None)
            return {"raw_report": {**raw, "reason": expert.plain_report(raw["reason"])},
                    "review": review, "trace": trace}
        except Exception as exc:
            return {"source": "model_error", "model_error": f"{type(exc).__name__}: {exc}",
                    "message": "专家诊断未完成，已保存数值证据和频率匹配。",
                    "review": review, "trace": trace}

    def validate_report(state: DiagnosisState):
        raw, chain, review = state["raw_report"], state["chain"], state["review"]

        def check():
            issues = evidence.validate_analysis(raw, chain)
            disputes = [item["problem"] for item in review["issues"] if item["category"] == "分类疑点"]
            issues += ["分类疑点待人工确认：" + text for text in disputes]
            if disputes:
                support = evidence.classification_support(chain, raw["fault_type"])
                if support is not None and support["pattern_known"]:
                    issues.append("分类疑点核验：" + support["assessment"])
            return {"checks": issues}

        try:
            issues = observe_call(on_event, "validate_diagnosis_report", {}, check, kind="stage",
                                  output_view=lambda _: {"message": "报告核验已完成。"})["checks"]
            if raw["fault_type"] == "normal" and evidence.normal_coverage_issue(chain):
                return {"issues": issues, "review": {**review, "coverage_blocked_analysis": raw},
                        "source": "insufficient_numeric_evidence",
                        "message": "当前频率覆盖不足以判断为正常，已保存证据供复核。"}
            return {"analysis": raw, "message": raw["reason"], "issues": issues}
        except Exception as exc:
            return {"source": "model_error", "model_error": f"{type(exc).__name__}: {exc}",
                    "message": "专家诊断未完成，已保存数值证据和频率匹配。"}

    def render_plots(state: DiagnosisState):
        spectra, features, values = state["spectra"], state["features"], state["selected_signal"]
        domains = [domain for domain, available in (
            ("time", "raw.kurtosis_pearson" in features),
            ("frequency", "fft.dominant_bins" in features),
            ("time_frequency", features.get("stft.available")),
        ) if available]
        if spectra and on_envelope_spectra is not None:
            on_envelope_spectra(spectra)
        if domains and on_analysis_signal is not None:
            on_analysis_signal(values, domains)
        if not plot:
            return {}
        plots = {"analysis_plots": [], "envelope_plot": False}
        errors = {}
        fs = case["signal"]["sampling_rate_hz"]

        def draw_envelope():
            config = load_config(config_path)
            artifacts, path = save_multiband_envelope_evidence(
                spectra, state["frequencies"], output, run_id="plots",
                max_harmonics=int(config["matching"].get("max_harmonics", 5)),
                metadata={"case_id": case_id},
            )
            return {"artifacts": artifacts, "manifest": path}

        def draw_signal(domain):
            path = output / "plots" / f"{domain}.png"
            save_signal_analysis_plot(values, fs, domain, path, unit=source_unit)
            return {"saved_file": str(path), "message": f"{ANALYSIS_PLOTS[domain]}已生成。"}

        with plot_lock if plot_lock is not None else nullcontext():
            if spectra:
                try:
                    plotted = observe_call(on_event, "draw_envelope_plot", {"band_count": len(spectra)},
                                           draw_envelope, kind="stage")
                    plots["envelope_plot"] = bool(plotted["artifacts"])
                except Exception as exc:
                    errors["envelope"] = str(exc)
                    _LOG.exception("Could not render envelope plot for %s", case_id)
            for domain in domains:
                try:
                    observe_call(on_event, f"draw_{domain}_plot",
                                 {"round": state["selected_round"], "sampling_rate_hz": fs},
                                 lambda: draw_signal(domain), kind="stage")
                    plots["analysis_plots"].append(domain)
                except Exception as exc:
                    errors[domain] = str(exc)
                    _LOG.exception("Could not render %s plot for %s", domain, case_id)
        if not plots["envelope_plot"]:
            plots["plot_notice"] = ("包络谱绘制失败，数值证据已保存。" if "envelope" in errors
                                    else "当前记录未产生可绘制的包络谱。")
        return {"plots": plots, "plot_errors": errors}

    def save_result(state: DiagnosisState):
        features, analysis, message = state["features"], state["analysis"], state["message"]
        assessment = {
            "quality": {**features.get("quality", {}), "sample_count": len(signal)},
            "time_domain": {"rms": features.get("raw.rms_" + ("g" if source_unit == "g" else "au")),
                            "kurtosis": features.get("raw.kurtosis_pearson"),
                            "crest_factor": features.get("raw.crest_factor_ratio")},
            "frequency_domain": {"dominant_peaks": features.get("fft.dominant_bins", []),
                                 "spectral_peak_snr_db": features.get("fft.local_peak_snr_db")},
            "time_frequency": {"transient_count": features.get("stft.high_energy_frame_count"),
                               "nonstationarity_index": features.get("stft.nonstationarity_index_ratio")},
        }
        chain = evidence.round_evidence({**state["chain"], "analysis_text": analysis})
        final = {"fault_type": analysis["fault_type"] if analysis else None, "reasoning": message,
                 "source": state["source"], "model_error": state["model_error"],
                 "review_status": ("needs_review" if state["issues"] else "checked") if analysis else None}
        payload = {"schema_version": "dataset-diagnosis/v1", "status": "completed" if analysis else "limited",
                   "final": final, "assessment": assessment, "adaptive": adaptive, "rounds": len(state["rounds"]),
                   "selected_round": state["selected_round"], "llm_calls": len(controller.calls),
                   "speed_reference": state["speed"], "narrative": {"text": message, "diagnosis": analysis},
                   "evidence_chain": chain, "audit_path": str(output / "result.json"), **state.get("plots", {})}
        audit = {**payload, "case_id": case_id, "review": state["review"], "checks": state["issues"],
                 "processing_rounds": state["rounds"], "processing_error": state.get("processing_error"),
                 "expert_calls": state["trace"], "plot_errors": state.get("plot_errors", {}),
                 "graph_steps": [*state["graph_steps"], "save_diagnosis"]}

        def save_audit():
            expert.write_json(output / "result.json", public_value(audit))
            return {"saved_file": payload["audit_path"]}

        observe_call(on_event, "save_diagnosis", {}, save_audit, kind="stage")
        return {"result": DatasetResult(public_value(payload), payload["audit_path"], {
            "assessment": assessment, "speed_reference": state["speed"],
            "numeric_result": {"evidence": {**chain, "fault_frequencies_hz": features.get("theory.fundamentals_hz", {})},
                               "bands": [row["band_hz"] for row in features.get("envelope.bands", [])]},
        })}

    graph = StateGraph(DiagnosisState)
    for name, action in (
        ("extract_dataset_evidence", extract_raw), ("adaptive_processing", process_candidate),
        ("extract_processed_evidence", extract_processed), ("diagnosis_gate", check_diagnosability),
        ("diagnose_from_evidence", expert_diagnosis), ("validate_diagnosis_report", validate_report),
        ("render_diagnosis_plots", render_plots), ("save_diagnosis", save_result),
    ):
        def tracked(state: DiagnosisState, name=name, action=action):
            updates = action(state)
            return {**updates, "graph_steps": [*state["graph_steps"], name]}

        graph.add_node(name, tracked)
    graph.add_edge(START, "extract_dataset_evidence")
    graph.add_conditional_edges("extract_dataset_evidence",
        lambda state: bool(prepare is not None and state["chain"]["frequency_matches"]),
        {True: "adaptive_processing", False: "diagnosis_gate"})
    graph.add_conditional_edges("adaptive_processing", lambda state: bool(state["selected_round"]),
        {True: "extract_processed_evidence", False: "diagnosis_gate"})
    graph.add_edge("extract_processed_evidence", "diagnosis_gate")
    graph.add_conditional_edges("diagnosis_gate", lambda state: state["source"] == "skill_llm",
        {True: "diagnose_from_evidence", False: "render_diagnosis_plots"})
    graph.add_conditional_edges("diagnose_from_evidence", lambda state: bool(state.get("raw_report")),
        {True: "validate_diagnosis_report", False: "render_diagnosis_plots"})
    graph.add_edge("validate_diagnosis_report", "render_diagnosis_plots")
    graph.add_edge("render_diagnosis_plots", "save_diagnosis")
    graph.add_edge("save_diagnosis", END)
    return graph.compile().invoke({"graph_steps": []})["result"]
