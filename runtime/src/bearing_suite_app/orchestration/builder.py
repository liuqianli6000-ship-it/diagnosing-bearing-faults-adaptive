"""Entry point for the evidence-based bearing diagnosis modes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from bearing_suite_app.intent import resolve_mode
from bearing_suite_app.llm_agent.controller import LLMController
from bearing_suite_app.toolbox.registry import build_default_registry
from bearing_suite_app.execution import observe_call, record_call
from bearing_suite_app.dataset_diagnosis import public_case, run_dataset_case
from bearing_suite_app.evidence_expert import DIAGNOSIS_CALLS
from bearing_suite_app import geometry as bearing_geometry
from bearing_suite_app.config import load_config
from bearing_suite_app.assessment.signal_assessment import assess_signal
from bearing_suite_app.processing.plan import build_plan, normalize_plan, valid_processing_plan
from bearing_suite_app.processing.rounds import run_rounds
from bearing_suite_app.toolbox.contracts import ToolContext

if TYPE_CHECKING:
    from bearing_suite_app.agent import BearingDiagnosisAgent



def run_case(
    signal: np.ndarray,
    *,
    sampling_rate_hz: float,
    rpm: float,
    geometry: dict[str, Any],
    case_id: str,
    provider: str = "qwen",
    env_path: str | None = None,
    config_path: str | Path | None = None,
    max_rounds: int | None = None,
    max_trials: int = 3,
    max_plan_rounds: int = 2,
    max_llm_calls: int = 9,
    adaptive: bool | None = None,
    request_text: str | None = None,
    output_dir: str | Path | None = None,
    agent: BearingDiagnosisAgent | None = None,
    controller: LLMController | None = None,
    on_event=None,
    on_envelope_spectra=None,
    on_analysis_signal=None,
    diagnosis_input=None,
    unit: str = "g",
    plot: bool = False,
    plot_lock=None,
):
    adaptive = resolve_mode(request_text, adaptive)
    max_rounds = min(max_rounds if max_rounds is not None else 2, 2) if adaptive else 0
    if controller is None:
        controller = LLMController.from_provider(
            provider, env_path=env_path, max_calls=max_llm_calls
        )
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        output_dir = Path("outputs") / "diagnosis_v2" / stamp
    case = public_case(signal, sampling_rate_hz=sampling_rate_hz, rpm=rpm, geometry=geometry,
                       diagnosis_input=diagnosis_input, unit=unit)

    def prepare(chain):
        fs = float(sampling_rate_hz)
        config = agent.config if agent is not None else load_config(config_path)
        assessment = assess_signal(signal, fs, rpm, geometry, config["signal"]["quality"],
                                   config["signal"]["diagnosability"], select_bands=False)
        bands = next(([row["band_hz"] for row in feature["value"]]
                      for feature in chain["extracted_features"] if feature["name"].endswith(".envelope.bands")), [])
        if bands:
            assessment["spectral"]["spectral_kurtosis_bands_hz"] = bands
        registry = build_default_registry()
        registry.on_event = on_event
        context = ToolContext(fs, rpm, geometry, bearing_geometry.fault_frequencies_hz(
            rpm, bearing_geometry.fault_frequency_ratios(geometry)))

        def plan_processing():
            plan = controller.plan_processing(assessment, reserve_calls=DIAGNOSIS_CALLS)
            if not valid_processing_plan(plan, fs):
                plan = build_plan(assessment)
            return normalize_plan(plan)

        plan = observe_call(on_event, "plan_adaptive_processing", {}, plan_processing, kind="stage")
        result = observe_call(on_event, "run_adaptive_processing", {"max_rounds": max_rounds},
                              lambda: run_rounds(signal, fs, plan, registry, context, assessment,
                                                 max_rounds=max_rounds, max_trials=min(max_trials, 3)),
                              kind="overview", output_view=lambda _: None)
        if on_event is not None:
            for item in result["rounds"]:
                record_call(on_event, "compare_processing_round", {"round": item["round"]},
                            {key: item[key] for key in ("before", "after", "hard_gate", "deltas")})
        return result

    return run_dataset_case(case, signal, controller=controller, case_id=case_id, output_dir=output_dir,
                            adaptive=adaptive, prepare=prepare if max_rounds > 0 else None,
                            on_event=on_event, on_envelope_spectra=on_envelope_spectra,
                            on_analysis_signal=on_analysis_signal, plot=plot, plot_lock=plot_lock,
                            config_path=config_path)
