"""LangGraph state and runtime context.

The raw signal lives in :class:`Runtime`, not in ``GraphState``.  This keeps
the checkpointable state small and prevents a raw array or true label from
being serialized by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from bearing_suite_app.agent import BearingDiagnosisAgent
from bearing_suite_app.llm_agent.controller import LLMController
from bearing_suite_app.toolbox.registry import ToolRegistry


class GraphState(TypedDict, total=False):
    case_id: str
    status: str
    adaptive: bool
    round: int
    max_rounds: int
    gate: dict[str, Any]
    baseline: dict[str, Any]
    evidence_summary: dict[str, Any]
    signal_profile: dict[str, Any]
    analysis_plan: list[dict[str, Any]]
    plan_index: int
    plan_round: int
    plan_history: list[dict[str, Any]]
    step_observations: list[dict[str, Any]]
    step_decisions: list[dict[str, Any]]
    next_route: str | None
    evidence_adopted: bool
    assessment: dict[str, Any]
    speed_reference: dict[str, Any] | None
    preprocessing_plan: dict[str, Any]
    preprocess_result: dict[str, Any]
    preprocessed_signal: Any
    processing_plan: dict[str, Any]
    round_history: list[dict[str, Any]]
    best_round: dict[str, Any]
    best_signal: Any
    best_bands: list[list[float]]
    numeric_result: dict[str, Any]
    preprocess_results: list[dict[str, Any]]
    l2_results: list[dict[str, Any]]
    precision_results: list[dict[str, Any]]
    planner_decisions: list[dict[str, Any]]
    evaluator_decisions: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    next_action: str | None
    final: dict[str, Any] | None
    errors: list[str]
    audit: dict[str, Any]


@dataclass
class Runtime:
    agent: BearingDiagnosisAgent
    registry: ToolRegistry
    controller: LLMController
    skill: dict[str, Any]
    raw_signal: np.ndarray
    sampling_rate_hz: float
    rpm: float
    geometry: dict[str, Any]
    fault_frequencies_hz: dict[str, float]
    case_id: str
    max_rounds: int = 3
    max_trials: int = 3
    max_plan_rounds: int = 2
    adaptive: bool = True
    output_dir: Path = field(default_factory=Path)
    notes: dict[str, Any] = field(default_factory=dict)
    signal: np.ndarray = field(default_factory=lambda: np.asarray([]))
    base_signal: np.ndarray = field(default_factory=lambda: np.asarray([]))
    evidence: Any = None
    baseline: Any = None
    evidence_history: list[dict[str, Any]] = field(default_factory=list)
    preprocess_results: list[dict[str, Any]] = field(default_factory=list)
    precision_results: list[dict[str, Any]] = field(default_factory=list)
    precision_arrays: list[dict[str, np.ndarray]] = field(default_factory=list)
    step_arrays: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    on_event: Any = None
    on_envelope_spectra: Any = None

    def __post_init__(self) -> None:
        if self.signal.size == 0:
            # Intake owns shape/dtype validation; keep the original channel layout.
            self.signal = self.raw_signal
        if self.base_signal.size == 0:
            self.base_signal = self.signal

    def tool_context(self):
        from bearing_suite_app.toolbox.contracts import ToolContext

        return ToolContext(
            sampling_rate_hz=self.sampling_rate_hz,
            rpm=self.rpm,
            geometry=self.geometry,
            fault_frequencies_hz=self.fault_frequencies_hz,
            notes=self.notes,
        )
