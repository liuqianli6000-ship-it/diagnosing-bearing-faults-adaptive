"""Round 0 baseline + up to three refined processing rounds."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from bearing_suite_app.toolbox.contracts import ToolContext

from . import metrics as metric_mod
from .refine import decide_mode, refine_plan, retry_plan


def _apply_filters(
    x: np.ndarray,
    fs: float,
    filters: list[dict[str, Any]],
    registry: Any,
    context: ToolContext,
    *,
    max_trials: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    normalized = []
    for step in filters:
        item = dict(step)
        params = dict(step.get("params") or {})
        if item.get("tool") == "bandpass":
            params["low_hz"] = params.pop("low_hz", params.pop("low", 5.0))
            params["high_hz"] = params.pop("high_hz", params.pop("high", fs * 0.475))
        item["params"] = params
        normalized.append(item)
    return _apply_processing(x, normalized, registry, context, max_trials=max_trials)


def _apply_processing(
    x: np.ndarray,
    steps: list[dict[str, Any]],
    registry: Any,
    context: ToolContext,
    *,
    max_trials: int = 2,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    current = np.asarray(x, dtype=float).reshape(-1)
    applied = []
    for step in steps:
        tool = str(step.get("tool") or "")
        try:
            params = dict(step.get("params") or {})
            auto = bool(params.pop("auto", False))
            requested_trials = int(params.pop("max_trials", max_trials))
            result = registry.execute(
                tool, current, context, params,
                auto=auto, max_trials=max(1, min(max_trials, requested_trials)),
            )
        except Exception as exc:  # noqa: BLE001
            applied.append({"tool": tool, "status": "error", "error": str(exc)})
            continue
        if result.status == "ok" and result.arrays.get("signal") is not None:
            current = np.asarray(result.arrays["signal"], dtype=float).reshape(-1)
        applied.append(
            {
                "tool": tool,
                "params": result.params,
                "status": result.status,
                "metrics": result.metrics,
                "error": result.error,
            }
        )
    return current, applied


def _bands(plan: Mapping[str, Any], assessment: Mapping[str, Any]) -> list[list[float]]:
    bands = [
        [float(band[0]), float(band[1])]
        for band in (plan.get("bands") or [])
        if len(band) >= 2
    ]
    if bands:
        return bands
    return [
        [float(band[0]), float(band[1])]
        for band in (assessment.get("spectral") or {}).get("spectral_kurtosis_bands_hz", [])
        if len(band) >= 2
    ]


def run_rounds(
    raw: np.ndarray,
    fs: float,
    initial_plan: Mapping[str, Any],
    registry: Any,
    context: ToolContext,
    assessment: Mapping[str, Any],
    *,
    max_rounds: int = 3,
    max_trials: int = 2,
) -> dict[str, Any]:
    if context is not None:
        context.notes["processing_objective"] = "signal_state"
    bands = _bands(initial_plan, assessment)
    baseline_gate = metric_mod.hard_gate(raw, raw, fs, bands)
    if not baseline_gate["passed"]:
        raise ValueError("invalid processing baseline: " + ", ".join(baseline_gate["reasons"]))
    baseline_after = metric_mod.metrics(raw, fs, bands)
    baseline = {
        "round": 0,
        "parent": None,
        "plan": {"round": 0, "method": "none", "filters": [], "processing": [], "bands": bands},
        "before": baseline_after,
        "after": baseline_after,
        "deltas": {},
        "relative": {},
        "qualitative": {"parts": [], "overall": "baseline"},
        "hard_gate": {"passed": True, "reasons": []},
        "rank": None,
    }
    plans = list(initial_plan.get("rounds") or [])[:1]
    if not plans:
        return {
            "baseline": baseline,
            "rounds": [],
            "best": baseline,
            "best_bands": bands,
            "ranking": [baseline],
            "best_signal": np.asarray(raw, dtype=float).reshape(-1),
        }

    history: list[dict[str, Any]] = []
    signals: dict[int, np.ndarray] = {0: np.asarray(raw, dtype=float).reshape(-1)}
    previous_plan = plans[0]
    previous_result = baseline
    for round_no in range(1, max_rounds + 1):
        if round_no == 1:
            plan = dict(previous_plan)
        elif round_no == 2:
            mode = decide_mode(previous_result, baseline)
            plan = (
                retry_plan(previous_plan, previous_result, assessment, round_no, fs)
                if mode == "retry"
                else None
            )
            if plan is None:
                plan = refine_plan(previous_plan, previous_result, assessment, round_no, fs)
        else:
            plan = refine_plan(previous_plan, previous_result, assessment, round_no, fs)
        round_bands = _bands(plan, assessment)
        filtered, applied_filters = _apply_filters(
            raw, fs, plan.get("filters") or [], registry, context, max_trials=max_trials,
        )
        processed, applied_processing = _apply_processing(
            filtered, plan.get("processing") or [], registry, context, max_trials=max_trials,
        )
        signals[round_no] = processed
        gate = metric_mod.hard_gate(raw, processed, fs, round_bands)
        if any(item.get("status") != "ok" for item in applied_filters + applied_processing):
            gate["reasons"].append("tool_failed")
            gate["passed"] = False
        # A fixed evaluation band bank makes metrics comparable across rounds.
        # Rejected outputs must not reach Hilbert/FFT or contaminate ranking.
        after = metric_mod.metrics(processed, fs, bands) if gate["passed"] else {}
        comparison = metric_mod.compare(baseline_after, after)
        record = {
            "round": round_no,
            "parent": int(plan.get("parent", round_no - 1)),
            "mode": plan.get("mode", "initial" if round_no == 1 else "refine"),
            "plan": plan,
            "filters": applied_filters,
            "processing": applied_processing,
            "bands": round_bands,
            "before": baseline_after,
            "after": after,
            "deltas": comparison["deltas"],
            "relative": comparison["relative"],
            "qualitative": comparison["qualitative"],
            "hard_gate": gate,
        }
        history.append(record)
        previous_plan = plan
        previous_result = record
        if not gate["passed"]:
            break

    candidates = [baseline] + history
    ranking = metric_mod.rank_candidates(candidates)
    best = ranking.get("best") or baseline
    best_bands = best.get("plan", {}).get("bands") or bands
    return {
        "baseline": baseline,
        "rounds": history,
        "best": best,
        "best_bands": best_bands,
        "ranking": ranking.get("ranking", []),
        "best_signal": signals.get(int(best.get("round", 0)), signals[0]),
    }
