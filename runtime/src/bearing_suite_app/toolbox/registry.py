"""Tool registry and bounded adaptive parameter search.

The registry is the single source of truth for what the LangGraph agent may
call.  ``execute(..., auto=True)`` tries a small, bounded candidate set and
returns the best result plus every trial.  The LLM only proposes a method or
a parameter direction; bounds, validation and scoring stay local.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np

from .contracts import ToolContext, ToolParameter, ToolResult, ToolSpec
from bearing_suite_app.execution import observe_call

ToolFunction = Callable[[np.ndarray, ToolContext, dict[str, Any]], ToolResult]


@dataclass
class RegisteredTool:
    spec: ToolSpec
    function: ToolFunction


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}
        self.on_event = None

    def register(self, spec: ToolSpec, function: ToolFunction) -> None:
        if spec.name in self._tools:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._tools[spec.name] = RegisteredTool(spec=spec, function=function)

    def get(self, name: str) -> RegisteredTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def names(self, layer: str | None = None) -> list[str]:
        return sorted(
            name
            for name, item in self._tools.items()
            if layer is None or item.spec.layer == layer
        )

    def specs(self, layer: str | None = None) -> list[dict[str, Any]]:
        return [
            self._tools[name].spec.to_dict()
            for name in self.names(layer)
        ]

    def validate(self, spec: ToolSpec, params: dict[str, Any]) -> dict[str, Any]:
        declared = {p.name: p for p in spec.parameters}
        unknown = set(params) - set(declared)
        if unknown:
            raise ValueError(f"unknown parameters for {spec.name}: {sorted(unknown)}")
        normalized = dict(spec.defaults())
        normalized.update(params)
        for name, value in normalized.items():
            parameter = declared[name]
            if parameter.type in {"int", "float"}:
                if value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"{spec.name}.{name} must be numeric") from exc
                if not math.isfinite(number):
                    raise ValueError(f"{spec.name}.{name} must be finite")
                if parameter.type == "int":
                    if not number.is_integer():
                        raise ValueError(f"{spec.name}.{name} must be an integer")
                    number = int(number)
                if parameter.minimum is not None and number < parameter.minimum:
                    raise ValueError(
                        f"{spec.name}.{name}={number} below minimum {parameter.minimum}"
                    )
                if parameter.maximum is not None and number > parameter.maximum:
                    raise ValueError(
                        f"{spec.name}.{name}={number} above maximum {parameter.maximum}"
                    )
                normalized[name] = number
            elif parameter.type == "enum":
                if value not in parameter.candidates:
                    raise ValueError(
                        f"{spec.name}.{name}={value!r} not in {parameter.candidates}"
                    )
        return normalized

    def execute(
        self, name: str, signal: np.ndarray, context: ToolContext, params: dict[str, Any] | None = None,
        *, auto: bool = False, max_trials: int = 4, min_gain: float = 0.10,
    ) -> ToolResult:
        return observe_call(self.on_event, name, {**(params or {}), "auto": auto, "max_trials": max_trials},
                            lambda: self._execute(name, signal, context, params, auto=auto,
                                                  max_trials=max_trials, min_gain=min_gain))

    def _execute(
        self,
        name: str,
        signal: np.ndarray,
        context: ToolContext,
        params: dict[str, Any] | None = None,
        *,
        auto: bool = False,
        max_trials: int = 4,
        min_gain: float = 0.10,
    ) -> ToolResult:
        registered = self.get(name)
        normalized = self.validate(registered.spec, dict(params or {}))
        if not auto:
            return registered.function(signal, context, normalized)

        candidates = self._candidates(registered.spec, context, params or {})
        candidates = candidates[: max(1, int(max_trials))]
        trials: list[dict[str, Any]] = []
        best: ToolResult | None = None
        best_score = -math.inf
        default_result: ToolResult | None = None
        first_result: ToolResult | None = None
        defaults = self.validate(registered.spec, {})
        for candidate in candidates:
            normalized = self.validate(registered.spec, candidate)
            result = registered.function(signal, context, normalized)
            if first_result is None:
                first_result = result
            score = float(result.metrics.get("objective", 0.0))
            if normalized == defaults:
                default_result = result
            trials.append(
                {
                    "params": normalized,
                    "status": result.status,
                    "objective": score,
                    "metrics": dict(result.metrics),
                    "error": result.error,
                }
            )
            if result.status == "ok" and score > best_score:
                best = result
                best_score = score
        if best is None:
            first_result.trials = trials
            return first_result
        # The default is a mandatory control.  A tuned parameter set is only
        # adopted when it beats the default by a meaningful margin; otherwise
        # the extra search is not a real gain.
        if (
            default_result is not None
            and default_result.status == "ok"
            and best.params != defaults
        ):
            default_score = float(default_result.metrics.get("objective", 0.0))
            threshold = default_score + abs(default_score) * max(0.0, float(min_gain))
            if best_score <= threshold:
                best = default_result
                best_score = default_score
                selection = "default_no_significant_gain"
            else:
                selection = "tuned_gain"
        else:
            selection = "default" if best.params == defaults else "tuned_gain"
        best.trials = trials
        best.data = dict(best.data)
        best.data["selected_trial"] = {
            "params": best.params,
            "objective": best_score,
            "selection": selection,
        }
        return best

    def _candidates(
        self,
        spec: ToolSpec,
        context: ToolContext,
        overrides: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if spec.candidates_fn is not None:
            generated = spec.candidates_fn(context)
        else:
            generated = []
        defaults = spec.defaults()
        result = [defaults]
        requested = dict(defaults)
        requested.update(overrides)
        if requested not in result:
            result.append(requested)
        for item in generated:
            merged = dict(defaults)
            merged.update(item)
            merged.update({key: value for key, value in overrides.items() if value is not None})
            if merged not in result:
                result.append(merged)
        return result


def build_default_registry() -> ToolRegistry:
    """Build the three-layer registry lazily to avoid import cycles."""

    from .l1 import register_l1
    from .l2 import register_l2
    from .l3 import register_l3
    from .assessment_tools import register_assessment

    registry = ToolRegistry()
    register_l1(registry)
    register_l2(registry)
    register_l3(registry)
    register_assessment(registry)
    return registry


def parameter(
    name: str,
    type_: str,
    description: str,
    *,
    default: Any = None,
    unit: str | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    adaptive: bool = False,
    candidates: Iterable[Any] = (),
) -> ToolParameter:
    return ToolParameter(
        name=name,
        type=type_,
        description=description,
        default=default,
        unit=unit,
        minimum=minimum,
        maximum=maximum,
        adaptive=adaptive,
        candidates=tuple(candidates),
    )
