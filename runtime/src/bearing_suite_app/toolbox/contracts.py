"""Tool contracts shared by all layers.

A tool is not just a function: version, layer, parameter ranges, adaptive
policy, output metrics and error semantics are part of the contract.  The
LangGraph nodes and the adaptive tuner use the same definition, so the model
cannot invent a tool name or a parameter outside the declared range.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class ToolParameter:
    name: str
    type: str
    description: str
    default: Any = None
    unit: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    adaptive: bool = False
    candidates: tuple[Any, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    layer: str
    family: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    output_fields: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ("invalid_parameter", "execution_error")
    candidates_fn: Callable[["ToolContext"], list[dict[str, Any]]] | None = field(
        default=None, compare=False, repr=False
    )

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.parameters if p.default is not None}

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("candidates_fn", None)
        payload["parameters"] = [asdict(p) for p in self.parameters]
        return payload


@dataclass
class ToolContext:
    sampling_rate_hz: float
    rpm: float
    geometry: Mapping[str, Any]
    fault_frequencies_hz: Mapping[str, float]
    seed: int = 42
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def max_fault_frequency_hz(self) -> float:
        values = [float(v) for v in self.fault_frequencies_hz.values() if v > 0]
        return max(values) if values else 0.0


@dataclass
class ToolResult:
    tool: str
    version: str
    layer: str
    status: str  # ok | unavailable | error | rejected
    params: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    error_code: str | None = None
    elapsed_ms: float = 0.0
    evidence_id: str | None = None
    trials: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": self.tool,
            "version": self.version,
            "layer": self.layer,
            "status": self.status,
            "params": self.params,
            "data": self.data,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "error": self.error,
            "error_code": self.error_code,
            "elapsed_ms": self.elapsed_ms,
            "evidence_id": self.evidence_id,
            "trials": self.trials,
            "array_keys": sorted(self.arrays),
        }
        if include_arrays:
            payload["arrays"] = self.arrays
        return payload


def timed(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    def wrapper(*args: Any, **kwargs: Any) -> ToolResult:
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        result.elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
        return result

    return wrapper
