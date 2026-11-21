"""Public execution records; arrays are described, never copied into UI events."""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import asdict, is_dataclass

import numpy as np


def public_value(value):
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype), "values_omitted": True}
    if hasattr(value, "model_dump"):
        return public_value(value.model_dump(mode="python"))
    if is_dataclass(value):
        return public_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [public_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, np.generic):
        return public_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def observe_call(callback, name, params, function, *, kind="tool", output_view=None):
    if callback is None:
        return function()
    identity = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    callback({"type": "start", "id": identity, "name": name, "kind": kind, "params": public_value(params)})
    try:
        result = function()
    except Exception as exc:
        callback({"type": "finish", "id": identity, "name": name, "kind": kind, "status": "failed",
                  "output": {"error": str(exc)}, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
        raise
    output = result.to_dict() if hasattr(result, "to_dict") else result
    status = "completed"
    if isinstance(output, dict):
        state = output.get("status")
        if state in {"error", "rejected"}:
            status = "failed"
        elif state in {"unavailable", "needs_input", "limited"}:
            status = "limited"
    callback({"type": "finish", "id": identity, "name": name, "kind": kind, "status": status,
              "output": public_value(output_view(output) if output_view else output),
              "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)})
    return result


def record_call(callback, name, params, output):
    """Publish a completed tool step recorded by the instrumented algorithm."""
    callback({"type": "record", "id": uuid.uuid4().hex[:12], "name": name, "kind": "tool",
              "params": public_value(params), "output": public_value(output), "status": "completed"})
