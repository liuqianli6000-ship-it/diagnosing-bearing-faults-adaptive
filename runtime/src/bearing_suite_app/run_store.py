"""诊断运行的可重放审计存储。"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def signal_sha256(values: Any) -> str:
    import numpy as np

    canonical = np.asarray(values, dtype="<f8").reshape(-1)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


class DiagnosticRunStore:
    """每次诊断一个目录；JSON原子写入，事件使用哈希链追加。"""

    def __init__(
        self,
        root: str | Path,
        *,
        mode: str,
        config: Mapping[str, Any],
        input_metadata: Mapping[str, Any],
        run_id: str | None = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex
        self.root = Path(root).expanduser().resolve()
        self.run_dir = (self.root / self.run_id).resolve()
        if self.root != self.run_dir and self.root not in self.run_dir.parents:
            raise ValueError("诊断运行目录越界")
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("evidence", "prompts", "responses"):
            (self.run_dir / name).mkdir()
        self._sequence = 0
        self._previous_event_hash: str | None = None
        config_snapshot = dict(config)
        self.write_json("config_snapshot.json", config_snapshot)
        self.write_json(
            "run.json",
            {
                "schema_version": "bearing-diagnostic-run/v1",
                "run_id": self.run_id,
                "mode": mode,
                "created_at_unix": time.time(),
                "config_sha256": payload_sha256(config_snapshot),
                "input": dict(input_metadata),
            },
        )
        self.append_event("run_started", {"mode": mode})

    def _resolve(self, relative_path: str) -> Path:
        path = (self.run_dir / relative_path).resolve()
        if self.run_dir != path and self.run_dir not in path.parents:
            raise ValueError("审计文件路径越界")
        return path

    def write_json(self, relative_path: str, payload: Any) -> str:
        path = self._resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return str(path)

    def append_event(self, event_type: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._sequence += 1
        event = {
            "sequence": self._sequence,
            "timestamp_unix": time.time(),
            "event_type": str(event_type),
            "payload": dict(payload),
            "previous_event_sha256": self._previous_event_hash,
        }
        event_hash = payload_sha256(event)
        event["event_sha256"] = event_hash
        path = self._resolve("events.jsonl")
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._previous_event_hash = event_hash
        return event

    def record_evidence(self, full: Mapping[str, Any], compact: Mapping[str, Any]) -> None:
        full_path = "evidence/full.json"
        compact_path = "evidence/compact.json"
        self.write_json(full_path, full)
        self.write_json(compact_path, compact)
        self.append_event(
            "evidence_recorded",
            {
                "full_path": full_path,
                "compact_path": compact_path,
                "full_sha256": payload_sha256(full),
                "compact_sha256": payload_sha256(compact),
            },
        )

    def record_exchange(self, exchange: Mapping[str, Any], *, attempt: int) -> None:
        request = {
            key: exchange.get(key)
            for key in (
                "provider",
                "model",
                "messages",
                "request_options",
                "started_at_unix",
            )
        }
        response = {
            key: exchange.get(key)
            for key in (
                "raw_text",
                "tool_calls",
                "request_id",
                "response_model",
                "system_fingerprint",
                "usage",
                "finish_reason",
                "duration_ms",
                "error",
            )
        }
        request_path = f"prompts/request_{attempt:03d}.json"
        response_path = f"responses/response_{attempt:03d}.json"
        self.write_json(request_path, request)
        self.write_json(response_path, response)
        self.append_event(
            "llm_exchange_recorded",
            {
                "attempt": int(attempt),
                "request_path": request_path,
                "response_path": response_path,
                "request_sha256": payload_sha256(request),
                "response_sha256": payload_sha256(response),
            },
        )

    def finalize(self, result: Mapping[str, Any]) -> str:
        relative_path = "final_result.json"
        self.write_json(relative_path, result)
        self.append_event(
            "run_completed",
            {
                "result_path": relative_path,
                "result_sha256": payload_sha256(result),
            },
        )
        return str(self.run_dir)


def verify_run(run_dir: str | Path) -> dict[str, Any]:
    """校验审计目录的事件哈希链和关键文件内容哈希。"""

    root = Path(run_dir).expanduser().resolve()
    errors: list[str] = []

    def load_json(relative_path: str) -> Any:
        path = (root / relative_path).resolve()
        if root != path and root not in path.parents:
            raise ValueError(f"审计引用路径越界: {relative_path}")
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        run = load_json("run.json")
        config = load_json("config_snapshot.json")
        if run.get("config_sha256") != payload_sha256(config):
            errors.append("config_snapshot_hash_mismatch")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"run_metadata_invalid:{type(exc).__name__}: {exc}")

    events_path = root / "events.jsonl"
    previous: str | None = None
    expected_sequence = 1
    events: list[dict[str, Any]] = []
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            events.append(event)
            if event.get("sequence") != expected_sequence:
                errors.append(f"event_sequence_mismatch:{expected_sequence}")
            if event.get("previous_event_sha256") != previous:
                errors.append(f"event_previous_hash_mismatch:{expected_sequence}")
            claimed = event.get("event_sha256")
            unhashed = dict(event)
            unhashed.pop("event_sha256", None)
            actual = payload_sha256(unhashed)
            if claimed != actual:
                errors.append(f"event_hash_mismatch:{expected_sequence}")
            previous = claimed
            expected_sequence += 1
    except Exception as exc:  # noqa: BLE001
        errors.append(f"events_invalid:{type(exc).__name__}: {exc}")

    for event in events:
        payload = event.get("payload", {})
        try:
            if event.get("event_type") == "evidence_recorded":
                full = load_json(str(payload["full_path"]))
                compact = load_json(str(payload["compact_path"]))
                if payload_sha256(full) != payload.get("full_sha256"):
                    errors.append("full_evidence_hash_mismatch")
                if payload_sha256(compact) != payload.get("compact_sha256"):
                    errors.append("compact_evidence_hash_mismatch")
            elif event.get("event_type") == "llm_exchange_recorded":
                request = load_json(str(payload["request_path"]))
                response = load_json(str(payload["response_path"]))
                if payload_sha256(request) != payload.get("request_sha256"):
                    errors.append("llm_request_hash_mismatch")
                if payload_sha256(response) != payload.get("response_sha256"):
                    errors.append("llm_response_hash_mismatch")
            elif event.get("event_type") == "run_completed":
                result = load_json(str(payload["result_path"]))
                if payload_sha256(result) != payload.get("result_sha256"):
                    errors.append("final_result_hash_mismatch")
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"referenced_artifact_invalid:{event.get('sequence')}:"
                f"{type(exc).__name__}: {exc}"
            )
    return {
        "valid": not errors,
        "run_dir": str(root),
        "event_count": len(events),
        "errors": errors,
    }
