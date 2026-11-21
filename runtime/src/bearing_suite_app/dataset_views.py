"""为 Web、应用 CLI、主链批测和独立 Skill 读取诊断清单。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CRRC_GEOMETRY: dict[str, Any] = {
    "type": "F-801804.ZL",
    "feature_frequency_ratios": {"BPFI": 10.2736, "BPFO": 7.7264, "BSF": 6.9252},
}

BJTU_GEOMETRY: dict[str, Any] = {
    "type": "HRB_32305",
    "feature_frequency_ratios": {"BPFI": 7.889, "BPFO": 5.111, "BSF": 2.203},
}


def normalize_bjtu_record(record: dict[str, Any]) -> dict[str, Any]:
    """把 BJTU 800 条原始索引适配为统一诊断输入，标签只用于结果评分。"""

    duration = float(record.get("duration_seconds", 1.0))
    return {
        "case_id": record.get("case_id"),
        "task_type": "bearing_fault_diagnosis",
        "bearing_type": "HRB_32305",
        "dataset": "BJTU_RAO_Bogie_Datasets",
        "input": {
            "signal_ref": record.get("signal_ref"),
            "signal_format": "npy_float32",
            "signal_shape": record.get("shape", [record.get("segment_samples")]),
            "channel_names": [record.get("channel", "CH11")],
            "channel": record.get("channel", "CH11"),
            "sampling_rate_hz": 64000.0,
            "segment_samples": record.get("segment_samples"),
            "duration_seconds": duration,
            "frequency_resolution_hz": 1.0 / duration if duration > 0 else None,
            "start_sample": record.get("start_sample"),
            "start_second": record.get("start_second"),
            "speed_hz": record.get("speed_hz"),
            "rpm": record.get("rpm"),
            "lateral_load_kN": record.get("lateral_load_kN"),
            "component": record.get("component"),
            "bearing_geometry": dict(BJTU_GEOMETRY),
            "notes": "由BJTU 800条原始manifest在诊断时适配；真实标签不传入诊断路径。",
        },
        "ground_truth": {"fault_type": record.get("fault_type")},
    }


def normalize_crrc_record(record: dict[str, Any]) -> dict[str, Any]:
    """把 CRRC 216 条原始索引适配为统一诊断输入，标签只用于结果评分。"""

    duration = float(record.get("duration_seconds") or 0.0)
    sampling_rate_hz = float(
        record.get("sampling_rate_hz")
        or (
            float(record["segment_samples"]) / duration
            if duration > 0 and record.get("segment_samples")
            else 0.0
        )
    )
    return {
        "case_id": record.get("case_id"),
        "task_type": "bearing_fault_diagnosis",
        "bearing_type": "F-801804.ZL",
        "dataset": "CRRC_Research_Bearing",
        "input": {
            "signal_ref": record.get("signal_ref"),
            "signal_format": "npy_float32",
            "signal_shape": record.get("shape", [record.get("segment_samples")]),
            "channel_names": [record.get("channel", "")],
            "channel": record.get("channel", ""),
            "sampling_rate_hz": sampling_rate_hz,
            "segment_samples": record.get("segment_samples"),
            "duration_seconds": duration,
            "frequency_resolution_hz": record.get("frequency_resolution_hz"),
            "start_sample": record.get("start_sample"),
            "start_second": record.get("start_second"),
            "speed_hz": record.get("speed_hz"),
            "rpm": record.get("rpm"),
            "bearing_side": record.get("bearing_side"),
            "damage_subtype": record.get("damage_subtype"),
            "bearing_geometry": dict(CRRC_GEOMETRY),
            "notes": "由CRRC 216条原始manifest在诊断时适配；真实标签不传入诊断路径。",
        },
        "ground_truth": {"fault_type": record.get("fault_type")},
    }


def read_manifest_payload(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSON 数组或 JSONL 清单，返回原始记录列表。"""

    manifest = Path(path)
    content = manifest.read_text(encoding="utf-8-sig")
    payload = (
        [json.loads(line) for line in content.splitlines() if line.strip()]
        if manifest.suffix.lower() == ".jsonl"
        else json.loads(content)
    )
    if not isinstance(payload, list):
        raise ValueError("JSON manifest 顶层必须是数组")
    return [dict(record) for record in payload]


def normalize_delivery_record(record: dict[str, Any], manifest: Path) -> dict[str, Any]:
    """交付 v2 metadata：只接入信号和设备输入，排除已计算证据链。"""
    signal = record["signal"]
    equipment = record["equipment"]
    condition = record["operating_condition"]
    geometry = dict(equipment["bearing_geometry"])
    aliases = {"pitch_diameter_mm": "pitch_diameter", "ball_diameter_mm": "ball_diameter",
               "rolling_element_diameter_mm": "rolling_element_diameter",
               "rolling_element_count_per_row": "rolling_element_count", "contact_angle_deg": "contact_angle_degrees"}
    for source, target in aliases.items():
        if source in geometry and target not in geometry:
            geometry[target] = geometry[source]
    payload = {
        "signal_ref": signal["signal_ref"], "sampling_rate_hz": signal["sampling_rate_hz"],
        "rpm": condition["rpm"], "bearing_geometry": geometry,
        "channel": signal.get("channel", ""), "component": equipment.get("component", ""),
        "unit": signal.get("unit", ""),
        "bearing_side": equipment.get("bearing_side", ""),
        "speed_type": "variable" if "变" in str(condition.get("operating_state", "")) else "steady",
    }
    if signal.get("segment_samples") is not None:
        payload["segment_samples"] = signal["segment_samples"]
        payload["duration_seconds"] = signal["segment_samples"] / signal["sampling_rate_hz"]
    loads = condition.get("loads")
    if isinstance(loads, dict) and loads.get("load_controller_current_a") is not None:
        payload["load_controller_current_a"] = loads["load_controller_current_a"]
    if isinstance(loads, list):
        for load in loads:
            key = {"横向载荷": "lateral_load_kN", "垂直向下载荷": "vertical_load_kN"}.get(load.get("type"))
            if key and load.get("unit") == "kN":
                payload[key] = load["value"]
    truth = record.get("ground_truth") or {}
    return {"case_id": record["case_id"], "dataset": "delivery_v2_" + manifest.parent.name,
            "bearing_type": equipment.get("bearing_type", ""), "input": payload,
            "diagnosis_input": {
                "signal": {key: signal[key] for key in ("channel", "sampling_rate_hz", "segment_samples", "unit") if key in signal},
                "equipment": equipment, "operating_condition": condition,
            },
            "ground_truth": {k: truth[k] for k in ("fault_type", "damage_subtype") if k in truth}}


def load_manifest(path: str | Path, *, signal_root: str | Path | None = None) -> list[dict[str, Any]]:
    """统一清单输入。相对波形引用默认相对于清单，或使用指定的波形根目录。"""

    manifest = Path(path).resolve()
    records = read_manifest_payload(manifest)
    if records and all(key in records[0] for key in ("signal", "equipment", "operating_condition")):
        records = [normalize_delivery_record(record, manifest) for record in records]
    elif records and "input" not in records[0]:
        name = manifest.name.lower()
        if "crrc" in name:
            records = [normalize_crrc_record(record) for record in records]
        elif "bjtu" in name:
            records = [normalize_bjtu_record(record) for record in records]
        else:
            raise ValueError("该原始manifest尚无诊断输入适配器（按文件名识别 bjtu/crrc）")
    root = Path(signal_root).resolve() if signal_root is not None else manifest.parent
    for record in records:
        payload = record["input"]
        if payload.get("signal_ref"):
            payload["signal_ref"] = str((root / payload["signal_ref"]).resolve())
    return records


def select_case(records: list[dict[str, Any]], *, case_id: str | None, index: int) -> dict[str, Any]:
    if case_id:
        for record in records:
            if record.get("case_id") == case_id:
                return record
        raise ValueError(f"案例不存在：{case_id}")
    if not 0 <= index < len(records):
        raise ValueError(f"样本索引 {index} 超出清单范围（共 {len(records)} 条）")
    return records[index]


def manifest_source_type(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "crrc" in name:
        return "crrc_main_manifest"
    if "bjtu" in name:
        return "bjtu_main_manifest"
    return "custom_manifest"
