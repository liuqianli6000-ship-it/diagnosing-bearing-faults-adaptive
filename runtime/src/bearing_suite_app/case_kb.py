"""案例库检索（保留，精简版）。

保留案例库数据与一个最小可用检索接口：确定性基线路径可按主导故障族取解释参考，
RAG 路径由编排层按 BPFO/BPFI/BSF/NONE 平衡取样，避免基线标签锚定。

说明：案例库的完整检索/排序/加权方案后续单独设计，这里只提供最基础的
family 匹配 + 参考视图，避免原项目里 0.65/0.35 等手设权重的冗余设计。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

FAMILY_LABEL = {
    "BPFO": "外圈故障",
    "BPFI": "内圈故障",
    "BSF": "滚动体故障",
    "NONE": "正常/健康",
}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    """加载 JSONL 案例库。"""

    if not Path(path).is_file():
        return []
    cases: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cases


def retrieve_similar(
    cases: Iterable[dict[str, Any]],
    *,
    family: str | None,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """按故障族匹配参考案例（family == NONE 时匹配健康参考）。"""

    target = (family or "NONE").upper()
    scored: list[tuple[float, dict[str, Any]]] = []
    for case in cases:
        case_family = str(case.get("bearing", {}).get("fault_family") or "NONE").upper()
        if case_family == target:
            score = 2.0
        elif target == "NONE" and case_family == "NONE":
            score = 2.0
        elif target != "NONE" and case_family == "NONE":
            score = 0.0
        else:
            score = 0.0
        if score > 0:
            scored.append((score, case))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [case for _, case in scored[: int(top_k)]]


def reference_view(case: Mapping[str, Any]) -> dict[str, Any]:
    """把案例压缩成解释用的参考视图（字段对齐 diagnosis-output-contract 的 similar_cases）。"""

    return {
        "case_id": case.get("case_id"),
        "fault_family": case.get("bearing", {}).get("fault_family"),
        "fault_type": case.get("diagnosis", {}).get("fault_type"),
        "trust_level": case.get("source", {}).get("trust_level"),
        "use_boundary": case.get("use_boundary"),
    }
