"""多模态包络谱观察器：只描述图像，并与数值证据交叉校验。"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .diagnosis_schemas import EvidenceBundle, VisualObservation
from .prompts import VISUAL_OBSERVER_SYSTEM


def _parse_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    value = text.strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(value[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MultimodalEvidenceObserver:
    """使用支持图像输入的模型生成结构化观察，不能修改数值测量。"""

    def __init__(self, llm: Any, *, max_image_bytes: int = 8 * 1024 * 1024) -> None:
        self.llm = llm
        self.max_image_bytes = max(1024, int(max_image_bytes))

    def observe(
        self,
        artifact: dict[str, Any],
        evidence: EvidenceBundle,
    ) -> dict[str, Any]:
        if not getattr(self.llm, "supports_image_input", False):
            return {
                "artifact_id": str(artifact.get("evidence_id", "")),
                "image_sha256": str(artifact.get("sha256", "")),
                "observable": False,
                "observations": [],
                "conflicts": [],
                "limitations": ["provider_image_input_not_enabled"],
            }
        path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise ValueError("可视化证据文件不存在")
        if path.stat().st_size > self.max_image_bytes:
            raise ValueError("可视化证据超过多模态输入大小上限")
        digest = _file_sha256(path)
        expected_digest = str(artifact.get("sha256", ""))
        if not expected_digest or digest != expected_digest:
            raise ValueError("可视化证据 SHA-256 不匹配")

        mime_type = str(artifact.get("mime_type") or "image/png")
        data_url = (
            f"data:{mime_type};base64,"
            + base64.b64encode(path.read_bytes()).decode("ascii")
        )
        numeric_context = {
            "artifact_id": artifact.get("evidence_id"),
            "image_sha256": digest,
            "fault_frequencies_hz": evidence.fault_frequencies_hz,
            "assessments": evidence.to_llm_payload().get("assessments", {}),
            "band_summaries": evidence.to_llm_payload().get("band_summaries", []),
            "artifact_processing": artifact.get("processing", {}),
        }
        messages = [
            {
                "role": "system",
                "content": VISUAL_OBSERVER_SYSTEM,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "请观察多频带包络谱。输出 JSON Schema：\n"
                            + json.dumps(
                                VisualObservation.model_json_schema(),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n数值上下文：\n"
                            + json.dumps(
                                numeric_context,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ]
        response_format = None
        if getattr(self.llm, "supports_strict_output", False):
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "bearing_visual_observation",
                    "strict": True,
                    "schema": VisualObservation.model_json_schema(),
                },
            }
        try:
            message = self.llm.complete_messages(
                messages,
                response_format=response_format,
            )
        except Exception:
            if response_format is None:
                raise
            message = self.llm.complete_messages(messages)
        raw = str(getattr(message, "content", None) or "")
        payload = _parse_object(raw)
        if payload is None:
            raise ValueError("多模态观察器未返回合法JSON")
        observation = VisualObservation.model_validate(payload)
        if observation.artifact_id != artifact.get("evidence_id"):
            raise ValueError("多模态观察引用了错误的artifact_id")
        if observation.image_sha256 != digest:
            raise ValueError("多模态观察返回的image_sha256不匹配")

        allowed = evidence.evidence_ids()
        for finding in observation.observations:
            missing = [
                ref for ref in finding.numeric_evidence_refs if ref not in allowed
            ]
            if missing:
                raise ValueError(f"多模态观察引用不存在的数值证据: {missing}")
            if finding.family in {"BPFO", "BPFI", "BSF"} and not any(
                ref == f"assessment.{finding.family}"
                or f".{finding.family}" in ref
                for ref in finding.numeric_evidence_refs
            ):
                raise ValueError(
                    f"多模态{finding.family}观察未引用对应频族数值证据"
                )
        return observation.model_dump(mode="json")
