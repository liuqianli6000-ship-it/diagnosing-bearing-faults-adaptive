from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .schemas import CaseRecord


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]:
        ...


@dataclass(frozen=True)
class CaseTextDocument:
    text_id: str
    case_id: str
    segment_type: str
    text: str
    auxiliary_only: bool = True
    metadata: dict[str, object] = field(default_factory=dict)


def _first_env(names: Sequence[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        model: str | None = None,
        dimensions: int = 1536,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_envs: Sequence[str] = ("OPENAI_API_KEY",),
        base_url_envs: Sequence[str] = ("OPENAI_BASE_URL",),
        model_envs: Sequence[str] = ("OPENAI_EMBEDDING_MODEL",),
        default_model: str = "text-embedding-3-small",
    ) -> None:
        self.api_key_envs = tuple(api_key_envs)
        self.base_url_envs = tuple(base_url_envs)
        self.model_envs = tuple(model_envs)
        self.api_key = api_key or _first_env(self.api_key_envs)
        self.base_url = base_url or _first_env(self.base_url_envs)
        self.model = model or _first_env(self.model_envs) or default_model
        self.dimensions = int(dimensions)
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive.")

    def embed(self, text: str) -> list[float]:
        if not text:
            raise ValueError("Cannot embed empty text.")
        from openai import OpenAI

        if not self.api_key:
            env_names = ", ".join(self.api_key_envs)
            raise RuntimeError(f"Embedding API key is not configured. Set one of: {env_names}.")
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(model=self.model, input=text, dimensions=self.dimensions)
        return [float(value) for value in response.data[0].embedding]


class QwenEmbeddingProvider(OpenAIEmbeddingProvider):
    def __init__(
        self,
        model: str | None = None,
        dimensions: int = 1536,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            api_key=api_key,
            base_url=base_url,
            api_key_envs=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
            base_url_envs=(
                "DASHSCOPE_EMBEDDING_BASE_URL",
                "QWEN_EMBEDDING_BASE_URL",
                "DASHSCOPE_BASE_URL",
                "QWEN_BASE_URL",
            ),
            model_envs=("DASHSCOPE_EMBEDDING_MODEL", "QWEN_EMBEDDING_MODEL"),
            default_model="text-embedding-v4",
        )


def build_case_text_documents(case: CaseRecord) -> list[CaseTextDocument]:
    documents: list[CaseTextDocument] = []

    def add(segment_type: str, text: str, index: int = 0, metadata: dict[str, object] | None = None) -> None:
        cleaned = " ".join(str(text).split())
        if not cleaned:
            return
        documents.append(
            CaseTextDocument(
                text_id=f"{case.case_id}:{segment_type}:{index}",
                case_id=case.case_id,
                segment_type=segment_type,
                text=cleaned,
                auxiliary_only=True,
                metadata=metadata or {},
            )
        )

    primary = case.primary_evidence
    add(
        "structured_summary",
        (
            f"case {case.case_id}; bearing {case.bearing_type}; fault family {case.fault_family}; "
            "MFD envelope family "
            f"{primary.mfd_envelope_primary_family}; envelope harmonics {primary.envelope_harmonic_count}; "
            f"envelope support {primary.envelope_support_score}."
        ),
        metadata={"auxiliary_role": "retrieval_text_summary"},
    )
    add(
        "diagnosis",
        f"diagnosis {case.fault_type}; family {case.fault_family}.",
        metadata={"fault_family": case.fault_family or ""},
    )
    for index, item in enumerate(case.counter_evidence):
        add("counter_evidence", item, index=index, metadata={"fault_family": case.fault_family or ""})
    for index, item in enumerate(case.use_boundary):
        add("use_boundary", item, index=index, metadata={"fault_family": case.fault_family or ""})
    return documents


def pgvector_literal(values: Sequence[float], expected_dimension: int | None = None) -> str:
    vector = [float(value) for value in values]
    if expected_dimension is not None and len(vector) != expected_dimension:
        raise ValueError(f"pgvector literal expected {expected_dimension} dimensions, got {len(vector)}.")
    if not vector:
        raise ValueError("pgvector literal cannot be empty.")
    for value in vector:
        if not math.isfinite(value):
            raise ValueError("pgvector literal values must be finite.")
    return "[" + ",".join(f"{value:.10f}" for value in vector) + "]"
