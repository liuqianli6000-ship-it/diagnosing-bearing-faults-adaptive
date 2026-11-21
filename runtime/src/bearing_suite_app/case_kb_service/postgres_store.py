from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .db_schema import DEFAULT_VECTOR_DIMENSION, schema_sql
from .embedding_store import (
    EmbeddingProvider,
    QwenEmbeddingProvider,
    build_case_text_documents,
    pgvector_literal,
)
from .schemas import CaseRecord


@dataclass(frozen=True)
class SqlStatement:
    sql: str
    params: dict[str, Any]


class PostgresCaseStore:
    def __init__(
        self,
        dsn: str,
        embedding_provider: EmbeddingProvider | None = None,
        vector_dimension: int = DEFAULT_VECTOR_DIMENSION,
    ) -> None:
        if not dsn:
            raise ValueError("dsn is required.")
        self.dsn = dsn
        self.vector_dimension = int(vector_dimension)
        if self.vector_dimension <= 0:
            raise ValueError("vector_dimension must be positive.")
        self.embedding_provider = embedding_provider or QwenEmbeddingProvider(dimensions=self.vector_dimension)

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cursor:
                cursor.execute(self.build_pgvector_available_query())
                if cursor.fetchone() is None:
                    raise RuntimeError(
                        "PostgreSQL is reachable, but pgvector is not installed on the server. "
                        "Install the pgvector extension for this PostgreSQL instance before importing case embeddings."
                    )
                cursor.execute(schema_sql(self.vector_dimension))
            conn.commit()

    def build_pgvector_available_query(self) -> str:
        return "SELECT name FROM pg_available_extensions WHERE name = 'vector'"

    def build_upsert_statements(self, case: CaseRecord) -> list[SqlStatement]:
        primary = case.primary_evidence
        auxiliary = case.auxiliary_evidence
        operating_condition = {
            "rpm_min": case.operating_condition.rpm_min,
            "rpm_max": case.operating_condition.rpm_max,
            "load": case.operating_condition.load,
        }
        statements = [
            SqlStatement(
                sql="""
                INSERT INTO kb_cases (
                    case_id, case_type, bearing_type, fault_family, fault_location, fault_type,
                    trust_level, operating_condition, counter_evidence, use_boundary, raw_record, updated_at
                )
                VALUES (
                    %(case_id)s, %(case_type)s, %(bearing_type)s, %(fault_family)s, %(fault_location)s, %(fault_type)s,
                    %(trust_level)s, %(operating_condition)s::jsonb, %(counter_evidence)s::jsonb,
                    %(use_boundary)s::jsonb, %(raw_record)s::jsonb, now()
                )
                ON CONFLICT (case_id) DO UPDATE SET
                    case_type = EXCLUDED.case_type,
                    bearing_type = EXCLUDED.bearing_type,
                    fault_family = EXCLUDED.fault_family,
                    fault_location = EXCLUDED.fault_location,
                    fault_type = EXCLUDED.fault_type,
                    trust_level = EXCLUDED.trust_level,
                    operating_condition = EXCLUDED.operating_condition,
                    counter_evidence = EXCLUDED.counter_evidence,
                    use_boundary = EXCLUDED.use_boundary,
                    raw_record = EXCLUDED.raw_record,
                    updated_at = now()
                """,
                params={
                    "case_id": case.case_id,
                    "case_type": case.case_type,
                    "bearing_type": case.bearing_type,
                    "fault_family": case.fault_family,
                    "fault_location": case.fault_location,
                    "fault_type": case.fault_type,
                    "trust_level": case.trust_level,
                    "operating_condition": _json(operating_condition),
                    "counter_evidence": _json(case.counter_evidence),
                    "use_boundary": _json(case.use_boundary),
                    "raw_record": _json(case.raw_record),
                },
            ),
            SqlStatement(
                sql="""
                INSERT INTO kb_primary_features (
                    case_id, mfd_envelope_primary_family, envelope_harmonic_count,
                    envelope_support_score,
                    frequency_error_hz_by_family, frequency_tolerance_hz_by_family,
                    primary_harmonic_closeness, updated_at
                )
                VALUES (
                    %(case_id)s, %(mfd_envelope_primary_family)s, %(envelope_harmonic_count)s,
                    %(envelope_support_score)s,
                    %(frequency_error_hz_by_family)s::jsonb,
                    %(frequency_tolerance_hz_by_family)s::jsonb,
                    %(primary_harmonic_closeness)s,
                    now()
                )
                ON CONFLICT (case_id) DO UPDATE SET
                    mfd_envelope_primary_family = EXCLUDED.mfd_envelope_primary_family,
                    envelope_harmonic_count = EXCLUDED.envelope_harmonic_count,
                    envelope_support_score = EXCLUDED.envelope_support_score,
                    frequency_error_hz_by_family = EXCLUDED.frequency_error_hz_by_family,
                    frequency_tolerance_hz_by_family = EXCLUDED.frequency_tolerance_hz_by_family,
                    primary_harmonic_closeness = EXCLUDED.primary_harmonic_closeness,
                    updated_at = now()
                """,
                params={
                    "case_id": case.case_id,
                    "mfd_envelope_primary_family": primary.mfd_envelope_primary_family,
                    "envelope_harmonic_count": primary.envelope_harmonic_count,
                    "envelope_support_score": primary.envelope_support_score,
                    "frequency_error_hz_by_family": _json(primary.frequency_error_hz_by_family),
                    "frequency_tolerance_hz_by_family": _json(primary.frequency_tolerance_hz_by_family),
                    "primary_harmonic_closeness": primary.primary_harmonic_closeness,
                },
            ),
            SqlStatement(
                sql="""
                INSERT INTO kb_auxiliary_features (
                    case_id, impulsiveness_score, raw_fft_family, order_domain_family,
                    order_domain_support_score, updated_at
                )
                VALUES (
                    %(case_id)s, %(impulsiveness_score)s, %(raw_fft_family)s,
                    %(order_domain_family)s, %(order_domain_support_score)s, now()
                )
                ON CONFLICT (case_id) DO UPDATE SET
                    impulsiveness_score = EXCLUDED.impulsiveness_score,
                    raw_fft_family = EXCLUDED.raw_fft_family,
                    order_domain_family = EXCLUDED.order_domain_family,
                    order_domain_support_score = EXCLUDED.order_domain_support_score,
                    updated_at = now()
                """,
                params={
                    "case_id": case.case_id,
                    "impulsiveness_score": auxiliary.impulsiveness_score,
                    "raw_fft_family": auxiliary.raw_fft_family,
                    "order_domain_family": auxiliary.order_domain_family,
                    "order_domain_support_score": auxiliary.order_domain_support_score,
                },
            ),
            SqlStatement(
                sql="DELETE FROM kb_case_text WHERE case_id = %(case_id)s",
                params={"case_id": case.case_id},
            ),
        ]
        for document in build_case_text_documents(case):
            embedding = self.embedding_provider.embed(document.text)
            statements.append(
                SqlStatement(
                    sql="""
                    INSERT INTO kb_case_text (
                        text_id, case_id, segment_type, text, auxiliary_only, metadata, embedding, updated_at
                    )
                    VALUES (
                        %(text_id)s, %(case_id)s, %(segment_type)s, %(text)s,
                        %(auxiliary_only)s, %(metadata)s::jsonb, %(embedding)s::vector, now()
                    )
                    ON CONFLICT (text_id) DO UPDATE SET
                        segment_type = EXCLUDED.segment_type,
                        text = EXCLUDED.text,
                        auxiliary_only = EXCLUDED.auxiliary_only,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = now()
                    """,
                    params={
                        "text_id": document.text_id,
                        "case_id": document.case_id,
                        "segment_type": document.segment_type,
                        "text": document.text,
                        "auxiliary_only": document.auxiliary_only,
                        "metadata": _json(document.metadata),
                        "embedding": pgvector_literal(embedding, self.vector_dimension),
                    },
                )
            )
        return statements

    def upsert_case(self, case: CaseRecord) -> None:
        statements = self.build_upsert_statements(case)
        with self._connect() as conn:
            with conn.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement.sql, statement.params)
            conn.commit()

    def build_text_search_query(self, query_text: str, top_k: int = 5) -> tuple[str, dict[str, Any]]:
        if not query_text:
            raise ValueError("query_text is required.")
        query_embedding = self.embedding_provider.embed(query_text)
        sql = """
            SELECT
                text_id,
                case_id,
                segment_type,
                text,
                auxiliary_only,
                metadata,
                1.0 - (embedding <=> %(query_embedding)s::vector) AS similarity
            FROM kb_case_text
            ORDER BY embedding <=> %(query_embedding)s::vector
            LIMIT %(top_k)s
        """
        return sql, {
            "query_embedding": pgvector_literal(query_embedding, self.vector_dimension),
            "top_k": max(1, int(top_k)),
        }

    def search_case_text(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        sql, params = self.build_text_search_query(query_text, top_k)
        with self._connect(row_factory=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]

    def build_load_cases_query(self, limit: int = 1000) -> tuple[str, dict[str, Any]]:
        sql = """
            SELECT
                c.case_id,
                c.case_type,
                c.bearing_type,
                c.fault_family,
                c.fault_location,
                c.fault_type,
                c.trust_level,
                c.operating_condition,
                c.counter_evidence,
                c.use_boundary,
                c.raw_record,
                p.mfd_envelope_primary_family,
                p.envelope_harmonic_count,
                p.envelope_support_score,
                p.frequency_error_hz_by_family,
                p.frequency_tolerance_hz_by_family,
                p.primary_harmonic_closeness,
                a.impulsiveness_score,
                a.raw_fft_family,
                a.order_domain_family,
                a.order_domain_support_score
            FROM kb_cases c
            LEFT JOIN kb_primary_features p ON p.case_id = c.case_id
            LEFT JOIN kb_auxiliary_features a ON a.case_id = c.case_id
            ORDER BY c.case_id
            LIMIT %(limit)s
        """
        return sql, {"limit": max(1, int(limit))}

    def load_cases(self, limit: int = 1000) -> list[CaseRecord]:
        sql, params = self.build_load_cases_query(limit)
        with self._connect(row_factory=True) as conn:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                return [_row_to_case(dict(row)) for row in cursor.fetchall()]

    def _connect(self, row_factory: bool = False):
        import psycopg

        if row_factory:
            from psycopg.rows import dict_row

            return psycopg.connect(self.dsn, row_factory=dict_row)
        return psycopg.connect(self.dsn)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_value(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _row_to_case(row: dict[str, Any]) -> CaseRecord:
    from .schemas import AuxiliaryEvidenceProfile, OperatingCondition, PrimaryEvidenceProfile

    operating = _json_value(row.get("operating_condition"), {})
    operating = operating if isinstance(operating, dict) else {}
    counter_evidence = _json_value(row.get("counter_evidence"), [])
    use_boundary = _json_value(row.get("use_boundary"), [])
    raw_record = _json_value(row.get("raw_record"), {})
    psd_record = raw_record.get("psd_profile", {}) if isinstance(raw_record, dict) else {}
    psd_record = psd_record if isinstance(psd_record, dict) else {}
    primary = PrimaryEvidenceProfile(
        mfd_envelope_primary_family=row.get("mfd_envelope_primary_family") or row.get("fault_family"),
        envelope_harmonic_count=int(row.get("envelope_harmonic_count") or 0),
        envelope_support_score=row.get("envelope_support_score"),
        frequency_error_hz_by_family=_json_value(row.get("frequency_error_hz_by_family"), {}),
        frequency_tolerance_hz_by_family=_json_value(row.get("frequency_tolerance_hz_by_family"), {}),
        primary_harmonic_closeness=row.get("primary_harmonic_closeness"),
    )
    auxiliary = AuxiliaryEvidenceProfile(
        impulsiveness_score=row.get("impulsiveness_score"),
        raw_fft_family=row.get("raw_fft_family"),
        order_domain_family=row.get("order_domain_family"),
        order_domain_support_score=row.get("order_domain_support_score"),
        psd_profile_version=psd_record.get("profile_version"),
        psd_sampling_rate_hz=psd_record.get("sampling_rate_hz"),
        psd_log_band_power_ratio_db=psd_record.get(
            "log_band_power_ratio_db", []
        ),
    )
    if not isinstance(raw_record, dict) or not raw_record:
        raw_record = {
            "case_id": row.get("case_id"),
            "case_type": row.get("case_type"),
            "source": {"source_type": "postgres", "trust_level": row.get("trust_level")},
            "bearing": {"bearing_type": row.get("bearing_type"), "fault_family": row.get("fault_family")},
            "operating_condition": {
                "rpm_range": [operating.get("rpm_min"), operating.get("rpm_max")],
                "load": operating.get("load"),
            },
            "diagnosis": {
                "fault_type": row.get("fault_type"),
                "fault_location": row.get("fault_location"),
                "fault_family": row.get("fault_family"),
            },
            "signal_evidence": {
                "envelope_spectrum": {
                    "primary_family": primary.mfd_envelope_primary_family,
                    "minimum_harmonics": primary.envelope_harmonic_count,
                }
            },
            "counter_evidence": counter_evidence,
            "use_boundary": use_boundary,
        }
    return CaseRecord(
        case_id=str(row.get("case_id")),
        case_type=str(row.get("case_type") or "expert_reference"),
        bearing_type=str(row.get("bearing_type") or "generic"),
        fault_family=row.get("fault_family"),
        fault_location=row.get("fault_location"),
        fault_type=row.get("fault_type"),
        operating_condition=OperatingCondition(
            rpm_min=operating.get("rpm_min"),
            rpm_max=operating.get("rpm_max"),
            load=operating.get("load"),
        ),
        primary_evidence=primary,
        auxiliary_evidence=auxiliary,
        trust_level=str(row.get("trust_level") or "unknown"),
        counter_evidence=counter_evidence if isinstance(counter_evidence, list) else [],
        use_boundary=use_boundary if isinstance(use_boundary, list) else [],
        raw_record=raw_record,
    )
