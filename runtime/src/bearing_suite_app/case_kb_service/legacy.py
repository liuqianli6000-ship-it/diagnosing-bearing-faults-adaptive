from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .retriever import HybridCaseRetriever
from .schemas import (
    AuxiliaryEvidenceProfile,
    CaseRecord,
    OperatingCondition,
    PrimaryEvidenceProfile,
    RetrievalQuery,
    normalize_family,
)


@dataclass(frozen=True)
class LegacyRetrievedCase:
    record: dict[str, Any]
    score: float
    similarity_basis: list[str]
    matched_evidence: list[str]
    score_components: dict[str, float]
    counter_evidence: list[str]
    use_boundary: list[str]


def retrieve_legacy_cases(records: list[dict[str, Any]], query: dict[str, Any], top_k: int) -> list[LegacyRetrievedCase]:
    case_records = [_record_to_case(item) for item in records]
    return retrieve_case_records(case_records, query, top_k)


def retrieve_case_records(case_records: list[CaseRecord], query: dict[str, Any], top_k: int) -> list[LegacyRetrievedCase]:
    result = HybridCaseRetriever(case_records).retrieve(_query_to_retrieval_query(query), top_k=top_k)
    by_id = {item.case_id: item for item in case_records}
    retrieved: list[LegacyRetrievedCase] = []
    for item in result.matched_cases:
        case_record = by_id.get(item.case_id)
        if case_record is None:
            continue
        retrieved.append(
            LegacyRetrievedCase(
                record=case_record.raw_record,
                score=item.score,
                similarity_basis=item.similarity_basis,
                matched_evidence=item.matched_evidence,
                score_components=item.score_components,
                counter_evidence=item.counter_evidence,
                use_boundary=item.use_boundary,
            )
        )
    return retrieved


def record_to_case(record: dict[str, Any]) -> CaseRecord:
    return _record_to_case(record)


def _query_to_retrieval_query(query: dict[str, Any]) -> RetrievalQuery:
    family = _family(
        query.get("mfd_envelope_primary_family")
        or query.get("dominant_fault_family")
        or query.get("primary_family")
    )
    primary = PrimaryEvidenceProfile(
        mfd_envelope_primary_family=family,
        envelope_harmonic_count=max(0, int(_float(query.get("envelope_harmonic_count"), 0.0))),
        envelope_support_score=_optional_score(query.get("envelope_support_score") or query.get("candidate_support_score")),
        frequency_error_hz_by_family=_family_float_map(query.get("frequency_error_hz_by_family")),
        frequency_tolerance_hz_by_family=_family_float_map(query.get("frequency_tolerance_hz_by_family")),
        primary_harmonic_closeness=_optional_score(query.get("primary_harmonic_closeness")),
    )
    auxiliary = AuxiliaryEvidenceProfile(
        impulsiveness_score=_optional_score(query.get("impulsiveness_score")),
        raw_fft_family=_family(query.get("raw_fft_family")),
        order_domain_family=_family(query.get("order_domain_family")),
        order_domain_support_score=_optional_score(query.get("order_domain_support_score")),
    )
    return RetrievalQuery(
        bearing_type=str(query.get("bearing_type") or "generic"),
        rpm=_float(query.get("rpm"), 0.0),
        primary_evidence=primary,
        auxiliary_evidence=auxiliary,
    )


def _record_to_case(record: dict[str, Any]) -> CaseRecord:
    diagnosis = _dict(record.get("diagnosis"))
    bearing = _dict(record.get("bearing"))
    evidence = _dict(record.get("signal_evidence"))
    envelope = _dict(evidence.get("envelope_spectrum"))
    source = _dict(record.get("source"))
    profile = _dict(record.get("retrieval_profile"))
    family = _family(
        diagnosis.get("fault_family")
        or record.get("dominant_fault_family")
        or bearing.get("fault_family")
        or envelope.get("primary_family")
    )
    primary = PrimaryEvidenceProfile(
        mfd_envelope_primary_family=_family(envelope.get("primary_family") or family),
        envelope_harmonic_count=max(0, int(_float(envelope.get("minimum_harmonics"), 0.0))),
        envelope_support_score=_profile_level_score(profile.get("envelope_decisiveness")),
    )
    auxiliary = AuxiliaryEvidenceProfile(
        impulsiveness_score=_profile_level_score(profile.get("impulsiveness")),
    )
    return CaseRecord(
        case_id=str(record.get("case_id") or record.get("_line_number") or "case"),
        case_type=str(record.get("case_type") or "expert_reference"),
        bearing_type=str(bearing.get("bearing_type") or record.get("bearing_type") or "generic"),
        fault_family=family,
        fault_location=str(diagnosis.get("fault_location") or _family_to_location(family) or ""),
        fault_type=str(diagnosis.get("fault_type") or diagnosis.get("fault_type_text") or ""),
        operating_condition=_operating_condition(record),
        primary_evidence=primary,
        auxiliary_evidence=auxiliary,
        trust_level=str(source.get("trust_level") or "unknown"),
        counter_evidence=_string_list(record.get("counter_evidence")),
        use_boundary=_string_list(record.get("use_boundary")),
        raw_record=record,
    )


def _operating_condition(record: dict[str, Any]) -> OperatingCondition:
    operating = _dict(record.get("operating_condition"))
    rpm_range = operating.get("rpm_range")
    if isinstance(rpm_range, list) and len(rpm_range) == 2:
        return OperatingCondition(
            rpm_min=_float(rpm_range[0], 0.0),
            rpm_max=_float(rpm_range[1], 0.0),
            load=str(operating.get("load") or "") or None,
        )
    rpm = _float(record.get("rpm"), 0.0)
    return OperatingCondition(rpm_min=rpm or None, rpm_max=rpm or None, load=str(operating.get("load") or "") or None)


def _dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _family(value: Any) -> str | None:
    return normalize_family(value)


def _family_to_location(family: str | None) -> str | None:
    return {
        "BPFI": "inner_race",
        "BPFO": "outer_race",
        "BSF": "rolling_element",
        "NONE": "none",
    }.get(family or "")


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    return max(0.0, min(1.0, _float(value, 0.0)))


def _family_float_map(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        family = _family(key)
        if family:
            result[family] = max(0.0, _float(item, 0.0))
    return result


def _band(value: Any) -> tuple[float, float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = _float(value[0], 0.0)
    high = _float(value[1], 0.0)
    if low <= 0 and high <= 0:
        return None
    return min(low, high), max(low, high)


def _profile_level_score(value: Any) -> float | None:
    text = str(value or "").lower()
    return {
        "strong": 0.9,
        "moderate": 0.65,
        "medium": 0.55,
        "weak": 0.25,
        "weak_or_ambiguous": 0.25,
        "low": 0.2,
        "none": 0.05,
        "high": 0.85,
    }.get(text)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]
