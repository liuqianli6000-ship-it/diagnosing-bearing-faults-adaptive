from __future__ import annotations

import math
from collections.abc import Iterable

from .schemas import (
    PRIMARY_EVIDENCE_FOCUS,
    AuxiliaryEvidenceProfile,
    CaseRecord,
    PrimaryEvidenceProfile,
    RetrievedCase,
    RetrievalQuery,
    RetrievalResult,
    clamp01,
    FAULT_FAMILIES,
)


SCORE_WEIGHTS: dict[str, float] = {
    "mfd_envelope_family_match": 0.34,
    "harmonic_pattern_similarity": 0.20,
    "frequency_reference_error_similarity": 0.15,
    "operating_condition_similarity": 0.08,
    "psd_profile_similarity": 0.12,
    "auxiliary_evidence_similarity": 0.06,
    "trust_level_score": 0.05,
}


class HybridCaseRetriever:
    def __init__(self, cases: Iterable[CaseRecord]) -> None:
        self._cases = [case for case in cases if _case_family(case) in FAULT_FAMILIES]

    def retrieve(self, query: RetrievalQuery, top_k: int = 3) -> RetrievalResult:
        top_k = max(1, int(top_k))
        ranked = [self._score_case(query, case) for case in self._cases]
        ranked.sort(key=lambda item: (item.score, item.case_type == "negative_control"), reverse=True)
        return RetrievalResult(
            primary_evidence_focus=PRIMARY_EVIDENCE_FOCUS,
            score_weights=dict(SCORE_WEIGHTS),
            matched_cases=[item for item in ranked[:top_k] if item.score > 0.0],
        )

    def _score_case(self, query: RetrievalQuery, case: CaseRecord) -> RetrievedCase:
        components = {
            "mfd_envelope_family_match": _family_match(query.primary_evidence, case),
            "harmonic_pattern_similarity": _harmonic_similarity(query.primary_evidence, case.primary_evidence),
            "frequency_reference_error_similarity": _frequency_error_similarity(query.primary_evidence, case),
            "operating_condition_similarity": _operating_similarity(query, case),
            "psd_profile_similarity": _psd_similarity(
                query.auxiliary_evidence, case.auxiliary_evidence
            ),
            "auxiliary_evidence_similarity": _auxiliary_similarity(query.auxiliary_evidence, case.auxiliary_evidence),
            "trust_level_score": _trust_score(case.trust_level),
        }
        score = sum(SCORE_WEIGHTS[name] * value for name, value in components.items())
        basis = _basis(query, case, components)
        matched = _matched_evidence(query, case, components)
        return RetrievedCase(
            case_id=case.case_id,
            case_type=case.case_type,
            bearing_type=case.bearing_type,
            fault_family=_case_family(case),
            fault_type=case.fault_type,
            score=round(clamp01(score), 6),
            score_components={key: round(value, 6) for key, value in components.items()},
            similarity_basis=basis,
            matched_evidence=matched,
            counter_evidence=list(case.counter_evidence),
            use_boundary=list(case.use_boundary),
        )


def _case_family(case: CaseRecord) -> str | None:
    return case.fault_family or case.primary_evidence.mfd_envelope_primary_family


def _family_match(query_primary: PrimaryEvidenceProfile, case: CaseRecord) -> float:
    query_family = query_primary.mfd_envelope_primary_family
    case_family = _case_family(case)
    if not query_family or not case_family:
        return 0.0
    if query_family == case_family:
        return 1.0
    return 0.0


def _harmonic_similarity(query_primary: PrimaryEvidenceProfile, case_primary: PrimaryEvidenceProfile) -> float:
    query_family = query_primary.mfd_envelope_primary_family
    case_family = case_primary.mfd_envelope_primary_family
    if query_family != case_family:
        return 0.0
    if query_family == "NONE":
        return 1.0 if query_primary.envelope_harmonic_count == case_primary.envelope_harmonic_count else 0.5
    query_count = max(0, query_primary.envelope_harmonic_count)
    case_count = max(0, case_primary.envelope_harmonic_count)
    if query_count == 0 and case_count == 0:
        return 0.5
    if query_count == 0 or case_count == 0:
        return 0.0
    return clamp01(1.0 - abs(query_count - case_count) / max(query_count, case_count, 1))


def _frequency_error_similarity(query_primary: PrimaryEvidenceProfile, case: CaseRecord) -> float:
    query_family = query_primary.mfd_envelope_primary_family
    if not query_family or _case_family(case) != query_family:
        return 0.0
    error = query_primary.frequency_error_hz_by_family.get(query_family)
    tolerance = query_primary.frequency_tolerance_hz_by_family.get(query_family)
    if error is not None and tolerance is not None and tolerance > 0:
        return clamp01(1.0 - error / tolerance)
    if query_primary.primary_harmonic_closeness is not None:
        return clamp01(query_primary.primary_harmonic_closeness)
    return 0.5


def _operating_similarity(query: RetrievalQuery, case: CaseRecord) -> float:
    query_bearing = (query.bearing_type or "").lower()
    case_bearing = (case.bearing_type or "").lower()
    if query_bearing and case_bearing == query_bearing:
        bearing_score = 1.0
    elif case_bearing == "generic":
        bearing_score = 0.8
    else:
        bearing_score = 0.0
    rpm_score = case.operating_condition.rpm_similarity(query.rpm)
    return clamp01(0.65 * rpm_score + 0.35 * bearing_score)


def _auxiliary_similarity(query_aux: AuxiliaryEvidenceProfile, case_aux: AuxiliaryEvidenceProfile) -> float:
    components: list[float] = []
    if query_aux.impulsiveness_score is not None and case_aux.impulsiveness_score is not None:
        components.append(clamp01(1.0 - abs(query_aux.impulsiveness_score - case_aux.impulsiveness_score)))
    if query_aux.order_domain_family and case_aux.order_domain_family:
        components.append(1.0 if query_aux.order_domain_family == case_aux.order_domain_family else 0.0)
    if query_aux.raw_fft_family and case_aux.raw_fft_family:
        components.append(1.0 if query_aux.raw_fft_family == case_aux.raw_fft_family else 0.0)
    if not components:
        return 0.0
    return clamp01(sum(components) / len(components))


def _psd_similarity(
    query_aux: AuxiliaryEvidenceProfile, case_aux: AuxiliaryEvidenceProfile
) -> float:
    """PSD 画像相似度，只在同版本、同维度时计算。"""

    query_profile = query_aux.psd_log_band_power_ratio_db
    case_profile = case_aux.psd_log_band_power_ratio_db
    if (
        not query_profile
        or len(query_profile) != len(case_profile)
        or query_aux.psd_profile_version != case_aux.psd_profile_version
        or query_aux.psd_sampling_rate_hz is None
        or case_aux.psd_sampling_rate_hz is None
        or abs(query_aux.psd_sampling_rate_hz - case_aux.psd_sampling_rate_hz)
        / query_aux.psd_sampling_rate_hz
        > 0.01
    ):
        return 0.0
    squared_error = sum(
        (float(query_value) - float(case_value)) ** 2
        for query_value, case_value in zip(query_profile, case_profile)
    ) / len(query_profile)
    rmse_db = math.sqrt(squared_error)
    return clamp01(math.exp(-rmse_db / 6.0))


def _trust_score(trust_level: str) -> float:
    return {
        "expert_curated": 1.0,
        "verified_dataset": 0.95,
        "field_verified": 0.9,
        "synthetic": 0.6,
        "unknown": 0.5,
    }.get((trust_level or "unknown").lower(), 0.5)


def _basis(query: RetrievalQuery, case: CaseRecord, components: dict[str, float]) -> list[str]:
    family = query.primary_evidence.mfd_envelope_primary_family
    basis: list[str] = []
    if components["mfd_envelope_family_match"] > 0:
        basis.append(f"MFD envelope family match: {family}")
    if components["harmonic_pattern_similarity"] > 0:
        basis.append(f"MFD envelope harmonic pattern similarity: {components['harmonic_pattern_similarity']:.3f}")
    if components["frequency_reference_error_similarity"] > 0:
        basis.append(
            "theoretical fault-frequency error similarity: "
            f"{components['frequency_reference_error_similarity']:.3f}"
        )
    if components["operating_condition_similarity"] > 0:
        basis.append(f"operating condition similarity: {components['operating_condition_similarity']:.3f}")
    if components["psd_profile_similarity"] > 0:
        basis.append(
            f"Welch PSD profile similarity: {components['psd_profile_similarity']:.3f}"
        )
    if components["auxiliary_evidence_similarity"] > 0:
        basis.append(f"auxiliary evidence similarity: {components['auxiliary_evidence_similarity']:.3f}")
    return basis


def _matched_evidence(query: RetrievalQuery, case: CaseRecord, components: dict[str, float]) -> list[str]:
    family = query.primary_evidence.mfd_envelope_primary_family
    matched: list[str] = []
    if components["mfd_envelope_family_match"] > 0:
        matched.append(
            f"Current MFD envelope spectrum primary family {family} matches case family {_case_family(case)}."
        )
    if query.primary_evidence.envelope_harmonic_count > 0 and components["harmonic_pattern_similarity"] > 0:
        matched.append(
            "Current MFD envelope spectrum contains "
            f"{query.primary_evidence.envelope_harmonic_count} matched harmonics."
        )
    if components["psd_profile_similarity"] > 0:
        matched.append(
            "Current Welch PSD band-energy profile is comparable with the case profile."
        )
    return matched
