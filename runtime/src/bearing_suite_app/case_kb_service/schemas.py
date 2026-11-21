from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


PRIMARY_EVIDENCE_FOCUS = "mfd_envelope"
FAULT_FAMILIES = {"BPFI", "BPFO", "BSF", "NONE"}


def normalize_family(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return text if text in FAULT_FAMILIES else None


def clamp01(value: float | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(0.0, min(1.0, number))


def finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


class OperatingCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rpm_min: float | None = None
    rpm_max: float | None = None
    load: str | None = None

    @model_validator(mode="after")
    def normalize_range(self) -> "OperatingCondition":
        if self.rpm_min is not None and self.rpm_max is not None and self.rpm_max < self.rpm_min:
            self.rpm_min, self.rpm_max = self.rpm_max, self.rpm_min
        return self

    def rpm_similarity(self, rpm: float | None) -> float:
        if rpm is None or rpm <= 0 or self.rpm_min is None or self.rpm_max is None:
            return 0.0
        if self.rpm_min <= rpm <= self.rpm_max:
            return 1.0
        center = (self.rpm_min + self.rpm_max) / 2.0
        return max(0.0, 1.0 - abs(rpm - center) / max(abs(rpm), abs(center), 1.0))


class PrimaryEvidenceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mfd_envelope_primary_family: str | None = None
    envelope_harmonic_count: int = Field(default=0, ge=0)
    envelope_support_score: float | None = Field(default=None, ge=0.0, le=1.0)
    frequency_error_hz_by_family: dict[str, float] = Field(default_factory=dict)
    frequency_tolerance_hz_by_family: dict[str, float] = Field(default_factory=dict)
    primary_harmonic_closeness: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("mfd_envelope_primary_family")
    @classmethod
    def normalize_primary_family(cls, value: Any) -> str | None:
        return normalize_family(value)

    @field_validator("frequency_error_hz_by_family", "frequency_tolerance_hz_by_family")
    @classmethod
    def normalize_family_maps(cls, value: dict[str, float]) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for key, item in value.items():
            family = normalize_family(key)
            number = finite_float(item)
            if family and number is not None:
                normalized[family] = max(0.0, number)
        return normalized

class AuxiliaryEvidenceProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    impulsiveness_score: float | None = Field(default=None, ge=0.0, le=1.0)
    raw_fft_family: str | None = None
    order_domain_family: str | None = None
    order_domain_support_score: float | None = Field(default=None, ge=0.0, le=1.0)
    psd_profile_version: str | None = None
    psd_sampling_rate_hz: float | None = Field(default=None, gt=0.0)
    psd_log_band_power_ratio_db: list[float] = Field(default_factory=list)

    @field_validator("raw_fft_family", "order_domain_family")
    @classmethod
    def normalize_aux_family(cls, value: Any) -> str | None:
        return normalize_family(value)

    @field_validator("psd_log_band_power_ratio_db")
    @classmethod
    def validate_psd_profile(cls, value: list[float]) -> list[float]:
        profile: list[float] = []
        for item in value:
            number = finite_float(item)
            if number is None:
                raise ValueError("PSD profile values must be finite")
            profile.append(number)
        return profile


class CaseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    case_type: str = "expert_reference"
    bearing_type: str = "generic"
    fault_family: str | None = None
    fault_location: str | None = None
    fault_type: str | None = None
    operating_condition: OperatingCondition = Field(default_factory=OperatingCondition)
    primary_evidence: PrimaryEvidenceProfile = Field(default_factory=PrimaryEvidenceProfile)
    auxiliary_evidence: AuxiliaryEvidenceProfile = Field(default_factory=AuxiliaryEvidenceProfile)
    trust_level: str = "unknown"
    counter_evidence: list[str] = Field(default_factory=list)
    use_boundary: list[str] = Field(default_factory=list)
    raw_record: dict[str, Any] = Field(default_factory=dict)

    @field_validator("fault_family")
    @classmethod
    def normalize_fault_family(cls, value: Any) -> str | None:
        return normalize_family(value)


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bearing_type: str = "generic"
    rpm: float | None = None
    primary_evidence: PrimaryEvidenceProfile = Field(default_factory=PrimaryEvidenceProfile)
    auxiliary_evidence: AuxiliaryEvidenceProfile = Field(default_factory=AuxiliaryEvidenceProfile)


class RetrievedCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    case_type: str
    bearing_type: str
    fault_family: str | None
    fault_type: str | None
    score: float
    score_components: dict[str, float]
    similarity_basis: list[str]
    matched_evidence: list[str]
    counter_evidence: list[str]
    use_boundary: list[str]


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_evidence_focus: str = PRIMARY_EVIDENCE_FOCUS
    score_weights: dict[str, float]
    matched_cases: list[RetrievedCase]
