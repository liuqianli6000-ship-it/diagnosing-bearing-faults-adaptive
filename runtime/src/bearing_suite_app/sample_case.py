"""样本案例库：把诊断后的样本（特征 + 诊断 + 解释）写入 PostgreSQL，并检索相似样本。

复用 ``case_kb_service`` 的 CaseRecord / HybridCaseRetriever / PostgresCaseStore。
新样本的特征向量与特征描述存进 ``raw_record``（jsonb），不修改既有表结构。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .case_kb_service import (
    AuxiliaryEvidenceProfile,
    CaseRecord,
    HybridCaseRetriever,
    OperatingCondition,
    PostgresCaseStore,
    PrimaryEvidenceProfile,
    RetrievedCase,
    RetrievalQuery,
)

DEFAULT_DSN = "host=localhost port=5432 dbname=case_kb user=case_kb password=case_kb"

FAMILIES = {"BPFO", "BPFI", "BSF", "NONE"}
FAMILY_TO_LOCATION = {
    "BPFO": "outer_race",
    "BPFI": "inner_race",
    "BSF": "rolling_element",
    "NONE": None,
}
PSD_MAD_SCALE_FLOOR_DB = 1.5
PSD_SIMILAR_DISTANCE_MAX = 1.5
PSD_DIFFERENT_DISTANCE_MIN = 3.0
PSD_ABNORMAL_BAND_Z_MIN = 3.0
PSD_MIN_ABNORMAL_BANDS = 2


def _normalize_family(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in FAMILIES else None


def _impulsiveness_score(crest: float) -> float:
    # 波峰因子粗略映射到 0~1 冲击性分数
    return round(max(0.0, min(1.0, float(crest) / 15.0)), 3)


def _psd_profile(features: Mapping[str, Any]) -> tuple[str | None, list[float], dict[str, Any]]:
    frequency = features.get("frequency", {})
    psd = frequency.get("psd", {}) if isinstance(frequency, Mapping) else {}
    if not isinstance(psd, Mapping):
        return None, [], {}
    version = str(psd.get("profile_version") or "") or None
    values = [float(value) for value in psd.get("log_band_power_ratio_db", [])]
    return version, values, dict(psd)


def build_sample_case(
    case_id: str,
    signal_ref: str,
    rpm: float,
    fs: float,
    geometry: Mapping[str, Any],
    features: Mapping[str, Any],
    fault_type: str,
    leading_family: str | None,
    confidence: float,
    explanation: str,
    *,
    source_group: str | None = None,
    measurement_point: str | None = None,
) -> CaseRecord:
    """把诊断后的样本构造成案例记录（特征向量 + 描述存入 raw_record）。"""

    family = _normalize_family(leading_family)
    env = features.get("envelope", {})
    families = env.get("families", {}) if isinstance(env.get("families"), dict) else {}
    if family and isinstance(families.get(family), dict):
        max_harmonics = len(families[family].get("qualified_orders", []))
    else:
        max_harmonics = max(
            (len(f.get("qualified_orders", [])) for f in families.values() if isinstance(f, dict)),
            default=0,
        )
    crest = float(features.get("time", {}).get("crest_factor", 0.0) or 0.0)
    psd_version, psd_profile, psd_record = _psd_profile(features)

    return CaseRecord(
        case_id=case_id,
        case_type="diagnosed_sample",
        bearing_type=str(geometry.get("type", "generic")) if isinstance(geometry, dict) else "generic",
        fault_family=family,
        fault_location=FAMILY_TO_LOCATION.get(family or ""),
        fault_type=fault_type,
        operating_condition=OperatingCondition(rpm_min=float(rpm), rpm_max=float(rpm)),
        primary_evidence=PrimaryEvidenceProfile(
            mfd_envelope_primary_family=family,
            envelope_harmonic_count=max_harmonics,
            envelope_support_score=round(float(confidence), 3),
        ),
        auxiliary_evidence=AuxiliaryEvidenceProfile(
            impulsiveness_score=_impulsiveness_score(crest),
            psd_profile_version=psd_version,
            psd_sampling_rate_hz=float(psd_record.get("sampling_rate_hz") or fs),
            psd_log_band_power_ratio_db=psd_profile,
        ),
        trust_level="verified_dataset",
        raw_record={
            "signal_ref": signal_ref,
            "source_group": source_group,
            "measurement_point": measurement_point,
            "rpm": float(rpm),
            "fs": float(fs),
            "geometry": dict(geometry) if isinstance(geometry, dict) else {},
            "feature_vector": list(features.get("feature_vector", [])),
            "feature_description": str(features.get("description", "")),
            "feature_schema_version": features.get("feature_schema_version"),
            "psd_profile": psd_record,
            "diagnosis": {
                "fault_type": fault_type,
                "confidence": float(confidence),
                "leading_family": family,
            },
            "explanation": explanation,
        },
    )


def query_from_features(
    bearing_type: str,
    rpm: float,
    features: Mapping[str, Any],
    leading_family: str | None,
) -> RetrievalQuery:
    """从新样本特征构造结构化检索查询。"""

    family = _normalize_family(leading_family)
    env = features.get("envelope", {})
    families = env.get("families", {}) if isinstance(env.get("families"), dict) else {}
    if family and isinstance(families.get(family), dict):
        max_harmonics = len(families[family].get("qualified_orders", []))
    else:
        max_harmonics = max(
            (len(f.get("qualified_orders", [])) for f in families.values() if isinstance(f, dict)),
            default=0,
        )
    crest = float(features.get("time", {}).get("crest_factor", 0.0) or 0.0)
    psd_version, psd_profile, psd_record = _psd_profile(features)

    return RetrievalQuery(
        bearing_type=bearing_type,
        rpm=float(rpm),
        primary_evidence=PrimaryEvidenceProfile(
            mfd_envelope_primary_family=family,
            envelope_harmonic_count=max_harmonics,
        ),
        auxiliary_evidence=AuxiliaryEvidenceProfile(
            impulsiveness_score=_impulsiveness_score(crest),
            psd_profile_version=psd_version,
            psd_sampling_rate_hz=(
                float(psd_record["sampling_rate_hz"])
                if psd_record.get("sampling_rate_hz")
                else None
            ),
            psd_log_band_power_ratio_db=psd_profile,
        ),
    )


def compare_psd_to_normal_cases(
    current_psd: Mapping[str, Any],
    cases: Sequence[CaseRecord],
    *,
    min_reference_cases: int = 3,
) -> dict[str, Any]:
    """用已验证正常案例的 PSD 分布评估当前样本是否偏离健康基线。

    该结果只表达正常性/异常性，不负责判断 BPFO/BPFI/BSF。使用中位数和
    MAD 抵抗少量异常正常样本，并用 1.5 dB 尺度下限避免零 MAD 过度放大。
    """

    version = str(current_psd.get("profile_version") or "")
    current = np.asarray(current_psd.get("log_band_power_ratio_db", []), dtype=float)
    band_rows = current_psd.get("band_power_ratios", [])
    if not version or current.size == 0 or not np.all(np.isfinite(current)):
        return {
            "evidence_id": "signal.psd_normality",
            "status": "unavailable",
            "reason": "当前样本缺少有效 PSD 画像",
            "reference_count": 0,
        }

    references: list[np.ndarray] = []
    reference_ids: list[str] = []
    reference_source_groups: set[str] = set()
    trusted_levels = {"verified_dataset", "field_verified"}
    for case in cases:
        family = _normalize_family(
            case.fault_family or case.primary_evidence.mfd_envelope_primary_family
        )
        profile = case.auxiliary_evidence.psd_log_band_power_ratio_db
        if (
            family != "NONE"
            or case.trust_level not in trusted_levels
            or case.auxiliary_evidence.psd_profile_version != version
            or len(profile) != current.size
        ):
            continue
        values = np.asarray(profile, dtype=float)
        if np.all(np.isfinite(values)):
            references.append(values)
            reference_ids.append(case.case_id)
            source_group = case.raw_record.get("source_group")
            if source_group:
                reference_source_groups.add(str(source_group))

    required = max(1, int(min_reference_cases))
    if len(references) < required:
        return {
            "evidence_id": "signal.psd_normality",
            "status": "unavailable",
            "reason": f"同工况已验证正常 PSD 案例不足（{len(references)}/{required}）",
            "reference_count": len(references),
        }

    matrix = np.vstack(references)
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0)
    robust_scale = np.maximum(1.4826 * mad, PSD_MAD_SCALE_FLOOR_DB)
    robust_z = np.abs(current - median) / robust_scale

    observable = np.ones(current.size, dtype=bool)
    if isinstance(band_rows, list) and len(band_rows) == current.size:
        observable = np.asarray(
            [bool(row.get("observable", False)) if isinstance(row, Mapping) else False for row in band_rows]
        )
    usable_z = robust_z[observable]
    robust_distance = float(np.sqrt(np.mean(np.square(usable_z)))) if usable_z.size else math.inf
    abnormal_indices = [
        int(index)
        for index in np.where(observable & (robust_z >= PSD_ABNORMAL_BAND_Z_MIN))[0]
    ]

    # PSD 是辅助异常证据，优先保证特异性：整体谱形和多个局部频带必须同时偏离。
    # 单独一个条件只标为灰区，避免工况/传感链小变化制造假故障线索。
    if (
        robust_distance >= PSD_DIFFERENT_DISTANCE_MIN
        and len(abnormal_indices) >= PSD_MIN_ABNORMAL_BANDS
    ):
        status = "different_from_normal"
    elif robust_distance <= PSD_SIMILAR_DISTANCE_MAX and not abnormal_indices:
        status = "similar_to_normal"
    else:
        status = "borderline"

    abnormal_bands: list[dict[str, Any]] = []
    for index in sorted(abnormal_indices, key=lambda item: robust_z[item], reverse=True)[:3]:
        band = band_rows[index] if isinstance(band_rows, list) and index < len(band_rows) else {}
        abnormal_bands.append(
            {
                "band_hz": band.get("band_hz") if isinstance(band, Mapping) else None,
                "deviation_db": float(current[index] - median[index]),
                "robust_z": float(robust_z[index]),
            }
        )

    return {
        "evidence_id": "signal.psd_normality",
        "status": status,
        "reference_count": len(references),
        "reference_source_count": len(reference_source_groups) or None,
        "robust_distance": robust_distance,
        "similarity_score": float(math.exp(-robust_distance / 3.0)),
        "abnormal_band_count": len(abnormal_indices),
        "abnormal_bands": abnormal_bands,
        "reference_case_ids": reference_ids,
        "role": "normality_only_not_fault_localization",
        "thresholds": {
            "mad_scale_floor_db": PSD_MAD_SCALE_FLOOR_DB,
            "similar_distance_max": PSD_SIMILAR_DISTANCE_MAX,
            "different_distance_min": PSD_DIFFERENT_DISTANCE_MIN,
            "abnormal_band_z_min": PSD_ABNORMAL_BAND_Z_MIN,
            "min_abnormal_bands": PSD_MIN_ABNORMAL_BANDS,
        },
    }


def filter_by_condition(
    cases: Sequence[CaseRecord],
    bearing_type: str,
    rpm: float,
    rpm_ratio: float = 0.5,
) -> list[CaseRecord]:
    """第一段筛选：按数据集(轴承型号) + 工况(转速范围)找同类案例。

    - 数据集：同 bearing_type 或 generic（专家通用案例）才保留；
    - 工况：案例的 rpm 范围与查询 rpm±ratio 有交集才保留（未标注 rpm 的保留）。
    """

    query_bearing = (bearing_type or "").strip().lower()
    query_rpm = float(rpm or 0.0)
    filtered: list[CaseRecord] = []
    for case in cases:
        case_bearing = (case.bearing_type or "").strip().lower()
        if query_bearing and case_bearing != query_bearing and case_bearing != "generic":
            continue
        if query_rpm > 0 and case.operating_condition.rpm_min is not None:
            low = query_rpm * (1.0 - float(rpm_ratio))
            high = query_rpm * (1.0 + float(rpm_ratio))
            case_low = float(case.operating_condition.rpm_min)
            case_high = (
                float(case.operating_condition.rpm_max)
                if case.operating_condition.rpm_max is not None
                else case_low
            )
            if case_low > high or case_high < low:
                continue
        filtered.append(case)
    return filtered


class SampleCaseStore:
    """样本案例库：PostgreSQL 存储 + 结构化/语义检索。"""

    def __init__(self, dsn: str = DEFAULT_DSN, embedding_provider=None) -> None:
        self.store = PostgresCaseStore(dsn, embedding_provider=embedding_provider)

    def save(self, case: CaseRecord) -> None:
        self.store.upsert_case(case)

    def load_all(self, limit: int = 10000) -> list[CaseRecord]:
        return self.store.load_cases(limit=limit)

    def retrieve(self, query: RetrievalQuery, top_k: int = 3) -> list[RetrievedCase]:
        cases = self.load_all()
        return HybridCaseRetriever(cases).retrieve(query, top_k=top_k).matched_cases

    def retrieve_by_condition(
        self, query: RetrievalQuery, top_k: int = 3, rpm_ratio: float = 0.5
    ) -> tuple[list[RetrievedCase], int]:
        """两段式检索：先按数据集+工况筛选同类，再在同类里按特征相似度检索。

        返回 (检索到的 top-K 案例, 筛选后同类案例总数)；若同类为空则回退全库。
        """

        cases = self.load_all()
        filtered = filter_by_condition(cases, query.bearing_type, query.rpm, rpm_ratio)
        if not filtered:
            filtered = cases
        result = HybridCaseRetriever(filtered).retrieve(query, top_k=top_k)
        return result.matched_cases, len(filtered)

    def retrieve_balanced_by_condition(
        self,
        bearing_type: str,
        rpm: float,
        features: Mapping[str, Any],
        top_k: int = 4,
        rpm_ratio: float = 0.5,
    ) -> tuple[list[RetrievedCase], int]:
        """按工况过滤后，每个故障族取一个最佳案例，避免使用基线标签预筛。"""

        cases = self.load_all()
        filtered = filter_by_condition(cases, bearing_type, rpm, rpm_ratio) or cases
        retriever = HybridCaseRetriever(filtered)
        candidates: list[RetrievedCase] = []
        seen: set[str] = set()
        for family in ("BPFO", "BPFI", "BSF", "NONE"):
            query = query_from_features(bearing_type, rpm, features, family)
            matched = retriever.retrieve(query, top_k=max(1, len(filtered))).matched_cases
            family_match = next((item for item in matched if item.fault_family == family), None)
            if family_match is not None and family_match.case_id not in seen:
                candidates.append(family_match)
                seen.add(family_match.case_id)
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: max(1, int(top_k))], len(filtered)

    def compare_normal_psd_by_condition(
        self,
        bearing_type: str,
        rpm: float,
        current_psd: Mapping[str, Any],
        *,
        rpm_ratio: float = 0.05,
        min_reference_cases: int = 3,
    ) -> dict[str, Any]:
        """在同型号、近转速且同采样率的正常案例中比较 PSD。

        正常基线不允许回退到跨型号案例；否则 ``unavailable`` 比一个失真的
        “正常/异常”结论更诚实。当前接口尚无测点/传感器标识，因此该结果仍是辅助证据。
        """

        query_bearing = (bearing_type or "generic").strip().lower()
        cases = [
            case
            for case in filter_by_condition(
                self.load_all(), bearing_type, rpm, rpm_ratio=rpm_ratio
            )
            if (case.bearing_type or "").strip().lower() == query_bearing
            and case.operating_condition.rpm_min is not None
        ]
        query_fs = float(current_psd.get("sampling_rate_hz") or 0.0)
        condition_matched: list[CaseRecord] = []
        for case in cases:
            case_fs = float(
                case.auxiliary_evidence.psd_sampling_rate_hz
                or case.raw_record.get("fs")
                or 0.0
            )
            if query_fs > 0.0 and case_fs > 0.0:
                if abs(case_fs - query_fs) / query_fs > 0.01:
                    continue
            elif query_fs > 0.0 or case_fs > 0.0:
                # 只有一侧缺采样率时无法证明 PSD 物理频带可比。
                continue
            condition_matched.append(case)

        result = compare_psd_to_normal_cases(
            current_psd,
            condition_matched,
            min_reference_cases=min_reference_cases,
        )
        result["condition_scope"] = {
            "bearing_type": bearing_type,
            "rpm": float(rpm),
            "rpm_tolerance_ratio": float(rpm_ratio),
            "sampling_rate_hz": query_fs or None,
            "measurement_point_matched": False,
        }
        return result

    def semantic_search(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.store.search_case_text(query_text, top_k=top_k)
