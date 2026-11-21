"""精简配置：只保留有物理意义的少量参数，其余一律用默认值。

matching 参数的唯一来源是 skill 的 ``mfd_method.json``（profiles.default）；
工程参数（质量门、包络、频带数、裁决档位、案例库 top_k）在本文件给出默认值，
外部 JSON 可覆盖。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "signal": {
        "quality": {
            "min_samples": 1024,
            "max_clipping_ratio": 0.05,
            "max_trend_ratio": 0.35,
            "max_dropout_run_ratio": 0.05,
            "max_speed_relative_span": 0.02,
            "max_speed_reference_relative_error": 0.02,
        },
        "diagnosability": {
            "min_valid_bands": 2,
            "min_observable_harmonics": 3,
            "max_df_relative_to_fault": 0.05,
            "reliable_nyquist_fraction": 0.95,
            "min_band_energy_ratio": 1e-4,
            "min_envelope_snr_db": 8.0,
        },
        "envelope": {
            "mode": "squared",
            "window": "hann",
        },
        "band_count": 5,
    },
    "matching": {
        # 兜底值；正常路径由 mfd_method.json 提供（见 load_config）
        "max_harmonics": 5,
        "tolerance_resolution_multiplier": 2.5,
        "minimum_tolerance_hz": 0.5,
        "speed_relative_uncertainty": 0.005,
        "geometry_relative_uncertainty": 0.005,
        "slip_relative_uncertainty_by_family": {
            "BPFI": 0.01,
            "BPFO": 0.01,
            "BSF": 0.02,
        },
        "min_relative_amplitude": 0.10,
        "min_robust_z": 4.5,
        "background_radius_bins": 12,
    },
    "decision": {
        "strong_min_bands": 3,
        "moderate_min_bands": 2,
        "dominance_margin": 2,
        # 细化频率必须在同一批阶次上比理论频率显著这么多（单位：局部稳健 z），
        # 否则视为在搜索范围内挑到偶然峰、不予采用。与轴承型号/数据集无关。
        "refined_gain_min": 3.0,
        "confidence": {
            "strong_unique": 0.85,
            "strong_dominant": 0.8,
            "moderate_unique": 0.65,
            "moderate_dominant": 0.6,
            "normal": 0.75,
            "unknown": 0.0,
        },
    },
    "case_kb": {
        "top_k": 3,
    },
    "visualization": {
        "enabled": False,
        "artifact_dir": "outputs/evidence",
        "save_multiband_envelope": True,
        "dpi": 160,
        "max_plot_points": 2000,
    },
    "audit": {
        "enabled": False,
        "trace_dir": "outputs/runs",
    },
    "agent_runtime": {
        "enabled": True,
        "skill_driven": True,
        "max_tool_calls": 4,
        "max_turns": 6,
        "max_no_progress_actions": 2,
        "tool_timeout_seconds": 30.0,
    },
    "multimodal": {
        "enabled": False,
        "max_image_bytes": 8 * 1024 * 1024,
    },
    "llm": {
        "fallback_policy": "deterministic",
        "require_evidence_refs": True,
        "max_semantic_repair_attempts": 1,
        "confidence_mapping": {
            "high": 0.85,
            "medium": 0.65,
            "low": 0.40,
            "unknown": 0.0,
        },
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_mfd_matching() -> dict[str, Any]:
    """从 skill 的 mfd_method.json 加载 matching 参数（profiles.default）。"""

    path = (
        _repo_root()
        / "skills"
        / "diagnosing-bearing-faults-adaptive"
        / "mfd_method.json"
    )
    if not path.exists():
        return dict(DEFAULT_CONFIG["matching"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload["profiles"][payload.get("default_profile", "default")])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return dict(DEFAULT_CONFIG["matching"])


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载配置：mfd_method.json（matching）→ 默认值 → 外部 JSON 覆盖。"""

    config = json.loads(json.dumps(DEFAULT_CONFIG))
    config["matching"] = load_mfd_matching()
    if path is not None:
        with open(path, "r", encoding="utf-8") as handle:
            override = json.load(handle)
        config = _deep_merge(config, override)
    return config
