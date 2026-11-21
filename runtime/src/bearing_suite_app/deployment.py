"""Deployment settings shared by the web and single-case CLI entry points."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

APP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEPLOYMENT = APP_ROOT / "config" / "deployment.json"


class Deployment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    output_dir: Path
    env_file: Path
    diagnosis_config: Path | None
    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    open_browser: bool
    web_max_llm_calls: int = Field(ge=0)
    max_llm_calls: int = Field(ge=0)
    max_rounds: int | None = Field(ge=0)
    max_trials: int = Field(ge=1)
    max_plan_rounds: int = Field(ge=0)


def load_deployment(path: str | Path | None = None) -> Deployment:
    values = json.loads(DEFAULT_DEPLOYMENT.read_text(encoding="utf-8-sig"))
    if path is not None:
        override = json.loads((APP_ROOT / path).read_text(encoding="utf-8-sig"))
        if not isinstance(override, dict):
            raise ValueError("应用部署配置必须是 JSON 对象")
        values.update(override)
    settings = Deployment.model_validate(values)
    for key in ("data_dir", "output_dir", "env_file", "diagnosis_config"):
        value = getattr(settings, key)
        if value is not None:
            setattr(settings, key, (APP_ROOT / value).resolve())
    return settings
