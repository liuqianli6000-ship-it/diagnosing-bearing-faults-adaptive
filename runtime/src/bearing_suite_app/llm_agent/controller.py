"""Multi-round LLM controller for v2.

Three roles share one OpenAI-compatible client:

* planner: choose the next Layer-3 method and parameter direction;
* evaluator: judge whether the new trial changed the evidence enough to stop;
* synthesizer: write the final diagnosis from the accumulated evidence.

The controller never decides by itself that evidence is clear; that is the
local ``EvidenceGate``.  Every call is recorded with role/round/usage.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping
from time import sleep

from bearing_suite_app.llm import LLMClient, load_env

ALLOWED_ACTIONS = (
    "adaptive_notch",
    "med",
    "mckd",
    "momeda",
    "iterative_sk",
    "subband_analysis",
    "stop",
)
FAULT_TYPES = (
    "outer_race_fault",
    "inner_race_fault",
    "rolling_element_fault",
    "normal",
    "unknown",
)

PLANNER_SYSTEM = """你负责为轴承精密诊断选择下一项处理。
根据初诊证据、既往精密分析 trial 和剩余预算，选择下一步 Layer-3 方法。
只输出 JSON：{"action": "...", "parameter_hint": {...}, "reason": "..."}。
action 只能是 adaptive_notch、med、mckd、momeda、iterative_sk、subband_analysis、stop。
证据充足或没有可用的下一步时选择 stop。"""

EVALUATOR_SYSTEM = """你负责评估本轮处理效果。
比较本轮精密分析前后的证据，判断是否继续、换方法或停止。
只输出 JSON：{"decision": "continue|stop|switch", "next_action": "...", "reason": "...", "confidence": 0.0}。
目标故障族的谐波证据无改善时，选择换方法或停止。"""

SYNTH_SYSTEM = """你是轴承故障诊断综合器。
只依据提供的工具证据输出最终 JSON：
{"fault_type": "outer_race_fault|inner_race_fault|rolling_element_fault|normal|unknown",
 "confidence": 0.0, "severity": "早期|中期|晚期|无法判断",
 "reasoning": "中文，引用证据ID或工具结果", "key_evidence": ["..."],
 "uncertainty": "证据不足或冲突时的说明"}
不得编造未出现的测量；证据不足时 fault_type=unknown。"""

PLANNER_PLAN_SYSTEM = """你是轴承诊断的分析规划器。先读 signal_profile，再提交分析计划。
只输出 JSON：
{"reason": "...", "plan": [{"tool": "...", "params": {...}, "purpose": "..."}]}
要求：
1. plan 最多 3 步；
2. 至少包含一个 L2 证据步骤（envelope_spectrum、harmonic_match、emd_decompose、wavelet_decompose）；
3. 可选 L3：adaptive_notch、med、mckd、momeda、iterative_sk、subband_analysis，最多 2 步；
4. 每步必须说明 purpose；params 可写 {"auto": true}；
5. 此轮输出仅包含上述计划字段。"""

STEP_EVALUATOR_SYSTEM = """你负责评估轴承分析计划的当前步骤。
根据当前步骤结果、信号画像和剩余计划，判断继续、换计划或停止。
只输出 JSON：
{"decision": "continue|switch|stop", "reason": "...", "next_tool": "..."}
根据当前步骤返回的指标说明决定。"""

PROCESSING_PLAN_SYSTEM = """你是轴承信号处理的规划器。
根据 signal_assessment 只规划第 1 轮滤波/处理方案；第 2/3 轮由后续真实指标微调。
只输出 JSON：
{"reason": "...", "rounds": [
  {"round": 1, "parent": 0,
   "filters": [{"tool": "bandpass|adaptive_notch", "params": {...}}],
   "processing": [{"tool": "med", "params": {...}}],
   "bands": [[low, high]],
   "direction": {"envelope_snr": "up", "out_of_band": "down"},
   "purpose": "..."}
]}
要求：
1. 只输出 1 轮；
2. filters 只能使用 bandpass、adaptive_notch。bandpass 的参数为 low_hz、high_hz、order（整数 1–8），频带须满足 0 < low_hz < high_hz <= 采样率的一半。adaptive_notch 自行检测窄带干扰，参数仅为 max_notches（整数 1–4）和 q_factor（5–60）；
3. processing 只能使用 med，参数为 filter_length 和 max_iter；
4. bands 必须来自 assessment 的谱峭度带，或明确给出合法频带；
5. 此轮输出仅包含上述处理计划字段。"""


@dataclass
class ControllerCall:
    role: str
    round: int
    prompt_chars: int
    response: str
    usage: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class LLMController:
    client: LLMClient | None
    max_calls: int = 9
    calls: list[ControllerCall] = field(default_factory=list)

    @classmethod
    def from_provider(
        cls,
        provider: str = "qwen",
        *,
        env_path: str | None = None,
        max_calls: int = 9,
        model: str | None = None,
    ) -> "LLMController":
        try:
            client = LLMClient(provider, env_path=env_path, model=model,
                               thinking=False if provider == "deepseek" else None)
        except Exception:  # noqa: BLE001
            client = None
        return cls(client=client, max_calls=max_calls)

    @property
    def available(self) -> bool:
        return self.client is not None and self.client.available

    def budgeted_client(self, client: Any, *, role: str, round_index: int = 0):
        """Share the controller budget with a branch using complete_messages."""
        return _BudgetedClient(self, client, role, round_index)

    def _call(self, role: str, round_index: int, system: str, prompt: str) -> dict[str, Any]:
        if len(self.calls) >= self.max_calls:
            return {"error": "llm_budget_exhausted"}
        if not self.available:
            return {"error": "llm_unavailable"}
        try:
            text = self.client.chat(prompt, system=system)
            parsed = _parse_json(text)
            exchange = self.client.last_exchange or {}
            self.calls.append(
                ControllerCall(
                    role=role,
                    round=round_index,
                    prompt_chars=len(prompt),
                    response=text,
                    usage=exchange.get("usage"),
                )
            )
            if parsed is None:
                return {"error": "invalid_json", "raw": text[:1000]}
            return parsed
        except Exception as exc:  # noqa: BLE001
            self.calls.append(
                ControllerCall(
                    role=role,
                    round=round_index,
                    prompt_chars=len(prompt),
                    response="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return {"error": f"{type(exc).__name__}: {exc}"}

    def plan(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        prompt = (
            "当前诊断摘要：\n"
            + json.dumps(summary, ensure_ascii=False, indent=1)
            + "\n请选择下一步动作。"
        )
        parsed = self._call("planner", _round(summary), PLANNER_SYSTEM, prompt)
        if parsed.get("error"):
            return self._fallback_plan(summary, parsed["error"])
        action = str(parsed.get("action") or "")
        if action not in ALLOWED_ACTIONS:
            return self._fallback_plan(summary, f"invalid_action:{action}")
        parsed["action"] = action
        parsed["source"] = "llm"
        return parsed

    def evaluate(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        prompt = (
            "紧凑诊断状态：\n"
            + json.dumps(summary, ensure_ascii=False, indent=1)
            + "\n请判断是否继续精密分析。"
        )
        parsed = self._call("evaluator", _round(summary), EVALUATOR_SYSTEM, prompt)
        if parsed.get("error"):
            return self._fallback_evaluate(summary, parsed["error"])
        decision = str(parsed.get("decision") or "")
        if decision not in {"continue", "stop", "switch"}:
            return self._fallback_evaluate(summary, f"invalid_decision:{decision}")
        parsed["source"] = "llm"
        return parsed

    def synthesize(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        prompt = (
            "完整证据摘要：\n"
            + json.dumps(summary, ensure_ascii=False, indent=1)
            + "\n请输出最终诊断 JSON。"
        )
        parsed = self._call("synthesizer", _round(summary), SYNTH_SYSTEM, prompt)
        if parsed.get("error") or parsed.get("fault_type") not in FAULT_TYPES:
            return self._fallback_synthesize(summary, parsed.get("error") or "invalid_fault_type")
        parsed["source"] = "llm"
        return parsed

    def judge_diagnosis(self, evidence: Mapping[str, Any], skill: str) -> dict[str, Any]:
        if not self.available:
            return {"error": "llm_unavailable"}
        allowed_refs = set(evidence)
        aliases = {}
        for band in evidence.get("envelope", {}).get("band_summaries", []):
            for item in (band, *band.get("families", {}).values()):
                ref = item.get("evidence_id")
                if isinstance(ref, str):
                    aliases[f"envelope.band_summaries.{ref}"] = ref
        if "segments" in evidence:
            aliases["envelope.segments"] = "segments"
        for key in evidence.get("signal", {}):
            if key not in evidence:
                aliases[key] = f"signal.{key}"
        stack = list(evidence.values())
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                ref = value.get("evidence_id")
                if isinstance(ref, str):
                    allowed_refs.add(ref)
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
        system = (
            "根据下列 Skill 判据和当前工具证据进行轴承诊断。只输出 JSON，字段为 "
            "fault_type（normal、inner_race_fault、outer_race_fault、rolling_element_fault、unknown 之一）、"
            "confidence（0 到 1 的判断分值）、reasoning（简短中文证据解释）、"
            "evidence_refs（非空数组，使用实际出现的 evidence_id 或输入字段路径，如 signal.time）、"
            "follow_up（需要进一步分析时写明项目，否则为空字符串）。"
            "引用频带和故障族时直接复制对应 evidence_id，保留选带方法前缀。"
            "\n\n" + skill
        )
        prompt = json.dumps(evidence, ensure_ascii=False)
        parsed = self._call("diagnosis", 0, system, prompt)
        if (str(parsed.get("error", "")).startswith(("APIConnectionError:", "APITimeoutError:"))
                and len(self.calls) < self.max_calls):
            sleep(1)
            parsed = self._call("diagnosis", 0, system, prompt)
        if parsed.get("error"):
            return {"error": parsed["error"]}
        fault = parsed.get("fault_type")
        confidence = parsed.get("confidence")
        reasoning = parsed.get("reasoning")
        refs = parsed.get("evidence_refs")
        follow_up = parsed.get("follow_up")
        if isinstance(refs, list):
            refs = [aliases.get(ref, ref) if isinstance(ref, str) and ref not in allowed_refs else ref
                    for ref in refs]
            for ref in refs:
                if not isinstance(ref, str) or ref in allowed_refs:
                    continue
                value = evidence
                for key in ref.split("."):
                    if not isinstance(value, Mapping) or key not in value:
                        break
                    value = value[key]
                else:
                    allowed_refs.add(ref)
        if (fault not in FAULT_TYPES or type(confidence) not in (int, float)
                or not math.isfinite(confidence) or not 0 <= confidence <= 1
                or not isinstance(reasoning, str) or not reasoning.strip()
                or not isinstance(follow_up, str) or not isinstance(refs, list) or not refs
                or any(not isinstance(ref, str) or ref not in allowed_refs for ref in refs)):
            return {"error": "invalid_diagnosis"}
        return {"source": "skill_llm", "fault_type": fault,
                "confidence": 0.0 if fault == "unknown" else confidence,
                "reasoning": reasoning.strip(), "evidence_refs": refs, "follow_up": follow_up.strip()}

    def review_diagnosis(self, evidence: Mapping[str, Any], draft: Mapping[str, Any],
                         report: str, skill: str) -> dict[str, Any]:
        prompt = json.dumps({"evidence": evidence, "diagnosis": draft, "report": report}, ensure_ascii=False)
        result = self._call("reason_reviewer", 0, skill, prompt)
        if result.get("error"):
            return {"status": "unavailable", "error": result["error"]}
        issues = result.get("issues")
        categories = {"证据引用", "结果输出"}
        if set(result) != {"issues"} or not isinstance(issues, list) or any(
            not isinstance(issue, dict) or set(issue) != {"category", "quote", "problem", "suggestion"}
            or not all(isinstance(value, str) for value in issue.values())
            or issue["category"] not in categories or not issue["problem"].strip() or not issue["suggestion"].strip()
            for issue in issues
        ):
            return {"status": "unavailable", "error": "invalid_review"}
        return {"status": "reviewed", "issues": issues}

    def plan_processing(self, assessment: Mapping[str, Any], *, adaptive: bool = True,
                        reserve_calls: int = 0) -> dict[str, Any]:
        summary = {"adaptive": adaptive, "signal_assessment": assessment, "max_rounds": 1}
        prompt = (
            "信号评估：\n"
            + json.dumps(summary, ensure_ascii=False, indent=1)
            + "\n请输出第 1 轮滤波/处理计划 JSON。"
        )
        parsed = {}
        if len(self.calls) + reserve_calls < self.max_calls:
            parsed = self._call("planner", 1, PROCESSING_PLAN_SYSTEM, prompt)
        if parsed.get("error") or not isinstance(parsed.get("rounds"), list):
            from bearing_suite_app.processing.plan import build_plan

            fallback = build_plan(assessment, adaptive=adaptive)
            fallback["source"] = "local_fallback"
            return fallback
        parsed["source"] = "llm"
        return parsed

    def plan_analysis(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        prompt = (
            "信号画像与分析状态：\n"
            + json.dumps(summary, ensure_ascii=False, indent=1)
            + "\n请输出分析计划 JSON。"
        )
        parsed = self._call("planner", _round(summary), PLANNER_PLAN_SYSTEM, prompt)
        if parsed.get("error") or not isinstance(parsed.get("plan"), list):
            return self._fallback_plan_analysis(summary, parsed.get("error") or "invalid_plan")
        parsed["source"] = "llm"
        parsed["round"] = _round(summary)
        return parsed

    def evaluate_step(self, summary: Mapping[str, Any]) -> dict[str, Any]:
        prompt = (
            "当前分析步骤状态：\n"
            + json.dumps(summary, ensure_ascii=False, indent=1)
            + "\n请判断下一步。"
        )
        parsed = self._call("evaluator", _round(summary), STEP_EVALUATOR_SYSTEM, prompt)
        if parsed.get("error"):
            return self._fallback_step_evaluate(summary, parsed["error"])
        if parsed.get("decision") not in {"continue", "switch", "stop"}:
            return self._fallback_step_evaluate(summary, "invalid_decision")
        parsed["source"] = "llm"
        return parsed

    def _fallback_plan_analysis(
        self, summary: Mapping[str, Any], reason: str
    ) -> dict[str, Any]:
        profile = summary.get("signal_profile") or {}
        kurtosis = float(profile.get("kurtosis") or 0.0)
        peak_count = int(profile.get("narrowband_peak_count") or 0)
        plan = [
            {
                "tool": "envelope_spectrum",
                "params": {"auto": True},
                "purpose": "建立包络谱基线",
            }
        ]
        if summary.get("adaptive", True):
            if kurtosis >= 6.0:
                plan.append(
                    {
                        "tool": "med",
                        "params": {"auto": True},
                        "purpose": "高峭度/冲击性：增强周期冲击",
                    }
                )
            elif peak_count > 0:
                plan.append(
                    {
                        "tool": "adaptive_notch",
                        "params": {"auto": True},
                        "purpose": "窄带干扰：抑制干扰峰",
                    }
                )
            else:
                plan.append(
                    {
                        "tool": "iterative_sk",
                        "params": {"auto": True},
                        "purpose": "精化解调频带",
                    }
                )
        return {
            "reason": f"本地回退计划：{reason}",
            "plan": plan,
            "source": "local",
            "round": _round(summary),
        }

    def _fallback_step_evaluate(
        self, summary: Mapping[str, Any], reason: str
    ) -> dict[str, Any]:
        index = int(summary.get("plan_index", 0))
        plan = summary.get("analysis_plan") or []
        if index < len(plan):
            return {
                "decision": "continue",
                "reason": f"本地回退评估，继续剩余 {len(plan) - index} 步：{reason}",
                "source": "local",
            }
        return {
            "decision": "stop",
            "reason": f"计划已执行完：{reason}",
            "source": "local",
        }

    def _fallback_plan(self, summary: Mapping[str, Any], reason: str) -> dict[str, Any]:
        round_index = int(summary.get("round", 1))
        gate = summary.get("gate") or {}
        if gate.get("verdict") == "clear":
            action = "stop"
        elif round_index == 1:
            action = "iterative_sk"
        elif round_index == 2:
            action = "med"
        elif round_index == 3:
            action = "subband_analysis"
        else:
            action = "stop"
        return {
            "action": action,
            "parameter_hint": {},
            "reason": f"本地回退规划：{reason}",
            "source": "local",
        }

    def _fallback_evaluate(self, summary: Mapping[str, Any], reason: str) -> dict[str, Any]:
        gate = summary.get("gate") or {}
        round_index = int(summary.get("round", 1))
        max_rounds = int(summary.get("max_rounds", 3))
        if gate.get("verdict") == "clear":
            decision = "stop"
        elif round_index >= max_rounds:
            decision = "stop"
        else:
            decision = "continue"
        return {
            "decision": decision,
            "reason": f"本地回退评估：{reason}",
            "source": "local",
        }

    def _fallback_synthesize(self, summary: Mapping[str, Any], reason: str) -> dict[str, Any]:
        baseline = summary.get("baseline") or {}
        fault_type = baseline.get("fault_type") or "unknown"
        if fault_type not in FAULT_TYPES:
            fault_type = "unknown"
        return {
            "fault_type": fault_type,
            "confidence": float(baseline.get("confidence") or 0.0),
            "severity": "无法判断",
            "reasoning": "LLM 不可用或输出无效，采用确定性基线结论。",
            "key_evidence": [
                f"baseline:{fault_type}",
                f"gate:{summary.get('gate', {}).get('verdict')}",
            ],
            "uncertainty": reason,
            "source": "local_fallback",
        }


class _BudgetedClient:
    def __init__(self, controller: LLMController, client: Any, role: str, round_index: int):
        self.controller = controller
        self.client = client
        self.role = role
        self.round_index = round_index

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)

    def complete_messages(self, messages, **kwargs):
        if len(self.controller.calls) >= self.controller.max_calls:
            raise RuntimeError("llm_budget_exhausted")
        response = ""
        error = None
        try:
            message = self.client.complete_messages(messages, **kwargs)
            response = getattr(message, "content", "") or ""
            return message
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            exchange = getattr(self.client, "last_exchange", None) or {}
            self.controller.calls.append(ControllerCall(
                role=self.role,
                round=self.round_index,
                prompt_chars=sum(len(str(item.get("content", ""))) for item in messages),
                response=response,
                usage=exchange.get("usage"),
                error=error,
            ))


def _round(summary: Mapping[str, Any]) -> int:
    try:
        return int(summary.get("round", 1))
    except (TypeError, ValueError):
        return 1


def _parse_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    value = text.strip()
    for candidate in (value, _between_braces(value)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _between_braces(value: str) -> str | None:
    start, end = value.find("{"), value.rfind("}")
    return value[start : end + 1] if start >= 0 and end > start else None
