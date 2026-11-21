"""LLM 客户端（OpenAI 兼容接口）。

职责：
1. 读取项目根目录的 ``.env``，解析 DashScope(Qwen) / 火山方舟(Doubao) / OpenAI 配置。
2. ``diagnose_evidence``：读取工具证据包，输出结构化最终诊断与解释。
3. 兼容保留 ``explain_verdict`` / ``explain_with_context`` 两个旧解释接口。

确定性规则由编排层作为审计基线与失败回退，不发送给 LLM，避免锚定偏差。
无 python-dotenv 依赖，手动解析 .env；缺失 key 时自动降级为不可用。

本模块负责屏蔽供应商差异。同一需求在不同模型上的能力并不一致：

  * 严格 JSON Schema 输出（``response_format``）支持程度不同，部分端点完全不支持；
  * tool calling 与严格工具参数校验的支持范围不同；
  * 图像输入支持情况不同；
  * 错误形态多样，包括超时、限流、空内容与 Markdown 包裹的 JSON。

若由上层编排处理这些差异，代码将退化为大量供应商分支判断。本模块将其收敛为一组
能力开关（``supports_*``），上层仅需查询客户端能力，无需区分供应商，
即适配器模式在 Agent 项目中的应用。

贯穿全项目的约定：模型输出属于建议而非事实。本模块仅负责获取文本并剥离明显的
非结构化包裹，不执行业务判断；结构化校验、证据引用校验与业务一致性校验均由上层
负责。客户端不参与业务决策，编排层不涉及供应商细节。

与模型交互的三项约束：

1. 能力开关需具备自愈能力。``supports_strict_output`` 初始按供应商推断，
   实际调用失败时置为 False 并回退至提示词约定 JSON（见 ``diagnose_evidence``），
   因供应商声明与具体端点、模型的实际能力可能不一致。
2. 每次调用须保留完整记录。``complete_messages`` 将 messages、tools、
   response_format、原始文本、token 用量、耗时与请求 ID 写入 ``exchanges``，
   作为排障与复现的第一手依据。
3. 失败须区分可修复与不可修复。模型返回内容但结构不符时可执行一次修复
   （``repair_diagnosis``）；连接失败或缺少配置时直接标记为不可用，
   由上层回退确定性基线，不再消耗修复调用。
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .diagnosis_schemas import EvidenceBundle, LLMDiagnosis
from .prompts import (
    EVIDENCE_DIAGNOSIS_SYSTEM,
    LEGACY_EXPLANATION_SYSTEM,
    REPAIR_DIAGNOSIS_SYSTEM,
)

# ---------------------------------------------------------------------------
# 供应商注册表：将配置项名称与默认端点集中为数据，避免分散的条件分支。
# 新增 OpenAI 兼容服务通常只需在此添加一项并在 .env 中配置密钥，无需改动逻辑代码。
# ---------------------------------------------------------------------------
PROVIDERS: dict[str, dict[str, Any]] = {
    "qwen": {
        "api_key": "DASHSCOPE_API_KEY",
        "base_url": "DASHSCOPE_BASE_URL",
        "model": "DASHSCOPE_MODEL",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "api_key": "DEEPSEEK_API_KEY",
        "base_url": "DEEPSEEK_BASE_URL",
        "model": "DEEPSEEK_MODEL",
        "default_base_url": "https://api.deepseek.com",
        "default_model": None,
    },
    "doubao": {
        "api_key": "ARK_API_KEY",
        "base_url": "ARK_BASE_URL",
        "model": "ARK_MODEL",
        "default_base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": None,
    },
    "openai": {
        "api_key": "OPENAI_API_KEY",
        "base_url": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
        "default_base_url": None,
        "default_model": None,
    },
    "glm": {
        "api_key": "GLM_API_KEY",
        "base_url": "GLM_BASE_URL",
        "model": "GLM_MODEL",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-plus",
    },
}


def load_env(env_path: str | Path | None = None) -> dict[str, str]:
    """从 .env 读取环境变量（手动解析，无 dotenv 依赖）。"""

    if env_path is None:
        env_path = Path(__file__).resolve().parents[2] / ".env"
    # Process variables take precedence; a portable skill need not carry a .env file.
    env: dict[str, str] = dict(os.environ)
    path = Path(env_path)
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env.setdefault(key.strip(), value.strip())
    return env


def _parse_text(text: str | None) -> str | None:
    """取出纯文本回答，去掉可能的 Markdown 代码块包裹。"""
    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:\w+)?\s*(.*?)\s*```", cleaned, flags=re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    return cleaned or None


def _parse_json(text: str | None) -> dict[str, Any] | None:
    """从纯 JSON 或 Markdown fenced JSON 中提取对象。

    与 ``agent_runtime._parse_object`` 采用相同的三层降级：整体解析、剥离
    Markdown 代码块、按首尾花括号截取。

    容忍这些形式的原因：在不支持严格 JSON Schema 的端点上，fenced JSON 是最常见的
    输出形态，容错的成本低于反复重试。前提是仅在解析层容忍，校验层不作放宽：
    剥离包裹后仍须通过 Pydantic 与证据引用校验。
    """

    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.S)
    if fenced:
        cleaned = fenced.group(1)
    try:
        payload = json.loads(cleaned)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


class LLMClient:
    """OpenAI 兼容 LLM 客户端（证据诊断器）。

    对外提供三项能力，分别对应三个层次：
      * ``complete_messages``：单轮底层请求，支持 tools，供 Agent 循环调用；
      * ``diagnose_evidence``：由证据生成结构化诊断（单轮抽取模式）；
      * ``repair_diagnosis``：受约束的修复调用，仅修正结构，不改变结论。
    """

    def __init__(
        self,
        provider: str = "qwen",
        *,
        env_path: str | Path | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
        thinking: bool | None = None,
    ) -> None:
        self.provider = provider
        self.temperature = temperature
        self.thinking = thinking
        env = load_env(env_path)
        conf = PROVIDERS.get(provider)
        if conf is None:
            raise ValueError(f"不支持的 provider: {provider}")

        api_key = env.get(conf["api_key"])
        base_url = env.get(conf["base_url"]) or conf["default_base_url"]
        model = model or env.get(conf["model"]) or conf["default_model"]

        self.client = None
        self.model = model
        self.initialization_error: str | None = None
        # ------------------------------------------------------------------
        # 能力开关（capability flags）：本模块的核心抽象。
        # 统一遵循"环境变量显式指定优先于供应商默认推断"的规则。
        # 上层（agent / agent_runtime / dual_channel）仅读取这些布尔值决定执行路径，
        # 无需区分具体模型。
        #
        # 默认值取保守值的原因：
        #   * 严格 JSON Schema：仅 openai 默认开启，其它端点常见声明支持但实际报错，
        #     故需通过 LLM_STRUCTURED_OUTPUT_MODE=json_schema 显式启用；
        #   * 严格工具参数：同上，需 LLM_STRICT_TOOLS 显式启用；
        #   * tool calling：三家主流供应商默认开启，为本项目主路径；
        #   * 图像输入：默认关闭。多模态将改变证据来源与置信度口径，
        #     须由使用者显式启用。
        # ------------------------------------------------------------------
        structured_mode = env.get("LLM_STRUCTURED_OUTPUT_MODE", "auto").lower()
        self.supports_strict_output = structured_mode == "json_schema" or (
            structured_mode == "auto" and provider == "openai"
        )
        strict_tools_mode = env.get("LLM_STRICT_TOOLS", "auto").lower()
        self.supports_strict_tools = strict_tools_mode in {
            "1",
            "true",
            "yes",
            "on",
        } or (strict_tools_mode == "auto" and provider == "openai")
        tool_mode = env.get("LLM_TOOL_CALLING", "auto").lower()
        self.supports_tool_calling = tool_mode in {"1", "true", "yes", "on"} or (
            tool_mode == "auto" and provider in {"openai", "qwen", "doubao", "deepseek"}
        )
        image_mode = env.get("LLM_IMAGE_INPUT", "false").lower()
        self.supports_image_input = image_mode in {"1", "true", "yes", "on"}
        self.exchanges: list[dict[str, Any]] = []
        self.last_exchange: dict[str, Any] | None = None
        # 缺少密钥或模型时不抛出异常，仅将 client 置空并记录原因。
        # 无 API 时仍须支持确定性基线，故"不可用"（available=False）属正常状态，
        # 由编排层决定回退。
        # max_retries=0 为有意设置：重试策略集中于 Agent 层（一次修复加重试回退），
        # 不与 SDK 的隐式重试叠加，否则实际失败次数难以审计。
        if api_key and model:
            try:
                from openai import OpenAI

                kwargs: dict[str, Any] = {
                    "api_key": api_key,
                    "timeout": timeout,
                    "max_retries": 0,
                }
                if base_url:
                    kwargs["base_url"] = base_url
                self.client = OpenAI(**kwargs)
            except Exception:  # noqa: BLE001
                self.client = None
                self.initialization_error = "openai_sdk_initialization_failed"
        elif not api_key:
            self.initialization_error = f"missing_api_key:{conf['api_key']}"
        elif not model:
            self.initialization_error = f"missing_model:{conf['model']}"

    @property
    def available(self) -> bool:
        """是否可以发起请求；上层据此决定执行 LLM 链或回退确定性基线。"""
        return self.client is not None and bool(self.model)

    @staticmethod
    def _model_dump(value: Any) -> Any:
        """将 SDK 返回对象递归转换为可 JSON 序列化的基础类型，供审计落盘。

        SDK 响应对象同时包含 pydantic 模型与普通对象，直接保存会引入不可序列化内容。
        此处统一拆解以保证 ``exchanges`` 始终可落盘；审计链路上任一处序列化失败
        都会导致该次运行缺失证据。
        """
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            return LLMClient._model_dump(value.model_dump(mode="json"))
        if isinstance(value, Mapping):
            return {
                str(key): LLMClient._model_dump(nested)
                for key, nested in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [LLMClient._model_dump(item) for item in value]
        return value

    def complete_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        response_format: Mapping[str, Any] | None = None,
        on_delta: Callable[[str], None] | None = None,
        tool_choice: Mapping[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        """发送单轮兼容 Chat Completions 请求，并记录完整审计元数据。

        本方法是 Agent 循环唯一的对外请求出口，同时承担发送与记账两项职责。

        ``exchange`` 记录内容较为完整，用于支持排障与复现：
          * 请求侧：messages 与 tools / response_format 原样留存。
            提示词与工具 schema 随版本变化，缺少记录将无法复现历史结论；
          * 响应侧：``raw_text``（未解析原文，解析失败时的唯一线索）、
            ``tool_calls``、``finish_reason``（区分正常结束与长度截断）、
            ``usage``（token 用量）、``request_id`` / ``response_model`` /
            ``system_fingerprint``（供应商对账所需）；
          * 时间与错误：``duration_ms`` 与 ``error``。

        两项实现约定：
          1. 失败路径同样写入 exchange 后再抛出，保证失败原因可查；
          2. ``temperature`` 默认 0，诊断结论须可复现。

        返回原始 message 对象而非解析后的 dict：tool_calls 的格式在不同供应商间
        存在差异，统一交由 ``agent_runtime`` 的适配函数处理，
        客户端仅负责通信。

        指定 on_delta 时使用真实流式请求，实时回调公开文本；工具参数先完整组装，
        返回同样的 message 接口。默认非流式调用保持不变。
        """

        if not self.available:
            raise RuntimeError("LLM 不可用（缺少 API key 或 model）")
        # 复制后再发送：避免 SDK 或后续代码就地修改调用方的 messages 列表，
        # 导致审计记录与实际请求不一致（审计使用该副本）。
        request_messages = [dict(message) for message in messages]
        request_options: dict[str, Any] = {"temperature": self.temperature}
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.temperature,
        }
        if self.provider == "deepseek" and self.thinking is not None:
            thinking = {"type": "enabled" if self.thinking else "disabled"}
            kwargs["extra_body"] = {"thinking": thinking}
            request_options["thinking"] = thinking
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
            request_options["max_tokens"] = max_tokens
            if self.provider == "qwen":
                kwargs["extra_body"] = {"enable_thinking": False}
        if tool_choice is not None:
            kwargs["tool_choice"] = dict(tool_choice)
            request_options["tool_choice"] = kwargs["tool_choice"]
        if tools:
            kwargs["tools"] = [dict(tool) for tool in tools]
            request_options["tools"] = kwargs["tools"]
        if response_format:
            kwargs["response_format"] = dict(response_format)
            request_options["response_format"] = kwargs["response_format"]
        if on_delta is not None:
            kwargs.update(stream=True, stream_options={"include_usage": True})
            request_options["stream"] = True

        started = time.time()
        # 先构造记录再发起请求，使请求中途异常时仍可补全错误信息后落盘。
        exchange: dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "messages": request_messages,
            "request_options": request_options,
            "started_at_unix": started,
        }
        try:
            response = self.client.chat.completions.create(**kwargs)
            if on_delta is not None:
                message, stream_meta = self._consume_stream(response, on_delta)
                exchange.update(stream_meta)
            else:
                choice = response.choices[0]
                message = choice.message
                exchange.update(
                    request_id=getattr(response, "id", None),
                    response_model=getattr(response, "model", None),
                    system_fingerprint=getattr(response, "system_fingerprint", None),
                    usage=self._model_dump(getattr(response, "usage", None)),
                    finish_reason=getattr(choice, "finish_reason", None),
                )
            exchange.update(
                {
                    "raw_text": message.content or "",
                    "tool_calls": self._model_dump(
                        getattr(message, "tool_calls", None)
                    ),
                    "duration_ms": round((time.time() - started) * 1000.0, 3),
                    "error": None,
                }
            )
        except Exception as exc:
            # 失败也记账：raw_text=None + error 字符串，事后能区分“网络没通”、
            # “被限流”还是“内容策略拒绝”，而不是只看到一句笼统的失败。
            exchange.update(
                {
                    "raw_text": None,
                    "request_id": None,
                    "response_model": None,
                    "system_fingerprint": None,
                    "usage": None,
                    "finish_reason": None,
                    "duration_ms": round((time.time() - started) * 1000.0, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.last_exchange = exchange
            self.exchanges.append(exchange)
            raise
        self.last_exchange = exchange
        self.exchanges.append(exchange)
        return message

    def _consume_stream(self, response, on_delta):
        """Reassemble tool arguments before execution; only public content is streamed."""
        from openai.types.chat import ChatCompletionMessage

        content, tool_calls, metadata = [], {}, {}
        with response:
            for chunk in response:
                for key, attribute in (("request_id", "id"), ("response_model", "model"),
                                       ("system_fingerprint", "system_fingerprint"), ("usage", "usage")):
                    value = getattr(chunk, attribute, None)
                    if value is not None:
                        metadata[key] = self._model_dump(value)
                for choice in chunk.choices:
                    if choice.index != 0:
                        continue
                    if choice.finish_reason:
                        metadata["finish_reason"] = choice.finish_reason
                    delta = choice.delta
                    if delta.content:
                        content.append(delta.content)
                        on_delta(delta.content)
                    for fragment in delta.tool_calls or []:
                        call = tool_calls.setdefault(fragment.index, {
                            "id": "", "type": "function", "function": {"name": "", "arguments": ""},
                        })
                        if fragment.id:
                            call["id"] = fragment.id
                        if fragment.function:
                            call["function"]["name"] += fragment.function.name or ""
                            call["function"]["arguments"] += fragment.function.arguments or ""
        if not metadata.get("finish_reason"):
            raise RuntimeError("LLM stream ended before completion")
        return ChatCompletionMessage(role="assistant", content="".join(content) or None,
            tool_calls=[tool_calls[index] for index in sorted(tool_calls)] or None), metadata

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        response_format: Mapping[str, Any] | None = None,
    ) -> str:
        """单轮对话，返回纯文本（无工具调用的最简形态）。"""

        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        message = self.complete_messages(messages, response_format=response_format)
        return message.content or ""

    # ------------------------------------------------------ 基于证据的最终诊断
    @staticmethod
    def diagnosis_response_format() -> dict[str, Any]:
        """将输出模型的 Pydantic schema 包装为 ``response_format`` 参数。

        schema 直接取自 ``LLMDiagnosis.model_json_schema()``，与本地校验使用同一
        模型定义，避免服务端约束与本地校验口径不一致。

        该参数为附加约束而非替代：即使服务端限制输出结构，本地仍执行 Pydantic 与
        证据引用校验。服务端仅能约束字段结构，无法保证引用指向真实获取的证据。
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "bearing_diagnosis",
                "strict": True,
                "schema": LLMDiagnosis.model_json_schema(),
            },
        }

    def diagnose_evidence(self, evidence: EvidenceBundle) -> LLMDiagnosis:
        """读取工具证据并给出最终诊断；不接收确定性基线，避免锚定。

        单轮结构化抽取模式：不提供工具，一次性传入证据并要求直接输出结构化诊断。
        该模式既是工具型 Agent 的兼容路径，也作为对照实现：两者共用同一 policy、
        输出契约与 Pydantic 模型，结论口径可比。

        不向本方法传入确定性基线的原因：模型若获知基线结论将倾向于沿用该结论，
        使复核退化为复述，两路一致性失去独立证据的意义。即去锚定设计：
        基线仅作为对照字段保存，不进入提示词。

        降级策略：优先请求最强结构化输出；端点运行时不支持时关闭该开关并改用
        提示词约定 JSON，且仅降级一次。异常处理中 ``response_format is None``
        分支直接抛出，表明失败与输出格式无关，重试无意义，
        由上层回退基线。即仅对存在成功理由的情况重试。
        """

        evidence_payload = evidence.to_llm_payload()
        prompt = (
            "请依据本次精简工具证据给出最终诊断。解释中保留支持结论的关键测量和实际限制。\n"
            "本次工具证据 JSON：\n"
            + json.dumps(evidence_payload, ensure_ascii=False, separators=(",", ":"))
        )
        system = EVIDENCE_DIAGNOSIS_SYSTEM
        response_format = (
            self.diagnosis_response_format()
            if getattr(self, "supports_strict_output", False)
            else None
        )
        try:
            raw = (
                self.chat(prompt, system=system, response_format=response_format)
                if response_format is not None
                else self.chat(prompt, system=system)
            )
        except Exception:
            if response_format is None:
                raise
            # 部分兼容端点未实现严格 JSON Schema；记录失败请求后降级为提示词约定 JSON。
            # 关闭开关后后续调用不再尝试该路径，避免重复产生必然失败的请求。
            self.supports_strict_output = False
            raw = self.chat(prompt, system=system)
        payload = _parse_json(raw)
        if payload is None:
            raise ValueError("LLM 未返回合法 JSON 对象")
        # 仅执行结构校验，不作业务裁决：引用是否存在、结论与证据是否一致由编排层
        # 调用 validate_evidence_references 判断。
        return LLMDiagnosis.model_validate(payload)

    def repair_diagnosis(
        self,
        evidence: EvidenceBundle,
        validation_error: str,
        *,
        invalid_output: str | None = None,
    ) -> LLMDiagnosis:
        """执行一次结构、一致性与引用修复，不接收确定性基线。

        结构化输出中模型可能遗漏字段或违反字段互斥关系，直接丢弃整轮结果代价过高，
        故设置修复环节。修复须有界且受约束，本实现以四条规则约束：

          1. 仅修复一次：次数由编排层控制，避免反复修复直至偶然通过；
          2. 反馈校验错误：原样传入 ``validation_error``，使模型明确修改目标；
          3. 提供证据 ID 白名单：``allowed_evidence_ids`` 限定可引用范围，
             避免模型为通过校验而构造引用；
          4. 不传入基线：防止修复由结构调整变为结论替换，
             该情形会以校验通过的形式掩盖结论变更。

        返回前同样仅执行结构校验；引用与业务一致性由调用方通过
        ``runtime.validate_final`` 复核，修复产物与首次产物均须重新校验。
        """

        compact = evidence.to_llm_payload()
        previous = invalid_output
        if previous is None and self.last_exchange:
            previous = str(self.last_exchange.get("raw_text") or "")
        prompt = "请修复以下校验错误，仅返回修复后的诊断 JSON。输入数据：\n" + json.dumps(
            {
                "validation_error": validation_error,
                "allowed_evidence_ids": sorted(evidence.evidence_ids()),
                "previous_output": previous or "",
                "evidence": compact,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response_format = (
            self.diagnosis_response_format()
            if getattr(self, "supports_strict_output", False)
            else None
        )
        system = REPAIR_DIAGNOSIS_SYSTEM
        try:
            raw = self.chat(
                prompt,
                system=system,
                response_format=response_format,
            )
        except Exception:
            if response_format is None:
                raise
            # 与 diagnose_evidence 一致的"仅降级一次"策略。
            self.supports_strict_output = False
            raw = self.chat(prompt, system=system)
        payload = _parse_json(raw)
        if payload is None:
            raise ValueError("LLM 修复后仍未返回合法 JSON 对象")
        return LLMDiagnosis.model_validate(payload)

    # ------------------------------------------------------------- 解释
    # 以下两个方法属于早期解释链，仅作兼容保留。其形态代表一类受限的 Agent 设计：
    #   * 结论由确定性规则计算，模型仅负责转换为自然语言，
    #     提示词中明确禁止修改故障类型与置信度；
    #   * 模型不参与判读，仅承担表达功能；
    #   * 输出为纯文本，程序无法校验结论是否被修改，仅能依赖提示词约束，
    #     而提示词约束不具备强制力。
    # 新链路（agentic / rag）改为模型判读、结构化输出与本地校验相结合，
    # 使结论有证据可查、偏离可被拒绝。该演进方向可归纳为：
    # 凡需模型承担判断的环节，均须提供可校验的输出结构。
    @staticmethod
    def _evidence_lines(
        fault_frequencies_hz: Mapping[str, float], assessments: Mapping[str, Mapping[str, Any]]
    ) -> list[str]:
        """将逐族证据压缩为若干行文本，供旧解释链构造上下文。

        此处仅选取基频、命中频带数与谐波阶次三项，以保证解释具备依据。
        该手工挑选字段的方式在新链路中由 ``EvidenceBundle.to_llm_payload()`` 取代，
        后者具备明确的精简规则与 evidence_id 追溯能力。
        """
        return [
            f"- {fam}: 基频 {fault_frequencies_hz.get(fam, 0):.2f} Hz, "
            f"在 {assessments.get(fam, {}).get('evidence_bands', assessments.get(fam, {}).get('fundamental_bands', 0))} 个频带形成一致谱列, "
            f"谐波阶次 {assessments.get(fam, {}).get('qualified_orders', [])}"
            for fam in ("BPFO", "BPFI", "BSF")
        ]

    def explain_verdict(
        self,
        verdict: Mapping[str, Any],
        fault_frequencies_hz: Mapping[str, float],
        assessments: Mapping[str, Mapping[str, Any]],
    ) -> str:
        """读取确定性结论与逐族证据，生成自然语言解释，不重新判定结论。

        提示词中"禁止修改故障类型、置信度或给出不同诊断结论"是该类仅承担表达职责的
        Agent 的必要约束。其局限在于该约束属于请求而非可执行限制，
        模型实际修改结论时程序无法检测。可靠方案是要求模型输出结构化字段并逐项比对。
        """

        prompt = (
            f"确定性结论：fault_type={verdict['fault_type']}，"
            f"leading_family={verdict['leading_family']}，置信度 {verdict['confidence']:.2f}。\n"
            f"依据：{verdict['reason']}\n\n"
            "逐族证据：\n" + "\n".join(self._evidence_lines(fault_frequencies_hz, assessments))
        )
        return _parse_text(self.chat(prompt, system=LEGACY_EXPLANATION_SYSTEM)) or ""

    def explain_with_context(
        self,
        verdict: Mapping[str, Any],
        fault_frequencies_hz: Mapping[str, float],
        assessments: Mapping[str, Mapping[str, Any]],
        feature_description: str,
        similar_cases: Sequence[Mapping[str, Any]] = (),
        retrieval_scope: str = "",
    ) -> str:
        """「RAG 解释」：读确定性结论 + 时频特征 + 同类参考案例，写解释。不重算结论。"""

        scope_note = f"（{retrieval_scope}）" if retrieval_scope else ""
        if similar_cases:
            case_lines = "\n".join(
                f"  - {c.get('case_id')}: {c.get('fault_type')}（相似度 {c.get('score', 0):.2f}）"
                for c in similar_cases
            )
        else:
            case_lines = "  （无同类历史案例）"

        prompt = (
            f"确定性结论：fault_type={verdict['fault_type']}，"
            f"leading_family={verdict['leading_family']}，置信度 {verdict['confidence']:.2f}。\n"
            f"依据：{verdict['reason']}\n\n"
            "逐族证据：\n" + "\n".join(self._evidence_lines(fault_frequencies_hz, assessments)) + "\n\n"
            f"【信号时频特征】\n{feature_description}\n\n"
            f"【同类历史案例（背景参考）】{scope_note}\n{case_lines}"
        )
        return _parse_text(self.chat(prompt, system=LEGACY_EXPLANATION_SYSTEM)) or ""
