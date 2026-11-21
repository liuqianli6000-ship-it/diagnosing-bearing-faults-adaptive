"""受约束的工具调用型诊断运行时。

本模块实现 Agent 主循环，不依赖 LangChain / LlamaIndex 等框架，循环所需的各项机制
均显式实现。

对话式 Agent 的最小构成：
  1) 模型：读取消息并决定调用工具或给出最终答案；
  2) 工具：具备名称、JSON Schema 参数与本地执行函数；
  3) 循环：串联"模型决策 → 本地执行 → 结果回填"，直至模型给出答案或预算耗尽。

Chat Completions 的 tool calling 协议围绕上述三者设计，单轮流程为：
    发送 messages 与 tools → 模型返回 tool_calls（仅含工具名与 JSON 参数字符串）
    → 本地执行 → 追加 role="tool" 结果消息 → 再次请求
模型不执行任何代码，其参数构成仅属建议；是否执行、如何校验、是否拒绝，
均由本模块在本地判定。

四项核心设计：

(1) 能力渐进开放（progressive tool exposure），见 ``_tools()``
    每轮暴露的工具集合动态变化：先仅提供 ``read_diagnosis_skill``，读取后提供
    ``collect_core_evidence``，核心证据就绪后才开放可选工具；预算不足时返回空列表，
    使模型只能输出最终答案。工具可见性构成硬约束，其可靠性高于提示词约定。

(2) 状态机驱动流程，见 ``agent_state.AgentState``
    每次运行持有可序列化状态对象，包含阶段、已获取证据、未决疑点、工具调用记录与
    剩余预算。流程合法性由状态字段判定。

(3) 证据引用闭包，见 ``validate_final()`` 与 ``_details()``
    最终 ``evidence_refs`` 必须属于本次实际获取的 evidence_id，否则判定失败，
    使"结论可追溯至具体测量"由程序强制保证。

(4) 失败显式终止，见 ``AgentRuntimeStopped``
    可修复错误（JSON 非法、引用越界）抛出 ValueError，由上层决定修复或回退；
    无效运行（质量门不通过、预算耗尽）抛出 ``AgentRuntimeStopped`` 并记录
    ``termination_reason``，禁止调用方以新的自由循环绕开限制。

相关概念：
* 审计事件（``self.events``）：记录循环内每个可观察动作，支持复现模型当轮的
  输入与行为，是可观测性的主要来源。
* 预算：模型轮数、工具调用次数、无进展次数与单工具耗时上限。
* 无进展检测：模型重复调用同一工具或仅命中缓存时累加计数，达到阈值即收回工具权限。
* 疑点（OpenQuestion）：以结构化摘要替代模型的隐藏推理，同时使取证具有指向性。
* 观察掩码：工具原始输出通常较大，仅回填精简的 ``task_state.view()``，
  避免上下文被历史结果占满。

本模块的一条基本约束：模型可被约束，但不被信任，全部校验均在本地执行。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .diagnosis_schemas import EvidenceBundle, LLMDiagnosis, validate_evidence_references
from .agent_state import AgentState, OpenQuestion, ToolExecution
from .agent_tools import (ToolInput, OptionalInput, DetailsInput, VisualInput, ToolSpec,
                          SkillReadInput, ToolResult, ToolExecutor, record_ids, require_keys)
from .skill_loader import DiagnosisSkillLoader
from pydantic import ValidationError
from uuid import uuid4
from .prompts import tool_agent_system_prompt


# =============================================================================
# 一、兼容性辅助函数
# =============================================================================
# 不同供应商的返回形态存在差异：即使要求严格 JSON 输出，模型仍可能附加 Markdown
# 代码块或说明文字，也可能返回 SDK 对象而非 dict。以下适配层集中收敛这些差异，
# 使主循环只需处理统一形状。


def _parse_object(text: str | None) -> dict[str, Any] | None:
    """将模型返回的文本解析为 dict；解析失败返回 None，不抛出异常。

    三层降级：
      1) 直接 ``json.loads``；
      2) 截取首个 ``{`` 至末个 ``}`` 之间的片段后解析，容忍 Markdown 代码块
         与说明性前后缀；
      3) 均失败则返回 None，由调用方（``_run``）转为明确的失败信号，
         交由上层编排决定修复或回退。

    解析成功时仍须确认结果为 dict：模型可能返回数组或字符串。
    """
    if not text:
        return None
    value = text.strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(value[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _tool_calls(message: Any) -> list[Any]:
    """取出本轮模型请求的工具调用列表，兼容 SDK 对象与 dict。"""
    if isinstance(message, Mapping):
        value = message.get("tool_calls")
    else:
        value = getattr(message, "tool_calls", None)
    return list(value or [])


def _message_content(message: Any) -> str:
    """取出 assistant 消息的文本内容，缺失时统一为空串。

    工具调用轮的 content 常为 None 或空串，统一为空串后，``if content``
    等判断无需区分 None 与字段缺失。该取值同时用于最终答案判定：
    本轮无工具调用且含文本时视为模型已给出结论（见 ``_run``）。
    """
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(getattr(message, "content", None) or "")


def _tool_call_parts(call: Any) -> tuple[str, str, str]:
    """将一次工具调用拆分为 (call_id, name, arguments 原始字符串)。

    三者用途互不相同：
      * ``call_id``：回填 tool 结果消息时必须原样携带，供应商据此配对请求与结果，
        不匹配会导致请求被拒绝；
      * ``name``：模型请求的工具名，用于在 registry 中查找；
      * ``arguments``：尚未解析的 JSON 字符串，属模型生成的自由文本，
        必须经 ``json.loads`` 与 Pydantic 校验后方可作为参数使用。
    """
    if isinstance(call, Mapping):
        call_id = str(call.get("id") or "")
        function = call.get("function", {})
        return (
            call_id,
            str(function.get("name") or ""),
            str(function.get("arguments") or "{}"),
        )
    function = getattr(call, "function", None)
    return (
        str(getattr(call, "id", "")),
        str(getattr(function, "name", "")),
        str(getattr(function, "arguments", "{}")),
    )


def _assistant_message(message: Any) -> dict[str, Any]:
    """将助手回复转换为可回填至后续请求的 dict。

    Chat Completions 无服务端会话状态，须由调用方将带 tool_calls 的 assistant
    消息与对应的 tool 结果消息一并回填，否则会破坏
    assistant(tool_calls) → tool(result) 的配对关系。

    此处将三种形态（SDK 对象 / dict / 兼容对象）统一为 dict，并重建标准
    ``tool_calls`` 结构：
        {"id": ..., "type": "function", "function": {"name": ..., "arguments": "..."}}
    ``exclude_none=True`` 用于清理空字段，部分端点不接受 ``content: null`` 与
    ``tool_calls: null`` 同时出现。
    """
    if hasattr(message, "model_dump"):
        payload = message.model_dump(mode="json", exclude_none=True)
        return dict(payload)
    if isinstance(message, Mapping):
        return dict(message)
    payload: dict[str, Any] = {
        "role": "assistant",
        "content": _message_content(message) or None,
    }
    calls = _tool_calls(message)
    if calls:
        normalized = []
        for call in calls:
            call_id, name, arguments = _tool_call_parts(call)
            normalized.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        payload["tool_calls"] = normalized
    return payload


class AgentRuntimeStopped(ValueError):
    """有意结束的受限运行；调用者不得再启动独立模型循环绕过限制。

    主动终止与可修复错误分为两类异常：
      * 可修复错误（JSON 非法、引用未获取的证据）说明本轮输出不合格，
        上层可尝试修复一次或退回确定性基线；
      * 主动终止（质量门判定不应诊断、轮数与工具预算耗尽）表明继续运行无意义，
        新开循环只会绕开既有的安全边界。

    上层（``agent._run_constrained_agent``）据此将本异常特判为
    ``status="limited"`` 并直接回退基线，不进入修复流程。
    """


@dataclass
class AgentRuntimeResult:
    """单次运行的产出：最终诊断、审计事件与终止状态快照。

    三者分开返回：``diagnosis`` 供业务使用，``events`` 与 ``state`` 构成过程记录，
    用于事后排障、复现与 Agent 行为评测。
    """

    diagnosis: LLMDiagnosis
    events: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)


class ConstrainedDiagnosisRuntime:
    """固定核心步骤与围绕疑点选择的可选工具共用一个运行状态。

    调用方式：
        runtime = ConstrainedDiagnosisRuntime(llm, evidence, ...)
        result  = runtime.run()
        runtime.validate_final(result.diagnosis)

    ``run()`` 的生命周期（对应 ``_run`` 的循环体）：
        1) 组装 system 与 user 消息，system 中声明预算与可用工具；
        2) 逐轮请求模型，本轮暴露的工具由 ``_tools()`` 依据状态决定；
        3) 返回 tool_calls 时本地逐个执行，将结果作为 role="tool" 消息回填后进入下一轮；
        4) 无 tool_calls 且有文本时视为最终答案，依次执行 JSON 解析、Pydantic 校验
           与证据引用校验，通过则返回，失败则抛出异常；
        5) 轮数耗尽仍未给出结论时抛出 ``AgentRuntimeStopped``（budget_exhausted）。

    实例为主循环与状态对象的组合，不可重复 ``run()``
    （见 ``run`` 开头的 ``runtime_already_started`` 检查），
    否则审计事件与状态将相互混杂。新一次诊断须新建实例。
    """

    def __init__(
        self, llm: Any, evidence: EvidenceBundle | None = None, *, max_tool_calls: int = 4,
        max_turns: int = 3,
        evidence_provider: Callable[[], EvidenceBundle] | None = None,
        skill_loader: DiagnosisSkillLoader | None = None,
        require_skill_first: bool = False,
        numeric_only: bool = False,
        visual_observer: Callable[[dict[str, Any], EvidenceBundle], Mapping[str, Any]] | None = None,
        compare_normal_psd: Callable[[], Mapping[str, Any]] | None = None,
        retrieve_balanced_cases: Callable[[], Mapping[str, Any]] | None = None,
        progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
        run_id: str | None = None, sample_id: str | None = None,
        max_no_progress_actions: int = 2, tool_timeout_seconds: float = 30.0,
    ) -> None:
        """构造运行时。

        参数按用途分为四组：

        A. 预算类，决定循环的终止条件
           ``max_tool_calls``：整轮运行的工具调用次数上限；
           ``max_turns``：模型响应轮数上限，最小为 2，须保留一轮用于输出结论；
           ``max_no_progress_actions``：连续无进展次数上限，达到后收回工具权限；
           ``tool_timeout_seconds``：单次工具耗时上限，超限仅标记不中断（见下）。

        B. 依赖注入类，决定运行时可见的事实来源
           ``evidence`` / ``evidence_provider``：核心证据包，可立即提供，也可延迟至
             模型调用 ``collect_core_evidence`` 时计算（懒加载，避免过早占用算力
             与上下文）；
           ``llm``：仅需满足 ``complete_messages`` 协议，测试中可注入替身对象。

        C. 可选工具类，决定围绕疑点可查询的补充证据
           三个 ``Callable`` 分别对应正常性 PSD 对比、平衡案例检索与图像观察。
           传入 None 表示该工具不存在：既不出现于模型可见的 schema，也不登记于本地
           registry。按能力开关工具比由工具返回"不支持"更安全，
           可消除模型对不存在工具的调用。

        D. 观测类
           ``progress_callback``：向 UI 与日志推送循环内部事件，回调异常被忽略
             （见 ``_emit_progress``），观测失败不影响诊断；
           ``run_id`` / ``sample_id``：审计关联标识，写入每次工具调用的 provenance。

        ``require_skill_first``：要求先读取 Skill 再获取核心证据。传入
        ``skill_loader`` 时自动开启，因流程顺序影响结论可比性。
        ``numeric_only``：双路模式下的数值分支标识，仅影响提示词表述，
        避免该分支代为陈述视觉结论。
        """
        self.llm, self.evidence = llm, evidence
        self.evidence_provider = evidence_provider
        self.skill_loader = skill_loader
        self.require_skill_first = bool(require_skill_first or skill_loader is not None)
        self.numeric_only = numeric_only
        # 缺少证据来源时拒绝构造，使配置错误在启动阶段暴露。
        if self.evidence is None and self.evidence_provider is None:
            raise ValueError("evidence_or_evidence_provider_required")
        # 预算参数收敛至有效下界，避免非法取值破坏循环语义。
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.max_turns = max(2, int(max_turns))
        self.max_no_progress_actions = max(1, int(max_no_progress_actions))
        self.visual_observer = visual_observer
        self.progress_callback = progress_callback
        self.events: list[dict[str, Any]] = []
        self.last_output: str | None = None
        # 状态对象持有全部流程合法性依据，审计时完整落盘。
        self.state = AgentState(remaining_model_turns=self.max_turns,
                                remaining_tool_attempts=self.max_tool_calls)
        if run_id:
            self.state.run_id = run_id
        if sample_id:
            self.state.sample_id = sample_id
        # 执行器负责缓存、重试、超时与输出合法性，与调用策略分离。
        self.executor = ToolExecutor(self.state.run_id)
        # registry 仅登记本次运行可用的工具，schema 与本地校验同源。
        self.registry: dict[str, ToolSpec] = {}

        def register(name, description, input_model, handler, kinds=(), keys=()):
            """登记工具至 registry。

            四个要素分别对应工具定义的不同方面：
              1) ``description``：工具用途，供模型判断调用时机；
              2) ``input_model``：可接受的参数，Pydantic 模型同时产出 JSON Schema；
              3) ``handler``：本地实现，模型无法影响其行为；
              4) ``kinds`` 与 ``keys``：本地反向约束，限定该工具可响应的疑点类型，
                 并要求返回的 data 包含指定字段（由 ``require_keys`` 校验）。

            ``kinds`` 为空表示该工具不接受疑点参数，属流程必需步骤；
            非空表示其为可选工具，模型须说明调用理由。
            """
            self.registry[name] = ToolSpec(
                name, description, input_model, handler, frozenset(kinds),
                timeout_seconds=tool_timeout_seconds, output_validator=require_keys(*keys),
            )

        # 流程第一步：读取 Skill（受控流程规范摘要），仅在启用时登记。
        if self.require_skill_first:
            register("read_diagnosis_skill", "首轮读取当前诊断 Skill 的适用范围和固定分析流程。",
                     SkillReadInput, self._read_skill, keys=("skill_id", "workflow_steps"))
        # 流程第二步：核心证据。同为必需步骤，不接受疑点参数。
        register("collect_core_evidence", "读取 Skill 流程要求的基础证据；只在 Skill 已读取后执行。",
                 ToolInput, self._core, keys=("assessments", "evidence_catalog"))
        # 以下为围绕疑点的可选工具，allowed_kinds 限定各自可响应的疑点类型。
        register("get_evidence_details", "针对频族竞争或峰值疑点读取目录中的1至6条记录，不重新计算。",
                 DetailsInput, self._details, ("family_conflict", "peak_verification"), ("evidence",))
        if compare_normal_psd is not None:
            register("compare_normal_psd", "存在正常性疑点时实际查询可比健康PSD参考，参考不足返回unavailable。",
                     OptionalInput, lambda _: self._mapping_result(compare_normal_psd()),
                     ("normality_uncertain",), ("status", "evidence_id"))
        if retrieve_balanced_cases is not None:
            register("retrieve_balanced_cases", "针对疑点实际检索BPFO/BPFI/BSF/NONE案例背景，不代替当前测量。",
                     OptionalInput, lambda _: self._mapping_result(retrieve_balanced_cases()),
                     ("family_conflict", "normality_uncertain", "peak_verification"), ("similar_cases",))
        if visual_observer is not None and evidence is not None and evidence.visual_evidence:
            register("inspect_visual_evidence", "针对可见谱形疑点观察指定图像，不独立裁决。",
                     VisualInput, self._visual, ("visual_verification", "family_conflict"),
                     ("observable", "observations", "conflicts", "limitations"))

    @staticmethod
    def _mapping_result(value: Mapping[str, Any]) -> ToolResult:
        """将可选工具返回的 dict 包装为 ToolResult，并判定是否存在可用参考。

        两种状态的语义区分贯穿整个运行时：
          * ``ok``：调用成功且包含可用于诊断的证据；
          * ``unavailable``：调用成功但客观无参考数据（案例库为空、无可比健康 PSD）。
            该结果不构成证据，不进入 ``observed_evidence_ids``，模型不得据此支持结论。
        区分"无数据"与"有数据"使 unknown 结论的归因明确：可判定为证据矛盾或缺少参考。
        """
        if not isinstance(value, Mapping):
            raise ValueError("tool_result_must_be_object")
        data = dict(value)
        unavailable = data.get("status") == "unavailable" or (
            "similar_cases" in data and not data["similar_cases"])
        return ToolResult(status="unavailable" if unavailable else "ok", data=data)

    def _emit_progress(self, event: str, **details: Any) -> None:
        """向外部观察者推送进度事件，回调异常一律忽略。

        可观测性属旁路能力，不应因 UI 或日志故障导致诊断失败。
        与工具执行路径不同：后者将异常收敛为结构化失败结果，此处直接忽略。
        """
        if self.progress_callback:
            try:
                self.progress_callback(event, details)
            except Exception:
                pass

    def _tools(self) -> list[dict[str, Any]]:
        """计算本轮暴露给模型的工具 schema，是能力渐进开放的唯一入口。

        工具可见性按状态收敛：
          1) 未读取 Skill：仅提供 ``read_diagnosis_skill``；
          2) 已读取 Skill 但无核心证据：仅提供 ``collect_core_evidence``；
          3) 核心证据就绪：开放 registry 中的全部工具；
          4) 已完成的流程工具从目录中移除，减少 token 开销并避免重复调用；
          5) 预算不足（剩余一轮、工具次数耗尽、连续无进展超阈值）时返回空列表。
             空工具列表为合法请求，模型因无法调用任何工具而只能输出最终 JSON，
             该约束的可靠性高于提示词提示。

        返回的 schema 取自 Pydantic 的 ``model_json_schema()``，与本地校验共用同一
        模型定义，模型的参数契约与本地强制执行的契约始终一致。
        """
        if self.require_skill_first and not self.state.skill_loaded:
            names = ["read_diagnosis_skill"]
        elif not self.state.core_collected:
            names = ["collect_core_evidence"]
        else:
            names = list(self.registry)
        if self.state.core_collected:
            names = [name for name in names if name != "collect_core_evidence"]
        if self.state.skill_loaded:
            names = [name for name in names if name != "read_diagnosis_skill"]
        if (self.state.remaining_model_turns <= 1 or self.state.remaining_tool_attempts <= 0
                or self.state.no_progress_count >= self.max_no_progress_actions):
            names = []
        # supports_strict_tools 由客户端按供应商能力置位：支持时附加 strict 标记，
        # 由服务端同步校验参数；本地校验始终执行，两者为叠加关系。
        return [self.registry[name].schema(bool(getattr(self.llm, "supports_strict_tools", False)))
                for name in names]

    def _read_skill(self, _: SkillReadInput) -> ToolResult:
        """执行 ``read_diagnosis_skill``，将受控 Skill 摘要载入上下文并记录状态。

        采用渐进披露：流程规范不以长提示词形式常驻，而由模型按需读取，
        并在状态中记录 ``skill_loaded`` 与 ``workflow_steps`` 供后续步骤放行判断。
        """
        if self.skill_loader is None:
            return ToolResult(status="failed", error_code="skill_loader_unavailable")
        payload = self.skill_loader.read()
        self.state.skill_loaded = True
        self.state.skill_id = str(payload.get("skill_id") or "")
        self.state.workflow_steps = [str(step) for step in payload.get("workflow_steps", [])]
        return ToolResult(status="ok", data=payload)

    def _core(self, _: ToolInput) -> ToolResult:
        """执行 ``collect_core_evidence``，产出供模型判读的证据视图。

        工具返回值并非内部对象的直接导出，而是面向模型的一次上下文构造：

        (a) 顺序约束：未读取 Skill 即调用核心证据时返回 ``invalid_input``。
            该约束由 handler 内的条件判断实施，模型无法跳过流程步骤。

        (b) 懒加载：证据可由 ``evidence_provider`` 在此时计算，模型未执行到该步骤
            即不产生算力开销。provider 失败被转换为结构化 error_code 而非异常，
            使工具始终返回结果。

        (c) 观测掩码：在精简视图基础上移除 ``similar_cases``、``psd_normality`` 与
            ``feature_description`` 三个字段。三者体积较大且易引导判读偏离
            （案例标签会锚定结论，PSD 正常性易被当作主要依据），故默认不提供，
            仅在模型提出相应疑点时通过专门工具获取。

        (d) 证据目录：仅列出允许引用的 ID，被隐藏字段对应的 ID 从目录中剔除
            （``hidden``）。模型由此知晓可查询的证据范围但不获取其内容，
            获取内容须调用 ``get_evidence_details``，使引用成为有成本且可审计的动作。

        (e) 疑点生成：多个故障族同时达到 moderate/strong 时，运行时主动登记
            ``family_conflict`` 疑点。程序负责定义问题，模型决定以何种工具响应。
        """
        if self.require_skill_first and not self.state.skill_loaded:
            return ToolResult(status="invalid_input", error_code="read_diagnosis_skill_must_be_called_first")
        if self.evidence is None:
            if self.evidence_provider is None:
                return ToolResult(status="failed", error_code="core_evidence_provider_unavailable")
            try:
                self.evidence = self.evidence_provider()
            except Exception as exc:
                # 工具执行不向主循环传播异常，统一转换为 error_code，
                # 使主循环与审计只需处理 ToolResult 一种结构。
                return ToolResult(status="failed", error_code=f"core_evidence_failed:{type(exc).__name__}")
            if not isinstance(self.evidence, EvidenceBundle):
                return ToolResult(status="failed", error_code="core_evidence_provider_invalid_result")
            # 懒加载路径下需待证据就绪才能确定图像是否存在，故在此补登记。
            if self.visual_observer is not None and self.evidence.visual_evidence:
                self.registry["inspect_visual_evidence"] = ToolSpec(
                    name="inspect_visual_evidence",
                    description="针对可见谱形疑点观察指定图像，不独立裁决。",
                    input_model=VisualInput,
                    handler=self._visual,
                    allowed_kinds=frozenset({"visual_verification", "family_conflict"}),
                    output_validator=require_keys("observable", "observations", "conflicts", "limitations"),
                )
        payload = self.evidence.to_llm_payload()
        # 观测掩码：以空值占位并移除大体积字段，使初始上下文保留判读所需的聚合信息，
        # 同时排除可能引导判读偏离的辅助信息。
        payload["similar_cases"] = []
        payload["signal_features"].pop("psd_normality", None)
        payload["signal_features"].pop("feature_description", None)
        # 被移除内容的 evidence_id 属隐藏证据，不得出现在目录中，
        # 否则模型会索取不可见 ID 并消耗工具预算。
        hidden = record_ids(self.evidence.similar_cases) | record_ids(
            self.evidence.signal_features.get("psd_normality", {}))
        self.state.catalog_evidence_ids = sorted(self.evidence.evidence_ids() - hidden)
        payload["evidence_catalog"] = self.state.catalog_evidence_ids
        # 仅提供图像工件的 ID 与类型，内容由专门工具解析。
        payload["visual_artifacts"] = [
            {"artifact_id": item.get("evidence_id"), "kind": "envelope_spectrum"}
            for item in self.evidence.visual_evidence
        ]
        # 多个频族同时获得支持时，由程序登记待解释疑点。
        supported = [family for family, a in self.evidence.assessments.items()
                     if a.get("evidence_level") in {"moderate", "strong"}]
        if len(supported) > 1:
            self.state.questions.append(OpenQuestion(
                question_id="q1", kind="family_conflict",
                question="多个故障族同时得到支持：" + ",".join(supported),
                needed_evidence="核对候选频族的峰位、谐波与碰撞明细",
                evidence_refs=[f"assessment.{family}" for family in supported],
            ))
        self.state.core_collected = True
        return ToolResult(status="ok", data=payload)

    def _details(self, args: DetailsInput) -> ToolResult:
        """执行 ``get_evidence_details``，按 ID 取回指定证据明细。

        三项设计要点：

        1) 引用闭包白名单校验：仅允许索取 ``catalog_evidence_ids`` 中的 ID，
           其余返回 ``unknown_evidence_id``，模型无法通过构造 ID 获取数据。

        2) 只查询不重算：在已生成的证据树中检索并返回，因此可缓存、可重复调用，
           不改变测量结果（纯读取、幂等）。

        3) 嵌套截断（trim）：命中记录的子节点若自身带有 ``evidence_id``，
           替换为 ``{"detail_ref": ...}`` 引用。既保留层级结构，又避免单次调用
           将整棵子树载入上下文，是上下文预算管理在工具返回值上的实现。

        检索采用递归 ``visit`` 而非固定路径，使证据结构演进时无需改动此处，
        代价是每次调用遍历一次 JSON。
        """
        if self.evidence is None:
            return ToolResult(status="failed", error_code="core_evidence_not_collected")
        ids = set(args.evidence_ids)
        if not ids <= set(self.state.catalog_evidence_ids):
            return ToolResult(status="invalid_input", error_code="unknown_evidence_id")
        found: dict[str, Any] = {}

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                if value.get("evidence_id") in ids:
                    # 返回当前记录；嵌套记录保留目录引用，避免父节点展开整树。
                    def trim(item: Any) -> Any:
                        if isinstance(item, Mapping):
                            if item.get("evidence_id"):
                                return {"detail_ref": item["evidence_id"]}
                            return {k: trim(v) for k, v in item.items()}
                        if isinstance(item, list):
                            return [trim(v) for v in item]
                        return item
                    found[value["evidence_id"]] = {k: trim(v) for k, v in value.items()}
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)

        visit(self.evidence.model_dump(mode="json"))
        # 请求 N 条仅命中 M<N 条表明证据视图与目录不一致：返回 unavailable 而非残缺结果，
        # 避免模型将缺项理解为"此处无证据"。
        if ids != set(found):
            return ToolResult(status="unavailable", error_code="evidence_record_unavailable")
        return ToolResult(status="ok", data={"evidence": found})

    def _visual(self, args: VisualInput) -> ToolResult:
        """执行 ``inspect_visual_evidence``，将指定图像交由视觉观察回调处理。

        可选工具的标准形态：先校验工件真实存在，再委托注入的观察函数，
        最后统一包装为 ToolResult。观察函数返回 ``status=unavailable`` 时，
        ``_mapping_result`` 将其归为"无可引用证据"而非失败。
        """
        if self.evidence is None:
            return ToolResult(status="failed", error_code="core_evidence_not_collected")
        artifact = next((item for item in self.evidence.visual_evidence
                         if item.get("evidence_id") == args.artifact_id), None)
        if artifact is None:
            return ToolResult(status="invalid_input", error_code="unknown_artifact_id")
        return self._mapping_result(self.visual_observer(artifact, self.evidence))

    def _question(self, spec: ToolSpec, args: OptionalInput) -> OpenQuestion:
        """校验并解析本次可选调用绑定的疑点，是本地约束最集中的一环。

        模型调用工具的意图本身不构成充分理由，故要求以结构化字段说明依据。
        运行时据此执行四类校验，并生成可审计的行动记录：

          a) ``question_tool_mismatch``：疑点类型须在该工具声明的 ``allowed_kinds`` 内；
          b) ``unobserved_question_evidence``：疑点引用的证据须为模型已获取的证据，
             模型不得为未读取的证据发起取证；
          c) ``unknown_or_mismatched_question``：填写的 ``question_id`` 须存在且类型一致，
             即复用既有疑点而非新建；
          d) 幂等：按 (kind, question) 去重，重复登记返回既有对象，
             避免模型通过持续新建疑点掩盖无进展行为、绕过无进展检测。

        ``evidence_refs`` 随疑点保存，用于后续判定疑点是否解决，并进入审计材料。
        """
        purpose = args.purpose
        if purpose.kind not in spec.allowed_kinds:
            raise ValueError("question_tool_mismatch")
        if not set(purpose.evidence_refs) <= set(self.state.observed_evidence_ids):
            raise ValueError("unobserved_question_evidence")
        if purpose.question_id is not None:
            question = next((q for q in self.state.questions if q.question_id == purpose.question_id), None)
            if question is None or question.kind != purpose.kind:
                raise ValueError("unknown_or_mismatched_question")
            return question
        # 相同问题不能靠重复创建来隐藏无进展行为。
        for question in self.state.questions:
            if question.kind == purpose.kind and question.question == purpose.question:
                return question
        question = OpenQuestion(question_id=f"q{len(self.state.questions) + 1}",
                                **purpose.model_dump(exclude={"question_id"}))
        self.state.questions.append(question)
        return question

    def _execute_call(self, name: str, raw: str, *, core_at_start: bool,
                      tools_allowed: bool) -> dict[str, Any]:
        """执行单次工具调用，返回回填给模型的结果对象。

        执行顺序为逐层收窄的校验链，每步失败均转换为结构化 ``error_code`` 而非异常：
        工具结果属正常对话内容，仅主循环层面的错误需要抛出异常：

          1) ``tool_budget_exhausted``：工具预算已用尽；
          2) ``final_response_required``：本轮工具列表为空（``_tools()`` 已收回权限），
             模型仍请求工具时予以拒绝；
          3) ``unknown_tool``：模型请求了不存在的工具；
          4) ``json.loads`` 与 Pydantic 校验：参数须为合法 JSON 且符合该工具的契约
             （``invalid_tool_arguments``）；
          5) 顺序约束：核心证据未就绪时禁止调用可选工具；
          6) 疑点校验（``_question``）：可选工具须绑定合法疑点；
          7) 交由 ``ToolExecutor``：缓存、重试、超时与输出字段校验。

        执行后进行三项状态更新，共同构成运行时的约束：
          * 将结果中的 evidence_ids 合并至 ``observed_evidence_ids``
            （仅 ``status=ok`` 计为已获取，``unavailable`` 不计）；
          * 更新疑点状态（blocked / evidence_received / resolved）；
          * 更新无进展计数：仅当取得新数据（ok、非缓存命中且 data 非空）计为有进展。
            缓存命中不计，超时则直接置满计数，使重复空转与工具阻塞均能及时终止。

        返回对象的两项约定：
          * 证据字段位于顶层，执行元数据置于 ``_tool`` 子对象。前者供模型使用，
            后者为运行时记录，分离可避免模型将 ``cache_hit`` 等元数据当作诊断依据；
          * 附加 ``task_state``（不含完整执行历史的精简状态视图），
            使模型每轮获知剩余预算与未决疑点，从而自行安排输出时机。
        """
        execution_id = uuid4().hex
        arguments: Any = {}
        question = None
        key, elapsed = None, 0.0
        spec = self.registry.get(name)
        result = ToolResult(status="invalid_input", error_code="unknown_tool")
        try:
            if self.state.remaining_tool_attempts <= 0:
                raise ValueError("tool_budget_exhausted")
            self.state.remaining_tool_attempts -= 1
            if not tools_allowed:
                raise ValueError("final_response_required")
            if spec is None:
                raise ValueError("unknown_tool")
            arguments = json.loads(raw)
            # 保留旧供应商与旧测试对无参数视觉调用的兼容路径；正式请求仍由 schema
            # 强制携带 purpose，自动补全结果记录于事件中以便迁移观测。
            if name == "inspect_visual_evidence" and arguments == {}:
                artifact_id = str(self.evidence.visual_evidence[0].get("evidence_id", ""))
                refs = [ref for ref in ("assessment.BPFO", "assessment.BPFI", "assessment.BSF")
                        if ref in self.state.observed_evidence_ids]
                arguments = {"artifact_id": artifact_id, "purpose": {
                    "question_id": None, "kind": "visual_verification",
                    "question": "需要核对当前谱图中的可见谐波结构",
                    "needed_evidence": "图像观察及其对应数值证据",
                    "evidence_refs": refs[:6] or ["quality.signal"],
                }}
            args = spec.input_model.model_validate(arguments)
            # 上一行同时完成严格校验（extra="forbid" 拒绝多余字段）与类型转换；
            # 模型参数仅通过此处后才会进入 handler。
            # 兼容旧调用方直接请求核心工具的情形，隐式补读 Skill；
            # 新协议下模型收到的工具目录仅含 read_diagnosis_skill，会先显式读取。
            if (self.require_skill_first and not self.state.skill_loaded
                    and name == "collect_core_evidence"):
                self._read_skill(SkillReadInput())
            if not core_at_start and name not in {"collect_core_evidence", "read_diagnosis_skill"}:
                raise ValueError("collect_core_evidence_must_be_called_first")
            if isinstance(args, OptionalInput):
                question = self._question(spec, args)
            self.state.transition("gathering_evidence")
            result, key, elapsed = self.executor.execute(spec, args)
        except (ValidationError, json.JSONDecodeError):
            # 参数非法 JSON 或不符合契约，属模型可自行纠正的错误，
            # 因此返回错误结果而非抛出异常。
            result = ToolResult(status="invalid_input", error_code="invalid_tool_arguments")
        except ValueError as exc:
            # 各校验环节统一以 ValueError 表达拒绝，error_code 即异常消息，
            # 新增校验项时无需扩展异常类型映射。
            result = ToolResult(status="invalid_input", error_code=str(exc))
        finally:
            # 无论成败均从"取证中"回到"分析中"，避免阶段机停留在中间态。
            if self.state.phase == "gathering_evidence":
                self.state.transition("analyzing")
        before = set(self.state.observed_evidence_ids)
        # unavailable 仅表示已尝试但无可用参考，不构成可引用的诊断证据。
        if result.status == "ok":
            self.state.observed_evidence_ids = sorted(before | set(result.evidence_ids))
            self.state.catalog_evidence_ids = sorted(set(self.state.catalog_evidence_ids) | set(result.evidence_ids))
        if question:
            question.result_evidence_ids = sorted(set(question.result_evidence_ids) | set(result.evidence_ids))
            if result.status in {"unavailable", "failed"}:
                # 疑点受阻：并非已解决，而是确认所需参考不可得。
                # 该状态进入审计材料，用于说明结论限定为 unknown 的原因。
                question.status = "blocked"
                question.resolution = result.error_code or "required_reference_unavailable"
            elif result.status == "ok" and not result.cache_hit:
                question.status = "evidence_received"
                question.resolution = "已收到补充结果，尚需核对是否消除疑点"
                # 运行时唯一可确认"已解决"的情形：峰值核对工具已送达所请求的明细；
                # 其余情况由模型在最终诊断中说明。
                if question.kind == "peak_verification" and name == "get_evidence_details":
                    question.status = "resolved"
                    question.resolution = "requested_numeric_details_delivered"
        # 进展判定：仅取得新数据计为有进展，缓存命中、空数据与失败均不计。
        progress = result.status == "ok" and not result.cache_hit and bool(result.data)
        self.state.no_progress_count = 0 if progress else self.state.no_progress_count + 1
        if result.provenance.get("deadline_exceeded"):
            # 工具超出单次耗时上限：直接置满无进展计数，下一轮收回工具权限。
            # 同步工具无法强制中断，此处采取返回后立即止损的策略。
            self.state.no_progress_count = self.max_no_progress_actions
        execution = ToolExecution(
            execution_id=execution_id, tool_name=name, tool_version=spec.version if spec else "unknown",
            question_id=question.question_id if question else None,
            # 审计保留原始参数；参数非法时亦如实记录，便于复盘模型的请求内容。
            arguments=arguments if isinstance(arguments, dict) else {"invalid_value": arguments},
            status=result.status, error_code=result.error_code, retryable=result.retryable,
            cache_key=key, cache_hit=result.cache_hit, elapsed_seconds=elapsed,
            evidence_ids=result.evidence_ids,
        )
        self.state.tool_executions.append(execution)
        # 证据字段保持在顶层供模型直接使用；执行元数据置于 _tool 字段，
        # 避免将运行时字段混入证据 data。
        output: dict[str, Any] = dict(result.data)
        output["_tool"] = {
            "status": result.status,
            "error_code": result.error_code,
            "retryable": result.retryable,
            "cache_hit": result.cache_hit,
            "evidence_ids": result.evidence_ids,
            "execution_id": execution_id,
            "question_id": execution.question_id,
        }
        output["task_state"] = self.state.view()
        return output

    def validate_final(self, diagnosis: LLMDiagnosis) -> None:
        """最终答案的兜底校验：引用闭包与证据一致性。

        两层检查分工不同：
          1) 引用闭包（前半段）：模型引用的每个 evidence_id 须属于本次实际获取的证据。
             该检查基于 ``observed_evidence_ids`` 的可验证记录，不依赖模型自述；
          2) 语义一致性（``validate_evidence_references``）：校验业务规则，例如判定具体
             故障须有对应频族证据、判定 normal 时不得引用故障族证据。
             规则定义于 schema 模块，与提示词中的输出契约一一对应。

        主循环成功路径已调用一次本方法；上层在修复成功后会再次调用，
        因修复路径同属模型产出，须重新校验。
        """
        missing = set(diagnosis.evidence_refs) - set(self.state.observed_evidence_ids)
        if missing:
            raise ValueError(f"unobserved_final_evidence:{sorted(missing)}")
        errors = validate_evidence_references(diagnosis, self.evidence)
        if errors:
            raise ValueError("; ".join(errors))

    def run(self) -> AgentRuntimeResult:
        """对外入口：执行一次完整循环，并保证异常时状态正确收尾。

        循环内抛出未预期异常时，状态将停留在 ``analyzing`` 或 ``gathering_evidence``，
        导致审计材料显示任务仍在运行。故此处捕获异常，将状态迁移至 ``failed``
        并记录 ``termination_reason``，随后原样抛出：状态收尾由运行时负责，
        是否回退由上层编排决定。
        """
        if self.events or self.state.core_collected:
            raise ValueError("runtime_already_started")
        try:
            return self._run()
        except Exception:
            if self.state.phase in {"analyzing", "gathering_evidence"}:
                self.state.transition("failed")
                self.state.termination_reason = "runtime_error"
            raise

    def _run(self) -> AgentRuntimeResult:
        """Agent 主循环：请求模型、执行工具、回填结果，直至模型给出结论。

        循环分为五个阶段：

        【1】初始化消息
        system 提示词由 ``prompts.tool_agent_system_prompt`` 生成，其中包含本次运行的
        轮数与工具次数预算，使模型能够安排取证与输出时机。user 消息仅提供最小上下文
        （工况与本轮可用工具名），实际数据由工具返回，避免初始上下文过大。

        【2】循环与预算递减
        每轮先由 ``_tools()`` 计算可用工具，再递减轮数预算后请求模型。
        ``response_format`` 仅在客户端声明支持严格结构化输出时传入；否则退化为
        "提示词约定 JSON + 本地校验"，与 ``llm.py`` 中的降级记录一致。
        【3】工具调用：执行并回填
        协议要求先追加含 tool_calls 的 assistant 消息，再为每个调用追加
        ``role="tool"`` 且带 ``tool_call_id`` 的结果消息，顺序与配对不可颠倒。
        传入的 ``core_at_start`` 表示本轮开始时核心证据是否已就绪，
        同一轮内多个调用以此为准，故允许在同一轮内先取核心证据再查询明细。
        核心证据就绪后立即检查 ``evidence.analyzable``：质量门不通过时无需继续取证，
        直接主动终止，既节省后续请求，也使不可诊断的结论源于确定性门控。

        【4】无工具调用：作为最终答案处理
        依次执行文本存在性检查、JSON 解析、Pydantic 校验与引用校验，随后写入终态。
        ``state.transition`` 拦截非法状态迁移，``termination_reason`` 以固定枚举
        记录结束原因。所有失败均抛出明确异常，由上层编排决定修复或回退，
        运行时不做隐式降级。

        【5】轮数耗尽
        循环结束仍未输出结论时主动终止并标记 ``budget_exhausted``。
        该路径与质量门不通过共同构成运行时仅有的两种主动终止。
        """
        context = {"operating_context": self.evidence.operating_context if self.evidence else {},
                   "available_tools": [tool["function"]["name"] for tool in self._tools()]}
        messages = [
            {"role": "system", "content": tool_agent_system_prompt(
                self.max_turns, self.max_tool_calls,
                require_skill_first=self.require_skill_first,
            )},
            {"role": "user", "content": "先读取核心证据，再围绕疑点选择行动。初始上下文："
             + json.dumps(context, ensure_ascii=False)},
        ]
        if self.numeric_only:
            # 双路模式下的分支隔离：数值分支不陈述视觉结论，
            # 也不假设视觉结论已存在，汇总由独立 Agent 负责。
            messages[0]["content"] += (
                "\n当前只负责数值工具分支，输出本分支的诊断。视觉分支独立运行，"
                "不要假设或编造视觉意见；两路结果稍后由汇总 Agent 裁决。"
            )
        for turn in range(1, self.max_turns + 1):
            tools = self._tools()
            self.state.remaining_model_turns -= 1
            response_format = (self.llm.diagnosis_response_format()
                               if getattr(self.llm, "supports_strict_output", False)
                               and callable(getattr(self.llm, "diagnosis_response_format", None)) else None)
            self._emit_progress("agent_llm_request", turn=turn,
                                available_tools=[tool["function"]["name"] for tool in tools])
            message = self.llm.complete_messages(messages, tools=tools or None, response_format=response_format)
            calls, content = _tool_calls(message), _message_content(message)
            # 每轮记录一条事件，保留本轮可观察行为的最小事实集合。
            self.events.append({"turn": turn, "event": "assistant_turn", "tool_call_count": len(calls),
                                "has_content": bool(content)})
            self._emit_progress("agent_llm_response", turn=turn, tool_call_count=len(calls),
                                has_final_content=bool(content) and not calls)
            if calls:
                # 顺序要求：先回填 assistant(tool_calls)，再逐条回填工具结果。
                messages.append(_assistant_message(message))
                core_at_start = self.state.core_collected
                for call in calls:
                    call_id, name, raw = _tool_call_parts(call)
                    output = self._execute_call(name, raw, core_at_start=core_at_start, tools_allowed=bool(tools))
                    event = {"turn": turn, "event": "tool_result", "tool_name": name,
                             "arguments": self.state.tool_executions[-1].arguments, "output": output}
                    self.events.append(event)
                    self._emit_progress("agent_tool_result", **{k: v for k, v in event.items() if k != "event"})
                    # 工具结果采用紧凑 JSON 序列化；该体积计入每轮输入 token。
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "content": json.dumps(output, ensure_ascii=False, separators=(",", ":"))})
                if self.state.core_collected and self.evidence is not None and not self.evidence.analyzable:
                    # 硬门控优先于模型判读：输入不满足诊断条件时立即终止。
                    self.state.transition("limited")
                    self.state.termination_reason = "missing_required_input_or_quality_gate"
                    raise AgentRuntimeStopped("核心可诊断性门未通过")
                continue
            # 本轮无工具调用，按最终答案处理
            if not self.state.core_collected:
                # 结论须建立在实际测量之上，不允许跳过取证直接输出。
                raise ValueError("Agent未调用必需的collect_core_evidence")
            self.last_output = content
            payload = _parse_object(content)
            if payload is None:
                raise ValueError("Agent最终输出不是合法JSON对象")
            diagnosis = LLMDiagnosis.model_validate(payload)
            if self.evidence is None:
                raise ValueError("Agent未获得核心证据")
            self.validate_final(diagnosis)
            self.state.transition("limited" if diagnosis.status == "unknown" else "completed")
            self.state.termination_reason = "unresolved_evidence" if diagnosis.status == "unknown" else "sufficient_evidence"
            self._emit_progress("agent_final_output", turn=turn, diagnosis=diagnosis.model_dump(mode="json"))
            return AgentRuntimeResult(diagnosis, self.events, self.state.model_dump(mode="json"))
        # 循环结束仍未输出结论：预算耗尽，属主动终止，不再自动重试。
        self.state.transition("limited")
        self.state.termination_reason = "budget_exhausted"
        raise AgentRuntimeStopped("Agent在最大轮数内未给出最终诊断")
