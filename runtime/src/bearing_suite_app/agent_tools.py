"""工具契约与本地执行边界；模型 schema 和本地校验使用相同定义。

本模块定义 Agent 工具层的完整契约。生产环境中的工具不只是执行函数，还须明确
以下五项，均由 ``ToolSpec`` 建模：

  1. 名称与用途        → ``name`` / ``description``，写入 JSON Schema 供模型使用
  2. 参数与边界        → ``input_model``，同时是 schema 来源与本地校验规则
  3. 本地实现          → ``handler``，模型无法影响其行为
  4. 允许的调用情境    → ``allowed_kinds``，限定可响应的疑点类型
  5. 返回值要求        → ``output_validator``，约束必需字段

第 2 与第 5 项对应同一原则：入参与出参均需校验。仅校验入参时，模型可能基于
字段缺失的工具结果继续生成解释。

单一事实来源：``ToolSpec.schema()`` 生成的 JSON Schema 直接取自
``input_model.model_json_schema()``，模型请求参数则由同一 ``input_model`` 校验，
两个方向自动同步，避免文档与实现不一致。

严格模式：``extra="forbid"`` 拒绝未声明字段而非静默忽略，防止模型形成错误调用习惯
并将问题掩盖至后期；``strict=True`` 禁用隐式类型转换，避免模型长期使用错误类型。

执行边界：``ToolExecutor`` 将单次工具调用视为有状态的远程调用，统一处理缓存、
重试上限、耗时统计与异常收敛。缓存键的构造规则见 ``ToolExecutor.key()``。
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from .agent_state import QuestionKind

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=240)]
EvidenceId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=160)]

# ---------------------------------------------------------------------------
# 工具输入契约
# ---------------------------------------------------------------------------
# 以下模型同时充当模型的 JSON Schema 与本地校验规则。基类 ToolInput 关闭未知字段
# 并启用严格模式，全部工具输入继承这两项约束。
#
# 输入模型亦构成提示词的一部分：模型可从 schema 读取"证据引用须 1~6 条"等边界，
# 因此能够用 schema 表达的约束无需再以自然语言重复。


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Purpose(ToolInput):
    """疑点：模型调用可选工具前必须提交的结构化理由。

    以可审查的行动摘要替代模型的自由推理：

      * ``kind``：疑点类型（频族竞争 / 峰值核对 / 正常性不确定 / 图像核对等），
        运行时据此校验工具与疑点是否匹配；
      * ``question``：待解决问题的简短陈述（限长以防夹带长篇推理）；
      * ``needed_evidence``：回答该问题所需的证据，要求模型先明确目的；
      * ``evidence_refs``：疑点所依据的已读证据（1~6 条），运行时校验其必须属于
        ``observed_evidence_ids``，因此模型无法为未读取的证据发起取证。

    相较模型的中间推理，结构化疑点可校验、可统计、可入审计材料，
    并可用于判定疑点是否已解决。
    """

    question_id: EvidenceId | None = None
    kind: QuestionKind
    question: ShortText
    needed_evidence: ShortText
    evidence_refs: list[EvidenceId] = Field(min_length=1, max_length=6)


class OptionalInput(ToolInput):
    """可选工具的统一输入：仅包含一个疑点。"""

    purpose: Purpose


class DetailsInput(OptionalInput):
    """证据明细工具的输入：除疑点外，指定要读取的证据记录。

    ``max_length=6`` 为上下文预算上限，防止单次调用索取整份证据。
    """

    evidence_ids: list[EvidenceId] = Field(min_length=1, max_length=6)


class VisualInput(OptionalInput):
    """图像观察工具的输入：必须指定具体工件 ID。"""

    artifact_id: EvidenceId


class SkillReadInput(ToolInput):
    """Skill 读取无模型参数，输入对象必须为空。

    无参数工具同样需要输入模型：不接受任何参数本身即为需要强制执行的契约，
    模型传入参数时会因 ``extra="forbid"`` 被判定非法。
    """

    pass


class ToolResult(BaseModel):
    """工具的统一输出信封：状态、数据、错误码与溯源信息。

    工具不直接返回 dict 或抛出异常，因为返回值需同时服务三方：

      * 模型：需区分"成功但无数据"（``unavailable``）与"调用错误"
        （``invalid_input``），否则可能将缺少参考误判为正常；
      * 运行时：依赖 ``evidence_ids`` 维护引用闭包，依赖 ``cache_hit``
        判定本次调用是否产生进展；
      * 审计：依赖 ``provenance`` 记录结果所属的 run、工具版本与缓存键，
        以证明该证据确由当次运行产生。

    ``status`` 的取值语义互不重叠：
      * ``ok``             成功，且包含可用于诊断的证据；
      * ``unavailable``    成功但客观无数据（无参考案例 / 无对比基线）；
                           不是失败，也不构成可引用证据；
      * ``invalid_input``  模型调用方式错误（未知工具 / 参数非法 / 顺序错误），
                           可由模型自行纠正；
      * ``retryable_error``临时性故障（超时 / 连接中断），可重试；
      * ``failed``         确定性失败（工具内部错误），重试无效。
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    status: Literal["ok", "unavailable", "invalid_input", "retryable_error", "failed"]
    data: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False
    cache_hit: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    tool_version: str = "1"
    provenance: dict[str, Any] = Field(default_factory=dict)


def record_ids(value: Any) -> set[str]:
    """递归收集嵌套结构中的所有 ``evidence_id``。

    该函数支撑证据引用机制：工具输出中出现的每个 evidence_id 均被收集，
    运行时据此确定模型已获取的证据范围，并在最终校验中检查引用是否越界。

    实现不依赖证据包的具体结构，仅识别嵌套 JSON 中的 ``"evidence_id"`` 字段，
    使证据结构演进无需改动此处；代价是要求所有证据记录携带 evidence_id，
    该要求由 schema 层约定。
    """
    ids: set[str] = set()
    if isinstance(value, Mapping):
        if isinstance(value.get("evidence_id"), str):
            ids.add(value["evidence_id"])
        for item in value.values():
            ids.update(record_ids(item))
    elif isinstance(value, list):
        for item in value:
            ids.update(record_ids(item))
    return ids


@dataclass(frozen=True)
class ToolSpec:
    """单个工具的完整定义，对应模块说明中的五项要素。

    ``frozen=True`` 保证工具定义在运行期不可变。调整工具集合应替换 registry
    而非就地修改 ToolSpec，否则审计记录中的 schema 可能与实际执行不一致。

    ``max_retries=1`` 表示同一参数最多尝试两次（初始一次 + 重试一次），
    由执行器记账。是否重试仍取决于返回状态。

    ``schema(strict)`` 中的 ``strict`` 为供应商能力开关：支持时附加
    ``"strict": True``，由服务端同步校验参数。该能力仅作为附加防线，
    本地 ``model_validate`` 始终执行。
    """

    name: str
    description: str
    input_model: type[ToolInput]
    handler: Callable[[ToolInput], ToolResult]
    allowed_kinds: frozenset[str] = frozenset()
    version: str = "1"
    timeout_seconds: float = 30.0
    max_retries: int = 1
    output_validator: Callable[[dict[str, Any]], None] | None = None

    def schema(self, strict: bool) -> dict[str, Any]:
        # 复用 Pydantic 生成的 JSON Schema，与本地校验共用同一定义。
        function = {"name": self.name, "description": self.description,
                    "parameters": self.input_model.model_json_schema()}
        if strict:
            function["strict"] = True
        return {"type": "function", "function": function}


@dataclass
class ToolExecutor:
    """run 私有缓存；同步调用在返回后核查耗时，不做可强制取消的假设。

    缓存生命周期与 run 绑定：随 run 建立、随 run 结束丢弃。改为进程级全局缓存会
    导致跨样本复用数据，进而污染诊断结论。

    ``cache``：缓存键 → 结果；命中时深拷贝并置 ``cache_hit=True``，
    供运行时判定无进展行为。
    ``attempts``：缓存键 → 已尝试次数，用于实施重试上限。

    超时处理：Python 无法安全中断执行中的同步函数，因此不提供"超时即终止"的
    语义，而是在返回后比较耗时，超限时记录 ``deadline_exceeded``，
    由运行时据此立即停止后续动作。
    """

    run_id: str
    cache: dict[str, ToolResult] = field(default_factory=dict)
    attempts: dict[str, int] = field(default_factory=dict)

    def key(self, spec: ToolSpec, args: ToolInput) -> str:
        """计算缓存键：run_id + 工具名 + 工具版本 + 参数（不含疑点）。

        两项设计保证缓存正确性：

          1. 键包含 ``run_id`` 与 ``version``，跨运行、跨工具版本一律不复用结果；
          2. 排除 ``purpose``（疑点）。疑点为模型的表述，同一测量换用不同问法
             不应视为新任务。若纳入疑点，模型可通过改写问题文本反复触发实际计算，
             既消耗算力，也绕过无进展检测；排除后重复调用命中缓存，
             无进展计数上升，空转行为可被识别。

        ``sort_keys=True`` 保证参数字典顺序不同时生成相同键。
        """
        # 疑点措辞不影响测量身份，避免换问题文本绕过缓存。
        parameters = args.model_dump(mode="json", exclude={"purpose"})
        raw = json.dumps([self.run_id, spec.name, spec.version, parameters], sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    def execute(self, spec: ToolSpec, args: ToolInput) -> tuple[ToolResult, str, float]:
        """执行一次工具调用，返回 ``(结果, 缓存键, 耗时秒)``。

        异常收敛：handler 的异常不向主循环传播，统一转换为结构化结果：
          * ``TimeoutError`` / ``ConnectionError`` → ``retryable_error``；
          * ``ValidationError`` / ``TypeError`` / ``ValueError`` →
            ``failed: invalid_tool_output``（产出不符合 ToolResult 契约）；
          * 其它异常 → ``failed: tool_failed:<异常类名>``。
        异常穿透会中断整个 Agent 循环；结构化失败则允许模型感知工具不可用，
        改用其它证据得出结论并如实说明限制。

        产出校验：成功结果须可被 ``json.dumps(allow_nan=False)`` 序列化
        （工具输出将写入消息与审计文件，NaN/Infinity 会导致链路后期失败），
        随后执行 ``output_validator`` 确认必需字段完整，与入参校验构成完整闭环。

        重试与缓存语义：
          * 尝试次数达到 ``1 + max_retries`` 返回 ``retry_exhausted``；
          * 仅 ``ok`` / ``unavailable`` / 不可重试的 ``failed`` 写入缓存；
            ``retryable_error`` 不缓存，避免临时故障被固化；
          * 写入缓存使用深拷贝，防止返回对象被修改后污染缓存。
        """
        key = self.key(spec, args)
        if key in self.cache:
            # 命中缓存：深拷贝避免调用方修改污染缓存，并标记 cache_hit 供无进展检测使用。
            result = self.cache[key].model_copy(deep=True)
            result.cache_hit = True
            return result, key, 0.0
        if self.attempts.get(key, 0) >= 1 + spec.max_retries:
            return ToolResult(status="failed", error_code="retry_exhausted"), key, 0.0
        self.attempts[key] = self.attempts.get(key, 0) + 1
        started = time.monotonic()
        try:
            result = ToolResult.model_validate(spec.handler(args))
            # 校验嵌套数据可被协议与审计序列化。
            json.dumps(result.model_dump(mode="json"), allow_nan=False)
            if result.status == "ok" and spec.output_validator:
                spec.output_validator(result.data)
        except (TimeoutError, ConnectionError):
            result = ToolResult(status="retryable_error", error_code="temporary_tool_failure", retryable=True)
        except (ValidationError, TypeError, ValueError):
            result = ToolResult(status="failed", error_code="invalid_tool_output")
        except Exception as exc:
            result = ToolResult(status="failed", error_code=f"tool_failed:{type(exc).__name__}")
        elapsed = time.monotonic() - started
        # 溯源信息记录结果所属的 run、工具版本与缓存键，用于审计、复现与结果一致性排查。
        result.tool_version = spec.version
        result.provenance = {"run_id": self.run_id, "tool": spec.name,
                             "version": spec.version, "cache_key": key}
        # evidence_ids 由输出内容反推，不依赖各 handler 自行填写，避免引用闭包遗漏。
        result.evidence_ids = sorted(record_ids(result.data))
        if elapsed > spec.timeout_seconds:
            # 结果已返回时保留有效结果，是否阻止后续动作由运行时决定。
            result.provenance["deadline_exceeded"] = True
        if result.status in {"ok", "unavailable"} or (result.status == "failed" and not result.retryable):
            self.cache[key] = copy.deepcopy(result)
        return result, key, elapsed


def require_keys(*keys: str) -> Callable[[dict[str, Any]], None]:
    """生成输出校验器，要求结果包含指定字段。

    用法：``output_validator=require_keys("assessments", "evidence_catalog")``。
    字段缺失时抛出 ValueError，由执行器统一转换为 ``failed: invalid_tool_output``。

    该函数是工具层的输出契约：完整类型校验交由各工具自身的结构化数据，
    此处仅约束关键字段是否存在。
    """

    def validate(data: dict[str, Any]) -> None:
        if any(key not in data for key in keys):
            raise ValueError("missing_tool_result_fields")
    return validate
