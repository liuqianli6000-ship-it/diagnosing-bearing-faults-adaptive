"""单次诊断的公开任务状态；仅保存行动摘要，不保存模型内部推理。

本模块是运行时唯一的流程状态来源。无状态地反复调用模型无法回答以下问题：
当前处于哪个阶段、核心证据是否就绪、剩余预算多少、是否存在空转、
未决疑点是否已解决、本次运行因何结束。将这些依据集中到 ``AgentState``，
使流程约束可校验、可序列化、可审计。

三条设计原则：

1. 状态可序列化。所有字段均为基础类型或嵌套模型，可直接
   ``model_dump(mode="json")`` 落盘，因此不接纳 LLM 客户端、文件句柄等
   运行时对象。可落盘即意味着可重放、可对比、可做回归评测。

2. 只记录行动摘要，不记录思维链。``OpenQuestion`` 描述待查问题、依据证据与
   处置结果。模型中间推理不可校验、体积大且易被注入，不适合持久化。

3. 合法性内建于状态机。``transition()`` 以白名单拒绝非法迁移，
   流程约束由状态自身保证，不依赖调用方顺序正确。
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# 疑点类型：模型调用可选工具前必须声明的原因。
# 取值同时决定可响应它的工具（见 agent_tools.ToolSpec.allowed_kinds）。
QuestionKind = Literal[
    "family_conflict", "peak_verification", "normality_uncertain",
    "visual_verification", "temporal_consistency", "metadata_missing",
]
# 运行阶段：
#   analyzing           默认态，正在判读
#   gathering_evidence  正在执行工具取证（执行前后成对迁移）
#   awaiting_input / awaiting_human  等待外部输入（预留，当前链路未使用）
#   completed / limited / failed     终态：证据充分 / 主动受限 / 出错
Phase = Literal[
    "analyzing", "gathering_evidence", "awaiting_input", "awaiting_human",
    "completed", "limited", "failed",
]


class StateModel(BaseModel):
    """状态类公共配置：禁止未知字段 + 赋值时校验。

    ``validate_assignment=True`` 针对状态被频繁就地修改的场景（扣减预算、
    累加计数、更新疑点状态）。缺少该校验时，类型错误的赋值会静默通过，
    并在后续若干轮之后以难以定位的方式失败；开启后错误在赋值处即暴露。
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class OpenQuestion(StateModel):
    """待解释的疑点：提问 → 取证 → 结论，全过程可审查。

    ``status`` 的四个取值构成一次取证闭环：
        open               已提出，尚未取证
        evidence_received  已收到补充结果（不代表疑点已消除）
        resolved           运行时可确认已解决（目前仅限峰值明细送达）
        blocked            所需参考客观不可得

    ``result_evidence_ids`` 与实际 ``resolution`` 记录未解决疑点的处置痕迹，
    使 unknown 结论在审计上可追溯至具体未决项，而非笼统的"证据不足"。
    """

    question_id: str
    kind: QuestionKind
    question: str
    needed_evidence: str
    evidence_refs: list[str] = Field(default_factory=list)
    status: Literal["open", "evidence_received", "resolved", "blocked"] = "open"
    result_evidence_ids: list[str] = Field(default_factory=list)
    resolution: str | None = None


class ToolExecution(StateModel):
    """单次工具调用的完整记录：工具、参数、结果、状态与耗时。

    该表是可观测性的核心数据源，支撑以下分析：
      * 工具的失败率、缓存命中率与平均耗时（优化提示词与实现的依据）；
      * 按 ``arguments`` 与 ``status`` 顺序还原模型当时的取证路径；
      * 通过 ``question_id`` 关联疑点，评估取证是否具有针对性。

    ``cache_key`` 标识测量同一性，``cache_hit`` 标识本次是否实际执行；
    两者结合才能准确判定"无进展"行为。
    """

    execution_id: str
    tool_name: str
    tool_version: str
    question_id: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str
    error_code: str | None = None
    retryable: bool = False
    cache_key: str | None = None
    cache_hit: bool = False
    elapsed_seconds: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)


class AgentState(StateModel):
    """一次诊断运行的完整状态快照。

    字段按用途分组：
      标识    task_id / run_id / sample_id / schema_version
      流程    phase / core_collected / skill_loaded / skill_id / workflow_steps
      证据    catalog_evidence_ids（可见）/ observed_evidence_ids（已获取）
      疑点    questions
      账本    tool_executions
      预算    remaining_model_turns / remaining_tool_attempts / no_progress_count
      收尾    termination_reason / pending_input

    证据字段的两个层次构成引用闭包的基础：
      * ``catalog_evidence_ids``：模型已知存在、可索取明细的 evidence_id；
      * ``observed_evidence_ids``：模型已实际获取、可写入最终引用的 evidence_id。
    仅后者可进入最终诊断的 ``evidence_refs``，使"结论可追溯至具体测量"由程序强制
    保证，而非依赖提示词约定。

    ``schema_version`` 供审计使用：状态结构演进后，历史运行记录仍可正确解读。
    """

    schema_version: str = "bearing-agent-state/v1"
    task_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    sample_id: str = Field(default_factory=lambda: uuid4().hex)
    phase: Phase = "analyzing"
    core_collected: bool = False
    skill_loaded: bool = False
    skill_id: str | None = None
    workflow_steps: list[str] = Field(default_factory=list)
    catalog_evidence_ids: list[str] = Field(default_factory=list)
    observed_evidence_ids: list[str] = Field(default_factory=list)
    questions: list[OpenQuestion] = Field(default_factory=list)
    tool_executions: list[ToolExecution] = Field(default_factory=list)
    remaining_model_turns: int = 0
    remaining_tool_attempts: int = 0
    no_progress_count: int = 0
    termination_reason: str | None = None
    pending_input: dict[str, Any] | None = None

    def transition(self, phase: Phase) -> None:
        """按白名单迁移阶段，非法迁移抛出异常。

        合法迁移以数据表形式集中声明，便于直接读取流程拓扑：

            analyzing ⇄ gathering_evidence → （completed | limited | failed）
            analyzing 可直接进入任一终态（如质量门初始即不通过）
            三个终态为吸收态，无任何出边

        终态必须封闭：运行结束后继续写入会使审计记录自相矛盾（例如"已完成"
        之后回到"分析中"），导致无法判定本次运行是否成功。
        ``phase == self.phase`` 时直接放行，以支持幂等设置。
        """
        allowed = {
            "analyzing": {"gathering_evidence", "awaiting_input", "awaiting_human", "completed", "limited", "failed"},
            "gathering_evidence": {"analyzing", "awaiting_input", "awaiting_human", "limited", "failed"},
            "awaiting_input": {"analyzing", "failed"},
            "awaiting_human": {"analyzing", "failed"},
            "completed": set(), "limited": set(), "failed": set(),
        }
        if phase != self.phase and phase not in allowed[self.phase]:
            raise ValueError(f"invalid_state_transition:{self.phase}->{phase}")
        self.phase = phase

    def view(self) -> dict[str, Any]:
        """返回同步给模型的精简状态，不重复注入完整执行记录。

        每轮均需向模型同步状态，但 ``tool_executions`` 承载的历史调用明细已作为
        ``role="tool"`` 消息存在于对话历史中，重复注入既增加 token 开销，
        也容易使模型混淆新旧数据。

        ``exclude={"tool_executions"}`` 保留预算、阶段与未决疑点，
        即模型编排下一步所需的信息。审计使用全量状态，模型上下文使用精简视图。
        """
        return self.model_dump(mode="json", exclude={"tool_executions"})
