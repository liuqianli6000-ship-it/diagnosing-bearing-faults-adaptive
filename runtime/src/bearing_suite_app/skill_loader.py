"""诊断 Skill 的受控读取器。

Skill 是运行时的流程契约，不把仓库目录直接暴露给模型。读取结果只包含当前
版本、适用范围、固定分析链和必要的参考文本；真实数据仍必须由诊断工具产生。

将流程规范直接写入 system 提示词存在三项成本：

  1. token 成本与注意力稀释：规范每轮重复发送，篇幅增加会导致模型忽略其中部分内容；
  2. 变更成本：规范修改后需重新验证全部提示词组合，回归范围扩大；
  3. 可控性不足：允许模型自行读取仓库目录等于将读取范围的决定权交给模型，
     而文件系统访问是 Agent 系统中最应受控的能力之一。

本模块将规范实现为受控摘要与工具化读取：
  * 仅提供 ``read()`` 一个入口，返回字段为白名单（版本、范围、步骤、契约、文本片段）；
  * 各文本段设有字符上限（``max_text_chars``），文档增长不会导致读取失控；
  * 超限时置位 ``truncated``，使上层与审计可知模型接收的内容为节选。

读取规范由此成为 Agent 循环中的一个常规工具调用：可审计（记录所读版本）、
可约束（仅可读取摘要）、可度量（记录体积）。

Skill 与工具数据的区别：
  * Skill 描述处理方式，包括流程顺序、判读规范与输出边界，相对稳定；
  * 工具提供事实，即本次样本的实际测量，每次运行不同。
两者分离后，判读规则变更仅需修改 Skill（影响全部模式），
测量逻辑变更仅需修改工具（不影响流程规范），回归测试范围因此明确。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DiagnosisSkillLoader:
    """从 skill 目录读取受控摘要；不向模型暴露目录本身。"""

    def __init__(self, root: str | Path | None = None, *, max_text_chars: int = 16000) -> None:
        # 默认路径由文件位置推导（仓库内固定位置），不依赖调用方的当前工作目录。
        # 资源文件读取均应采用该方式，避免启动目录变化导致文件无法定位。
        self.root = Path(root) if root else Path(__file__).resolve().parents[2] / "skills" / "diagnosing-bearing-faults-adaptive"
        # 下界保护：限制上限的最小值，防止摘要不可用；上限本身属于上下文预算的一部分。
        self.max_text_chars = max(2000, int(max_text_chars))

    def _json(self, name: str) -> dict[str, Any]:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def read(self) -> dict[str, Any]:
        """读取并组装 Skill 摘要。

        返回结构即流程契约，按用途分为三部分：

        【流程骨架】``workflow_steps`` 与 ``regime``
            取自 manifest 的 ``tool_chains.branches.steady``，包含固定步骤与默认工况分支
            （当前仅实现稳态）。该字段决定模型的行动顺序，故被运行时用作放行条件之一
            （``AgentState.workflow_steps``）。

        【契约细节】``workflow_contract``
            包含公共门控、终态叶子节点、policy 与多模态流程，用于界定模型的行为边界
            （何时可判定具体故障、何时必须判定 unknown）。
            规则以结构化 JSON 而非散文给出，便于模型逐条对照，也便于程序复用同一份数据。

        【参考文本】``skill_text``、``diagnosis_criteria`` 与 ``diagnostic_tree``
            SKILL.md 入口、完整诊断判据与诊断树节选。入口与诊断树有独立上限：``skill_text`` 使用
            ``max_text_chars``，诊断树为三分之一（属辅助细节，不与主规范占用同等预算）。
            超限时仅截断并置位 ``truncated``，不抛出异常：流程可用性优先于完整性。

        ``skill_id`` 取自 manifest 版本号并写入状态与审计，
        使结论可追溯至当时依据的规范版本。
        """

        manifest = self._json("runtime_manifest.json")
        policy = self._json("runtime_policy.json")
        skill_text = (self.root / "SKILL.md").read_text(encoding="utf-8")
        criteria_path = self.root / "references" / "diagnosis-criteria.md"
        diagnosis_criteria = criteria_path.read_text(encoding="utf-8").strip()
        review_path = self.root / "references" / "diagnosis-review.md"
        diagnosis_review = review_path.read_text(encoding="utf-8").strip() if review_path.is_file() else ""
        tree_path = self.root / "references" / "diagnostic-tree.md"
        # 可选文件缺失时退化为空串而非抛出异常：缺少参考资料不应导致整条链路不可用。
        tree_text = tree_path.read_text(encoding="utf-8") if tree_path.is_file() else ""
        branches = manifest.get("tool_chains", {}).get("branches", {})
        steady_steps = list(branches.get("steady", []))
        return {
            # 版本标识：写入状态与审计，用于追溯所依据的规范版本。
            "skill_id": manifest.get("skill_version", "diagnosing-bearing-faults-adaptive"),
            "schema_version": manifest.get("schema_version"),
            # 适用范围：明确流程覆盖与不覆盖的内容。
            "scope": "稳态滚动轴承频率域诊断；目标故障族为 BPFO/BPFI/BSF，并允许 normal/unknown",
            "regime": manifest.get("regime_selection", {}).get("default_branch", "steady"),
            # 流程骨架：模型的行动顺序依据，亦为运行时的放行条件之一。
            "workflow_steps": steady_steps,
            # 契约细节：以结构化数据而非散文给出规则，便于逐条对照与程序复用。
            "workflow_contract": {
                "common_gates": manifest.get("tool_chains", {}).get("common_gates", []),
                "final_leaves": manifest.get("diagnostic_contract", {}).get("final_leaves", []),
                "policy": policy,
            },
            # 参考文本：带独立上限的节选，超限仅标记不报错。
            "skill_text": skill_text[: self.max_text_chars],
            "diagnosis_criteria": diagnosis_criteria,
            "diagnosis_review": diagnosis_review,
            "diagnostic_tree": tree_text[: max(2000, self.max_text_chars // 3)],
            # 如实标记截断状态，避免模型将未提供的内容误判为规范中不存在。
            "truncated": len(skill_text) > self.max_text_chars or len(tree_text) > self.max_text_chars // 3,
        }
