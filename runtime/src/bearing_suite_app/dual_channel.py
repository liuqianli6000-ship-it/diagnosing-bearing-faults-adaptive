"""独立图像诊断与双路汇总；两个分支不共享模型消息。

本模块实现多 Agent 协作的一种基础形态，其适用判据为：存在两个可独立完成、
且结论需要裁决的任务。

    数值分支（agent_runtime 的工具循环）  ─┐
                                          ├─→ 汇总 Agent（裁决两路结果） → 最终诊断
    视觉分支（仅基于图像的独立诊断）       ─┘

三项关键设计：

一、信息隔离：视觉分支不接收数值结论
   ``diagnose_image`` 仅接收图像、哈希、理论频率坐标与频带 ID，
   不接收数值匹配等级、数值诊断结论与案例标签。
   图像与数值由同一信号产生，属同源证据，不得重复计权。若视觉模型预先获知数值结论，
   其输出将趋同于该结论，两路一致性退化为自证而非独立复核。

二、阶段化结构化输出：各阶段独立 schema 与校验
   ImageDiagnosis（视觉）与 FusionDiagnosis（汇总）各自执行本地 schema 校验与领域
   规则校验（引用、一致性、冲突标记、置信度上限）。各 Agent 通过可校验的结构化
   中间产物交接，而非共享同一对话上下文，使错误定位更为明确。

三、裁决规则由程序计算，模型负责解释
   汇总前由程序比较两路的 ``fault_type`` 得出 ``agreement``
   （agree / conflict / insufficient），并作为 ``required_agreement`` 传入；
   模型输出与该值不一致时判定失败（见 ``summarize_channels`` 末尾校验）。
   一致性属事实判定，不由模型裁量，模型负责解释冲突与取舍。
   该分工可推广至其它 Agent 场景：程序可确定性计算的内容不应交由模型判断。

同源证据规则构成多模态设计的底线，图像与数值来自同一信号：
  * 两路一致不等于两份独立证据，不得据此提高置信度；
  * 图像不能推翻数值质量门，门控不通过时视觉分支同样无权判定正常或故障；
  * 图像可用于提示漏检与竞争结构，但不能单独建立具体故障；
  * 图像失败或缺失时须如实传递缺失状态，允许给出受限结论，
    但不得声称已完成双路验证。
上述规则以代码校验实施，而非仅写入提示词。
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .diagnosis_schemas import EvidenceBundle, LLMDiagnosis, validate_evidence_references
from .prompts import FUSION_DIAGNOSIS_SYSTEM, FUSION_REPAIR_SYSTEM, VISUAL_DIAGNOSIS_SYSTEM
from .visual_observer import _parse_object


# ---------------------------------------------------------------------------
# 阶段产物 schema：每个 Agent 阶段一个，均包含领域规则校验
# ---------------------------------------------------------------------------
# schema 不仅描述字段结构，还通过 model_validator 强制业务自洽性。
# 结构校验仅保证字段已填写，业务校验用于拦截字段完整但内容自相矛盾的输出。


class ImageFinding(BaseModel):
    """单条图像观察。

    ``band_ids`` 为必填且非空：视觉结论须对应具体频带，
    不接受无法核对的概括性描述。
    ``pattern`` 以枚举限定可表达的形态（谐波、跨频带一致、边带、无清晰形态、模糊），
    等效于为视觉模型提供受限词表，便于校验与统计。
    """

    model_config = ConfigDict(extra="forbid")
    band_ids: list[str] = Field(min_length=1)
    pattern: Literal["harmonics", "cross_band_consistency", "sidebands", "no_clear_pattern", "ambiguous"]
    description: str = Field(min_length=1)


class ImageDiagnosis(BaseModel):
    """视觉分支的完整产物：可读性、观察、诊断与限制。

    ``check_visibility`` 将互斥关系写入 schema，同时拦截两类相反的输出缺陷：
      * 图像不可读却给出观察或具体诊断：不允许；
      * 图像可读但未给出任何观察：不允许。
    两类情形均在真实运行中出现过，提示词仅能降低发生概率，schema 可完全排除。
    """

    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    image_sha256: str
    observable: bool
    findings: list[ImageFinding]
    diagnosis: LLMDiagnosis
    limitations: list[str]

    @model_validator(mode="after")
    def check_visibility(self):
        if not self.observable and (self.findings or self.diagnosis.fault_type != "unknown"):
            raise ValueError("不可读图像只能输出 unknown，且不得生成图像观察")
        if self.observable and not self.findings:
            raise ValueError("可读图像必须提供观察")
        return self


class FusionDiagnosis(BaseModel):
    """汇总 Agent 的产物：一致性判定、两路摘要、取舍说明与最终诊断。

    ``agreement`` 由模型填写，``required_agreement`` 由程序计算，两者须一致
    （在 ``summarize_channels`` 中强制）。要求模型复述程序计算的结论可使其在撰写
    摘要前先行确认冲突状态，避免出现既承认冲突又声称两路一致的叙述。
    """

    model_config = ConfigDict(extra="forbid")
    agreement: Literal["agree", "conflict", "insufficient"]
    numeric_summary: str = Field(min_length=1)
    visual_summary: str = Field(min_length=1)
    resolution: str = Field(min_length=1)
    diagnosis: LLMDiagnosis


class _StructuredOutputError(ValueError):
    """模型已返回，但内容未满足结构化输出要求。

    与 ValidationError 分开定义，使 ``_complete`` 能准确捕获可修复的失败类型；
    网络与超时异常不属于可修复范围，不应消耗修复调用。
    即异常分类决定重试策略。
    """


def _complete_once(llm: Any, system: str, content: Any, schema: type[BaseModel]):
    """发起一次结构化输出请求，并执行三项检查。

    三项检查构成结构化输出在业务中可用的前提：

      1. 仅在 ``supports_strict_output`` 为真时传入 ``response_format``，
         并将 schema 作为 ``json_schema`` 交由服务端约束；
      2. 拒绝工具调用：观察与裁决阶段属纯产出阶段，模型在此发起工具调用表明
         角色理解错误，直接判定失败。该阶段能力白名单可防止子 Agent 调用工具
         而超出编排层设定的流程；
      3. 解析后以 ``model_validate(payload, strict=True)`` 严格校验，
         禁用隐式类型转换，避免类型错误被掩盖。

    ``content`` 类型为 ``Any``：文本阶段传入字符串，图像阶段传入多模态消息数组，
    归一化由调用方完成。
    """

    response_format = None
    if getattr(llm, "supports_strict_output", False):
        response_format = {"type": "json_schema", "json_schema": {
            "name": schema.__name__, "strict": True, "schema": schema.model_json_schema(),
        }}
    message = llm.complete_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": content}],
        response_format=response_format,
    )
    if getattr(message, "tool_calls", None):
        raise _StructuredOutputError("此诊断阶段不能调用工具")
    payload = _parse_object(getattr(message, "content", None))
    if payload is None:
        raise _StructuredOutputError("模型未返回合法 JSON 对象")
    return schema.model_validate(payload, strict=True)


def _complete(
    llm: Any,
    system: str,
    content: Any,
    schema: type[BaseModel],
    *,
    repair_system: str | None = None,
):
    """完成结构化输出；仅在 schema 校验失败时执行一次受限修复。

    三项约束：

      * 仅捕获可修复异常：``ValidationError``（结构或业务校验失败）与
        ``_StructuredOutputError``（未返回 JSON，或在本阶段调用了工具）。
        网络超时等异常不在其中，重试不具意义；
      * 至多一次：不设循环与重试次数参数。单次修复已可覆盖多数格式波动，
        继续重试的收益有限而成本与不确定性累积，且会掩盖真实原因；
      * 向修复调用传递错误原文：``repair_content`` 包含上一份输出与校验错误信息，
        使模型明确修改目标；同时附上原输入 ``content``，确保修复基于同一份输入，
        避免退化为重新诊断。

    ``repair_system is None`` 表示该阶段不执行修复，此时直接抛出原异常，由上层决定回退。
    """
    try:
        return _complete_once(llm, system, content, schema)
    except (ValidationError, _StructuredOutputError) as exc:
        if repair_system is None:
            raise
        repair_content = (
            str(content)
            + "\n上一份输出未通过本地结构校验。请根据同一份两路输入重新输出，校验错误如下：\n"
            + str(exc)
        )
        return _complete_once(llm, repair_system, repair_content, schema)


def diagnose_image(llm: Any, artifact: dict, frequencies: dict, max_image_bytes: int) -> dict:
    """仅接收图像与理论坐标，不接受数值诊断结论。

    返回 ``{"status": ...}`` 而不抛出异常：双路模式下视觉分支允许失败，
    失败状态本身具有信息价值（汇总阶段需记录图像不可用），不应中断整条链路。

    代码中包含四类入参可信性检查，均为多模态 Agent 的必要校验：
      1) 能力检查：端点不支持图像输入时不再继续；
      2) 产物合规检查：``independent_visual_view`` 标识图像是否仅包含谱线、
         未叠加数值匹配结果。若将含匹配标记的审计图输入视觉模型，
         模型可从图中读取数值结论，独立视觉链将失效；
      3) 资源与完整性检查：大小上限避免单次请求超出上下文与带宽限制，
         SHA-256 比对确认磁盘图像未被替换；凡读取外部文件的环节均应执行该校验；
      4) 输出引用校验：模型回填的 artifact_id 与 image_sha256 须与本次输入一致，
         ``evidence_refs`` 仅可引用本次图像，findings 中的频带须属于本次实际展示的频带。
         缺少这三项校验时，模型可沿用上一次图像的结论而程序无法分辨。

    ``context`` 中携带 ``independent_band_normalization: True``，将图像的物理含义
    （各频带独立归一化）告知模型，以避免跨频带比较绝对幅值，
    该约束的表述精度高于提示词层面的说明。
    """
    if not getattr(llm, "supports_image_input", False):
        return {"status": "unavailable", "reason": "provider_image_input_not_enabled"}
    try:
        if not artifact.get("processing", {}).get("independent_visual_view"):
            raise ValueError("独立视觉链拒绝包含数值匹配结果的审计图")
        path = Path(artifact["path"])
        if path.stat().st_size > max_image_bytes:
            raise ValueError("图像超过输入大小上限")
        image = path.read_bytes()
        digest = hashlib.sha256(image).hexdigest()
        if digest != artifact.get("sha256"):
            raise ValueError("图像 SHA-256 不匹配")
        band_ids = list(artifact["processing"]["displayed_points_by_band"])
        context = {"artifact_id": artifact["evidence_id"], "image_sha256": digest,
                   "theoretical_frequencies_hz": frequencies, "band_ids": band_ids,
                   "independent_band_normalization": True}
        # 多模态消息由文本段与图像段组成：文本段包含输出契约与坐标说明，
        # 图像段以 data URL 内联 base64，避免额外的文件托管与鉴权依赖。
        content = [
            {"type": "text", "text": "输出契约：\n" + json.dumps(ImageDiagnosis.model_json_schema(), ensure_ascii=False)
             + "\n图像坐标说明：\n" + json.dumps(context, ensure_ascii=False)},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")}},
        ]
        output = _complete(llm, VISUAL_DIAGNOSIS_SYSTEM, content, ImageDiagnosis)
        if output.artifact_id != context["artifact_id"] or output.image_sha256 != digest:
            raise ValueError("视觉诊断引用了错误的图像或哈希")
        if set(output.diagnosis.evidence_refs) != {context["artifact_id"]}:
            raise ValueError("视觉诊断只能引用本次图像")
        if any(not set(f.band_ids) <= set(band_ids) for f in output.findings):
            raise ValueError("视觉诊断引用了不存在的频带")
        return {"status": "completed", "result": output.model_dump(mode="json")}
    except Exception as exc:
        # 兜底为 failed：视觉分支的异常仅影响自身，
        # 由汇总阶段如实记录图像不可用及失败原因。
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def summarize_channels(llm: Any, evidence: EvidenceBundle, numeric: dict, visual: dict) -> FusionDiagnosis:
    """分支结束后执行汇总，并在本地校验最终数值依据与冲突记录。

    执行顺序为先计算事实，再交由模型裁决：

    【第 1 步】程序判定一致性
    仅当视觉结果可读、两路均给出诊断且均非 unknown 时才可能为 agree 或 conflict，
    其余情形一律为 ``insufficient``。该结果作为 ``required_agreement`` 写入输入，
    要求模型如实填写并在后续核对，以避免模型将 insufficient 表述为 agree。

    【第 2 步】向模型提供两路结果与数值证据
    payload 同时包含 ``numeric_evidence``（可引用的数值证据）与两路结论，
    以及 ``shared_signal_source: True``，明确两路同源，
    避免模型将一致性视为独立验证。

    【第 3 步】本地领域校验
    模型返回后逐条检查，任一条不通过即抛出异常，由编排层回退或记录失败：
      * ``agreement`` 须等于程序计算值，模型无权自行认定一致性；
      * 引用须真实存在（``validate_evidence_references``，视觉证据 ID 单独传入）；
      * 质量门失败时仅可判定 unknown；
      * 判定具体故障须有对应族的 moderate/strong 数值证据；
      * 存在故障族证据时不得判定 normal；
      * 冲突须写入 ``diagnosis.conflicts``；
      * 非 agree 情形下不得给出 high 置信度。

    上述规则可归纳为：视觉分支可补充证据与提示冲突，但不能替代数值证据。
    该结论以七条可执行断言实施，而非依赖提示词约定。
    """
    visual_result = visual.get("result", {})
    visual_diagnosis = visual_result.get("diagnosis")
    numeric_diagnosis = numeric.get("diagnosis")
    # 一致性由程序判定：默认 insufficient（图像缺失/不可读/任一路为 unknown）。
    agreement = "insufficient"
    if (visual_result.get("observable") and visual_diagnosis and numeric_diagnosis
            and "unknown" not in {visual_diagnosis["fault_type"], numeric_diagnosis["fault_type"]}):
        agreement = "agree" if visual_diagnosis["fault_type"] == numeric_diagnosis["fault_type"] else "conflict"
    # 数值证据与两路结论置于同一 payload，使最终引用可核对：
    # 模型仅可引用实际存在的 evidence_id，不得引用视觉分支的自述。
    payload = {"numeric_evidence": evidence.to_llm_payload(), "numeric_channel": numeric,
               "visual_channel": visual, "required_agreement": agreement, "shared_signal_source": True}
    output = _complete(llm, FUSION_DIAGNOSIS_SYSTEM,
        "输出契约：\n" + json.dumps(FusionDiagnosis.model_json_schema(), ensure_ascii=False)
        + "\n两路结果：\n" + json.dumps(payload, ensure_ascii=False), FusionDiagnosis,
        repair_system=FUSION_REPAIR_SYSTEM)
    # 1) 一致性须与程序判定一致，该事实不由模型裁量。
    if output.agreement != agreement:
        raise ValueError("汇总未如实标记两路一致性")
    diagnosis = output.diagnosis
    # 视觉证据 ID 作为白名单的一部分单独传入：汇总可引用图像工件，
    # 但不得引用图像中"看似存在"的数值证据。
    visual_ids = {
        str(visual_result.get("artifact_id"))
    } if visual_result.get("artifact_id") else set()
    errors = validate_evidence_references(diagnosis, evidence, visual_ids)
    # 2) 质量门不可被图像绕过。
    if not evidence.analyzable and diagnosis.fault_type != "unknown":
        errors.append("质量门失败时只能判 unknown")
    # 3) 具体故障须由数值证据支撑，视觉分支不得单独建立故障结论。
    if diagnosis.status == "diagnosed" and evidence.assessments.get(diagnosis.leading_family, {}).get("evidence_level") not in {"moderate", "strong"}:
        errors.append("具体故障缺少对应族数值支持")
    # 4) 反向约束：存在故障族证据时不得判定 normal。
    if diagnosis.status == "normal" and any(a.get("evidence_level") in {"moderate", "strong"} for a in evidence.assessments.values()):
        errors.append("存在故障族证据时不能判 normal")
    # 5) 冲突须留痕：未写入 conflicts 即等同抹除分歧。
    if agreement == "conflict" and not diagnosis.conflicts:
        errors.append("两路不一致必须记录冲突")
    # 6) 置信度上限：信息不完整或存在冲突时不得给出 high。
    if agreement != "agree" and diagnosis.confidence_level == "high":
        errors.append("冲突或信息不完整时不得给出 high 置信度")
    if errors:
        raise ValueError("; ".join(errors))
    return output
