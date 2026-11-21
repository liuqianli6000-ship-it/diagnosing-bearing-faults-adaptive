"""Evidence diagnosis, independent reason review, then one bounded revision."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .dataset_evidence import FAULT_NAMES, public_input, validate_analysis
from .execution import observe_call
from .llm import _parse_json


ROOT = Path(__file__).resolve().parents[2]
SKILL_PATH = ROOT / "skills/diagnosing-bearing-faults-adaptive/references/expert-diagnosis.md"
REVIEW_SKILL_PATH = ROOT / "skills/diagnosing-bearing-faults-adaptive/references/expert-review.md"
DIAGNOSIS_CALLS = 3
SYSTEM_PROMPT = "你是轴承诊断专家。按下列工作规范和实测证据返回 JSON。reason 是可交付的中文诊断意见，四段合计严格300—500字。用具体谱峰、倍频关系和频率偏差解释判断。"
REVIEW_SYSTEM_PROMPT = "你是独立的轴承诊断文本审查专家。按下列工作规范对照实测证据，审查诊断初稿中的引用、推理和表达，返回有证据的修改意见。"
REVIEW_CATEGORIES = {"证据引用", "推理逻辑", "措辞术语", "格式内容", "分类疑点"}


def plain_report(reason):
    for term, wording in (
        ("最佳模型胜出", "该频率规律与实测峰列更接近"),
        ("最佳解释模型", "与实测峰列最接近的频率规律"),
        ("倍频晶格", "倍频关系"),
        ("晶格拟合", "谱峰倍频拟合"),
        ("晶格", "倍频排列"),
        ("形成了以基频为锚点的连续支撑结构", "从基频起连续出现"),
        ("以基频为锚点", "从基频开始"),
        ("连续谐波支撑", "连续多阶倍频峰"),
        ("不通约", "倍频排列未见明显重合"),
        ("通约关系", "倍频重合关系"),
        ("通约", "在倍频排列上存在重合"),
    ):
        reason = reason.replace(term, wording)
    return reason


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")


def validate_reason_review(review):
    if set(review) != {"issues"} or not isinstance(review["issues"], list):
        raise ValueError("审查输出必须且仅包含 issues 数组")
    for issue in review["issues"]:
        if (not isinstance(issue, dict) or set(issue) != {"category", "quote", "problem", "suggestion"}
                or not all(isinstance(value, str) for value in issue.values())
                or issue["category"] not in REVIEW_CATEGORIES or not issue["problem"].strip() or not issue["suggestion"].strip()):
            raise ValueError("审查问题字段或类型不正确")
    return review["issues"]


class ExpertState(TypedDict, total=False):
    draft: dict
    checks: dict
    review: dict
    analysis: dict


def diagnose(record, evidence, client, trace=None, review_log=None, *, robust_periodicity=False, on_event=None):
    if type(robust_periodicity) is not bool:
        raise ValueError("robust_periodicity must be a boolean")
    audit = review_log if review_log is not None else {}

    def request(name, messages, **options):
        entry = {"step": name, "request": json.loads(json.dumps({"model": client.model, "messages": messages, **options}, ensure_ascii=False))}
        if trace is not None:
            trace.append(entry)
        try:
            options.setdefault("max_tokens", 4096)
            message = client.complete_messages(messages, **options)
            exchange = client.last_exchange or {}
            return SimpleNamespace(message=message, finish_reason=exchange.get("finish_reason"))
        finally:
            if trace is not None:
                entry["exchange"] = client.last_exchange

    def load_skill(path, system):
        return [{"role": "system", "content": system + "\n\n" + path.read_text(encoding="utf-8")}]

    def respond(name, messages):
        def read_response():
            choice = request(name, messages, response_format={"type": "json_object"})
            if choice.finish_reason != "stop":
                raise ValueError("专家输出未正常结束: "+str(choice.finish_reason))
            value = _parse_json(choice.message.content)
            if not isinstance(value, dict):
                raise ValueError("专家未返回 JSON 对象")
            return value

        messages_for_card = {"02_diagnose": "已生成诊断初稿。", "04_review": "已完成独立评审。",
                             "05_revise": "已完成诊断说明修订。"}
        return observe_call(on_event, name, {"model": client.model, "response_format": "JSON"},
                            read_response, kind="stage",
                            output_view=lambda _: {"message": messages_for_card[name]})

    messages = load_skill(SKILL_PATH, SYSTEM_PROMPT)
    payload = {"case_input": public_input(record), "evidence": {key: evidence[key] for key in
               ("tool_call_chain", "extracted_features", "frequency_matches")}}
    messages.append({"role": "user", "content": "依据以下证据完成一次判读，只返回 JSON：\n"+json.dumps(payload, ensure_ascii=False, allow_nan=False)})
    messages.append({"role": "user", "content": "请据上述证据交付最终JSON，仅含fault_type、reason。优先依据归属核验说明区分候选周期与故障支持；正常也是待比较的状态。只有已有证据支持状态异常、但不足以定位时用abnormal，并说明具体异常和定位限制；高包络峭度、零散候选峰或归属不确定本身不能单独判异常，不输出未知。reason用四段连续中文写成诊断意见，全文严格300—500字，建议350—430字，含标题、标点和引用。围绕2—3个关键证据点解释判断，不逐类罗列，不贴字段名、数组、布尔值或英文模型名称，括号内也不保留原文。保留必要数值与实际步骤引用。提交前检查字数、术语和结论一致性，超长先压缩，再输出最终结果。"})

    draft_messages = messages

    def draft_report(_: ExpertState):
        draft = respond("02_diagnose", messages)
        if (set(draft) != {"fault_type", "reason"} or not isinstance(draft["fault_type"], str)
                or draft["fault_type"] not in FAULT_NAMES or not isinstance(draft["reason"], str)):
            raise ValueError("诊断初稿字段或类型不正确")
        audit["initial_analysis"] = draft
        checks = {"reason_chars": len("".join(draft["reason"].split())), "errors": validate_analysis(draft, evidence)}
        audit["initial_checks"] = checks
        return {"draft": draft, "checks": checks}

    def review_report(state: ExpertState):
        draft, checks = state["draft"], state["checks"]
        reviewer = load_skill(REVIEW_SKILL_PATH, REVIEW_SYSTEM_PROMPT)
        reviewer.append({"role": "user", "content": "对照证据审查初稿，只返回规定的审查JSON：\n"+
                         json.dumps({**payload, "draft": draft, "checks": checks}, ensure_ascii=False, allow_nan=False)})
        review = respond("04_review", reviewer)
        validate_reason_review(review)
        audit["issues"] = review["issues"]
        return {"review": review}

    def revise_report(state: ExpertState):
        draft, checks, review = state["draft"], state["checks"], state["review"]
        messages = list(draft_messages)
        reconsider = robust_periodicity and any(issue["category"] == "分类疑点" for issue in review["issues"])
        messages.append({"role": "assistant", "content": json.dumps(draft, ensure_ascii=False)})
        messages.append({"role": "user", "content": "依据独立审查意见对照原始证据润色reason。审查意见也可能有误，合理意见逐条落实；无证据的建议不能采纳。"
                         "本轮仅修改说明，原fault_type固定为"+draft["fault_type"]+"，不可改判或编造支持。若存在分类疑点，保留具体限制供人工复核。"
                         "仍用四段自然中文，严格300—500字，修正术语、过度推断、量值和步骤引用。只返回含reason一个字段的JSON：\n"+
                         json.dumps({"review": review, "checks": checks}, ensure_ascii=False, allow_nan=False)})
        messages.append({"role": "user", "content": "交付前通读四段全文，尤其综合判断，不仅替换审查引用的局部原句。"
                         "同一术语在其他段落也要改写，全文不得出现晶格、通约、最佳解释模型或内部字段。"
                         "直接写实测峰是否接近理论倍频、峰间距是否符合预测、谱峰是否也能用转频倍频解释。"
                         "审查建议也须核对：分段均方根稳定不能证明有无周期性冲击；无冲击周期检测，就不写未形成连续冲击序列。"
                         "核验中的拟合基频偏差不能换成实测峰位超出容差，拟合间隔也不是残差。"
                         "不补写高斯分布、损伤阶段或所有干扰已排除。保留初稿类别，只输出reason字段，四段共300—500字。"})
        if reconsider:
            messages[-2]["content"] = ("依据审查意见对照原始证据重新审议分类及说明。审查也可能出错，可有据改判或维持原类别。"
                "本次允许修订fault_type；此前类别锁定与仅返回reason的要求不适用。只返回fault_type、reason两个字段。"
                "不读取或猜测真实标签；未定位不等于正常，有周期不等于异常或损伤，接近标称转频不等于证明旋转来源。"
                "periodicity为当前片段自身观测：半窗、自相关及不同频带不是独立测量；未测冲击事件不能宣称证实冲击。"
                "频率不相容和未建模旋转来源须如实保留，不改参数来归属故障。审查疑点仍保留供验收，不因改判自动解除。\n"
                +json.dumps({"review": review, "checks": checks}, ensure_ascii=False, allow_nan=False))
            messages[-1]["content"] = ("最终reason仍为四段300—500个非空白字符，引用实际步骤与数值，末段类别与fault_type一致。"
                "说明改判或维持的证据依据，不输出unknown，不为保留类别编造论据。")
        revised = respond("05_revise", messages)
        if reconsider:
            if (set(revised) != {"fault_type", "reason"} or not isinstance(revised["fault_type"], str)
                    or revised["fault_type"] not in FAULT_NAMES or not isinstance(revised["reason"], str)):
                raise ValueError("分类复议必须返回合法fault_type、reason")
            audit["classification_reconsideration"] = {"initial_fault_type": draft["fault_type"],
                "final_fault_type": revised["fault_type"], "changed": revised["fault_type"] != draft["fault_type"],
                "final_reason": revised["reason"], "unresolved_issues_retained": True}
            return {"analysis": revised}
        if (set(revised) - {"reason", "fault_type"} or not isinstance(revised.get("reason"), str)
                or revised.get("fault_type", draft["fault_type"]) != draft["fault_type"]):
            raise ValueError("润色输出必须包含 reason 文本，分类须与初稿一致")
        return {"analysis": {"fault_type": draft["fault_type"], "reason": revised["reason"]}}

    graph = StateGraph(ExpertState)
    graph.add_node("draft_report", draft_report)
    graph.add_node("review_report", review_report)
    graph.add_node("revise_report", revise_report)
    graph.add_edge(START, "draft_report")
    graph.add_edge("draft_report", "review_report")
    graph.add_edge("review_report", "revise_report")
    graph.add_edge("revise_report", END)
    return graph.compile().invoke({})["analysis"]
