"""Validate v2 compact results or result.json audits; does not verify authenticity."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

FAULT_TYPES = {"outer_race_fault", "inner_race_fault", "rolling_element_fault", "normal", "unknown"}


def validate(payload):
    errors = []
    if not isinstance(payload, dict):
        return ["Result must be an object"]
    if payload.get("schema_version") == "signal-analysis/v1":
        return [] if isinstance(payload.get("assessment"), dict) and isinstance(payload.get("analysis_plots"), list) \
            else ["Invalid signal analysis result"]
    if payload.get("schema_version") == "dataset-diagnosis/v1":
        chain = payload.get("evidence_chain")
        if not isinstance(chain, dict) or set(chain) != {
                "tool_call_chain", "extracted_features", "frequency_matches", "analysis_text"}:
            return ["Invalid dataset evidence chain"]
        if any(not isinstance(chain[key], list) for key in ("tool_call_chain", "extracted_features", "frequency_matches")):
            errors.append("Numeric evidence sections must be arrays")
        analysis = chain["analysis_text"]
        fault = (payload.get("final") or {}).get("fault_type")
        if analysis is not None and (not isinstance(analysis, dict) or analysis.get("fault_type") != fault
                or fault not in (FAULT_TYPES - {"unknown"}) | {"abnormal"}
                or not isinstance(analysis.get("reason"), str)):
            errors.append("Invalid or inconsistent final diagnosis")
        if analysis is None and fault is not None:
            errors.append("Unfinished diagnosis must have no fault_type")
        if payload.get("status") != ("limited" if analysis is None else "completed"):
            errors.append("Diagnosis status does not match the result")
        rounds = payload.get("rounds")
        if type(rounds) is not int or not 0 <= rounds <= (2 if payload.get("adaptive") else 0):
            errors.append("Invalid number of processing rounds")
        return errors
    if payload.get("status") not in {"completed", "limited"}:
        errors.append("Invalid or missing v2 status")
    final = payload.get("final")
    if not isinstance(final, dict):
        return errors + ["Missing v2 final object (v1 payloads are unsupported)"]
    fault = final.get("fault_type")
    if fault not in FAULT_TYPES:
        errors.append("Invalid final.fault_type")
    confidence = final.get("confidence")
    valid_confidence = (type(confidence) in (int, float) and math.isfinite(confidence)
                        and 0 <= confidence <= 1)
    if not valid_confidence:
        errors.append("final.confidence must be finite and within [0, 1]")
    if fault == "unknown" and confidence != 0:
        errors.append("unknown requires zero confidence")
    if payload.get("status") == "limited" and fault != "unknown":
        errors.append("limited requires unknown")
    if not isinstance(payload.get("adaptive"), bool):
        errors.append("Missing boolean adaptive mode")
    rounds = payload.get("rounds")
    if type(rounds) is not int or not 0 <= rounds <= 2:
        errors.append("rounds must be 0..2")
    elif not payload.get("adaptive") and rounds:
        errors.append("Stable mode cannot contain adaptive rounds")
    assessment = payload.get("assessment", final.get("assessment"))
    if not isinstance(assessment, dict):
        errors.append("Missing signal assessment")
    else:
        if "fault_type" in assessment or "confidence" in assessment:
            errors.append("Assessment must not contain a diagnosis")
        quality = assessment.get("quality") or {}
        if not isinstance(quality, dict):
            errors.append("Assessment quality must be an object")
            quality = {}
        if payload.get("status") == "completed" and (
                assessment.get("schema_version") != "signal-assessment/v1" or
                quality.get("valid_for_analysis") is not True):
            errors.append("Completed result requires a valid v1 signal assessment")
        if quality.get("valid_for_analysis") is False and fault != "unknown":
            errors.append("Failed quality gate cannot produce a specific diagnosis")
    numeric = payload.get("numeric_result") or final.get("numeric") or {}
    if not isinstance(numeric, dict):
        errors.append("Numeric result must be an object")
        numeric = {}
    if fault in FAULT_TYPES - {"unknown"} and not numeric.get("fault_type"):
        errors.append("Concrete result requires numeric diagnosis")
    source = final.get("source")
    if source == "skill_llm":
        if not final.get("evidence_refs") or not final.get("reasoning"):
            errors.append("Skill model verdict requires evidence references and reasoning")
    elif source == "evidence_only":
        if fault != "unknown":
            errors.append("Evidence-only result requires unknown")
    elif numeric.get("fault_type") and numeric["fault_type"] != fault:
        errors.append("Final fault must match numeric fault")
    # Historical v2 audits can include the former visual validation fields.
    visual = payload.get("visual_validation") or final.get("visual_validation") or {}
    if not isinstance(visual, dict):
        errors.append("Visual validation must be an object")
        visual = {}
    if visual.get("status") == "disagree" and valid_confidence and confidence > 0.5:
        errors.append("Visual disagreement caps confidence at 0.5")
    narrative = payload.get("narrative")
    if not isinstance(narrative, dict) or not isinstance(narrative.get("text"), str):
        errors.append("Missing narrative.text")
    return errors


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("Usage: validate_diagnosis_payload.py <v2-result.json>")
        failures = validate(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
        if failures:
            raise ValueError("; ".join(failures))
        print("Valid v2 result structure; authenticity and accuracy not evaluated")
    except (OSError, ValueError, TypeError) as error:
        raise SystemExit(str(error))
