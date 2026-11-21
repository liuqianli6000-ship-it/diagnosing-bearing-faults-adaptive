"""Portable v2 diagnosis. Default: offline; --provider explicitly enables model access."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from runtime import runtime
from validate_diagnosis_payload import validate


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def run(args):
    import numpy as np
    from bearing_suite_app import diagnosability
    from bearing_suite_app.config import load_config
    from bearing_suite_app.dataset_views import load_manifest, select_case
    from bearing_suite_app.llm_agent.controller import LLMController
    from bearing_suite_app.orchestration.builder import run_case
    from bearing_suite_app.toolbox.contracts import ToolContext
    from bearing_suite_app.toolbox.registry import build_default_registry
    from bearing_suite_app.visualization import save_signal_analysis_plot

    record = {}
    if args.manifest:
        manifest = Path(args.manifest).resolve()
        record = select_case(load_manifest(manifest, signal_root=args.data_root),
                             case_id=args.case_id, index=args.index)
        item = record["input"]
        if not args.domain and not all(item.get(name) is not None for name in ("sampling_rate_hz", "rpm", "bearing_geometry")):
            raise ValueError("Manifest requires sampling_rate_hz, rpm and bearing_geometry")
        signal_path = Path(item["signal_ref"])
        fs, rpm, geometry = item["sampling_rate_hz"], item.get("rpm"), item.get("bearing_geometry") or {}
        unit = item.get("unit") or "g"
        case_id = record.get("case_id") or signal_path.stem
    else:
        if args.fs is None or not args.unit or (not args.domain and
                (args.rpm is None or not (args.geometry or args.geometry_json))):
            raise ValueError("Provide --fs and --unit; diagnosis also requires --rpm and --geometry/--geometry-json")
        signal_path = Path(args.signal)
        fs, rpm = args.fs, args.rpm
        geometry = (json.loads(args.geometry_json) if args.geometry_json else
                    json.loads(args.geometry) if args.geometry and args.geometry.lstrip().startswith("{")
                    else read_json(args.geometry) if args.geometry else {})
        case_id = args.case_id or signal_path.stem
        unit = args.unit
    if not isinstance(geometry, dict):
        raise ValueError("Geometry must be a JSON object")
    if not all(math.isfinite(float(value)) and float(value) > 0 for value in ((fs,) if args.domain else (fs, rpm))):
        raise ValueError("Sampling rate and rpm must be finite and positive")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Output folder already exists; choose a new folder")
    signal = np.load(signal_path, allow_pickle=False)
    if args.domain:
        signal, normalization = diagnosability.normalize_real_series(signal)
        config = load_config(args.diagnosis_config)
        quality = diagnosability.assess_input_quality(
            signal, fs, rpm, config["signal"]["quality"], input_normalization=normalization, require_rpm=False)
        assessment = {"quality": quality}
        plots = []
        tools = {"time": "assess_time_domain", "frequency": "assess_frequency_domain",
                 "time_frequency": "assess_time_frequency"}
        if quality["valid_for_analysis"]:
            registry = build_default_registry()
            context = ToolContext(fs, rpm, geometry, {})
            for domain in dict.fromkeys(args.domain):
                result = registry.execute(tools[domain], signal, context, {})
                if result.status != "ok":
                    raise ValueError(result.error or f"{domain} analysis failed")
                assessment[domain] = result.data
                if domain != "time_frequency" or result.data["stft_status"] == "ok":
                    save_signal_analysis_plot(signal, fs, domain, output / "plots" / f"{domain}.png", unit=unit)
                    plots.append(domain)
        payload = {"schema_version": "signal-analysis/v1", "assessment": assessment, "analysis_plots": plots}
        output.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
        (output / "result.json").write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0
    offline = args.offline or args.provider is None
    result = run_case(
        signal, sampling_rate_hz=fs, rpm=rpm, geometry=geometry, case_id=str(case_id),
        provider=args.provider or "qwen", env_path=args.env_file, config_path=args.diagnosis_config,
        max_rounds=args.max_rounds, max_plan_rounds=args.max_plan_rounds,
        max_trials=args.max_trials, max_llm_calls=0 if offline else args.max_llm_calls,
        adaptive=args.adaptive, request_text=args.request, output_dir=output,
        controller=LLMController(client=None, max_calls=0) if offline else None,
        diagnosis_input=record.get("diagnosis_input"), unit=unit, plot=True,
    )
    payload = result.to_dict()
    errors = validate(payload)
    if errors:
        raise ValueError("Invalid v2 result: " + "; ".join(errors))
    print(payload["narrative"]["text"] if args.narrative else
          json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.status in {"completed", "limited"} else 1


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="Run the reviewed v2 graph; no source repository required")
    inputs = runner.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--signal")
    inputs.add_argument("--manifest")
    runner.add_argument("--fs", type=float)
    runner.add_argument("--rpm", type=float)
    runner.add_argument("--unit", choices=("g", "a.u."))
    runner.add_argument("--domain", nargs="+", choices=("time", "frequency", "time_frequency"),
                        help="Run only the selected signal analyses")
    geometry = runner.add_mutually_exclusive_group()
    geometry.add_argument("--geometry", help="Geometry JSON file or inline object")
    geometry.add_argument("--geometry-json", help="Inline geometry object")
    runner.add_argument("--case-id")
    runner.add_argument("--index", type=int, default=0)
    runner.add_argument("--data-root", help="Relative manifest references resolve here (default: manifest folder)")
    models = runner.add_mutually_exclusive_group()
    models.add_argument("--offline", action="store_true", help="Disable all model clients, even if API keys exist")
    models.add_argument("--provider", choices=("qwen", "deepseek", "openai"))
    runner.add_argument("--env-file")
    runner.add_argument("--diagnosis-config", help="Numeric diagnosis parameters JSON")
    mode = runner.add_mutually_exclusive_group()
    mode.add_argument("--adaptive", dest="adaptive", action="store_const", const=True, default=None)
    mode.add_argument("--no-adaptive", dest="adaptive", action="store_const", const=False)
    runner.add_argument("--request")
    runner.add_argument("--max-rounds", type=int, choices=(0, 1, 2))
    runner.add_argument("--max-plan-rounds", type=int, default=2, choices=(1, 2))
    runner.add_argument("--max-trials", type=int, default=3, choices=range(1, 11))
    runner.add_argument("--max-llm-calls", type=int, default=9, choices=range(0, 31))
    runner.add_argument("--narrative", action="store_true")
    runner.add_argument("--output", required=True, help="New output folder")
    checker = commands.add_parser("validate", help="Check v2 result/audit structure, not measurement authenticity")
    checker.add_argument("result")
    for old in ("collect", "finalize", "verify"):
        commands.add_parser(old, help="Retired v1 protocol; migrate to run / validate")
    if len(sys.argv) > 1 and sys.argv[1] in {"collect", "finalize", "verify"}:
        parser.error("v1 collect/finalize/verify are retired: use run --offline and validate <result.json>; "
                     "v2 uses numeric evidence and the Skill diagnosis criteria")
    args = parser.parse_args()
    if getattr(args, "env_file", None) and not args.provider:
        parser.error("--env-file requires an explicit --provider")
    try:
        if args.command == "validate":
            errors = validate(read_json(args.result))
            if errors:
                raise ValueError("; ".join(errors))
            print("Valid v2 result structure (not an authenticity or accuracy certification)")
            return 0
        with runtime():
            return run(args)
    except (ValueError, TypeError, KeyError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
