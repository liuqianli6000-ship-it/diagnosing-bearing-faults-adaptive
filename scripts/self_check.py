"""Offline integration check using only a copied skill and caller-generated data."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    import numpy as np
    source = Path(__file__).resolve().parents[1]
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if temporary_root.is_relative_to(source):
        temporary_root = source.parent
    with tempfile.TemporaryDirectory(prefix="bearing-v2-skill-check-", dir=temporary_root) as temporary:
        directory = Path(temporary)
        skill = directory / source.name
        shutil.copytree(source, skill, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        entry = skill / "scripts" / "diagnose.py"
        environment = dict(os.environ, PYTHONPATH="", PYTHONIOENCODING="utf-8",
                           DASHSCOPE_API_KEY="offline-must-not-use-this", OPENAI_API_KEY="offline-must-not-use-this")

        def command(*arguments, success=True):
            process = subprocess.run(
                [sys.executable, "-E", "-X", "utf8", str(entry), *map(str, arguments)],
                cwd=directory, env=environment, capture_output=True, encoding="utf-8", timeout=90,
            )
            assert (process.returncode == 0) == success, process.stdout + process.stderr
            return process

        fs, rpm = 20000.0, 1800.0
        geometry = {"n": 8, "d": 7.94, "D": 39.04, "contact_angle_degrees": 0.0}
        t = np.arange(int(2 * fs)) / fs
        frequency = 0.5 * geometry["n"] * (1 - geometry["d"] / geometry["D"]) * rpm / 60
        period = int(round(fs / frequency))
        train = np.exp(-(np.arange(t.size) % period) / (fs * 0.0008))
        signal = 0.8 * train * np.sin(2 * np.pi * 3000 * t)
        signal += np.random.default_rng(42).normal(0, 0.04, t.size)
        np.save(directory / "signal.npy", signal)
        common = ["--signal", directory / "signal.npy", "--fs", fs, "--rpm", rpm, "--unit", "g",
                  "--geometry-json", json.dumps(geometry), "--max-trials", 1]
        for adaptive in (False, True):
            output = directory / ("adaptive" if adaptive else "stable")
            process = command("run", "--offline", *common, *( ["--adaptive"] if adaptive else []),
                              "--output", output)
            result = json.loads(process.stdout)
            assert result["status"] in {"completed", "limited"}
            assert result["adaptive"] is adaptive and result["llm_calls"] == 0
            assert result["final"]["fault_type"] is None
            assert result["final"]["source"] == "evidence_only"
            assert result["final"]["model_error"] is None
            assert result["rounds"] <= 2
            audit = json.loads((output / "result.json").read_text(encoding="utf-8"))
            assert audit["evidence_chain"]["frequency_matches"]
            assert audit["expert_calls"] == []
            assert list((output / "plots").glob("*.png"))
            command("validate", output / "result.json")
        print("PASS: isolated stable/adaptive evidence, plots, zero model calls and audit validation")
        np.save(directory / "constant.npy", np.ones(2048))
        limited = command("run", "--offline", "--signal", directory / "constant.npy", "--fs", fs,
                          "--rpm", rpm, "--unit", "g", "--geometry-json", json.dumps(geometry),
                          "--output", directory / "limited")
        result = json.loads(limited.stdout)
        assert result["status"] == "limited" and result["final"]["fault_type"] is None
        assert result["llm_calls"] == 0
        command("validate", directory / "limited" / "result.json")
        domain = directory / "domain"
        command("run", "--offline", *common, "--domain", "time", "--output", domain)
        assert {p.name for p in (domain / "plots").glob("*.png")} == {"time.png"}
        command("validate", domain / "result.json")
        print("PASS: invalid quality keeps no verdict; a time-only request generates its plot")
        for old in ("collect", "finalize", "verify"):
            failure = command(old, success=False)
            assert "retired" in failure.stderr
        archive = skill / "assets" / "runtime.zip"
        archive.write_bytes(archive.read_bytes() + b"tampered")
        failure = command("run", "--offline", *common, "--output", directory / "tampered", success=False)
        assert "hash mismatch" in failure.stderr
        assert not (directory / "tampered").exists()
        print("PASS: v1 commands reject explicitly and runtime tampering fails before diagnosis")


if __name__ == "__main__":
    main()
