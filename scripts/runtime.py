"""Verify and load the plain-source runtime in a fresh process."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile

SKILL_ROOT = Path(__file__).resolve().parents[1]
MAX_UNPACKED_BYTES = 32 * 1024 * 1024


def validated_files(root: Path, inventory: dict[str, str]) -> dict[str, bytes]:
    names = list(inventory)
    if len(names) > 1024 or len(names) != len({name.casefold() for name in names}):
        raise ValueError("Runtime inventory exceeds limits or contains duplicate paths")
    entries = list(root.rglob("*"))
    if any(path.is_symlink() for path in entries):
        raise ValueError("Unsafe runtime symlink")
    if {path.relative_to(root).as_posix() for path in entries if path.is_file()} != set(names):
        raise ValueError("Runtime inventory mismatch")
    contents = {}
    total = 0
    for name in names:
        path = PurePosixPath(name)
        source = root / name
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name
                or "\\" in name or ":" in name or not source.resolve().is_relative_to(root)):
            raise ValueError("Unsafe runtime member")
        total += source.stat().st_size
        if total > MAX_UNPACKED_BYTES:
            raise ValueError("Runtime exceeds the loading limit")
        contents[name] = source.read_bytes()
        if hashlib.sha256(contents[name]).hexdigest() != inventory[name]:
            raise ValueError(f"Runtime member hash mismatch: {name}")
    return contents


@contextmanager
def runtime():
    manifest = json.loads((SKILL_ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))
    bundle = manifest["runtime_bundle"]
    source = SKILL_ROOT / bundle["path"]
    if source.is_symlink() or not source.resolve().is_relative_to(SKILL_ROOT) or not source.is_dir():
        raise ValueError("Runtime directory must be inside the skill folder")
    if any(name == "bearing_suite_app" or name.startswith("bearing_suite_app.") for name in sys.modules):
        raise RuntimeError("Run the skill entrypoint in a fresh Python process")
    contents = validated_files(source.resolve(), bundle["files"])
    with tempfile.TemporaryDirectory(prefix="bearing-v2-skill-") as directory:
        root = Path(directory)
        for name, data in contents.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        resources = root / "skills" / "diagnosing-bearing-faults-adaptive"
        # Excluding the manifest from its own inventory avoids recursive hashes.
        (resources / "runtime_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path = str(root / "src")
        sys.path.insert(0, path)
        try:
            yield root
        finally:
            sys.path.remove(path)
