"""Verify and load the bundled v2 implementation in a fresh process."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys
import tempfile
import zipfile

SKILL_ROOT = Path(__file__).resolve().parents[1]
MAX_UNPACKED_BYTES = 32 * 1024 * 1024


def validate_archive(archive: zipfile.ZipFile, inventory: dict[str, str]) -> None:
    entries = archive.infolist()
    if len(entries) > 1024 or sum(item.file_size for item in entries) > MAX_UNPACKED_BYTES:
        raise ValueError("Runtime archive exceeds the extraction limit")
    names = [item.filename for item in entries]
    if len(names) != len({name.casefold() for name in names}) or set(names) != set(inventory):
        raise ValueError("Runtime archive inventory mismatch")
    for entry in entries:
        name = entry.filename
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or path.as_posix() != name or "\\" in name or ":" in name
                or stat.S_ISLNK(entry.external_attr >> 16) or entry.is_dir()):
            raise ValueError("Unsafe runtime archive member")
        if hashlib.sha256(archive.read(entry)).hexdigest() != inventory[name]:
            raise ValueError(f"Runtime member hash mismatch: {name}")


@contextmanager
def runtime():
    manifest = json.loads((SKILL_ROOT / "runtime_manifest.json").read_text(encoding="utf-8"))
    bundle = manifest["runtime_bundle"]
    archive_path = (SKILL_ROOT / bundle["path"]).resolve()
    if SKILL_ROOT not in archive_path.parents:
        raise ValueError("Runtime bundle must be inside the skill folder")
    if hashlib.sha256(archive_path.read_bytes()).hexdigest() != bundle["sha256"]:
        raise ValueError("Runtime bundle hash mismatch; rebuild the reviewed v2 bundle")
    if any(name == "bearing_suite_app" or name.startswith("bearing_suite_app.") for name in sys.modules):
        raise RuntimeError("Run the skill entrypoint in a fresh Python process")
    with tempfile.TemporaryDirectory(prefix="bearing-v2-skill-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(archive_path) as archive:
            validate_archive(archive, bundle["files"])
            archive.extractall(root)
        resources = root / "skills" / "diagnosing-bearing-faults-adaptive"
        # Excluding the manifest from its own archive avoids a recursive hash.
        (resources / "runtime_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        path = str(root / "src")
        sys.path.insert(0, path)
        try:
            yield root
        finally:
            sys.path.remove(path)
