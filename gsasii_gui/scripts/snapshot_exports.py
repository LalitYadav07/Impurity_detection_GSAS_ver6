#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path


def _manifest(root: Path) -> list[dict[str, int | str]]:
    entries: list[dict[str, int | str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return entries


def snapshot(export_dir: Path, archive: Path, state_file: Path) -> bool:
    export_dir.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    state_file.parent.mkdir(parents=True, exist_ok=True)

    before = _manifest(export_dir)
    try:
        previous = json.loads(state_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        previous = None
    if archive.is_file() and before == previous:
        return False

    fd, temporary_name = tempfile.mkstemp(prefix=f".{archive.name}.", dir=archive.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as bundle:
            for entry in before:
                source = export_dir / str(entry["path"])
                bundle.write(source, arcname=str(entry["path"]))
        after = _manifest(export_dir)
        if before != after:
            return False
        os.replace(temporary, archive)
        archive.chmod(0o644)
        state_file.write_text(json.dumps(after, sort_keys=True), encoding="utf-8")
        print(f"Saved {len(after)} exported file(s) to the Galaxy archive")
        return True
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: snapshot_exports.py EXPORT_DIR ARCHIVE STATE_FILE", file=sys.stderr)
        return 64
    snapshot(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
