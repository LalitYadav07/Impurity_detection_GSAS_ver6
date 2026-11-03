"""Copy a built, reviewed guide bundle for the public application image.

Build help/ with build_user_guide.py first, then pass that directory and the
destination. Only manifest-listed files are copied. No CIF or source-CIF ZIP is
redistributed; the dated reference and current guide travel together.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def safe_file(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe guide path: {name}")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Guide file escapes its bundle: {name}")
    return target


def package(source: Path, destination: Path) -> None:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "radar-pd-user-guide/v2":
        raise ValueError("Build the current task guide with build_user_guide.py before packaging")
    if manifest.get("licensed_cif_files_redistributed") is not False:
        raise ValueError("Guide must not redistribute licensed CIFs")
    for name, digest in manifest["files"].items():
        path = safe_file(source, name)
        if path.suffix.lower() in {".cif", ".zip"}:
            raise ValueError(f"Scientific source archive is not a public guide asset: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Guide manifest is stale: {name}")
    if source.resolve() == destination.resolve():
        print(json.dumps({"destination": str(destination), "status": "verified in place"}))
        return
    old_files = {}
    if (destination / "manifest.json").is_file():
        old_files = json.loads((destination / "manifest.json").read_text()).get("files", {})
    destination.mkdir(parents=True, exist_ok=True)
    for name in manifest["files"]:
        target = safe_file(destination, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(safe_file(source, name), target)
    shutil.copyfile(source / "manifest.json", destination / "manifest.json")
    # Remove only obsolete assets owned by the prior bundle's manifest.
    for name in old_files.keys() - manifest["files"].keys():
        safe_file(destination, name).unlink(missing_ok=True)
    print(json.dumps({"destination": str(destination), "application_version": manifest["application_version"],
                      "pages": len(manifest["pages"]), "files": len(manifest["files"]),
                      "screenshots": manifest["screenshots"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    package(args.source, args.destination)
