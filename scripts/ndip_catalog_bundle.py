#!/usr/bin/env python3
"""Build and validate the immutable built-in catalog bundle used by NDIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DATABASES = ("database_neutron", "database_xray")
REQUIRED_RELATIVE_PATHS = {
    "database_neutron": (
        "catalog_deduplicated.csv",
        "mp_experimental_stable.csv",
        "highsymm_metadata.json",
        "profiles64/profiles64.npz",
        "profiles64/index.csv",
    ),
    "database_xray": (
        "catalog_deduplicated.csv",
        "mp_experimental_stable.csv",
        "highsymm_metadata.json",
        "profiles64.npz",
        "profiles64/index.csv",
    ),
}
SHARED_METADATA = Path("database_xray/highsymm_metadata.json")
SHARED_METADATA_SOURCE = Path("database_neutron/highsymm_metadata.json")


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_paths(root: Path) -> Iterable[Path]:
    for database, paths in REQUIRED_RELATIVE_PATHS.items():
        for relative in paths:
            yield root / database / relative


def validate_source(root: Path) -> None:
    missing = [str(path) for path in _required_paths(root) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required catalog files:\n" + "\n".join(missing))

    neutron_hash = _sha256(root / SHARED_METADATA_SOURCE)
    xray_hash = _sha256(root / SHARED_METADATA)
    if neutron_hash != xray_hash:
        raise ValueError(
            "Neutron and X-ray highsymm_metadata.json differ; refusing to deduplicate them"
        )


def _iter_bundle_files(root: Path) -> Iterable[Path]:
    for database in DATABASES:
        database_root = root / database
        for path in sorted(database_root.rglob("*")):
            if not path.is_file():
                continue
            if path.relative_to(root) == SHARED_METADATA:
                continue
            yield path


def build_bundle(root: Path, output: Path, version: str) -> dict:
    root = root.resolve()
    output = output.resolve()
    validate_source(root)

    files = list(_iter_bundle_files(root))
    manifest_files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    manifest = {
        "schema": "radar-pd-catalog-bundle/v1",
        "version": version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root_name": root.name,
        "shared_metadata": {
            "canonical": SHARED_METADATA_SOURCE.as_posix(),
            "linked_path": SHARED_METADATA.as_posix(),
            "sha256": _sha256(root / SHARED_METADATA_SOURCE),
        },
        "files": manifest_files,
        "uncompressed_bytes": sum(item["size"] for item in manifest_files),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_name)
    try:
        with tarfile.open(temp_path, "w:gz", compresslevel=6) as archive:
            for path in files:
                archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)

            payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n"
            info = tarfile.TarInfo("catalog_manifest.json")
            info.size = len(payload)
            info.mode = 0o644
            info.mtime = int(datetime.now(timezone.utc).timestamp())
            archive.addfile(info, io.BytesIO(payload))
        temp_path.replace(output)
    finally:
        temp_path.unlink(missing_ok=True)

    return manifest


def inspect_bundle(path: Path) -> dict:
    with tarfile.open(path, "r:gz") as archive:
        member = archive.getmember("catalog_manifest.json")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("Catalog archive has no readable manifest")
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a catalog source tree")
    validate_parser.add_argument("--source", required=True, type=Path)

    build_parser = subparsers.add_parser("build", help="build a catalog tar.gz bundle")
    build_parser.add_argument("--source", required=True, type=Path)
    build_parser.add_argument("--output", required=True, type=Path)
    build_parser.add_argument("--version", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print a bundle manifest")
    inspect_parser.add_argument("--archive", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "validate":
        validate_source(args.source)
        print(f"Catalog source is valid: {args.source}")
        return 0
    if args.command == "build":
        manifest = build_bundle(args.source, args.output, args.version)
        print(
            f"Wrote {args.output} with {len(manifest['files'])} files "
            f"({manifest['uncompressed_bytes']} uncompressed bytes)"
        )
        return 0

    print(json.dumps(inspect_bundle(args.archive), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
