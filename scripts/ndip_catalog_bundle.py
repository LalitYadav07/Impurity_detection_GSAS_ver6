#!/usr/bin/env python3
"""Build and validate the immutable built-in catalog bundle used by NDIP."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
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
RUNTIME_ALIASES = {
    Path('database_xray/profiles64/profiles64.npz'): Path('database_xray/profiles64.npz'),
}


def prepare_runtime_layout(root: Path) -> None:
    """Expose legacy packaged fingerprints at the current loader's path."""
    root = root.resolve()
    for alias, canonical in RUNTIME_ALIASES.items():
        path = root / alias
        if path.exists():
            continue
        source = root / canonical
        if not source.is_file():
            raise ValueError('Packaged catalog has no fingerprint source: ' + str(canonical))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(os.path.relpath(source, path.parent))


class BuiltinCatalogUnavailable(FileNotFoundError):
    """Expected readiness failure, with technical details for the saved log."""
    def __init__(self, database_root: Path, radiation: str, missing: list[str]):
        self.catalog_diagnostic = {'schema': 'radar-pd-catalog-readiness/v1', 'ready': False,
                                   'radiation': radiation, 'database_root': str(database_root),
                                   'missing_or_empty': missing}
        super().__init__(f'Built-in {radiation} catalog is unavailable on this worker. '
                         'No scientific analysis was started; restore the worker\'s packaged catalog '
                         'or select a complete custom library.')


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


def verify_installed_catalog(root: Path, *, verify_hashes: bool = True) -> dict:
    """Verify the immutable package, including its manifest, before release."""
    root = root.resolve()
    manifest_path = root / 'catalog_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'radar-pd-catalog-bundle/v1' or not manifest.get('version'):
        raise ValueError('Unsupported or unversioned built-in catalog manifest')
    entries = manifest.get('files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('Built-in catalog manifest has no files')
    registered = {}
    for entry in entries:
        relative = entry.get('path')
        if not isinstance(relative, str) or not relative or not Path(relative).parts:
            raise ValueError('Invalid built-in catalog manifest path')
        path = root / relative
        if (relative in registered or Path(relative).is_absolute() or '..' in Path(relative).parts
                or Path(relative).parts[0] not in DATABASES
                or not path.resolve().is_relative_to(root)):
            raise ValueError('Unsafe or duplicate built-in catalog manifest path')
        if (type(entry.get('size')) is not int or entry['size'] < 0
                or not re.fullmatch(r'[a-f0-9]{64}', str(entry.get('sha256', '')))):
            raise ValueError('Invalid built-in catalog manifest size or checksum')
        if not path.is_file() or path.stat().st_size != entry['size']:
            raise ValueError('Built-in catalog file missing or size changed: ' + relative)
        if verify_hashes and _sha256(path) != entry['sha256']:
            raise ValueError('Built-in catalog checksum mismatch: ' + relative)
        registered[relative] = entry
    for path in _required_paths(root):
        relative = path.relative_to(root)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Required built-in catalog asset is absent or empty: ' + str(relative))
        canonical = SHARED_METADATA_SOURCE if relative == SHARED_METADATA else relative
        if canonical.as_posix() not in registered:
            raise ValueError('Required built-in catalog asset is not in the manifest: ' + str(relative))
    shared = manifest.get('shared_metadata') or {}
    if (shared.get('canonical') != SHARED_METADATA_SOURCE.as_posix()
            or shared.get('linked_path') != SHARED_METADATA.as_posix()
            or shared.get('sha256') != registered[SHARED_METADATA_SOURCE.as_posix()]['sha256']):
        raise ValueError('Built-in shared metadata manifest is inconsistent')
    if not (root / SHARED_METADATA).resolve().is_relative_to(root):
        raise ValueError('Built-in shared metadata link escapes the catalog')
    if verify_hashes and _sha256(root / SHARED_METADATA) != shared['sha256']:
        raise ValueError('Built-in shared metadata content differs')
    for alias, canonical in RUNTIME_ALIASES.items():
        path = root / alias
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError('Required runtime catalog path is absent or unsafe: ' + str(alias))
        if alias.as_posix() not in registered:
            source = registered.get(canonical.as_posix())
            if not source or path.stat().st_size != source['size'] or (
                    verify_hashes and _sha256(path) != source['sha256']):
                raise ValueError('Runtime catalog alias differs from its manifest source: ' + str(alias))
    return {'schema': 'radar-pd-installed-catalog/v1', 'version': manifest['version'],
            'manifest_sha256': _sha256(manifest_path), 'files': len(registered),
            'content_hashes_verified': verify_hashes}


def require_runtime_catalog(database_root: Path, radiation: str) -> None:
    """Cheap per-job readiness check; full package hashes are verified in CI."""
    if radiation not in ('neutron', 'xray'):
        raise ValueError('Unsupported built-in catalog radiation')
    required = list(REQUIRED_RELATIVE_PATHS['database_' + radiation])
    if radiation == 'xray':
        required[required.index('profiles64.npz')] = 'profiles64/profiles64.npz'
    missing = [name for name in required
               if not (database_root / name).is_file() or (database_root / name).stat().st_size == 0]
    if missing:
        raise BuiltinCatalogUnavailable(database_root, radiation, missing)


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

    installed_parser = subparsers.add_parser('verify-installed', help='verify every packaged file against its manifest')
    installed_parser.add_argument('--source', required=True, type=Path)
    layout_parser = subparsers.add_parser('prepare-runtime-layout', help='link legacy packaged fingerprints without changing their contents')
    layout_parser.add_argument('--source', required=True, type=Path)

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

    if args.command == 'verify-installed':
        print(json.dumps(verify_installed_catalog(args.source), sort_keys=True))
        return 0
    if args.command == 'prepare-runtime-layout':
        prepare_runtime_layout(args.source)
        return 0

    print(json.dumps(inspect_bundle(args.archive), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
