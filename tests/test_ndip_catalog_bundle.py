import tarfile
from pathlib import Path

import pytest

from scripts.ndip_catalog_bundle import build_bundle, inspect_bundle, validate_source


def _write_catalog(root: Path, *, xray_metadata: bytes = b"shared metadata") -> None:
    files = {
        "database_neutron/catalog_deduplicated.csv": b"id,formula\n1,Cu\n",
        "database_neutron/mp_experimental_stable.csv": b"id\n1\n",
        "database_neutron/highsymm_metadata.json": b"shared metadata",
        "database_neutron/profiles64/profiles64.npz": b"neutron profiles",
        "database_neutron/profiles64/index.csv": b"id\n1\n",
        "database_xray/catalog_deduplicated.csv": b"id,formula\n2,Fe\n",
        "database_xray/mp_experimental_stable.csv": b"id\n2\n",
        "database_xray/highsymm_metadata.json": xray_metadata,
        "database_xray/profiles64.npz": b"xray profiles",
        "database_xray/profiles64/index.csv": b"id\n2\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def test_bundle_excludes_duplicate_metadata_and_records_manifest(tmp_path: Path) -> None:
    source = tmp_path / "catalog"
    _write_catalog(source)
    output = tmp_path / "catalog.tar.gz"

    manifest = build_bundle(source, output, "test-v1")

    assert manifest["version"] == "test-v1"
    assert inspect_bundle(output)["shared_metadata"]["linked_path"] == (
        "database_xray/highsymm_metadata.json"
    )
    with tarfile.open(output, "r:gz") as archive:
        names = set(archive.getnames())
    assert "database_neutron/highsymm_metadata.json" in names
    assert "database_xray/highsymm_metadata.json" not in names
    assert "catalog_manifest.json" in names


def test_validate_rejects_nonidentical_shared_metadata(tmp_path: Path) -> None:
    source = tmp_path / "catalog"
    _write_catalog(source, xray_metadata=b"different")

    with pytest.raises(ValueError, match="refusing to deduplicate"):
        validate_source(source)


def test_validate_reports_missing_required_file(tmp_path: Path) -> None:
    source = tmp_path / "catalog"
    _write_catalog(source)
    (source / "database_xray/profiles64.npz").unlink()

    with pytest.raises(FileNotFoundError, match="profiles64.npz"):
        validate_source(source)
