import tarfile
import json
from pathlib import Path

import pytest

from scripts.ndip_catalog_bundle import (
    build_bundle, inspect_bundle, validate_source, verify_installed_catalog, require_runtime_catalog,
    prepare_runtime_layout,
)


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


def installed_catalog(tmp_path):
    source = tmp_path / 'catalog'
    _write_catalog(source)
    manifest = build_bundle(source, tmp_path / 'catalog.tar.gz', 'test-v1')
    (source / 'catalog_manifest.json').write_text(json.dumps(manifest))
    prepare_runtime_layout(source)
    return source, manifest


def test_installed_catalog_verifies_manifest_and_both_runtime_roots(tmp_path):
    source, manifest = installed_catalog(tmp_path)
    receipt = verify_installed_catalog(source)
    assert receipt['version'] == 'test-v1' and receipt['content_hashes_verified']
    for radiation in ('neutron', 'xray'):
        require_runtime_catalog(source / ('database_' + radiation), radiation)


def test_legacy_xray_fingerprint_link_preserves_manifest_bytes(tmp_path):
    source, _ = installed_catalog(tmp_path)
    alias = source / 'database_xray/profiles64/profiles64.npz'
    assert alias.is_symlink() and alias.read_bytes() == b'xray profiles'
    assert verify_installed_catalog(source)['content_hashes_verified']


def test_runtime_xray_accepts_current_layout_without_legacy_duplicate(tmp_path):
    source, _ = installed_catalog(tmp_path)
    alias = source / 'database_xray/profiles64/profiles64.npz'
    content = alias.read_bytes()
    alias.unlink()
    alias.write_bytes(content)
    (source / 'database_xray/profiles64.npz').unlink()
    require_runtime_catalog(source / 'database_xray', 'xray')


def test_shared_metadata_link_cannot_escape_package(tmp_path):
    source, _ = installed_catalog(tmp_path)
    outside = tmp_path / 'outside.json'
    outside.write_bytes((source / 'database_neutron/highsymm_metadata.json').read_bytes())
    link = source / 'database_xray/highsymm_metadata.json'
    link.unlink()
    link.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes the catalog'):
        verify_installed_catalog(source)


def test_installed_catalog_rejects_same_size_changed_content(tmp_path):
    source, _ = installed_catalog(tmp_path)
    path = source / 'database_neutron/catalog_deduplicated.csv'
    path.write_bytes(path.read_bytes().replace(b'Cu', b'Fe'))
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_installed_catalog(source)


def test_installed_catalog_rejects_unregistered_required_asset(tmp_path):
    source, manifest = installed_catalog(tmp_path)
    manifest['files'] = [x for x in manifest['files'] if x['path'] != 'database_neutron/profiles64/index.csv']
    (source / 'catalog_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='not in the manifest'):
        verify_installed_catalog(source)


@pytest.mark.parametrize('path', ['', '../outside', '/absolute', 'database_neutron/../../outside'])
def test_installed_catalog_rejects_unsafe_manifest_path(tmp_path, path):
    source, manifest = installed_catalog(tmp_path)
    manifest['files'][0]['path'] = path
    (source / 'catalog_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest path'):
        verify_installed_catalog(source)


def test_runtime_catalog_fails_clearly_on_empty_volume(tmp_path):
    with pytest.raises(FileNotFoundError, match='No scientific analysis was started'):
        require_runtime_catalog(tmp_path / 'database_neutron', 'neutron')


def test_runtime_catalog_rejects_empty_required_asset(tmp_path):
    source, _ = installed_catalog(tmp_path)
    (source / 'database_neutron/profiles64/index.csv').write_bytes(b'')
    with pytest.raises(FileNotFoundError, match='catalog is unavailable') as caught:
        require_runtime_catalog(source / 'database_neutron', 'neutron')
    assert caught.value.catalog_diagnostic['missing_or_empty'] == ['profiles64/index.csv']
