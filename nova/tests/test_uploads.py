import io
import zipfile
from pathlib import Path

import pytest

from radar_pd_nova.uploads import (
    build_cif_source_archive,
    browser_library_upload_js,
    browser_named_file_js,
    display_filename,
    inspect_cif_archive,
    inspect_cif_upload,
    safe_client_filename,
    store_browser_upload,
)


VALID_CIF = b"""data_Fe
_chemical_formula_sum 'Fe'
_space_group_IT_number 229
_cell_length_a 2.86
_cell_length_b 2.86
_cell_length_c 2.86
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""


def test_safe_client_filename_removes_browser_paths_and_unsafe_characters() -> None:
    assert safe_client_filename(r"C:\fakepath\TbSSL.cif") == "TbSSL.cif"
    assert safe_client_filename("../../hb2a_si_ge113.instprm") == "hb2a_si_ge113.instprm"
    assert safe_client_filename("sample/<bad>.dat") == "_bad_.dat"


def test_store_browser_upload_preserves_name_suffix_and_bytes() -> None:
    contents = b"data_TbSSL\n_cell_length_a 4.1\n"

    target = store_browser_upload(contents, r"C:\fakepath\TbSSL.cif")

    assert isinstance(target, Path)
    assert target.name == "TbSSL.cif"
    assert target.suffix == ".cif"
    assert target.read_bytes() == contents


def test_display_filename_never_exposes_the_temporary_or_windows_path() -> None:
    assert display_filename(r"C:\\fakepath\\HB2A_TbSSL.dat") == "HB2A_TbSSL.dat"
    assert display_filename("/tmp/radar_pd_upload_123/hb2a_si_ge113.instprm") == "hb2a_si_ge113.instprm"
    assert display_filename("") == ""


def test_inspect_cif_upload_reports_formula_space_group_and_digest() -> None:
    result = inspect_cif_upload(VALID_CIF, r"C:\fakepath\Fe.cif")

    assert result["name"] == "Fe.cif"
    assert result["formula"] == "Fe"
    assert result["space_group"] == "229"
    assert len(result["digest"]) == 64


@pytest.mark.parametrize(
    ("contents", "name", "message"),
    [
        (b"", "empty.cif", "empty"),
        (b"data_x\n_cell_length_a 1\n", "broken.cif", "missing unit-cell"),
        (b"data_x\n", "wrong.txt", "expected a .cif"),
    ],
)
def test_inspect_cif_upload_rejects_obviously_unusable_inputs(contents: bytes, name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        inspect_cif_upload(contents, name)


def test_inspect_cif_archive_reports_usable_and_rejected_members() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("structures/Fe.cif", VALID_CIF)
        archive.writestr("structures/broken.cif", b"data_broken\n")
        archive.writestr("notes.txt", b"ignored")

    result = inspect_cif_archive(payload.getvalue(), "candidate_structures.zip")

    assert result["source_type"] == "ZIP archive"
    assert result["cif_count"] == 1
    assert result["rejected_count"] == 1
    assert Path(result["path"]).is_file()


def test_build_cif_source_archive_deduplicates_loose_and_zipped_cifs(tmp_path: Path) -> None:
    loose = tmp_path / "Fe.cif"
    loose.write_bytes(VALID_CIF)
    source_zip = tmp_path / "more.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr("duplicate_Fe.cif", VALID_CIF)
        archive.writestr("bad.cif", b"data_bad\n")

    bundle, stats = build_cif_source_archive([str(loose), str(source_zip)], "Fe candidates")

    assert stats["cif_count"] == 1
    assert stats["skipped_count"] == 2
    with zipfile.ZipFile(bundle) as archive:
        assert len([name for name in archive.namelist() if name.endswith(".cif")]) == 1


def test_file_upload_handler_uses_supported_trame_file_event() -> None:
    handler = browser_named_file_js("decode_source")

    assert "Array.isArray(value)" in handler
    assert "file.arrayBuffer()" in handler
    assert "trigger('decode_source', [contents, file.name])" in handler
    assert "$event.target.files" not in handler


def test_library_upload_handler_accepts_single_or_multiple_files_and_uses_json_safe_transport() -> None:
    handler = browser_library_upload_js("decode_archive", "mark_upload", "ZIP archive")

    assert "Array.isArray(value)" in handler
    assert "Array.from(value)" in handler
    assert "typeof value.length === 'number'" in handler
    assert "raw.target.files" in handler
    assert "for (const file of selected)" in handler
    assert "trigger('mark_upload', [file.name, 'ZIP archive'])" in handler
    assert "reader.readAsDataURL(file)" in handler
    assert "await trigger('decode_archive', [file.name, encoded])" in handler
    assert "raw.target.value=''" in handler


def test_native_file_inputs_register_the_change_event() -> None:
    source = Path(__file__).parents[1] / "src" / "radar_pd_nova" / "uploads.py"
    text = source.read_text(encoding="utf-8")

    assert text.count('__events=["change"]') == 2
