from pathlib import Path

import pytest

from radar_pd_nova.uploads import display_filename, inspect_cif_upload, safe_client_filename, store_browser_upload


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
    contents = b"""data_Fe
_chemical_formula_sum 'Fe'
_space_group_IT_number 229
_cell_length_a 2.86
_cell_length_b 2.86
_cell_length_c 2.86
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""

    result = inspect_cif_upload(contents, r"C:\fakepath\Fe.cif")

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
