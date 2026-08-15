from pathlib import Path

from radar_pd_nova.uploads import display_filename, safe_client_filename, store_browser_upload


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
