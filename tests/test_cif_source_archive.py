from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from cif_source_archive import validate_cif_member_names


def make(tmp_path, names):
    path=tmp_path/'sources.zip'
    with zipfile.ZipFile(path,'w') as archive:
        for index,name in enumerate(names):archive.writestr(name,f'data_source_{index}\n')
    return path


def test_two_folder_same_name_fails_before_any_source_is_dropped(tmp_path):
    path=make(tmp_path,['phase_a/model.cif','phase_b/model.cif'])
    original=path.read_bytes()
    with pytest.raises(ValueError,match='phase_a/model.cif.*phase_b/model.cif.*distinct names'):
        validate_cif_member_names(path)
    assert path.read_bytes()==original
    assert list(tmp_path.iterdir())==[path]


def test_ordinary_prepared_zip_names_remain_unchanged(tmp_path):
    path=make(tmp_path,['00001_model.cif','00002_model.cif','receipt.json'])
    assert validate_cif_member_names(path)==2
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist()==['00001_model.cif','00002_model.cif','receipt.json']


def test_case_and_windows_separator_collisions_fail_closed(tmp_path):
    with pytest.raises(ValueError,match='collision'):
        validate_cif_member_names(make(tmp_path,['a/Model.cif','b\\model.CIF']))


def test_empty_or_overlarge_member_inventory_is_explicit(tmp_path):
    with pytest.raises(ValueError,match='no CIF'):
        validate_cif_member_names(make(tmp_path,['notes.txt']))
    with pytest.raises(ValueError,match='more than'):
        validate_cif_member_names(make(tmp_path,['one.cif','two.cif']),max_members=1)
