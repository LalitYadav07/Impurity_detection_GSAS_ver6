"""Signed reduction data must retain real uncertainty; absent bins stay masks."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'scripts/gsas_signed_fxye.py'
spec=importlib.util.spec_from_file_location('signed_fxye',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def fixture(tmp_path,*,instrument='SNAP'):
    data=tmp_path/'test.gsa'
    marker={'schema':'radar-pd-input-provenance/v1','instrument':instrument}
    data.write_text('Sample Run: 42\n# RADAR-PD input provenance: '+json.dumps(marker)+'\n'
                    'BANK 1 4 4 SLOG 1000 1300 .01 0 FXYE\n'
                    '1000 5 2\n1100 -3 0.5\n1200 nan 0\n1300 0 1\n')
    reader=SimpleNamespace(formatName='GSAS powder data',powderdata=[
        np.array([10.,11.,12.,13.]),np.array([5.,0.,0.,0.]),np.array([.25,0.,0.,0.])])
    return reader,data


def test_snap_restores_signed_observations_without_weighting_absent_bins(tmp_path):
    reader,path=fixture(tmp_path)
    before=path.read_bytes()
    assert module.restore_snap_signed_observations(reader,path)
    assert np.array_equal(reader.powderdata[1],[5.,-3.,0.,0.])
    assert np.array_equal(reader.powderdata[2],[.25,4.,0.,1.])
    assert path.read_bytes()==before


@pytest.mark.parametrize('instrument',['PG3','NOM'])
def test_existing_unmarked_beamline_imports_are_unchanged(tmp_path,instrument):
    reader,path=fixture(tmp_path,instrument=instrument)
    before=[a.copy() for a in reader.powderdata]
    assert module.restore_snap_signed_observations(reader,path) is False
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


@pytest.mark.parametrize('old,new,message',[
    ('1100 -3 0.5','1100 nan 0.5','Nonfinite'),
    ('1100 -3 0.5','1100 -3 -1','uncertainty'),
    ('BANK 1','BANK 2','one staged'),
    ('1100 -3 0.5','1000 -3 0.5','grid disagrees'),
])
def test_invalid_weighted_or_wrong_bank_inputs_do_not_silently_import(tmp_path,old,new,message):
    reader,path=fixture(tmp_path);path.write_text(path.read_text().replace(old,new))
    with pytest.raises(ValueError,match=message):module.restore_snap_signed_observations(reader,path)
