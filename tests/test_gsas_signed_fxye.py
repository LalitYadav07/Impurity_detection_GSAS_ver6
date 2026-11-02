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


PARAMETERS = [{'Type': ['PNT', 'PNT', False]}, {}]


def fixture(tmp_path,*,instrument='SNAP',family=None,invalid=None):
    data=tmp_path/'test.gsa'
    marker={'schema':'radar-pd-input-provenance/v1','instrument':instrument}
    if family is not None:marker['reduction_family']=family
    if invalid is not None:marker['invalid_count_bin_indices']=invalid
    data.write_text('Sample Run: 42\n# RADAR-PD input provenance: '+json.dumps(marker)+'\n'
                    'BANK 1 4 4 SLOG 1000 1300 .01 0 FXYE\n'
                    '1000 5 2\n1100 -3 0.5\n1200 nan 0\n1300 0 1\n')
    reader=SimpleNamespace(formatName='GSAS powder data',GSAS=True,numbanks=1,
        powderentry=[str(data),0,1],clockWd={},powderdata=[
        np.array([10.,11.,12.,13.]),np.array([5.,0.,0.,0.]),np.array([.25,0.,0.,0.])])
    return reader,data


def test_snap_restores_signed_observations_without_weighting_absent_bins(tmp_path):
    reader,path=fixture(tmp_path)
    before=path.read_bytes()
    assert module.restore_snap_signed_observations(reader,path,PARAMETERS)
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
    with pytest.raises(ValueError,match=message):module.restore_snap_signed_observations(reader,path,PARAMETERS)


@pytest.mark.parametrize('instrument,family',[
    ('NOM','NOMAD individual GSAS/SLOG'),('NOM','NOMAD IDL individual GSAS/SLOG'),
    ('PG3','POWGEN individual GSAS/FXYE'),('SNAP','SNAP individual GSAS/SLOG'),
])
def test_verified_staged_tof_restores_signed_and_zero_counts(tmp_path,instrument,family):
    reader,path=fixture(tmp_path,instrument=instrument,family=family)
    before=path.read_bytes()
    source=module.read_staged_tof_source(path)
    positive=[reader.powderdata[i][0].tobytes() for i in range(3)]
    assert module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert np.array_equal(reader.powderdata[1],[5.,-3.,0.,0.])
    assert np.array_equal(reader.powderdata[2],[.25,4.,0.,1.])
    assert [reader.powderdata[i][0].tobytes() for i in range(3)]==positive
    assert module.restore_staged_tof_observations(reader,source,PARAMETERS)  # Idempotent before native conversion.
    assert path.read_bytes()==before


@pytest.mark.parametrize('masked_array', [0,1,2])
def test_existing_native_masks_are_not_reinterpreted_as_intensity_suppression(tmp_path,masked_array):
    reader,path=fixture(tmp_path)
    reader.powderdata[masked_array]=np.ma.array(reader.powderdata[masked_array],mask=[False,True,False,False])
    source=module.read_staged_tof_source(path)
    module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert reader.powderdata[2][1] is np.ma.masked
    assert np.ma.getdata(reader.powderdata[2])[1]==0
    assert np.ma.getmaskarray(reader.powderdata[masked_array])[1]
    assert np.ma.getdata(reader.powderdata[2])[3]==1


def test_source_invalid_placeholder_is_not_restored_to_positive_weight(tmp_path):
    reader,path=fixture(tmp_path,instrument='NOM',family='NOMAD individual GSAS/SLOG',invalid=[1,3])
    source=module.read_staged_tof_source(path)
    module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert np.ma.getdata(reader.powderdata[2]).tolist()==[.25,0.,0.,0.]
    assert np.ma.getmaskarray(reader.powderdata[2]).tolist()==[False,True,False,True]


@pytest.mark.parametrize('invalid', [[True],[-1],[4],[1,1],'1'])
def test_invalid_source_mask_declarations_fail(tmp_path,invalid):
    _,path=fixture(tmp_path,invalid=invalid)
    with pytest.raises(ValueError,match='mask indices'):module.read_staged_tof_source(path)


@pytest.mark.parametrize('kind', ['PXC','PXE','PXT','PNC',None])
def test_only_native_neutron_tof_parameters_are_allowed(tmp_path,kind):
    reader,path=fixture(tmp_path)
    before=[a.copy() for a in reader.powderdata]
    with pytest.raises(ValueError,match='native PNT'):
        module.restore_staged_tof_observations(reader,module.read_staged_tof_source(path),[{'Type':[kind]},{}])
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


@pytest.mark.parametrize('change', ['file','bank','multiple','clock','reader','positive','weight','grid'])
def test_unexpected_reader_state_fails_without_modifying_arrays(tmp_path,change):
    reader,path=fixture(tmp_path)
    if change=='file':reader.powderentry[0]=str(tmp_path/'other.gsa')
    elif change=='bank':reader.powderentry[2]=2
    elif change=='multiple':reader.numbanks=2
    elif change=='clock':reader.clockWd={'1':1}
    elif change=='reader':reader.formatName='Other reader'
    elif change=='positive':reader.powderdata[1][0]=6
    elif change=='weight':reader.powderdata[2][0]=.3
    else:reader.powderdata[0][1]+=1e-12
    before=[a.copy() for a in reader.powderdata]
    with pytest.raises(ValueError):
        module.restore_staged_tof_observations(reader,module.read_staged_tof_source(path),PARAMETERS)
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


@pytest.mark.parametrize('old,new', [('4 4 SLOG','5 5 SLOG'),('SLOG','CONST'),('FXYE','FXY'),
    ('1300 0 1','1300 0 1\nBANK 1 1 1 SLOG 1 2 .1 0 FXYE\n1400 1 1')])
def test_bank_kind_count_and_single_histogram_are_required(tmp_path,old,new):
    _,path=fixture(tmp_path);path.write_text(path.read_text().replace(old,new))
    with pytest.raises(ValueError):module.read_staged_tof_source(path)


def test_unknown_family_and_unmarked_data_are_unchanged(tmp_path):
    for instrument in ['NOM','PG3','SNAP','HB2A']:
        _,path=fixture(tmp_path,instrument=instrument,family='unverified custom data')
        assert module.read_staged_tof_source(path) is None


def test_marked_file_and_header_size_limits_fail_closed(tmp_path,monkeypatch):
    _,path=fixture(tmp_path)
    monkeypatch.setattr(module,'_MAX_BYTES',20)
    with pytest.raises(ValueError,match='import limit'):module.read_staged_tof_source(path)


def test_source_change_with_identical_grid_is_rejected(tmp_path):
    reader,path=fixture(tmp_path)
    source=module.read_staged_tof_source(path)
    path.write_text(path.read_text().replace('1100 -3','1100 -4'))
    with pytest.raises(ValueError,match='source changed'):
        module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert reader.powderdata[2][1]==0


def test_native_latin1_title_does_not_change_numeric_source(tmp_path):
    reader,path=fixture(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'Sample Run: 42',b'Sample caf\xe9 Run: 42'))
    assert module.restore_staged_tof_observations(reader,module.read_staged_tof_source(path),PARAMETERS)
    assert reader.powderdata[2][3]==1
