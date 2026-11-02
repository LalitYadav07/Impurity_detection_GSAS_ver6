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
    ('BANK 1','BANK 2','bank label'),
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


def test_instrument_or_family_claim_does_not_control_numeric_semantics(tmp_path):
    for instrument in ['NOM','PG3','SNAP','HB2A']:
        reader,path=fixture(tmp_path,instrument=instrument,family='unverified custom data')
        source=module.read_staged_tof_source(path)
        assert module.restore_staged_tof_observations(reader,source,PARAMETERS)
        assert np.array_equal(reader.powderdata[1],[5.,-3.,0.,0.])
        assert np.array_equal(reader.powderdata[2],[.25,4.,0.,1.])
        assert 'acquisition' not in source


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


def unmarked_fixture(tmp_path, *, bank=1):
    reader,path=fixture(tmp_path)
    path.write_text('\n'.join(line for line in path.read_text().splitlines()
                             if not line.startswith('# RADAR-PD input provenance: '))+'\n')
    path.write_text(path.read_text().replace('BANK 1','BANK '+str(bank)))
    reader.powderentry[2]=bank
    return reader,path


@pytest.mark.parametrize('bank',[1,2,3,17,'001','+2',0])
def test_unmarked_single_pattern_restores_supplied_signed_and_zero_observations(tmp_path,bank):
    reader,path=unmarked_fixture(tmp_path,bank=bank)
    reader.powderentry[2]=int(bank)
    raw=path.read_bytes()
    positive=[a[0].tobytes() for a in reader.powderdata]
    source=module.read_staged_tof_source(path)
    assert source['instrument'] is None and source['marked'] is False
    assert module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert np.array_equal(reader.powderdata[1],[5.,-3.,0.,0.])
    assert np.array_equal(reader.powderdata[2],[.25,4.,0.,1.])
    assert [a[0].tobytes() for a in reader.powderdata]==positive
    assert path.read_bytes()==raw


@pytest.mark.parametrize('kind',['PNC','PXC','PXT','PXE',None])
def test_unmarked_non_neutron_tof_input_retains_native_arrays(tmp_path,kind):
    reader,path=unmarked_fixture(tmp_path)
    before=[a.copy() for a in reader.powderdata]
    assert not module.restore_staged_tof_observations(reader,module.read_staged_tof_source(path),[{'Type':[kind]},{}])
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


@pytest.mark.parametrize('change',[('1000 5 2','0 5 2'),('1000 5 2','-100 5 2'),
                                 ('4 4 SLOG','999 999 CONST'),('1100 -3 0.5','1100 -3 -1')])
def test_unmarked_tof_parse_error_waits_for_loaded_native_type(tmp_path,change):
    reader,path=unmarked_fixture(tmp_path)
    path.write_text(path.read_text().replace(*change))
    source=module.read_staged_tof_source(path,defer_unmarked_errors=True)
    assert source['deferred_error']
    before=[a.copy() for a in reader.powderdata]
    assert not module.restore_staged_tof_observations(reader,source,[{'Type':['PXC']},{}])
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))
    with pytest.raises(ValueError):module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


@pytest.mark.parametrize('claim',[None,{}, {'temperature':10,'status':'recorded'},
                                {'instrument':'PG3','sample_id':'made-up','channels':[]}])
def test_acquisition_claim_cannot_change_or_certify_supplied_sigma(tmp_path,claim):
    reader,path=unmarked_fixture(tmp_path)
    if claim is not None:
        marker={'schema':'radar-pd-input-provenance/v1','instrument':'unverified','acquisition':claim}
        path.write_text(path.read_text().replace('BANK 1','# RADAR-PD input provenance: '+json.dumps(marker)+'\nBANK 1'))
    source=module.read_staged_tof_source(path)
    assert 'acquisition' not in source
    assert module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert reader.powderdata[1][1]==-3 and reader.powderdata[2][1]==4
    assert reader.powderdata[2][2]==0  # A metadata claim cannot invent sigma.


def test_selected_multibank_label_binds_its_own_grid_and_observations(tmp_path):
    reader,path=unmarked_fixture(tmp_path,bank=2)
    path.write_text(path.read_text()+'BANK 5 3 3 SLOG 2000 2200 .01 0 FXYE\n2000 9 3\n2100 -7 2\n2200 0 4\n')
    reader.numbanks=2
    reader.powderentry[2]=5
    reader.powderdata=[np.array([20.,21.,22.]),np.array([9.,0.,0.]),np.array([1./9,0.,0.])]
    source=module.read_staged_tof_source(path)
    assert module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert np.array_equal(reader.powderdata[1],[9.,-7.,0.])
    assert np.array_equal(reader.powderdata[2],[1./9,.25,1./16])
    reader.powderentry[2]=2
    with pytest.raises(ValueError,match='grid disagrees'):
        module.restore_staged_tof_observations(reader,source,PARAMETERS)


def test_multibank_source_mask_must_not_be_guessed(tmp_path):
    _,path=fixture(tmp_path,invalid=[1])
    path.write_text(path.read_text()+'BANK 2 2 2 SLOG 2000 2100 .01 0 FXYE\n2000 1 1\n2100 2 1\n')
    with pytest.raises(ValueError,match='ambiguous across multiple banks'):
        module.read_staged_tof_source(path)


@pytest.mark.parametrize('kind',['CONST','CONQ','TIME_MAP'])
def test_unmarked_unsupported_selected_tof_format_is_not_silently_clipped(tmp_path,kind):
    reader,path=unmarked_fixture(tmp_path)
    path.write_text(path.read_text().replace('SLOG',kind))
    before=[a.copy() for a in reader.powderdata]
    source=module.read_staged_tof_source(path)
    with pytest.raises(ValueError,match='requires supported SLOG/FXYE'):
        module.restore_staged_tof_observations(reader,source,PARAMETERS)
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


def test_positive_other_gsas_format_keeps_its_native_import_contract(tmp_path):
    reader,path=unmarked_fixture(tmp_path)
    path.write_text('Sample Run: 42\nBANK 1 4 4 CONST 1000 100 0 0 FXYE\n'
                    '1000 5 2\n1100 6 3\n1200 7 4\n1300 8 5\n')
    before=[a.copy() for a in reader.powderdata]
    assert not module.restore_staged_tof_observations(reader,module.read_staged_tof_source(path),PARAMETERS)
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


@pytest.mark.parametrize('change',['negative_sigma','nan_sigma','bad_count','duplicate_bank','resumed_after_comment'])
def test_unmarked_invalid_format_or_sigma_is_rejected(tmp_path,change):
    _,path=unmarked_fixture(tmp_path)
    text=path.read_text()
    if change=='negative_sigma':text=text.replace('1100 -3 0.5','1100 -3 -1')
    elif change=='nan_sigma':text=text.replace('1100 -3 0.5','1100 -3 nan')
    elif change=='bad_count':text=text.replace('4 4 SLOG','5 5 SLOG')
    elif change=='duplicate_bank':text+='BANK 1 2 2 SLOG 2000 2100 .01 0 FXYE\n2000 1 1\n2100 2 1\n'
    else:text=text.replace('1100 -3 0.5','# comment\n1100 -3 0.5')
    path.write_text(text)
    with pytest.raises(ValueError):module.read_staged_tof_source(path)


def test_unbanked_fxye_suffix_does_not_change_other_reader_contracts(tmp_path):
    path=tmp_path/'plain.fxye';path.write_text('1000 2 1\n1100 -3 2\n1200 0 1\n')
    assert module.read_staged_tof_source(path) is None


@pytest.mark.parametrize('instrument,family', [
    ('HB2A','HB2A individual reduced XYE'),('other','custom constant wavelength')])
@pytest.mark.parametrize('body', ['10 2 1\n11 3 2\n12 0 0\n',
    'BANK 001 40 4 CONST 1000 10 0 0 STD\nopaque native records\n'])
def test_non_tof_provenance_does_not_reclassify_unrelated_formats(tmp_path,instrument,family,body,monkeypatch):
    marker={'schema':'radar-pd-input-provenance/v1','instrument':instrument,'reduction_family':family}
    path=tmp_path/'marked.xye'
    path.write_text('# RADAR-PD input provenance: '+json.dumps(marker)+'\n'+body)
    assert module.read_staged_tof_source(path) is None
    monkeypatch.setattr(module,'_MAX_BYTES',20)
    assert module.read_staged_tof_source(path) is None


def test_only_known_legacy_tof_marker_requires_banked_fxye(tmp_path):
    path=tmp_path/'legacy.xye'
    marker={'schema':'radar-pd-input-provenance/v1','instrument':'SNAP'}
    path.write_text('# RADAR-PD input provenance: '+json.dumps(marker)+'\n10 2 1\n11 3 2\n')
    with pytest.raises(ValueError,match='requires SLOG/FXYE'):
        module.read_staged_tof_source(path)


@pytest.mark.parametrize('bad_coordinate', [False,True])
def test_non_tof_marker_does_not_assert_pnt_on_banked_cw(tmp_path,bad_coordinate):
    reader,path=unmarked_fixture(tmp_path)
    marker={'schema':'radar-pd-input-provenance/v1','instrument':'HB2A',
            'reduction_family':'HB2A individual reduced XYE'}
    text=path.read_text()
    if bad_coordinate:text=text.replace('1000 5 2','0 5 2')
    path.write_text('# RADAR-PD input provenance: '+json.dumps(marker)+'\n'+text)
    source=module.read_staged_tof_source(path,defer_unmarked_errors=True)
    before=[a.copy() for a in reader.powderdata]
    assert source['marked'] is False
    assert module.restore_staged_tof_observations(reader,source,[{'Type':['PNC']},{}]) is False
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))


@pytest.mark.parametrize('header',['BANK 001 40 4 CONST 1000 10 0 0 STD',
                                  'BANK legacy 40 4 CONST 1000 10 0 0 ESD'])
def test_non_fxye_native_formats_are_not_revalidated(tmp_path,header):
    path=tmp_path/'legacy.gsa';path.write_text('Legacy title\n'+header+'\nopaque native records\n')
    assert module.read_staged_tof_source(path) is None


def test_duplicate_native_integer_spellings_are_ambiguous(tmp_path):
    _,path=unmarked_fixture(tmp_path)
    path.write_text(path.read_text()+'BANK 001 2 2 SLOG 2000 2100 .01 0 FXYE\n2000 1 1\n2100 2 1\n')
    with pytest.raises(ValueError,match='Duplicate GSAS bank label'):
        module.read_staged_tof_source(path)


def test_unmarked_late_bank_header_is_not_silently_ignored(tmp_path):
    reader,path=unmarked_fixture(tmp_path)
    path.write_text('# header\n'*70+path.read_text())
    assert module.restore_staged_tof_observations(reader,module.read_staged_tof_source(path),PARAMETERS)
    assert reader.powderdata[2][1]==4


def test_unmarked_file_bank_limit_and_source_race(tmp_path,monkeypatch):
    reader,path=unmarked_fixture(tmp_path)
    source=module.read_staged_tof_source(path)
    path.write_text(path.read_text().replace('1100 -3','1100 -4'))
    with pytest.raises(ValueError,match='source changed'):
        module.restore_staged_tof_observations(reader,source,PARAMETERS)
    monkeypatch.setattr(module,'_MAX_BANKS',0)
    with pytest.raises(ValueError,match='bank limit'):module.read_staged_tof_source(path)


def test_unmarked_file_size_limit_is_not_bypassed_without_provenance(tmp_path,monkeypatch):
    _,path=unmarked_fixture(tmp_path)
    monkeypatch.setattr(module,'_MAX_BYTES',20)
    with pytest.raises(ValueError,match='import limit'):module.read_staged_tof_source(path)


@pytest.mark.parametrize('sigma',['1e-300','1e300'])
def test_unrepresentable_supplied_weights_fail_without_mutation(tmp_path,sigma):
    reader,path=unmarked_fixture(tmp_path)
    path.write_text(path.read_text().replace('1100 -3 0.5','1100 -3 '+sigma))
    before=[a.copy() for a in reader.powderdata]
    with pytest.raises(ValueError,match='finite positive weights'):
        module.restore_staged_tof_observations(reader,module.read_staged_tof_source(path),PARAMETERS)
    assert all(np.array_equal(a,b) for a,b in zip(before,reader.powderdata))
