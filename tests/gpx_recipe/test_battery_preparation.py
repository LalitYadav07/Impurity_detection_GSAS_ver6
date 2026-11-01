from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.battery.preparation import validate_preparation, preserve_nomad_observations, input_path
from gpx_recipe.battery.preparation import verify_excluded_bins_outside_fit, set_sample_widths


@pytest.fixture
def specification(tmp_path):
    def ref(name):
        path=tmp_path/name;path.write_text(name)
        return dict(path=name,sha256=sha256(path.read_bytes()).hexdigest())
    data=ref('NOM_42.gsa')
    return dict(schema='radar-battery-preparation/v1',reduction_family='shared/autoNOM/GSAS',review_note='Synthetic validation',
        acquisition=dict(instrument='NOM',ipts='IPTS-1',run=42,sample_id='sample',cell_id='cell',
                         start_utc='2026-01-01T00:00:00Z',end_utc='2026-01-01T00:05:00Z',metadata_sha256='a'*64),
        banks=[dict(key=f'b{b}',physical_bank=b,data=data,profile=ref(f'bank{b}.instprm'),profile_bank=b,
                    fit_d=[.5,3],profile_review='Explicit synthetic profile') for b in (2,5)],
        phases=[dict(key='graphite',name='Graphite',role='anode',cif=ref('graphite.cif'),hypothesis_note='Synthetic',
                     links=[dict(bank=f'b{b}',hap_scale=.5) for b in (2,5)])])


def test_preparation_keeps_explicit_links_and_does_not_mutate_input(specification,tmp_path):
    result=validate_preparation(specification,tmp_path)
    result['phases'][0]['links'].clear()
    assert len(specification['phases'][0]['links'])==2


def test_width_starts_are_explicit_per_physical_link_and_do_not_change_other_hap_inputs(specification,tmp_path):
    first,second=specification['phases'][0]['links']
    first['sample_widths']={'size_um':.01,'microstrain_ustrain':0.}
    second['sample_widths']={'size_um':.1}
    result=validate_preparation(specification,tmp_path)
    assert result['phases'][0]['links']==[first,second]
    hap={'Size':['isotropic',[1.,1.,.6],[False,False,False]],
         'Mustrain':['isotropic',[1000.,1000.,.5],[False,False,False]],'Scale':[.5,False]}
    original=deepcopy(hap)
    set_sample_widths(hap,first['sample_widths'])
    assert hap['Size'][1]==[.01,1.,.6] and hap['Mustrain'][1]==[0.,1000.,.5]
    assert hap['Scale']==original['Scale'] and hap['Size'][2]==original['Size'][2]
    set_sample_widths(hap,{})
    hap['Size'][0]='uniaxial'
    with pytest.raises(ValueError,match='isotropic'):set_sample_widths(hap,{'size_um':.1})


@pytest.mark.parametrize('widths',[
    {'size_um':True},{'size_um':float('nan')},{'size_um':.0009},{'size_um':10.1},
    {'size_um':'0.1'},{'microstrain_ustrain':-1},{'microstrain_ustrain':float('inf')},
    {'size_nm':10},None,[],
])
def test_invalid_or_unit_ambiguous_width_starts_stop_preparation(specification,tmp_path,widths):
    specification['phases'][0]['links'][0]['sample_widths']=widths
    with pytest.raises(ValueError):validate_preparation(specification,tmp_path)


def test_powgen_reduction_evidence_seals_presence_and_content(specification,tmp_path):
    specification['acquisition']['instrument']='PG3'
    with pytest.raises(ValueError,match='reduction-evidence list'):
        validate_preparation(specification,tmp_path)
    specification['reduction_evidence']=[]
    validate_preparation(specification,tmp_path)
    evidence=tmp_path/'NOM_42.py';evidence.write_text('literal reduction evidence')
    with pytest.raises(ValueError,match='unrecorded'):
        validate_preparation(specification,tmp_path)
    specification['reduction_evidence']=[dict(path=evidence.name,sha256=sha256(evidence.read_bytes()).hexdigest())]
    validate_preparation(specification,tmp_path)
    evidence.write_text('modified reduction evidence')
    with pytest.raises(ValueError,match='source changed'):
        validate_preparation(specification,tmp_path)


def test_nexus_instrument_cannot_select_another_instruments_catalog(specification,tmp_path):
    specification['banks'][0]['profile_origin']='powgen_catalog'
    with pytest.raises(ValueError,match='profile origin'):
        validate_preparation(specification,tmp_path)


@pytest.mark.parametrize('change,message',[
    (lambda s:s['banks'][1].update(physical_bank=2),'physical bank'),
    (lambda s:s['banks'][0].update(profile_bank=True),'Bank label'),
    (lambda s:s['banks'][0].update(fit_d=[2,1]),'fit limits'),
    (lambda s:s['banks'][0].update(invalid_bin_policy='drop_any_bad_data'),'Invalid bins'),
    (lambda s:s['banks'][0].update(invalid_bin_policy=['outside_fit_only']),'Invalid bins'),
    (lambda s:s['phases'][0]['links'].pop(),'Every bank'),
    (lambda s:s['phases'][0]['links'][0].update(hap_scale=-1),'positive starting'),
    (lambda s:s['acquisition'].update(end_utc='2026-01-01T00:00:00Z'),'exposure'),
    (lambda s:s['acquisition'].update(start_utc='2026-01-01T00:00:00'),'UTC offset'),
    (lambda s:s['phases'][0].update(role='measured_lithium'),'physical role'),
    (lambda s:s.update(execute='arbitrary code'),'Unexpected'),
])
def test_invalid_or_ambiguous_preparations_rejected(specification,tmp_path,change,message):
    change(specification)
    with pytest.raises(ValueError,match=message):validate_preparation(specification,tmp_path)


def test_changed_input_and_mixed_reductions_rejected(specification,tmp_path):
    (tmp_path/'NOM_42.gsa').write_text('a new reduction')
    with pytest.raises(ValueError,match='changed'):validate_preparation(specification,tmp_path)
    other=tmp_path/'other.gsa';other.write_text('other reduction')
    specification['banks'][0]['data']=dict(path='NOM_42.gsa',sha256=sha256((tmp_path/'NOM_42.gsa').read_bytes()).hexdigest())
    specification['banks'][1]['data']=dict(path=other.name,sha256=sha256(other.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='mix reduction'):validate_preparation(specification,tmp_path)


@pytest.mark.parametrize('name',['../file','/tmp/file','nested/../file','dir\\file'])
def test_input_paths_cannot_escape_bundle(tmp_path,name):
    with pytest.raises(ValueError,match='confined'):input_path(tmp_path,dict(path=name,sha256='a'*64))


def test_signed_observations_are_preserved_after_checked_native_conversion(tmp_path):
    data=tmp_path/'bank.gsa'
    data.write_text('test\nBANK 1 4 4 SLOG 1000 1300 .1 0 FXYE\n1000 20 2\n1100 -5 1\n1200 0 3\n1300 7 1\n')
    arrays=[np.array([1050.,1150.,1250.]),np.array([.2,0.,0.]),np.array([2500.,0.,0.])]
    histogram=SimpleNamespace(data={'data':[{},arrays]})
    receipt=preserve_nomad_observations(histogram,data)
    np.testing.assert_allclose(arrays[1],[.2,-.05,0])
    np.testing.assert_allclose(arrays[2],[2500,10000,10000/9])
    assert receipt['nonpositive_observations_retained']==2
    arrays[0]+=1
    with pytest.raises(ValueError,match='bin convention'):preserve_nomad_observations(histogram,data)


def test_invalid_count_placeholders_remain_masked_and_cannot_enter_fit(tmp_path):
    path=tmp_path/'bank.gsa';path.write_text('test\nBANK 1 4 4 SLOG 1000 1300 .1 0 FXYE\n1000 20 2\n1100 10 1\n1200 0 1\n1300 0 1\n')
    arrays=[np.array([1050.,1150.,1250.]),np.array([.2,.1,0.]),np.array([2500.,10000.,0.])]
    h=SimpleNamespace(data={'data':[{},arrays],'Limits':[[1000,1300],[1000,1150]]})
    invalid=[{'record_index':2,'intensity':'nan','uncertainty':'nan'},{'record_index':3,'intensity':'nan','uncertainty':'nan'}]
    receipt=preserve_nomad_observations(h,path,invalid)
    assert receipt['invalid_bin_indices_excluded']==[2]
    assert arrays[1].mask.tolist()==[False,False,True]
    assert arrays[2][2]==0. and arrays[2][1]==10000.
    verify_excluded_bins_outside_fit(h,invalid)
    # GSAS-II includes the next bin even when the upper limit is between bins.
    h.data['Limits'][1]=[1000,1200]
    with pytest.raises(ValueError,match='inside the selected fit range'):verify_excluded_bins_outside_fit(h,invalid)
