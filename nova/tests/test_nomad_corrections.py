"""Declared reduction chemistry is provenance, not inferred sample identity."""
import json

import h5py
import pytest

from radar_pd_nova import nomad
from radar_pd_nova import nomad_controller as controller
from radar_pd_nova.nomad_corrections import (
    declared_corrections, material_comparison, strict_formula_species, with_material_review,
)
from radar_pd_nova.nomad_metadata import read_nomad_material_metadata
from radar_pd_nova.powgen_watch import WatchDefinition
from test_nomad import source, supplied, Galaxy


def change_input(source, **sample):
    path=source.with_suffix('.json')
    value=json.loads(path.read_text())
    value['Sample'].update(sample)
    path.write_text(json.dumps(value))
    return value


def formula(value):
    return {'sample_formula':value, 'formula_source':'/entry/sample/chemical_formula',
            'nexus_path':'/SNS/NOM/IPTS-1/nexus/NOM_42.nxs.h5','formula_source_verified':True}


def test_absent_dummyinfo_is_preserved_and_existing_warning_remains(source):
    path=source.with_suffix('.json');value=json.loads(path.read_text())
    value['Sample'].pop('DummyInfo')
    value['Sample'].update(Material='Si O2', PackingFraction=.5,
        Geometry={'Radius':.15,'Height':1.8},AbsorptionCorrection={'Type':'Carpenter'},
        MultipleScatteringCorrection={'Type':'Carpenter'},
        InelasticCorrection={'Type':'Placzek','Order':'1st','Self':True,'Interference':False})
    path.write_text(json.dumps(value))
    setup=nomad.inspect_reduced_setup(source)
    recorded=setup.declared_corrections['Sample']
    assert recorded['DummyInfo_present'] is False and 'DummyInfo' not in recorded
    assert recorded['Material']=='Si O2' and recorded['Geometry']=={'Radius':.15,'Height':1.8}
    for field in ('AbsorptionCorrection','MultipleScatteringCorrection','InelasticCorrection'):
        assert recorded[field]==value['Sample'][field]
    assert any('placeholder or unconfirmed' in notice for notice in setup.warnings)
    assert setup.declared_corrections['status']=='declared_in_reduction_input_not_execution_verified'


@pytest.mark.parametrize('dummy',[False,True,None,'false'])
def test_dummyinfo_value_is_not_coerced_to_a_review_decision(source,dummy):
    change_input(source,DummyInfo=dummy)
    setup=nomad.inspect_reduced_setup(source)
    assert setup.declared_corrections['Sample']['DummyInfo_present'] is True
    assert setup.declared_corrections['Sample']['DummyInfo']==dummy
    assert any('placeholder or unconfirmed' in x for x in setup.warnings)==(dummy is not False)


@pytest.mark.parametrize('declared,authoritative,status',[
    ('Si O2','Li2ODCl','different_species'),
    ('Li2OHCl','Li2ODCl','different_species'),
    ('SiO2','Si2O4','same_species_not_identity'),
    ('Li2ODCl','1g Li2ODCl Li: 0.21g O: 0.24g D: 0.015g Cl: 0.534g','unassessable'),
    ('silica','SiO2','unassessable'),
])
def test_strict_comparison_preserves_isotopes_and_refuses_prose(declared,authoritative,status):
    result=material_comparison(declared_corrections({'Sample':{'Material':declared}}),formula(authoritative))
    assert result['status']==status and result['identity_claim'] is False
    assert result['reduction_material']==declared and result['source_formula']==authoritative


def test_name_or_title_does_not_supply_an_authoritative_formula():
    result=material_comparison(declared_corrections({'Sample':{'Material':'Si O2'}}),
        {'sample_name':'Li2ODCl','run_title':'Li2ODCl at 380K'})
    assert result['status']=='unavailable' and result['source_formula'] is None


@pytest.mark.parametrize('value',['1g SiO2','SiO2 at 300K','Li2(OH)Cl','H0O','Xx2O','Si1e3O2'])
def test_formula_parser_does_not_salvage_ambiguous_or_unsupported_text(value):
    assert strict_formula_species(value) is None


def test_correction_snapshot_is_detached_and_bounded():
    config={'Sample':{'Material':'SiO2','Geometry':{'Radius':.15}},
            'Vanadium':{'Material':'V','Background':{'Runs':'5'},'MassDensity':6.11},
            'Merging':{'QThreshold':3.0},'HighQLinearFitRange':[30,40]}
    saved=declared_corrections(config)
    config['Sample']['Geometry']['Radius']=2
    assert saved['Sample']['Geometry']['Radius']==.15
    assert saved['Vanadium']['Background']=={'Runs':'5'}
    assert saved['Processing']=={'Merging':{'QThreshold':3.0},'HighQLinearFitRange':[30,40]}
    with pytest.raises(ValueError,match='bounded provenance'):
        declared_corrections({'Sample':{'Material':'A'*5000}})


def test_staged_header_and_receipt_retain_declared_settings_without_changing_data(source,supplied,tmp_path):
    value=change_input(source,Material='Si O2',AbsorptionCorrection={'Type':'Carpenter'})
    original=source.read_bytes()
    setup=with_material_review(nomad.inspect_reduced_setup(source),formula('Li2ODCl'))
    assert any('different element/isotope sets' in warning for warning in setup.warnings)
    data,_,receipt=nomad.stage_bank(source,setup,5,supplied,1,tmp_path/'stage',acknowledged=True)
    marker=next(line for line in data.read_text().splitlines() if line.startswith('# RADAR-PD input provenance: '))
    provenance=json.loads(marker.split(': ',1)[1])['reduction_setup']
    assert provenance['declared_corrections']['Sample']['Material']==value['Sample']['Material']
    assert provenance['material_comparison']['status']=='different_species'
    assert receipt['setup']['declared_corrections']==setup.declared_corrections
    assert receipt['setup']['material_comparison']['source_formula']=='Li2ODCl'
    assert b'1000 -5.5 0.1\n1100 5.25 0.2\n' in data.read_bytes()
    assert source.read_bytes()==original


def test_source_formula_reader_uses_exact_field_and_checks_run_and_proposal(tmp_path):
    path=tmp_path/'source.nxs.h5'
    with h5py.File(path,'w') as h:
        h['entry/run_number']=42;h['entry/experiment_identifier']='IPTS-1'
        h['entry/sample/chemical_formula']='Li2ODCl';h['entry/sample/name']='Unrelated display name'
    value=read_nomad_material_metadata('IPTS-1',42,nexus_path=path)
    assert value['sample_formula']=='Li2ODCl' and value['formula_source']=='/entry/sample/chemical_formula'
    assert value['formula_source_verified'] is True
    with pytest.raises(ValueError,match='source run differs'):
        read_nomad_material_metadata('IPTS-1',43,nexus_path=path)
    with pytest.raises(ValueError,match='source proposal differs'):
        read_nomad_material_metadata('IPTS-2',42,nexus_path=path)


def test_missing_source_identity_cannot_establish_a_chemical_mismatch(tmp_path):
    path=tmp_path/'source.nxs.h5'
    with h5py.File(path,'w') as h:
        h['entry/sample/chemical_formula']='Li2ODCl'
    metadata=read_nomad_material_metadata('IPTS-1',42,nexus_path=path)
    assert metadata['sample_formula']=='Li2ODCl' and not metadata['formula_source_verified']
    assessment=material_comparison(declared_corrections({'Sample':{'Material':'SiO2'}}),metadata)
    assert assessment['status']=='unverified_source' and not assessment['identity_claim']


def test_multi_value_formula_is_not_reduced_to_its_first_component(tmp_path):
    path=tmp_path/'source.nxs.h5'
    with h5py.File(path,'w') as h:
        h['entry/run_number']=42;h['entry/experiment_identifier']='IPTS-1'
        h['entry/sample/chemical_formula']=[b'Li2ODCl',b'SiO2']
    metadata=read_nomad_material_metadata('IPTS-1',42,nexus_path=path)
    assert not metadata['sample_formula']
    assert metadata['formula_review_reason']=='Source formula is not one text value'


@pytest.mark.parametrize('runs,proposals',[
    ([42,99],'IPTS-1'),
    (42,[b'IPTS-1',b'IPTS-2']),
    ([42,42],'IPTS-1'),
])
def test_multi_value_source_identity_cannot_certify_formula(tmp_path,runs,proposals):
    path=tmp_path/'source.nxs.h5'
    with h5py.File(path,'w') as h:
        h['entry/run_number']=runs;h['entry/experiment_identifier']=proposals
        h['entry/sample/chemical_formula']='Li2ODCl'
    metadata=read_nomad_material_metadata('IPTS-1',42,nexus_path=path)
    assessment=material_comparison(declared_corrections({'Sample':{'Material':'SiO2'}}),metadata)
    assert not metadata['formula_source_verified']
    assert assessment['status']=='unverified_source'
    assert 'missing or ambiguous' in assessment['reason']


@pytest.mark.parametrize('fixed_width',[False,True])
def test_source_formula_size_is_bounded(tmp_path,fixed_width):
    path=tmp_path/'source.nxs.h5'
    with h5py.File(path,'w') as h:
        h['entry/run_number']=42;h['entry/experiment_identifier']='IPTS-1'
        if fixed_width:
            h.create_dataset('entry/sample/chemical_formula',data=b'H'*5000,dtype='S5000')
        else:
            h['entry/sample/chemical_formula']='H'*5000
    with pytest.raises(ValueError,match='bounded provenance'):
        read_nomad_material_metadata('IPTS-1',42,nexus_path=path)


def test_preflight_and_dispatch_require_review_of_a_strict_mismatch(source,supplied,monkeypatch):
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr(controller,'read_nomad_scan_metadata',lambda *args:{})
    monkeypatch.setattr(controller,'read_nomad_product_material_metadata',lambda *args:formula('Li2ODCl'))
    args=dict(selected_banks=(5,),profile_path=supplied,profile_bank=1,profile_acknowledged=True)
    blocked=controller.preflight_nomad_experiment('IPTS-1',**args)
    assert not blocked['ready']
    assert blocked['reduction_material_reviews'][0]['status']=='different_species'
    assert any('different element/isotope sets' in notice for notice in blocked['quality_notices'])
    reviewed=controller.preflight_nomad_experiment('IPTS-1',allow_exploratory=True,**args)
    assert reviewed['ready'] and reviewed['reduction_material_reviews'][0]['identity_claim'] is False
    service=Galaxy(supplied.read_text())
    settings=controller.NomadExperimentSettings('IPTS-1','h','config',instrument_profile_dataset_id='original',
        selected_banks=(5,),profile_bank=1,profile_acknowledged=True)
    live=controller.NomadWatchController(service,settings)
    live._read_scan_metadata=lambda run:{}
    run=live.discover(checkpoint=False)[0]
    with pytest.raises(ValueError,match='exploratory-analysis acknowledgement'):
        live.launch_submission(run)
    assert not service.jobs and not service.snapshots


def test_freeform_formula_keeps_raw_review_evidence_without_claiming_mismatch(source):
    raw='1g Li2ODCl Li:0.21g O:0.24g D:0.015g Cl:0.534g'
    setup=with_material_review(nomad.inspect_reduced_setup(source),formula(raw))
    assert setup.material_comparison['status']=='unassessable'
    assert setup.material_comparison['source_formula']==raw
    assert any('free-form or unsupported text' in notice for notice in setup.warnings)
    assert not any('different element/isotope sets' in notice for notice in setup.warnings)


def test_preflight_compares_each_run_formula_not_only_the_latest(source,supplied,monkeypatch):
    later=source.with_name('NOM_43.gsa')
    later.write_text(source.read_text().replace('Run: 42','Run: 43'))
    config=json.loads(source.with_suffix('.json').read_text());config['Sample']['Runs']='43'
    later.with_suffix('.json').write_text(json.dumps(config))
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(source.parent)))
    monkeypatch.setattr(controller,'read_nomad_scan_metadata',lambda *args:{})
    read=[]
    def material(ipts,run,path):
        read.append(run)
        return formula('Li2ODCl' if run==42 else 'Si')
    monkeypatch.setattr(controller,'read_nomad_product_material_metadata',material)
    result=controller.preflight_nomad_experiment('IPTS-1',selected_banks=(5,),profile_path=supplied,
        profile_bank=1,profile_acknowledged=True)
    assert not result['ready'] and sorted(read)==[42,43]
    assessments={r['run_number']:r['status'] for r in result['reduction_material_reviews']}
    assert assessments=={42:'different_species',43:'same_species_not_identity'}
