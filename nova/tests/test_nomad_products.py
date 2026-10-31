"""Derived-product identities and acquisition windows, using small synthetic logs."""
from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from radar_pd_nova import nomad, nomad_metadata
from radar_pd_nova.nomad_products import member_runs, product_from_path, slice_metadata_path, slice_window
from radar_pd_nova.nomad_controller import NomadExperimentSettings, NomadWatchController
from radar_pd_nova.powgen_watch import WatchDefinition, WatchState, discover_from_listing
from test_nomad import Galaxy, source, supplied


def processed(path, *, start='2025-01-31T10:51:30', duration=30., run=42, times=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path,'w') as f:
        root='mantid_workspace_1/logs/'
        for key,value in [('run_number',[str(run).encode()]),('duration',[duration]),
                          ('end_time',[b'2025-01-31T10:56:30'])]:
            f.create_dataset(root+key+'/value',data=value)
        dataset=f.create_dataset(root+'proton_charge/time',data=np.linspace(0,duration,1801) if times is None else times)
        dataset.attrs['start']=start


def sliced(source,tmp_path,token,start):
    path=tmp_path/'shared/sliced/reduced/GSAS'/f'nom_42_{token}.h5.gsa'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(source.read_bytes())
    processed(slice_metadata_path(path),start=start)
    return path


def test_slice_uses_filtered_duration_not_retained_parent_end(tmp_path):
    path=tmp_path/'slice.h5';processed(path)
    value=slice_window(path)
    assert value['run_number']==42 and value['duration_seconds']==30
    assert value['end']=='2025-01-31T10:52:00+00:00'
    assert value['retained_parent_end_time']=='2025-01-31T10:56:30'
    value['run_number']=99
    assert slice_window(path)['run_number']==42  # callers cannot alter the cache


@pytest.mark.parametrize('times',[[],[0],[0,1,1,30],[0,1,2,30],[0,1,2],[0,float('nan'),30]])
def test_invalid_slice_pulses_are_not_replaced_with_parent_metadata(tmp_path,times):
    path=tmp_path/'invalid.h5';processed(path,times=times)
    with pytest.raises(ValueError):slice_window(path)


def test_short_tail_uses_last_logged_pulse_not_administrative_duration_padding(tmp_path):
    path=tmp_path/'tail.h5'
    processed(path,duration=.021562,times=[0.,.016667])
    value=slice_window(path)
    assert value['end']=='2025-01-31T10:51:30.016667+00:00'
    assert value['duration_seconds']==pytest.approx(.016667)
    assert value['recorded_duration_seconds']==pytest.approx(.021562)


def test_slice_identity_order_and_staging_guards(source,supplied,tmp_path,monkeypatch):
    warm=sliced(source,tmp_path,'300','2025-01-31T10:51:30')
    cold=sliced(source,tmp_path,'100','2025-01-31T10:52:00')
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(warm.parent)))
    definition=WatchDefinition('IPTS-1',nomad.SLICED_FOLDER,'h','c','p',instrument='NOM')
    runs=discover_from_listing(definition,nomad.directory_listing(warm.parent))
    # Filename tokens are not temperature coordinates or acquisition order.
    assert [r.run_id for r in runs]==['NOM_42_SLICE_300','NOM_42_SLICE_100']
    banked=[replace(r,bank=5) for r in runs]
    state=WatchState('h',instrument='NOM',discovered={r.run_id:r for r in banked})
    assert set(WatchState.from_json(state.to_json()).discovered)=={r.run_id for r in banked}
    setup=nomad.inspect_reduced_setup(warm)
    data,_,_=nomad.stage_bank(warm,setup,5,supplied,1,tmp_path/'stage',acknowledged=True)
    assert b'1000 -5.5 0.1' in data.read_bytes()
    processed(slice_metadata_path(warm),start='2025-01-31T10:52:00')
    with pytest.raises(ValueError,match='slice metadata changed'):
        nomad.stage_bank(warm,setup,5,supplied,1,tmp_path/'changed',acknowledged=True)


@pytest.mark.parametrize('limit',[0,1,None])
def test_late_slice_of_same_parent_is_found_after_restart(source,supplied,tmp_path,monkeypatch,limit):
    initial=sliced(source,tmp_path,'300','2025-01-31T10:51:30')
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(initial.parent)))
    settings=NomadExperimentSettings('IPTS-1','h','c',subfolder=nomad.SLICED_FOLDER,
        instrument_profile_dataset_id='p',selected_banks=(5,),profile_bank=1,
        profile_acknowledged=True,allow_exploratory=True,initial_backfill_limit=limit)
    service=Galaxy(supplied.read_text());first=NomadWatchController(service,settings)
    first._read_scan_metadata=lambda run:{}
    assert len(first.discover())==(0 if limit==0 else 1)
    first.persist_state()
    restored=NomadWatchController(service,settings);restored._read_scan_metadata=lambda run:{}
    assert restored.restore_latest_state()
    sliced(source,tmp_path,'100','2025-01-31T10:52:00')
    assert [r.run_id for r in restored.discover()]==['NOM_42_SLICE_100_BANK5']
    assert not restored.discover()
    assert not service.jobs  # discovery must not submit refinements


def test_sum_requires_explicit_membership_and_does_not_alias_last_run(tmp_path):
    path=tmp_path/'multi_banks_summed/GSAS/NOM_Fe_250.gsa';path.parent.mkdir(parents=True)
    with pytest.raises(ValueError,match='reduction input'):product_from_path(path)
    side=path.parent.parent/'Input'/f'{path.stem}.json';side.parent.mkdir()
    side.write_text(json.dumps({'Sample':{'Runs':'41-42'}}))
    product=product_from_path(path)
    assert product.kind=='sum' and product.members==(41,42) and product.run_number==42
    assert product.product_id.startswith('SUM_')
    side2=side.with_name('NOM_Fe_251.json');side2.write_text(side.read_text())
    assert product_from_path(path.with_name('NOM_Fe_251.gsa')).product_id!=product.product_id
    from radar_pd_nova.powgen_samples import include_run
    from radar_pd_nova.powgen_watch import WatchedRun
    run=WatchedRun(42,str(path),'hash',instrument='NOM',product_id=product.product_id)
    assert not include_run(run,excluded_run_numbers=(41,))
    assert include_run(run,excluded_run_numbers=(43,))
    # A numeric name in a summed-family folder must keep its summed identity.
    side.with_name('NOM_42.json').write_text(side.read_text())
    assert product_from_path(path.with_name('NOM_42.gsa')).kind=='sum'


@pytest.mark.parametrize('value',['','1-3000','2-1','1,1','1;rm','1-2,2',42])
def test_invalid_sum_membership(value):
    with pytest.raises(ValueError):member_runs(value)


def test_slice_conditions_are_recomputed_on_its_exact_interval(source,tmp_path,monkeypatch):
    path=sliced(source,tmp_path,'100','2025-01-31T10:51:30')
    parent=dict(available=True,start_time='2025-01-31T10:51:00Z',end_time='2025-01-31T10:56:30Z',
        sample_id='A',duration_seconds=330,temperature={'value':200})
    monkeypatch.setattr(nomad_metadata,'read_nomad_scan_metadata',lambda *a:dict(parent))
    seen=[]
    def attach(metadata,nexus,instrument,*,window):
        seen.append(window)
        return {**metadata,'temperature':{'value':100},'conditions':{'window_scope':'explicit_product_window'}}
    monkeypatch.setattr('radar_pd_nova.scan_conditions.attach_conditions',attach)
    value=nomad_metadata.read_nomad_product_metadata('IPTS-1',42,path)
    assert value['duration_seconds']==30 and value['temperature']['value']==100
    assert seen==[{'start':'2025-01-31T10:51:30+00:00','end':'2025-01-31T10:52:00+00:00'}]
    parent['end_time']='2025-01-31T10:51:31Z'
    with pytest.raises(ValueError,match='outside'):nomad_metadata.read_nomad_product_metadata('IPTS-1',42,path)


def test_missing_slice_sensor_never_falls_back_to_a_parent_sweep_average(source,tmp_path,monkeypatch):
    path=sliced(source,tmp_path,'100','2025-01-31T10:51:30')
    parent=dict(available=True,start_time='2025-01-31T10:51:00Z',end_time='2025-01-31T10:56:30Z',
        temperature={'value':300},magnetic_field={'value':2},conditions={'parent':'full run'})
    monkeypatch.setattr(nomad_metadata,'read_nomad_scan_metadata',lambda *a:dict(parent))
    def missing(metadata,*a,**kw):
        assert not {'temperature','magnetic_field','conditions'} & set(metadata)
        return {**metadata,'conditions':{'window_scope':'explicit_product_window','channels':[]}}
    monkeypatch.setattr('radar_pd_nova.scan_conditions.attach_conditions',missing)
    value=nomad_metadata.read_nomad_product_metadata('IPTS-1',42,path)
    assert 'temperature' not in value and 'magnetic_field' not in value


def test_excluding_one_slice_does_not_exclude_its_whole_parent(source,supplied,tmp_path,monkeypatch):
    initial=sliced(source,tmp_path,'100','2025-01-31T10:51:30')
    sliced(source,tmp_path,'103','2025-01-31T10:52:00')
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(initial.parent)))
    settings=NomadExperimentSettings('IPTS-1','h','c',subfolder=nomad.SLICED_FOLDER,
        instrument_profile_dataset_id='p',selected_banks=(5,),profile_bank=1,
        profile_acknowledged=True,allow_exploratory=True,excluded_run_numbers=('NOM_42_SLICE_103',))
    controller=NomadWatchController(Galaxy(supplied.read_text()),settings);controller._read_scan_metadata=lambda run:{}
    assert [r.run_id for r in controller.discover()]==['NOM_42_SLICE_100_BANK5']
    from radar_pd_nova.powgen_watch import normalize_exclusions
    assert normalize_exclusions(['42','NOM_42_SLICE_103'],'NOM',from_form=True)==(42,'NOM_42_SLICE_103')
    with pytest.raises(ValueError):normalize_exclusions(['NOM_42_SLICE_103'],'PG3')


def test_changed_sum_membership_is_a_revision_not_a_new_automatic_acquisition(source,supplied,tmp_path,monkeypatch):
    path=tmp_path/'multi_banks_summed/GSAS/NOM_held_temperature.gsa';path.parent.mkdir(parents=True)
    path.write_bytes(source.read_bytes())
    side=path.parent.parent/'Input'/f'{path.stem}.json';side.parent.mkdir()
    config=json.loads(source.with_suffix('.json').read_text());config['Sample']['Runs']='41-42'
    side.write_text(json.dumps(config))
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(path.parent)))
    settings=NomadExperimentSettings('IPTS-1','h','c',subfolder=nomad.SUMMED_FOLDER,
        instrument_profile_dataset_id='p',selected_banks=(5,),profile_bank=1,
        profile_acknowledged=True,allow_exploratory=True)
    service=Galaxy(supplied.read_text());controller=NomadWatchController(service,settings)
    controller._read_scan_metadata=lambda run:{}
    old=controller.discover()[0]
    config['Sample']['Runs']='41-43';side.write_text(json.dumps(config))
    path.write_text(path.read_text().replace('Sample Run: 42','Sample Run: 43'))
    assert not controller.discover()
    assert controller.state.discovered[old.run_id].scan_metadata['source_revision_warning']
    assert not service.jobs
