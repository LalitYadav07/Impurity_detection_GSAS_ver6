"""Portable planner and History recovery preserve settings without pickle trust."""
import asyncio
import copy
import hashlib
import io
import json
from pathlib import Path
import pickle
import threading
from types import SimpleNamespace
import zipfile
import uuid

import numpy as np
import pytest

from radar_pd_nova import planning_bundle as bundle
from radar_pd_nova.planning_controller import FIELDS,PlanningMixin
from radar_pd_nova.planning_storage import EXTRA_FIELDS,_planning_history_label,_planning_restore_settings
from test_model_runtime import Service


class State(SimpleNamespace):
    def change(self,*_):return lambda callback:callback
    def flush(self):pass


class History(Service):
    galaxy_key='PRIVATE-KEY'
    def _dataset_metadata(self,key):
        if key not in self.data:raise KeyError(key)
        return dict(id=key,state='ok',history_id=self.history_id)


def planner(tmp_path,service=None):
    instance=PlanningMixin();instance.server=SimpleNamespace(state=State())
    instance.service=service or SimpleNamespace(output_root=tmp_path)
    instance._init_planning();instance.load_planning_example()
    return instance,instance.server.state


def result(instance):
    root=instance._planning_root()/'simulation';root.mkdir(parents=True)
    np.savez(root/'curves.npz',d=[1.,2.,3.],expected=[1.,2.,1.],nuclear=[1.,2.,1.],magnetic=[0.,0.,0.])
    # This is intentionally a real pickle payload, which must remain opaque.
    (root/'forward.gpx').write_bytes(pickle.dumps({'never':'load uploaded pickle'}))
    receipt=dict(schema='powgen-planning-result/v1',mode='profile_only',looks=[],intensity_unit='shape',
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})
    (root/'result.json').write_text(json.dumps(receipt))
    instance._planning_output=root;instance._planning_receipt_sha256=hashlib.sha256((root/'result.json').read_bytes()).hexdigest()
    instance.server.state.plan_current=True;instance.server.state.plan_result_ready=True
    return root


def test_portable_bundle_rebases_settings_and_results_without_executing_gpx(tmp_path,monkeypatch):
    instance,s=planner(tmp_path/'original');root=result(instance)
    s.plan_mass=2.5;s.plan_isotopes='{"Si":{"28":0.25,"29":0.75}}';s.plan_seconds='60, 900'
    document,files=instance._planning_build_bundle();raw=bundle.archive(document,files)
    assert document['schema']==bundle.SCHEMA and not Path(document['settings']['plan_cif']).is_absolute()
    assert str(tmp_path) not in json.dumps(document)
    monkeypatch.setattr(pickle,'load',lambda *a,**k:pytest.fail('Never unpickle uploaded native data'))
    monkeypatch.setattr(pickle,'loads',lambda *a,**k:pytest.fail('Never unpickle uploaded native data'))
    fresh,t=planner(tmp_path/'different')
    restored,values=bundle.read_archive(raw);fresh._planning_apply_bundle(restored,values)
    assert (t.plan_mass,t.plan_seconds,t.plan_isotopes)==(2.5,'60, 900',s.plan_isotopes)
    assert t.plan_current and t.plan_result_ready and t.plan_figure['data']
    assert 'provenance unverified' in t.plan_qualification
    assert Path(t.plan_cif).is_file() and fresh._planning_output!=root
    with pytest.raises(ValueError,match='Uploaded GPX'):
        fresh._planning_native_input(fresh._planning_output/'forward.gpx')
    fresh._planning_edited()
    assert not t.plan_current and not t.plan_result_ready


def test_real_history_save_and_fresh_controller_recovery(tmp_path):
    service=History(tmp_path);service.output_root=tmp_path/'one'
    instance,s=planner(tmp_path/'one',service);result(instance)
    s.plan_name='Portable silicon';s.plan_mass=3.25
    instance.save_planning_plan()
    assert not s.plan_error and s.plan_saved_id.startswith('history:')
    selected=s.plan_saved_id
    # Simulate losing every original local path with a separate output root.
    service.output_root=tmp_path/'two'
    fresh,t=planner(tmp_path/'two',service)
    fresh.refresh_planning_plans()
    assert selected in {row['value'] for row in t.plan_saved_options}
    t.plan_saved_id=selected;fresh.restore_planning_plan()
    assert not t.plan_error and t.plan_mass==3.25 and t.plan_current
    assert fresh._planning_output.is_relative_to(tmp_path/'two')
    assert 'Galaxy' in s.plan_save_message


def test_other_history_cannot_restore_even_an_accessible_copied_bundle(tmp_path):
    service=History(tmp_path);service.output_root=tmp_path/'one'
    instance,s=planner(tmp_path/'one',service);instance.save_planning_plan();selected=s.plan_saved_id
    service.history_id='other-history';service.output_root=tmp_path/'other'
    fresh,t=planner(tmp_path/'other',service);fresh.refresh_planning_plans()
    t.plan_saved_id=selected;fresh.restore_planning_plan()
    assert 'another History' in t.plan_error and not t.plan_current


def test_history_selector_distinguishes_named_plans_with_same_archive_filename(tmp_path,monkeypatch):
    service=History(tmp_path);service.output_root=tmp_path
    instance,s=planner(tmp_path,service)
    names=['Silicon shape · NDIP acceptance','Silicon counts · NDIP acceptance']
    rows=[{'id':identifier,'name':'RADAR-PD Planner bundle '+name+' '+'a'*64+' | plan.zip',
           'display_name':'plan.zip · 2026-11-10 06:35 · '+identifier[:8],
           'update_time':'2026-11-10T06:35:24.000000'}
          for identifier,name in zip(['6a00c9f0aa11bb22','d2203311aa11bb22'],names)]
    monkeypatch.setattr(service,'search_history_datasets',lambda **kwargs:rows)
    instance.refresh_planning_plans()
    assert not s.plan_error
    titles={row['value']:row['title'] for row in s.plan_saved_options}
    for row,name in zip(rows,names):
        assert titles['history:'+row['id']]=='Galaxy · '+name+' · 2026-11-10 06:35 · '+row['id'][:8]
    assert instance._planning_history_ids=={row['id'] for row in rows}


@pytest.mark.parametrize('name', ['RADAR-PD Planner bundle | plan.zip',
    'RADAR-PD Planner bundle Legacy experiment | plan.zip',
    'RADAR-PD Planner bundle Bad hash nope | plan.zip',
    'RADAR-PD Planner bundle '+'x'*101+' '+'a'*64+' | plan.zip',
    'RADAR-PD Planner bundle Split\nname '+'a'*64+' | plan.zip',None,{},42])
def test_history_label_uses_legacy_display_fallback_for_unrecognized_metadata(name):
    row={'name':name,'display_name':'plan.zip · 2026-11-10 06:35 · 6a00c9f0'}
    assert _planning_history_label(row,'6a00c9f0aa11bb22')==row['display_name']


def test_history_label_handles_missing_or_malformed_display_metadata():
    name='RADAR-PD Planner bundle Named experiment '+'a'*64+' | plan.zip'
    assert _planning_history_label({'name':name,'display_name':{},'update_time':{}},'6a00c9f0aa11bb22')=='Named experiment · 6a00c9f0'
    assert _planning_history_label({'name':None,'display_name':None},'6a00c9f0aa11bb22')=='6a00c9f0aa11bb22'
    assert len(_planning_history_label({'name':{},'display_name':'x'*10000},'6a00c9f0aa11bb22'))==256


def connected_planner(tmp_path,service=None):
    """Real Trame state, NOVA download and wslink publication; only the wire is a sink."""
    from nova.trame import ThemedApp
    from trame.app import get_server
    from trame.widgets import client
    from trame_server.protocol import CoreServer
    from trame_server.state import State as TrameState
    from wslink.publish import PublishManager
    frames=[];commits=[];tasks=[];download_sent=asyncio.Event()
    publisher=PublishManager()
    class Wire:
        network_monitor=SimpleNamespace(on_enter=lambda:None,on_exit=lambda *_:None)
        async def sendWrappedMessage(self,topic,data,**kwargs):
            frames.append((threading.get_ident(),topic,copy.deepcopy(data)))
            if 'trame.actions.topic' in topic:download_sent.set()
    publisher.registerProtocol(Wire())
    def commit(data):
        commits.append((threading.get_ident(),copy.deepcopy(data)))
        publisher.publish('trame.state.topic',copy.deepcopy(data))
    server=get_server('planner-storage-'+uuid.uuid4().hex,client_type='vue3')
    server._state=TrameState(commit_fn=commit,ready=True)
    # wslink normally attaches publish to the root protocol during server startup.
    protocol=object.__new__(CoreServer);protocol.publish=publisher.publish
    server._root_protocol=protocol
    instance=PlanningMixin();instance.server=server
    instance.service=service or SimpleNamespace(output_root=tmp_path)
    instance._init_planning();server.state.flush()
    instance.load_planning_example();server.state.flush()
    def schedule(coroutine,name):
        task=asyncio.create_task(coroutine,name=name);tasks.append(task);return task
    instance._refinement_schedule=schedule
    instance.download_file=ThemedApp.download_file.__get__(instance,PlanningMixin)
    instance._download_file=client.JSEval(trame_server=server,exec='async () => {}').exec
    return instance,server.state,tasks,frames,commits,download_sent


def test_export_delivers_real_nova_download_on_owner_loop_with_exact_bundle_bytes(tmp_path,monkeypatch):
    async def check():
        owner=threading.get_ident()
        instance,s,tasks,frames,_,download_sent=connected_planner(tmp_path)
        result(instance);s.flush()
        expected_document,expected_files=instance._planning_build_bundle()
        built=[]
        def build():
            built.append(threading.get_ident());return expected_document,expected_files
        monkeypatch.setattr(instance,'_planning_build_bundle',build)
        instance.export_planning_plan();await tasks[-1]
        assert not s.plan_error
        await asyncio.wait_for(download_sent.wait(),1)
        actions=[(thread,data) for thread,topic,data in frames if 'trame.actions.topic' in topic]
        assert len(actions)==1 and actions[0][0]==owner
        action=actions[0][1][0]
        assert action['method']=='exec'
        name,mime,raw=action['args'][0]
        assert (name,mime)==('powgen-experiment-plan.zip','application/zip')
        actual_document,actual_files=bundle.read_archive(raw)
        assert actual_document==expected_document and actual_files==expected_files
        assert built and all(thread!=owner for thread in built)
        assert s.plan_save_message=='Portable plan download requested.'
        assert not s.plan_busy and not s.plan_can_cancel
    asyncio.run(check())


def test_history_restore_publishes_values_and_callbacks_on_owner_loop_before_clearing_guard(tmp_path,monkeypatch):
    service=History(tmp_path);service.output_root=tmp_path/'original'
    original,source=planner(tmp_path/'original',service);result(original)
    source.plan_name='Restored silicon';source.plan_mass=3.25
    original.save_planning_plan();selected=source.plan_saved_id
    async def check():
        owner=threading.get_ident();service.output_root=tmp_path/'fresh'
        instance,s,tasks,frames,commits,_=connected_planner(tmp_path/'fresh',service)
        instance._planning_history_ids={selected.removeprefix('history:')}
        s.plan_saved_id=selected;s.flush();await asyncio.sleep(0)
        frames.clear();commits.clear();callbacks=[];downloads=[]
        @s.change('plan_mass','plan_cif','plan_current')
        def changed(**_):callbacks.append((threading.get_ident(),instance._planning_restoring))
        download=service._download_dataset
        def tracked_download(*args,**kwargs):
            downloads.append(threading.get_ident());return download(*args,**kwargs)
        monkeypatch.setattr(service,'_download_dataset',tracked_download)
        instance.restore_planning_plan();await tasks[-1];await asyncio.sleep(.01)
        assert not s.plan_error and s.plan_mass==3.25 and s.plan_name=='Restored silicon'
        assert s.plan_current and s.plan_result_ready and s.plan_figure['data']
        assert s.plan_save_message.startswith('Opened Restored silicon.')
        assert callbacks and all(thread==owner and guarded for thread,guarded in callbacks)
        assert downloads and all(thread!=owner for thread in downloads)
        published={key:value for _,topic,data in frames if 'trame.state.topic' in topic for key,value in data.items()}
        assert published['plan_mass']==3.25 and published['plan_current'] is True
        assert published['plan_result_ready'] is True and published['plan_figure']['data']
        assert commits and all(thread==owner for thread,_ in commits)
        assert not s.plan_busy and not s.plan_can_cancel and not instance._planning_restoring
    asyncio.run(check())


def test_failed_export_keeps_download_silent_and_invalidates_results_on_owner_loop(tmp_path,monkeypatch):
    async def check():
        owner=threading.get_ident();instance,s,tasks,frames,commits,_=connected_planner(tmp_path)
        result(instance);s.flush();await asyncio.sleep(0);commits.clear();frames.clear()
        def fail():raise ValueError('The saved result changed')
        monkeypatch.setattr(instance,'_planning_build_bundle',fail)
        instance.export_planning_plan();await tasks[-1];await asyncio.sleep(.01)
        assert s.plan_error=='The saved result changed' and not s.plan_current and not s.plan_result_ready
        assert not any('trame.actions.topic' in topic for _,topic,_ in frames)
        assert all(thread==owner for thread,_ in commits)
        assert not s.plan_busy and not s.plan_can_cancel
    asyncio.run(check())


def test_failed_history_restore_preserves_inputs_and_publishes_error_on_owner_loop(tmp_path):
    async def check():
        owner=threading.get_ident();service=History(tmp_path);service.output_root=tmp_path
        instance,s,tasks,frames,commits,_=connected_planner(tmp_path,service)
        service.data['broken']=b'not a planner archive'
        instance._planning_history_ids={'broken'};s.plan_saved_id='history:broken'
        result(instance);s.flush();await asyncio.sleep(0);commits.clear();frames.clear()
        original_cif=s.plan_cif;original_mass=s.plan_mass
        instance.restore_planning_plan();await tasks[-1];await asyncio.sleep(.01)
        assert s.plan_error and not s.plan_current and not s.plan_result_ready
        assert (s.plan_cif,s.plan_mass)==(original_cif,original_mass)
        assert not s.plan_busy and not s.plan_can_cancel and not instance._planning_restoring
        assert commits and all(thread==owner for thread,_ in commits)
        published={key:value for _,topic,data in frames if 'trame.state.topic' in topic for key,value in data.items()}
        assert published['plan_error']==s.plan_error and published['plan_current'] is False
    asyncio.run(check())


def test_failed_owner_delivery_propagates_timeout_and_invalidates_export(tmp_path,monkeypatch):
    async def check():
        instance,s,tasks,_,_,_=connected_planner(tmp_path)
        result(instance);s.flush()
        def fail(*_):raise TimeoutError('Download delivery failed')
        monkeypatch.setattr(instance,'download_file',fail)
        instance.export_planning_plan();await asyncio.wait_for(tasks[-1],2)
        assert s.plan_error=='Download delivery failed' and not s.plan_current and not s.plan_result_ready
        assert not s.plan_busy and not s.plan_can_cancel
        # A timeout raised by an owner-loop action is not a Future wait timeout.
        with pytest.raises(TimeoutError,match='Download delivery failed'):
            await asyncio.wait_for(asyncio.to_thread(instance._planning_on_owner,lambda:fail()),2)
    asyncio.run(check())


@pytest.mark.parametrize('legacy_strings',[False,True])
def test_real_trame_numeric_text_edits_restore_using_declared_types(tmp_path,legacy_strings):
    service=History(tmp_path);service.output_root=tmp_path/'original'
    original,source=planner(tmp_path/'original',service);result(original)
    source.plan_mass=3.25
    if legacy_strings:source.plan_step='0.0004';source.plan_seed='42'
    original.save_planning_plan();selected=source.plan_saved_id
    async def check():
        service.output_root=tmp_path/'fresh'
        instance,s,tasks,_,_,_=connected_planner(tmp_path/'fresh',service)
        instance._planning_history_ids={selected.removeprefix('history:')}
        s.plan_step=.0001 if legacy_strings else '0.0001'
        s.plan_seed=45
        s.plan_saved_id=selected;s.flush()
        assert not s.plan_current
        instance.restore_planning_plan();await tasks[-1]
        assert not s.plan_error and s.plan_current and s.plan_result_ready
        assert s.plan_step==.0004 and type(s.plan_step) is float
        assert s.plan_seed==42 and type(s.plan_seed) is int
        assert s.plan_mass==3.25 and type(s.plan_mass) is float
        assert instance._planning_setting_types['plan_step'] is float
        assert instance._planning_request()['relative_step']==.0004
        assert s.plan_save_message.startswith('Opened ')
    asyncio.run(check())


@pytest.mark.parametrize('key,value,expected',[
    ('plan_step','0.0001',.0001),('plan_seed','45',45),('plan_profile_bank','2',2),
    ('plan_step_ui','3',3),('plan_look','23',23),('plan_mass','2.2',2.2),
    ('plan_radius','0.295',.295),('plan_packing','1',1.),('plan_beam_width','0.59',.59),
    ('plan_beam_height','4',4.),('plan_current_ua','1500',1500.),('plan_power','1.4',1.4),
    ('plan_energy','1.3',1.3),('plan_yield_ratio','0.6',.6),('plan_yield_ratio','',None),
    ('plan_moment_multiplier','0',0.),('plan_phase_1_percent','100',100.),
    ('plan_phase_2_size',1.5,'1.5'),('plan_phase_3_microstrain',0,'0'),
    ('plan_phase_1_size','',''),('plan_phase_2_microstrain','',''),
])
def test_declared_numeric_controls_accept_only_their_numeric_text_schema(tmp_path,key,value,expected):
    instance,_=planner(tmp_path)
    normalized=_planning_restore_settings({key:value},instance._planning_setting_types)
    assert normalized[key]==expected and type(normalized[key]) is type(expected)


@pytest.mark.parametrize('key,value',[
    ('plan_step',True),('plan_seed',False),('plan_phase_1_percent',True),
    ('plan_step','nan'),('plan_mass','inf'),('plan_step',float('-inf')),('plan_step','1'*129),
    ('plan_step',0),('plan_radius',-1),('plan_packing',0),('plan_packing',1.1),
    ('plan_seed',-1),('plan_seed',2**32),('plan_seed',3.5),
    ('plan_profile_bank',0),('plan_profile_bank',101),('plan_profile_bank','1.5'),
    ('plan_step_ui',4),('plan_look',24),('plan_moment_multiplier',11),
    ('plan_phase_1_percent',101),('plan_phase_2_percent',-1),
    ('plan_phase_1_size',0),('plan_phase_1_microstrain',-1),('plan_yield_ratio',False),
    ('plan_configuration',123),('plan_sites','Er1'),('plan_current',1),
])
def test_declared_setting_schema_rejects_invalid_numeric_and_generic_values(tmp_path,key,value):
    instance,_=planner(tmp_path)
    with pytest.raises(ValueError,match='Invalid saved planner setting: '+key):
        _planning_restore_settings({key:value},instance._planning_setting_types)


def test_every_saved_field_has_a_declared_type_and_bad_numeric_input_is_atomic(tmp_path):
    instance,s=planner(tmp_path/'original');document,files=instance._planning_build_bundle()
    assert set(FIELDS)|set(EXTRA_FIELDS)<=instance._planning_setting_types.keys()
    document['settings']['plan_step']='nan'
    fresh,target=planner(tmp_path/'fresh');target.plan_mass=2.2
    before=target.plan_cif
    with pytest.raises(ValueError,match='Invalid saved planner setting: plan_step'):
        fresh._planning_apply_bundle(document,files)
    assert target.plan_mass==2.2 and target.plan_cif==before


@pytest.mark.parametrize('name',['../outside.cif','/absolute.cif','C:\\outside.cif','inputs/../outside.cif'])
def test_archive_path_escape_is_rejected(name):
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w') as archive:archive.writestr(name,b'unsafe')
    with pytest.raises(ValueError,match='relative'):bundle.read_archive(data.getvalue())


def test_duplicate_link_and_unmanifested_files_are_rejected(tmp_path):
    instance,_=planner(tmp_path);document,files=instance._planning_build_bundle()
    raw=bundle.archive(document,files)
    for mode in ('duplicate','link','extra'):
        data=io.BytesIO(raw)
        with zipfile.ZipFile(data,'a') as archive:
            if mode=='duplicate':archive.writestr('plan.json',b'{}')
            elif mode=='link':
                item=zipfile.ZipInfo('inputs/link.cif');item.external_attr=0o120777<<16;archive.writestr(item,b'../elsewhere')
            else:archive.writestr('unmanifested.cif',b'extra')
        with pytest.raises(ValueError):bundle.read_archive(data.getvalue())


def test_corrupt_result_and_absolute_backend_fail_before_state_changes(tmp_path):
    instance,_=planner(tmp_path/'original');result(instance);document,files=instance._planning_build_bundle()
    fresh,t=planner(tmp_path/'fresh');t.plan_mass=6.5
    wrong=copy.deepcopy(document);wrong['backend']['_planning_output']='/etc'
    with pytest.raises(ValueError,match='outside'):
        fresh._planning_apply_bundle(wrong,files)
    assert t.plan_mass==6.5
    changed=dict(files);changed['results/output/curves.npz']=b'changed'
    with pytest.raises(ValueError,match='changed'):
        fresh._planning_apply_bundle(document,changed)
    assert t.plan_mass==6.5


def test_object_arrays_nonfinite_and_oversized_headers_are_rejected():
    typed=io.BytesIO();np.savez(typed,model_type='compound_poisson_reference')
    bundle.validate_arrays(typed.getvalue())  # fixed text metadata, without pickle
    for values in ([object()],[float('nan')]):
        data=io.BytesIO();np.savez(data,values=values)
        with pytest.raises(ValueError):bundle.validate_arrays(data.getvalue())
    array=io.BytesIO()
    np.lib.format.write_array_header_1_0(array,dict(descr='<f8',fortran_order=False,shape=(10**12,)))
    data=io.BytesIO()
    with zipfile.ZipFile(data,'w',zipfile.ZIP_DEFLATED) as archive:archive.writestr('values.npy',array.getvalue())
    with pytest.raises(ValueError,match='shape'):bundle.validate_arrays(data.getvalue())


def test_client_cannot_invent_saved_native_authority(tmp_path):
    instance,s=planner(tmp_path);root=result(instance)
    document,files=instance._planning_build_bundle()
    document['backend']['_planning_prepared']={'gpx':{'$path':'results/output/forward.gpx'},'gpx_sha256':hashlib.sha256((root/'forward.gpx').read_bytes()).hexdigest()}
    document['backend']['_planning_prepared_kind']='magnetic'
    instance._planning_apply_bundle(document,files,trusted=False)
    assert instance._planning_prepared is None and instance._planning_prepared_kind is None
    with pytest.raises(ValueError):instance._planning_native_input(instance._planning_output/'forward.gpx')


def test_magnetic_catalog_snapshot_excludes_native_scratch_and_dispatch_locks(tmp_path):
    instance,_=planner(tmp_path);root=instance._planning_root()/'discovery';root.mkdir()
    catalog=dict(candidates=[],k=['0','0','0'])
    (root/'catalog.json').write_text(json.dumps(catalog))
    scratch=root/'native-dispatches';scratch.mkdir();(scratch/'job.lock').write_text('lock')
    (root/'worker.log').write_text('private scratch should not be needed to restore a catalog')
    instance._planning_catalog=(catalog,{}, {},root)
    document,files=instance._planning_build_bundle()
    assert 'catalogs/catalog/catalog.json' in files
    assert not any('worker.log' in path or '.lock' in path for path in files)


def test_original_scenario_binding_detects_edited_sample_isotopes_and_exposure(tmp_path):
    instance,s=planner(tmp_path)
    original=instance._planning_request()
    assert instance._planning_matches_original(original,tmp_path)
    s.plan_mass*=2
    assert not instance._planning_matches_original(original,tmp_path)
    s.plan_mass/=2;s.plan_seconds='1, 2'
    assert not instance._planning_matches_original(original,tmp_path)
    s.plan_seconds='60, 300, 1800';s.plan_isotopes='{"Si":"28"}'
    assert not instance._planning_matches_original(original,tmp_path)


def test_authentic_result_cannot_be_assigned_to_edited_saved_settings(tmp_path):
    instance,_=planner(tmp_path/'source');root=result(instance)
    original=instance._planning_request()
    curve=root/'curves.npz'
    instance._planning_native_artifacts={str(curve):dict(sha256=hashlib.sha256(curve.read_bytes()).hexdigest(),
        handle={'fixture':'verified separately'},relative='result/curves.npz')}
    document,files=instance._planning_build_bundle()
    fresh,s=planner(tmp_path/'fresh')
    fresh._planning_runtime_cache=SimpleNamespace(
        verify_artifacts=lambda values,_:values,verify_result=lambda *_:(original,tmp_path))
    fresh._planning_apply_bundle(document,files,trusted=True)
    assert s.plan_current
    changed=copy.deepcopy(document);changed['settings']['plan_mass']=99.
    fresh._planning_apply_bundle(changed,files,trusted=True)
    assert not s.plan_current and not s.plan_result_ready and not s.plan_figure['data']
    assert 'differ from the original' in s.plan_message


def test_structural_restore_binds_added_removed_and_changed_cell_overrides(tmp_path):
    instance,s=planner(tmp_path)
    original=instance._planning_request()
    original['prepared_structural']=dict(candidate={'id':'child-a'},modes={'mode-a':.1},
        cell=[5.,5.,8.,90.,90.,90.],requested_cell=None)
    s.plan_transition='structural';s.plan_structural_candidate='child-a';s.plan_structural_parameters='{"mode-a":0.1}'
    assert instance._planning_matches_original(original,tmp_path)
    s.plan_structural_cell='[5,5,9,90,90,90]'
    assert not instance._planning_matches_original(original,tmp_path)
    original['prepared_structural']['requested_cell']=[5.,5.,9.,90.,90.,90.]
    assert instance._planning_matches_original(original,tmp_path)
    s.plan_structural_cell=''
    assert not instance._planning_matches_original(original,tmp_path)
    original['prepared_structural'].pop('requested_cell')
    assert not instance._planning_matches_original(original,tmp_path)


@pytest.mark.parametrize('count,expected', [
    (0, 'No saved plans found in Galaxy History.'),
    (1, 'Found 1 saved plan in Galaxy History.'),
    (3, 'Found 3 saved plans in Galaxy History.'),
])
def test_refresh_publishes_terminal_status_after_worker_and_guards_duplicate_clicks(tmp_path, count, expected):
    async def check():
        owner=threading.get_ident()
        service=History(tmp_path);service.output_root=tmp_path
        instance,s,tasks,frames,commits,_=connected_planner(tmp_path,service)
        s.plan_message='Existing simulation status';s.flush()
        await asyncio.sleep(.01);frames.clear();commits.clear()
        entered=threading.Event();release=threading.Event();calls=[];callbacks=[]
        rows=[dict(id='saved'+str(index),name='Saved plan '+str(index)) for index in range(count)]
        def search(**kwargs):
            calls.append((threading.get_ident(),kwargs));entered.set()
            assert release.wait(5), 'The test must release its bounded worker'
            return rows
        service.search_history_datasets=search
        @s.change('plan_save_message','plan_saved_options')
        def changed(**_):callbacks.append(threading.get_ident())
        instance.refresh_planning_plans()
        try:
            assert await asyncio.to_thread(entered.wait,2)
            assert s.plan_busy and not s.plan_can_cancel
            assert s.plan_save_message=='Reading saved plans from Galaxy History…'
            instance.refresh_planning_plans()
            assert len(tasks)==1 and len(calls)==1
        finally:release.set()
        await asyncio.wait_for(tasks[-1],2);await asyncio.sleep(.01)
        assert not s.plan_error and not s.plan_busy and not s.plan_can_cancel
        assert s.plan_save_message==expected
        assert s.plan_message=='Existing simulation status'
        assert {row['value'] for row in s.plan_saved_options}=={'history:'+row['id'] for row in rows}
        assert all(thread!=owner for thread,_ in calls)
        assert callbacks and all(thread==owner for thread in callbacks)
        assert commits and all(thread==owner for thread,_ in commits)
        messages=[data['plan_save_message'] for _,topic,data in frames
                  if 'trame.state.topic' in topic and 'plan_save_message' in data]
        assert messages==['Reading saved plans from Galaxy History…',expected]
        published={key:value for _,topic,data in frames if 'trame.state.topic' in topic for key,value in data.items()}
        assert published['plan_busy'] is False
        if count:assert published['plan_saved_options']==s.plan_saved_options
    asyncio.run(check())


def test_failed_refresh_keeps_existing_choices_and_publishes_error_without_success(tmp_path):
    async def check():
        service=History(tmp_path);service.output_root=tmp_path
        instance,s,tasks,frames,_,_=connected_planner(tmp_path,service)
        previous=[dict(title='Previously found plan',value='history:existing')]
        s.plan_saved_options=previous;s.plan_saved_id='history:existing';s.flush()
        await asyncio.sleep(.01);frames.clear()
        def fail(**_):raise TimeoutError('History search timed out')
        service.search_history_datasets=fail
        instance.refresh_planning_plans();await asyncio.wait_for(tasks[-1],2)
        await asyncio.sleep(.01)
        assert s.plan_error=='History search timed out'
        assert s.plan_save_message=='Saved-plan refresh failed.'
        assert not s.plan_busy and not s.plan_can_cancel
        assert s.plan_saved_options==previous and s.plan_saved_id=='history:existing'
        published={key:value for _,topic,data in frames if 'trame.state.topic' in topic for key,value in data.items()}
        assert published['plan_error']==s.plan_error
        assert published['plan_save_message']==s.plan_save_message
    asyncio.run(check())


def test_unavailable_history_does_not_claim_an_empty_successful_search(tmp_path):
    async def check():
        instance,s,tasks,_,_,_=connected_planner(tmp_path)
        instance.refresh_planning_plans();await asyncio.wait_for(tasks[-1],2)
        assert not s.plan_error and not s.plan_busy and not s.plan_can_cancel
        assert s.plan_save_message=='Galaxy History is unavailable in this session.'
    asyncio.run(check())
