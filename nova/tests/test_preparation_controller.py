"""User interaction and stale-setup controls, with an in-memory Galaxy service."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.preparation_controller import PreparationMixin
from radar_pd_nova import preparation_controller as module
from test_model_runtime import remote
from test_preparation_workspace import workspace,finish_inspection,catalog,PARENT


class State(SimpleNamespace):
    def change(self,*names):return lambda callback:callback
    def flush(self):pass


@pytest.fixture
def app(workspace):
    store,key,service=workspace
    app=PreparationMixin();app.server=SimpleNamespace(state=State(workspace_view='experiment',
        setup_collapsed=False,live_assistant_open=True));app.service=service;app._init_preparation()
    app._preparation_workspace=store;app._prep_start_driver=lambda:None
    return app,store,key,service


def load_context(app,store,key,service,tmp_path):
    finish_inspection(store,key,service,tmp_path)
    app.server.state.prep_reference=key
    asyncio.run(app._prep_load_reference(key))


def test_open_and_close_preserve_live_workspace_and_do_not_start_jobs(app):
    app,store,key,service=app;s=app.server.state
    app.open_preparation_workspace()
    assert s.workspace_view=='preparation' and s.setup_collapsed and not s.live_assistant_open
    assert s.prep_reference_options and not service.submissions
    app.close_preparation_workspace()
    assert s.workspace_view=='experiment' and not s.setup_collapsed


def test_unregistered_parent_path_is_rejected_before_any_server_file_read(app,tmp_path,monkeypatch):
    app,_,_,service=app;path=tmp_path/'private.cif';path.write_text('must not be read')
    app._prep_parent_upload=SimpleNamespace(current_path=None);app.server.state.prep_parent_path=str(path)
    monkeypatch.setattr(Path,'read_bytes',lambda *a:pytest.fail('Do not read unregistered server inputs'))
    with pytest.raises(ValueError,match='Choose a completed'):asyncio.run(app._inspect_preparation_inputs())
    assert not service.submissions


def test_profile_type_and_grid_are_checked_and_bank_capture_is_immutable(app,tmp_path):
    app,_,_,_=app;s=app.server.state;path=tmp_path/'cw.instprm';path.write_text('Type:PNC;Lam:1.5\n')
    app._prep_profile_upload=SimpleNamespace(current_path=path);s.prep_profile_path=str(path)
    app._preparation_profile_changed()
    assert s.prep_grid_kind=='PNC' and s.prep_upper=='160'
    s.prep_bank_label='Detector A';app.add_preparation_bank()
    captured=app._prep_banks['Detector A']['instrument'];assert captured.read_bytes()==path.read_bytes()
    assert s.prep_bank_rows[0]['profile']=='cw.instprm'
    path.write_text('changed upload');assert captured.read_text()=='Type:PNC;Lam:1.5\n'
    assert not s.prep_context_ready
    s.prep_bank_label='Invalid';s.prep_bins='2.5';app.add_preparation_bank()
    assert s.prep_error and list(app._prep_banks)==['Detector A']


def test_changed_inputs_after_capture_do_not_launch_an_old_inspection(app,tmp_path,monkeypatch):
    app,store,_,service=app;s=app.server.state
    parent=tmp_path/'parent.cif';parent.write_text('contract parent')
    profile=tmp_path/'bank.instprm';profile.write_text('contract profile')
    app._prep_parent_upload=SimpleNamespace(current_path=parent);s.prep_parent_path=str(parent)
    app._prep_banks={'bank':dict(instrument=profile,grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1))}
    monkeypatch.setattr(module,'inspect_cif_upload',lambda *a:None)
    original=store.inspect
    def changed(**kwargs):
        result=original(**kwargs);app._preparation_inputs_changed();return result
    monkeypatch.setattr(store,'inspect',changed)
    asyncio.run(app._inspect_preparation_inputs())
    assert not service.submissions and 'without starting jobs' in s.prep_message


def test_inspected_parent_populates_exact_k_sites_and_explicit_ion_assumptions(app,tmp_path):
    app,store,key,service=app;load_context(app,store,key,service,tmp_path);s=app.server.state
    assert s.prep_context_ready and s.prep_parent_title=='Parent · P 1'
    assert s.prep_selected_k==['0,0,0'] and s.prep_selected_banks==['bank 1','bank 2']
    s.prep_sites=['Fe1'];app._preparation_site_selection_changed()
    assert s.prep_site_rows==[dict(label='Fe1',element='Fe',form_factor='',lande_g='2')]
    app.set_preparation_ion('Fe1','Fe+3');app.set_preparation_g('Fe1','2.1')
    assert s.prep_site_rows[0]['form_factor']=='Fe+3' and s.prep_site_rows[0]['lande_g']=='2.1'
    s.prep_custom_k='0,0,2/6';app.add_preparation_k()
    assert '0,0,1/3' in s.prep_selected_k


def test_failed_family_remains_visible_and_cannot_become_a_silent_shortlist(app,tmp_path,monkeypatch):
    app,store,key,service=app;load_context(app,store,key,service,tmp_path);s=app.server.state
    s.prep_custom_k='0,0,1/2';app.add_preparation_k()
    class Discovery:
        def __init__(self,store):pass
        def discover(self,reference,kind,k,**kwargs):
            if k[-1]=='1/2':raise ValueError('Family service unavailable')
            return dict(catalog=catalog(PARENT,k),directory='',cached=False)
    monkeypatch.setattr(module,'PreparationDiscovery',Discovery)
    asyncio.run(app._discover_preparation_models())
    assert len(s.prep_discovery_rows)==2 and not s.prep_discovery_ready
    assert s.prep_discovery_rows[1]['state']=='Family service unavailable'
    with pytest.raises(ValueError,match='currently selected'):asyncio.run(app._start_preparation_models())
    assert len(service.submissions)==1


def test_late_discovery_cannot_replace_a_changed_parent_or_k_scope(app,tmp_path,monkeypatch):
    app,store,key,service=app;load_context(app,store,key,service,tmp_path);s=app.server.state
    class Discovery:
        def __init__(self,store):pass
        def discover(self,reference,kind,k,**kwargs):
            s.prep_selected_k=[];app._preparation_scope_changed()
            return dict(catalog=catalog(PARENT,k),directory='',cached=False)
    monkeypatch.setattr(module,'PreparationDiscovery',Discovery)
    asyncio.run(app._discover_preparation_models())
    assert not s.prep_discovery_ready and not s.prep_discovery_rows and not app._prep_catalogs


def test_changed_site_assumptions_after_plan_capture_do_not_start_old_models(app,tmp_path,monkeypatch):
    app,store,key,service=app;load_context(app,store,key,service,tmp_path);s=app.server.state
    s.prep_sites=['Fe1'];app.set_preparation_ion('Fe1','Fe+3')
    app._prep_catalogs=[dict(catalog=catalog(PARENT,['0','0','0']))]
    app._prep_discovered_identity=app._scope_identity();s.prep_discovery_ready=True
    original=store.create_models
    def changed(*args,**kwargs):
        result=original(*args,**kwargs);app.set_preparation_g('Fe1','2.5');return result
    monkeypatch.setattr(store,'create_models',changed)
    asyncio.run(app._start_preparation_models())
    assert len(service.submissions)==1 and 'without starting jobs' in s.prep_message


def test_repeated_clicks_have_one_action_owner_before_the_first_await(app):
    app,_,_,_=app;calls=[]
    async def operation():calls.append('once');await asyncio.sleep(0)
    async def run():
        first=app._prep_action(operation(),'One action')
        assert app.server.state.prep_busy
        second=app._prep_action(operation(),'Second action')
        assert second is None
        await first
    asyncio.run(run())
    assert calls==['once'] and not app.server.state.prep_busy


def test_transient_queue_observation_is_repolled_without_a_new_dispatch(app,monkeypatch):
    app,store,_,_=app;calls=[];s=app.server.state;s.prep_library=''
    def tick():
        calls.append('same job')
        if len(calls)==1:raise TimeoutError('temporary observation failure')
        return dict(active=0,queue={'paused':True},coverage={'complete':True,'all_prepared':True})
    store._queues={'same':SimpleNamespace(tick=tick)}
    async def immediate(*args):pass
    monkeypatch.setattr(module.asyncio,'sleep',immediate)
    asyncio.run(app._prep_drive())
    assert calls==['same job','same job']
    assert not s.prep_error


@pytest.mark.parametrize('base',['https://person:secret@galaxy.invalid','file:///private',
    'https://galaxy.invalid?token=secret','https://galaxy.invalid#fragment'])
def test_download_link_cannot_include_embedded_credentials_or_other_url_schemes(base):
    with pytest.raises(ValueError):module.download_url(base,'dataset1')


def test_download_link_retains_galaxy_proxy_prefix():
    assert module.download_url('https://galaxy.invalid/prototype/','dataset1')==(
        'https://galaxy.invalid/prototype/api/datasets/dataset1/display?to_ext=zip')


def test_library_table_is_bounded_without_hiding_total_scope(app,tmp_path):
    app,store,key,service=app;load_context(app,store,key,service,tmp_path)
    c=catalog(PARENT,['0','0','0']);template=c['candidates'][0]
    c['candidates']=[dict(copy.deepcopy(template),id='magnetic:'+str(i)) for i in range(40)]
    key=store.create_models(key,name='Large display',kind='magnetic',catalogs=[{'catalog':c}],
        sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)])
    s=app.server.state;s.prep_library=key;app._preparation_library_changed()
    assert s.prep_table_count==80 and len(s.prep_entries)==25 and s.prep_pages==4
    s.prep_page=4;app._preparation_table_changed()
    assert len(s.prep_entries)==5 and '80 total' in s.prep_coverage
    assert s.prep_can_resume and not s.prep_can_pause


def test_completed_library_has_no_pending_work_controls(app,tmp_path):
    app,store,key,service=app;load_context(app,store,key,service,tmp_path)
    s=app.server.state;s.prep_library=key;app._preparation_library_changed()
    assert not s.prep_can_resume and not s.prep_can_pause
