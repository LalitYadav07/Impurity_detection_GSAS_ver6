import json
from types import SimpleNamespace
import numpy as np
import pytest
from radar_pd_nova.webmon.session import WebmonSession,digest
from radar_pd_nova.webmon.plots import checkpoint_review
from radar_pd_nova.webmon.ui import WebmonMixin


def session(tmp_path):
    sources=[]
    for run in [1,2]:
        p=tmp_path/f'{run}.gpx';p.write_bytes(str(run).encode())
        sources.append(dict(id=str(run),source_path=str(p),source_sha256=digest(p)))
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(facility_root=str(tmp_path),archive=str(tmp_path/'archive'),checkpoint_options={'IPTS-7/10':sources})))
    return WebmonSession(manifest),sources


def test_selection_is_scoped_and_source_hash_is_enforced(tmp_path):
    s,options=session(tmp_path)
    assert s.selected_checkpoint('7','10')['id']=='1'
    s.select_checkpoint('7','10','2')
    assert s.selected_checkpoint('IPTS-7','10')['id']=='2'
    with pytest.raises(ValueError,match='this experiment'):s.select_checkpoint('8','10','2')
    with pytest.raises(ValueError,match='this experiment'):s.select_checkpoint('7','11','2')
    from pathlib import Path
    Path(options[1]['source_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):s.selected_checkpoint('7','10')


def test_reference_q_plot_keeps_sorted_observations_and_residuals():
    data=dict(x=np.array([100,200,300]),y=np.array([2,3,4]),calc=np.array([1,4,2]),sigma=np.ones(3),valid=[True,True,True],live_x=np.array([100,200,300]),live_scaled=np.array([2,3,4]))
    f=checkpoint_review(data,axis='q',difc=100)
    np.testing.assert_allclose(f.data[0].x,2*np.pi/np.array([3,2,1]))
    np.testing.assert_array_equal(f.data[-1].y,[2,-1,1])
    assert f.layout.xaxis2.title.text=='Q (Å⁻¹)'
    assert f.to_json()


def test_changed_checkpoint_clears_old_acceptance_and_results(tmp_path):
    backend,_=session(tmp_path)
    backend.binding=lambda *a:dict(description='second',review=[],k=['0','0','0'],ion='Fe+3')
    obj=WebmonMixin();obj._webmon=backend;obj._webmon_review_checkpoint=lambda c:None;obj.webmon_saved_list=lambda:None
    s=SimpleNamespace(powgen_ipts='7',webmon_run='10',webmon_busy=False,webmon_accepted=True,webmon_has_result=True,flush=lambda:None)
    obj.server=SimpleNamespace(state=s);obj.webmon_checkpoint_changed('2')
    assert not s.webmon_accepted and not s.webmon_has_result and s.webmon_context_ready
    assert s.webmon_target=='' and s.webmon_checkpoint=='2'


def test_busy_selection_does_not_switch_source(tmp_path):
    backend,_=session(tmp_path)
    s=SimpleNamespace(powgen_ipts='7',webmon_run='10',webmon_busy=True,flush=lambda:None)
    obj=WebmonMixin();obj._webmon=backend;obj.server=SimpleNamespace(state=s)
    obj.webmon_checkpoint_changed('2')
    assert backend.selected_checkpoint('7','10')['id']=='1'
    assert s.webmon_checkpoint=='1'


def test_luna_handoff_pins_one_source_and_never_arms_future_runs(tmp_path):
    backend,opts=session(tmp_path)
    src=dict(id='source-1',source_sha256=opts[0]['source_sha256'],experiment='expt',experiment_label='Experiment',scan_key='run-1')
    opts[0]['refinement_source']=src;backend.config['checkpoint_options']['IPTS-7/10'][0]=opts[0]
    workspace=SimpleNamespace(state=dict(sources={},jobs={}))
    workspace.observe=lambda items:workspace.state['sources'].update({x['id']:x for x in items})
    s=SimpleNamespace(powgen_ipts='7',webmon_run='10',webmon_busy=False,flush=lambda:None)
    obj=WebmonMixin();obj._webmon=backend;obj.server=SimpleNamespace(state=s);obj._refinement_store=lambda:workspace
    called=[];obj._refinement_sync=lambda:called.append('sync');obj.inspect_refinement_project=lambda:called.append('inspect')
    obj.webmon_open_refinement()
    assert s.rf_source=='source-1' and s.rf_scans==['run-1'] and not s.rf_future
    assert s.experiment_result_view=='refinement' and called==['sync','inspect']


def test_register_polish_checks_ancestry_and_rolls_back_invalid_binding(tmp_path):
    s,options=session(tmp_path);view=tmp_path/'view.npz';view.write_bytes(b'view')
    child=dict(options[0],id='child',parent_checkpoint='1',view_path=str(view),view_sha256=digest(view))
    with pytest.raises(ValueError,match='No reviewed'):s.register_checkpoint('7','10',child)
    assert s.selected_checkpoint('7','10')['id']=='1'
    assert not s.extra_checkpoints['IPTS-7/10']
    with pytest.raises(ValueError,match='no parent'):s.register_checkpoint('8','10',child)


def test_return_path_rejects_a_completed_job_from_another_checkpoint(tmp_path):
    import asyncio
    s,options=session(tmp_path);job=dict(status='completed',source=dict(source_sha256='wrong'))
    obj=WebmonMixin();obj._webmon_process=None;obj._webmon_cancelled=False;obj._webmon=s;obj._refinement_store=lambda:SimpleNamespace(state=dict(jobs={'job':job}))
    state=SimpleNamespace(webmon_busy=True,flush=lambda:None)
    obj.server=SimpleNamespace(state=state)
    asyncio.run(obj._webmon_prepare_polish(('7','10','1','job')))
    assert 'exact checkpoint' in state.webmon_message and not state.webmon_busy
    assert not s.extra_checkpoints


def test_reflection_worker_mode_requires_owning_active_backend(tmp_path):
    from radar_pd_nova.webmon.reflections import export_reflections
    with pytest.raises(RuntimeError,match='owning process'):
        export_reflections(tmp_path,tmp_path,backend_active=True)
