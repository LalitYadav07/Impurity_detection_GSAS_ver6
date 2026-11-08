"""User actions share reviewed Model settings and the existing acquisition owner."""
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.contracts import digest
from radar_pd_nova.webmon.ui import WebmonMixin
from radar_pd_nova.webmon import structural_ui
from radar_pd_nova.webmon.stream import LatestQueue,ReviewRequired
from test_structural_stream import setup


class State(SimpleNamespace):
    def update(self,values):self.__dict__.update(values)
    def flush(self):pass


def view(tmp_path,count=1):
    stream,producer,source=setup(tmp_path,count)
    root=stream.root;p=__import__('gpx_recipe.structural_live.job',fromlist=['reviewed_job']).reviewed_job(stream.templates[0]['directory'])[1]
    template=copy.deepcopy(stream.templates[0]);row,receipt,_=stream.session.capture('1','2')
    template['mapping']['snapshot_sha256']=receipt['content_sha256'];template['mapping']['source_id']=source['id']
    obj=WebmonMixin();s=State(webmon_busy=False,webmon_refine_transfer=False,webmon_run='2',
        webmon_published_source=source['id'],sx_kind='structural',sx_approved=True,webmon_stream_active=False,
        webmon_stale=False,webmon_has_result=False,webmon_message='',webmon_accepted=False,
        webmon_revision=receipt['content_sha256'][:12],webmon_target='',webmon_analysis_revision='')
    obj.server=SimpleNamespace(state=s);obj._init_structural_live();s.webmon_analysis_kind='structural'
    obj._webmon=stream.session;obj._webmon.config=dict(jobs=str(root/'live-jobs'),allow_conditional_stream=True)
    obj._webmon_published_catalog={source['id']:source};obj._webmon_reviewed_mapping=template['mapping']
    obj._webmon_scope_snapshot=lambda:'owner';obj._webmon_scope_current=lambda owner:owner=='owner'
    obj._webmon_can_publish=lambda owner:obj._webmon_scope_current(owner)
    obj._webmon_effective_ipts=lambda:'1';obj._webmon_require_target=lambda **kwargs:row
    obj._webmon_require_reference=lambda *args:None;obj._discovery_identity=lambda:'discovery'
    obj._model_form=lambda:'form';obj._symmetry_approved=digest({'reviewed':True})
    obj._symmetry_draft=dict(source_id=source['id'],discovery_identity='discovery',form_signature='form',
        template={'reviewed':True},protocol=p,root=root/'model-inputs',catalog=json.loads((root/'model-inputs/catalog.json').read_text()))
    runtime=SimpleNamespace(resolve=lambda selected:dict(selected))
    obj._model_remote=lambda:runtime;obj._capture_prepared_model_inputs=lambda *args:None
    obj._webmon_stream=None;obj._webmon_queue=None;obj._webmon_task=None;obj._webmon_cancelled=False
    obj._webmon_observed=SimpleNamespace(update=lambda plot:None)
    return obj,producer


def test_adopted_structural_plan_shares_inputs_and_keeps_all_reviewed_children(tmp_path):
    obj,_=view(tmp_path,2)
    asyncio.run(obj.webmon_adopt_structural_plan())
    s=obj.server.state;plan=obj._webmon_structural_plan
    assert s.webmon_structural_ready and not s.webmon_busy
    assert len(plan['templates'])==2
    assert len({p['source'] for p in plan['templates']})==1
    assert len(list((plan['root']/'models').rglob('*.gpx')))==2  # one target and one reviewed reference
    for template in plan['templates']:
        assert json.loads((Path(template['directory'])/'job.json').read_text())['schema']=='radar-symmetry-job/v2'
    assert 'all physical modes' in s.webmon_structural_summary
    assert not s.webmon_structural_has_result


@pytest.mark.parametrize('change',['approval','source','form','mapping'])
def test_adoption_rejects_unreviewed_or_changed_inputs_without_starting_jobs(tmp_path,change):
    obj,_=view(tmp_path);s=obj.server.state
    if change=='approval':s.sx_approved=False
    if change=='source':obj._symmetry_draft['source_id']='another'
    if change=='form':obj._model_form=lambda:'changed'
    if change=='mapping':obj._webmon_reviewed_mapping['snapshot_sha256']='a'*64
    asyncio.run(obj.webmon_adopt_structural_plan())
    assert not s.webmon_structural_ready and obj._webmon_structural_plan is None
    assert not s.webmon_busy and s.webmon_structural_message


def test_transfer_edit_during_capture_cannot_adopt_a_different_plan(tmp_path):
    obj,_=view(tmp_path)
    obj._capture_prepared_model_inputs=lambda *args:setattr(obj.server.state,'webmon_refine_transfer',True)
    asyncio.run(obj.webmon_adopt_structural_plan())
    assert not obj.server.state.webmon_structural_ready
    assert 'changed during capture' in obj.server.state.webmon_structural_message


def test_user_can_fit_once_then_follow_without_rebuilding_or_losing_paused_latest(tmp_path):
    obj,producer=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan());plan=obj._webmon_structural_plan
    seen=[]
    async def execute(item,stream,selected):seen.append(item)
    obj._webmon_execute_structural=execute
    async def single():
        obj.webmon_start_structural();await obj._webmon_task;obj.server.state.webmon_busy=False
    asyncio.run(single());assert len(seen)==1
    stream=plan['stream'];obj.webmon_stream_tick=lambda:None
    obj.webmon_follow_structural();assert obj._webmon_stream is stream
    producer['raw']=producer['raw'].replace(b'-12',b'-13')
    pending=stream.capture();obj._webmon_queue.offer(pending)
    obj.webmon_pause_stream();assert plan['paused_pending']==pending
    obj.webmon_follow_structural()
    assert obj._webmon_queue.pending==pending and 'paused_pending' not in plan
    assert obj._webmon_stream is stream


def test_existing_webmon_tick_dispatches_structural_work_and_never_the_magnetic_engine(tmp_path):
    obj,_=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan());calls=[]
    async def execute(item,stream,plan):
        calls.append(item);obj.server.state.webmon_calculation_status='Completed'
    obj._webmon_execute_structural=execute
    obj._webmon_execute=lambda *a:pytest.fail('Structural snapshots must not enter the posterior worker')
    async def run():
        obj.webmon_follow_structural()
        await obj._webmon_task
    asyncio.run(run())
    assert len(calls)==1 and calls[0]['kind']=='structural'
    assert obj._webmon_queue.state['completed']==[calls[0]['id']]


def test_structural_dispatch_recovery_is_explicit_and_retains_the_reviewed_plan(tmp_path,monkeypatch):
    obj,_=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan());plan=obj._webmon_structural_plan
    stream=obj._new_structural_stream();item=stream.capture()
    async def unresolved(*args,**kwargs):raise ReviewRequired('Acknowledgment needs recovery')
    monkeypatch.setattr(structural_ui,'analyse_snapshot',unresolved)
    asyncio.run(obj._webmon_execute_structural(item,stream,plan))
    assert obj.server.state.webmon_structural_recovery
    asyncio.run(obj.webmon_adopt_structural_plan())
    assert obj._webmon_structural_plan is plan and 'Recover' in obj.server.state.webmon_structural_message
    calls=[]
    async def recover(item_arg,stream_arg,plan_arg):calls.append((item_arg,stream_arg,plan_arg))
    obj._webmon_execute_structural=recover
    async def resume():obj.webmon_recover_structural();await obj._webmon_task
    asyncio.run(resume())
    assert calls==[(item,stream,plan)]


def test_partial_structural_fit_appears_without_erasing_previous_checked_results(tmp_path):
    obj,_=view(tmp_path);s=obj.server.state
    obj._webmon_structural_results={'old:model':dict(name='Old',rwp=3.,revision=1)}
    s.webmon_structural_selected='old:model';s.webmon_structural_has_result=True
    s.webmon_structural_choices=[dict(title='Old',value='old:model')]
    obj._structural_result_choices('new')
    assert s.webmon_structural_selected=='old:model' and s.webmon_structural_has_result
    obj._webmon_structural_results['new:model']=dict(name='Current',rwp=2.,revision=2)
    obj._structural_result_choices('new')
    assert s.webmon_structural_selected=='new:model'
    assert 'review required' in s.webmon_structural_choices[0]['title']


@pytest.mark.parametrize('available',[True,False])
def test_current_snapshot_ranking_status_is_visible_without_claiming_a_structure(tmp_path,monkeypatch,available):
    from test_structural_snapshot_ranking import ranking_result
    obj,_=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan());plan=obj._webmon_structural_plan
    stream=obj._new_structural_stream();item=stream.capture();seen=[]
    async def ranking_updates(*args,on_ranking_progress,**kwargs):
        on_ranking_progress(dict(state='queued'))
        seen.append(obj.server.state.webmon_calculation_status)
        on_ranking_progress(dict(state='ranked',result=ranking_result(item['payloads'][0],available=available)))
        return []
    monkeypatch.setattr(structural_ui,'analyse_snapshot',ranking_updates)
    asyncio.run(obj._webmon_execute_structural(item,stream,plan))
    assert seen==['Checking prepared model ranking']
    message=obj.server.state.webmon_structural_ranking
    assert ('SG 63' in message and 'Every reviewed model remains' in message) if available else message=='Public transport fixture'
    assert not obj.server.state.webmon_structural_has_result


def test_job_status_polls_do_not_rehash_every_completed_view_or_redraw_the_selected_fit(tmp_path,monkeypatch):
    obj,_=view(tmp_path,2);asyncio.run(obj.webmon_adopt_structural_plan());plan=obj._webmon_structural_plan
    stream=obj._new_structural_stream();item=stream.capture();draws=[];hashed=[]
    obj.webmon_render_structural=lambda **kwargs:draws.append(obj.server.state.webmon_structural_selected)
    original=structural_ui.file_sha
    def checksum(path):hashed.append(str(path));return original(path)
    monkeypatch.setattr(structural_ui,'file_sha',checksum)
    rows=[]
    for index,payload in enumerate(item['payloads']):
        from gpx_recipe.structural_live.job import reviewed_job
        candidate=reviewed_job(payload['directory'])[2]['id']
        path=tmp_path/f'view-{index}.json';path.write_text('{}')
        rows.append(dict(candidate=candidate,name=candidate,state='queued',revision=item['revision'],rwp=2.-index,
            result=dict(view=str(path))))
    async def status_updates(*args,on_progress,**kwargs):
        on_progress(copy.deepcopy(rows))
        rows[0]['state']='verified_point_fit'
        for _ in range(5):on_progress(copy.deepcopy(rows))
        rows[1]['state']='verified_point_fit';on_progress(copy.deepcopy(rows))
        return rows
    monkeypatch.setattr(structural_ui,'analyse_snapshot',status_updates)
    asyncio.run(obj._webmon_execute_structural(item,stream,plan))
    assert len(obj._webmon_structural_results)==2 and len(hashed)==2
    assert len(draws)==2  # first available fit and final status; queued polls do no plot work
    assert draws[0]==draws[1] and len(obj.server.state.webmon_structural_choices)==2
    assert {r['display_state'] for r in obj.server.state.webmon_structural_rows}=={'Fit available · review required'}


def test_result_collection_recovery_message_is_reachable_in_live_job_details(tmp_path,monkeypatch):
    obj,_=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan());plan=obj._webmon_structural_plan
    stream=obj._new_structural_stream();item=stream.capture()
    reason='Galaxy job finished; result collection needs recovery: Download interrupted'
    async def lost_result(*args,on_progress,**kwargs):
        on_progress([dict(candidate='public',name='Public model',state='needs_recovery',reason=reason)])
        raise ReviewRequired('Structural dispatch is unresolved')
    monkeypatch.setattr(structural_ui,'analyse_snapshot',lost_result)
    asyncio.run(obj._webmon_execute_structural(item,stream,plan))
    row=obj.server.state.webmon_structural_rows[0]
    assert row['reason']==reason and row['display_state']=='Recovery needed'
    assert obj.server.state.webmon_structural_recovery


def test_structural_plot_uses_native_calibration_and_rejects_changed_display_artifact(tmp_path):
    obj,_=view(tmp_path);updates=[]
    from trame.app import get_server
    from trame.ui.vuetify3 import SinglePageLayout
    from trame.widgets import plotly
    server=get_server('structural-live-real-widget',client_type='vue3')
    with SinglePageLayout(server) as layout:
        with layout.content:widget=plotly.Figure()
    def update(figure):
        widget.update(figure);updates.append(figure)
    obj._webmon_structural_fit=SimpleNamespace(update=update)
    path=tmp_path/'view.json'
    write_json(path,dict(schema='radar-symmetry-view/v1',x=[100.,200.,300.],d=[1.,2.,3.],coordinate='tof',
        observed=[1.,3.,2.],calculated=[1.,2.9,2.],background=[0.,0.,0.],weights=[1.,1.,1.],fitted=[True]*3,
        limits=[100.,300.],ticks=[dict(phase='Public',native=[200.],d=[2.])],phases=[dict(name='Public',type='nuclear')]))
    from gpx_recipe.symmetry.compiler import file_sha
    obj._webmon_structural_results['current:model']=dict(name='Public',view_sha256=file_sha(path),
        result=dict(view=str(path),snapshot_sha256='a'*64))
    from gpx_recipe.structural_live.service import workflow_timings
    obj._webmon_structural_results['current:model']['result']['workflow_timings']=workflow_timings(12.,5.,scope='acquisition')
    s=obj.server.state;s.webmon_structural_selected='current:model';s.webmon_structural_axis='q'
    obj.webmon_render_structural();assert updates
    assert any(trace['name']=='Nuclear Bragg' for trace in updates[-1]['data'])
    assert '12.0 s preparation' in s.webmon_structural_timing
    obj._webmon_structural_results['current:model']['result'].pop('workflow_timings')
    obj.webmon_render_structural();assert s.webmon_structural_timing==''
    path.write_text('{}');obj.webmon_render_structural()
    assert len(updates)==2 and 'changed' in s.webmon_structural_message


def test_structural_axis_control_applies_the_new_coordinate_before_plotting(tmp_path):
    obj,_=view(tmp_path);seen=[]
    obj.webmon_render_structural=lambda:seen.append(obj.server.state.webmon_structural_axis)
    obj.webmon_structural_control('webmon_structural_axis','q')
    assert seen==['q']
    obj.webmon_structural_control('webmon_structural_axis','unrelated')
    assert seen==['q']


def test_switching_analysis_kind_keeps_magnetic_and_structural_status_distinct(tmp_path):
    obj,_=view(tmp_path);s=obj.server.state
    obj._webmon_display_kind='magnetic';s.webmon_analysis_kind='magnetic'
    s.webmon_calculation_status='Magnetic complete';s.webmon_count_summary='3 magnetic models'
    obj.webmon_analysis_kind_changed('structural')
    assert s.webmon_calculation_status=='Not started'
    s.webmon_calculation_status='Structural complete';s.webmon_count_summary='2 point fits'
    obj.webmon_analysis_kind_changed('magnetic')
    assert s.webmon_calculation_status=='Magnetic complete' and s.webmon_count_summary=='3 magnetic models'
    obj.webmon_analysis_kind_changed('structural')
    assert s.webmon_calculation_status=='Structural complete' and s.webmon_count_summary=='2 point fits'


def test_two_rapid_clicks_reserve_one_task_and_do_not_consume_another_revision(tmp_path):
    obj,producer=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan());calls=[]
    async def execute(item,stream,plan):calls.append(item)
    obj._webmon_execute_structural=execute
    async def run():
        obj.webmon_start_structural();task=obj._webmon_task
        producer['raw']=producer['raw'].replace(b'-12',b'-13')
        obj.webmon_start_structural()
        assert obj._webmon_task is task
        await task
    asyncio.run(run())
    assert len(calls)==1 and obj._webmon_structural_plan['stream'].revision==1


@pytest.mark.parametrize('change',['transfer','mapping'])
def test_changed_snapshot_settings_cannot_silently_use_the_adopted_plan(tmp_path,change):
    obj,_=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan())
    if change=='transfer':obj.server.state.webmon_refine_transfer=True
    else:obj._webmon_reviewed_mapping=copy.deepcopy(obj._webmon_reviewed_mapping);obj._webmon_reviewed_mapping['evidence']='updated review'
    with pytest.raises(ValueError,match='changed'):
        obj._new_structural_stream()
    assert not obj.server.state.webmon_structural_ready


def test_transfer_edit_pauses_following_without_rewriting_existing_review(tmp_path):
    obj,_=view(tmp_path);asyncio.run(obj.webmon_adopt_structural_plan());plan=obj._webmon_structural_plan
    obj.server.state.webmon_stream_active=True
    obj.webmon_structural_transfer_changed(True)
    assert not obj.server.state.webmon_stream_active and not obj.server.state.webmon_structural_ready
    assert plan['templates'][0]['refine_transfer'] is False
    assert 'existing jobs and results' in obj.server.state.webmon_structural_message


def structural_publication(obj,path):
    from gpx_recipe.symmetry.compiler import file_sha
    obj.server.state.webmon_structural_selected='snapshot:child'
    obj._webmon_structural_results['snapshot:child']=dict(name='Child',revision=2,
        result=dict(result=dict(continuation=dict(path=str(path),sha256=file_sha(path)))))


def test_structural_download_uses_the_real_three_argument_download_contract(tmp_path):
    obj,_=view(tmp_path);path=tmp_path/'point.gpx';path.write_bytes(b'public native point-fit fixture')
    structural_publication(obj,path);downloads=[]
    def download(name,mime,content):downloads.append((name,mime,content))
    obj.download_file=download;obj.webmon_download_structural()
    assert downloads[0][0].endswith('.gpx') and downloads[0][1:] == ('application/octet-stream',path.read_bytes())
    path.write_bytes(b'changed');obj.webmon_download_structural()
    assert len(downloads)==1 and 'changed' in obj.server.state.webmon_structural_message


def test_structural_desktop_keeps_exact_clicked_gpx_and_scope(tmp_path):
    obj,_=view(tmp_path);path=tmp_path/'point.gpx';path.write_bytes(b'public native point-fit fixture')
    structural_publication(obj,path);s=obj.server.state;s.gsasii_session_status='idle'
    uploads=[];actions=[]
    def upload(path,**kwargs):uploads.append(Path(path).read_bytes());return 'uploaded-point'
    obj.service=SimpleNamespace(upload_document=upload)
    async def submit(**kwargs):actions.append(kwargs);return object()
    obj._submit_utility_action=submit
    def schedule(coroutine,_):asyncio.run(coroutine)
    obj._schedule_utility=schedule
    obj.webmon_open_structural_gsasii()
    assert uploads==[b'public native point-fit fixture']
    assert actions[0]['inputs']['project_source|gpx_project']=={'dataset_id':'uploaded-point'}
    assert 'revision 2' in actions[0]['name']
    # The desktop keeps its existing project rather than silently replacing it.
    s.gsasii_session_status='ready';obj.webmon_open_structural_gsasii();assert len(actions)==1


def test_structural_desktop_stale_owner_cannot_submit_a_different_scope(tmp_path):
    obj,_=view(tmp_path);path=tmp_path/'point.gpx';path.write_bytes(b'public fixture')
    structural_publication(obj,path);obj.server.state.gsasii_session_status='idle'
    obj._webmon_scope_current=lambda owner:False
    obj.service=SimpleNamespace(upload_document=lambda *a,**k:pytest.fail('No stale upload'))
    obj._schedule_utility=lambda coroutine,_:asyncio.run(coroutine)
    obj.webmon_open_structural_gsasii()
    assert obj.server.state.gsasii_session_status=='error'
    assert 'scope changed' in obj.server.state.webmon_structural_message
