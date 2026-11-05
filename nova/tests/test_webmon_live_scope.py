"""Acquisition scope is the saved watch, not the newest file in its IPTS."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from radar_pd_nova.refinement_sources import experiment_key
from radar_pd_nova.webmon.scope import WatchScope, ScopeCancellation
from radar_pd_nova.webmon.session import WebmonSession
from radar_pd_nova.webmon.ui import WebmonMixin


RAW=b'# X , Y , E Distribution=true\n0\n1,10,2\n2,20,3\n3,30,4\n'


class State(SimpleNamespace):
    def flush(self):pass


@pytest.fixture
def view(tmp_path):
    folder=tmp_path/'SNS/PG3/IPTS-2767/shared/livereduce'
    folder.mkdir(parents=True)
    summary=folder.parent/'autoreduce/PG3_IPTS-2767_runsummary.csv'
    summary.parent.mkdir()
    summary.write_text('RUN,IPTS#,SAMPLE,ITEMS-ID,TEMP,WAVELENGTH,DURATION\n'
        '7,2767,Fe,100,100,1.5,20\n8,2767,Fe,100,200,1.5,30\n'
        '9,2767,Cu,200,300,1.5,40\n10,2767,Unknown,,300,1.5,40\n')
    for run in (7,8,9,10):(folder/f'PG3_{run}.dat').write_bytes(RAW)
    app=WebmonMixin()
    app.service=SimpleNamespace(galaxy_url='https://example.invalid',galaxy_key='test-key',history_id='history')
    settings=SimpleNamespace(instrument='PG3',history_id='history',ipts='IPTS-2767',
        selected_sample_ids=('100',),selected_banks=(2,),excluded_run_numbers=())
    app._powgen_controller=SimpleNamespace(settings=settings)
    app._result_context=object()
    state=State(live_beamline='PG3',powgen_ipts='2767',powgen_processing_scope='selected',
        powgen_processing_sample_ids=['100'],powgen_bank_scope='selected',powgen_processing_banks=[2],
        webmon_follow_latest=True,webmon_has_result=False,webmon_accepted=False,
        webmon_context_ready=False,webmon_busy=False,webmon_target='',webmon_run='',
        webmon_analysis_revision='',webmon_revision='',webmon_stale=False,
        webmon_published_source='',webmon_source_notice='',webmon_k='0,0,0',webmon_ion='Fe+3',
        webmon_lower=0.,webmon_upper=5.,webmon_mapping_accepted=False,
        webmon_prepare_accepted=False,webmon_refine_transfer=False)
    app.server=SimpleNamespace(state=state)
    app._webmon=WebmonSession(dict(facility_root=str(tmp_path/'SNS'),archive=str(tmp_path/'archive'),
                                 jobs=str(tmp_path/'jobs'),local_replay=False))
    app._webmon_start_scope='2767';app._webmon_display_scope=None
    app._webmon_binding=None;app._webmon_last_preview=None;app._webmon_stream=None
    app._webmon_queue=None;app._webmon_task=None;app._webmon_cancelled=False
    app._webmon_observed=SimpleNamespace(update=lambda figure:None)
    app.webmon_saved_list=lambda:None
    app.summary_path=summary
    return app


def reference(identifier='source',sample='100',bank='2',history='history'):
    return dict(id=identifier,experiment=experiment_key(history,'PG3','IPTS-2767'),
        sample_key='id:'+sample,metadata={'sample_id':sample},bank=bank,
        source_sha256='a'*64,source_path='/unused.gpx',run_number=7,
        scan_label='Reference scan',sample_label=sample)


def test_latest_preview_is_latest_matching_sample_and_never_analyzes(view):
    view.webmon_refresh()
    state=view.server.state
    assert [row['run'] for row in state.webmon_runs]==['8','7']
    assert state.webmon_run==state.webmon_latest=='8'
    assert 'sample 100' in state.webmon_scope_notice
    assert 'physical bank 2' in state.webmon_scope_notice
    assert 'does not identify its physical bank' in state.webmon_scope_notice
    assert not state.webmon_accepted and not state.webmon_context_ready
    assert view._webmon_task is None


@pytest.mark.parametrize('run',['9','10','999'])
def test_manual_unmatched_or_unknown_run_is_rejected_before_capture(view,monkeypatch,run):
    view.webmon_refresh()
    monkeypatch.setattr(view._webmon,'capture',lambda *a:pytest.fail('Do not read an ineligible acquisition'))
    view.webmon_select(run)
    assert view.server.state.webmon_run=='8'
    assert not view.server.state.webmon_accepted and not view.server.state.webmon_scope_allowed


def test_invalid_client_v_model_selection_restores_the_validated_preview_label(view):
    view.webmon_refresh()
    view.server.state.webmon_run='9'  # Browser v-model precedes the handler RPC.
    view.webmon_select('9')
    assert view.server.state.webmon_run=='8'
    assert not view.server.state.webmon_scope_allowed


def test_quiet_refresh_does_not_keep_an_out_of_scope_manual_selection(view):
    view.server.state.webmon_follow_latest=False
    view.server.state.webmon_run='9'
    view.server.state.webmon_accepted=True
    view.webmon_refresh(quiet=True)
    assert not view.server.state.webmon_run and not view.server.state.webmon_accepted
    assert 'outside the saved Live selection' in view.server.state.webmon_message


@pytest.mark.parametrize('kind',['duplicate','foreign','missing','duplicate_column'])
def test_bad_summary_identity_does_not_match_selected_sample(view,kind):
    if kind=='duplicate':text='RUN,IPTS#,SAMPLE,ITEMS-ID\n8,2767,Fe,100\n8,2767,Cu,200\n'
    elif kind=='foreign':text='RUN,IPTS#,SAMPLE,ITEMS-ID\n8,9999,Fe,100\n'
    elif kind=='missing':text='RUN,IPTS#,SAMPLE,ITEMS-ID\n8,2767,Fe,\n'
    else:text='RUN,IPTS#,SAMPLE,ITEMS-ID,ITEMS-ID\n8,2767,Fe,100,100\n'
    view.summary_path.write_text(text)
    view.webmon_refresh()
    assert not view.server.state.webmon_runs and not view.server.state.webmon_run
    assert not view.server.state.webmon_context_ready


def test_new_saved_sample_scope_clears_old_result_but_preserves_artifact(view,tmp_path):
    view.webmon_refresh()
    receipt=tmp_path/'accepted-job.json';receipt.write_text('{"job":"accepted-old"}')
    view.server.state.webmon_has_result=True;view.server.state.webmon_target='8'
    view.server.state.webmon_calculation_status='Completed'
    view.server.state.webmon_validity='Old sample fit assessed'
    view.server.state.webmon_count_summary='5 sampled'
    view.server.state.webmon_fit_metrics=[{'label':'Old sample Rwp','value':'1.0%'}]
    view._webmon_response_cache='old-response-hda'
    view._webmon_pending_binding={'old':'binding'};view._webmon_pending_request={'old':'request'}
    view._webmon_controller_old=view._powgen_controller
    view._powgen_controller=SimpleNamespace(settings=SimpleNamespace(**{
        **vars(view._powgen_controller.settings),'selected_sample_ids':('200',)}))
    view.webmon_refresh()
    assert view.server.state.webmon_run=='9'
    assert not view.server.state.webmon_has_result and not view.server.state.webmon_target
    assert view.server.state.webmon_calculation_status=='Not started'
    assert view.server.state.webmon_validity=='No fit assessed'
    assert view.server.state.webmon_count_summary=='No candidates analyzed'
    assert not view.server.state.webmon_fit_metrics
    assert view._webmon_response_cache is None and view._webmon_pending_binding is None
    assert view._webmon_pending_request is None
    assert receipt.read_text()=='{"job":"accepted-old"}'


@pytest.mark.parametrize('sample,bank,history,metadata',[('200','2','history',None),
    ('100','3','history',None),('100','2','other',None),('','2','history',None),
    ('100','2','history','200')])
def test_published_reference_cannot_cross_sample_bank_or_history(view,sample,bank,history,metadata):
    view.webmon_refresh();source=reference(sample=sample,bank=bank,history=history)
    if metadata is not None:source['metadata']['sample_id']=metadata
    with pytest.raises(ValueError):view._webmon_require_reference(source)


def test_published_source_catalog_filters_and_preserves_exact_identity(view):
    view.webmon_refresh()
    sources=[reference(),reference('other-sample','200'),reference('other-bank',bank='3'),reference('other-history',history='other')]
    workspace=SimpleNamespace(state={'sources':{source['id']:source for source in sources}})
    async def refresh():pass
    view._refresh_refinement_sources=refresh;view._refinement_store=lambda:workspace
    asyncio.run(view.webmon_load_published_sources())
    assert set(view._webmon_published_catalog)=={'source'}
    assert view._webmon_published_catalog['source']==sources[0]


@pytest.mark.parametrize('change',['sample','bank','history','actor','context'])
def test_slow_catalog_cannot_apply_into_changed_parent_scope(view,change):
    view.webmon_refresh();entered=asyncio.Event();release=asyncio.Event()
    async def refresh():entered.set();await release.wait()
    view._refresh_refinement_sources=refresh
    view._refinement_store=lambda:SimpleNamespace(state={'sources':{'source':reference()}})
    async def run():
        task=asyncio.create_task(view.webmon_load_published_sources())
        await entered.wait();mutate(view,change);release.set();await task
    asyncio.run(run())
    assert not getattr(view,'_webmon_published_catalog',{})


def mutate(view,change):
    if change=='sample':view._powgen_controller.settings.selected_sample_ids=('200',)
    elif change=='bank':view._powgen_controller.settings.selected_banks=(3,)
    elif change=='history':view.service.history_id='different-history'
    elif change=='actor':view.service.galaxy_key='different-key'
    else:view._result_context=object()


def test_bad_manual_start_is_rejected_even_if_checkbox_was_set(view):
    view.server.state.webmon_run='9';view.server.state.webmon_accepted=True
    view.server.state.webmon_context_ready=True
    view.webmon_start()
    assert view.server.state.webmon_calculation_status=='Rejected before launch'
    assert view._webmon_task is None


def test_stream_stops_admitting_updates_after_parent_selection_change(view):
    view.webmon_refresh();view._webmon_stream_scope=view._webmon_scope_snapshot()
    view._webmon_stream=SimpleNamespace(capture=lambda:pytest.fail('No out-of-scope capture'))
    class Queue:
        closed=False
        def close(self):self.closed=True
    queue=Queue();view._webmon_queue=queue
    mutate(view,'sample');view.webmon_stream_tick()
    assert queue.closed and view._webmon_stream is None
    assert not view.server.state.webmon_accepted


@pytest.mark.parametrize('change',['sample','bank','history','actor','context'])
def test_preparation_rechecks_scope_after_reference_resolution(view,monkeypatch,change):
    view.webmon_refresh();row,receipt,_=view._webmon.capture('2767','8')
    source=reference();view._webmon_published_catalog={'source':source}
    view.server.state.webmon_published_source='source'
    view.server.state.webmon_prepare_accepted=True;view.server.state.sx_kind='magnetic'
    view._symmetry_draft={'source_id':'source','discovery_identity':'reviewed'}
    view._discovery_identity=lambda:'reviewed';view._model_form=lambda:{}
    view._webmon_reviewed_mapping={'source_id':'source','snapshot_sha256':receipt['content_sha256']}
    entered,release=threading.Event(),threading.Event()
    def resolve(source):entered.set();assert release.wait(5);return source
    view._refinement_store=lambda:SimpleNamespace(runtime=SimpleNamespace(resolve=resolve))
    monkeypatch.setattr('radar_pd_nova.webmon.galaxy_runtime.GalaxyLiveRuntime',
        lambda *a,**k:pytest.fail('No stale preparation dispatch'))
    async def run():
        task=asyncio.create_task(view.webmon_prepare_from_models())
        assert await asyncio.to_thread(entered.wait,5)
        mutate(view,change);release.set();await task
    asyncio.run(run())
    assert not view.server.state.webmon_busy


def test_scope_cancellation_uses_existing_admission_contract():
    current=[True];event=ScopeCancellation(lambda:current[0])
    assert not event.is_set()
    current[0]=False
    assert event.is_set() and event.wait(0)
    current[0]=True
    assert event.is_set(), 'An observed withdrawn scope must not become authorized again'


def test_matching_reviewed_start_queues_once_without_automatic_extra_work(view):
    view.webmon_refresh()
    choice={'refinement_source':reference()}
    view._webmon.selected_checkpoint=lambda *a:choice
    view._webmon.request=lambda *a,**k:({}, {})
    view.server.state.webmon_accepted=True
    seen=[]
    async def run():
        done=asyncio.Event()
        async def execute(launch):seen.append(launch);await done.wait()
        view._webmon_execute=execute
        view.webmon_start();task=view._webmon_task
        view.webmon_start()
        await asyncio.sleep(0)
        assert len(seen)==1 and seen[0]['webmon_run']=='8'
        assert seen[0]['scope_owner']['scope'].sample_ids==('100',)
        done.set();await task
    asyncio.run(run())


def test_explicit_bank_metadata_is_checked_without_inventing_missing_bank():
    scope=WatchScope('history','IPTS-2767',('100',),(2,))
    row={'run':'8','sample_id':'100','identity_status':'resolved'}
    assert not scope.row_error(row)
    assert scope.row_error({**row,'physical_bank':3})


def test_all_sample_watch_still_requires_reference_and_target_to_match():
    scope=WatchScope('history','IPTS-2767',None,(2,))
    row={'run':'9','sample_id':'200','identity_status':'resolved'}
    assert not scope.row_error(row)
    assert 'different sample IDs' in scope.source_error(reference(sample='100'),row)
    assert not scope.source_error(reference(sample='200'),row)


def test_beamline_switch_revokes_scope_even_when_old_controller_remains(view):
    view.webmon_refresh();owner=view._webmon_scope_snapshot()
    view.server.state.live_beamline='NOM'
    assert not view._webmon_scope_current(owner)
    view.webmon_scope_changed()
    assert not view.server.state.webmon_run and not view.server.state.webmon_scope_allowed


def test_changing_scope_away_and_back_does_not_revive_old_permission(view):
    view.webmon_refresh();owner=view._webmon_scope_snapshot()
    view.server.state.webmon_accepted=True
    view.server.state.powgen_processing_sample_ids=['200'];view.webmon_scope_changed()
    view.server.state.powgen_processing_sample_ids=['100'];view.webmon_scope_changed()
    assert not view._webmon_scope_current(owner)
    assert not view.server.state.webmon_accepted and not view.server.state.webmon_context_ready


@pytest.mark.parametrize('sample',['100','200'])
def test_posterior_handoff_keeps_exact_output_and_sample_owner(view,sample):
    view.webmon_refresh();state=view.server.state
    state.webmon_model='posterior';state.webmon_target='8'
    state.webmon_rows=[{'posterior':'posterior','fitted_dataset_id':'exact-posterior-hda'}]
    view._webmon_request={'reference_revision':'PG3_7:'+'a'*64,'source_revision':'PG3_8:'+'b'*64}
    view._webmon_plot_binding={'source_scope':{'run':'8','sample_id':'100','sample':'Fe'}}
    view._webmon_post=lambda _:dict(fitted_gpx='/verified/posterior.gpx',gpx_export={'fitted_gpx_sha256':'c'*64},rwp=5.)
    source=reference(sample=sample);source.update(dataset_id='parent-hda',collection_id='parent-hdca')
    choice={'source_sha256':'a'*64,'refinement_source':source}
    view._webmon.selected_checkpoint=lambda *a:choice
    selected={};view._webmon_handoff_source=lambda result,prompt:selected.update(result)
    view.webmon_open_posterior_refinement()
    if sample=='100':
        assert not selected
        assert 'nuclear projects only' in state.webmon_message
    else:
        assert not selected and 'outside' in state.webmon_message


def test_scope_revocation_after_packet_upload_makes_no_scientific_post(tmp_path):
    from radar_pd_nova.webmon.galaxy_runtime import GalaxyLiveRuntime
    from radar_pd_nova.refinement_runtime import RuntimeUnavailable
    from test_webmon_galaxy_runtime import Service
    from test_webmon_hardening import scientific_request
    src=tmp_path/'source';src.mkdir();request,_=scientific_request(src)
    allowed=[True];event=ScopeCancellation(lambda:allowed[0])
    service=Service(None)
    def upload(*args,**kwargs):allowed[0]=False;return 'retained-packet-hda'
    service.upload_document=upload
    runtime=GalaxyLiveRuntime(tmp_path/'runtime',service)
    with pytest.raises(RuntimeUnavailable,match='no new job'):
        runtime.call(request,tmp_path/'call',dispatch_key='e'*24,cancel_event=event)
    assert not service.submissions


def test_accepted_dispatch_survives_scope_loss_and_recovers_without_resubmit(tmp_path):
    from radar_pd_nova.webmon.galaxy_runtime import GalaxyLiveRuntime
    from radar_pd_nova.refinement_runtime import RuntimeUnavailable
    from radar_pd_nova.models import RunStatus
    from test_webmon_galaxy_runtime import Service, bundle_for
    from test_webmon_hardening import scientific_request
    src=tmp_path/'source';src.mkdir();request,_=scientific_request(src)
    allowed=[True];event=ScopeCancellation(lambda:allowed[0])
    service=Service(bundle_for(tmp_path,request));original=service.submit_utility
    def submit(**kwargs):
        job=original(**kwargs);job.status=RunStatus.QUEUED;allowed[0]=False;return job
    service.submit_utility=submit
    runtime=GalaxyLiveRuntime(tmp_path/'runtime',service);runtime.poll_seconds=0
    with pytest.raises(RuntimeUnavailable,match='recoverable'):
        runtime.call(request,tmp_path/'first',dispatch_key='f'*24,cancel_event=event)
    receipt=tmp_path/'runtime/dispatches'/('f'*24+'.json')
    before=receipt.read_bytes();assert len(service.submissions)==1
    allowed[0]=True
    service.refresh_utility=lambda job:setattr(job,'status',RunStatus.OK)
    with pytest.raises(RuntimeUnavailable,match='no longer active'):
        runtime.call(request,tmp_path/'withdrawn',dispatch_key='f'*24,cancel_event=event)
    # Explicitly renewed review gets a fresh lease; an old cancelled event never revives.
    renewed=ScopeCancellation(lambda:allowed[0])
    result=runtime.call(request,tmp_path/'recovery',dispatch_key='f'*24,cancel_event=renewed)
    assert result['galaxy_job_id']=='galaxy-job' and len(service.submissions)==1
    assert receipt.read_bytes()==before


@pytest.mark.parametrize('change',['sample','bank','actor','context'])
def test_accepted_preparation_is_retained_but_never_adopted_in_new_scope(view,tmp_path,monkeypatch,change):
    view.webmon_refresh();row,receipt,_=view._webmon.capture('2767','8')
    source=reference();view._webmon_published_catalog={'source':source}
    s=view.server.state
    s.webmon_published_source='source';s.webmon_prepare_accepted=True;s.sx_kind='magnetic'
    s.sx_candidates=['candidate'];s.sx_sites='';s.sx_form_factors='';s.sx_lande=''
    s.sx_parent='phase';s.sx_k='0,0,0';s.sx_max_moment=5.;s.sx_slots=1
    view._symmetry_draft={'source_id':'source','discovery_identity':'reviewed',
                         'catalog':{'candidates':[{'id':'candidate'}]}}
    view._discovery_identity=lambda:'reviewed';view._model_form=lambda:{}
    view._webmon_reviewed_mapping={'source_id':'source','snapshot_sha256':receipt['content_sha256'],
                                  'evidence':'Explicit producer review'}
    view._refinement_store=lambda:SimpleNamespace(runtime=SimpleNamespace(resolve=lambda value:value))
    monkeypatch.setattr('gpx_recipe.symmetry.chemistry.sites_from_inputs',
                        lambda *a:[{'label':'Fe1','form_factor':'Fe+3'}])
    entered,release=threading.Event(),threading.Event();saved=tmp_path/'accepted.json'
    class Runtime:
        def __init__(self,*args):pass
        def call(self,*args,**kwargs):
            saved.write_text('{"job":"accepted-before-scope-change"}')
            entered.set();assert release.wait(5)
            return {'preparation':{},'directory':str(tmp_path)}
    monkeypatch.setattr('radar_pd_nova.webmon.galaxy_runtime.GalaxyLiveRuntime',Runtime)
    async def run():
        task=asyncio.create_task(view.webmon_prepare_from_models())
        assert await asyncio.to_thread(entered.wait,5)
        mutate(view,change);release.set();await task
    asyncio.run(run())
    assert saved.read_text()=='{"job":"accepted-before-scope-change"}'
    assert not view._webmon.config.get('checkpoint_options')
    assert not s.webmon_context_ready
