import asyncio,json,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from radar_pd_nova.webmon.stream import LatestQueue,ObservationStream,ReviewRequired
from radar_pd_nova.webmon.observations import replace_observations
from radar_pd_nova.webmon.contracts import file_hash,validate_request
from radar_pd_nova.webmon.session import WebmonSession
from test_webmon_hardening import scientific_request


def fixture(tmp_path):
    r,b=scientific_request(tmp_path)
    b.update(reference_gpx=b['source_gpx'],reference_sha256=b['source_sha256'])
    r['reference_revision']='PG3_2:'+b['reference_sha256']
    root=tmp_path/'PG3/IPTS-1/shared/livereduce';root.mkdir(parents=True)
    p=root/'PG3_2.dat';p.write_bytes(b'# X , Y , E Distribution=true\n0\n1,10,2\n2,20,3\n3,30,4\n')
    b.update(source_dat=str(p),source_dat_sha256=file_hash(p),display_export_difc=100.)
    r['source_revision']='PG3_2:'+b['source_dat_sha256']
    template=tmp_path/'review.json';template.write_text(json.dumps(r));b['request']=str(template)
    m=tmp_path/'context.json';m.write_text(json.dumps(dict(facility_root=str(tmp_path),archive=str(tmp_path/'archive'))))
    stream=ObservationStream(WebmonSession(m),'1','2',b,r,tmp_path/'stream')
    return stream,p


def test_revision_replaces_cumulative_likelihood_and_preserves_response(tmp_path):
    stream,p=fixture(tmp_path);first=stream.capture();assert stream.capture() is None
    k=tmp_path/'kernel.npz';np.savez_compressed(k,x=[100.,200.],y=[0.,0.],sigma=[1.,1.],base=[5.,6.],quadratic=np.ones((1,1,2)))
    replace_observations(k,first['request'])
    p.write_bytes(p.read_bytes().replace(b'1,10,2',b'1,12,1'))
    second=stream.capture();replace_observations(k,second['request'])
    with np.load(k) as a:
        np.testing.assert_allclose(a['y'],[.12,.2]);np.testing.assert_allclose(a['base'],[5.,6.]);np.testing.assert_allclose(a['sigma'],[.01,.03])
    assert first['request']['source_sha256']==second['request']['source_sha256']
    assert first['request']['source_revision']!=second['request']['source_revision']
    bad=copy.deepcopy(second['request']);bad['cumulative_policy']='add'
    with pytest.raises(ValueError,match='replace'):validate_request(bad,second['binding'])


def test_truncated_snapshot_retries_but_grid_change_needs_review(tmp_path):
    s,p=fixture(tmp_path);s.capture();raw=p.read_bytes();p.write_bytes(raw[:-1])
    with pytest.raises(ValueError,match='Incomplete'):s.capture()
    p.write_bytes(raw);assert s.capture() is None
    p.write_bytes(raw.replace(b'3,30,4',b'4,30,4'))
    with pytest.raises(ReviewRequired):s.capture()


def test_configuration_change_does_not_reuse_response(tmp_path):
    s,p=fixture(tmp_path);s.capture()
    folder=p.parent.parent/'autoreduce';folder.mkdir()
    (folder/'PG3_IPTS-1_runsummary.csv').write_text('RUN,SAMPLE,WAVELENGTH\n2,New sample,2.665\n')
    with pytest.raises(ReviewRequired):s.capture()


def test_latest_queue_coalesces_without_overlapping_jobs(tmp_path):
    async def scenario():
        q=LatestQueue(tmp_path/'q.json');q.offer(dict(id='a'));entered=asyncio.Event();release=asyncio.Event();seen=[]
        async def execute(item):
            seen.append(item['id'])
            if item['id']=='a':entered.set();await release.wait()
        task=asyncio.create_task(q.drain(execute));await entered.wait()
        assert not q.offer(dict(id='a'))
        q.offer(dict(id='b'));q.offer(dict(id='c'));await q.drain(execute)
        release.set();await task
        assert seen==['a','c'];assert q.state['coalesced']==1
        assert q.state['completed']==seen
    asyncio.run(scenario())


def test_restart_does_not_automatically_resume_and_user_queues_are_isolated(tmp_path):
    a=LatestQueue(tmp_path/'user-a/q.json');b=LatestQueue(tmp_path/'user-b/q.json')
    a.offer(dict(id='x'));b.offer(dict(id='y'))
    restarted=LatestQueue(a.path)
    assert restarted.state['interrupted_previous'];assert restarted.pending is None
    assert b.pending['id']=='y'


def test_failed_worker_stops_automatic_retry(tmp_path):
    async def scenario():
        q=LatestQueue(tmp_path/'q.json');q.offer(dict(id='x'))
        async def fail(_):raise RuntimeError('crash')
        with pytest.raises(RuntimeError):await q.drain(fail)
        assert q.closed and q.state['failed'][0]['id']=='x'
    asyncio.run(scenario())


def test_global_native_capacity_is_bounded_and_released(tmp_path):
    from radar_pd_nova.webmon.resources import acquire
    async def scenario():
        first=await acquire(tmp_path,1,.1)
        with pytest.raises(TimeoutError):await acquire(tmp_path,1,.01)
        first.release();second=await acquire(tmp_path,1,.1);second.release()
    asyncio.run(scenario())


def test_restore_stream_requires_current_snapshot_and_reviewed_reference(tmp_path):
    stream,p=fixture(tmp_path);first=stream.capture()
    stream.binding['reference_gpx']=stream.binding['source_gpx']
    choice=dict(id='review',source_path=stream.binding['reference_gpx'],source_sha256=stream.binding['reference_sha256'],binding=stream.binding)
    stream.session.config['checkpoint_options']={'IPTS-1/2':[choice]}
    restored=stream.session.restore_binding('1','2',first['request'])
    assert restored['observation_sha256']==first['request']['observation_sha256']
    p.write_bytes(p.read_bytes().replace(b'1,10,2',b'1,12,1'))
    with pytest.raises(ValueError,match='current compatible'):
        stream.session.restore_binding('1','2',first['request'])
    second=stream.capture()
    assert stream.session.restore_binding('1','2',second['request'])['source_dat_sha256']==file_hash(p)
    Path(choice['source_path']).write_bytes(b'changed gpx')
    with pytest.raises(ValueError,match='changed'):
        stream.session.restore_binding('1','2',second['request'])


def test_profile_width_diagnostic_known_gaussian_and_missing_crossing():
    from radar_pd_nova.webmon.profile_diagnostics import half_height_width
    x=np.linspace(2.9,3.1,1001);sigma=.01;y=3+20*np.exp(-.5*((x-3)/sigma)**2)
    assert half_height_width(x,y,3)['fwhm_A']==pytest.approx(2.35482*sigma,rel=.001)
    assert half_height_width(x[:10],y[:10],3) is None
    assert half_height_width(x,np.ones_like(x),3) is None


def test_pause_during_execution_stays_stopped(tmp_path):
    async def scenario():
        q=LatestQueue(tmp_path/'q.json');q.offer(dict(id='a'))
        async def execute(item):q.close()
        await q.drain(execute)
        assert q.closed and q.state['status']=='stopped'
        with pytest.raises(RuntimeError):q.offer(dict(id='b'))
    asyncio.run(scenario())


def test_idle_stream_reconciles_displayed_and_analyzed_revisions(tmp_path):
    from radar_pd_nova.webmon.ui import WebmonMixin
    obj=WebmonMixin();obj._webmon_stream=SimpleNamespace(ipts='1',run='2',capture=lambda:None)
    obj._webmon_queue=LatestQueue(tmp_path/'q.json');obj._webmon_task=None
    s=SimpleNamespace(powgen_ipts='1',webmon_run='2',webmon_has_result=True,webmon_revision='a',webmon_analysis_revision='a',webmon_stale=True,flush=lambda:None)
    obj.server=SimpleNamespace(state=s);obj.webmon_stream_tick()
    assert s.webmon_stale is False
    s.webmon_run='3';obj.webmon_stream_tick();assert s.webmon_stale is True
    s.webmon_run='2';s.webmon_revision='b';obj.webmon_stream_tick();assert s.webmon_stale is True


def test_archived_revision_stays_reviewable_after_live_source_changes(tmp_path):
    stream,p=fixture(tmp_path);first=stream.capture()
    stream.binding['reference_gpx']=stream.binding['source_gpx']
    choice=dict(id='review',source_path=stream.binding['reference_gpx'],source_sha256=stream.binding['reference_sha256'],binding=stream.binding)
    stream.session.config['checkpoint_options']={'IPTS-1/2':[choice]}
    original=p.read_bytes();p.write_bytes(original.replace(b'1,10,2',b'1,19,2'))
    # Historical review checks its immutable archived observations. It neither
    # treats them as the current scan nor authorizes a new launch from them.
    restored=stream.session.restore_binding('1','2',first['request'],allow_archived=True)
    assert restored['source_dat_sha256']==first['request']['observation_source_sha256']
    with pytest.raises(ValueError,match='current compatible snapshot'):
        stream.session.restore_binding('1','2',first['request'])


def test_production_cannot_execute_local_native_commands():
    from radar_pd_nova.webmon.ui import WebmonMixin
    obj=WebmonMixin();obj._webmon=SimpleNamespace(config={'local_replay':False,'worker_command':['python','native.py']})
    with pytest.raises(ValueError,match='Galaxy transport'):
        obj._webmon_local_command('worker_command')
    obj._webmon.config['local_replay']=True
    assert obj._webmon_local_command('worker_command')==['python','native.py']


def test_normal_monitor_attaches_snapshot_polling_without_launching_analysis():
    from radar_pd_nova.webmon.ui import WebmonMixin
    async def scenario():
        obj=WebmonMixin();obj._webmon=SimpleNamespace(config={});obj._webmon_poll_task=None
        obj.server=SimpleNamespace(state=SimpleNamespace(live_beamline='PG3',powgen_ipts='123',powgen_monitoring=True))
        refreshed=[];obj.webmon_refresh=lambda **kw:refreshed.append(kw)
        stopped=asyncio.Event()
        async def poll():
            await stopped.wait()
        obj._webmon_poll=poll
        obj.webmon_attach_monitor();task=obj._webmon_poll_task
        obj.webmon_attach_monitor()
        assert obj._webmon_poll_task is task
        assert refreshed==[{'quiet':True},{'quiet':True}]
        stopped.set();await task
    asyncio.run(scenario())


def test_preparing_new_revision_retains_last_verified_result(tmp_path):
    from radar_pd_nova.webmon.ui import WebmonMixin
    import hashlib
    out=tmp_path/'next';out.mkdir()
    request=dict(source_revision='PG3_2:new',reference_revision='PG3_1:ref')
    (out/'request.json').write_text(json.dumps(request))
    (out/'result.json').write_text(json.dumps(dict(request_sha256=file_hash(out/'request.json'),**request,
        status='running',rows=[dict(id='next',name='Next',bns='1',status='preparing')])) )
    obj=WebmonMixin();obj._webmon_out=tmp_path/'previous'
    state=SimpleNamespace(webmon_has_result=True,webmon_model='previous-posterior',webmon_rows=['previous'])
    obj.server=SimpleNamespace(state=state)
    obj._webmon_load_result(out)
    assert obj._webmon_out==tmp_path/'previous'
    assert state.webmon_model=='previous-posterior'
    assert state.webmon_rows==['previous']
    assert state.webmon_pending_rows[0]['status']=='preparing'
