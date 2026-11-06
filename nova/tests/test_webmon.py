import json
from pathlib import Path
import numpy as np
import pytest
from radar_pd_nova.webmon.statistics import QuadraticObservation,sample_posterior
from radar_pd_nova.webmon.snapshot import parse_live_ascii,SnapshotArchive,density_to_tof
from radar_pd_nova.webmon.session import WebmonSession


def test_reduced_gaussian_compression_matches_full_signed_likelihood():
    rng=np.random.default_rng(11);q=rng.normal(size=(3,3,101));q=(q+q.transpose(1,0,2))/2
    y=rng.normal(size=101);sigma=rng.uniform(.1,3,101);base=rng.normal(size=101);forms=np.eye(3)[None]
    model=QuadraticObservation(base,q,y,sigma,forms)
    p=rng.normal(size=(80,3));direct=-.5*np.sum(((model.predict(p)-y)/sigma)**2,axis=1)
    np.testing.assert_allclose(model.loglike(p),direct,rtol=1e-12,atol=1e-9)
    np.testing.assert_allclose(model.predict(p),model.predict(-p))


def test_smc_moment_interval_against_dense_quadrature():
    m=QuadraticObservation(np.zeros(1),np.ones((1,1,1)),np.array([4.]),np.array([.6]),np.ones((1,1,1)),0,4)
    x=np.linspace(-4,4,100001);w=np.exp(m.loglike(x[:,None])-m.loglike([[2]])[0]);order=np.argsort(abs(x));cdf=np.cumsum(w[order]);cdf/=cdf[-1]
    exact=np.interp([.025,.5,.975],cdf,abs(x)[order])
    result=sample_posterior(m,particles=2048,seed=81,max_seconds=30)
    np.testing.assert_allclose(np.quantile(abs(result['samples'][:,0]),[.025,.5,.975]),exact,atol=.07)
    assert result['history'][-1]['beta']==1


def test_duplicate_capture_and_unknown_exposure(tmp_path):
    p=tmp_path/'PG3_12.dat';p.write_bytes(b'# X , Y , E Distribution=true\n0\n1,-1,2\n2,0,0\n3,4,1\n')
    a=SnapshotArchive(tmp_path/'archive');kwargs=dict(ipts='1',run='12',epoch='one',metadata={})
    first=a.capture(p,**kwargs);again=a.capture(p,**kwargs)
    assert again['duplicate'] and first['revision_id']==again['revision_id']
    assert not first['admission']['cumulative_exposure_known']
    assert first['signed_points']==1 and first['masked_points']==1
    p.write_bytes(p.read_bytes()[:-1])
    with pytest.raises(ValueError,match='Incomplete'):a.capture(p,**kwargs)


def test_exact_density_integrals_and_masks():
    p=parse_live_ascii(b'# X , Y , E Distribution=true\n0\n1,-1,2\n2,0,0\n3,4,1\n')
    t=density_to_tof(p,d_edges=[.5,1.5,2.5,3.5],export_difc=100)
    np.testing.assert_allclose(t['y']*np.diff(t['edges']),p.y)
    assert t['weights'][1]==0


def test_scope_and_other_run_not_inherits_checkpoint(tmp_path):
    manifest=tmp_path/'context.json';manifest.write_text(json.dumps(dict(facility_root=str(tmp_path/'inputs'),archive=str(tmp_path/'archive'))))
    s=WebmonSession(manifest)
    with pytest.raises(ValueError):s.scope('../../2')
    with pytest.raises(ValueError,match='No reviewed'):s.binding('1','3')


def test_inventory_cache_detaches_rows_and_detects_new_runs(tmp_path,monkeypatch):
    root=tmp_path/'SNS';folder=root/'PG3/IPTS-1/shared/livereduce';folder.mkdir(parents=True)
    (folder/'PG3_1.dat').write_text('snapshot')
    session=WebmonSession(dict(facility_root=str(root),archive=str(tmp_path/'archive')))
    original=Path.glob;calls=[]
    def glob(path,pattern):
        calls.append((path,pattern));return original(path,pattern)
    monkeypatch.setattr(Path,'glob',glob)
    first=session.runs('1');first[0]['sample']='changed by caller'
    assert session.runs('1')[0]['sample']=='Unknown'
    assert len(calls)==1
    (folder/'PG3_2.dat').write_text('next snapshot')
    assert [r['run'] for r in session.runs('1')]==['2','1']
    assert len(calls)==2


def test_per_site_bound_box_does_not_remove_valid_points():
    forms=np.array([[[1.,0],[0,0]],[[0,0],[0,1.]]]);m=QuadraticObservation(np.zeros(3),np.zeros((2,2,3)),np.zeros(3),np.ones(3),forms,5,7)
    p=m.prior(np.random.default_rng(2),1000);assert m.valid(p).all()
    assert (np.max(abs(p),axis=0)>6.9).all()


def test_many_independent_orbits_can_sample_tight_bounds():
    n=24;forms=np.zeros((8,n,n))
    for j in range(8):forms[j,3*j:3*j+3,3*j:3*j+3]=np.eye(3)
    m=QuadraticObservation(np.zeros(2),np.zeros((n,n,2)),np.zeros(2),np.ones(2),forms,5,7)
    draws=m.prior(np.random.default_rng(100),256)
    assert m.valid(draws).all()
    assert (np.std(m.moments(draws),axis=0)>.4).all()


def test_start_dispatches_background_task_and_ignores_double_click():
    import asyncio
    from types import SimpleNamespace
    from radar_pd_nova.webmon.ui import WebmonMixin
    async def scenario():
        done=asyncio.Event();calls=[]
        async def execute(launch):calls.append(launch['webmon_run']);await done.wait()
        obj=WebmonMixin();obj.server=SimpleNamespace(state=SimpleNamespace(webmon_busy=False,flush=lambda:None,powgen_ipts='1',webmon_run='2',webmon_k='0,0,0',webmon_ion='Er+3',webmon_lower=5,webmon_upper=7,webmon_accepted=True));obj._webmon_execute=execute
        assert obj.webmon_start() is None
        task=obj._webmon_task;obj.webmon_start();assert obj._webmon_task is task
        obj.server.state.webmon_run='3'
        await asyncio.sleep(0);assert calls==['2'] and not task.done()
        done.set();await task
    asyncio.run(scenario())


def test_worker_identity_is_bound_at_import():
    from radar_pd_nova.webmon.worker import code_identity
    identity=code_identity()
    assert 'compile_candidate' in identity and 'infer' in identity
    assert all(v=='unavailable' or len(v)==64 for v in identity.values())
    identity['infer']='changed'
    assert code_identity()['infer']!='changed'


def test_worker_cache_identity_detects_a_transitive_scientific_helper_change(tmp_path,monkeypatch):
    from gpx_recipe.live import worker
    from gpx_recipe.symmetry import compiler
    root=tmp_path/'gpx_recipe/symmetry';root.mkdir(parents=True)
    entry=root/'compiler.py';entry.write_text('# isolated source inventory\n')
    helper=root/'displacements.py';helper.write_text('# earlier construction semantics\n')
    monkeypatch.setattr(compiler,'__file__',str(entry))
    before=worker._read_code_identity()
    helper.write_text('# revised construction semantics\n')
    after=worker._read_code_identity()
    assert before['gpx_recipe_package_sha256']!=after['gpx_recipe_package_sha256']
    assert {k:v for k,v in before.items() if k!='gpx_recipe_package_sha256'}=={
        k:v for k,v in after.items() if k!='gpx_recipe_package_sha256'}
