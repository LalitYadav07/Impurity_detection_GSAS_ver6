from pathlib import Path
import json
import os
import sys
import numpy as np
import pytest

from gpx_recipe.structural_live.optimization.refinable_gpx import origin_reference_modes


@pytest.mark.parametrize('n,g', [(0,0),(2,0),(6,1),(12,3),(30,3)])
def test_origin_reference_keeps_entire_physical_quotient(n,g):
    rng=np.random.default_rng(142)
    gauge=np.linalg.qr(rng.normal(size=(n,g)))[0] if g else np.empty((n,0))
    held=origin_reference_modes(gauge)
    free=[i for i in range(n) if i not in held]
    assert len(held)==g
    projector=np.eye(n)-gauge@gauge.T
    allowed=np.eye(n)[:,free]
    assert np.linalg.matrix_rank(projector@allowed,tol=1e-9)==n-g if n else len(held)==0
    # Any desired change can be represented using the free modes plus only
    # an origin translation, while keeping held modes at their starting value.
    delta=rng.normal(size=n)
    gauge_shift=np.linalg.solve(gauge[held],-delta[held]) if g else np.empty(0)
    represented=delta+gauge@gauge_shift
    np.testing.assert_allclose(represented[held],0.,atol=1e-12)
    np.testing.assert_allclose(projector@represented,projector@delta,atol=1e-12)


@pytest.mark.parametrize('bad', [np.ones(3),np.array([[np.nan]]),np.ones((2,3)),np.ones((3,2))])
def test_invalid_origin_gauge_rejected(bad):
    with pytest.raises(ValueError):origin_reference_modes(bad)


def test_continuation_refuses_implicit_phase_or_bank_selection():
    from types import SimpleNamespace
    from gpx_recipe.structural_live.optimization.refinable_gpx import continue_native_refinement
    with pytest.raises(ValueError,match='transforming phase'):
        continue_native_refinement(SimpleNamespace(phases=lambda:[None,None]),[])
    with pytest.raises(ValueError,match='histogram'):
        continue_native_refinement(SimpleNamespace(phases=lambda:[None],histograms=lambda:[None,None]),[])


@pytest.fixture
def native_fixture():
    value=os.environ.get('RADAR_NATIVE_EXPORT_FIXTURE')
    if not value:pytest.skip('Requires the registered native two-mode benchmark fixture')
    return Path(value)


def test_native_export_refines_perturbed_modes_and_saves_covariance(tmp_path,native_fixture):
    from gpx_recipe.structural_live.native import modules,apply_modes,sha
    from gpx_recipe.structural_live.optimization.refinable_gpx import export_refinable,mode_amplitudes,continue_native_refinement
    g,m,*_=modules();source_hash=sha(native_fixture)
    p=g.G2Project(gpxfile=str(native_fixture),newgpx=str(tmp_path/'ideal.gpx'))
    h=p.histograms()[0];h.data['data'][1][1]=np.array(h.data['data'][1][3],copy=True)
    p.data['Controls']['data']['max cyc']=0;p.save()
    result=export_refinable(p.filename,tmp_path/'export')
    p=g.G2Project(gpxfile=result['path'],newgpx=str(tmp_path/'perturbed.gpx'))
    q=mode_amplitudes(p.phases()[0].data);assert len(q)==2
    apply_modes(p.phases()[0].data,q+np.array([.01,-.01]),m)
    result2=continue_native_refinement(p,result['contract']['native_active_variables'])
    assert result2['objective_verified'] and result2['chemistry_feasible']
    np.testing.assert_allclose(mode_amplitudes(p.phases()[0].data),q,atol=2e-6,rtol=0)
    assert len(p.data['Covariance']['data']['varyList'])==8
    assert sha(native_fixture)==source_hash


def test_background_holds_rebind_across_projects(tmp_path,native_fixture):
    other=os.environ.get('RADAR_NATIVE_EXPORT_SECOND_FIXTURE')
    if not other:pytest.skip('Requires a second histogram identity')
    from gpx_recipe.structural_live.native import modules
    from gpx_recipe.structural_live.optimization.refinable_gpx import configure
    g,*_=modules()
    for i,path in enumerate([native_fixture,Path(other),native_fixture]):
        p=g.G2Project(gpxfile=str(path),newgpx=str(tmp_path/f'project-{i}.gpx'))
        c=configure(p)
        assert ':0:Back;0' in c['native_active_variables']
        assert not any(v in c['native_active_variables'] for v in [':0:Back;1',':0:Back;2'])
        # The same project can be exported again without duplicate holds.
        configure(p)
        assert len(p.data['Constraints']['data']['Hist'])==2
