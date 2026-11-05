from types import SimpleNamespace
import numpy as np
import pytest
from gpx_recipe.structural_live.observations import Acquisition,CumulativeCounts
from gpx_recipe.structural_live.optimization.fit_fast_native import residual_and_factor
from gpx_recipe.structural_live.optimization.chemistry_sqp import contacts
from gpx_recipe.structural_live.optimization.chemistry_sqp_native_fit import fit

def fixture():
    x=np.linspace(0,1,60);b=np.array([np.exp(-((x-v)/.07)**2) for v in [.2,.45,.7]])
    background=np.full(60,.002);base=background+.008+.004*b[1]
    geometry=SimpleNamespace(baseline=np.eye(3),cell_directions=np.array([np.zeros((3,3)),np.diag([2.,0.,0.])]),
        delta=np.array([[1.,0,0]]),directions=np.array([[[1.,0],[0,0],[0,0]]]),threshold=np.array([1.]),
        offsets=[0,1,2],radii=[1.,1.])
    def kernel(points):
        return np.array([0. if 1+2*p[1]>0 and np.all(contacts(geometry,p,np.ones(2))[0]>=-1e-14) else -np.inf for p in points])
    geometry.log_kernel=kernel;called=[]
    def native(p):
        assert np.isfinite(kernel(np.array(p)[None,:])[0])
        called.append(np.array(p));return base+.002*p[0]*b[0]+.002*p[1]*b[2]
    source=SimpleNamespace(radii=np.ones(2),mode_dimensions=1,cell_dimensions=1,chemistry=geometry,native=native)
    target=native([-.1,-.2]);counts=np.random.default_rng(2221).poisson(target*1e8)
    acquisition=Acquisition('sample','run','instrument','config','run','hash','q_inverse_angstrom','counts','unit')
    obs=CumulativeCounts.from_counts(acquisition,'sqp',0,1e8,x,counts,1.,np.ones(60,bool))
    table=SimpleNamespace(renderer=SimpleNamespace(background=background),background_reference=float(background[0]),fallback=True)
    bank=SimpleNamespace(proposal_cell_steps=np.array([.002,.002]))
    cache=SimpleNamespace(evaluate=native,mode_jacobian=lambda p:(native(p),(.002*b[0])[None,:]))
    return source,table,obs,bank,cache,called

def test_native_constrained_fit_can_leave_boundary_with_coupled_motion():
    s,t,o,b,c,called=fixture();result,rate=fit(s,t,o,b,np.zeros(4),seconds=3,evaluator=c)
    residual,_=residual_and_factor(o,rate)
    assert result['best']['deviance']==pytest.approx(residual@residual)
    z=np.array(result['best']['parameters'])
    assert z[0]<-.08 and z[1]<-.16
    assert np.all(np.diff([r['deviance'] for r in result['trace']])<0)
    assert np.isfinite(s.chemistry.log_kernel(np.array(called))).all()
    assert result['all_allowed_modes_retained'] and not result['model_exclusion_authorized']

def test_native_sqp_cache_error_is_fatal():
    s,t,o,b,c,_=fixture();old=c.mode_jacobian
    c.mode_jacobian=lambda p:(old(p)[0]+.001*np.sin(o.x*23),old(p)[1])
    with pytest.raises(ValueError,match='cache mismatch'):fit(s,t,o,b,np.zeros(4),seconds=2,evaluator=c)
