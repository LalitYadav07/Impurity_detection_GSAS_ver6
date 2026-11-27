"""Small float64 oracles for scale-conditioned coefficient construction.

These are algebraic controls, not substitutes for the separate GSAS-II tests.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint


def oracle(scale, *, masked=False, fixed=True):
    rng=np.random.default_rng(570241);n=2301;dimension=12
    reference=5907124.940466692
    factors=rng.normal(size=(n,3,dimension))*1e-4
    physical_q=reference*np.einsum('nki,nkj->nij',factors,factors)
    physical_n=reference*rng.uniform(2e-6,6e-6,n)
    truth=rng.normal(size=dimension)*.45
    observed=physical_n+np.einsum('i,nij,j->n',truth,physical_q,truth)+rng.normal(0,.03,n)
    obj=NativeMagneticStartingPoint.__new__(NativeMagneticStartingPoint)
    obj.source_scale=scale;obj.fixed_scale=scale if fixed else None
    obj.moment_indices=list(range(dimension));obj.background_indices=[]
    obj.initial=truth+.05;obj.moment_names=[f'2::AM{a}:{i}'for i in range(4)for a in 'xyz']
    obj.cartesian_maps=np.eye(dimension).reshape(4,3,dimension);obj.atoms=list(range(4));obj.maximum_moment=9.
    mask=np.arange(n)%97==0 if masked else np.zeros(n,dtype=bool)
    obj.residual_mask=mask.copy();obj.calls=0;obj.derivative_calls=0;obj.scales=[]
    obj.histogram={'Data':[np.arange(n),observed.copy(),np.ones(n)],'Residuals':{'sumwYo':float(observed@observed)}}
    obj.histogram_id=0;obj.histogram_name='public arithmetic oracle';obj.phases={};obj.moment_basis=None
    obj.math=None;obj.lattice=None;obj.parms={};obj.args=(None,None,None,None,{':0:wtFactor':1.},None,None)
    obj.peak_memo=SimpleNamespace(diagnostics=lambda:{})
    obj._check=lambda:None
    def residual(moments,current_scale,background):
        obj.calls+=1;obj.scales.append(current_scale);obj.last_scale=current_scale
        # A unit-scale signal is tiny when the physical common scale is large.
        return (-observed+current_scale*(physical_n/scale+
            np.einsum('i,nij,j->n',moments,physical_q/scale,moments)))[~mask]
    def jacobian(moments,current_scale,background):
        obj.derivative_calls+=1
        return (2*current_scale*np.einsum('nij,j->ni',physical_q/scale,moments))[~mask].T
    obj.residual=residual;obj.magnetic_jacobian=jacobian
    return obj,truth,physical_n[~mask],physical_q[~mask],mask


@pytest.fixture(autouse=True)
def no_optional_native_cache(monkeypatch):
    from gpx_recipe.symmetry import profiled_qualification
    # This synthetic oracle has no GSAS dependency identity. The compulsory
    # full-profile holdouts and final-point checks are not mocked.
    monkeypatch.setattr(profiled_qualification,'identity',lambda *_:None)


def build(obj,path,monkeypatch):
    if path=='reflection':
        from gpx_recipe.live import native_reflection
        n=len(obj.histogram['Data'][0])
        # Independent exact amplitude oracle, evaluated at the scale actually
        # installed by the caller before invoking the reflection compiler.
        _,_,_,full_q,_=oracle(obj.source_scale)
        monkeypatch.setattr(native_reflection,'compile_reflection_quadratic',lambda **_:
            (full_q/obj.source_scale*obj.last_scale,{'fit_slice':[0,n]}))
        return obj._build()
    return obj._build(use_derivative=path=='derivative')


@pytest.mark.parametrize('scale',[1.,5907124.940466692,1e-8])
@pytest.mark.parametrize('path',['reflection','derivative','profiles'])
def test_same_physical_coefficients_and_objective_at_ordinary_large_and_tiny_scales(scale,path,monkeypatch):
    obj,truth,nuclear,quadratic,_=oracle(scale)
    model,proof=build(obj,path,monkeypatch)
    assert proof['compiler']=={'reflection':'native_reflection_amplitudes','derivative':'native_derivatives','profiles':'native_profiles'}[path]
    assert proof['coefficient_construction']['reference_scale']==scale
    np.testing.assert_allclose(model.nuclear*scale,nuclear,rtol=1e-11,atol=1e-10)
    np.testing.assert_allclose(model.quadratic*scale,quadratic,rtol=1e-10,atol=1e-10)
    for point in (truth,truth*1.1):
        native=obj.residual(point,scale,np.empty(0))
        assert model.heldout_error(point,scale,np.empty(0),native)<1e-10
    assert all(v in (0.,scale)or .8*scale<=v<=1.2*scale for v in obj.scales)


def test_unit_scale_subtraction_can_pass_random_holdouts_but_fail_low_residual_point(monkeypatch):
    scale=5907124.940466692;obj,truth,_,q,_=oracle(scale)
    anchor=np.random.default_rng(77831).uniform(.6,1.6,12)
    offset=obj.residual(anchor,0.,np.empty(0))
    old_n=(obj.residual(anchor,1.,np.empty(0))-offset)-np.einsum('i,nij,j->n',anchor,q/scale,anchor)
    error=lambda m:float(np.max(np.abs(offset+scale*(old_n+np.einsum('i,nij,j->n',m,q/scale,m))-obj.residual(m,scale,np.empty(0))))/
        max(1.,float(np.max(np.abs(obj.residual(m,scale,np.empty(0)))))))
    rng=np.random.default_rng(18019)
    for _ in range(4):
        point=rng.normal(size=12);point*=5.4/np.max(np.linalg.norm(point.reshape(4,3),axis=1))
        assert error(point)<1e-8
    assert error(truth)>1e-8
    model,_=build(obj,'reflection',monkeypatch)
    assert model.heldout_error(truth,scale,np.empty(0),obj.residual(truth,scale,np.empty(0)))<1e-12


def test_masked_points_and_free_scale_keep_the_same_physical_units(monkeypatch):
    obj,truth,nuclear,quadratic,mask=oracle(5907124.940466692,masked=True,fixed=False)
    original=obj.histogram['Data'][1].copy();model,_=build(obj,'reflection',monkeypatch)
    assert model.fixed_scale is None and len(model.offset)==np.count_nonzero(~mask)
    np.testing.assert_array_equal(obj.residual_mask,mask)
    np.testing.assert_array_equal(obj.histogram['Data'][1],original)
    for scale in (.8*obj.source_scale,1.2*obj.source_scale):
        assert model.heldout_error(truth,scale,np.empty(0),obj.residual(truth,scale,np.empty(0)))<1e-11


@pytest.mark.parametrize('scale',[0.,-1.,float('nan'),float('inf'),True])
def test_invalid_construction_scale_rejected_before_native_calls(scale):
    obj,_,_,_,_=oracle(1.);obj.source_scale=scale
    with pytest.raises(ValueError,match='positive finite source scale'):obj._build(use_derivative=False)
    assert obj.calls==0
