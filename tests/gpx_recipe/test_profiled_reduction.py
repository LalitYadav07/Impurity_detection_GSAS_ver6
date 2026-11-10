"""Complete-column reduction against original residuals and fitting guards."""
import numpy as np
import pytest

from gpx_recipe.symmetry.profiled_magnetic import ProfiledMagneticObjective
from gpx_recipe.symmetry.profiled_reduction import MagneticResidualReduction
from test_profiled_magnetic import fixture_model


def large_model(*, dimension=3, sites=1, fixed_scale=None, free_background=True, direct=False):
    source,truth,scale,background=fixture_model(dimension=dimension,sites=sites,
        fixed_scale=fixed_scale,free_background=free_background)
    cls=ProfiledMagneticObjective
    if direct:
        class OriginalObjective(ProfiledMagneticObjective):
            _evaluate=ProfiledMagneticObjective._evaluate_full
        cls=OriginalObjective
    repeats=max(12,8192//(len(source.offset)*dimension*(dimension+1)//2)+1)
    model=cls(offset=np.tile(source.offset,repeats),nuclear=np.tile(source.nuclear,repeats),
        quadratic=np.tile(source.quadratic,(repeats,1,1)),background=np.tile(source.background,(repeats,1)),
        moment_maps=source.moment_maps,maximum_moment=source.maximum_moment,
        normalization=source.normalization*repeats,fixed_scale=fixed_scale)
    return model,truth,scale,background


@pytest.mark.parametrize('fixed_scale',[None,1.7])
@pytest.mark.parametrize('free_background',[False,True])
@pytest.mark.parametrize('dimension,sites',[(1,1),(3,1),(6,2)])
def test_cost_gradient_and_common_scale_match_all_original_bins(dimension,sites,fixed_scale,free_background):
    model,*_=large_model(dimension=dimension,sites=sites,fixed_scale=fixed_scale,
        free_background=free_background)
    assert model.reduction is not None
    assert model.reduction_diagnostics['rows']==2+dimension*(dimension+1)//2
    random=np.random.default_rng(29176)
    for point in [np.zeros(dimension)]+list(random.normal(size=(15,dimension))):
        value,gradient,scale=model._evaluate(point)
        expected=model._evaluate_full(point)
        assert value==pytest.approx(expected[0],rel=1e-11,abs=1e-9)
        np.testing.assert_allclose(gradient,expected[1],rtol=1e-10,atol=1e-9)
        assert scale==pytest.approx(expected[2],rel=1e-11,abs=1e-12)
    assert model.reduction_evaluations>0


@pytest.mark.parametrize('fixed_scale',[None,1.7])
def test_complete_optimization_and_reassessment_agree_with_original(fixed_scale):
    model,truth,scale,_=large_model(dimension=5,sites=2,fixed_scale=fixed_scale)
    original,*_=large_model(dimension=5,sites=2,fixed_scale=fixed_scale,direct=True)
    fitted=model.fit(truth*.65)
    reference=original.fit(truth*.65)
    assert fitted.success and reference.success
    assert fitted.weighted_cost<1e-8 and reference.weighted_cost<1e-8
    np.testing.assert_allclose(fitted.coefficients,reference.coefficients,atol=3e-6,rtol=3e-6)
    assert fitted.scale==pytest.approx(scale,rel=1e-5)
    assert model.reduction_evaluations>0 and model.full_evaluations>0
    reassessed=original.reassess(fitted)
    assert reassessed.success
    assert fitted.weighted_cost==pytest.approx(reassessed.weighted_cost,abs=1e-10)


def test_final_acceptance_never_uses_the_reduced_cost_or_curvature(monkeypatch):
    model,truth,*_=large_model()
    monkeypatch.setattr(model.reduction,'evaluate',lambda *args:(-1.,np.zeros(3),1.))
    checked=model._assess(np.zeros(3),optimizer_success=True,optimizer_status=0,
        message='Forged reduction',iterations=1,evaluations=1,objective_ftol=1e-14,optimizer_scale=1.)
    assert checked.weighted_cost>0 and checked.local_descent_found and not checked.success
    assert model.reduction_evaluations==0 and model.full_evaluations>0


def test_high_contrast_cancellation_automatically_uses_original_residual():
    n=8193
    quadratic=np.zeros((n,1,1));quadratic[-2048:,0,0]=5.
    nuclear=np.full(n,1e16)
    observed=nuclear+4*quadratic[:,0,0]
    model=ProfiledMagneticObjective(offset=-observed,nuclear=nuclear,quadratic=quadratic,
        background=np.empty((n,0)),moment_maps=np.array([[[1.],[0.],[0.]]]),
        maximum_moment=4.,normalization=float(observed@observed),fixed_scale=1.)
    assert model.reduction is not None
    point=np.array([2.])
    assert model.reduction.evaluate(point,model.fixed_scale) is None
    value,gradient,scale=model._evaluate(point)
    assert value==0 and np.array_equal(gradient,np.zeros(1)) and scale==1.
    assert model.full_evaluations==1 and model.reduction_evaluations==0


def test_nonnegative_scale_boundary_uses_original_branch():
    model,*_=large_model(free_background=False)
    model.projected_offset*=-1
    reduced,report=MagneticResidualReduction.build(model.projected_offset,
        model.projected_nuclear,model.projected_quadratic)
    assert report['method']=='stable_qr'
    assert reduced.evaluate(np.zeros(3),None) is None


def test_rank_deficient_profile_columns_are_retained_without_truncation():
    model,*_=large_model()
    quadratic=model.projected_quadratic.copy()
    quadratic[:,1,1]=quadratic[:,0,0]
    quadratic[:,2,2]=quadratic[:,0,0]*(1+1e-13)
    reduced,_=MagneticResidualReduction.build(model.projected_offset,
        model.projected_nuclear,quadratic)
    assert reduced.root.shape==(8,8)
    assert not reduced.root.flags.writeable
    with pytest.raises(ValueError):
        reduced.root.flags.writeable=True
    point=np.array([1.,-.3,.7]);result=reduced.evaluate(point,1.7)
    assert result is not None
    residual=model.projected_offset+1.7*(model.projected_nuclear+
        np.einsum('i,nij,j->n',point,quadratic,point))
    assert result[0]==pytest.approx(.5*(residual@residual),rel=1e-11)


def test_failed_qr_is_an_internal_fallback_without_changing_scope(monkeypatch):
    def failed(*args,**kwargs):
        raise np.linalg.LinAlgError('Injected QR failure')
    model,*_=large_model(free_background=False)
    monkeypatch.setattr(np.linalg,'qr',failed)
    reduced,report=MagneticResidualReduction.build(model.projected_offset,
        model.projected_nuclear,model.projected_quadratic)
    assert reduced is None and report['method']=='full_pattern'
    assert 'LinAlgError' in report['reason']


def test_small_or_wide_objectives_preserve_the_original_path():
    model,*_=fixture_model()
    assert model.reduction is None
    assert model.reduction_diagnostics['method']=='full_pattern'
    source,*_=fixture_model()
    cheap=ProfiledMagneticObjective(offset=np.tile(source.offset,5),nuclear=np.tile(source.nuclear,5),
        quadratic=np.tile(source.quadratic,(5,1,1)),background=np.tile(source.background,(5,1)),
        moment_maps=source.moment_maps,maximum_moment=6.,normalization=source.normalization*5)
    assert len(cheap.offset)>128 and cheap.reduction is None


def test_new_observations_weights_and_masks_get_a_fresh_reduction():
    original,*_=large_model()
    random=np.random.default_rng(163)
    for variant in ('counts','weights','mask'):
        mask=np.arange(len(original.offset))%5!=0 if variant=='mask' else np.ones(len(original.offset),dtype=bool)
        weights=random.uniform(.2,2.,size=len(original.offset)) if variant=='weights' else np.ones(len(original.offset))
        whitening=np.sqrt(weights[mask])
        offset=original.offset.copy()
        if variant=='counts':offset+=random.normal(size=len(offset))*.1
        model=ProfiledMagneticObjective(offset=offset[mask]*whitening,
            nuclear=original.nuclear[mask]*whitening,
            quadratic=original.quadratic[mask]*whitening[:,None,None],
            background=original.background[mask]*whitening[:,None],moment_maps=original.moment_maps,
            maximum_moment=6.,normalization=original.normalization)
        assert model.reduction is not original.reduction
        if variant=='mask':
            # Enough exclusions can make the remaining objective inexpensive;
            # either path must use the new observation and declared masks.
            assert model.reduction_diagnostics['method']=='full_pattern'
        else:
            assert model.reduction is not None
        for point in random.normal(size=(5,3)):
            value,gradient,scale=model._evaluate(point)
            direct=model._evaluate_full(point)
            assert value==pytest.approx(direct[0],rel=1e-11,abs=1e-9)
            np.testing.assert_allclose(gradient,direct[1],rtol=1e-10,atol=1e-9)
            assert scale==pytest.approx(direct[2],rel=1e-11)
