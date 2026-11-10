"""Reduced current-observation arithmetic against independent GSAS-II calls."""
import json
import time
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profiled import artificial_project
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.symmetry.profiled_magnetic import ProfiledMagneticObjective


@pytest.mark.parametrize('constrained,masked,free',[(False,False,True),(True,True,True),(False,True,False)])
def test_complete_reduced_cost_and_gradient_match_native_constraints_and_masks(tmp_path,constrained,masked,free):
    _,path=artificial_project(tmp_path,sites=2,constrained=constrained,masked=masked,
        scale_free=free,background_free=free,npoints=2101)
    native=NativeMagneticStartingPoint(path,refine_scale=free,refine_background=free,maximum_moment=6.)
    model,receipt=native.build()
    assert model.reduction is not None and receipt['residual_reduction']['method']=='stable_qr'
    random=np.random.default_rng(10837)
    for point in random.uniform(.4,1.6,size=(4,model.dimension)):
        cost,gradient,scale=model._evaluate(point)
        coefficients=model.to_native@point
        residual=model.offset+scale*(model.nuclear+
            np.einsum('i,nij,j->n',coefficients,model.quadratic,coefficients))
        background=np.linalg.lstsq(model.background,-residual,rcond=None)[0] if free else np.empty(0)
        independent=native.residual(coefficients,scale,background)
        native_gradient=model.to_native.T @ (native.magnetic_jacobian(coefficients,scale,background) @ independent)
        assert cost==pytest.approx(.5*(independent@independent),rel=1e-8,abs=1e-6)
        np.testing.assert_allclose(gradient,native_gradient,rtol=2e-7,atol=1e-5)
    assert model.reduction_evaluations>0


def test_matched_native_model_multistart_timing_keeps_full_acceptance_checks(tmp_path):
    _,path=artificial_project(tmp_path,npoints=2101)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,receipt=native.build()
    class OriginalObjective(ProfiledMagneticObjective):
        _evaluate=ProfiledMagneticObjective._evaluate_full
    original=OriginalObjective(offset=model.offset,nuclear=model.nuclear,quadratic=model.quadratic,
        background=model.background,moment_maps=model.moment_maps,maximum_moment=6.,
        normalization=model.normalization,fixed_scale=model.fixed_scale)
    seed=native.initial[native.moment_indices]
    seeds=[seed*(.4+i/150)*(-1 if i%2 else 1) for i in range(96)]
    samples={};fits={}
    for label,objective in [('original',original),('automatic',model)]:
        began=time.monotonic()
        fits[label]=[objective.fit(point,max_iterations=120) for point in seeds]
        samples[label]=time.monotonic()-began
    assert len(fits['original'])==len(fits['automatic'])==96
    assert all(fit.success for values in fits.values() for fit in values)
    for full,reduced in zip(fits['original'],fits['automatic']):
        assert reduced.weighted_cost==pytest.approx(full.weighted_cost,rel=1e-6,abs=1e-4)
    best=min(fits['automatic'],key=lambda fit:fit.weighted_cost)
    independently_checked=original.reassess(best)
    assert independently_checked.success
    residual=native.residual(best.coefficients,best.scale,best.background)
    assert best.weighted_cost==pytest.approx(.5*(residual@residual),rel=1e-7,abs=1e-6)
    (tmp_path/'multistart-timing.json').write_text(json.dumps(dict(
        scope='One public artificial GSAS-II model; 96 identical starts, full-pattern acceptance retained; excludes Galaxy, transfers and GPX publication',
        starts=96,bins=len(model.offset),dimension=model.dimension,
        native_build=receipt,reduction=model.reduction_diagnostics,
        original_fit_seconds=samples['original'],automatic_fit_seconds=samples['automatic'],
        fit_only_ratio=samples['original']/samples['automatic'],
        reduced_evaluations=model.reduction_evaluations,full_evaluations=model.full_evaluations,
        all_qualified=True,best_native_weighted_cost=float(.5*(residual@residual))),indent=2)+'\n')
