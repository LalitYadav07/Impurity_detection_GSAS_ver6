"""Opt-in full-basis magnetic least squares with an independent native oracle.

No GSAS-II convergence flag or covariance is synthesized. The objective is
the unchanged native weighted residual with only declared linear nuisances
profiled out. Every retained candidate is forward evaluated from its GPX.
"""
from __future__ import annotations

import dataclasses
import copy
import time
import numpy as np

from .native_profiled import NativeMagneticStartingPoint
from .native_audit import profile_residual_metrics
from ..snapshot import json_safe


SCHEMA='radar-profiled-magnetic-optimizer/v1'
TOLERANCES=dict(normalized_objective_ftol=1e-12,native_identity_rtol=1e-8,
                physical_stationarity=1e-7,curvature_relative=1e-10,
                nuclear_improvement_relative=1e-8)


class ProfiledScopeError(ValueError):
    pass


def validate_scope(protocol):
    p=protocol['policy']
    if (protocol['kind']!='magnetic' or protocol['k']!=['0','0','0'] or
            p.get('moment_model')!='symmetry_allowed' or p.get('refine_cell')):
        raise ProfiledScopeError('Profiled optimizer requires fixed-geometry symmetry_allowed magnetic k=0')


def _native_diagnostics(context,model):
    """Read the serialized GPX independently of the interpolation fit."""
    vector=context.initial.copy()
    residual=context.residual(vector[context.moment_indices],
        vector[context.scale_index] if context.scale_index is not None else context.source_scale,
        vector[context.background_indices])
    jacobian=np.asarray(np.ma.getdata(context.math.dervRefine(vector,*context.args)))[:,~context.residual_mask].T
    if jacobian.shape!=(len(residual),len(vector)) or not np.isfinite(jacobian).all():
        raise ValueError('Native final Jacobian is nonfinite or has the wrong independent basis')
    gradient=jacobian.T@residual
    magnetic=context.moment_indices
    nuisance=[i for i in range(len(vector)) if i not in magnetic]
    physical_gradient=model.to_native.T@gradient[magnetic]/model.normalization
    nuisance_cosines=[]
    for i in nuisance:
        denominator=np.linalg.norm(jacobian[:,i])*np.linalg.norm(residual)
        nuisance_cosines.append(abs(gradient[i])/max(denominator,np.finfo(float).tiny))
    magnetic_jacobian=jacobian[:,magnetic]@model.to_native
    if nuisance:
        columns=jacobian[:,nuisance]
        magnetic_jacobian-=columns@np.linalg.lstsq(columns,magnetic_jacobian,rcond=1e-12)[0]
    singular=np.linalg.svd(magnetic_jacobian,compute_uv=False)
    context.residual(vector[magnetic],vector[context.scale_index] if context.scale_index is not None else context.source_scale,
                     vector[context.background_indices])
    return dict(physical_gradient=physical_gradient,physical_gradient_norm=float(np.linalg.norm(physical_gradient)),
        maximum_nuisance_gradient_cosine=max(nuisance_cosines,default=0.),
        profiled_magnetic_jacobian_singular_values=singular,
        jacobian_rank=int(np.sum(singular>max(singular,default=0.)*1e-6)),
        magnetic_dimension=len(magnetic),weighted_points=len(residual)),residual


def _retain_forward_profile(project,path,context):
    """Retain truthful arrays and variable inventory, with no native fit claim."""
    context.residual(context.initial[context.moment_indices],
        context.initial[context.scale_index] if context.scale_index is not None else context.source_scale,
        context.initial[context.background_indices])
    histogram=project.histogram(context.histogram_name)
    arrays=histogram.data['data'][1]
    for i in (3,4,5):arrays[i]=np.ma.asarray(context.histogram['Data'][i]).copy()
    # Native evaluation has its own histogram dictionary. Persist its final
    # nuclear and magnetic reflection tables as well as the calculated curve.
    histogram.data['Reflection Lists']=copy.deepcopy(context.histogram['Reflection Lists'])
    metrics=profile_residual_metrics(arrays[:3],arrays[3],histogram.data['Limits'])
    histogram.data['data'][0]['wR']=metrics['rwp_native']
    project.data['Covariance']['data']=dict(varyList=list(context.vary),variables=context.initial.tolist(),
        sig=[None]*len(context.vary),covMatrix=[],
        Rvals=dict(Rwp=metrics['rwp_native'],Nvars=len(context.vary),Nobs=metrics['fitted_points'],converged=None),
        diagnostic_source='profiled_quadratic variable inventory; no GSAS-II covariance or convergence supplied')
    project.save(str(path))
    return metrics


def refine(project,path,protocol):
    """Return separate optimizer qualification; leave native flags unavailable."""
    validate_scope(protocol)
    p=protocol['policy'];started=time.monotonic();deadline=started+p['timeout_seconds']
    try:
        context=NativeMagneticStartingPoint(path,refine_scale=p['refine_histogram_scale'],
            refine_background=p['refine_background'],maximum_moment=p['maximum_moment'],deadline=deadline)
        model,identity=context.build(heldouts=4)
    except ValueError as exc:
        raise ProfiledScopeError(str(exc)) from exc
    null_cost,_,null_scale=model._evaluate(np.zeros(model.dimension))
    null_curve=model.offset+null_scale*model.nuclear
    null_background=np.linalg.lstsq(model.background,-null_curve,rcond=None)[0] if model.background.shape[1] else np.empty(0)
    null_residual=context.residual(np.zeros(model.dimension),null_scale,null_background)
    null_identity=model.heldout_error(np.zeros(model.dimension),null_scale,null_background,null_residual)
    fit=model.fit(context.initial[context.moment_indices],max_iterations=p['optimizer_iterations'],
        objective_ftol=TOLERANCES['normalized_objective_ftol'],callback=lambda _:context._check())
    context._check()
    proof=dict(schema=SCHEMA,name='profiled_quadratic',iterations=fit.iterations,
        maximum_iterations=p['optimizer_iterations'],success=fit.optimizer_success,status=fit.optimizer_status,
        message=fit.message,tolerances=dict(TOLERANCES),uncertainty_available=False,native_convergence_declared=False,
        native_covariance_supplied=False,variable_inventory=list(context.vary),
        magnetic_variables=list(context.moment_names),nuisance_scope=dict(common_scale=p['refine_histogram_scale'],
            existing_background_coefficients=p['refine_background']),
        objective_identity=identity,raw_solution=json_safe(dataclasses.asdict(fit)),
        nuclear_baseline=dict(rwp=100*np.sqrt(2*null_cost/model.normalization),native_identity_error=null_identity,
            common_scale=null_scale,background=null_background.tolist(),same_nuisance_scope=True))
    checks=dict(raw_optimizer_success=fit.optimizer_success and fit.optimizer_status==0,
        independent_stationarity=fit.scaled_stationarity_norm<=TOLERANCES['physical_stationarity'],
        no_local_descent=not fit.local_descent_found,
        curvature_nonnegative=bool(np.min(fit.curvature_eigenvalues)>=-TOLERANCES['curvature_relative']*max(1.,float(np.max(np.abs(fit.curvature_eigenvalues))))),
        interior_solution=not fit.bound_active,
        improves_nuclear_baseline=fit.weighted_cost<null_cost*(1-TOLERANCES['nuclear_improvement_relative']),
        nuclear_null_identity=null_identity<=TOLERANCES['native_identity_rtol'])
    if fit.success:
        proof['final_native_identity']=context.apply_to_project(project,fit)
        project.save(str(path))
        fresh=NativeMagneticStartingPoint(path,refine_scale=p['refine_histogram_scale'],
            refine_background=p['refine_background'],maximum_moment=p['maximum_moment'],deadline=deadline)
        checks['serialized_basis_preserved']=fresh.vary==context.vary and np.allclose(fresh.cartesian_maps,context.cartesian_maps,rtol=1e-9,atol=1e-10)
        diagnostics,residual=_native_diagnostics(fresh,model)
        proof['diagnostics']=json_safe(diagnostics)
        checks['serialized_native_stationarity']=diagnostics['physical_gradient_norm']<=TOLERANCES['physical_stationarity'] and diagnostics['maximum_nuisance_gradient_cosine']<=TOLERANCES['physical_stationarity']
        error=model.heldout_error(fit.coefficients,fit.scale,fit.background,residual)
        proof['serialized_native_identity_error']=error
        checks['serialized_native_identity']=error<=TOLERANCES['native_identity_rtol']
        context=fresh
        proof['candidate_applied']=True
    else:
        proof['candidate_applied']=False
        checks['serialized_basis_preserved']=False
        checks['serialized_native_stationarity']=False
        checks['serialized_native_identity']=False
    proof['final_profile']=_retain_forward_profile(project,path,context)
    proof['qualification']=dict(passed=all(checks.values()),checks=checks,reasons=[key for key,value in checks.items() if not value])
    proof['elapsed_seconds']=time.monotonic()-started
    return json_safe(proof)
