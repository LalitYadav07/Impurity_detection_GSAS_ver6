"""Retain the GPX background and obtain Back;0 response from GSAS-II itself."""
from types import SimpleNamespace
import numpy as np
from ...live import native_background


def _background_column(source,baseline,hist):
    """Reuse the reviewed background routine, not another phase calculation.

    The full native snapshot supplies an independent baseline. Two further
    evaluations of the original background routine check the compiled slope
    at the actual coefficient, retaining fixed higher and auxiliary terms.
    All dictionaries and fixed-background data passed to these checks are
    private; the owned diffraction context stays at its original anchor.
    """
    native=source.fast.native;native._check()
    hfx=f":{hist['hId']}:";key=hfx+'Back;0'
    axis=np.asarray(source.fast.x[source.fast.slice],float)
    columns,proof=native_background.compile_columns(math=native.math,histogram=hist,
        parameters=native.parms,controls=native.controls,axis=axis,names=[key])
    if columns.shape!=(len(axis),1) or not np.isfinite(columns).all():
        raise native_background.BackgroundCompilerUnavailable('Incomplete structural background column')
    fixback=hist['Background'][1].get('fixback')
    if fixback is not None:fixback=np.array(fixback,copy=True)
    parameters=native.parms.copy();initial=float(parameters[key]);checks=[]
    try:
        for delta in [0.,1.,2.]:
            parameters[key]=initial+delta
            value,_=native.math.G2pwd.getBackground(hfx,parameters,
                native.controls[hfx+'bakType'],native.controls[hfx+'histType'],axis,fixback)
            value=np.asarray(np.ma.getdata(value),float)
            expected=baseline[source.fast.slice]+delta*columns[:,0]
            if (value.shape!=axis.shape or not np.isfinite(value).all()
                    or not np.allclose(value,expected,rtol=1e-10,atol=1e-10)):
                raise native_background.BackgroundCompilerUnavailable('Structural background failed native anchor/slope checks')
            checks.append(float(np.max(abs(value-expected))))
    except (KeyError,TypeError,ValueError,FloatingPointError) as error:
        raise native_background.BackgroundCompilerUnavailable(str(error)) from error
    response=np.zeros_like(baseline);response[source.fast.slice]=columns[:,0]
    return response,dict(proof,independent_background_checks=len(checks),
        maximum_absolute_check_error=max(checks),native_background_calls=proof['native_background_calls']+len(checks),
        full_powder_calls_avoided=3,full_native_calls=1)


def _original_column(source,zero,baseline,hist):
    """Original full-powder construction for unsupported or failed helpers."""
    responses=[]
    try:
        for delta in [1.,2.]:
            theta=source.theta(zero);theta[-1]=delta
            source.fast.evaluate(theta,chemistry_domain=(source.chemistry,zero))
            changed=np.zeros_like(baseline);changed[source.fast.slice]=hist['Data'][4][source.fast.slice]
            responses.append((changed-baseline)/delta)
    finally:source.native(zero)
    if not np.allclose(responses[0],responses[1],rtol=1e-10,atol=1e-10):
        raise ValueError('Background coefficient is not a verified linear native response')
    return responses[0]

def fit_table(source,reference):
    zero=np.zeros(len(source.radii));_,_,baseline,_,hist=source.snapshot(zero)
    baseline=baseline.copy();initial=float(source.fast.slow.background0)
    try:
        response,execution=_background_column(source,baseline,hist)
    except native_background.BackgroundCompilerUnavailable as error:
        response=_original_column(source,zero,baseline,hist)
        execution=dict(method='native full powder',full_native_calls=4,
            full_powder_calls_avoided=0,fallback_reason=str(error))
    return SimpleNamespace(renderer=SimpleNamespace(background=baseline),fallback=True,evaluate=source.native,
        background_reference=float(reference),fixed_background=baseline-initial*response,
        background_response=response,baseline_background_coefficient=initial,
        background_execution=execution,
        background_driver='GSAS-II native Back;0 response; all other GPX background terms retained')
