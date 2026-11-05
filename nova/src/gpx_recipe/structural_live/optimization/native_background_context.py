"""Retain the GPX background and obtain Back;0 response from GSAS-II itself."""
from types import SimpleNamespace
import numpy as np

def fit_table(source,reference):
    zero=np.zeros(len(source.radii));_,_,baseline,_,hist=source.snapshot(zero)
    baseline=baseline.copy();initial=float(source.fast.slow.background0)
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
    response=responses[0]
    return SimpleNamespace(renderer=SimpleNamespace(background=baseline),fallback=True,evaluate=source.native,
        background_reference=float(reference),fixed_background=baseline-initial*response,
        background_response=response,baseline_background_coefficient=initial,
        background_driver='GSAS-II native Back;0 response; all other GPX background terms retained')
