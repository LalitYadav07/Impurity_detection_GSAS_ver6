"""Multiscale secants for proposals when native profiles are numerically rough.

These directions are optimization aids, never substitutes for exact likelihoods.
Choosing the most consistent adjacent scales does not certify a derivative.
"""
import numpy as np

class FeasibleDirectionUnavailable(ValueError):
    """The declared domain admits no usable coordinate stencil at this point."""


def feasible_secant(function, point, column, step, *, allowed):
    points,detail=feasible_stencil(point,column,step,allowed=allowed)
    values=[np.asarray(function(p)) for p in points]
    derivative=(values[0]-values[1])/((2 if detail['scheme']=='central' else 1)*detail['effective_step'])
    if not np.isfinite(derivative).all():raise ValueError('Nonfinite secant')
    return derivative,detail


def feasible_stencil(point,column,step,*,allowed):
    """Select the same feasible stencil before scheduling native evaluations."""
    point=np.asarray(point,float)
    if point.ndim!=1 or not np.isfinite(point).all() or not 0<=column<len(point):
        raise ValueError('Finite point and valid derivative column required')
    if not np.isfinite(step) or step<=0 or not allowed(point):
        raise ValueError('Positive step and allowed base point required')
    for reduction in range(17):
        h=float(step)/2**reduction;delta=np.zeros(len(point));delta[column]=h
        plus,minus=point+delta,point-delta
        upper,lower=allowed(plus),allowed(minus)
        if upper and lower:
            points=[plus,minus];scheme='central'
        elif upper or lower:
            points=[plus,point.copy()] if upper else [point.copy(),minus]
            scheme='forward' if upper else 'backward'
        else:continue
        return points,dict(effective_step=h,scheme=scheme,reductions=reduction)
    raise FeasibleDirectionUnavailable('No feasible finite-difference direction within the declared search')


def select_secant_steps(function, point, columns, *, allowed,
                        scales=(.001,.004,.016,.064,.256)):
    point=np.asarray(point,float);scales=np.asarray(scales,float)
    if scales.ndim!=1 or len(scales)<2 or not np.isfinite(scales).all() or np.any(scales<=0) or np.any(np.diff(scales)<=0):
        raise ValueError('At least two increasing finite positive secant scales required')
    steps=np.full(len(point),.001);receipts=[]
    for column in columns:
        previous=None;rows=[];best=None
        for step in scales:
            derivative,detail=feasible_secant(function,point,column,step,allowed=allowed)
            row=dict(requested_step=float(step),**detail,norm=float(np.linalg.norm(derivative)))
            # Compare the same stencil at genuinely different effective scales.
            # Repeated backtracking must not masquerade as independent agreement.
            if previous is not None and previous[1]['scheme']==detail['scheme'] and not np.isclose(previous[1]['effective_step'],detail['effective_step'],rtol=1e-12,atol=0):
                disagreement=float(np.linalg.norm(derivative-previous[0])/max(np.linalg.norm(derivative),np.linalg.norm(previous[0]),1e-30))
                row['adjacent_relative_difference']=disagreement
                if best is None or disagreement<best[0]:best=(disagreement,float(step))
            rows.append(row);previous=(derivative,detail)
        if best is None:raise ValueError('No distinct feasible scales for proposal direction')
        steps[column]=best[1]
        receipts.append(dict(column=int(column),selected_requested_step=best[1],
            minimum_adjacent_relative_difference=best[0],scales=rows,derivative_qualified=False))
    return steps,receipts
