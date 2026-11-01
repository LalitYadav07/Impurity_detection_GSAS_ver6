"""Compile fixed-nuisance magnetic intensity from checked native derivatives.

For I(p)=I(0)+p'Qp, [dI/dp_j(e_i)-dI/dp_j(-e_i)]/4=Q_ji.
Opposite signs cancel the native zero-component guard's first-order offsets.
Only magnetic columns are used; known problematic native cell columns are
irrelevant here. Independent native profile checks remain compulsory.
"""
import time
import numpy as np


def compile_native_magnetic_jacobian(objective, reference, indices, *, seed=192, checks=5):
    started=time.monotonic();reference=np.asarray(reference,float);indices=list(indices)
    if not indices or len(set(indices))!=len(indices) or checks<1:raise ValueError('Distinct magnetic indices and positive validation budget required')
    dimension=len(indices);zero=reference.copy();zero[indices]=0.
    original_numerical=list(objective.numerical_indices)
    def state(p):
        x=reference.copy();x[indices]=p;return x
    try:
        # Avoid computing unused finite-difference nuclear columns. Their native
        # analytic values are discarded; the objective's policy is restored below.
        objective.numerical_indices=[]
        base=objective.profile(zero)
        q=np.empty((dimension,dimension,len(base)))
        for i in range(dimension):
            p=np.zeros(dimension);p[i]=1.
            _,positive=objective.profile(state(p),True)
            _,negative=objective.profile(state(-p),True)
            q[:,i,:]=(positive[indices]-negative[indices])/4.
        magnitude=max(float(np.max(abs(q))),1e-12)
        asymmetry=float(np.max(abs(q-q.swapaxes(0,1)))/magnitude)
        if not np.isfinite(q).all() or asymmetry>1e-7:raise ValueError('Native magnetic derivative kernel is not symmetric')
        q=(q+q.swapaxes(0,1))/2
        rng=np.random.default_rng(seed);errors=[]
        for _ in range(checks):
            p=rng.normal(size=dimension)*3.
            direct=objective.profile(state(p));predicted=base+np.einsum('i,ijb,j->b',p,q,p,optimize=True)
            error=float(np.max(abs(direct-predicted))/max(np.max(abs(direct)),1e-12));errors.append(error)
        if max(errors)>1e-7:raise ValueError('Derivative-compiled magnetic kernel fails native forward checks')
    finally:
        objective.numerical_indices=original_numerical
        objective.profile(reference)
    return dict(base=base,quadratic=q,diagnostics=dict(dimensions=dimension,derivative_calls=2*dimension,
        symmetry_relative_error=asymmetry,native_forward_validation_errors=errors,
        seconds=time.monotonic()-started,numerical_nuclear_policy_restored=True,
        scope='Fixed nuisance magnetic quadratic model; native cell derivative columns not used'))
