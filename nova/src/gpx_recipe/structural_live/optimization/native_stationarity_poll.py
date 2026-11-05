"""Exact-native descent checks when a proposal Jacobian reports stagnation.

No truth, likelihood approximation, domain relaxation or convergence guarantee.
"""
import numpy as np

def stationarity_poll(point,cost,jacobian,mode_dimensions,cell_dimensions,cell_steps,evaluate,allowed):
    z=np.asarray(point,float);j=np.asarray(jacobian,float);m=mode_dimensions;c=cell_dimensions
    if j.ndim!=2 or j.shape[1]!=len(z) or m+c+2!=len(z):raise ValueError('Matching complete parameter/Jacobian dimensions required')
    if not np.isfinite(z).all() or not np.isfinite(j).all() or not np.isfinite(cost):raise ValueError('Finite polling context required')
    steps=np.asarray(cell_steps,float)
    if steps.shape!=(m+c,) or np.any(steps<=0) or not np.isfinite(steps).all():raise ValueError('Positive cell proposal scales required')
    directions=[]
    for i in range(m,m+c):
        v=np.zeros(len(z));v[i]=1.
        for h in sorted(set(np.clip(steps[i]*np.array([.25,1.,4.]),1e-4,.2))):directions.append((v,float(h),'cell'))
    if m:
        nuisance=j[:,m:];jm=j[:,:m]
        compensation=np.linalg.lstsq(nuisance,jm,rcond=1e-10)[0]
        projected=jm-nuisance@compensation
        _,_,vh=np.linalg.svd(projected,full_matrices=False)
        for v in vh[-min(m,3):]:
            joint=np.r_[v,-compensation@v];joint/=max(np.max(abs(joint)),1e-30)
            for h in (.02,.08,.2):directions.append((joint,h,'weak_joint_mode'))
    best=z.copy();best_cost=float(cost);tried=0;infeasible=0;kind=None
    for v,h,label in directions:
        for sign in (-1,1):
            candidate=z+sign*h*v
            if not allowed(candidate):infeasible+=1;continue
            value=float(evaluate(candidate));tried+=1
            if not np.isfinite(value):raise ValueError('Native polling objective must be finite')
            if value<best_cost-1e-10:best=candidate;best_cost=value;kind=label
    return best,dict(initial_cost=float(cost),best_cost=best_cost,improved=best_cost<float(cost)-1e-10,
        evaluations=tried,infeasible=infeasible,direction_kind=kind,global_stationarity_certified=False)
