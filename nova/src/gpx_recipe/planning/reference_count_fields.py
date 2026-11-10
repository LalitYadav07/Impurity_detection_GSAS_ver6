"""Positive reference-derived count fields on an explicit minute clock.

This sampler implements the mathematical law used in development reference
calibrations. It supplies joint integer counts in disjoint output regions and
shares each of four spectral multipliers across all regions receiving that
quarter's rate. It does not infer hardware, condition on observed latent state,
propagate fitted-parameter uncertainty, or qualify physical transfer.
"""
import numpy as np

VERSION='powgen-reference-lognormal-count-field/1'


def sample_reference_counts(region_quarter_rates,charge_uah,times_minutes,fit,draws,rng,*,max_bytes=128*1024**2):
    rates,q,t=[np.asarray(v) for v in (region_quarter_rates,charge_uah,times_minutes)]
    if (rates.ndim!=2 or rates.shape[1]!=4 or not 1<=len(rates)<=256
            or q.ndim!=1 or not 1<=len(q)<=100000 or t.shape!=q.shape
            or type(draws) is not int or not 1<=draws<=100000
            or type(max_bytes) is not int or max_bytes<1024
            or not isinstance(rng,np.random.Generator)
            or any(v.dtype.kind not in 'iuf' or not np.isfinite(v).all() for v in (rates,q,t))
            or np.any(rates<0) or np.any(q<0) or np.any(t[1:]<=t[:-1])):
        raise ValueError('Bounded nonnegative region/quarter rates and dose, increasing minute times, explicit draws and Generator required')
    required=draws*len(q)*(32*len(rates)+96)+rates.size*16
    if required>max_bytes:
        raise ValueError('Count campaign exceeds its working-memory budget')
    rates,q,t=[np.array(v,float,copy=True) for v in (rates,q,t)]
    if (np.any(np.diff(t)<=0) or t[-1]-t[0]>1e7
            or not np.isfinite(t[-1]-t[0])):
        raise ValueError('Minute clock loses ordering or exceeds the supported span')
    a=np.asarray(fit['log_rate_loadings'],float);kind=fit['chosen']
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>.05):
        raise ValueError('Four log-rate loadings between 0 and 0.05 are required')
    if kind=='poisson':
        if np.any(a!=0):raise ValueError('Poisson field requires zero loadings')
    elif kind=='oscillatory':
        damping,frequency=fit['damping_per_minute'],fit['frequency_cycles_per_minute']
        if not np.isfinite([damping,frequency]).all() or not 0<=damping<=4 or not 0<=frequency<=.5:
            raise ValueError('Invalid oscillatory damping or frequency')
    elif kind!='white':raise ValueError('Unknown reference field family')
    result=np.zeros((draws,len(q),len(rates)),dtype=np.int64)
    active=np.flatnonzero(q>0)
    if not len(active):return result
    clock=t[active]
    if kind=='poisson':field=np.zeros((draws,len(active)))
    elif kind=='white':field=rng.standard_normal((draws,len(active)))
    else:
        state=rng.standard_normal((draws,2));field=np.empty((draws,len(active)))
        for i,time in enumerate(clock):
            field[:,i]=state[:,0]
            if i+1<len(clock):
                delta=clock[i+1]-time;decay=np.exp(-damping*delta);angle=2*np.pi*frequency*delta
                rotation=decay*np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                state=state@rotation.T+np.sqrt(-np.expm1(-2*damping*delta))*rng.standard_normal((draws,2))
    multipliers=np.exp(field[:,:,None]*a-.5*a*a)
    expected=(multipliers@rates.T)*q[active][None,:,None]
    if not np.isfinite(expected).all() or np.any(expected>1e15):
        raise ValueError('Conditional count means exceed the finite sampling domain')
    result[:,active]=rng.poisson(expected)
    return result
