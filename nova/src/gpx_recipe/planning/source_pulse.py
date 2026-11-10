"""Positive source/monitor response on the same wavelength-scaled pulse clock.

The spectrum, monitor area/transport and timing law remain hypotheses supplied
by the caller. This operator does not identify them separately from counts.
"""
import math
import numpy as np
from scipy import sparse
from .energy_response import TOF_US_PER_M_A
from .monitor_response import MonitorResponse, build_monitor_response, _grid, _positive
from .timing_response import centred_normal_laplace_probability, interval_probability

VERSION='powgen-wavelength-scaled-source-monitor/2'


def _finite(value,name,*,nonnegative=False):
    if (isinstance(value,(bool,np.bool_)) or not np.isscalar(value)
            or not math.isfinite(float(value)) or (nonnegative and value<0)):
        raise ValueError(name+' needs a finite '+('nonnegative ' if nonnegative else '')+'scalar')
    return float(value)


def build_scaled_pulse_monitor_response(incident_nodes_a,effective_area_cm2,tof_edges_us,*,
        source_to_monitor_m,wavelength_reference_a,pulse_mean_us,time_zero_us,
        sigma_us,tau_left_us,tau_right_us,tof_windows_us=None,
        prompt_period_us=None,prompt_width_us=0.,tof_us_per_m_a=TOF_US_PER_M_A,
        quadrature_order=16,max_bytes=256*1024**2):
    """Integrate linear source nodes/area through raw monitor count intervals.

    At actual wavelength lambda, raw time is
    k*L*lambda + time_zero + lambda/lambda_ref*(pulse_mean + centred_NL).
    Sigma and the two exponential widths are specified at lambda_ref. This is
    the same mean/width convention as the scaled powder timing kernel. The
    upstream monitor path and its time zero must be declared independently;
    sharing a detector electronic offset requires evidence outside this API.

    Gaussian-Legendre integration has positive weights. Source intervals are
    split where the pulse's exponential junction crosses any recorded bin,
    gate or prompt boundary. This makes deterministic zero-width timing exact
    within polynomial arithmetic and helps resolve narrow continuous pulses.
    Increase quadrature_order and compare full count predictions; no formal
    continuous quadrature error bound is claimed. Nothing is renormalized to
    the recorded domain. The unobserved response includes gates, prompt vetoes
    and the effective pulse's tails outside recorded time coverage.
    """
    nodes=_grid(incident_nodes_a,'Incident nodes',4097)
    edges=_grid(tof_edges_us,'Monitor TOF edges',100001,positive=False)
    area=np.asarray(effective_area_cm2,float)
    if area.shape!=nodes.shape or not np.isfinite(area).all() or np.any(area<0):
        raise ValueError('Matching finite nonnegative effective areas are required')
    path=_positive(source_to_monitor_m,'Monitor flight path')
    ref=_positive(wavelength_reference_a,'Reference wavelength')
    k=_positive(tof_us_per_m_a,'TOF constant')
    mean=_finite(pulse_mean_us,'Pulse mean');zero=_finite(time_zero_us,'Time zero')
    sigma=_finite(sigma_us,'Gaussian width',nonnegative=True)
    left=_finite(tau_left_us,'Left exponential width',nonnegative=True)
    right=_finite(tau_right_us,'Right exponential width',nonnegative=True)
    slope=k*path+(mean-right+left)/ref
    if slope<=0 or k*path*nodes[0]+zero+nodes[0]/ref*mean<0:
        raise ValueError('Increasing pulse junction and nonnegative mean arrival clocks are required')
    if type(quadrature_order) is not int or not 2<=quadrature_order<=64:
        raise ValueError('An explicit quadrature order from 2 to 64 is required')
    if type(max_bytes) is not int or max_bytes<1024:
        raise ValueError('An explicit positive working memory budget is required')
    if tof_windows_us is None:
        windows=None
    else:
        windows=np.asarray(tof_windows_us,float)
        if (windows.ndim!=2 or windows.shape[1]!=2 or not 1<=len(windows)<=1024
                or not np.isfinite(windows).all() or np.any(windows[:,0]<0)
                or np.any(windows[:,1]<=windows[:,0]) or np.any(windows[1:,0]<windows[:-1,1])):
            raise ValueError('Recorded TOF windows must be bounded, sorted and disjoint')
    width=_finite(prompt_width_us,'Prompt width',nonnegative=True)
    if prompt_period_us is None:
        if width:raise ValueError('A prompt veto needs an explicit period')
        period=None
    else:
        period=_positive(prompt_period_us,'Prompt period')
        if width>=period:raise ValueError('Prompt width must be smaller than its period')
    cuts=[edges]
    if windows is not None:cuts.append(np.clip(windows.ravel(),edges[0],edges[-1]))
    if period is not None and width:
        first=math.floor((edges[0]-width)/period);last=math.floor(edges[-1]/period)
        if last-first+1>100000:raise ValueError('Prompt partition exceeds the time-piece budget')
        pulse=np.arange(first,last+1)*period
        cuts.extend([np.clip(pulse,edges[0],edges[-1]),np.clip(pulse+width,edges[0],edges[-1])])
    boundaries=np.unique(np.concatenate(cuts));mid=(boundaries[:-1]+boundaries[1:])/2
    accepted=np.ones(len(mid),bool)
    if windows is not None:
        index=np.searchsorted(windows[:,0],mid,side='right')-1
        accepted&=(index>=0)&(mid<=windows[np.maximum(index,0),1])
    if period is not None and width:
        from .prompt_gate import periodic_prompt_acceptance
        accepted&=periodic_prompt_acceptance(mid,period,width,upper_inclusive=True)
    lo=boundaries[:-1][accepted];hi=boundaries[1:][accepted]
    rows=np.searchsorted(edges,mid[accepted],side='right')-1
    matrix_bytes=(len(edges)-1)*len(nodes)*8
    fixed_bytes=4*matrix_bytes+64*(len(boundaries)+len(nodes)+len(edges))
    # CDF/SF branches allocate several temporaries. Limit every broadcast block
    # conservatively, rather than allowing source/TOF size to imply an OOM.
    block_bytes=max(1,len(lo))*quadrature_order*1024
    if fixed_bytes+block_bytes>max_bytes or 2*len(lo)*quadrature_order>2000000:
        raise ValueError('Source/monitor integration exceeds its working memory budget')
    if sigma==left==right==0:
        # Use exact source/time intersections at the deterministic limit. A
        # discontinuous CDF evaluated at rounded wavelengths can otherwise
        # misassign a sub-ulp sliver next to a time boundary.
        response=build_monitor_response(nodes,area,edges,source_to_monitor_m=path,
            mean_delay_us=zero+nodes/ref*mean,tof_windows_us=windows,prompt_period_us=period,
            prompt_width_us=width,tof_us_per_m_a=k,max_bytes=max_bytes)
        meta=dict(response.metadata,version=VERSION,
            clock='k*L*lambda+time_zero+lambda/lambda_ref*(pulse_mean+centred_normal_Laplace)',
            wavelength_reference_a=ref,pulse_mean_us=mean,time_zero_us=zero,
            sigma_us=sigma,tau_left_us=left,tau_right_us=right,
            exact_deterministic_limit=True,quadrature_order=quadrature_order,
            source_spectrum_timing_and_monitor_efficiency_separately_identified=False,
            continuous_quadrature_error_bound_established=False)
        return MonitorResponse(response.incident_nodes_a,response.tof_edges_us,
            response.matrix_cm2_a,response.unobserved_cm2_a,meta)
    dense=np.zeros((len(edges)-1,len(nodes)))
    roots,weights=np.polynomial.legendre.leggauss(quadrature_order)
    junction_wavelength=(boundaries-zero)/slope
    pieces=0
    for j in range(len(nodes)-1):
        inside=junction_wavelength[(junction_wavelength>nodes[j])&(junction_wavelength<nodes[j+1])]
        span=nodes[j+1]-nodes[j]
        split=np.r_[0.,(inside-nodes[j])/span,1.]
        for a,b in zip(split[:-1],split[1:]):
            fraction=(a+b)/2+(b-a)/2*roots
            wave=nodes[j]+span*fraction;scale=wave/ref
            effective=area[j]+fraction*(area[j+1]-area[j])
            arrive=k*path*wave+zero+scale*mean
            low=(lo[:,None]-arrive[None,:])/scale[None,:]
            high=(hi[:,None]-arrive[None,:])/scale[None,:]
            probability=interval_probability(low,high,lambda x:centred_normal_laplace_probability(x,sigma,left,right))
            measure=(b-a)*span/2*weights*effective
            np.add.at(dense[:,j],rows,probability@(measure*(1-fraction)))
            np.add.at(dense[:,j+1],rows,probability@(measure*fraction))
            pieces+=1
    matrix=sparse.csr_matrix(dense);matrix.eliminate_zeros();matrix.sort_indices()
    total=np.zeros(len(nodes))
    total[:-1]+=np.diff(nodes)*(2*area[:-1]+area[1:])/6
    total[1:]+=np.diff(nodes)*(area[:-1]+2*area[1:])/6
    unobserved=total-np.asarray(matrix.sum(axis=0)).ravel()
    if (not np.isfinite(matrix.data).all() or np.any(matrix.data<0)
            or np.any(unobserved < -512*np.finfo(float).eps*total)):
        raise ValueError('Source pulse integration failed positive count conservation')
    clipped=float(np.maximum(-unobserved,0).sum());unobserved=np.maximum(unobserved,0)
    for array in (nodes,edges,unobserved):array.setflags(write=False)
    return MonitorResponse(nodes,edges,matrix,unobserved,dict(version=VERSION,
        clock='k*L*lambda+time_zero+lambda/lambda_ref*(pulse_mean+centred_normal_Laplace)',
        tof_us_per_m_a=k,source_to_monitor_m=path,wavelength_reference_a=ref,
        pulse_mean_us=mean,time_zero_us=zero,sigma_us=sigma,tau_left_us=left,tau_right_us=right,
        effective_area_units='cm2; includes unknown efficiency, sampled area and declared upstream transfer',
        source_basis='linear_nodes',acquisition_model='unwindowed_hypothesis' if windows is None else 'specified_windows',
        tof_windows_us=None if windows is None else windows.tolist(),prompt_period_us=period,prompt_width_us=width,
        quadrature='positive Gauss-Legendre after pulse-junction/time-boundary splitting',
        quadrature_order=quadrature_order,wavelength_integration_pieces=pieces,
        source_basis_evaluated_in_dimensionless_interval_coordinates=True,exact_deterministic_limit=False,
        maximum_working_memory_budget_bytes=max_bytes,sparse_coefficients=matrix.nnz,
        floating_subtraction_clipping_cm2_a=clipped,unobserved_response_renormalized=False,
        source_count_floor_added=False,pulse_has_unbounded_gaussian_or_left_tail=bool(sigma>0 or left>0),
        source_spectrum_timing_and_monitor_efficiency_separately_identified=False,
        continuous_quadrature_error_bound_established=False,experimental_accuracy_certified=False))
