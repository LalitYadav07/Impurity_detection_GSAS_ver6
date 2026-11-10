"""Quantify observed peak discrepancies against a fixed density prediction.

Local amplitude/background, shift and dilation estimate observations. They do
not recalibrate the prediction. Supplied reduced marginal errors define a
working least-squares metric, not an independent count law or calibrated
predictive coverage. Specimen, instrument and shared-reference uncertainty
require separate evidence.
"""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks

VERSION = 'powgen-fixed-profile-peak-observations/1'


@dataclass(frozen=True)
class LinearDensity:
    x: np.ndarray
    density: np.ndarray

    def __post_init__(self):
        x,y = np.asarray(self.x),np.asarray(self.density)
        if (x.ndim!=1 or y.shape!=x.shape or not 3<=len(x)<=100000
                or any(a.dtype.kind not in 'iuf' or not np.isfinite(a).all() for a in [x,y])
                or np.any(np.diff(x)<=0) or np.any(y<0) or not np.any(y>0)):
            raise ValueError('Bounded increasing coordinates and nonnegative finite density required')
        x,y=np.array(x,float,copy=True),np.array(y,float,copy=True)
        slope=np.diff(y)/np.diff(x)
        integral=np.r_[0.,np.cumsum(np.diff(x)*(y[:-1]+y[1:])/2)]
        if not np.isfinite(integral).all() or integral[-1]<=0:
            raise ValueError('Finite positive integrated prediction required')
        for name,a in [('x',x),('density',y),('_slope',slope),('_integral',integral)]:
            a.setflags(write=False);object.__setattr__(self,name,a)

    def antiderivative(self,points):
        p=np.asarray(points,float)
        if p.ndim!=1 or not np.isfinite(p).all() or len(p)>250001:
            raise ValueError('Finite bounded integration coordinates required')
        index=np.clip(np.searchsorted(self.x,p,side='right')-1,0,len(self.x)-2)
        delta=np.clip(p,self.x[0],self.x[-1])-self.x[index]
        return self._integral[index]+self.density[index]*delta+self._slope[index]*delta**2/2

    def average(self,edges,*,center,shift=0.,dilation=1.):
        e=np.asarray(edges,float)
        if (e.ndim!=1 or not 2<=len(e)<=250001 or not np.isfinite(e).all() or np.any(np.diff(e)<=0)
                or not np.isfinite([center,shift,dilation]).all() or dilation<=0):
            raise ValueError('Increasing finite bin edges and a positive dilation required')
        # The density transforms as f((x-center-shift)/dilation+center)/dilation:
        # integrated area is preserved rather than multiplied by fitted width.
        u=(e-center-shift)/dilation+center
        area=np.diff(self.antiderivative(u))
        if np.any(area < -1e-12*self._integral[-1]):
            raise ValueError('Invalid integrated profile')
        return np.maximum(area,0)/np.diff(e)


def isolated_peaks(profile,*,prominence_fraction=.01,window_half_fwhm=4.):
    """Select a fixed, prediction-only denominator; report every candidate."""
    if not isinstance(profile,LinearDensity) or not 0<prominence_fraction<1 or not 1<=window_half_fwhm<=10:
        raise ValueError('A fixed profile and explicit bounded peak-selection rules required')
    y,x=profile.density,profile.x
    all_peaks,_=find_peaks(y,height=1e-6*y.max())
    peaks,_=find_peaks(y,prominence=prominence_fraction*y.max())
    records=[]
    for p in peaks:
        half=y[p]/2;left=np.flatnonzero(y[:p]<=half);right=np.flatnonzero(y[p+1:]<=half)
        record={'index':int(p),'center':float(x[p]),'height':float(y[p]),'eligible':False}
        if not len(left) or not len(right):
            record['reason']='half_maximum_outside_prediction';records.append(record);continue
        a,b=int(left[-1]),int(p+1+right[0])
        lo=x[a]+(x[a+1]-x[a])*(half-y[a])/(y[a+1]-y[a])
        hi=x[b-1]+(x[b]-x[b-1])*(half-y[b-1])/(y[b]-y[b-1])
        width=float(hi-lo);window=[float(x[p]-window_half_fwhm*width),float(x[p]+window_half_fwhm*width)]
        neighbors=all_peaks[all_peaks!=p]
        nearby=bool(np.any(abs(x[neighbors]-x[p])<=2*window_half_fwhm*width))
        reason=('window_outside_prediction' if window[0]<x[0] or window[1]>x[-1]
                else 'nearby_predicted_peak' if nearby else 'eligible')
        record.update(fwhm=width,window=window,eligible=reason=='eligible',reason=reason)
        records.append(record)
    return records


def fit_peak(profile,edges,density,sigma,*,center,fwhm,shift_limit_fwhm=1.5,
             dilation_bounds=(.25,4.),signal_to_noise_min=10.):
    """Fit one preselected window with area-preserving finite-bin integration.

The nine starts and parameter bounds are fixed. No prediction parameter is
changed. Zero supplied errors are retained in the input record and explicitly
excluded from this marginal-error fit; they are never assigned an error floor.
    """
    e,y,s=np.asarray(edges,float),np.asarray(density,float),np.asarray(sigma,float)
    bounds=np.asarray(dilation_bounds,float)
    if (not isinstance(profile,LinearDensity) or e.ndim!=1 or y.ndim!=1 or s.shape!=y.shape
            or e.shape!=(len(y)+1,) or len(y)>250000 or not np.isfinite(e).all()
            or not np.isfinite(y).all() or not np.isfinite(s).all() or np.any(s<0) or np.any(np.diff(e)<=0)
            or not np.isfinite([center,fwhm,shift_limit_fwhm,signal_to_noise_min]).all()
            or fwhm<=0 or not 0<shift_limit_fwhm<=3 or signal_to_noise_min<=0
            or bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<bounds[0]<1<bounds[1]<=8):
        raise ValueError('Matching finite physical observations and explicit fit bounds required')
    use=s>0
    if np.count_nonzero(use)<8:raise ValueError('At least eight positive-error bins required')
    coordinate=((e[:-1]+e[1:])/2-center)/fwhm
    template=profile.average(e,center=center)
    design=np.column_stack([template,np.ones_like(y),coordinate])
    initial=np.linalg.lstsq(design[use]/s[use,None],y[use]/s[use],rcond=None)[0]
    initial[0]=max(initial[0],1e-12*max(1.,float(np.max(abs(y)))/max(float(template.max()),1e-300)))
    lower=[0.,-np.inf,-np.inf,-shift_limit_fwhm,np.log(bounds[0])]
    upper=[np.inf,np.inf,np.inf,shift_limit_fwhm,np.log(bounds[1])]

    def expected(p):
        return p[0]*profile.average(e,center=center,shift=p[3]*fwhm,dilation=np.exp(p[4]))+p[1]+p[2]*coordinate

    def residual(p):return (expected(p)[use]-y[use])/s[use]

    attempts=[]
    for shift in [-.5,0.,.5]:
        for dilation in [.7,1.,1.4]:
            start=np.r_[initial,np.clip(shift,-.9*shift_limit_fwhm,.9*shift_limit_fwhm),
                        np.clip(np.log(dilation),np.log(bounds[0])+.0001,np.log(bounds[1])-.0001)]
            fit=least_squares(residual,start,bounds=(lower,upper),x_scale='jac',
                ftol=1e-10,xtol=1e-10,gtol=1e-10,max_nfev=2000)
            attempts.append(fit)
    best=min(attempts,key=lambda r:r.cost);parameters=best.x
    scales=np.linalg.norm(best.jac,axis=0)
    normalized=np.divide(best.jac,scales,out=np.zeros_like(best.jac),where=scales>0)
    singular=np.linalg.svd(normalized,compute_uv=False)
    rank=int(np.count_nonzero(singular>np.finfo(float).eps*max(best.jac.shape)*singular[0]))
    covariance=None
    if rank==5 and singular[0]/singular[-1]<1e8:
        _,v,vt=np.linalg.svd(normalized,full_matrices=False)
        covariance=((vt.T/v**2)@vt)/np.outer(scales,scales)
    amplitude_sd=float(np.sqrt(covariance[0,0])) if covariance is not None else None
    snr=float(parameters[0]/amplitude_sd) if amplitude_sd and amplitude_sd>0 else None
    boundary=bool(abs(parameters[3])>=shift_limit_fwhm-1e-6 or
        min(abs(parameters[4]-np.log(bounds)))<1e-6)
    status=('optimizer_failed' if not best.success else 'parameter_bound' if boundary else
            'unidentified_local_covariance' if covariance is None else
            'low_signal' if snr is None or snr<signal_to_noise_min else 'estimated')
    return {'schema':VERSION,'status':status,'fixed_multistarts':9,'successful_optimizer_starts':sum(r.success for r in attempts),
        'amplitude':float(parameters[0]),'amplitude_standard_error':amplitude_sd,'amplitude_signal_to_noise':snr,
        'background_constant':float(parameters[1]),'background_slope_per_fwhm':float(parameters[2]),
        'position_difference_in_predicted_fwhm':float(parameters[3]),
        'width_ratio_observed_to_predicted':float(np.exp(parameters[4])),
        'working_marginal_chi_squared':float(2*best.cost),'positive_error_bins':int(use.sum()),
        'excluded_zero_error_indices':np.flatnonzero(~use).tolist(),
        'conditional_local_covariance':covariance.tolist() if covariance is not None else None,
        'covariance_scope':'Local marginal-error least squares only; excludes instrument/CIF/specimen/shared-reference uncertainty; no coverage qualification',
        'prediction_recalibrated':False,'experimental_accuracy_certified':False}
