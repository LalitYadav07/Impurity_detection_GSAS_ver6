"""Checked fast common-width Gaussian/normal-Laplace mixtures on a time grid.

Positive cloud-in-cell deposition preserves position weights and their first
moment. The continuous common timing kernel is convolved spectrally; its CDF
and PDF supply cubic Hermite interval integration. Analytic deposition,
periodization and interpolation bounds are separate from a declared floating
point allowance, which must be checked against the direct mixture reference.
This helper supplies neither a measured pulse law nor physical counting noise.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.fft import rfft,irfft,rfftfreq
from scipy.special import ndtr

VERSION='powgen-common-timing-spectral-mixture/3'


def _mixture_fourth_bound(deposited,step,sigma):
    """Bound sigma**4 * sup|CDF''''| for the PERIODIC positive mixture.

    Inside A sigma use the established global2.4 bound; outside use the
    monotone Gaussian third-PDF-derivative envelope (A>=3). Circular window
    masses cover any continuous query point. Non-nearest periodic images have
    distance >=period/2, with an explicit Gaussian geometric-series bound.
    Convolution with either exponential probability kernel cannot increase
    this supremum. The cumsum guard covers ordinary positive floating sums;
    the executor's separate floating allowance is still retained.
    """
    n=len(deposited);period=n*step/sigma;eps=np.finfo(float).eps
    images=2*(period/2)**3*np.exp(-.5*(period/2)**2)/np.sqrt(2*np.pi)/(1-27*np.exp(-period**2))
    best=2.4;chosen=None
    for radius in [3.,4.,6.,8.]:
        width=min(n,2*int(np.ceil(radius*sigma/step))+3)
        if width==n:mass=1.
        else:
            cumulative=np.r_[0.,np.cumsum(np.r_[deposited,deposited[:width-1]])]
            mass=min(1.,float(np.max(cumulative[width:]-cumulative[:-width]))+8*n*eps)
        tail=(radius**3-3*radius)*np.exp(-.5*radius**2)/np.sqrt(2*np.pi)
        bound=2.4*mass+tail*(1-mass)+images
        if bound<best:best=bound;chosen=(radius,mass)
    return float(best),chosen


@dataclass(frozen=True)
class ProfileIntervals:
    mean: np.ndarray
    analytic_relative_L1_bound: float
    declared_roundoff_relative_allowance: float
    clipped_negative_relative_mass: float
    metadata: dict


@dataclass(frozen=True)
class CommonTimingGrid:
    total_weight: float
    origin_us: float
    step_us: float
    cdf: np.ndarray
    pdf_per_us: np.ndarray
    metadata: dict

    def probability(self,time_us):
        x=np.asarray(time_us,float)
        if np.isnan(x).any() or x.size>2000000:raise ValueError('Bounded real-or-infinite time queries are required')
        end=self.origin_us+self.step_us*(len(self.cdf)-1)
        p=np.empty(x.shape);p[x<=self.origin_us]=0.;p[x>=end]=1.
        inside=(x>self.origin_us)&(x<end)
        if inside.any():
            coordinate=(x[inside]-self.origin_us)/self.step_us
            i=np.minimum(np.floor(coordinate).astype(np.int64),len(self.cdf)-2);t=coordinate-i
            t2=t*t;t3=t2*t
            p[inside]=((2*t3-3*t2+1)*self.cdf[i]+(t3-2*t2+t)*self.step_us*self.pdf_per_us[i]
                +(-2*t3+3*t2)*self.cdf[i+1]+(t3-t2)*self.step_us*self.pdf_per_us[i+1])
        allowance=self.metadata['declared_CDF_roundoff_allowance']
        if not np.isfinite(p).all() or p.min(initial=0)<-allowance or p.max(initial=1)>1+allowance:
            raise ValueError('Spectral CDF violates its declared floating-point allowance')
        # This only limits checked floating point range error. No data count,
        # physical tail or source-domain loss is renormalized here.
        p=np.clip(p,0.,1.)
        return p,1-p

    def intervals(self,lower_us,upper_us):
        """Rates on ordered disjoint intervals with analytic/roundoff accounting."""
        lower=np.asarray(lower_us,float);upper=np.asarray(upper_us,float)
        if (lower.ndim!=1 or upper.shape!=lower.shape or lower.size>1000000
                or np.isnan(lower).any() or np.isnan(upper).any() or np.any(upper<lower)
                or np.any(lower[1:]<upper[:-1])):
            raise ValueError('Ordered nonoverlapping time intervals are required for the L1 bound')
        c0,s0=self.probability(lower);c1,s1=self.probability(upper)
        mass=np.where(c0<.5,c1-c0,s0-s1)
        allowance=self.metadata['declared_CDF_roundoff_allowance']
        if np.any(mass < -2*allowance) or mass.sum()>1+2*len(mass)*allowance:
            raise ValueError('Spectral interval mass violates its declared floating-point allowance')
        clipped=float(-mass[mass<0].sum());mass=np.maximum(mass,0.)
        end=self.origin_us+self.step_us*(len(self.cdf)-1)
        endpoints=int(((lower>self.origin_us)&(lower<end)).sum()+((upper>self.origin_us)&(upper<end)).sum())
        budget=self.metadata.get('maximum_interval_endpoints')
        if budget is not None and endpoints>budget:
            raise ValueError('Interval query exceeds the endpoint count used for adaptive precision')
        analytic=(self.metadata['deposition_relative_L1_bound']+self.metadata['periodization_relative_L1_bound']
            +endpoints*self.metadata['Hermite_CDF_absolute_error_bound']
            +endpoints*self.metadata['spectral_CDF_alias_allowance'])
        roundoff=2*endpoints*allowance+self.metadata['deposition_normalization_roundoff']
        mean=self.total_weight*mass;mean.setflags(write=False)
        return ProfileIntervals(mean,float(analytic),float(roundoff),clipped,
            dict(version=VERSION,intervals=len(mass),analytic_bound_excludes_floating_roundoff=True,
                experimental_accuracy_certified=False))


def common_timing_grid(centers_us,weights,sigma_us,*,tau_left_us=0.,tau_right_us=0.,
        center_residual_mean=True,step_over_sigma=.04,max_bytes=256*1024**2,max_grid_points=2**22,
        interval_relative_L1_tolerance=None,maximum_interval_endpoints=None,minimum_step_over_sigma=.005):
    """Convolve weighted point means with a common normalized timing family.

    Source = Gaussian(sigma) + Exp(tau_right) - Exp(tau_left). Zero exponential
    time means a degenerate zero variate. center_residual_mean subtracts
    tau_right-tau_left, preserving the supplied mean-clock centers. The finite
    grid wraps tails with an explicit union-bound allowance; no omitted source
    energy domain is represented by that timing approximation.

    All points must share these timing parameters. Wavelength-dependent laws
    require the direct finite-volume reference or a separately checked kernel.

    An optional interval tolerance halves the grid spacing BEFORE any FFT,
    using the analytic envelope and separately declared floating allowance for
    the supplied maximum endpoint count. Its denominator is TOTAL input rate,
    not the accepted interval rate. A caller must allocate an appropriate
    absolute budget when cuts reject substantial rate. Resource/minimum-step
    failures raise; they never relax the requested tolerance.
    """
    center=np.asarray(centers_us,float);weight=np.asarray(weights,float)
    numbers=[sigma_us,tau_left_us,tau_right_us,step_over_sigma,minimum_step_over_sigma]
    if (center.ndim!=1 or not 1<=len(center)<=2**22 or weight.shape!=center.shape
            or not np.isfinite(center).all() or not np.isfinite(weight).all() or np.any(weight<0)
            or any(isinstance(v,(bool,np.bool_,str,bytes)) or not np.isscalar(v) or not math.isfinite(float(v)) for v in numbers)
            or sigma_us<=0 or tau_left_us<0 or tau_right_us<0 or not 1e-4<=step_over_sigma<=.1
            or not 1e-4<=minimum_step_over_sigma<=.1
            or type(center_residual_mean) is not bool or type(max_bytes) is not int
            or not 1024<=max_bytes<=2*1024**3 or type(max_grid_points) is not int or not 16<=max_grid_points<=2**24):
        raise ValueError('Bounded finite point weights and physical common timing parameters are required')
    if interval_relative_L1_tolerance is None:
        if maximum_interval_endpoints is not None:raise ValueError('An endpoint budget requires a requested interval tolerance')
    elif (isinstance(interval_relative_L1_tolerance,(bool,np.bool_,str,bytes))
            or not np.isscalar(interval_relative_L1_tolerance) or not math.isfinite(float(interval_relative_L1_tolerance))
            or not 0<float(interval_relative_L1_tolerance)<1 or type(maximum_interval_endpoints) is not int
            or not 0<=maximum_interval_endpoints<=2000000 or minimum_step_over_sigma>step_over_sigma):
        raise ValueError('Adaptive precision requires a bounded tolerance, endpoint count and minimum spacing')
    total=float(weight.sum())
    if not math.isfinite(total) or total<=0:raise ValueError('A finite positive total mixture weight is required')
    active=weight>0;center=center[active];weight=weight[active]/total
    shift=tau_right_us-tau_left_us if center_residual_mean else 0.
    center=center-shift
    normal_pad=10.;exponential_pad=32.
    origin=float(center.min()-normal_pad*sigma_us-exponential_pad*tau_left_us)
    stop=float(center.max()+normal_pad*sigma_us+exponential_pad*tau_right_us)
    attempts=[]
    while True:
        step=float(sigma_us*step_over_sigma)
        if not math.isfinite(stop-origin) or not math.isfinite((stop-origin)/step):
            raise ValueError('Timing grid exceeds its finite coordinate domain')
        required=max(16,int(math.ceil((stop-origin)/step))+2);n=1<<(required-1).bit_length()
        if n>max_grid_points or n*160+len(center)*64>max_bytes:
            raise ValueError('Common timing grid exceeds its declared memory/grid budget')
        coordinate=(center-origin)/step;index=np.floor(coordinate).astype(np.int64);alpha=coordinate-index
        if index.min()<0 or index.max()+1>=n:raise ValueError('Timing deposition leaves its declared padded grid')
        deposited=np.bincount(index,weights=weight*(1-alpha),minlength=n)
        deposited+=np.bincount(index+1,weights=weight*alpha,minlength=n)
        normalization=float(deposited.sum());drift=abs(normalization-1.)
        roundoff=512*np.finfo(float).eps*(1+math.log2(n))**2
        if drift>roundoff:raise ValueError('Timing deposition normalization exceeds its roundoff allowance')
        deposited/=normalization  # Only a checked floating-point sum discrepancy.
        deposition_variance=float(weight@(alpha*(1-alpha)))*step**2
        # ||g_sigma''||_1 = 4*phi(1)/sigma². Positive probability convolution
        # cannot increase that norm; Taylor remainder gives variance*norm/2.
        deposition_bound=2*np.exp(-.5)/np.sqrt(2*np.pi)*deposition_variance/sigma_us**2
        left_tail=ndtr(-(normal_pad-step_over_sigma))+(np.exp(-exponential_pad) if tau_left_us else 0.)
        right_tail=ndtr(-(normal_pad-step_over_sigma))+(np.exp(-exponential_pad) if tau_right_us else 0.)
        derivative_coefficient,local_mass=_mixture_fourth_bound(deposited,step,sigma_us)
        hermite_bound=derivative_coefficient/384*step_over_sigma**4
        alias_allowance=float(8*n*np.exp(-.5*(np.pi/step_over_sigma)**2))
        endpoints=0 if maximum_interval_endpoints is None else maximum_interval_endpoints
        envelope=float(deposition_bound+2*(left_tail+right_tail)+endpoints*(hermite_bound+alias_allowance)+2*endpoints*roundoff+drift)
        attempts.append(dict(step_over_sigma=float(step_over_sigma),grid_points=n,maximum_interval_relative_L1_bound=envelope))
        if interval_relative_L1_tolerance is None or envelope<=interval_relative_L1_tolerance:break
        if step_over_sigma<=minimum_step_over_sigma:
            raise ValueError('Requested timing precision cannot be met at the declared minimum spacing')
        step_over_sigma=max(minimum_step_over_sigma,step_over_sigma/2)
    frequency=2*np.pi*rfftfreq(n,d=step)
    kernel=np.exp(-.5*(sigma_us*frequency)**2)/((1+1j*frequency*tau_right_us)*(1-1j*frequency*tau_left_us))
    spectrum=rfft(deposited)*kernel
    pdf=irfft(spectrum,n=n)/step
    antiderivative=np.zeros_like(spectrum);antiderivative[1:]=spectrum[1:]/(1j*frequency[1:])
    primitive=irfft(antiderivative,n=n)/step
    cdf=primitive-primitive[0]+np.arange(n)/n
    if (not np.isfinite(cdf).all() or cdf.min()<-roundoff or cdf.max()>1+roundoff
            or float(-pdf[pdf<0].sum()*step)>roundoff):
        raise ValueError('Timing spectral reconstruction exceeds its roundoff allowance')
    negative_pdf=float(-pdf[pdf<0].sum()*step);pdf=np.maximum(pdf,0.)
    cdf=np.r_[cdf,1.];pdf=np.r_[pdf,pdf[0]]
    for array in (cdf,pdf):array.setflags(write=False)
    return CommonTimingGrid(total,origin,step,cdf,pdf,dict(version=VERSION,grid_points=n,
        sigma_us=float(sigma_us),tau_left_us=float(tau_left_us),tau_right_us=float(tau_right_us),
        residual_mean_centered=center_residual_mean,step_over_sigma=step_over_sigma,
        requested_interval_relative_L1_tolerance=interval_relative_L1_tolerance,
        maximum_interval_endpoints=maximum_interval_endpoints,adaptive_spacing_attempts=attempts,
        adaptive_precision_denominator='total input rate before interval cuts',
        deposition_relative_L1_bound=float(deposition_bound),periodization_relative_L1_bound=float(2*(left_tail+right_tail)),
        Hermite_CDF_absolute_error_bound=float(hermite_bound),spectral_CDF_alias_allowance=alias_allowance,
        Hermite_derivative_coefficient=derivative_coefficient,
        Hermite_local_mass_bound=None if local_mass is None else list(local_mass),
        declared_CDF_roundoff_allowance=float(roundoff),deposition_normalization_roundoff=drift,
        clipped_negative_PDF_integral=negative_pdf,floating_roundoff_bound_is_declared_not_formally_verified=True,
        missing_source_domain_mass_renormalized=False,experimental_accuracy_certified=False))
