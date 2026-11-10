"""Repeated finite interval queries on a bounded scaled-pulse spectrum.

This evaluates the same periodic Fourier/Hermite calculation as
``ScaledTimingKernel.intervals`` at fixed interval endpoints. It changes the
order of linear operations, avoiding full inverse FFTs in repeated reference
fits. It supplies no absolute CDF or new physical calibration.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.fft import rfftfreq
from scipy.special import polygamma
from .scaled_timing import ScaledTimingIntervals,ScaledTimingKernel,VERSION


@dataclass(frozen=True)
class TimingOperatorEvaluation:
    intervals: ScaledTimingIntervals
    operator: "TimingIntervalOperator"
    parameters: np.ndarray
    weighted_spectrum: np.ndarray

    def parameter_vjp(self,rate_gradient):
        """Return (component, six parameters) derivatives of these rates.

        The small floating-point clipping of negative interval values is not
        differentiated, consistent with the FFT implementation. Derivatives
        of approximation errors and physical uncertainties are not supplied.
        """
        g=np.asarray(rate_gradient,float);op=self.operator
        if g.shape!=self.intervals.mean.shape or not np.isfinite(g).all():
            raise ValueError('A finite component-by-interval rate gradient is required')
        adjoint=g@op.coefficients
        a=op.kernel.scale_nodes[:,None]*op.frequency[None,:]
        q=op.kernel.zero_nodes[:,None];w=op.frequency[None,:]
        mean,zero,sigma,left,right,gamma=self.parameters
        products=self.weighted_spectrum*adjoint[:,None,:]
        answer=np.empty((len(op.kernel.component_totals),6))
        # Form one derivative factor at a time, bounding the transient memory.
        factors=(lambda:-1j*a,lambda:-1j*w*q,lambda:-sigma*a*a,
                 lambda:-a*a*left/(1-1j*a*left),
                 lambda:-a*a*right/(1+1j*a*right),lambda:-w)
        for j,f in enumerate(factors):
            answer[:,j]=np.real(np.sum(products*f()[None,:,:],axis=(1,2)))
        return answer

    def width_variance_vjp(self,rate_gradient):
        """Algorithmic derivatives for nonnegative squared Gaussian/left/right widths.

        Their finite zero-width limits avoid an artificial zero-gradient stop.
        No physical derivative-error or identifiability certificate is supplied.
        """
        g=np.asarray(rate_gradient,float);op=self.operator
        if g.shape!=self.intervals.mean.shape or not np.isfinite(g).all():
            raise ValueError('A finite component-by-interval rate gradient is required')
        adjoint=g@op.coefficients;a=op.kernel.scale_nodes[:,None]*op.frequency[None,:]
        products=self.weighted_spectrum*adjoint[:,None,:];left,right=self.parameters[3:5]
        factors=(lambda:-.5*a*a,lambda:-.5*a*a/(1-1j*a*left),lambda:-.5*a*a/(1+1j*a*right))
        return np.stack([np.real(np.sum(products*f()[None,:,:],axis=(1,2))) for f in factors],axis=1)


@dataclass(frozen=True)
class TimingIntervalOperator:
    kernel: ScaledTimingKernel
    lower_us: np.ndarray
    upper_us: np.ndarray
    frequency: np.ndarray
    coefficients: np.ndarray
    declared_additional_roundoff_relative_per_bin: np.ndarray
    construction_metadata: dict | None = None

    def evaluate(self,parameters):
        p=self.kernel.bounds.values(parameters)
        k=self.kernel;w=self.frequency[None,:]
        a=w*k.scale_nodes[:,None];q=k.zero_nodes[:,None]
        mean,zero,sigma,left,right,gamma=p
        factor=np.exp(-.5*(sigma*a)**2-gamma*w-1j*w*(q*zero+k.scale_nodes[:,None]*(mean-right+left)))
        factor/=((1+1j*a*right)*(1-1j*a*left))
        source=k.spectrum*factor[None,:,:]
        source.setflags(write=False);p.setflags(write=False)
        rates=np.real(source.sum(axis=1)@self.coefficients.T)
        allowance=k.metadata['declared_CDF_roundoff_relative_allowance']
        extra=self.declared_additional_roundoff_relative_per_bin
        limit=k.component_totals[:,None]*(2*allowance+extra[None,:])
        if np.any(rates < -limit):
            raise ValueError('Direct interval reconstruction exceeds its declared floating allowance')
        clipped=np.maximum(-rates,0.).sum(axis=1);rates=np.maximum(rates,0.)
        endpoints=2*len(self.lower_us);length=float(np.sum(self.upper_us-self.lower_us))
        period=k.grid_points*k.step_us
        alias=k.metadata['non_Cauchy_tail_probability_bound']
        if gamma>0:
            distance=k.metadata['maximum_query_minus_central_support_distance_us']/period
            alias+=length*2*gamma/(math.pi*period*period)*float(polygamma(1,1-distance))
        relative=(np.array(k.metadata['scale_interpolation_relative_l1_bounds'])
                  +np.array(k.metadata['zero_interpolation_relative_l1_bounds'])
                  +np.array(k.metadata['position_deposition_relative_l1_bounds'])
                  +alias+endpoints*(k.metadata['Hermite_CDF_absolute_error_bound']
                                    +k.metadata['spectral_CDF_truncation_bound']))
        analytic=relative*k.component_totals
        roundoff=(2*endpoints*allowance+extra.sum())*k.component_totals
        intervals=ScaledTimingIntervals(rates,None,analytic,roundoff,
                    dict(version=VERSION,arithmetic='fixed-interval spectral/Hermite linear operator',
                         parameters=p.tolist(),clipped_negative_mass=clipped.tolist(),
                         absolute_CDF_not_supplied=True,periodic_image_relative_l1_bound=alias,
                         parameter_derivative_error_not_certified=True,
                         floating_allowance_formally_verified=False,
                         geometry_QMC_and_physical_error_excluded=True,
                         experimental_accuracy_certified=False))
        # An evaluation is transient: consume its VJP before moving to the next
        # region rather than retaining every source-by-node matrix in a fit.
        return TimingOperatorEvaluation(intervals,self,p,source)


def compile_interval_operator(kernel,lower_us,upper_us,*,max_bytes=256*1024**2):
    """Compile the fixed linear functional for ordered finite intervals.

    The coefficient matrix is bounded explicitly. The additional roundoff
    allowance covers the declared floating evaluation model; it is not a
    formal proof for arbitrary BLAS, libm or hardware implementations.
    """
    if not isinstance(kernel,ScaledTimingKernel) or type(max_bytes) is not int or not 1024<=max_bytes<=2*1024**3:
        raise ValueError('A bounded scaled-timing kernel and explicit memory cap are required')
    lo=np.asarray(lower_us,float);hi=np.asarray(upper_us,float)
    if (lo.ndim!=1 or hi.shape!=lo.shape or not 1<=len(lo)<=1000000
            or not np.isfinite(lo+hi).all() or np.any(hi<lo) or np.any(lo[1:]<hi[:-1])
            or lo.min()<kernel.query_range_us[0] or hi.max()>kernel.query_range_us[1]
            or 2*len(lo)>kernel.metadata['maximum_interval_endpoints']):
        raise ValueError('Ordered disjoint finite intervals inside the compiled query range are required')
    n=kernel.grid_points;nk=kernel.spectrum.shape[2];size=len(lo)*nk
    # Endpoint phases, coefficients, and construction work are all included.
    original_peak=size*256+nk*128
    # Full endpoint expression: bound its simultaneous complex temporaries,
    # final/ownership coefficient arrays, vectors, and full rfftfreq temporary.
    peak=max(112*size+128*nk+256*len(lo),8*(n+2)+64*nk+64*len(lo))
    if peak>max_bytes:raise ValueError('Fixed interval operator exceeds its compile memory cap')
    frequency=2*np.pi*rfftfreq(n,d=kernel.step_us)[:nk];w=frequency[1:]
    def endpoint(e):
        coordinate=(e-kernel.origin_us)/kernel.step_us;index=np.floor(coordinate).astype(np.int64);t=coordinate-index
        if index.min()<0 or index.max()+1>=n:
            raise ValueError('Interval operator endpoints leave the periodic grid')
        t2=t*t;t3=t2*t
        phase=np.exp(1j*(index[:,None]*kernel.step_us)*w[None,:])
        next_phase=np.exp(1j*kernel.step_us*w)
        left=(2*t3-3*t2+1)[:,None]/(1j*w[None,:])+(t3-2*t2+t)[:,None]*kernel.step_us
        right=(-2*t3+3*t2)[:,None]/(1j*w[None,:])+(t3-t2)[:,None]*kernel.step_us
        return phase*(left+right*next_phase[None,:])
    coefficient=np.empty((len(lo),nk),complex);period=n*kernel.step_us
    coefficient[:,0]=(hi-lo)/period
    coefficient[:,1:]=2/period*(endpoint(hi)-endpoint(lo))
    if nk==n//2+1:coefficient[:,-1]*=.5
    # Each spectral coefficient is bounded by the positive component mass.
    # Allow formation, node summation, dot products, and phase arithmetic.
    extra=128*np.finfo(float).eps*(nk+len(kernel.scale_nodes)+32)*np.sum(abs(coefficient),axis=1)
    arrays=[np.array(v,copy=True) for v in [lo,hi,frequency,coefficient,extra]]
    for v in arrays:v.setflags(write=False)
    return TimingIntervalOperator(kernel,*arrays,construction_metadata=dict(refined_memory_accounting=original_peak>max_bytes,original_full_vector_expression_preserved=True,rows_per_block=len(lo),
        estimated_peak_compile_array_bytes=peak,compile_cap_bytes=max_bytes,coefficient_row_formula_unchanged=True))
