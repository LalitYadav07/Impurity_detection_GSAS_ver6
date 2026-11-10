"""Bounded reference bin probabilities for a Gaussian/asymmetric-Laplace law.

The positive exponential-component integral preserves a bin's width separately
from its shifted midpoint. This avoids subtracting rounded CDF values and
losing narrow intervals when a large centroid is removed. It is a reference
integration path, not a calibrated POWGEN moderator or a fast response pack.
"""
from dataclasses import dataclass
import math
import warnings
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import ndtr,log_ndtr

VERSION='powgen-positive-normal-laplace-intervals/3'
_NODE,_WEIGHT=np.polynomial.legendre.leggauss(8)
_ROUGH_NODE,_ROUGH_WEIGHT=np.polynomial.legendre.leggauss(16)
_REFINE_NODE,_REFINE_WEIGHT=np.polynomial.legendre.leggauss(32)


@dataclass(frozen=True)
class NormalLaplaceIntervals:
    probability: np.ndarray
    absolute_quadrature_error: np.ndarray
    metadata: dict


def _log_normal_interval(center,width,lower,upper,sigma):
    if math.isfinite(center) and math.isfinite(width):
        # A bin wholly beyond56sigma has mass belowPhi(-56)<exp(-1572).
        # Even the largest allowed integrand multiplier2**1000 leaves this
        # belowbinary64's smallest subnormal; exp(-u) only reduces it.
        # Preserve this guaranteed scaled underflow before shifted endpoints
        # can collapse. This is not a probability floor or area adjustment.
        nearest=(abs(center)-.5*width)/sigma
        if nearest>=56.:return -math.inf
        z=center/sigma;h=.5*width/sigma
        if h>0 and h*(1+abs(z)+h)<.25:
            return (math.log(h)-.5*z*z-.5*math.log(2*math.pi)
                +math.log(float(_WEIGHT@np.exp(-z*h*_NODE-.5*(h*_NODE)**2))))
    a,b=lower/sigma,upper/sigma
    if a>=0:leading=float(log_ndtr(-a));trailing=float(log_ndtr(-b))
    else:leading=float(log_ndtr(b));trailing=float(log_ndtr(a))
    if leading==-math.inf:return -math.inf
    difference=trailing-leading
    if difference>=0:raise ValueError('Unresolved log Gaussian interval width')
    return leading+math.log(-math.expm1(difference))


def _normal_interval(center,width,lower,upper,sigma):
    log_mass=_log_normal_interval(center,width,lower,upper,sigma)
    return math.exp(log_mass)


def _exponential_interval(center,width,lower,upper,tau):
    if upper<=0:return 0.
    if lower<0:return -math.expm1(-upper/tau)
    # Preserve a narrow width even when subtracting shifted endpoints would
    # erase it. This branch is also valid for an infinite upper endpoint.
    return math.exp(-lower/tau)*(-math.expm1(-width/tau))


def normal_laplace_intervals(lower,upper,*,sigma,tau_left,tau_right,centred=False,
        relative_tolerance=2e-10,max_intervals=4096,max_quad_segments=200000):
    """Integrate N(0,sigma²)+Exp(tau_right)-Exp(tau_left) in (lower,upper].

    ``centred=True`` subtracts the law's mean (tau_right-tau_left). All inputs
    use the same time or length unit. Zero widths are exact point-mass limits;
    continuous laws are unaffected by endpoint inclusion. Raw detector half-open
    atom policies must be supplied separately. Parameters broadcast with bins.

    Integration is positive, without probability clipping or area repair.
    The exponential latent variable is bounded at800, with omitted probability
    below4e-348. Warnings or unresolved quadrature errors raise. This explicit
    reference path is slower than the CDF path and makes no physical accuracy
    claim. Width/rate, source, geometry and material calibration remain external.
    The reported quadrature error is an adaptive estimate, not a certified
    bound on all floating-point roundoff and elementary-function error.
    """
    if (type(centred) is not bool or type(max_intervals) is not int or not 1<=max_intervals<=100000
            or type(max_quad_segments) is not int or not 1<=max_quad_segments<=2000000
            or isinstance(relative_tolerance,(bool,str,bytes)) or not np.isscalar(relative_tolerance)
            or not np.isfinite(relative_tolerance) or not 1e-12<=relative_tolerance<=1e-5):
        raise ValueError('Explicit bounded intervals, quadrature work and tolerance are required')
    values=np.broadcast_arrays(*[np.asarray(v,float) for v in [lower,upper,sigma,tau_left,tau_right]])
    lo,hi,sig,left,right=values
    if (lo.size>max_intervals or np.isnan(lo).any() or np.isnan(hi).any() or np.any(hi<lo)
            or any(not np.isfinite(v).all() or np.any(v<0) for v in [sig,left,right])):
        raise ValueError('Ordered non-NaN bins and finite nonnegative widths are required')
    probability=np.empty(lo.shape);error=np.empty(lo.shape);segments=0;complements=0;underflows=0

    def component(a,b,center,width,s,tau,sign):
        nonlocal segments
        if tau==0:return _normal_interval(center,width,a,b,s),0.,0
        nodes={0.,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,.001,.01,.1,1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,800.}
        for factor in [.001,.01,.1,1.,4.,12.,40.]:
            z=factor*s/tau
            if 0<z<800:nodes.add(z)
        # Resolve each smoothed endpoint crossing in exponential coordinates.
        for endpoint in [a,b]:
            if not math.isfinite(endpoint):continue
            crossing=sign*endpoint/tau
            for n in [-12.,-4.,-1.,-.1,0.,.1,1.,4.,12.]:
                z=crossing+n*s/tau
                if 0<z<800:nodes.add(z)
        ordered=sorted(nodes);bounds=[ordered[0]]
        # These are auxiliary integration partitions, not histogram edges.
        # Merge almost equal partitions so the affine quadrature transform
        # does not collapse its nodes onto identical binary64 values.
        for point in ordered[1:-1]:
            if point-bounds[-1]>8*max(math.ulp(point),math.ulp(bounds[-1])):bounds.append(point)
        if 800.-bounds[-1]>8*max(math.ulp(800.),math.ulp(bounds[-1])):bounds.append(800.)
        else:bounds[-1]=800.
        segments+=len(bounds)-1
        if segments>max_quad_segments:raise ValueError('Normal/Laplace quadrature exceeds its work budget')
        def integrand(u,scale_power=0):
            shift=sign*tau*u
            c=math.fsum([center,-shift]) if math.isfinite(center) else center
            aa=math.fsum([a,-shift]);bb=math.fsum([b,-shift])
            if scale_power:
                log_mass=_log_normal_interval(c,width,aa,bb,s)
                return math.exp(log_mass-u+scale_power*math.log(2.))
            mass=_normal_interval(c,width,aa,bb,s)
            if not 0<=mass<=1:raise ValueError('Gaussian component probability is outside its numerical domain')
            return math.exp(-u)*mass
        parts=[];estimates=[]
        # The error budget belongs to the complete component probability.
        # Requiring relative precision on every negligible subinterval causes
        # underflow/roundoff warnings without improving the requested integral.
        rough=[]
        for a0,b0 in zip(bounds[:-1],bounds[1:]):
            points=.5*(a0+b0)+.5*(b0-a0)*_ROUGH_NODE
            rough.append(.5*(b0-a0)*math.fsum(float(w)*integrand(float(x))
                for x,w in zip(points,_ROUGH_WEIGHT)))
        rough_total=math.fsum(rough)
        scale_power=1000 if rough_total<2.**-500 else 0
        scaled_integrand=lambda u:integrand(u,scale_power)
        if scale_power:
            rough=[]
            for a0,b0 in zip(bounds[:-1],bounds[1:]):
                points=.5*(a0+b0)+.5*(b0-a0)*_ROUGH_NODE
                rough.append(.5*(b0-a0)*math.fsum(float(w)*scaled_integrand(float(x)) for x,w in zip(points,_ROUGH_WEIGHT)))
            rough_total=math.fsum(rough)
        absolute_budget=relative_tolerance*rough_total/(8*(len(bounds)-1))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',IntegrationWarning)
            for a0,b0 in zip(bounds[:-1],bounds[1:]):
                if b0-a0<=math.sqrt(np.finfo(float).eps)*max(1.,abs(a0),abs(b0)):
                    values=[]
                    for node,weight in [(_ROUGH_NODE,_ROUGH_WEIGHT),(_REFINE_NODE,_REFINE_WEIGHT)]:
                        points=.5*(a0+b0)+.5*(b0-a0)*node
                        values.append(.5*(b0-a0)*math.fsum(float(w)*scaled_integrand(float(x)) for x,w in zip(points,weight)))
                    estimate=abs(values[1]-values[0])
                    if estimate<=absolute_budget+relative_tolerance*values[1]/8:
                        parts.append(values[1]);estimates.append(estimate);continue
                value,estimate=quad(scaled_integrand,a0,b0,epsabs=absolute_budget,epsrel=relative_tolerance/8,limit=200)
                parts.append(value);estimates.append(estimate)
        value=math.fsum(parts);estimate=math.fsum(estimates)
        if caught:raise ValueError('Normal/Laplace quadrature warning: '+str(caught[0].message))
        if estimate>relative_tolerance*value:
            raise ValueError('Normal/Laplace quadrature error exceeds its declared tolerance')
        return value,estimate,scale_power

    def integrated(a,b,c,w,s,l,r):
        if l==r==0:return _normal_interval(c,w,a,b,s),0.
        scale=max(l,r);den=l/scale+r/scale
        pieces=[]
        for tau,sign,weight in [(l,-1,(l/scale)/den),(r,1,(r/scale)/den)]:
            if not weight:continue
            p,e,power=component(a,b,c,w,s,tau,sign);pieces.append((weight,p,e,power))
        power=max(row[3] for row in pieces)
        result=[weight*math.ldexp(p,power-k) for weight,p,e,k in pieces]
        errors=[weight*math.ldexp(e,power-k) for weight,p,e,k in pieces]
        # Round only the final weighted mixture into binary64's subnormal
        # range. Rounding each component before multiplying its weight loses
        # another ULP even when the underlying positive integrals are accurate.
        return math.ldexp(math.fsum(result),-power),math.ldexp(math.fsum(errors),-power)

    for index in np.ndindex(lo.shape):
        a,b,s,l,r=[float(v[index]) for v in values]
        if a==b:probability[index]=error[index]=0.;continue
        if a==-math.inf and b==math.inf:probability[index]=1.;error[index]=0.;continue
        width=b-a
        center=math.fsum([a,.5*width]) if math.isfinite(width) else math.nan
        if centred:
            if math.isfinite(center):center=math.fsum([a,.5*width,r,-l])
            a=math.fsum([a,r,-l]);b=math.fsum([b,r,-l])
        if s==0:
            if l==r==0:value=float(a<0<=b)
            else:
                scale=max(l,r);den=l/scale+r/scale;terms=[]
                if l:terms.append((l/scale)/den*_exponential_interval(-center,width,-b,-a,l))
                if r:terms.append((r/scale)/den*_exponential_interval(center,width,a,b,r))
                value=math.fsum(terms)
            estimate=0.
        else:
            value,estimate=integrated(a,b,center,width,s,l,r)
            if value>.5:
                lower_loss,e0=integrated(-math.inf,a,math.nan,math.inf,s,l,r)
                upper_loss,e1=integrated(b,math.inf,math.nan,math.inf,s,l,r)
                value=1-math.fsum([lower_loss,upper_loss]);estimate=e0+e1;complements+=1
        if not math.isfinite(value) or not 0<=value<=1 or not math.isfinite(estimate):
            raise ValueError('Integrated normal/Laplace probability is outside its numerical domain')
        if value==0 and (s>0 or (l>0 and a<0) or (r>0 and b>0)):underflows+=1
        probability[index]=value;error[index]=estimate
    probability.setflags(write=False);error.setflags(write=False)
    return NormalLaplaceIntervals(probability,error,dict(version=VERSION,quadrature_segments=segments,
        complement_evaluations=complements,exponential_latent_upper_limit=800.,omitted_probability_bound='4e-348',
        positive_component_integrals=True,interval_width_preserved_before_centroid_shift=True,
        tiny_component_integrands_scaled_by_powers_of_two=True,
        Gaussian_interval_probabilities_evaluated_in_log_domain=True,
        scaled_components_combined_before_final_probability_rounding=True,
        continuous_law_probability_underflow_bins=underflows,
        quadrature_error_is_estimate_not_total_roundoff_bound=True,
        probability_clipping_or_area_renormalization=False,experimental_accuracy_certified=False))
