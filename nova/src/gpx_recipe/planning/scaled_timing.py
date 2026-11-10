"""Bounded interval response for positive mixtures of wavelength-scaled pulses.

Each input point has base coordinate x and positive pulse/time-zero multipliers
s,q. Its coordinate is x+q*t0+s*(mean+centred_normal_Laplace), optionally
convolved with a common symmetric Cauchy profile. Input weights are physical
quadrature weights, not counting-noise samples. No measured clock is inferred.

The Fourier cumulative primitive is PERIODIC. Only finite interval masses in
the declared query range are exposed, with a periodic-image bound. In
particular it is not the true absolute CDF of a Cauchy-broadened response.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.fft import rfft,irfft,rfftfreq
from scipy.special import exp1,ndtr,polygamma

VERSION='powgen-positive-scaled-timing-intervals/2'
PARAMETERS=('pulse_mean_us','time_zero_us','sigma_us','tau_left_us','tau_right_us','cauchy_hwhm_us')
PHI_ONE=math.exp(-.5)/math.sqrt(2*math.pi)


def _scalar(value,name):
    if (isinstance(value,(bool,np.bool_,str,bytes,complex,np.complexfloating))
            or not np.isscalar(value) or not math.isfinite(float(value))):
        raise ValueError(name+' must be a finite real scalar')
    return float(value)


@dataclass(frozen=True)
class ScaledTimingBounds:
    pulse_mean_us: tuple
    time_zero_us: tuple
    sigma_us: tuple
    tau_left_us: tuple
    tau_right_us: tuple
    cauchy_hwhm_us: tuple=(0.,0.)

    def __post_init__(self):
        for name in PARAMETERS:
            pair=getattr(self,name)
            if not isinstance(pair,(tuple,list)) or len(pair)!=2:
                raise ValueError('Every timing parameter requires two explicit bounds')
            lo,hi=[_scalar(v,name) for v in pair]
            if hi<lo or (name=='sigma_us' and lo<0) or (name in PARAMETERS[3:] and lo<0):
                raise ValueError('Timing bounds must be ordered and widths physical')
            object.__setattr__(self,name,(lo,hi))
        if self.sigma_us[0]==0 and self.cauchy_hwhm_us[0]==0:
            raise ValueError('A positive Gaussian or Cauchy smoothing lower bound is required')

    def values(self,parameters):
        a=np.asarray(parameters)
        if a.shape!=(6,) or not np.isrealobj(a):
            raise ValueError('Supply all six timing parameters in the documented order')
        a=np.array([_scalar(v,n) for v,n in zip(a,PARAMETERS)])
        if any(not lo<=v<=hi for v,(lo,hi) in zip(a,[getattr(self,n) for n in PARAMETERS])):
            raise ValueError('Timing evaluation leaves the compiled parameter domain')
        return a


def _brackets(values,nodes):
    if len(nodes)==1:
        return np.zeros(len(values),np.int64),np.zeros(len(values))
    i=np.clip(np.searchsorted(nodes,values,side='right')-1,0,len(nodes)-2)
    alpha=(values-nodes[i])/(nodes[i+1]-nodes[i])
    if np.any(alpha<0) or np.any(alpha>1):
        raise ValueError('Interpolation nodes do not cover the input measure')
    return i,alpha


def _adaptive_nodes(values,weights,coefficient,relative_budget,*,inverse_square=False,max_nodes=256):
    lo,hi=float(values.min()),float(values.max())
    totals=weights.sum(axis=0)
    if lo==hi or coefficient==0:
        return np.array([lo]),np.zeros(len(totals))
    nodes=np.array([lo,hi])
    while True:
        i,alpha=_brackets(values,nodes)
        variance=(values-nodes[i])*(nodes[i+1]-values)
        point_error=coefficient*variance/(nodes[i]**2 if inverse_square else 1.)
        per_cell=np.array([np.bincount(i,weights=weights[:,c]*point_error,minlength=len(nodes)-1)
                           for c in range(weights.shape[1])])
        relative=np.divide(per_cell.sum(axis=1),totals,out=np.zeros_like(totals),where=totals>0)
        if relative.max(initial=0)<=relative_budget:
            return nodes,relative
        if len(nodes)>=max_nodes:
            raise ValueError('Scaled-pulse interpolation exceeds its node budget')
        scores=np.divide(per_cell,totals[:,None],out=np.zeros_like(per_cell),where=totals[:,None]>0).max(axis=0)
        j=int(np.argmax(scores));mid=(nodes[j]+nodes[j+1])/2
        if not nodes[j]<mid<nodes[j+1]:
            raise ValueError('Scaled-pulse interpolation cannot be resolved in double arithmetic')
        nodes=np.insert(nodes,j+1,mid)


@dataclass(frozen=True)
class ScaledTimingIntervals:
    mean: np.ndarray
    derivatives: np.ndarray | None
    analytic_absolute_l1_bound: np.ndarray
    declared_roundoff_absolute_allowance: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class ScaledTimingKernel:
    origin_us: float
    step_us: float
    grid_points: int
    scale_nodes: np.ndarray
    zero_nodes: np.ndarray
    spectrum: np.ndarray
    component_totals: np.ndarray
    query_range_us: tuple
    bounds: ScaledTimingBounds
    metadata: dict

    def __post_init__(self):
        if (not isinstance(self.bounds,ScaledTimingBounds) or type(self.grid_points) is not int
                or not 16<=self.grid_points<=2**24 or self.grid_points&(self.grid_points-1)
                or not math.isfinite(self.origin_us) or not math.isfinite(self.step_us) or self.step_us<=0):
            raise ValueError('A finite positive power-of-two timing grid is required')
        for name in ['scale_nodes','zero_nodes','spectrum','component_totals']:
            a=np.array(getattr(self,name),copy=True);a.setflags(write=False);object.__setattr__(self,name,a)
        c,n,k=self.spectrum.shape
        if (n!=len(self.scale_nodes) or n!=len(self.zero_nodes) or c!=len(self.component_totals)
                or not 1<=c<=8 or not 1<=n<=4096 or not 2<=k<=self.grid_points//2+1
                or not np.isfinite(self.spectrum).all() or not np.isfinite(self.scale_nodes+self.zero_nodes).all()
                or np.any(self.scale_nodes<=0) or np.any(self.zero_nodes<0)
                or not np.isfinite(self.component_totals).all() or np.any(self.component_totals<0)):
            raise ValueError('Invalid scaled timing kernel arrays')

    def intervals(self,lower_us,upper_us,parameters,*,derivatives=False):
        """Positive component interval rates, plus optional parameter derivatives.

        Analytic bounds concern these disjoint intervals under the supplied
        point measure and parameter domain. Geometry/QMC error, physical model
        discrepancy and source/reference/metrology uncertainty are separate.
        The floating-point allowance is declared, not formally verified.
        """
        p=self.bounds.values(parameters)
        lower,upper=np.asarray(lower_us,float),np.asarray(upper_us,float)
        if (lower.ndim!=1 or upper.shape!=lower.shape or not 1<=len(lower)<=1000000
                or not np.isfinite(lower+upper).all() or np.any(upper<lower)
                or np.any(lower[1:]<upper[:-1])
                or lower.min()<self.query_range_us[0] or upper.max()>self.query_range_us[1]
                or type(derivatives) is not bool):
            raise ValueError('Ordered disjoint finite intervals inside the compiled query range are required')
        endpoints=2*len(lower)
        if endpoints>self.metadata['maximum_interval_endpoints']:
            raise ValueError('Interval endpoint budget exceeded')
        n=self.grid_points;k=self.spectrum.shape[2]
        frequency=2*np.pi*rfftfreq(n,d=self.step_us)[:k]
        s=self.scale_nodes[:,None];q=self.zero_nodes[:,None];w=frequency[None,:]
        mean,zero,sigma,left,right,gamma=p
        a=w*s
        factor=np.exp(-.5*(sigma*a)**2-gamma*w-1j*w*(q*zero+s*(mean-right+left)))
        factor/=((1+1j*a*right)*(1-1j*a*left))
        source=self.spectrum*factor[None,:,:]
        summed=source.sum(axis=1)
        spectra=[summed]
        if derivatives:
            logarithmic=[-1j*a,-1j*w*q,-sigma*a*a,
                         -a*a*left/(1-1j*a*left),-a*a*right/(1+1j*a*right),-np.broadcast_to(w,a.shape)]
            spectra.extend((source*v[None,:,:]).sum(axis=1) for v in logarithmic)
        endpoints_array=np.r_[lower,upper]
        coordinate=(endpoints_array-self.origin_us)/self.step_us
        idx=np.floor(coordinate).astype(np.int64);fraction=coordinate-idx
        if idx.min()<0 or idx.max()+1>=n:
            raise ValueError('Query endpoints leave the periodic primitive grid')
        t=fraction;t2=t*t;t3=t2*t
        basis=np.array([2*t3-3*t2+1,t3-2*t2+t,-2*t3+3*t2,t3-t2])
        answers=[]
        period=n*self.step_us
        for spectrum in spectra:
            full=np.zeros((len(self.component_totals),n//2+1),complex);full[:,:k]=spectrum
            pdf=irfft(full,n=n,axis=-1)/self.step_us
            primitive=np.zeros_like(full);primitive[:,1:]=full[:,1:]/(1j*(2*np.pi*rfftfreq(n,d=self.step_us))[None,1:])
            cdf=irfft(primitive,n=n,axis=-1)/self.step_us
            dc=full[:,0].real
            cdf=cdf-cdf[:,:1]+dc[:,None]*np.arange(n)[None,:]/n
            values=(basis[0]*cdf[:,idx]+basis[1]*self.step_us*pdf[:,idx]
                    +basis[2]*cdf[:,idx+1]+basis[3]*self.step_us*pdf[:,idx+1])
            answers.append(values[:,len(lower):]-values[:,:len(lower)])
        rates=answers[0];allowance=self.metadata['declared_CDF_roundoff_relative_allowance']
        limits=2*allowance*self.component_totals[:,None]
        if np.any(rates < -limits):
            raise ValueError('Interval reconstruction exceeds its floating-point allowance')
        clipped=np.maximum(-rates,0.).sum(axis=1);rates=np.maximum(rates,0.)
        length=float(np.sum(upper-lower))
        # Only aliasing INTO the finite query intervals is bounded here.
        alias=self.metadata['non_Cauchy_tail_probability_bound']
        if gamma>0:
            a0=self.metadata['maximum_query_minus_central_support_distance_us']/period
            alias+=length*2*gamma/(np.pi*period*period)*float(polygamma(1,1-a0))
        relative=(np.array(self.metadata['scale_interpolation_relative_l1_bounds'])
                  +np.array(self.metadata['zero_interpolation_relative_l1_bounds'])
                  +np.array(self.metadata['position_deposition_relative_l1_bounds'])
                  +alias+endpoints*(self.metadata['Hermite_CDF_absolute_error_bound']
                                    +self.metadata['spectral_CDF_truncation_bound']))
        analytic=relative*self.component_totals
        roundoff=2*endpoints*allowance*self.component_totals
        return ScaledTimingIntervals(rates,np.stack(answers[1:],axis=1) if derivatives else None,
                                      analytic,roundoff,dict(version=VERSION,parameters=dict(zip(PARAMETERS,p.tolist())),
                                      clipped_negative_mass=clipped.tolist(),periodic_image_relative_l1_bound=alias,
                                      absolute_CDF_not_supplied=True,parameter_derivative_error_not_certified=True,
                                      geometry_QMC_and_physical_error_excluded=True,experimental_accuracy_certified=False))

    def parameter_vjp(self,lower_us,upper_us,parameters,rate_gradient):
        """Exact adjoint of the declared periodic/Hermite interval calculation.

        Return (component, six parameters) gradients for supplied derivatives
        with respect to component interval rates. This avoids six additional
        inverse transforms per component during a reference fit. It is an
        algorithmic derivative, with no physical-uncertainty claim. As with the
        full Jacobian, floating clipping at negligible negative rates is not
        differentiated.
        """
        p=self.bounds.values(parameters)
        lower,upper=np.asarray(lower_us,float),np.asarray(upper_us,float)
        g=np.asarray(rate_gradient,float)
        c=len(self.component_totals)
        if (lower.ndim!=1 or upper.shape!=lower.shape or not 1<=len(lower)<=1000000
                or g.shape!=(c,len(lower)) or not np.isfinite(g).all()
                or not np.isfinite(lower+upper).all() or np.any(upper<lower)
                or np.any(lower[1:]<upper[:-1]) or lower.min()<self.query_range_us[0]
                or upper.max()>self.query_range_us[1] or 2*len(lower)>self.metadata['maximum_interval_endpoints']):
            raise ValueError('A finite component-rate gradient on valid disjoint intervals is required')
        n=self.grid_points;k=self.spectrum.shape[2];period=n*self.step_us
        frequency=2*np.pi*rfftfreq(n,d=self.step_us)[:k]
        endpoint=np.r_[lower,upper];coord=(endpoint-self.origin_us)/self.step_us
        idx=np.floor(coord).astype(np.int64);t=coord-idx
        if idx.min()<0 or idx.max()+1>=n:raise ValueError('Adjoint endpoints leave the timing grid')
        t2=t*t;t3=t2*t;cg=np.c_[-g,g]
        gcdf=np.zeros((c,n));gpdf=np.zeros((c,n))
        for j in range(c):
            np.add.at(gcdf[j],idx,cg[j]*(2*t3-3*t2+1))
            np.add.at(gcdf[j],idx+1,cg[j]*(-2*t3+3*t2))
            np.add.at(gpdf[j],idx,cg[j]*(t3-2*t2+t)*self.step_us)
            np.add.at(gpdf[j],idx+1,cg[j]*(t3-t2)*self.step_us)
            gcdf[j,0]-=gcdf[j].sum()  # derivative of the cumulative-origin subtraction
        adjoint=np.conj(rfft(gpdf,axis=-1)[:,:k])
        adjoint[:,1:]+=np.conj(rfft(gcdf,axis=-1)[:,1:k])/(1j*frequency[None,1:])
        adjoint*=2/period;adjoint[:,0]=0. # pulse parameters preserve the total mass
        if k==n//2+1:adjoint[:,-1]*=.5
        s=self.scale_nodes[:,None];q=self.zero_nodes[:,None];w=frequency[None,:];a=w*s
        mean,zero,sigma,left,right,gamma=p
        factor=np.exp(-.5*(sigma*a)**2-gamma*w-1j*w*(q*zero+s*(mean-right+left)))
        factor/=((1+1j*a*right)*(1-1j*a*left))
        source=self.spectrum*factor[None,:,:]
        logarithmic=[-1j*a,-1j*w*q,-sigma*a*a,
                     -a*a*left/(1-1j*a*left),-a*a*right/(1+1j*a*right),-np.broadcast_to(w,a.shape)]
        return np.array([np.real(np.sum((source*v[None,:,:]).sum(axis=1)*adjoint,axis=1))
                         for v in logarithmic]).T


def compile_scaled_timing(positions_us,scales,zero_multipliers,weights,*,query_range_us,bounds,
                          maximum_interval_endpoints,provenance_label,
                          interpolation_budget=.0001,position_budget=.00005,alias_budget=.0001,
                          max_points=2**22,max_nodes=256,max_grid_points=2**21,max_bytes=512*1024**2):
    """Compile a reusable positive kernel over an explicit parameter rectangle.

    Scale interpolation uses an L1 second-derivative envelope for the scaled
    Gaussian/exponential family. The envelope is at most
    [10+2sqrt(6)*sqrt(EU²)/sigma+4phi(1)*EU²/sigma²]/s²,
    where U=mean+Exp(right)-Exp(left)-(right-left). Tensor interpolation in q
    and x uses the Gaussian second-derivative L1 norm; convolution cannot
    increase these norms. Adaptive nodes use input weights only, never data.
    """
    if (not isinstance(bounds,ScaledTimingBounds) or type(maximum_interval_endpoints) is not int
            or not 2<=maximum_interval_endpoints<=2000000 or type(max_points) is not int
            or not 1<=max_points<=2**22 or type(max_nodes) is not int or not 2<=max_nodes<=1024
            or type(max_grid_points) is not int or not 16<=max_grid_points<=2**24
            or type(max_bytes) is not int or not 1024<=max_bytes<=2*1024**3
            or type(provenance_label) is not str or not provenance_label.strip() or len(provenance_label)>1024):
        raise ValueError('Explicit bounded compile resources, parameter bounds and provenance are required')
    budgets=[_scalar(v,'Approximation budget') for v in [interpolation_budget,position_budget,alias_budget]]
    if any(not 0<v<.1 for v in budgets):raise ValueError('Positive approximation budgets below0.1 are required')
    arrays=[np.asarray(v) for v in [positions_us,scales,zero_multipliers,weights]]
    x,s,q,w=arrays
    if w.ndim==1:w=w[:,None]
    if (x.ndim!=1 or not 1<=len(x)<=max_points or s.shape!=x.shape or q.shape!=x.shape
            or w.ndim!=2 or w.shape[0]!=len(x) or not 1<=w.shape[1]<=8
            or any(not np.isrealobj(a) for a in [x,s,q,w])):
        raise ValueError('Bounded real point and positive component-weight arrays are required')
    x,s,q,w=[np.array(v,float,copy=True) for v in [x,s,q,w]]
    if (any(not np.isfinite(v).all() for v in [x,s,q,w]) or np.any(s<=0) or np.any(q<0)
            or np.any(w<0) or not np.any(w>0)):
        raise ValueError('Finite points, positive scales and nonnegative rates/clock multipliers are required')
    if x.size*(128+16*w.shape[1])>max_bytes:raise ValueError('Point measure exceeds compile memory budget')
    roi=np.asarray(query_range_us,float)
    if roi.shape!=(2,) or not np.isfinite(roi).all() or roi[1]<=roi[0]:raise ValueError('A finite ordered query range is required')
    totals=w.sum(axis=0)
    if not np.isfinite(totals).all() or totals.sum()<=0:raise ValueError('Finite positive total rates are required')
    sigmin,sigmax=bounds.sigma_us;smin,smax=float(s.min()),float(s.max())
    meanmax=max(abs(v) for v in bounds.pulse_mean_us);zeromax=max(abs(v) for v in bounds.time_zero_us)
    leftmax=bounds.tau_left_us[1];rightmax=bounds.tau_right_us[1]
    second=leftmax*leftmax+rightmax*rightmax+meanmax*meanmax
    if sigmin>0:
        curvature=10+2*np.sqrt(6)*np.sqrt(second)/sigmin+4*PHI_ONE*second/(sigmin*sigmin)
        snodes,serror=_adaptive_nodes(s,w,.5*curvature,interpolation_budget,inverse_square=True,max_nodes=max_nodes)
        qcoefficient=2*PHI_ONE*zeromax**2/(sigmin*sigmin*smin*smin)
        qnodes,qerror=_adaptive_nodes(q,w,qcoefficient,interpolation_budget,max_nodes=max_nodes)
        si,sa=_brackets(s,snodes);qi,qa=_brackets(q,qnodes)
        step=sigmin*smin*np.sqrt(2*position_budget/PHI_ONE)
        smoothing='positive Gaussian lower bound'
    else:
        # The Cauchy smoothing belongs to the supplied conditional specimen
        # model. It is not an independently measured instrumental width.
        gamma_min=bounds.cauchy_hwhm_us[0]
        cauchy_d2=3*np.sqrt(3)/(2*np.pi*gamma_min*gamma_min)
        curvature=cauchy_d2*(second+sigmax*sigmax)
        snodes,serror=_adaptive_nodes(s,w,.5*curvature,interpolation_budget,max_nodes=max_nodes)
        qcoefficient=.5*cauchy_d2*zeromax*zeromax
        qnodes,qerror=_adaptive_nodes(q,w,qcoefficient,interpolation_budget,max_nodes=max_nodes)
        si,sa=_brackets(s,snodes);qi,qa=_brackets(q,qnodes)
        step=np.sqrt(8*position_budget/cauchy_d2)
        smoothing='strictly positive conditional Cauchy lower bound'
    meanlo=bounds.pulse_mean_us[0]-rightmax+bounds.tau_left_us[0]
    meanhi=bounds.pulse_mean_us[1]-bounds.tau_right_us[0]+leftmax
    meanlimits=[ss*mm for ss in [smin,smax] for mm in [meanlo,meanhi]]
    zerolimits=[qq*tt for qq in [float(q.min()),float(q.max())] for tt in bounds.time_zero_us]
    # Include the full x-deposition support, not just the original points.
    zlo=float(x.min()+min(meanlimits)+min(zerolimits)-9*sigmax*smax-28*leftmax*smax-step)
    zhi=float(x.max()+max(meanlimits)+max(zerolimits)+9*sigmax*smax+28*rightmax*smax+step)
    tail=float(2*ndtr(-9)+(np.exp(-28) if leftmax else 0)+(np.exp(-28) if rightmax else 0))
    origin=min(zlo,float(roi[0]),float(x.min()))-2*step
    stop=max(zhi,float(roi[1]),float(x.max()))+2*step
    distance=max(abs(float(roi[1])-zlo),abs(zhi-float(roi[0])))
    n=1<<max(4,int(math.ceil(math.log2(max(16,(stop-origin)/step)))) )
    gamma=bounds.cauchy_hwhm_us[1]
    while True:
        period=n*step
        image=(np.inf if period<=distance else tail+(roi[1]-roi[0])*2*gamma/(np.pi*period**2)*float(polygamma(1,1-distance/period)))
        if image<=alias_budget:break
        n*=2
        if n>max_grid_points:raise ValueError('Cauchy-image/period budget exceeds the timing grid limit')
    if n>max_grid_points:raise ValueError('Timing grid exceeds compiled resource limits')
    frequency=2*np.pi*rfftfreq(n,d=step)
    if sigmin>0:
        keep=min(len(frequency),max(2,int(np.searchsorted(frequency,8/(sigmin*smin),side='right'))))
        first_omitted=2*np.pi*keep/(n*step)
        exponent=.5*(sigmin*smin*first_omitted)**2
        truncation_cdf=float(2/(n*step)*np.exp(-exponent)/first_omitted+exp1(exponent)/(2*np.pi))
        truncation_pdf=float(2/(n*step)*np.exp(-exponent)
                             +np.sqrt(2/np.pi)/(sigmin*smin)*ndtr(-sigmin*smin*first_omitted))
        hermite=2.4/384*(step/(sigmin*smin))**4
    else:
        keep=min(len(frequency),max(2,int(np.searchsorted(frequency,32/gamma_min,side='right'))))
        first_omitted=2*np.pi*(keep-1 if keep==len(frequency) else keep)/(n*step)
        damping=np.exp(-gamma_min*first_omitted)
        truncation_cdf=float(2/(n*step)*damping/first_omitted+exp1(gamma_min*first_omitted)/np.pi)
        truncation_pdf=float(2/(n*step)*damping+damping/(np.pi*gamma_min))
        hermite=step**4/384*6/(np.pi*gamma_min**4)
    # Hermite endpoint interpolation also uses the PDF; its two derivative
    # basis magnitudes sum to at most1/4, so include that truncation term.
    truncation=truncation_cdf+step*truncation_pdf/4
    keys=[];pointweights=[]
    # Each point contributes to up to four positive scale/time-zero nodes.
    for ds in range(1 if len(snodes)==1 else 2):
        for dq in range(1 if len(qnodes)==1 else 2):
            fraction=(sa if ds else 1-sa)*(qa if dq else 1-qa)
            if len(snodes)==1:fraction=(qa if dq else 1-qa)
            if len(qnodes)==1:fraction=(sa if ds else 1-sa)
            if len(snodes)==len(qnodes)==1:fraction=np.ones(len(x))
            keys.append((si+ds)*len(qnodes)+(qi+dq));pointweights.append(fraction)
    unique=np.unique(np.concatenate([key[f>0] for key,f in zip(keys,pointweights)]))
    if len(unique)>4096:raise ValueError('Positive tensor-node budget exceeded')
    # Ownership copies, evaluation products, the six derivative factors and
    # FFT work arrays are included, in addition to the retained spectrum.
    required=(3*w.shape[1]*len(unique)*keep*16+10*len(unique)*keep*16
              +n*(96+128*w.shape[1])+len(x)*(192+32*w.shape[1]))
    if required>max_bytes:raise ValueError('Scaled timing spectrum exceeds the compile memory budget')
    coordinates=(x-origin)/step;xi=np.floor(coordinates).astype(np.int64);xa=coordinates-xi
    if xi.min()<0 or xi.max()+1>=n:raise ValueError('Point deposition leaves its grid')
    spectrum=np.empty((w.shape[1],len(unique),keep),complex)
    for j,node in enumerate(unique):
        fraction=np.zeros(len(x))
        for key,f in zip(keys,pointweights):fraction+=np.where(key==node,f,0.)
        active=fraction>0
        for c in range(w.shape[1]):
            weight=w[active,c]*fraction[active]
            deposited=np.bincount(xi[active],weights=weight*(1-xa[active]),minlength=n)
            deposited+=np.bincount(xi[active]+1,weights=weight*xa[active],minlength=n)
            spectrum[c,j]=rfft(deposited)[:keep]
    variance=(step*xa)*(step*(1-xa))
    if sigmin>0:
        perror=np.array([2*PHI_ONE*float(np.sum(w[:,c]*variance))/(sigmin*sigmin*smin*smin)/totals[c] if totals[c]>0 else 0. for c in range(w.shape[1])])
    else:
        perror=np.array([.5*cauchy_d2*float(np.sum(w[:,c]*variance))/totals[c] if totals[c]>0 else 0. for c in range(w.shape[1])])
    allowance=float(1024*np.finfo(float).eps*(1+math.log2(n))**2*(1+math.log2(max(1,len(unique)))))
    drift=np.abs(spectrum[:,:,0].real.sum(axis=1)-totals)
    if np.any(drift>allowance*totals):raise ValueError('Positive deposition mass exceeds its floating allowance')
    metadata=dict(version=VERSION,smoothing_bound=smoothing,provenance_label=provenance_label,points=len(x),components=w.shape[1],
                  tensor_nodes=len(unique),scale_grid_nodes=len(snodes),zero_grid_nodes=len(qnodes),
                  grid_points=n,stored_frequency_points=keep,estimated_compile_array_bytes=required,
                  maximum_interval_endpoints=maximum_interval_endpoints,
                  scale_interpolation_relative_l1_bounds=serror.tolist(),zero_interpolation_relative_l1_bounds=qerror.tolist(),
                  position_deposition_relative_l1_bounds=perror.tolist(),scale_family_curvature_coefficient=float(curvature),
                  maximum_query_minus_central_support_distance_us=distance,non_Cauchy_tail_probability_bound=tail,
                  maximum_finite_range_periodic_image_bound=float(image),
                  Hermite_CDF_absolute_error_bound=float(hermite),
                  spectral_CDF_truncation_bound=truncation,declared_CDF_roundoff_relative_allowance=allowance,
                  floating_allowance_formally_verified=False,input_rates_renormalized=False,
                  geometry_QMC_physical_and_parameter_derivative_errors_excluded=True,experimental_accuracy_certified=False)
    return ScaledTimingKernel(origin,float(step),n,snodes[unique//len(qnodes)],qnodes[unique%len(qnodes)],
                              spectrum,totals,tuple(float(v) for v in roi),bounds,metadata)
