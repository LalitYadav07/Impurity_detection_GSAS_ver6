"""Continuous TOF probability, integrated through the actual observation cuts.

The normal/Laplace law is a normalized back-to-back exponential convolved
with a Gaussian. It is a possible effective timing law, not a calibrated
POWGEN moderator model. A fitted focused profile must not be added to the
same broadening a second time. Lorentzian/specimen broadening is not supplied
by this law. CDF differences, rather than peak heights or point samples,
preserve line area and report the probability lost to cuts and histogram edges.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.special import erfcx, log_ndtr, ndtr
from .energy_response import ReductionWindow
from .energy_response import apparent_d, TOF_US_PER_M_A
from .elastic_powder import PowderLineResponse, _lines, _edges, _factors, _positive
from scipy import sparse

VERSION = 'powgen-continuous-timing-bin-response/1'


def _parameters(x, alpha, beta, sigma):
    arrays = np.broadcast_arrays(*[np.asarray(v, float) for v in (x, alpha, beta, sigma)])
    x, alpha, beta, sigma = arrays
    if (x.size > 4000000 or np.isnan(x).any() or any(not np.isfinite(v).all() for v in arrays[1:])
            or np.any(alpha <= 0) or np.any(beta <= 0) or np.any(sigma < 0)
            or not np.isfinite(alpha*sigma).all() or not np.isfinite(beta*sigma).all()):
        raise ValueError('Finite positive rates and nonnegative Gaussian widths are required within budget')
    scale = np.maximum(alpha, beta)
    left = (beta/scale)/(alpha/scale+beta/scale)
    return x, alpha, beta, sigma, left


def _positive_exponential_normal(x, rate, sigma):
    """CDF/SF of N(0,sigma²)+Exp(rate), including a zero Gaussian width.

    The erfcx branch avoids the cancellation of rate²*sigma² against a normal
    log-CDF in the large rate*sigma limit. The survival function is evaluated
    directly so a small tail is not computed as 1-rounded(CDF).
    """
    cdf, sf = np.empty(x.shape), np.empty(x.shape)
    zero = sigma == 0
    neg = zero & (x < 0)
    cdf[neg], sf[neg] = 0., 1.
    pos = zero & ~neg
    cdf[pos] = -np.expm1(-rate[pos]*x[pos])
    sf[pos] = np.exp(-rate[pos]*x[pos])
    active = ~zero & np.isfinite(x)
    z, a = x[active]/sigma[active], rate[active]*sigma[active]
    log_term = np.empty(z.shape)
    negative_argument = z < a
    with np.errstate(over='ignore', under='ignore', invalid='raise'):
        log_term[negative_argument] = (-.5*z[negative_argument]**2
            + np.log(.5*erfcx((a[negative_argument]-z[negative_argument])/np.sqrt(2))))
        other = ~negative_argument
        log_term[other] = -a[other]*(z[other]-.5*a[other])+log_ndtr(z[other]-a[other])
        log_normal = log_ndtr(z)
        normal_probability = np.exp(log_normal)
        # In a remote Gaussian tail both logarithms can be very large and
        # negative. Their subtraction can lose its sign even though the
        # normal probability, and hence this CDF, already underflows to zero.
        # Do not form that subtraction when exp(log_normal) is exactly zero.
        resolved = normal_probability > 0
        difference = log_term[resolved]-log_normal[resolved]
        if np.any(difference > 2e-12):
            raise ValueError('Normal/exponential CDF cancellation exceeds the numerical domain')
        value = np.zeros(z.shape)
        value[resolved] = normal_probability[resolved]*(-np.expm1(np.minimum(difference, 0.)))
        cdf[active] = value
        sf[active] = ndtr(-z)+np.exp(log_term)
    cdf[np.isneginf(x)], sf[np.isneginf(x)] = 0., 1.
    cdf[np.isposinf(x)], sf[np.isposinf(x)] = 1., 0.
    return cdf, sf


def normal_laplace_probability(x_us, alpha_per_us, beta_per_us, sigma_us):
    """Return CDF and survival probability about the exponential joining point.

    The mean offset is 1/beta-1/alpha, generally different from the joining
    point and the mode. Parameters broadcast; +/-infinity queries are allowed.
    Rates use 1/us and sigma uses us. No truncation or renormalization occurs.
    """
    x, alpha, beta, sigma, left = _parameters(x_us, alpha_per_us, beta_per_us, sigma_us)
    positive_cdf, positive_sf = _positive_exponential_normal(x, beta, sigma)
    negative_cdf, negative_sf = _positive_exponential_normal(-x, alpha, sigma)
    return (left*negative_sf+(1-left)*positive_cdf,
            left*negative_cdf+(1-left)*positive_sf)


def centred_normal_laplace_probability(x_us, sigma_us, tau_left_us, tau_right_us):
    """CDF/SF about the mean, including exactly zero exponential widths.

    The random variable is N(0,sigma²)+Exp(tau_right)-Exp(tau_left)
    -(tau_right-tau_left). A zero exponential width is a point mass at zero;
    if every width vanishes the returned CDF is right-continuous at zero.
    This is an effective timing law, not a measured moderator distribution.
    """
    x,sigma,left,right=np.broadcast_arrays(*[np.asarray(v,float) for v in
        (x_us,sigma_us,tau_left_us,tau_right_us)])
    if (x.size>4000000 or np.isnan(x).any()
            or any(not np.isfinite(v).all() or np.any(v<0) for v in (sigma,left,right))):
        raise ValueError('Nonnegative finite timing widths and bounded non-NaN queries are required')
    cdf,sf=np.empty(x.shape),np.empty(x.shape)
    both=(left>0)&(right>0)
    if np.any(both):
        cdf[both],sf[both]=normal_laplace_probability(x[both]+right[both]-left[both],
            1/left[both],1/right[both],sigma[both])
    positive=(left==0)&(right>0)
    if np.any(positive):
        cdf[positive],sf[positive]=_positive_exponential_normal(x[positive]+right[positive],
            1/right[positive],sigma[positive])
    negative=(left>0)&(right==0)
    if np.any(negative):
        sf[negative],cdf[negative]=_positive_exponential_normal(left[negative]-x[negative],
            1/left[negative],sigma[negative])
    gaussian=(left==0)&(right==0)&(sigma>0)
    cdf[gaussian]=ndtr(x[gaussian]/sigma[gaussian]);sf[gaussian]=ndtr(-x[gaussian]/sigma[gaussian])
    point=(left==0)&(right==0)&(sigma==0)
    cdf[point]=(x[point]>=0).astype(float);sf[point]=(x[point]<0).astype(float)
    return cdf,sf


def interval_probability(lower_us, upper_us, probability):
    """Integrate a supplied continuous law using its independently evaluated SF.

    probability(offsets) returns matching CDF and SF arrays. Its parameter
    provenance/calibration belongs to the caller. Invalid or inconsistent
    probabilities raise rather than being repaired or normalized.
    """
    lower, upper = np.broadcast_arrays(np.asarray(lower_us, float), np.asarray(upper_us, float))
    if lower.size > 2000000 or np.isnan(lower).any() or np.isnan(upper).any() or np.any(upper < lower) or not callable(probability):
        raise ValueError('Ordered interval endpoints and an explicit probability law are required within budget')
    endpoints = np.stack((lower, upper))
    cdf, sf = [np.asarray(v, float) for v in probability(endpoints)]
    if (cdf.shape != endpoints.shape or sf.shape != endpoints.shape
            or any(not np.isfinite(v).all() or np.any(v < 0) or np.any(v > 1) for v in (cdf, sf))
            or not np.allclose(cdf+sf, 1., rtol=0., atol=2e-12)):
        raise ValueError('The timing law must provide consistent physical CDF/SF arrays')
    mass = np.where(cdf[0] < .5, cdf[1]-cdf[0], sf[0]-sf[1])
    if np.any(mass < -2e-14):
        raise ValueError('The supplied timing law is not monotonic on the requested intervals')
    return np.maximum(mass, 0.)


@dataclass(frozen=True)
class ArrivalBins:
    probability: np.ndarray
    unobserved_probability: float
    metadata: dict


def binned_arrival_probability(d_edges_a, center_tof_us, probability, *, difc_us_per_a,
        cos_two_theta, difa_us_per_a2=0., tzero_us=0., window=None,
        timing_law_label, max_intervals=200000):
    """One line's continuous arrival law after diffraction calibration and cuts.

    Raw TOF = DIFC*d + DIFA*d² + TZERO. The supplied center already comes from
    the declared physical or calibrated mean clock; the probability callback
    describes offsets about it. Cuts are applied inside each bin before CDF
    integration, including every intersecting prompt-veto interval. Lost mass
    includes tails outside the histogram and every cut, without redistribution.

    This is a deterministic probability operator. Counting noise, source
    fluence, line factors, physical broadening and their uncertainty are separate.
    """
    edges = np.asarray(d_edges_a, float)
    values = (center_tof_us, difc_us_per_a, cos_two_theta, difa_us_per_a2, tzero_us)
    if (edges.ndim != 1 or not 2 <= len(edges) <= 100001 or not np.isfinite(edges).all()
            or edges[0] <= 0 or np.any(np.diff(edges) <= 0)
            or any(isinstance(v, (bool, np.bool_, str, bytes)) or not np.isscalar(v)
                or not math.isfinite(float(v)) for v in values)
            or difc_us_per_a <= 0 or not -1 <= cos_two_theta < 1
            or not np.all(difc_us_per_a+2*difa_us_per_a2*edges > 0)
            or type(timing_law_label) is not str or not timing_law_label.strip()
            or len(timing_law_label) > 512 or type(max_intervals) is not int
            or not 1 <= max_intervals <= 2000000 or not callable(probability)):
        raise ValueError('Explicit finite monotonic calibration, physical bins and timing provenance are required')
    window = ReductionWindow() if window is None else window
    if not isinstance(window, ReductionWindow):
        raise ValueError('An explicit ReductionWindow is required')
    lower, upper = edges[:-1].copy(), edges[1:].copy()
    if window.apparent_wavelength_range_a is not None:
        sine2 = np.sqrt(2*(1-cos_two_theta))
        low, high = np.asarray(window.apparent_wavelength_range_a)/sine2
        lower, upper = np.maximum(lower, low), np.minimum(upper, high)
    useful = upper > lower
    index = np.flatnonzero(useful)
    if len(index) > max_intervals:
        raise ValueError('Timing integration exceeds its declared interval budget')
    lo, hi = lower[useful], upper[useful]
    lo = difc_us_per_a*lo+difa_us_per_a2*lo**2+tzero_us
    hi = difc_us_per_a*hi+difa_us_per_a2*hi**2+tzero_us
    lo = np.maximum(lo, 0.)
    if window.tof_range_us is not None:
        lo, hi = np.maximum(lo, window.tof_range_us[0]), np.minimum(hi, window.tof_range_us[1])
    useful = hi > lo
    index, lo, hi = index[useful], lo[useful], hi[useful]
    # Integrate the allowed pieces directly. Subtracting two nearly equal
    # probabilities would erase a narrow surviving interval next to a veto.
    part_index, part_lo, part_hi = [], [], []
    veto_count = 0
    period, width = window.prompt_period_us, window.prompt_width_us
    for j, (start, stop) in enumerate(zip(lo, hi)):
        cursor = start
        if period is not None and width > 0:
            first = max(0, math.ceil((float(start)-width)/period))
            last = math.floor(float(stop)/period)
            if len(index)+veto_count+max(0, last-first+1) > max_intervals:
                raise ValueError('Prompt-veto integration exceeds its declared interval budget')
            for cycle in range(first, last+1):
                left, right = max(start, cycle*period), min(stop, cycle*period+width)
                if right > left:
                    if left > cursor:
                        part_index.append(j);part_lo.append(cursor);part_hi.append(left)
                    cursor = max(cursor, right);veto_count += 1
        if stop > cursor:
            part_index.append(j);part_lo.append(cursor);part_hi.append(stop)
    normalization = interval_probability(-np.inf, np.inf, probability)
    if abs(float(normalization)-1.) > 2e-12:
        raise ValueError('A normalized continuous timing CDF/SF is required')
    pieces = interval_probability(np.asarray(part_lo)-center_tof_us, np.asarray(part_hi)-center_tof_us, probability)
    mass = np.bincount(part_index, weights=pieces, minlength=len(index))
    if np.any(mass < -2e-12) or mass.sum() > 1+2e-12:
        raise ValueError('Integrated timing probability violates conservation')
    result = np.zeros(len(edges)-1);result[index] = np.maximum(mass, 0.)
    lost = max(0., 1-float(result.sum()))
    result.setflags(write=False)
    return ArrivalBins(result, lost, dict(version=VERSION, timing_law_label=timing_law_label,
        integrated_bin_intervals=len(part_index), integrated_prompt_intersections=veto_count,
        missing_or_cut_probability_renormalized=False, experimental_accuracy_certified=False))


def build_continuous_powder_line_response(d_a, multiplicity, cell_volume_a3, factor_squared_b,
        source_edges_a, d_edges_a, *, cos_two_theta, solid_angle_sr, observed_group,
        l1_m, l2_m, path_weight, detector_efficiency, timing_probability, mean_delay_us,
        timing_law_label, source_nodes_a=None, difc_ratio=1., difa_us_per_a2=0., tzero_us=0.,
        clock_model='physical', window=None, tof_us_per_m_a=TOF_US_PER_M_A,
        max_bytes=512*1024**2, max_line_detector_pairs=2000000, max_probability_intervals=2000000):
    """Physical elastic line counts with continuous recorded-time probability.

    Same cell/factor/source/path units as build_powder_line_response. The
    callback timing_probability(cell, scalar_wavelength, offsets_array) returns
    CDF and SF for the full, normalized offset law. Integration follows each
    detector's diffraction calibration, prompt veto and reduction window.
    Source coefficients remain at the incident elastic wavelength; clock
    broadening is not a change in the neutron's energy or an independent
    integration over the incident spectrum. Missing source coverage is reported.

    This bounded reference operator intentionally evaluates every output-bin
    interval for each nonzero line/detector pair. Large packs require separately
    checked sparse/local kernels. Resource exhaustion never silently drops tails.
    """
    d, mult, volume = _lines(d_a, multiplicity, cell_volume_a3)
    source, output = _edges(source_edges_a, 'Source edges'), _edges(d_edges_a, 'Observed d edges')
    nodes = None if source_nodes_a is None else _edges(source_nodes_a, 'Source nodes')
    if (len(source)>4098 or len(output)>100001 or (nodes is not None and
            (len(nodes)>4097 or nodes[0]!=source[0] or nodes[-1]!=source[-1]))):
        raise ValueError('Bounded density nodes must span the declared source domain')
    cosine, omega = np.asarray(cos_two_theta,float), np.asarray(solid_angle_sr,float)
    group = np.asarray(observed_group)
    if (cosine.ndim!=1 or not 1<=len(cosine)<=100000 or omega.shape!=cosine.shape
            or not np.isfinite(cosine).all() or np.any(cosine < -1) or np.any(cosine >= 1)
            or not np.isfinite(omega).all() or np.any(omega <= 0) or omega.sum()>4*np.pi*(1+1e-14)
            or group.shape!=cosine.shape or group.dtype.kind not in 'iu' or np.any(group<0)
            or group.max()>=2048 or not np.array_equal(np.unique(group),np.arange(group.max()+1))):
        raise ValueError('Physical detector geometry and contiguous bounded groups are required')
    ngroups, nbins = int(group.max()+1), len(output)-1
    nsource = len(source)-1 if nodes is None else len(nodes)
    if (type(max_bytes) is not int or not 1024<=max_bytes<=2*1024**3
            or type(max_line_detector_pairs) is not int or max_line_detector_pairs<1
            or len(d)*len(cosine)>max_line_detector_pairs or ngroups*nbins>1000000
            or type(max_probability_intervals) is not int or not 1<=max_probability_intervals<=20000000
            or max_probability_intervals*128+ngroups*nsource*32+ngroups*nbins*8+len(d)*128>max_bytes):
        raise ValueError('Continuous line response exceeds its explicit resource budget')
    if not all(callable(v) for v in (factor_squared_b,path_weight,detector_efficiency,timing_probability,mean_delay_us)):
        raise ValueError('Explicit factor, path, efficiency, delay and continuous timing laws are required')
    if type(timing_law_label) is not str or not timing_law_label.strip() or len(timing_law_label)>512:
        raise ValueError('Explicit bounded timing-law provenance is required')
    l1, k = _positive(l1_m,'Incident flight path'), _positive(tof_us_per_m_a,'TOF conversion')
    calibration=[]
    for value,positive in [(l2_m,True),(difc_ratio,True),(difa_us_per_a2,False),(tzero_us,False)]:
        item=np.broadcast_to(np.asarray(value,float),cosine.shape)
        if not np.isfinite(item).all() or (positive and np.any(item<=0)):
            raise ValueError('Matching finite physical paths and calibration are required')
        calibration.append(item)
    l2, ratio, difa, zero=calibration
    window=ReductionWindow() if window is None else window
    if not isinstance(window,ReductionWindow):raise ValueError('An explicit ReductionWindow is required')
    apparent_d([1.],[1.],[0.],l1_m=l1,l2_m=1.,clock_model=clock_model,tof_us_per_m_a=k)
    storage=[[],[],[]];lost=np.zeros((ngroups,nsource));total=np.zeros_like(lost)
    uncovered=0;outside_min,outside_max=math.inf,-math.inf;used=0
    for cell in range(len(cosine)):
        sine=np.sqrt((1-cosine[cell])/2);wave=2*d*sine
        inside=(wave>=source[0])&(wave<=source[-1])
        if (~inside).any():
            uncovered+=int((~inside).sum());outside_min=min(outside_min,float(wave[~inside].min()))
            outside_max=max(outside_max,float(wave[~inside].max()))
        indices=np.flatnonzero(inside);wave=wave[inside]
        if not len(wave):continue
        factors=_factors(factor_squared_b,indices,wave)
        path,eff,delay=[np.asarray(f(cell,wave),float) for f in (path_weight,detector_efficiency,mean_delay_us)]
        if (any(v.shape!=wave.shape or not np.isfinite(v).all() for v in (path,eff,delay))
                or any(np.any(v<0) or np.any(v>1) for v in (path,eff))):
            raise ValueError('Matching physical path/efficiency and finite delay arrays are required')
        weight=mult[indices]*factors*d[indices]**3*wave/(4*np.pi*volume)*omega[cell]*path*eff
        if not np.isfinite(weight).all():raise ValueError('Physical line areas exceed their finite numerical domain')
        _,arrival=apparent_d(wave,wave,cosine[cell],l1_m=l1,l2_m=l2[cell],difc_ratio=ratio[cell],
            difa_us_per_a2=difa[cell],tzero_us=zero[cell],delay_us=delay,clock_model=clock_model,tof_us_per_m_a=k)
        basis=source if nodes is None else nodes
        left=np.minimum(np.searchsorted(basis,wave,side='right')-1,len(basis)-2)
        for j in np.flatnonzero(weight>0):
            used+=nbins
            if used>max_probability_intervals:raise ValueError('Continuous timing integration exceeds its declared interval budget')
            # Bind the scalar wavelength and cell immediately; no callback is
            # retained across iterations and no timing probability is sampled.
            law=lambda offsets:timing_probability(cell,float(wave[j]),offsets)
            result=binned_arrival_probability(output,float(arrival[j]),law,
                difc_us_per_a=2*k*(l1+l2[cell])*sine*ratio[cell],cos_two_theta=float(cosine[cell]),
                difa_us_per_a2=float(difa[cell]),tzero_us=float(zero[cell]),window=window,
                timing_law_label=timing_law_label,max_intervals=max_probability_intervals-used+nbins)
            used+=result.metadata['integrated_prompt_intersections']
            if used>max_probability_intervals:raise ValueError('Continuous timing integration exceeds its declared interval budget')
            columns=[int(left[j])];fractions=[1.]
            if nodes is not None:
                fraction=(wave[j]-nodes[left[j]])/(nodes[left[j]+1]-nodes[left[j]])
                columns=[int(left[j]),int(left[j]+1)];fractions=[1-fraction,fraction]
            active=np.flatnonzero(result.probability>0)
            for column,fraction in zip(columns,fractions):
                value=float(weight[j]*fraction)
                if value==0:continue
                total[group[cell],column]+=value
                lost[group[cell],column]+=value*result.unobserved_probability
                storage[0].append(group[cell]*nbins+active)
                storage[1].append(np.full(len(active),column,dtype=np.int64))
                storage[2].append(value*result.probability[active])
    arrays=[np.concatenate(v) if v else np.array([],float) for v in storage]
    matrix=sparse.coo_matrix((arrays[2],(arrays[0].astype(np.int64),arrays[1].astype(np.int64))),
        shape=(ngroups*nbins,nsource)).tocsr()
    matrices=[matrix,sparse.csr_matrix(lost),sparse.csr_matrix(total)]
    for item in matrices:
        item.eliminate_zeros()
        if not np.isfinite(item.data).all():raise ValueError('Accumulated continuous response exceeds its finite domain')
        for array in (item.data,item.indices,item.indptr):array.setflags(write=False)
    for array in (source,nodes,output):
        if array is not None:array.setflags(write=False)
    return PowderLineResponse(source,nodes,output,ngroups,*matrices,dict(version=VERSION,
        coefficient_unit='barn*angstrom per declared crystallographic cell',
        factor_unit='barn per declared cell; multiplicity includes Friedel signs',
        line_area_convention='A_lambda=m*|F(lambda_h)|^2*d^3*lambda_h/(4*pi*V)',
        source_basis='piecewise_constant' if nodes is None else 'piecewise_linear',
        clock_model=clock_model,timing_law_label=timing_law_label,
        probability_intervals_budgeted=used,missing_timing_or_reduction_mass_renormalized=False,
        whole_powder_path_includes_illumination_once=True,
        outside_source_domain_candidate_line_detector_pairs=uncovered,
        outside_source_domain_candidate_wavelength_range_a=None if not uncovered else [outside_min,outside_max],
        experimental_accuracy_certified=False))
