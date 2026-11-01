"""Phase-independent local component fits, including evolution of existing doublets.

An asymmetric single-profile null competes with a two-component profile. These
are local descriptive models, not crystallographic phase or hkl assignments.
"""
import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def _pv(x,width,mixing):
    z=x/width
    return (1-mixing)*np.exp(-4*math.log(2)*z*z)+mixing/(1+4*z*z)


def _candidate(prepared,lo,hi):
    p=prepared.observation;use=(p.x>=lo)&(p.x<=hi)
    if use.sum()<24 or not np.all(p.valid[use]):
        return False
    y=gaussian_filter1d(p.intensity[use],.4)
    noise=float(np.median(prepared.noise[use]))
    peaks,_=find_peaks(y,prominence=max(3*noise,.015*(y.max()-y.min())),distance=2)
    return 2<=len(peaks)<=3


def component_profile(prepared,lo,hi):
    p=prepared.observation
    start,end=np.searchsorted(p.x,[lo,hi]);end=min(len(p.x),end+1)
    cache=getattr(prepared,"_component_profile_cache",None)
    if cache is None:
        cache=prepared._component_profile_cache={}
    key=(int(start),int(end))
    if key in cache:
        return cache[key]
    cache[key]=None
    use=slice(start,end)
    if end-start<24 or not np.all(p.valid[use]):
        return None
    xx=p.x[use];center=float((xx[0]+xx[-1])/2);scale=float((xx[-1]-xx[0])/3)
    height=float(np.max(prepared.excess[use]))
    if height<=0:
        return None
    x=(xx-center)/scale
    base=float(np.median(np.r_[p.intensity[start:start+3],p.intensity[end-3:end]]))
    y=(p.intensity[use]-base)/height
    sigma=np.maximum(prepared.noise[use]/height,1e-6)
    step=float(np.median(np.diff(x)));minimum=max(.04,step*1.25)
    def one(v):
        a,c,left,right,mixing,offset,slope=v
        return (a*_pv(x-c,np.where(x<c,left,right),mixing)+offset+slope*x-y)/sigma
    def two(v):
        a,b,c,gap,width,mixing,offset,slope=v
        return (a*_pv(x-c+gap/2,width,mixing)+b*_pv(x-c-gap/2,width,mixing)+offset+slope*x-y)/sigma
    try:
        single=least_squares(one,[1,0,.7,.7,.2,0,0],
            bounds=([0,-.8,minimum,minimum,0,-.8,-.5],[4,.8,2,2,1,.8,.5]),max_nfev=100)
        best=None
        for gap in (.35,.7,1.1):
            fit=least_squares(two,[.7,.5,0,gap,.45,.2,0,0],
                bounds=([0,0,-.8,step*1.5,minimum,0,-.8,-.5],[4,4,.8,2.2,1.8,1,.8,.5]),max_nfev=100)
            if fit.success and (best is None or np.dot(fit.fun,fit.fun)<np.dot(best.fun,best.fun)):
                best=fit
        if best is None or not single.success:
            return None
        a,b,c,gap,width,mixing,offset,slope=best.x
        chi=float(np.dot(best.fun,best.fun));scatter=max(1.,chi/max(1,len(x)-8))
        covariance=np.linalg.pinv(best.jac.T@best.jac)*scatter
        gap_error=max(math.sqrt(max(0,float(covariance[3,3])))*scale,width*scale*.015)
        amplitudes=[a,b];errors=[math.sqrt(max(0,float(covariance[k,k]))) for k in (0,1)]
        snr=min(v/max(e,1e-12) for v,e in zip(amplitudes,errors))
        improvement=float(np.dot(single.fun,single.fun)-chi-math.log(len(x)))
        adequate=(min(a,b)>=.10*max(a,b) and snr>=6 and scatter<25
                  and minimum*1.01<width<1.79 and gap<2.19)
        answer={"interval":[float(xx[0]),float(xx[-1])],"bic_improvement":improvement,
            "two_component_qualified":bool(adequate and improvement>=12),
            "separation":float(gap*scale),"separation_error":float(gap_error),
            "component_fwhm":float(width*scale),"component_snr":float(snr),
            "positions":[float(center+(c-gap/2)*scale),float(center+(c+gap/2)*scale)],
            "relative_amplitudes":[float(a),float(b)],"residual_scatter":scatter,
            "single_residual_scatter":float(np.dot(single.fun,single.fun)/max(1,len(x)-7)),
            "method":"asymmetric single versus common-width two-component pseudo-Voigt; native weighted data"}
        cache[key]=answer
        return answer
    except (ValueError,FloatingPointError,np.linalg.LinAlgError):
        return None


def _aligned_windows(before, after, p, q, extent, matches):
    """Retain the same local peak support as the matched anchor moves.

    A common absolute-x crop can include a neighbor in just one acquisition,
    making rigid peak motion look like a change in component count. Work in
    coordinates relative to each matched anchor, expanding both native-data
    windows together to retain each included feature's complete support.
    Corresponding neighbors are included even when they moved independently.
    Intruding tails of excluded neighbors are recorded so that the local fit
    cannot mistake them for a new component. More than three resolved features
    remains outside the local model's scope.
    """
    radius = extent * max(p.width / p.position, q.width / q.position)
    lo, hi = -radius, radius
    features = (before.features, after.features)
    anchors = (p.position, q.position)
    chosen = ({p.position}, {q.position})
    # Even an ambiguous assignment is a possible counterpart that must remain
    # inside the support. Dropping it can let a crowded anchor pair describe
    # unrelated neighboring doublets as if they were the same local feature.
    links = [(m['before_position'], m['after_position']) for m in matches]
    while True:
        old = (lo, hi, len(chosen[0]), len(chosen[1]))
        for side in (0, 1):
            anchor = anchors[side]
            for feature in features[side]:
                padding = 2.25 * feature.width + 2 * feature.bin_width
                left = (feature.position - padding) / anchor - 1
                right = (feature.position + padding) / anchor - 1
                center = feature.position / anchor - 1
                if feature.position in chosen[side] or lo <= center <= hi:
                    chosen[side].add(feature.position)
                    lo, hi = min(lo, left), max(hi, right)
        for left, right in links:
            if left in chosen[0] or right in chosen[1]:
                chosen[0].add(left)
                chosen[1].add(right)
        if max(map(len, chosen)) > 3:
            return None
        if old == (lo, hi, len(chosen[0]), len(chosen[1])):
            break
    edge_neighbor_overlap = False
    for side in (0, 1):
        anchor = anchors[side]
        for feature in features[side]:
            if feature.position in chosen[side]:
                continue
            padding = 2.25 * feature.width + 2 * feature.bin_width
            left = (feature.position - padding) / anchor - 1
            right = (feature.position + padding) / anchor - 1
            if left < hi and right > lo:
                edge_neighbor_overlap = True
    return ((p.position * (1 + lo), p.position * (1 + hi)),
            (q.position * (1 + lo), q.position * (1 + hi)),
            tuple(map(len, chosen)), edge_neighbor_overlap)


def component_changes(before,after,report):
    fa={f.position:f for f in before.features};fb={f.position:f for f in after.features}
    windows=[];events=[]
    for match in report.get("matches",[]):
        p,q=fa[match["before_position"]],fb[match["after_position"]]
        if min(p.snr,q.snr)<12:
            continue
        a=b=None
        counts = None
        # Both fits retain complete corresponding local support. Native data
        # are never interpolated or warped to make the profiles agree.
        for extent in (1.45,2.8):
            support = _aligned_windows(before, after, p, q, extent, report.get('matches', []))
            if support is None:
                continue
            (lo, hi), (other_lo, other_hi), counts, edge_neighbor_overlap = support
            if any(min(hi,v)-max(lo,u)>.65*min(hi-lo,v-u) for u,v in windows):
                break
            if not (_candidate(before,lo,hi) or _candidate(after,other_lo,other_hi)):
                continue
            a=component_profile(before,lo,hi);b=component_profile(after,other_lo,other_hi)
            if a and b and (a["two_component_qualified"] or b["two_component_qualified"]):
                windows.append((lo,hi))
                break
            a=b=None
        if not a or not b:
            continue
        qualified=[v for v in (a,b) if v["two_component_qualified"]]
        if not qualified:
            continue
        # Localize the components themselves, rather than painting the entire
        # (wider) background-fitting window as changed.
        padding=max(v["component_fwhm"] for v in qualified)
        positions=[x for v in qualified for x in v["positions"]]
        interval=[max(min(lo,other_lo),min(positions)-padding),
                  min(max(hi,other_hi),max(positions)+padding)]
        if not edge_neighbor_overlap and (a["two_component_qualified"] and counts[1] < 2
                and b["bic_improvement"]<=0 and b["single_residual_scatter"]<5
                or b["two_component_qualified"] and counts[0] < 2
                and a["bic_improvement"]<=0 and a["single_residual_scatter"]<5):
            events.append({"kind":"local_resolved_component_count_change","interval":interval,
                "before_components":2 if a['two_component_qualified'] else 1,
                "after_components":2 if b['two_component_qualified'] else 1,
                "before_profile":a,"after_profile":b,"error_ratio":min(v['component_snr'] for v in qualified),
                "physical_cause":"undetermined; local component resolution or coalescence"})
            continue
        if not (a["two_component_qualified"] and b["two_component_qualified"]):
            continue
        # Remove the matched anchor's common coordinate dilation. This does
        # not remove differential motion between the two local components.
        coordinate_scale = q.position / p.position
        change=abs(b["separation"] / coordinate_scale-a["separation"])
        error=math.hypot(a["separation_error"],b["separation_error"] / coordinate_scale)
        events.append({"kind":"local_component_separation_change","interval":interval,
            "separation_change_fwhm":change/np.mean([a["component_fwhm"],b["component_fwhm"] / coordinate_scale]),
            "error_ratio":change/max(error,1e-30),"before_profile":a,"after_profile":b,
            "coordinate_scale":coordinate_scale,
            "edge_neighbor_overlap":edge_neighbor_overlap,
            "physical_cause":"undetermined; evolving doublet/shoulder includes independently moving reflections"})
    return events
