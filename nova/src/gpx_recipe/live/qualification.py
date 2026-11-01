"""Held-out uncertainty/policy receipts; engineering success is not qualification."""
import math


def wilson(successes,total,z=1.959963984540054):
    if not isinstance(total,int) or total<=0 or not 0<=successes<=total:raise ValueError('Independent trial counts required')
    p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0.,center-half),min(1.,center+half)]


def summarize(trials):
    """Each entry is ONE complete independent experiment, not one snapshot/site."""
    if not trials:raise ValueError('No held-out experiments')
    ids=[r['experiment_id'] for r in trials]
    if len(ids)!=len(set(ids)):raise ValueError('Repeated snapshots are not independent calibration trials')
    n=len(trials);resolved=sum(bool(r['resolved']) for r in trials)
    covered=sum(bool(r['resolved']) and bool(r['all_target_intervals_cover_truth']) for r in trials)
    excluded=sum(bool(r['truth_excluded_at_any_step']) for r in trials)
    premature=sum(bool(r['premature_stop_at_any_step']) for r in trials)
    return dict(experiments=n,resolved=resolved,unresolved=n-resolved,simultaneous_coverage_including_unresolved=covered/n,
        simultaneous_coverage_interval=wilson(covered,n),false_exclusion_interval=wilson(excluded,n),premature_stop_interval=wilson(premature,n),
        policy_evaluated=all(r.get('policy_evaluated',False) for r in trials),scientifically_qualified=False,
        note='Intervals use independent experiments; sampler failures remain failures. No qualification is inferred without empirical noise/reference evidence and independent material/configuration holdouts.')
