"""Fresh native-result checks, following NDIP's recent refinement correction.

The freshCOV reset/check is adopted from gpx_recipe/execution.py at d52e61c
(ndip-battery-experiments). Qualification follows the native magnetic worker's
convergence/aborted/SVD policy at 573134d (fe2o3-ndip-benchmark-fixes).
"""
import numpy as np


def checked_native_refine(project):
    project.data.setdefault('Covariance',{}).setdefault('data',{})['freshCOV']=False
    project.save()
    project.do_refinements([{}])
    covariance=project.data.get('Covariance',{}).get('data',{})
    if not covariance.get('freshCOV'):
        raise ValueError('Native refinement did not produce fresh covariance; stale results rejected')
    values=np.asarray(covariance.get('variables',[]),float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('Native refinement did not produce finite fitted parameters')
    project.save()
    r=covariance.get('Rvals',{})
    return dict(qualified=bool(r.get('converged') and not r.get('Aborted') and not r.get('SVD0',0)),
                converged=bool(r.get('converged')),aborted=bool(r.get('Aborted')),rank_deficiencies=int(r.get('SVD0',0)),
                fresh_covariance=True,message=str(r.get('msg','')))


def refresh_saved_profiles(project, calculate=None):
    """Score profiles recomputed from saved parameters, as in NDIP's baseline refresh."""
    if calculate is None:
        from GSASII import GSASIIstrMain
        calculate = GSASIIstrMain.DoNoFit
    project.save()
    receipts = []
    for h in project.histograms():
        data = h.data['data'][1]
        old = np.asarray(data[3], float).copy()
        fresh = np.asarray(calculate(str(project.filename), h.name), float)
        if fresh.shape != old.shape or not np.isfinite(fresh).all():
            raise ValueError('Saved native profile could not be recalculated')
        receipts.append(dict(histogram=h.name, maximum_stale_curve_difference=float(np.max(abs(old-fresh))),
                             maximum_difference_in_sigma=float(np.max(abs(old-fresh)*np.sqrt(data[2])))))
        data[3] = fresh
        data[5] = np.asarray(data[1])-fresh
    project.save()
    return receipts
