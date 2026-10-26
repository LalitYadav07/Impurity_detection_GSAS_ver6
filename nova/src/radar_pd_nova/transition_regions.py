"""Bounded, optional localization for one selected scan, never a transition mask."""
import math

import numpy as np

from .diffraction_coordinates import coordinate_values


def localized_regions(decision,signals,pattern,target,calibration=None,limit=16,comparison_pattern=None,include_split_features=True):
    candidates=[]
    for key in signals:
        indicator=(decision.get("indicators") or {}).get(key) or {}
        if not indicator.get("localizable",False):
            continue
        regions=list(indicator.get("regions",[])) if indicator.get("detected") else []
        if key=="splitting" and include_split_features:
            regions += [v for v in indicator.get("features",[]) if not v.get("passes_rule")]
        for region in regions:
            interval=region.get("interval")
            if not isinstance(interval,(list,tuple)) or len(interval)!=2:
                continue
            try:
                lo,hi=sorted(map(float,interval))
            except (TypeError,ValueError):
                continue
            if not math.isfinite(lo+hi):
                continue
            lo=max(lo,float(pattern.x[0]));hi=min(hi,float(pattern.x[-1]))
            if lo<hi:
                passes=region.get("passes_rule",region in indicator.get("regions",[]))
                candidates.append({"lo":lo,"hi":hi,"signals":{key},
                    "strength":float(region.get("error_ratio") or 1),
                    "status":"recognized_split" if region.get("kind")=="local_component_separation_change" and not passes else "detected_change",
                    "diagnostics":[{**region,"passes_rule":bool(passes)}] if region.get("before_profile") else []})
    merged=[]
    for region in sorted(candidates,key=lambda r:r["lo"]):
        if merged and region["lo"]<=merged[-1]["hi"]:
            merged[-1]["hi"]=max(merged[-1]["hi"],region["hi"])
            merged[-1]["signals"].update(region["signals"])
            merged[-1]["strength"]=max(merged[-1]["strength"],region["strength"])
            if region["status"]=="detected_change":
                merged[-1]["status"]="detected_change"
            merged[-1]["diagnostics"]+=region["diagnostics"]
        else:
            merged.append(region.copy())
    localized=[]
    for region in merged:
        valid=pattern.valid & (pattern.x>=region["lo"]) & (pattern.x<=region["hi"])
        if comparison_pattern is not None:
            valid &= np.interp(pattern.x,comparison_pattern.x,comparison_pattern.valid.astype(float),left=0,right=0)>.999
        edges=np.diff(np.r_[False,valid,False].astype(int))
        for start,end in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if end-start<2:
                continue
            interval=sorted(coordinate_values([pattern.x[start],pattern.x[end-1]],
                pattern.scope.coordinate,target,calibration).tolist())
            localized.append({"interval":interval,"signals":sorted(region["signals"]),
                "strength":region["strength"],"status":region["status"],"diagnostics":region["diagnostics"]})
    # Prioritize corroborated/strong regions when the view would become dense;
    # all original evidence remains in the downloadable report.
    shown=sorted(localized,key=lambda r:("splitting" in r["signals"],r["status"]=="detected_change",len(r["signals"]),r["strength"]),reverse=True)[:limit]
    return sorted(shown,key=lambda r:r["interval"][0]),{"total":len(localized),"shown":len(shown)}
