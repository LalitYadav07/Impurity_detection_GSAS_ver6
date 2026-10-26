"""Optional whole-pattern evidence from completed RADAR-PD final refinements.

This module is deliberately separate from phase-independent measured-peak
analysis. Rwp changes are recipe/model/counting-statistics dependent, not proof
of a physical phase transition and not a spatially localized finding.
"""
from hashlib import sha256
from pathlib import Path
import math

import numpy as np

from .scan_interpretation import Pattern,load_pattern
from .transition_input import digest


def final_fit_key(scan):
    path=scan.get("pattern_path")
    try:
        stat=Path(path).stat() if path else None
        revision=[path,stat.st_size,stat.st_mtime_ns] if stat else scan.get("pattern")
    except OSError:
        revision=[path,"unavailable"]
    return digest([revision,scan.get("final_rwp"),scan.get("analysis_status"),scan.get("result_status"),
        scan.get("analysis_mode"),scan.get("configuration_id"),scan.get("candidate_library_id"),
        scan.get("observables"),scan.get("pattern_observation")])


def final_fit(scan):
    unavailable=lambda reason:{"available":False,"reason":reason}
    complete={"complete","completed","ok","success"}
    analysis=str(scan.get("analysis_status") or scan.get("result_status") or "").lower()
    result=str(scan.get("result_status") or "").lower()
    if analysis not in complete or result and result not in complete:
        return unavailable("Final refinement is pending or unsuccessful")
    if str(scan.get("analysis_mode") or "").lower()!="full":
        return unavailable("A completed Full RADAR-PD final refinement is required")
    value=scan.get("final_rwp")
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        return unavailable("An explicit final-project Rwp is not available")
    try:
        packet=scan.get("pattern")
        if not isinstance(packet,dict):
            packet=load_pattern(scan["pattern_path"],expected_sha256=(scan.get("pattern_observation") or {}).get("sha256"))
        fitted=Pattern.parse(packet)
    except (OSError,ValueError,TypeError,KeyError):
        return unavailable("The final measured/fit-domain receipt is unavailable")
    observables=scan.get("observables") or {}
    project=observables.get("source_gpx_sha256")
    if not project or packet.get("source_gpx_sha256")!=project:
        return unavailable("Final Rwp and measured-curve project provenance disagree")
    context=observables.get("context") or {}
    recipe=context.get("recipe_key")
    configuration=scan.get("configuration_id")
    profile=context.get("profile_sha256") or (packet.get("context") or {}).get("profile_sha256")
    if not recipe or not configuration or not profile or np.count_nonzero(fitted.fitted)<16:
        return unavailable("Refinement configuration, recipe, profile and fitted-domain provenance are required")
    grid_hash=sha256(np.asarray(fitted.x,dtype="<f8").tobytes()).hexdigest()
    protocol=digest([recipe,configuration,scan.get("candidate_library_id") or "builtin",
        profile,packet.get("coordinate"),packet.get("fit_domain"),grid_hash])
    return {"available":True,"value":float(value),"unit":"percent","project_sha256":project,
        "mask_bits":np.packbits(fitted.fitted).tobytes().hex(),"points":len(fitted.x),
        "protocol_sha256":protocol,"reason":"Completed final-project Rwp; whole-pattern model-dependent evidence"}


def compare_final_fits(before,after,policy):
    if not before or not after or not before.get("available") or not after.get("available"):
        reason=(after or {}).get("reason") if not (after or {}).get("available") else (before or {}).get("reason")
        return {"available":False,"detected":False,"reason":reason or "Final refinement is pending"}
    if before["protocol_sha256"]!=after["protocol_sha256"]:
        return {"available":False,"detected":False,
            "reason":"Rwp comparison withheld: configuration, library, recipe, profile or fitted domain differs"}
    a=np.unpackbits(np.frombuffer(bytes.fromhex(before["mask_bits"]),dtype=np.uint8))[:before["points"]].astype(bool)
    b=np.unpackbits(np.frombuffer(bytes.fromhex(after["mask_bits"]),dtype=np.uint8))[:after["points"]].astype(bool)
    agreement=float(np.count_nonzero(a&b)/max(1,np.count_nonzero(a|b)))
    if agreement<.98:
        return {"available":False,"detected":False,"mask_agreement":agreement,
            "reason":"Rwp comparison withheld: contributing fit points overlap by less than 98%"}
    delta=after["value"]-before["value"]
    relative=abs(delta)/max(before["value"],1e-8)
    detected=(abs(delta)>=policy.rwp_delta_pp and relative>=policy.rwp_relative
        and (policy.rwp_direction=="either" or delta>0))
    return {"available":True,"detected":bool(detected),"before":before["value"],"after":after["value"],
        "delta_pp":delta,"relative_change":relative,"protocol_sha256":before["protocol_sha256"],"mask_agreement":agreement,
        "reason":"Final Rwp change uses both the absolute and relative thresholds; no local peak interval is assigned"}
