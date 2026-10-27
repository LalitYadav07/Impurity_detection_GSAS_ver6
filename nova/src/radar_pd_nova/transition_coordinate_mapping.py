"""Resolve display calibration through the same physical/profile bank mapping as analysis."""
from pathlib import Path

from .diffraction_coordinates import TofCalibration


def cached_record(scan,controller,records):
    record=getattr(controller,"records",{}).get(scan.get("run_id"))
    uid=getattr(record,"uid",None)
    job=getattr(record,"galaxy_job_id",None) or uid
    cached=records.get(uid)
    if cached is None and job:
        cached=next((r for r in records.values() if getattr(r,"galaxy_job_id",None)==job),None)
    return cached or record


def mapped_calibration(scan,controller,records):
    """Never substitute nominal export DIFC or another physical bank's profile."""
    bank=scan.get("bank") or (scan.get("metadata") or {}).get("source_bank")
    if bank is None:
        raise ValueError("A physical bank is required for coordinate conversion")
    bank=int(bank)
    record=cached_record(scan,controller,records)
    receipts=getattr(record,"provenance",{}) or {}
    receipt=receipts.get("powgen_bank") or receipts.get("nomad_bank") or {}
    profile=receipt.get("profile") or {}
    expected=profile.get("sha256") or receipt.get("profile_sha256")
    if receipt and int(receipt.get("source_bank",bank))!=bank:
        raise ValueError("Analysis receipt belongs to another physical bank")
    if profile.get("path") and Path(profile["path"]).is_file():
        return TofCalibration.from_profile(profile["path"],source_bank=bank,
            profile_bank=int(profile["profile_bank"]),expected_sha256=expected)
    settings=getattr(controller,"settings",None)
    run=None
    for stage in ("discovered","submitted","completed","failed"):
        run=getattr(getattr(controller,"state",None),stage,{}).get(scan.get("run_id"))
        if run is not None:
            break
    if controller is None or run is None:
        raise ValueError("The mapped instrument profile is not available locally")
    if not receipt:
        metadata=getattr(run,"scan_metadata",{}) or {}
        receipt=metadata.get("nomad_bank_provenance") or metadata.get("powgen_bank_provenance") or {}
        profile=receipt.get("profile") or {}
        expected=profile.get("sha256") or receipt.get("profile_sha256")
        if receipt and int(receipt.get("source_bank",bank))!=bank:
            raise ValueError("Analysis receipt belongs to another physical bank")
    if scan.get("instrument")=="PG3" and not getattr(settings,"instrument_profile_dataset_id",None):
        resolution,path=controller.resolve_profile(run)
        if resolution.source_bank!=bank or expected and resolution.sha256!=expected:
            raise ValueError("Resolved profile disagrees with the analysis bank/profile receipt")
        return TofCalibration.from_profile(path,source_bank=bank,profile_bank=resolution.profile_bank,
            expected_sha256=resolution.sha256)
    if scan.get("instrument")=="NOM":
        # _supplied_profile is the already selected/downloaded local profile.
        # Moving a display control must not initiate a new remote transfer.
        path=getattr(controller,"_supplied_profile",None)
        if path is not None and Path(path).is_file():
            from .nomad import validate_profile
            saved_bank=receipt.get("profile_bank") or profile.get("profile_bank")
            if saved_bank is not None and int(saved_bank)!=settings.profile_bank:
                raise ValueError("Supplied NOMAD profile Bank label differs from this analysis receipt")
            _,mapping=validate_profile(controller._setup(run),bank,path,settings.profile_bank,
                acknowledged=settings.profile_acknowledged)
            if expected and mapping["profile_sha256"]!=expected:
                raise ValueError("Supplied NOMAD profile differs from this analysis receipt")
            return TofCalibration.from_profile(path,source_bank=bank,profile_bank=settings.profile_bank,
                expected_sha256=mapping["profile_sha256"])
        if getattr(controller,"display_profile_error",None):
            raise ValueError(controller.display_profile_error)
    raise ValueError("A locally available, explicitly mapped neutron TOF profile is required")
