"""Resolve verified calibration through the analysis physical/profile bank mapping."""
from hashlib import sha256
from pathlib import Path
import re
import tempfile

from .diffraction_coordinates import TofCalibration, CwCalibration


def cached_record(scan,controller,records):
    record=getattr(controller,"records",{}).get(scan.get("run_id"))
    if record is None:
        record=records.get(scan.get("run_id"))
    uid=getattr(record,"uid",None)
    job=(getattr(record,"galaxy_job_id",None) or scan.get("galaxy_job_id")
         or scan.get("job_id") or uid)
    cached=records.get(uid)
    if cached is None and job:
        cached=next((r for r in records.values() if getattr(r,"galaxy_job_id",None)==job),None)
    return cached or record


def _galaxy_profile_calibration(scan, record, receipt, service, bank):
    """Recover the exact staged Galaxy input after a temporary path disappears."""
    if service is None:
        return None
    profile=receipt.get('profile') or {}
    dataset_id=(receipt.get('staged_profile_dataset_id')
        or receipt.get('manual_profile_dataset_id')
        or scan.get('transition_profile_dataset_id')
        or (getattr(record,'input_dataset_ids',{}) or {}).get('instrument'))
    if not dataset_id:
        return None
    expected=(receipt.get('staged_profile_sha256') or profile.get('sha256')
        or receipt.get('profile_sha256')
        or ((scan.get('pattern') or {}).get('context') or {}).get('profile_sha256'))
    if not expected:
        raise ValueError('Saved instrument input has no trusted profile checksum')
    try:
        text=service.load_text_document(str(dataset_id))
    except Exception as exc:
        raise ValueError('The saved instrument profile could not be recovered from Galaxy') from exc
    raw=text.encode('utf-8')
    if not 0<len(raw)<=1024*1024 or sha256(raw).hexdigest()!=expected:
        raise ValueError('Saved Galaxy profile differs from the analysis checksum')
    instrument=scan.get('instrument')
    if instrument=='HB2A':
        suffix='.instprm';profile_bank=1
    else:
        suffix='.instprm' if 'Type:' in text and 'Bank:' in text else '.prm'
        profile_bank=(receipt.get('staged_bank') if receipt.get('staged_profile_sha256')
                      else receipt.get('profile_bank') or profile.get('profile_bank'))
        if profile_bank is None and suffix=='.instprm':
            labels={int(value) for value in re.findall(r'(?m)(?:^|;)\s*Bank:\s*(\d+)\b',text)}
            if len(labels)==1:
                profile_bank=labels.pop()
        if profile_bank is None:
            raise ValueError('Saved instrument input has no unambiguous profile Bank label')
    with tempfile.TemporaryDirectory(prefix='radar-transition-profile-') as folder:
        path=Path(folder)/('checked'+suffix)
        path.write_bytes(raw)
        if instrument=='HB2A':
            return CwCalibration.from_profile(path,source_bank=bank,profile_bank=profile_bank,
                expected_sha256=expected)
        return TofCalibration.from_profile(path,source_bank=bank,profile_bank=int(profile_bank),
            expected_sha256=expected)


def mapped_calibration(scan,controller,records,service=None):
    """Never substitute nominal export DIFC or another physical bank's profile."""
    bank=scan.get("bank") or (scan.get("metadata") or {}).get("source_bank")
    if bank is None:
        raise ValueError("A physical bank is required for coordinate conversion")
    bank=int(bank)
    record=cached_record(scan,controller,records)
    receipts=getattr(record,"provenance",{}) or {}
    receipt=(receipts.get("powgen_bank") or receipts.get("nomad_bank")
        or receipts.get("snap_bank") or receipts.get('hb2a_pattern') or {})
    if not receipt:
        metadata=scan.get('metadata') or {}
        receipt=(metadata.get('powgen_bank_provenance') or metadata.get('nomad_bank_provenance')
            or metadata.get('snap_bank_provenance') or metadata.get('hb2a_pattern_provenance') or {})
    profile=receipt.get("profile") or {}
    expected=profile.get("sha256") or receipt.get("profile_sha256")
    if receipt and int(receipt.get("source_bank",bank))!=bank:
        raise ValueError("Analysis receipt belongs to another physical bank")
    if scan.get('instrument')!='HB2A' and profile.get("path") and Path(profile["path"]).is_file():
        return TofCalibration.from_profile(profile["path"],source_bank=bank,
            profile_bank=int(profile["profile_bank"]),expected_sha256=expected)
    settings=getattr(controller,"settings",None)
    run=None
    for stage in ("discovered","submitted","completed","failed"):
        run=getattr(getattr(controller,"state",None),stage,{}).get(scan.get("run_id"))
        if run is not None:
            break
    if controller is None or run is None:
        recovered=_galaxy_profile_calibration(scan,record,receipt,service,bank)
        if recovered is not None:
            return recovered
        raise ValueError("The mapped instrument profile is not available locally")
    if not receipt:
        metadata=getattr(run,"scan_metadata",{}) or {}
        receipt=(metadata.get("nomad_bank_provenance") or metadata.get("powgen_bank_provenance")
            or metadata.get("snap_bank_provenance") or metadata.get("hb2a_pattern_provenance") or {})
        profile=receipt.get("profile") or {}
        expected=profile.get("sha256") or receipt.get("profile_sha256")
        if receipt and int(receipt.get("source_bank",bank))!=bank:
            raise ValueError("Analysis receipt belongs to another physical bank")
    if scan.get('instrument')=='HB2A':
        path=getattr(controller,'_supplied_profile',None)
        if path is not None and Path(path).is_file():
            from .hb2a import Setup, validate_profile
            setup=Setup(**(run.scan_metadata.get('hb2a_setup') or {}))
            mapping=validate_profile(setup,path,acknowledged=settings.profile_acknowledged)
            if expected and mapping['profile_sha256']!=expected:
                raise ValueError('Selected CW profile differs from the analysis receipt')
            return CwCalibration.from_profile(path,source_bank=bank,profile_bank=1,
                expected_sha256=mapping['profile_sha256'])
        recovered=_galaxy_profile_calibration(scan,record,receipt,service,bank)
        if recovered is not None:
            return recovered
        if getattr(controller,'display_profile_error',None):
            raise ValueError(controller.display_profile_error)
        raise ValueError('The selected CW profile is not available locally; retain native 2theta')
    if scan.get("instrument")=="PG3" and not getattr(settings,"instrument_profile_dataset_id",None):
        resolution,path=controller.resolve_profile(run)
        if resolution.source_bank!=bank or expected and resolution.sha256!=expected:
            raise ValueError("Resolved profile disagrees with the analysis bank/profile receipt")
        return TofCalibration.from_profile(path,source_bank=bank,profile_bank=resolution.profile_bank,
            expected_sha256=resolution.sha256)
    if scan.get("instrument") in {"NOM", "SNAP"}:
        # _supplied_profile is the already selected/downloaded local profile.
        # Moving a display control must not initiate a new remote transfer.
        path=getattr(controller,"_supplied_profile",None)
        if path is not None and Path(path).is_file():
            saved_bank=receipt.get("profile_bank") or profile.get("profile_bank")
            if saved_bank is not None and int(saved_bank)!=settings.profile_bank:
                raise ValueError("Supplied profile Bank label differs from this analysis receipt")
            if scan.get("instrument") == "SNAP":
                from .snap import validate_profile
            else:
                from .nomad import validate_profile
            _,mapping=validate_profile(controller._setup(run),bank,path,settings.profile_bank,
                acknowledged=settings.profile_acknowledged)
            # SNAP's nested profile receipt describes the staged Bank-1 copy;
            # this file is the original supplied profile with its original label.
            original_expected=receipt.get('profile_sha256') or expected
            if original_expected and mapping["profile_sha256"]!=original_expected:
                raise ValueError("Supplied profile differs from this analysis receipt")
            return TofCalibration.from_profile(path,source_bank=bank,profile_bank=settings.profile_bank,
                expected_sha256=mapping["profile_sha256"])
    recovered=_galaxy_profile_calibration(scan,record,receipt,service,bank)
    if recovered is not None:
        return recovered
    if getattr(controller,"display_profile_error",None):
        raise ValueError(controller.display_profile_error)
    raise ValueError("An explicitly mapped neutron TOF profile is required")
