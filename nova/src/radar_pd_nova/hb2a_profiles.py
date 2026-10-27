"""Exact dated HB-2A facility-published CW profile mappings.

No nearest-cycle fallback. Publication establishes calibration provenance;
sample corrections and the suitability of a refinement still require review.
"""
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).parent / 'data' / 'hb2a_calibrations'


def catalog_digest(root=None):
    return sha256(((Path(root) if root else ROOT) / 'catalog.json').read_bytes()).hexdigest()


def resolve_profile(setup, *, root=None):
    from .hb2a import bounded_bytes, validate_profile
    root = Path(root) if root else ROOT
    data, _ = bounded_bytes(root / 'catalog.json')
    catalog = json.loads(data)
    if catalog.get('schema') != 'radar-pd-hb2a-profiles/v1':
        raise ValueError('Unsupported HB-2A calibration registry')
    metadata = setup.metadata
    try:
        times = [datetime.fromisoformat(metadata[k]) for k in ('start_time','end_time')]
        if any(t.tzinfo is None for t in times) or times[1] < times[0]:
            raise ValueError()
        start,end = [t.astimezone(ZoneInfo('America/New_York')).date().isoformat() for t in times]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError('HB-2A automatic profile selection needs timezone-qualified acquisition start and end times') from exc
    matches = [p for p in catalog.get('profiles', [])
        if p['start_date'] <= start <= end <= p['end_date']
        and p['ge_reflection'] == metadata.get('ge_reflection')
        and p['collimator'] == metadata.get('collimator')]
    if len(matches) != 1:
        raise ValueError(f'No unique published HB-2A profile for {start}–{end}, Ge{metadata.get("ge_reflection")}, '
            f'collimator {metadata.get("collimator") or "unresolved"}. Supply a reviewed matching .instprm; no nearby cycle is substituted')
    entry = matches[0]
    path = root / entry['filename']
    if path.parent != root or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('HB-2A registry profile path escapes its calibration directory')
    profile, _ = bounded_bytes(path)
    if len(profile) != entry['bytes'] or sha256(profile).hexdigest() != entry['sha256']:
        raise ValueError('Packaged HB-2A calibration checksum mismatch')
    validated = validate_profile(setup,path,acknowledged=True)
    return path, {**entry, 'registry_sha256':sha256(data).hexdigest(),
        'mapping_basis':'Exact published cycle dates, measured Ge reflection and reviewed/measured collimation',
        'profile_parameters':validated['parameters']}
