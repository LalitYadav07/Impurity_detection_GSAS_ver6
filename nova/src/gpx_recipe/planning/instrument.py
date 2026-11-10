"""Exact POWGEN profile choices; profile coverage and rate evidence are distinct."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "radar_pd_nova/data/powgen_calibrations"
SOURCE_PAGE = "https://neutrons-user-facilities.ornl.gov/instruments/powgen/"
UNSUPPORTED_PROFILES = {"PG60HR_2011B.prm", "PG60HR_2013A.prm"}


def profiles():
    catalog = json.loads((DATA / "catalog.json").read_text())
    rows = []
    for record in catalog["rules"]:
        path = DATA / record["resource"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("A registered POWGEN profile is missing or changed")
        supported = path.name not in UNSUPPORTED_PROFILES
        title = f"{record['cycle']} · {record['mode']} · {record['wavelength']} Å · {record['frequency']:g} Hz · {record['grouping']} · source bank {record['source_bank']}"
        rows.append({**record, "title": title + (" · legacy lookup unavailable" if not supported else ""),
                     "value": record["id"], "profile": str(path),
                     "forward_supported": supported,
                     "unavailable_reason": "Legacy PAB table needs an audited native profile implementation" if not supported else None,
                     "response_status": "not_experimentally_qualified"})
    return rows


def selected_profile(config_id, custom_profile=None):
    if custom_profile:
        path = Path(custom_profile).resolve(strict=True)
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Instrument parameter file exceeds 2 MiB")
        return {"id": "custom-" + hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                "profile": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "profile_bank": None, "source_bank": None,
                "response_status": "custom_profile_without_qualified_response"}
    matches = [row for row in profiles() if row["id"] == config_id]
    if len(matches) != 1:
        raise ValueError("Choose an exact POWGEN configuration")
    if not matches[0]['forward_supported']:
        raise ValueError(matches[0]['unavailable_reason'])
    return matches[0]


def selected_resolution_table(profile):
    """Return the exact companion table, never a nearest-cycle substitute."""
    catalog=json.loads((DATA/'resolution-tables.json').read_text())
    matches=[r for r in catalog['rules'] if r['profile_id']==profile['id']
        and r['profile_sha256']==profile['sha256']]
    if len(matches)!=1:
        raise ValueError('Published tables are available for the three registered 2026B HR 60 Hz profiles')
    row=matches[0];path=DATA/row['resource']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('The published resolution table is missing or changed')
    return dict(row,path=str(path),source_url=catalog['source_url'])


def coverage():
    rows = profiles()
    return {"profile_mappings": len(rows), "cycles": sorted({r["cycle"] for r in rows}),
            "supported_forward_mappings": sum(r['forward_supported'] for r in rows),
            "unavailable_forward_mappings": sum(not r['forward_supported'] for r in rows),
            "modes": sorted({r["mode"] for r in rows}),
            "wavelengths_angstrom": sorted({r["wavelength"] for r in rows}),
            "frequencies_hz": sorted({r["frequency"] for r in rows}),
            "validated_count_response_mappings": 0,
            "source": SOURCE_PAGE}
