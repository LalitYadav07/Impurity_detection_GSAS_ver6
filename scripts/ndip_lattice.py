"""Read final-cell values and covariance uncertainties from our own GSAS-II outputs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import sys
from typing import Any

CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
GSAS_KEYS = ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma", "volume")


def final_project_key(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    if "seq_final_joint_polished" in name:
        return (4, 0, name)
    match = re.search(r"seq_pass(\d+)_kept_polished", name)
    if match:
        return (2, int(match[1]), name)
    return (3 if "seq_final_main_polished" in name else 1, 0, name)


def lattice_from_records(records, *, source: str, labels=None):
    """Fallback values remain useful when GSAS-II/covariance is unavailable."""
    rows = []
    covariance = {}
    for record in records:
        if record and record[0][0] == "Covariance":
            covariance = record[0][1] or {}
        if not record or record[0][0] != "Phases":
            continue
        for name, data in record[1:]:
            general = data.get("General") or {}
            cell = general.get("Cell") or []
            if len(cell) < 8:
                continue
            row: dict[str, Any] = {"phase_id": str(name), "phase": (labels or {}).get(str(name), str(name)),
                "space_group": str((general.get("SGData") or {}).get("SpGrp") or ""),
                "source_gpx": source, "uncertainty_status": "not_estimated"}
            for key, value in zip(CELL_KEYS, cell[1:8]):
                number = float(value)
                row[key] = number if math.isfinite(number) else None
                row[key + "_esd"] = None
            haps = [h for h in (data.get("Histograms") or {}).values() if h.get("Use", True)]
            scale = float(haps[0].get("Scale", [0])[0]) if len(haps) == 1 else None
            mass = float(general.get("Mass") or 0)
            # GSAS-II HAP scales already contain the volume normalization.
            row["_weight"] = scale * mass if scale is not None else None
            rows.append(row)
    total = sum(max(0, r["_weight"] or 0) for r in rows)
    for row in rows:
        weight = row.pop("_weight")
        row["weight_percent"] = 100 * max(0, weight) / total if total > 0 and weight is not None else None
    rwp = (covariance.get("Rvals") or {}).get("Rwp")
    return rows, float(rwp) if rwp is not None and math.isfinite(float(rwp)) else None


def extract_lattice(path: Path, records, *, source: str, labels=None):
    rows, rwp = lattice_from_records(records, source=source, labels=labels)
    # GSAS-II performs the symmetry-dependent covariance propagation. Do not
    # manufacture ESDs from independent lengths or report missing ones as zero.
    gsas_root = os.environ.get("RADAR_PD_GSASII_ROOT") or str(Path(__file__).resolve().parents[1] / "GSAS-II")
    if gsas_root not in sys.path:
        sys.path.insert(0, gsas_root)
    try:
        from GSASII import GSASIIscriptable as G2sc
        project = G2sc.G2Project(gpxfile=str(path))
        covariance = project.data.get("Covariance", {}).get("data", {})
        variables = [str(v) for v in covariance.get("varyList", [])]
        by_name = {p.name: p for p in project.phases()}
        for row in rows:
            phase = by_name[row["phase_id"]]
            prefix = f"{phase.id}::A"
            if not any(v.startswith(prefix) and v[len(prefix):] in {"0", "1", "2", "3", "4", "5"} for v in variables):
                continue
            values, errors = phase.get_cell_and_esd()
            if not covariance.get("freshCOV", False):
                row["uncertainty_status"] = "covariance_not_current"
                continue
            for key, gsas_key in zip(CELL_KEYS, GSAS_KEYS):
                error = float(errors[gsas_key])
                row[key + "_esd"] = error if math.isfinite(error) and error > 0 else None
            row["uncertainty_status"] = "estimated" if any(row[k + "_esd"] is not None for k in CELL_KEYS) else "not_estimated"
        if project.histograms():
            value = project.histograms()[0].get_wR()
            if value is not None and math.isfinite(float(value)):
                rwp = float(value)
    except (ImportError, KeyError, ValueError, TypeError, RuntimeError, AttributeError) as exc:
        print(f"[WARN] Lattice ESDs unavailable for {path.name}: {exc}")
    return rows, rwp


def lattice_quality(rows):
    issues = []
    for row in rows:
        lengths = [row.get(k) for k in ("a", "b", "c")]
        angles = [row.get(k) for k in ("alpha", "beta", "gamma")]
        valid = all(v is not None and math.isfinite(v) and v > 0 for v in lengths)
        valid = valid and all(v is not None and math.isfinite(v) and 0 < v < 180 for v in angles)
        if valid:
            ca, cb, cg = (math.cos(math.radians(v)) for v in angles)
            valid = 1 + 2*ca*cb*cg - ca*ca - cb*cb - cg*cg > 1e-10
        if not valid or row.get("volume") is None or row["volume"] <= 0:
            issues.append({"code": "invalid_cell", "message": f"{row['phase']}: invalid final unit cell"})
        weight = row.get("weight_percent")
        if len(rows) > 1 and weight is not None and weight < 0.05:
            issues.append({"code": "vanishing_phase", "message": f"{row['phase']}: final fraction below 0.05 wt%; review the accepted model"})
    return issues
