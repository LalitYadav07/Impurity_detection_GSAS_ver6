"""Read final-cell values and covariance uncertainties from our own GSAS-II outputs."""

from __future__ import annotations

import math
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

CELL_KEYS = ("a", "b", "c", "alpha", "beta", "gamma", "volume")
GSAS_KEYS = (
    "length_a",
    "length_b",
    "length_c",
    "angle_alpha",
    "angle_beta",
    "angle_gamma",
    "volume",
)


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
            row: dict[str, Any] = {
                "phase_id": str(name),
                "phase": (labels or {}).get(str(name), str(name)),
                "space_group": str((general.get("SGData") or {}).get("SpGrp") or ""),
                "source_gpx": source,
                "uncertainty_status": "not_estimated",
            }
            sg = general.get("SGData") or {}
            # Exclude fitted numerical cells/coordinates from stable series identity.
            pointers = general.get("AtomPtrs") or [3, 1, 7, 9]
            atom_types = [
                (str(a[0]), str(a[pointers[1]]))
                for a in data.get("Atoms", [])
                if len(a) > pointers[1]
            ]
            identity = [str(name), sg.get("SpGrp"), sg.get("SGUniq"), atom_types]
            row["phase_key"] = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode()
            ).hexdigest()
            row["identity_basis"] = (
                "phase model name, symmetry setting and atom labels/types; review model equivalence before merging"
            )
            row["basis_key"] = json.dumps([sg.get("SpGrp"), sg.get("SGUniq"), sg.get("SGLaue")])
            row["metric_states"] = {key: "unknown" for key in CELL_KEYS}
            row["cell_covariance"] = None
            row["cell_jacobian"] = None
            row["covariance_status"] = "unavailable"
            row["fraction_state"] = "unknown"
            row["covariance_fresh"] = bool(covariance.get("freshCOV", False))
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
        row["weight_percent"] = (
            100 * max(0, weight) / total if total > 0 and weight is not None else None
        )
    rwp = (covariance.get("Rvals") or {}).get("Rwp")
    return rows, float(rwp) if rwp is not None and math.isfinite(float(rwp)) else None


def extract_lattice(path: Path, records, *, source: str, labels=None):
    rows, rwp = lattice_from_records(records, source=source, labels=labels)
    # Use the complete covariance and the symmetry-constrained transformation,
    # including correlations. The current GSAS-II getCellEsd implementation
    # transposes its cell/reciprocal-metric derivative in the length-error
    # contraction; it disagrees with analytic propagation for hexagonal cells.
    # Preserve that API result as a diagnostic, rather than using it to discard
    # an independently validated covariance or publishing inconsistent errors.
    gsas_root = os.environ.get("RADAR_PD_GSASII_ROOT") or str(
        Path(__file__).resolve().parents[1] / "GSAS-II"
    )
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
            row.update(cell_fit_information(phase, covariance))
            scale_names = [
                f"{phase.id}:{h.id}:Scale"
                for h in project.histograms()
                if phase.data.get("Histograms", {}).get(h.name, {}).get("Use", True)
            ]
            row["fraction_state"] = (
                "refined"
                if any(name in variables for name in scale_names)
                else "fixed_or_unverified"
            )
            prefix = f"{phase.id}::A"
            if not any(
                v.startswith(prefix) and v[len(prefix) :] in {"0", "1", "2", "3", "4", "5"}
                for v in variables
            ):
                continue
            errors = None
            try:
                _, errors = phase.get_cell_and_esd()
                row["gsas_api_cell_esds"] = {
                    key: float(errors[gsas_key])
                    for key, gsas_key in zip(CELL_KEYS, GSAS_KEYS)
                    if math.isfinite(float(errors[gsas_key]))
                }
            except (KeyError, ValueError, TypeError, RuntimeError, AttributeError):
                row["gsas_esd_comparison"] = "unavailable"
            if not covariance.get("freshCOV", False):
                row["uncertainty_status"] = "covariance_not_current"
                continue
            matrix = row.get("cell_covariance")
            if matrix is None:
                row["uncertainty_status"] = "covariance_unqualified"
                continue
            for i, key in enumerate(CELL_KEYS):
                error = math.sqrt(max(0.0, matrix[i][i]))
                row[key + "_esd"] = error if error > 0 else None
            if errors is not None and all(k in errors for k in GSAS_KEYS):
                row["gsas_esd_comparison"] = "different" if any(
                    not math.isclose(
                        math.sqrt(max(0, matrix[i][i])),
                        float(errors[gsas_key]),
                        rel_tol=1e-3,
                        abs_tol=2e-7,
                    )
                    for i, gsas_key in enumerate(GSAS_KEYS)
                ) else "agrees"
            row["covariance_status"] = "symmetry_jacobian"
            row["uncertainty_method"] = "J C J^T from final GSAS-II reciprocal-cell covariance"
            row["uncertainty_status"] = (
                "estimated"
                if any(row[k + "_esd"] is not None for k in CELL_KEYS)
                else "not_estimated"
            )
        add_cross_phase_covariance(rows,covariance)
        if project.histograms():
            value = project.histograms()[0].get_wR()
            if value is not None and math.isfinite(float(value)):
                rwp = float(value)
    except (ImportError, KeyError, ValueError, TypeError, RuntimeError, AttributeError) as exc:
        print(f"[WARN] Lattice ESDs unavailable for {path.name}: {exc}")
    return rows, rwp


def cell_fit_information(phase, covariance):
    """Propagate the final reciprocal-cell covariance through GSAS-II symmetry.

    Constraint-generated variables outside the direct cell vary-list are not
    approximated as independent observations. They remain unverified here.
    """
    import numpy as np
    from GSASII import GSASIIlattice as lattice, GSASIIstrIO as strio

    fresh = bool(covariance.get("freshCOV", False))
    prefix = f"{phase.id}::"
    names = [str(name) for name in covariance.get("varyList", [])]
    direct = [name for name in names if name in {prefix + f"A{i}" for i in range(6)}]
    states = {key: "unknown" for key in CELL_KEYS}
    output = {"metric_states": states, "covariance_fresh": fresh,
              "cell_covariance": None, "covariance_status": "unavailable"}
    if len(set(names)) != len(names) or len(covariance.get('variables', [])) != len(names):
        output['covariance_status'] = 'invalid_covariance'
        return output
    if not direct:
        # A flagged but absent cell may be constraint-derived or from an older
        # refinement. Neither case can be certified from stored flags alone.
        state = "unknown" if phase.data["General"]["Cell"][0] else "fixed"
        output["metric_states"] = {key: state for key in CELL_KEYS}
        return output
    sg = phase.data["General"]["SGData"]
    independent = {
        "-1": (0, 1, 2, 3, 4, 5), "mmm": (0, 1, 2),
        "4/m": (0, 2), "4/mmm": (0, 2),
        "3": (0, 2), "3m1": (0, 2), "31m": (0, 2),
        "6/m": (0, 2), "6/mmm": (0, 2),
        "3R": (0, 3), "3mR": (0, 3), "m3": (0,), "m3m": (0,),
    }.get(sg["SGLaue"])
    if sg["SGLaue"] == "2/m":
        independent = (0, 1, 2, {"a": 5, "b": 4, "c": 3}[sg["SGUniq"]])
    expected = {prefix + f"A{i}" for i in independent or ()}
    if set(direct) != expected:
        # Missing symmetry-independent variables can have external constraints
        # or frozen parameters. Their dependence is not encoded by cellFill.
        output["covariance_status"] = "cell_constraints_unqualified"
        return output
    base = lattice.cell2A(phase.data["General"]["Cell"][1:7])
    parameters = {prefix + f"A{i}": float(value) for i, value in enumerate(base)}
    parameters.update(
        {
            name: float(value)
            for name, value in zip(names, covariance.get("variables", []))
            if name in parameters
        }
    )

    def evaluate(values):
        reciprocal, _ = strio.cellFill(prefix, sg, values, {})
        return np.asarray([*lattice.A2cell(reciprocal), lattice.calc_V(reciprocal)], dtype=float)

    jacobian = np.zeros((7, len(direct)))
    for j, name in enumerate(direct):
        step = max(abs(parameters[name]) * 1e-6, 1e-10)
        plus, minus = dict(parameters), dict(parameters)
        plus[name] += step
        minus[name] -= step
        jacobian[:, j] = (evaluate(plus) - evaluate(minus)) / (2 * step)
    for i, key in enumerate(CELL_KEYS):
        if not np.any(np.abs(jacobian[i]) > 1e-7):
            jacobian[i] = 0.0
        states[key] = (
            ("refined" if fresh else "stale") if np.any(np.abs(jacobian[i]) > 1e-7) else "fixed"
        )
    output["cell_jacobian"] = jacobian.tolist()
    output['cell_variable_names'] = direct
    matrix = covariance.get("covMatrix")
    if not fresh or matrix is None:
        return output
    array = np.asarray(matrix, dtype=float)
    if array.shape != (len(names), len(names)) or not np.isfinite(array).all():
        return output
    indices = [names.index(name) for name in direct]
    selected = array[np.ix_(indices, indices)]
    scale = max(float(np.max(np.abs(selected))), np.finfo(float).tiny)
    if not np.allclose(selected, selected.T, rtol=1e-9, atol=scale*1e-12):
        output["covariance_status"] = "invalid_covariance"
        return output
    selected = (selected + selected.T) / 2
    if np.linalg.eigvalsh(selected).min() < -scale * 1e-9:
        output["covariance_status"] = "invalid_covariance"
        return output
    propagated = jacobian @ selected @ jacobian.T
    propagated = (propagated + propagated.T) / 2
    if np.isfinite(propagated).all():
        output["cell_covariance"] = propagated.tolist()
    return output


def add_cross_phase_covariance(rows,covariance):
    """Retain only qualified pair blocks from one final joint covariance.

    Two individual phase ESDs cannot reconstruct their cross correlation.
    Old results without this optional block remain values without ratio ESDs.
    """
    import numpy as np
    for row in rows:row['cross_phase_cell_covariance']={}
    if not covariance.get('freshCOV'):return
    names=[str(n) for n in covariance.get('varyList',[])]
    if len(set(names))!=len(names):return
    try:matrix=np.asarray(covariance.get('covMatrix'),dtype=float)
    except (TypeError,ValueError):return
    if matrix.shape!=(len(names),len(names)) or not np.isfinite(matrix).all():return
    keys=[r.get('phase_key') for r in rows]
    qualified=[r for r in rows if r.get('covariance_status')=='symmetry_jacobian'
        and r.get('covariance_fresh') and r.get('phase_key') and keys.count(r['phase_key'])==1]
    for i,a in enumerate(qualified):
        for b in qualified[i+1:]:
            av,bv=a.get('cell_variable_names') or [],b.get('cell_variable_names') or []
            if not av or not bv or any(v not in names for v in [*av,*bv]):continue
            try:ja,jb=np.asarray(a.get('cell_jacobian'),dtype=float),np.asarray(b.get('cell_jacobian'),dtype=float)
            except (TypeError,ValueError):continue
            if ja.shape!=(7,len(av)) or jb.shape!=(7,len(bv)):continue
            if not np.isfinite(ja).all() or not np.isfinite(jb).all():continue
            ai,bi=[names.index(n) for n in av],[names.index(n) for n in bv]
            union=list(dict.fromkeys([*ai,*bi]));joint=matrix[np.ix_(union,union)]
            scale=max(float(np.max(np.abs(joint))),np.finfo(float).tiny)
            if not np.allclose(joint,joint.T,rtol=1e-9,atol=scale*1e-12):continue
            if np.linalg.eigvalsh((joint+joint.T)/2).min()<-scale*1e-9:continue
            cross=ja@((matrix[np.ix_(ai,bi)]+matrix[np.ix_(bi,ai)].T)/2)@jb.T
            if not np.isfinite(cross).all():continue
            a['cross_phase_cell_covariance'][b['phase_key']]=cross.tolist()
            b['cross_phase_cell_covariance'][a['phase_key']]=cross.T.tolist()


def lattice_quality(rows):
    issues = []
    for row in rows:
        lengths = [row.get(k) for k in ("a", "b", "c")]
        angles = [row.get(k) for k in ("alpha", "beta", "gamma")]
        valid = all(v is not None and math.isfinite(v) and v > 0 for v in lengths)
        valid = valid and all(v is not None and math.isfinite(v) and 0 < v < 180 for v in angles)
        if valid:
            ca, cb, cg = (math.cos(math.radians(v)) for v in angles)
            valid = 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg > 1e-10
        if not valid or row.get("volume") is None or row["volume"] <= 0:
            issues.append(
                {"code": "invalid_cell", "message": f"{row['phase']}: invalid final unit cell"}
            )
        weight = row.get("weight_percent")
        if len(rows) > 1 and weight is not None and weight < 0.05:
            issues.append(
                {
                    "code": "vanishing_phase",
                    "message": f"{row['phase']}: final fraction below 0.05 wt%; review the accepted model",
                }
            )
        if row.get("covariance_status") == "gsas_esd_mismatch":
            issues.append(
                {
                    "code": "cell_covariance_review",
                    "message": f"{row['phase']}: GSAS-II API cell ESDs disagree with symmetry-Jacobian propagation; derived covariance is withheld pending review",
                }
            )
        elif row.get("uncertainty_status") in {"covariance_unqualified", "covariance_not_current"}:
            issues.append({"code": "cell_covariance_review",
                "message": f"{row['phase']}: final cell uncertainty is not qualified ({row.get('covariance_status')}); covariance is withheld"})
    return issues
