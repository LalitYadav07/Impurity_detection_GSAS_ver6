"""Metadata qualification, conservative experiment splits, and coverage receipts.

An acquisition sample ID is never promoted to a verified physical specimen.
Unknown quantities remain unknown; a large inventory is not validation evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def read_inventory(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not records or records[0].get("schema") != "powgen-planning-inventory/v1":
        raise ValueError("POWGEN metadata inventory required")
    if records[-1].get("type") != "complete":
        raise ValueError("An incomplete inventory cannot be used for qualification")
    rows = [r for r in records if r.get("type") == "run"]
    keys = [(r["ipts"], r["run"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate acquisition records")
    return records, rows


def scalar(field):
    value = field.get("value") if isinstance(field, dict) else None
    return value if isinstance(value, (str, int, float)) and not isinstance(value, bool) else None


def quantity(field, units):
    value = scalar(field)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        return None
    unit = str(field.get("units", "")).lower().strip()
    return value * units[unit] if unit in units else None


def summarize_run(row):
    fields, logs = row["metadata"], row["logs"]
    mass = quantity(fields.get("entry/sample/mass"), {"g": 1., "gram": 1., "grams": 1., "mg": .001})
    duration = quantity(fields.get("entry/duration"), {"second": 1., "seconds": 1., "s": 1.})
    charge = quantity(fields.get("entry/proton_charge"), {"picocoulombs": 1/3.6e9, "picocoulomb": 1/3.6e9,
                       "coulomb": 1/.0036, "coulombs": 1/.0036, "c": 1/.0036,
                       "microampere_hour": 1., "uah": 1.})
    height = quantity(fields.get("entry/sample/height_in_container"), {"mm": .1, "cm": 1.})
    diameter = quantity(fields.get("entry/sample/interior_diameter"), {"mm": .1, "cm": 1.})
    formula = str(scalar(fields.get("entry/sample/chemical_formula")) or "").strip()
    sample_id = scalar(logs.get("SampleId")) or scalar(logs.get("SampleID"))
    sample_name = str(scalar(fields.get("entry/sample/name")) or scalar(logs.get("SampleName")) or "")
    title = str(scalar(fields.get("entry/title")) or "")
    # Retain changes such as gas recipe, loading or field history. Remove only
    # the changing readback suffix of an explicit producer-generated title.
    title_context = re.sub(r"( at temperature )[-+\d.]+( K\.)$", r"\1<readback>\2", title)
    temperature = logs.get("SampleTemp", {})
    tunit = str(temperature.get("units", "")).lower().strip()
    if "value" in temperature and type(temperature["value"]) in (int, float):
        tmin = tmax = temperature["value"]
    else:
        tmin, tmax = temperature.get("min"), temperature.get("max")
    steady = (tunit in {"k", "kelvin"} and type(tmin) in (int, float) and
              type(tmax) in (int, float) and tmin > 0 and tmax - tmin <= .5)
    # These are sampled extrema for long logs; they nominate repeats and never
    # certify thermodynamic equilibrium or unobserved portions of the log.
    calls = row.get("reduction", {}).get("calls", [])
    parameters = calls[0].get("parameters", {}) if len(calls) == 1 else {}
    calibration = next((parameters.get(k) for k in ("CalibrationFile", "CalFileName", "CalFilename")
                        if parameters.get(k)), None)
    setup = {"calibration": calibration,
             "characterization": parameters.get("CharacterizationRunsFile"),
             "binning": parameters.get("Binning"), "scale": parameters.get("ScaleData"),
             "correction": parameters.get("TypeOfCorrection"),
             "grouping": parameters.get("GroupingFileName"),
             "wavelength": scalar(logs.get("LambdaRequest")),
             "guide": scalar(logs.get("Guide_Position")),
             "sample_environment": scalar(logs.get("SampleEnvironment")),
             "field_log": logs.get("Mag08ActualField"),
             "proton_energy": scalar(logs.get("ProtonBeamEnergy"))}
    has_reduced = any(p["name"].endswith(".xye") for p in row["products"])
    missing = []
    for label, value in [("mass_with_units", mass), ("fill_height_with_units", height),
                         ("diameter_with_units", diameter), ("charge_with_units", charge),
                         ("duration_with_units", duration), ("formula", formula),
                         ("literal_reduction_calibration", calibration)]:
        if value is None or value == "":
            missing.append(label)
    start = scalar(fields.get("entry/start_time"))
    return {"ipts": row["ipts"], "run": row["run"], "sample_id": sample_id,
            "sample_name": sample_name, "formula_as_recorded": formula,
            "run_title": title, "acquisition_title_context": title_context,
            "specimen_identity": "unverified_acquisition_label", "mass_g": mass,
            "fill_height_cm": height, "diameter_cm": diameter,
            "duration_seconds": duration, "charge_uah": charge, "start_time": start,
            "setup": setup, "setup_sha256": identity(setup),
            "temperature_candidate_steady": steady,
            "temperature_sampled": bool(temperature.get("sampled")),
            "temperature_min_k": tmin if tunit in {"k", "kelvin"} else None,
            "temperature_max_k": tmax if tunit in {"k", "kelvin"} else None,
            "has_reduced_xye": has_reduced,
            "eligible_for_absolute_metadata_review": not missing and not row["issues"],
            "missing_absolute_metadata": missing,
            "absolute_validation_eligible": False,
            "absolute_validation_reason": "Illumination, specimen identity and authoritative structure still require independent review",
            "metadata_sha256": row.get("metadata_sha256"), "issues": row["issues"]}


def locked_split(rows, *, development_proposals=(), seed="powgen-planning-20261105", test_fraction=.25):
    """Whole-proposal split with global acquired-ID links and file-leakage gates.

    This is a metadata-only lock. A curator must add physical specimen/family
    links before declaring sample or material independence in a final benchmark.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("Declare an interior evaluation fraction")
    proposals = {r["ipts"] for r in rows}
    parent = {p: p for p in proposals}

    def find(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]; p = parent[p]
        return p

    def join(a, b):
        a, b = find(a), find(b)
        if a != b: parent[max(a, b)] = min(a, b)

    ids = defaultdict(set)
    for row in rows:
        if row.get("sample_id") is not None:
            ids[str(row["sample_id"])].add(row["ipts"])
    for values in ids.values():
        values = sorted(values)
        for other in values[1:]: join(values[0], other)
    components = defaultdict(list)
    for p in sorted(proposals): components[find(p)].append(p)
    development = set(development_proposals)
    result = {}
    for members in components.values():
        score = int(identity({"seed": seed, "proposals": members})[:16], 16)/2**64
        role = "development" if development.intersection(members) or score >= test_fraction else "locked_evaluation"
        result.update({p: role for p in members})
    return {"schema": "powgen-planning-corpus-lock/v1", "seed": seed,
            "proposal_roles": result, "connected_proposal_groups": list(components.values()),
            "physical_specimen_independence_verified": False,
            "material_family_independence_verified": False,
            "numerical_patterns_inspected_before_lock": False,
            "previously_inspected_proposals": sorted(development),
            "scope": "Metadata-only partition; prior inspection is explicitly assigned development",
            "assignments_sha256": identity(result)}


def coverage(rows):
    result = {"runs": len(rows), "proposals": len({r["ipts"] for r in rows}),
              "acquisition_sample_labels": len({(r["ipts"], str(r["sample_id"])) for r in rows if r["sample_id"] is not None}),
              "verified_independent_specimens": 0,
              "reduced_xye_runs": sum(r["has_reduced_xye"] for r in rows),
              "complete_absolute_metadata_candidates": sum(r["eligible_for_absolute_metadata_review"] for r in rows),
              "qualified_absolute_predictions": 0,
              "missing_metadata": dict(Counter(k for r in rows for k in r["missing_absolute_metadata"])),
              "wavelength_as_recorded": dict(Counter(str(r["setup"]["wavelength"]) for r in rows)),
              "status": "inventory_only_not_simulator_validation"}
    return result
