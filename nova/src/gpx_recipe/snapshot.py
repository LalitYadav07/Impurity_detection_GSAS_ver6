"""GSAS-II-side extraction. No diffraction arrays are sent to the planner."""
from pathlib import Path
import hashlib
import importlib
import json
import math
import os
import sys

from .studio_state import extract_gsasii_refinement_state


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def gsas():
    root = os.environ.get("GSASII_ROOT")
    if root and str(Path(root).resolve()) not in sys.path:
        sys.path.insert(0, str(Path(root).resolve()))
    try:
        return importlib.import_module("GSASII.GSASIIscriptable")
    except ModuleNotFoundError as exc:
        if exc.name not in {"GSASII", "GSASII.GSASIIscriptable"}:
            raise
        return importlib.import_module("GSASIIscriptable")


def snapshot(path):
    path = Path(path).resolve()
    G2 = gsas()
    project = G2.G2Project(gpxfile=str(path))
    if any(str(name).startswith("HKLF ") for name in project.data):
        raise ValueError("Mixed single-crystal/powder projects require a separate recipe implementation")
    symmetry = importlib.import_module((G2.__package__ + "." if G2.__package__ else "") + "GSASIIspc")
    state = extract_gsasii_refinement_state(path, project, G2, max_correlations=8)
    if state["extraction"]["status"] != "complete":
        raise ValueError("A complete effective-parameter snapshot is required: " + str(state["extraction"]["warnings"]))
    parameters = state["parameter_space"]["parameters"]
    histogram_keys = {h.name: f"h{i}" for i, h in enumerate(sorted(project.histograms(), key=lambda h: h.name))}
    phases = []
    for index, phase in enumerate(sorted(project.phases(), key=lambda p: p.name)):
        general = phase.data["General"]
        if general.get("Type", "nuclear") != "nuclear" or general.get("Modulated"):
            raise ValueError("Version 1 accepts ordinary nuclear phases; magnetic/modulated recipes need a separate implementation")
        atoms = []
        for atom_index, atom in enumerate(phase.atoms()):
            refs = [p for p in parameters if p["phase_id"] == int(phase.id) and p["atom_id"] == atom_index]
            site_symmetry = symmetry.SytSym(atom.coordinates, general["SGData"])[0]
            coordinate_terms = symmetry.GetCSxinel(site_symmetry)[0]
            atoms.append({"label": atom.label, "element": atom.element, "type": atom.type,
                          "coordinates": list(atom.coordinates), "occupancy": atom.occupancy,
                          "adp_type": atom.adp_flag, "adp": atom.ADP,
                          "multiplicity": atom.mult, "refinement_flags": atom.refinement_flags.strip(),
                          "site_symmetry": site_symmetry,
                          "position_dof": len(set(coordinate_terms) - {0}),
                          "variables": [{k: p[k] for k in ("name", "group", "current_value", "configured_now", "effective_now", "status")}
                                        for p in refs]})
        labels = [a["label"] for a in atoms]
        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate atom labels require explicit project cleanup before planning")
        phases.append({"key": f"p{index}", "name": phase.name, "space_group": general["SGData"]["SpGrp"],
                       "cell": phase.get_cell(), "cell_refined": bool(general["Cell"][0]), "atoms": atoms,
                       "phase_scales": [{"histogram": histogram_keys[name], "value": hap["Scale"][0], "refine": bool(hap["Scale"][1])}
                                        for name, hap in phase.data["Histograms"].items() if name in histogram_keys and hap.get("Use", True)],
                       "histogram_phase_models": {histogram_keys[name]: json_safe(hap) for name, hap in phase.data["Histograms"].items() if name in histogram_keys},
                       "histograms": sorted(name for name, hap in phase.data["Histograms"].items() if hap.get("Use", True))})
    histograms = []
    for index, hist in enumerate(sorted(project.histograms(), key=lambda h: h.name)):
        bg = hist.data["Background"]
        inst = hist.InstrumentParameters
        instrument = {k: json_safe(v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else v) for k, v in inst.items()}
        histograms.append({"key": f"h{index}", "name": hist.name, "instrument": instrument,
                           "limits": json_safe(hist.data["Limits"]), "point_count": len(hist.getdata("X")),
                           "background": {"function": bg[0][0], "coefficients": int(bg[0][2]), "values": bg[0][3:],
                                          "refine": bool(bg[0][1]), "fixed_point_count": len(bg[1].get("FixedPoints", []))},
                           "rwp": hist.get_wR(), "sample_parameters": json_safe(hist.SampleParameters)})
    if not histograms or not phases:
        raise ValueError("The project needs powder histograms and at least one nuclear phase")
    # Labels and atom topology bind recipes independently of mutable phase IDs,
    # cell values, atom coordinates, fitted scales and scan-specific filenames.
    # Histogram names map by stable h-key; instrument/calibration must match.
    binding = {"phases": [{"key": p["key"], "name": p["name"], "space_group": p["space_group"],
                            "atoms": [{k: a[k] for k in ("label", "element", "adp_type", "multiplicity", "site_symmetry")} for a in p["atoms"]]} for p in phases],
               "histograms": [{"key": h["key"], "instrument": h["instrument"],
                               "background": {k: h["background"][k] for k in ("function", "coefficients")}} for h in histograms]}
    covariance = {k: v for k, v in state["covariance"].items() if k not in {"matrix", "correlation_matrix"}}
    result = {"schema": "radar-gpx-context/v1", "source": {"name": path.name, "sha256": state["source"]["sha256"]},
              "model_signature": digest(binding), "binding": binding,
              "runtime": {"module": G2.__name__, "version": str(project.data.get("Controls", {}).get("data", {}).get("LastSavedUsing", "unknown"))},
              "histograms": histograms, "phases": phases, "constraints": state["constraints"],
              "restraints": json_safe(project.data.get("Restraints", {}).get("data", {})),
              "parameter_space": {k: v for k, v in state["parameter_space"].items() if k != "parameters"},
              "parameter_records": [{k: p[k] for k in ("name", "group", "phase_id", "histogram_id", "atom_id", "current_value", "configured_now", "effective_now", "lower_bound", "upper_bound", "frozen_at_limit")} for p in parameters],
              "controls": json_safe(state["controls"]),
              "covariance_summary": covariance, "extraction": state["extraction"]}
    result = json_safe(result)
    result["context_sha256"] = digest(result)
    return result
