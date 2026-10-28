"""GSAS-II-side extraction. No diffraction arrays are sent to the planner."""
from pathlib import Path
import hashlib
import importlib
import json
import math
import os
import sys

from .studio_state import extract_gsasii_refinement_state
from .advanced import occupancy_groups, _scale_sum_exists
from .identity import model_topology,bound_binding,explicit_constraints, constraint_binding, uiso_groups


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


def runtime_info(G2, project):
    """The executing API version is distinct from a project's last save label."""
    try:
        module=importlib.import_module((G2.__package__+'.' if G2.__package__ else '')+'GSASIIpath')
        version=str(module.GetVersionNumber())
    except (ImportError,AttributeError,TypeError,ValueError):version='unknown'
    return {'module':G2.__name__,'version':version,
            'source_saved_using':str(project.data.get('Controls',{}).get('data',{}).get('LastSavedUsing','unknown'))}


def snapshot(path):
    path = Path(path).resolve()
    G2 = gsas()
    project = G2.G2Project(gpxfile=str(path))
    if project.data.get("Controls", {}).get("data", {}).get("Seq Data"):
        raise ValueError("Sequential-refinement controls are active. This worker executes joint powder stages; review an explicitly converted joint or per-scan GPX before using it. Sequential execution is not implemented.")
    if any(str(name).startswith("HKLF ") for name in project.data):
        raise ValueError("Mixed single-crystal/powder projects require a separate recipe implementation")
    if not project.histograms() or not project.phases():
        raise ValueError("A Rietveld recipe needs powder histograms and at least one structural phase. A single-peak or phase-free project cannot be refined with this worker.")
    histogram_names={h.name for h in project.histograms()}
    if not any(name in histogram_names and hap.get('Use',True)
               for phase in project.phases() for name,hap in phase.data.get('Histograms',{}).items()):
        raise ValueError("No structural phase is actively linked to a powder histogram. Review the phase–histogram setup in GSAS-II; the recipe worker will not invent links or enable excluded phases.")
    # Unsupported model families are rejected before parameter reconstruction.
    for phase in project.phases():
        general=phase.data['General']
        if general.get('Type','nuclear')!='nuclear' or general.get('Modulated'):
            raise ValueError("Version 1 accepts ordinary nuclear phases; magnetic/modulated recipes need a separate implementation")
    symmetry = importlib.import_module((G2.__package__ + "." if G2.__package__ else "") + "GSASIIspc")
    state = extract_gsasii_refinement_state(path, project, G2, max_correlations=8)
    if state["extraction"]["status"] != "complete":
        raise ValueError("A complete effective-parameter snapshot is required: " + str(state["extraction"]["warnings"]))
    parameters = state["parameter_space"]["parameters"]
    histogram_keys = {h.name: f"h{i}" for i, h in enumerate(sorted(project.histograms(), key=lambda h: h.name))}
    phases = []
    for index, phase in enumerate(sorted(project.phases(), key=lambda p: p.name)):
        general = phase.data["General"]
        if phase.data.get("Pawley ref") or any(h.get("LeBail") for h in phase.data["Histograms"].values()):
            raise ValueError("Le Bail/Pawley intensity models require a dedicated recipe implementation; their intensities cannot be silently frozen or treated as ordinary Rietveld parameters")
        if general.get("Type", "nuclear") != "nuclear" or general.get("Modulated"):
            raise ValueError("Version 1 accepts ordinary nuclear phases; magnetic/modulated recipes need a separate implementation")
        atoms = []
        for atom_index, atom in enumerate(phase.atoms()):
            refs = [p for p in parameters if p["phase_id"] == int(phase.id) and p["atom_id"] == atom_index]
            site_symmetry = symmetry.SytSym(atom.coordinates, general["SGData"])[0]
            coordinate_terms = symmetry.GetCSxinel(site_symmetry)[0]
            atoms.append({"label": atom.label, "native_id": atom_index, "element": atom.element, "type": atom.type,
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
        laue = general["SGData"]["SGLaue"]
        lat = importlib.import_module((G2.__package__ + "." if G2.__package__ else "") + "GSASIIlattice")
        phases.append({"key": f"p{index}", "native_id": int(phase.id), "name": phase.name, "space_group": general["SGData"]["SpGrp"],
                       "global_texture": json_safe(general.get("SH Texture", {})),
                       "laue": laue, "mass": general.get("Mass"),
                       "occupancy_groups": occupancy_groups(atoms),
                       "texture_orders": [order for order in (2, 4, 6, 8) if lat.GenSHCoeffT(laue, "0", order, False)],
                       "cell": phase.get_cell(), "cell_refined": bool(general["Cell"][0]), "atoms": atoms,
                       "phase_scales": [{"histogram": histogram_keys[name], "value": hap["Scale"][0], "refine": bool(hap["Scale"][1])}
                                        for name, hap in phase.data["Histograms"].items() if name in histogram_keys and hap.get("Use", True)],
                       "histogram_phase_models": {histogram_keys[name]: json_safe(hap) for name, hap in phase.data["Histograms"].items() if name in histogram_keys},
                       "histograms": sorted(name for name, hap in phase.data["Histograms"].items() if name in histogram_keys and hap.get("Use", True))})
    histograms = []
    for index, hist in enumerate(sorted(project.histograms(), key=lambda h: h.name)):
        bg = hist.data["Background"]
        inst = hist.InstrumentParameters
        instrument = {k: json_safe(v[1] if isinstance(v, (list, tuple)) and len(v) > 1 else v) for k, v in inst.items()}
        histograms.append({"key": f"h{index}", "native_id": int(hist.id), "name": hist.name, "instrument": instrument,
                           "limits": json_safe(hist.data["Limits"]), "point_count": len(hist.getdata("X")),
                           "background": {"function": bg[0][0], "coefficients": int(bg[0][2]), "values": bg[0][3:],
                                          "refine": bool(bg[0][1]), "fixed_point_count": len(bg[1].get("FixedPoints", []))},
                           "rwp": hist.get_wR(), "sample_parameters": json_safe(hist.SampleParameters)})
        histograms[-1]['sample_parameter_provenance']='GSAS-II project/model fields, which may contain defaults. These are not independently verified acquisition logs; do not infer measured temperature, pressure or other scan conditions from them without an authoritative run-log mapping.'
    if not histograms or not phases:
        raise ValueError("The project needs powder histograms and at least one nuclear phase")
    normalized_histograms = []
    source_constraints = explicit_constraints(project,json_safe)
    shared_uiso_groups = uiso_groups(source_constraints,phases,histograms)
    for h in histograms:
        linked = [p for p in phases if h["name"] in p["histograms"]]
        if len(linked) > 1:
            names = [f"{p['native_id']}:{h['native_id']}:Scale" for p in linked]
            try:
                if _scale_sum_exists(project, names): normalized_histograms.append(h["key"])
            except ValueError:
                pass  # A different existing equation is not sum-to-one normalization.
    # Labels and atom topology bind recipes independently of mutable phase IDs,
    # cell values, atom coordinates, fitted scales and scan-specific filenames.
    # Histogram names map by stable h-key; instrument/calibration must match.
    binding = {"phases": [{"key": p["key"], "name": p["name"], "space_group": p["space_group"],
                            "atoms": [{k: a[k] for k in ("label", "element", "adp_type", "multiplicity", "site_symmetry")} for a in p["atoms"]]} for p in phases],
               "source_constraints": constraint_binding(source_constraints,phases,histograms),
               "source_bounds": bound_binding(project.data.get("Controls",{}).get("data",{}),phases,histograms,json_safe),
               "occupancy_groups": {p["key"]:p["occupancy_groups"] for p in phases},
               "model_topology": {p["key"]:model_topology(p) for p in phases},
               "source_restraints": json_safe(project.data.get("Restraints",{}).get("data",{})),
               "phase_histogram_links": {p['key']:[h['key'] for h in histograms if h['name'] in p['histograms']] for p in phases},
               "histograms": [{"key": h["key"], "instrument": h["instrument"],
                               "background": {k: h["background"][k] for k in ("function", "coefficients")}} for h in histograms]}
    covariance = {k: v for k, v in state["covariance"].items() if k not in {"matrix", "correlation_matrix"}}
    result = {"schema": "radar-gpx-context/v1", "source": {"name": path.name, "sha256": state["source"]["sha256"]},
              "model_signature": digest(binding), "binding": binding, "normalized_histograms": normalized_histograms,
              "source_constraints": source_constraints, "uiso_equivalence_groups": shared_uiso_groups,
              "runtime": runtime_info(G2,project),
              "histograms": histograms, "phases": phases, "constraints": state["constraints"],
              "restraints": json_safe(project.data.get("Restraints", {}).get("data", {})),
              "parameter_space": {k: v for k, v in state["parameter_space"].items() if k != "parameters"},
              "parameter_records": [{k: p[k] for k in ("name", "group", "phase_id", "histogram_id", "atom_id", "current_value", "configured_now", "effective_now", "lower_bound", "upper_bound", "frozen_at_limit")} for p in parameters],
              "controls": json_safe(state["controls"]),
              "covariance_summary": covariance, "extraction": state["extraction"]}
    result = json_safe(result)
    result["context_sha256"] = digest(result)
    return result
