"""The explicit model-setup view sent to a recipe drafting model.

Native snapshots retain numerical outcomes for auditing and the GUI. Drafting
uses topology, current starting values, constraints and available operations;
fit statistics, uncertainty results and hypothesis rankings stay internal.
Allowlisted fields prevent new diagnostic fields leaking into future prompts.
"""
from copy import deepcopy

from .snapshot import digest


def _pick(record, fields):
    return {key: deepcopy(record[key]) for key in fields if key in record}


def setup_context(context):
    result = _pick(context, (
        "model_signature", "source", "binding", "normalized_histograms",
        "source_constraints", "uiso_equivalence_groups", "runtime", "restraints",
    ))
    result["schema"] = "radar-gpx-setup-context/v1"
    result["purpose"] = "Model setup and typed recipe drafting only. Values are starting model values, not independently validated scientific results."
    result["histograms"] = []
    for histogram in context["histograms"]:
        result["histograms"].append(_pick(histogram, (
            "key", "native_id", "name", "instrument", "limits", "point_count",
            "background", "sample_parameters", "sample_parameter_provenance",
        )))
    result["phases"] = []
    for phase in context["phases"]:
        projected = _pick(phase, (
            "key", "native_id", "name", "space_group", "global_texture", "laue",
            "mass", "occupancy_groups", "texture_orders", "cell", "cell_refined",
            "phase_scales", "histograms",
        ))
        projected["atoms"] = []
        for atom in phase["atoms"]:
            item = _pick(atom, (
                "label", "native_id", "element", "type", "coordinates", "occupancy",
                "adp_type", "adp", "multiplicity", "refinement_flags",
                "site_symmetry", "position_dof",
            ))
            if "variables" in atom:
                item["variables"] = [_pick(row, ("name", "group", "current_value", "configured_now"))
                                     for row in atom["variables"]]
            projected["atoms"].append(item)
        if "histogram_phase_models" in phase:
            projected["histogram_phase_models"] = {key: _pick(hap, (
                "Use", "Scale", "Pref.Ori.", "Size", "Mustrain", "HStrain", "Extinction",
                "Babinet", "LeBail", "Layer Disp", "Fix FXU", "Type", "newLeBail",
            )) for key, hap in phase["histogram_phase_models"].items()}
        result["phases"].append(projected)
    if "parameter_records" in context:
        result["parameter_records"] = [_pick(row, (
            "name", "group", "phase_id", "histogram_id", "atom_id", "current_value",
            "configured_now", "lower_bound", "upper_bound",
        )) for row in context["parameter_records"]]
    if "controls" in context:
        result["controls"] = _pick(context["controls"], (
            "parmMinDict", "parmMaxDict", "Seq Data", "seqCopy", "Reverse Seq",
            "max cyc", "deriv type", "min dM/M",
        ))
    # The full snapshot has its own hash in the native audit. This hash binds
    # only the decoded projection actually supplied to the model.
    result["context_sha256"] = digest(result)
    return result
