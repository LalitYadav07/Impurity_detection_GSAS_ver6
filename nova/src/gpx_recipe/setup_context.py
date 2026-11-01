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


def _host_execution(value, histogram_keys):
    """Project host capabilities, never arbitrary host diagnostics or results."""
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ("recipe_unit", "scan_enrollment")
              if isinstance(value.get(key), str)}
    if isinstance(value.get("histogram_keys"), list):
        result["histogram_keys"] = [key for key in histogram_keys if key in value["histogram_keys"]]
    fields = {
        "future_enrollment": ("supported", "requires_approval"),
        "output_artifacts": ("stage_gpx", "final_gpx", "source_plan_lineage"),
        "condition_mapping": ("preserve_reviewed_temperature_axis",),
    }
    for name, booleans in fields.items():
        record = value.get(name)
        if not isinstance(record, dict):
            continue
        projected = {key: record[key] for key in booleans if type(record.get(key)) is bool}
        if isinstance(record.get("scope"), str):
            projected["scope"] = record["scope"]
        if projected:
            result[name] = projected
    return result


def setup_context(context):
    result = _pick(context, (
        "model_signature", "source", "binding", "normalized_histograms",
        "source_constraints", "uiso_equivalence_groups", "runtime", "restraints",
    ))
    result["schema"] = "radar-gpx-setup-context/v1"
    result["purpose"] = "Model setup and typed recipe drafting only. Values are starting model values, not independently validated scientific results."
    host = _host_execution(context.get("host_execution"), [h["key"] for h in context["histograms"]])
    if host:
        result["host_execution"] = host
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
