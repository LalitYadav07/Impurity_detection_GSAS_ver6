"""Closed, typed vocabulary: each stage defines its COMPLETE active set."""
import copy
import json
import math
import re

from . import SCHEMA
from .snapshot import digest

BACKGROUNDS = ["chebyschev-1", "chebyschev", "cosine", "Q^2 power series", "Q^-2 power series", "lin interpolate", "inv interpolate", "log interpolate"]
KINDS = {"background", "phase_scale", "histogram_scale", "preferred_orientation", "cell", "atom"}


def exact_keys(value, allowed, required=()):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError("Unexpected or missing fields: " + repr(value)[:200])


def validate_plan(plan, context):
    exact_keys(plan, {"schema", "model_signature", "request", "assumptions", "questions", "unsupported", "stages"},
               {"schema", "model_signature", "request", "assumptions", "questions", "unsupported", "stages"})
    if plan["schema"] != SCHEMA or plan["model_signature"] != context["model_signature"]:
        raise ValueError("Plan does not match the inspected GPX model")
    if not isinstance(plan["request"], str) or not 0 < len(plan["request"]) <= 16000:
        raise ValueError("A bounded user request is required")
    for field in ("assumptions", "questions", "unsupported"):
        if not isinstance(plan[field], list) or any(not isinstance(v, str) or len(v) > 3000 for v in plan[field]):
            raise ValueError("Invalid " + field)
    if (not isinstance(plan["stages"], list) or len(plan["stages"]) > 20
            or (not plan["stages"] and not (plan["questions"] or plan["unsupported"]))):
        raise ValueError("Use 1–20 explicit stages, or no stages when clarification/unsupported operations prevent a plan")
    phases = {p["key"]: p for p in context["phases"]}
    hists = {h["key"]: h for h in context["histograms"]}
    stage_ids = set()
    normalized = set(context.get("normalized_histograms", []))
    for stage in plan["stages"]:
        from .advanced import STAGE_FIELDS, validate_stage
        exact_keys(stage, {"id", "title", "cycles", "free", "background", "constraints"} | STAGE_FIELDS, {"id", "title", "cycles", "free", "constraints"})
        if not isinstance(stage["id"], str) or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_-]{0,40}", stage["id"]) or stage["id"] in stage_ids:
            raise ValueError("Stage IDs must be unique safe identifiers")
        stage_ids.add(stage["id"])
        if not isinstance(stage["title"], str) or not 0 < len(stage["title"]) <= 500:
            raise ValueError("Stage title required")
        if type(stage["cycles"]) is not int or not 1 <= stage["cycles"] <= 30:
            raise ValueError("Stage cycles must be an integer from 1 to 30")
        if not isinstance(stage["free"], list) or not stage["free"] or len(stage["free"]) > 500:
            raise ValueError("Every stage needs a bounded explicit active set")
        seen = set()
        for target in stage["free"]:
            kind = target.get("kind") if isinstance(target, dict) else None
            fields = {"background": {"kind", "histogram"}, "histogram_scale": {"kind", "histogram"},
                      "preferred_orientation": {"kind", "phase", "histogram"}, "phase_scale": {"kind", "phase", "histogram"},
                      "cell": {"kind", "phase"}, "atom": {"kind", "phase", "atom", "flags"}}
            if kind not in KINDS:
                raise ValueError("Unsupported active parameter family: " + str(kind))
            exact_keys(target, fields[kind], fields[kind])
            if "histogram" in target and target["histogram"] not in hists:
                raise ValueError("Unknown histogram " + str(target["histogram"]))
            phase = phases.get(target.get("phase"))
            if "phase" in target and phase is None:
                raise ValueError("Unknown phase " + str(target["phase"]))
            if kind in {"phase_scale", "preferred_orientation"} and hists[target["histogram"]]["name"] not in phase["histograms"]:
                raise ValueError("Phase is not active in the selected histogram")
            if kind == "atom":
                matches = [a for a in phase["atoms"] if a["label"] == target["atom"]]
                if len(matches) != 1:
                    raise ValueError("Unknown/ambiguous atom " + str(target["atom"]))
                flags = target["flags"]
                if not isinstance(flags, str) or not re.fullmatch(r"[FXU]{1,3}", flags) or len(set(flags)) != len(flags):
                    raise ValueError("Atom flags must select F (occupancy), X (position), U (isotropic ADP)")
                if "U" in flags and matches[0]["adp_type"] != "I":
                    raise ValueError("Version 1 supports isotropic displacement parameters only")
                if "X" in flags and matches[0].get("position_dof") == 0:
                    raise ValueError(f"{target['phase']}:{target['atom']} has no symmetry-independent positions")
            identity = (kind, target.get("phase"), target.get("histogram"), target.get("atom"))
            if identity in seen:
                raise ValueError("Duplicate target; combine flags explicitly")
            seen.add(identity)
        bg = stage.get("background")
        if bg is not None:
            exact_keys(bg, {"histogram", "function", "coefficients", "fixed_points"}, {"histogram", "function", "coefficients", "fixed_points"})
            if bg["histogram"] not in hists or bg["function"] not in BACKGROUNDS:
                raise ValueError("Unsupported background selection")
            if type(bg["coefficients"]) is not int or not 1 <= bg["coefficients"] <= 36:
                raise ValueError("Background coefficients must be an integer from 1 to 36")
            if bg["fixed_points"] not in {"preserve", "remove"}:
                raise ValueError("Choose preserve/remove explicitly for existing background support points")
        if not isinstance(stage["constraints"], list) or len(stage["constraints"]) > 30:
            raise ValueError("Constraint list is required")
        for constraint in stage["constraints"]:
            exact_keys(constraint, {"kind", "terms", "total", "reason"}, {"kind", "terms", "total", "reason"})
            if constraint["kind"] != "occupancy_sum" or not isinstance(constraint["reason"], str) or not constraint["reason"].strip():
                raise ValueError("Version 1 new constraints require an explained occupancy_sum")
            if type(constraint["total"]) not in (int, float) or not math.isfinite(constraint["total"]):
                raise ValueError("Constraint total must be finite")
            if not isinstance(constraint["terms"], list) or not 2 <= len(constraint["terms"]) <= 20:
                raise ValueError("Occupancy relation needs 2–20 explicit terms")
            identities = set()
            for term in constraint["terms"]:
                exact_keys(term, {"phase", "atom", "coefficient"}, {"phase", "atom", "coefficient"})
                if term["phase"] not in phases or term["atom"] not in {a["label"] for a in phases[term["phase"]]["atoms"]}:
                    raise ValueError("Unknown atom in occupancy relation")
                if type(term["coefficient"]) not in (float, int) or not math.isfinite(term["coefficient"]) or term["coefficient"] == 0:
                    raise ValueError("Constraint coefficient must be finite and nonzero")
                identity = (term["phase"], term["atom"])
                if identity in identities:
                    raise ValueError("Duplicate occupancy relation term")
                identities.add(identity)
            active_occupancies = {(t["phase"], t["atom"]) for t in stage["free"]
                                  if t["kind"] == "atom" and "F" in t["flags"]}
            if len(identities & active_occupancies) == 1:
                raise ValueError("A new occupancy equation with exactly one released occupant and all other terms held fixed removes that occupant's refinement freedom. Do not invent a site-total equation; preserve the user's requested fixed/free choices or ask for clarification.")
        validate_stage(stage, context, normalized)
    return copy.deepcopy(plan)


def plan_hash(plan):
    return digest(plan)


def target_labels(target):
    if target["kind"] == "atom":
        return {f"{target['phase']}:{target['atom']}:{flag}" for flag in target["flags"]}
    return {":".join(str(target.get(k, "")) for k in ("kind", "phase", "histogram")).rstrip(":")}


def stage_table(plan, context=None):
    previous = set()
    if context:
        for h in context["histograms"]:
            if h.get("background", {}).get("refine"):
                previous.update(target_labels({"kind": "background", "histogram": h["key"]}))
            if h.get("sample_parameters", {}).get("Scale", [0, False])[1]:
                previous.update(target_labels({"kind": "histogram_scale", "histogram": h["key"]}))
        for p in context["phases"]:
            if p.get("cell_refined"):
                previous.update(target_labels({"kind": "cell", "phase": p["key"]}))
            for scale in p.get("phase_scales", []):
                if scale["refine"]:
                    previous.update(target_labels({"kind": "phase_scale", "phase": p["key"], "histogram": scale["histogram"]}))
            for atom in p["atoms"]:
                flags = atom.get("refinement_flags", "").strip()
                if flags:
                    previous.update(target_labels({"kind": "atom", "phase": p["key"], "atom": atom["label"], "flags": flags}))
    rows = []
    for stage in plan["stages"]:
        current = set().union(*(target_labels(t) for t in stage["free"]))
        rows.append({"id": stage["id"], "title": stage["title"], "free": sorted(current),
                     "newly_free": sorted(current - previous), "newly_fixed": sorted(previous - current),
                     "fixed_policy": "Every other ordinary refinement parameter is fixed; original constraints/restraints remain.",
                     "cycles": stage["cycles"], "background": stage.get("background"), "new_constraints": stage["constraints"],
                     "edits": {k: stage[k] for k in ("fit_ranges", "exclusions", "bounds", "phase_normalization", "texture") if stage.get(k)}})
        previous = current
    return rows
