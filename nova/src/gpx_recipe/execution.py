"""Deterministic GSAS-II execution of an explicitly approved active-set plan."""
import hashlib
import json
from pathlib import Path
import time
import importlib

from .contracts import validate_plan, plan_hash, stage_table
from .snapshot import snapshot, gsas, json_safe
from .studio_state import extract_gsasii_refinement_state
from . import advanced
from .effects import parameter_changes, require_fixed_values_unchanged,require_native_covariance,constraint_bound_audit,require_constraint_bounds,blocking_frozen_variables


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze(project):
    """Studio's tested reset: no accidental cumulative ordinary flags."""
    for hist in project.histograms():
        hist.clear_refinements({"Background": {"peaks": True}})
        # In this native API, Background=True clears only the polynomial flag.
        # Explicitly freeze independent Debye and imported-pattern terms too;
        # keep every numerical value and FixedPoints support point unchanged.
        background_terms=hist.data['Background'][1]
        for term in background_terms.get('debyeTerms',[]):
            for index in range(1,len(term),2):term[index]=False
        imported=background_terms.get('background PWDR')
        if isinstance(imported,list) and len(imported)>2:imported[2]=False
        for category, data, flag in (("Sample Parameters", hist.SampleParameters, 1),
                                     ("Instrument Parameters", hist.InstrumentParameters, 2)):
            keys = [k for k, v in data.items() if isinstance(v, (list, tuple)) and len(v) > flag and isinstance(v[flag], bool)]
            if keys:
                hist.clear_refinements({category: keys})
        # All independent background submodels are held unless explicitly supported and requested.
    for phase in project.phases():
        phase.clear_refinements({"Cell": True})
        phase.set_refinements({"Atoms": {"all": ""}})
        texture = phase.data["General"].get("SH Texture", {})
        if texture.get("SH Coeff"): texture["SH Coeff"][0] = False
        for key in ("Sample omega", "Sample chi", "Sample phi"):
            if key in texture and len(texture[key]) > 1: texture[key][0] = False
        phase.clear_HAP_refinements({**{k: True for k in ("Scale", "Pref.Ori.", "Size", "Mustrain", "HStrain", "Extinction")},
                                     "Babinet": ["BabA", "BabU"]})
        if phase.data.get("Pawley ref") or any(h.get("LeBail") for h in phase.data["Histograms"].values()):
            raise ValueError("Le Bail/Pawley projects need a dedicated recipe; their intensity variables cannot be hidden")


def bindings(project, context):
    phases = {p["key"]: project.phase(p["name"]) for p in context["phases"]}
    hists = {h["key"]: project.histogram(h["name"]) for h in context["histograms"]}
    return phases, hists


def configure(project, stage, context, added_constraints):
    freeze(project)
    phases, hists = bindings(project, context)
    bg = stage.get("background")
    if bg:
        hist = hists[bg["histogram"]]
        hist.set_refinements({"Background": {"type": bg["function"], "no. coeffs": bg["coefficients"], "refine": False}})
        if bg["fixed_points"] == "remove":
            hist.data["Background"][1].pop("FixedPoints", None)
    for target in stage["free"]:
        kind = target["kind"]
        if kind == "background":
            hists[target["histogram"]].set_refinements({"Background": True})
        elif kind == "histogram_scale":
            hist = hists[target["histogram"]]
            hist.set_refinements({"Sample Parameters": ["Scale"]})
            variable = f":{hist.id}:Scale"
            lo = project.get_Controls("parmMin", variable=variable)
            project.set_Controls("parmMin", max(0.0, lo) if lo is not None else 0.0, variable=variable)
        elif kind == "preferred_orientation":
            phases[target["phase"]].set_HAP_refinements({"Pref.Ori.": True}, histograms=[hists[target["histogram"]].name])
        elif kind == "phase_scale":
            phase, hist = phases[target["phase"]], hists[target["histogram"]]
            phase.set_HAP_refinements({"Scale": True}, histograms=[hist.name])
            variable = f"{phase.id}:{hist.id}:Scale"
            lo = project.get_Controls("parmMin", variable=variable)
            project.set_Controls("parmMin", max(0.0, lo) if lo is not None else 0.0, variable=variable)
        elif kind == "cell":
            phases[target["phase"]].set_refinements({"Cell": True})
        elif kind == "atom":
            phase = phases[target["phase"]]
            phase.set_refinements({"Atoms": {target["atom"]: target["flags"]}})
            atom_record = next(a for p in context["phases"] if p["key"] == target["phase"] for a in p["atoms"] if a["label"] == target["atom"])
            # Resolve the exact observed variable; do not guess numeric phase or
            # atom IDs when establishing physical occupancy/ADP bounds.
            for row in atom_record["variables"]:
                if ("F" in target["flags"] and row["group"] == "occupancy"):
                    lo = project.get_Controls("parmMin", variable=row["name"])
                    hi = project.get_Controls("parmMax", variable=row["name"])
                    project.set_Controls("parmMin", max(0.0, lo) if lo is not None else 0.0, variable=row["name"])
                    capacity = advanced.site_capacity(context, target["phase"], target["atom"], phase, stage["free"])
                    project.set_Controls("parmMax", min(capacity, hi) if hi is not None else capacity, variable=row["name"])
                if "U" in target["flags"] and row["group"] == "atomic_displacement":
                    lo = project.get_Controls("parmMin", variable=row["name"])
                    project.set_Controls("parmMin", max(0.0, lo) if lo is not None else 0.0, variable=row["name"])
    for constraint in stage["constraints"]:
        identity = json.dumps(constraint, sort_keys=True)
        if identity in added_constraints:
            continue
        if constraint['kind']=='uiso_equal':
            names=[]
            for term in constraint['atoms']:
                atom=next(a for p in context['phases'] if p['key']==term['phase'] for a in p['atoms'] if a['label']==term['atom'])
                variables=[v['name'] for v in atom['variables'] if v['group']=='atomic_displacement']
                if len(variables)!=1:raise ValueError('Could not resolve a unique isotropic ADP variable')
                names.append(variables[0])
            if not advanced.has_equality(project,names):project.add_EquivConstr(names,multlist=[1.0]*len(names),override=False)
            added_constraints.add(identity)
            continue
        names, multipliers = [], []
        for term in constraint["terms"]:
            atom = next(a for p in context["phases"] if p["key"] == term["phase"] for a in p["atoms"] if a["label"] == term["atom"])
            names_for_atom = [v["name"] for v in atom["variables"] if v["group"] == "occupancy"]
            if len(names_for_atom) != 1:
                raise ValueError("Could not resolve a unique observed occupancy variable")
            names.append(names_for_atom[0]); multipliers.append(term["coefficient"])
        if not advanced.has_equation(project, names, multipliers, constraint["total"]):
            project.add_EqnConstr(constraint["total"], names, multlist=multipliers, override=False)
        added_constraints.add(identity)
    advanced.configure(project, stage, context, phases, hists, added_constraints)
    project.set_Controls("cycles", stage["cycles"])
    # Preserve the original constraints, restraints, fit range and calibrated
    # profile. Every stage rebuilds its ordinary free set from zero.


def parameter_matches_target(row, target, context):
    group = row["group"]
    phase = next((p for p in context["phases"] if p["key"] == target.get("phase")), None)
    hist = next((h for h in context["histograms"] if h["key"] == target.get("histogram")), None)
    kind = target["kind"]
    if kind in {"background", "histogram_scale"}:
        scope=hist is not None and row["histogram_id"] == hist["native_id"] and row["phase_id"] is None
        if kind=='background':
            family=row.get('family',row['name'].split(':')[2] if len(row['name'].split(':'))>2 else '')
            return scope and group=='background' and family.startswith('Back;')
        return scope and group=='histogram_scale'
    if phase is None or row["phase_id"] != phase["native_id"]: return False
    if kind == "cell": return group == "lattice" and row["histogram_id"] is None
    if kind in {"phase_scale", "preferred_orientation"}:
        return hist is not None and row["histogram_id"] == hist["native_id"] and group == ("phase_fraction" if kind == "phase_scale" else "preferred_orientation")
    if kind == "atom":
        atom = next(a for a in phase["atoms"] if a["label"] == target["atom"])
        flags = {"atomic_coordinates": "X", "atomic_displacement": "U", "occupancy": "F"}
        return row["atom_id"] == atom["native_id"] and flags.get(group, "!") in target["flags"]
    return False


def check_preflight(state, stage, context):
    if state["extraction"]["status"] != "complete" or state["constraints"].get("error") or state["constraints"].get("multiplier_error"):
        raise ValueError("GSAS-II constraint/parameter preflight failed: " + str(state["constraints"]))
    space = state["parameter_space"]
    blocked=set(space.get('frozen_variables',[])) & (set(space.get('configured_vary',[]))|set(space.get('effective_vary',[])))
    if blocked:
        raise ValueError('Requested variables are already frozen by preserved source bounds/history: '+str(sorted(blocked))+'. Review an explicit reset in the source project; this recipe never silently clears holds.')
    if not space["effective_vary"]:
        raise ValueError("The requested stage has no effective independent refinement variables")
    # A rigid-body or custom variable must not bypass our closed active set.
    allowed = {"background", "phase_fraction", "histogram_scale", "preferred_orientation", "lattice", "atomic_coordinates", "atomic_displacement", "occupancy"}
    permitted = lambda row: row["configured_now"] and any(parameter_matches_target(row, target, context) for target in stage["free"])
    permitted_names={name for p in space['parameters'] if permitted(p)
                     for name in (p['name'],p.get('refinement_variable',p['name']))}
    # Generated constraint coordinates need not have a row in parmDict. Check
    # the complete independent vary list, not just rows visible in that table.
    unexpected=sorted({str(name) for name in space['effective_vary'] if str(name) not in permitted_names} |
                      {p['name'] for p in space['parameters'] if p['effective_now'] and not permitted(p)})
    resolved = {}
    if unexpected:
        G2 = gsas()
        maps = importlib.import_module((G2.__package__ + "." if G2.__package__ else "") + "GSASIImapvars")
        resolved = resolve_constraint_provenance(space, unexpected, maps.GetDependentVars(), maps.getInvConstraintEq, allowed, permitted)
        unexpected = [name for name in unexpected if name not in resolved]
    if unexpected:
        raise ValueError("Unexpected active parameters outside the approved vocabulary: " + str(unexpected))
    for target in stage["free"]:
        if target["kind"] == "atom":
            continue  # Each requested atom flag is checked separately below.
        matching = [p for p in space["parameters"] if p["configured_now"] and parameter_matches_target(p, target, context)]
        if not matching:
            raise ValueError("A requested target has no configured native parameters: " + str(target))
        if not any(p["effective_now"] for p in matching):
            G2 = gsas()
            maps = importlib.import_module((G2.__package__ + "." if G2.__package__ else "") + "GSASIImapvars")
            dependent = {str(v) for v in maps.GetDependentVars()}
            influence = False
            for p in matching:
                variable = p.get("refinement_variable", p["name"])
                if variable not in dependent:
                    continue
                variables, multipliers = maps.getInvConstraintEq(variable, space["effective_vary"])
                if any(str(v) in space["effective_vary"] and abs(float(c)) > 1e-12 for v,c in zip(variables,multipliers)):
                    influence = True
            if not influence:
                raise ValueError("A requested target is fully constrained with no independent refinement influence: " + str(target))
    for target in stage["free"]:
        if target["kind"] != "atom":
            continue
        atom = next(a for p in context["phases"] if p["key"] == target["phase"] for a in p["atoms"] if a["label"] == target["atom"])
        for flag, group in (("X", "atomic_coordinates"), ("U", "atomic_displacement"), ("F", "occupancy")):
            if flag not in target["flags"]:
                continue
            names = {v["name"] for v in atom["variables"] if v["group"] == group}
            matching = [p for p in space["parameters"] if p["name"] in names and p["configured_now"]]
            if not matching:
                raise ValueError(f"{target['phase']}:{target['atom']} has no symmetry-independent {flag} variables")
            if not any(p["effective_now"] for p in matching):
                G2 = gsas()
                maps = importlib.import_module((G2.__package__ + "." if G2.__package__ else "") + "GSASIImapvars")
                dependent = {str(v) for v in maps.GetDependentVars()}
                influence = False
                for p in matching:
                    variable = p.get("refinement_variable", p["name"])
                    if variable not in dependent: continue
                    variables, multipliers = maps.getInvConstraintEq(variable, space["effective_vary"])
                    if any(str(v) in space["effective_vary"] and abs(float(c)) > 1e-12 for v, c in zip(variables, multipliers)):
                        influence = True
                if not influence:
                    raise ValueError(f"{target['phase']}:{target['atom']} {flag} is fully constrained with no independent refinement influence")
    return resolved


def resolve_constraint_provenance(space, unknown, dependent, inverse, allowed, permitted=None):
    """Admit generated independent variables only through approved native variables."""
    records = {}
    for row in space["parameters"]:
        records[row["name"]] = row
        records[row.get("refinement_variable", row["name"])] = row
    result = {}
    for name in unknown:
        origins = []
        for variable in dependent:
            variable = variable.varname() if hasattr(variable, "varname") else str(variable)
            variables, multipliers = inverse(variable, space["effective_vary"])
            if any(str(v) == name and abs(float(c)) > 1e-12 for v, c in zip(variables, multipliers)):
                row = records.get(variable)
                if row is None or not row["configured_now"] or row["group"] not in allowed or (permitted is not None and not permitted(row)):
                    origins = []; break
                origins.append({"variable": variable, "group": row["group"]})
        if origins: result[name] = origins
    return result


def run_plan(source, output, plan, *, approved=None, preview=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise ValueError("A distinct, new output GPX path is required")
    context = snapshot(source)
    validate_plan(plan, context)
    if plan["questions"] or plan["unsupported"]:
        raise ValueError("Resolve all questions and unsupported requests before approval/execution")
    if not preview and (not approved or approved.get("approved_plan_sha256") != plan_hash(plan)
                        or approved.get("model_signature") != context["model_signature"]):
        raise ValueError("This exact plan and model require an approval receipt")
    output.parent.mkdir(parents=True, exist_ok=True)
    G2 = gsas()
    initial_hash = context["source"]["sha256"]
    project = G2.G2Project(gpxfile=str(source))
    if hashlib.sha256(source.read_bytes()).hexdigest() != initial_hash:
        raise ValueError("Starting GPX changed between inspection and execution")
    project.save(str(output))
    result = {"schema": "radar-gpx-execution/v1", "preview": preview, "plan_sha256": plan_hash(plan),
              "source_sha256": initial_hash, "model_signature": context["model_signature"], "status": "running", "stages": []}
    result_path = output.with_suffix(".execution.json")
    added_constraints = set()
    declared_relations = {}
    normalized_histograms = set(context.get("normalized_histograms",[]))
    try:
        for stage, display in zip(plan["stages"], stage_table(plan, context)):
            started = time.monotonic()
            introduced=[c for c in stage['constraints'] if advanced.relation_key(c) not in declared_relations]
            configure(project, stage, context, added_constraints)
            normalized_histograms.update(stage.get("phase_normalization",[]))
            advanced.check_requested_relations(project,context,introduced,free=stage['free'])
            for constraint in introduced:declared_relations[advanced.relation_key(constraint)]=constraint
            configured = advanced.audit(project, context)
            project.save()
            state = extract_gsasii_refinement_state(output, project, G2, max_correlations=10)
            constraint_resolution = check_preflight(state, stage, context)
            row = {**display, "status": "preflight_passed", "effective_vary": state["parameter_space"]["effective_vary"],
                   "constraint_warning": state["constraints"].get("warning", ""), "constraint_parameter_provenance": constraint_resolution, "configured_state": configured}
            result["stages"].append(row)
            write_json(result_path, result)
            if not preview:
                # The input GPX can already contain a fresh covariance. Require
                # this particular refinement call to replace it explicitly.
                project.data.setdefault("Covariance", {}).setdefault("data", {})["freshCOV"] = False
                project.save()
                project.do_refinements([{}])
                row["status"] = "refined_needs_review"
                cov_native=project.data.get("Covariance",{}).get("data",{})
                if not cov_native.get("freshCOV"):raise ValueError("Stage did not produce fresh refinement covariance")
                native_maps=importlib.import_module((G2.__package__+"." if G2.__package__ else "")+"GSASIImapvars")
                # Capture while the actual last solver's map is still active;
                # the read-only extractor reconstructs its own maps afterward.
                native_bounds=constraint_bound_audit(state,cov_native,native_maps.Dict2Map)
                project.save()
                after = extract_gsasii_refinement_state(output, project, G2, max_correlations=10)
                cov = project.data.get("Covariance", {}).get("data", {})
                if not cov.get("freshCOV"):
                    raise ValueError("Stage did not produce fresh refinement covariance")
                row.update(status="refined_needs_review", rwp={h.name: h.get_wR() for h in project.histograms()},
                           actual_vary=after["parameter_space"]["last_vary"], covariance={k: v for k, v in after["covariance"].items() if k not in {"matrix", "correlation_matrix"}},
                           metric_rows=[p for p in after["parameter_space"]["parameters"] if p["group"] in {"lattice", "atomic_coordinates", "atomic_displacement", "occupancy"}],
                           frozen_variables=after["parameter_space"]["frozen_variables"])
                row["constraint_bound_audit"] = native_bounds
                row["blocking_frozen_variables"] = blocking_frozen_variables(state,after)
                row["final_state"] = advanced.audit(project, context)
                row["parameter_changes"] = parameter_changes(state, after)
                require_fixed_values_unchanged(row["parameter_changes"])
                require_native_covariance(cov,len(row['actual_vary']),row['covariance'])
                require_constraint_bounds(native_bounds)
                advanced.check_site_totals(project, context)
                row["normalization_checks"] = advanced.check_normalization(project,context,normalized_histograms)
                advanced.check_requested_relations(project,context,list(declared_relations.values()))
                if set(row["actual_vary"]) != set(row["effective_vary"]):
                    row["status"] = "needs_review"
                    raise ValueError("Actual independent vary list differs from the approved stage preflight")
                if row["blocking_frozen_variables"]:
                    row["status"] = "needs_review"
                    raise ValueError("GSAS-II froze a variable at its bound; remaining stages were not executed")
                row["status"] = "completed"
                checkpoint = output.with_name(output.stem + "_" + stage["id"] + ".gpx")
                project.save(str(checkpoint))
                project.save(str(output))
            row["elapsed_seconds"] = time.monotonic() - started
        result["status"] = "preview_passed" if preview else "completed_needs_scientific_review"
        result["final_phases"] = [{"name": p.name, "cell": p.get_cell(),
            "atoms": [{"label": a.label, "coordinates": list(a.coordinates), "occupancy": a.occupancy, "adp": a.ADP} for a in p.atoms()]}
            for p in project.phases()]
        result["output_gpx_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    except Exception as exc:
        if result["stages"]:
            result["stages"][-1].setdefault("elapsed_seconds",time.monotonic()-started)
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        write_json(result_path, result)
        raise
    finally:
        if hashlib.sha256(source.read_bytes()).hexdigest() != initial_hash:
            raise RuntimeError("The starting GPX was modified")
    write_json(result_path, result)
    return result
