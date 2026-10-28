"""Typed range, bounds, scale normalization and texture operations."""
import importlib
import math
import copy

STAGE_FIELDS = {"fit_ranges", "exclusions", "bounds", "phase_normalization", "texture"}
UNITS = {"native", "tof", "d", "Q", "two_theta"}


def occupancy_groups(atoms):
    """Conservative co-located sites only; do not guess symmetry-orbit equivalence."""
    groups = {}
    for atom in atoms:
        if "coordinates" not in atom or "multiplicity" not in atom: continue
        point = tuple(round(float(x) % 1, 7) % 1 for x in atom["coordinates"])
        groups.setdefault((point, atom["multiplicity"]), []).append(atom["label"])
    return [labels for labels in groups.values() if len(labels) > 1]


def site_capacity(context, phase_key, atom_label, phase, free):
    record = next(p for p in context["phases"] if p["key"] == phase_key)
    active = {t["atom"] for t in free if t["kind"] == "atom" and t["phase"] == phase_key and "F" in t["flags"]}
    for group in record.get("occupancy_groups", occupancy_groups(record["atoms"])):
        if atom_label in group:
            held = sum(phase.atom(name).occupancy for name in group if name not in active)
            return max(0.0, 1.0 - held)
    return 1.0


def check_site_totals(project, context):
    for p in context["phases"]:
        phase = project.phase(p["name"])
        for group in p.get("occupancy_groups", occupancy_groups(p["atoms"])):
            total = sum(phase.atom(name).occupancy for name in group)
            if total > 1 + 1e-6 or total < -1e-6:
                raise ValueError("Mixed-site occupancy exceeds its physical capacity: " + p["key"] + " " + str(group) + " total=" + str(total))
        for name, hap in phase.data["Histograms"].items():
            if not hap.get("Use", True): continue
            if hap["Scale"][0] < -1e-9:
                raise ValueError("Refinement produced a negative phase scale; scientific review is required")
            if hap["Pref.Ori."][0] == "MD" and hap["Pref.Ori."][1] <= 0:
                raise ValueError("Refinement produced a nonpositive March-Dollase ratio")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_stage(stage, context, normalized):
    from .contracts import exact_keys
    phases = {p["key"]: p for p in context["phases"]}
    hists = {h["key"]: h for h in context["histograms"]}
    free = stage["free"]
    for field in STAGE_FIELDS:
        if not isinstance(stage.get(field, []), list) or len(stage.get(field, [])) > 100:
            raise ValueError("Expected a bounded list for " + field)
    for edit in stage.get("fit_ranges", []):
        exact_keys(edit, {"histogram", "unit", "lower", "upper"}, {"histogram", "unit", "lower", "upper"})
        if edit["histogram"] not in hists or edit["unit"] not in UNITS or not finite(edit["lower"]) or not finite(edit["upper"]) or not 0 < edit["lower"] < edit["upper"]:
            raise ValueError("Fit ranges need a known histogram, explicit unit and increasing positive limits")
    for edit in stage.get("exclusions", []):
        exact_keys(edit, {"histogram", "unit", "mode", "regions"}, {"histogram", "unit", "mode", "regions"})
        if edit["histogram"] not in hists or edit["unit"] not in UNITS or edit["mode"] not in {"add", "replace", "clear"}:
            raise ValueError("Invalid excluded-region edit")
        if not isinstance(edit["regions"], list) or len(edit["regions"]) > 50 or (edit["mode"] == "clear" and edit["regions"]):
            raise ValueError("Use an explicit bounded region list, empty for clear")
        for pair in edit["regions"]:
            if not isinstance(pair, list) or len(pair) != 2 or not all(finite(x) for x in pair) or not 0 < pair[0] < pair[1]:
                raise ValueError("Excluded intervals must be increasing positive pairs")
    seen = set()
    for edit in stage.get("bounds", []):
        exact_keys(edit, {"phase", "atom", "parameter", "lower", "upper", "value"}, {"phase", "atom", "parameter", "lower", "upper"})
        ident = (edit["phase"], edit["atom"], edit["parameter"])
        if ident in seen: raise ValueError("Duplicate parameter bound")
        seen.add(ident)
        p = phases.get(edit["phase"])
        if p is None or edit["parameter"] not in {"F", "U"} or edit["atom"] not in {a["label"] for a in p["atoms"]}:
            raise ValueError("Bounds support exact occupancy F or Uiso U targets")
        if not any(t.get("kind") == "atom" and t.get("phase") == edit["phase"] and t.get("atom") == edit["atom"] and edit["parameter"] in t["flags"] for t in free):
            raise ValueError("A bounded parameter must be explicitly free in that stage")
        if not all(finite(edit[k]) for k in ("lower", "upper")) or not 0 <= edit["lower"] < edit["upper"] or (edit["parameter"] == "F" and edit["upper"] > 1):
            raise ValueError("Bounds must be finite and physical (occupancy 0–1, Uiso nonnegative)")
        if "value" in edit and (not finite(edit["value"]) or not edit["lower"] <= edit["value"] <= edit["upper"]):
            raise ValueError("The requested starting value must lie inside its bounds")
        if edit["parameter"] == "F":
            active = {t["atom"] for t in free if t["kind"] == "atom" and t["phase"] == edit["phase"] and "F" in t["flags"]}
            for group in p.get("occupancy_groups", occupancy_groups(p["atoms"])):
                if edit["atom"] in group:
                    capacity = 1 - sum(a["occupancy"] for a in p["atoms"] if a["label"] in group and a["label"] not in active)
                    if edit["upper"] > capacity + 1e-7:
                        raise ValueError("Requested occupancy bound permits overfilling a mixed site with other occupants held fixed")
    values = stage.get("phase_normalization", [])
    if any(not isinstance(h, str) or h not in hists for h in values) or len(set(values)) != len(values):
        raise ValueError("Phase normalization lists each selected histogram key once")
    normalized.update(values)
    for hist in hists:
        linked = {p["key"] for p in phases.values() if hists[hist]["name"] in p["histograms"]}
        scales = {t["phase"] for t in free if t["kind"] == "phase_scale" and t["histogram"] == hist}
        global_free = any(t["kind"] == "histogram_scale" and t["histogram"] == hist for t in free)
        if global_free and linked and scales == linked and hist not in normalized:
            raise ValueError("Histogram scale plus all phase scales requires explicit sum-to-one phase normalization")
        if hist in values and (not global_free or (len(linked)==1 and scales)):
            raise ValueError("Normalization requires the global histogram scale free; a single phase HAP scale is fixed to one")
        if hist in normalized and len(linked)>1 and len(scales)==1:
            raise ValueError("Only one requested free phase scale is fully determined by sum-to-one with the other phase scales held. Hold it too and refine the histogram scale, or explicitly release at least two phase scales; never silently free extra phases.")
    for edit in stage.get("texture", []):
        exact_keys(edit, {"phase", "histogram", "model", "direction", "order", "value"}, {"phase", "histogram", "model"})
        if edit["phase"] not in phases or edit["histogram"] not in hists or hists[edit["histogram"]]["name"] not in phases[edit["phase"]]["histograms"]:
            raise ValueError("Texture requires a linked phase/histogram")
        if edit["model"] == "march-dollase":
            direction = edit.get("direction")
            if not isinstance(direction, list) or len(direction) != 3 or any(type(x) is not int for x in direction) or direction == [0, 0, 0] or "order" in edit:
                raise ValueError("March-Dollase requires a nonzero integer hkl direction")
            if "value" in edit and (not finite(edit["value"]) or edit["value"] <= 0):
                raise ValueError("March-Dollase ratio must be positive")
        elif edit["model"] == "spherical-harmonics":
            if type(edit.get("order")) is not int or edit["order"] not in {2, 4, 6, 8} or "direction" in edit or "value" in edit:
                raise ValueError("Spherical harmonics requires explicit even order 2–8")
            if "texture_orders" in phases[edit["phase"]] and edit["order"] not in phases[edit["phase"]]["texture_orders"]:
                raise ValueError("This crystal symmetry has no texture coefficients at the requested order")
        else:
            raise ValueError("Unsupported texture model")


def lattice(project):
    module = project.__class__.__module__
    return importlib.import_module((module.rsplit(".", 1)[0] + "." if "." in module else "") + "GSASIIlattice")


def native_interval(hist, lower, upper, unit, project):
    inst = hist.InstrumentParameters
    tof = "T" in inst["Type"][0]
    if unit == "tof" and not tof or unit == "two_theta" and tof:
        raise ValueError("Requested native axis is incompatible with this histogram")
    points = [lower, upper]
    if unit in {"d", "Q"}:
        dsp = points if unit == "d" else [2 * math.pi / x for x in points]
        if "E" in inst["Type"][0]: raise ValueError("Energy-dispersive conversion is outside this recipe implementation")
        points = [float(lattice(project).Dsp2pos(inst, x)) for x in dsp]
    lo, hi = sorted(points)
    original = hist.data["Limits"][0]
    if not all(math.isfinite(x) for x in points) or not original[0] <= lo < hi <= original[1]:
        raise ValueError("Requested interval lies outside the measured native range")
    return [lo, hi]


def _scale_sum_exists(project, names):
    for group, constraints in project.data.get("Constraints", {}).get("data", {}).items():
        if not isinstance(constraints, list): continue
        for item in constraints:
            if not isinstance(item, list) or len(item) < 4 or item[-1] != "c": continue
            terms = item[:-3]
            try:
                variables = [str(v.varname()) if hasattr(v, "varname") else str(v) for _, v in terms]
                if set(variables) == set(names):
                    if len(variables) != len(names) or not has_equation(project, names, [1.0]*len(names), 1.0):
                        raise ValueError("An existing phase-scale equation conflicts with normalization")
                    return True
            except (TypeError, AttributeError): continue
    return False


def has_equation(project, names, coefficients, total):
    expected = dict(zip(names, coefficients, strict=True))
    for constraints in project.data.get("Constraints", {}).get("data", {}).values():
        if not isinstance(constraints, list): continue
        for item in constraints:
            if not isinstance(item, list) or len(item) < 4 or item[-1] != "c": continue
            try:
                actual = {v.varname() if hasattr(v, "varname") else str(v): float(c) for c, v in item[:-3]}
                if actual.keys() != expected.keys(): continue
                first = sorted(expected)[0]; factor = actual[first] / expected[first]
                if factor and all(math.isclose(actual[n], factor * expected[n], rel_tol=1e-10, abs_tol=1e-12) for n in expected) and math.isclose(float(item[-3]), factor * total, rel_tol=1e-10, abs_tol=1e-12):
                    return True
            except (TypeError, ValueError, ZeroDivisionError): continue
    return False


def has_equality(project,names):
    """Recognize equal-coefficient equivalence chains, preserving existing constraints."""
    neighbours={}
    for rows in project.data.get('Constraints',{}).get('data',{}).values():
        if not isinstance(rows,list):continue
        for row in rows:
            if not isinstance(row,list) or len(row)<5 or row[-1]!='e':continue
            try:
                terms=[(float(c),v.varname() if hasattr(v,'varname') else str(v)) for c,v in row[:-3]]
                if not terms or not terms[0][0] or any(not math.isclose(c,terms[0][0],rel_tol=1e-12,abs_tol=1e-12) for c,v in terms):continue
                first=terms[0][1]
                for _,v in terms[1:]:neighbours.setdefault(first,set()).add(v);neighbours.setdefault(v,set()).add(first)
            except (ValueError,TypeError):continue
    pending=[names[0]];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);pending.extend(neighbours.get(name,set())-seen)
    return set(names)<=seen


def configure(project, stage, context, phases, hists, added):
    for edit in stage.get("fit_ranges", []):
        hist = hists[edit["histogram"]]
        hist.set_refinements({"Limits": native_interval(hist, edit["lower"], edit["upper"], edit["unit"], project)})
    for edit in stage.get("exclusions", []):
        hist = hists[edit["histogram"]]
        regions = [native_interval(hist, *pair, edit["unit"], project) for pair in edit["regions"]]
        hist.Excluded((hist.Excluded() if edit["mode"] == "add" else []) + regions)
    for key in stage.get("phase_normalization", []):
        hist = hists[key]
        active = [p for p in phases.values() if hist.name in p.data["Histograms"] and p.data["Histograms"][hist.name].get("Use", True)]
        names = [f"{p.id}:{hist.id}:Scale" for p in active]
        total = sum(float(p.data["Histograms"][hist.name]["Scale"][0]) for p in active)
        if not math.isfinite(total) or total <= 0 or any(p.data["Histograms"][hist.name]["Scale"][0] < 0 for p in active):
            raise ValueError("Normalize only finite nonnegative phase scales with positive total")
        existing = _scale_sum_exists(project, names) if len(names) > 1 else True
        for phase in active:
            variable=f"{phase.id}:{hist.id}:Scale"
            lo=project.get_Controls("parmMin",variable=variable)
            hi=project.get_Controls("parmMax",variable=variable)
            lo=max(0.0,float(lo)) if lo is not None else 0.0
            hi=min(1.0,float(hi)) if hi is not None else 1.0
            value=float(phase.data["Histograms"][hist.name]["Scale"][0])/total
            if not lo<=value<=hi:
                raise ValueError("Normalized starting phase scale is outside its preserved source bounds; review the bounds or starting project rather than silently relaxing or clamping them")
            phase.data["Histograms"][hist.name]["Scale"][0]=value
            project.set_Controls("parmMin",lo,variable=variable)
            project.set_Controls("parmMax",hi,variable=variable)
        hist.SampleParameters["Scale"][0] *= total
        variable=f":{hist.id}:Scale";value=hist.SampleParameters["Scale"][0]
        lo=project.get_Controls("parmMin",variable=variable);hi=project.get_Controls("parmMax",variable=variable)
        if lo is not None and value<lo or hi is not None and value>hi:
            raise ValueError("Compensated histogram starting scale is outside its preserved source bounds; review the normalization recipe")
        if len(names) == 1:
            active[0].data["Histograms"][hist.name]["Scale"][1] = False
        elif not existing:
            project.add_EqnConstr(1.0, names, multlist=[1.0] * len(names))
    for edit in stage.get("texture", []):
        phase, hist = phases[edit["phase"]], hists[edit["histogram"]]
        record = phase.data["Histograms"][hist.name]["Pref.Ori."]
        if edit["model"] == "march-dollase":
            value = edit.get("value", record[1] if record[0] == "MD" else 1.0)
            record[0], record[1], record[3], record[4], record[5] = "MD", value, list(edit["direction"]), 0, {}
        else:
            order = edit["order"]
            names = lattice(project).GenSHCoeffT(phase.data["General"]["SGData"]["SGLaue"], "0", order, False)
            if not names: raise ValueError("This symmetry/order has no spherical-harmonics coefficients")
            old = record[5] if record[0] == "SH" else {}
            record[0], record[4], record[5] = "SH", order, {name: old.get(name, 0.0) for name in names}
    for target in stage["free"]:
        if target["kind"] != "preferred_orientation": continue
        phase, hist = phases[target["phase"]], hists[target["histogram"]]
        if phase.data["Histograms"][hist.name]["Pref.Ori."][0] == "MD":
            variable = f"{phase.id}:{hist.id}:MD"
            lo = project.get_Controls("parmMin", variable=variable)
            project.set_Controls("parmMin", max(1e-6, lo) if lo is not None else 1e-6, variable=variable)
    for edit in stage.get("bounds", []):
        phase = phases[edit["phase"]]; atom = phase.atom(edit["atom"])
        if edit["parameter"] == "F" and edit["upper"] > site_capacity(context, edit["phase"], edit["atom"], phase, stage["free"]) + 1e-7:
            raise ValueError("Requested occupancy bounds overfill the current mixed site after preceding stages; review the recipe rather than silently changing the bounds")
        record = next(a for p in context["phases"] if p["key"] == edit["phase"] for a in p["atoms"] if a["label"] == edit["atom"])
        group = "occupancy" if edit["parameter"] == "F" else "atomic_displacement"
        names = [v["name"] for v in record["variables"] if v["group"] == group]
        if len(names) != 1: raise ValueError("A unique occupancy/Uiso variable is required for bounds")
        if "value" in edit:
            if edit["parameter"] == "F": atom.occupancy = edit["value"]
            else: atom.ADP = edit["value"]
        value = atom.occupancy if edit["parameter"] == "F" else atom.ADP
        if not edit["lower"] <= value <= edit["upper"]:
            raise ValueError("Starting parameter is outside its requested bounds; specify an explicit valid starting value")
        project.set_Controls("parmMin", edit["lower"], variable=names[0])
        project.set_Controls("parmMax", edit["upper"], variable=names[0])


def audit(project, context):
    bounds = []
    for p in context["phases"]:
        phase = project.phase(p["name"])
        for atom in p["atoms"]:
            flags = phase.atom(atom["label"]).refinement_flags
            for flag, group in (("F", "occupancy"), ("U", "atomic_displacement")):
                if flag not in flags: continue
                for variable in atom["variables"]:
                    if variable["group"] != group: continue
                    bounds.append({"phase": p["key"], "atom": atom["label"], "parameter": flag,
                                   "variable": variable["name"], "lower": project.get_Controls("parmMin", variable=variable["name"]),
                                   "upper": project.get_Controls("parmMax", variable=variable["name"])})
    return copy.deepcopy({"effective_bounds": bounds, "histograms": [{"key": h["key"], "limits": project.histogram(h["name"]).data["Limits"],
                            "histogram_scale": project.histogram(h["name"]).SampleParameters["Scale"]} for h in context["histograms"]],
            "phases": [{"key": p["key"], "cell_refined": phase.data["General"]["Cell"][0],
                        "atoms": [{"label": a.label, "flags": a.refinement_flags.strip(), "occupancy": a.occupancy,
                                   "coordinates": list(a.coordinates), "adp": a.ADP} for a in phase.atoms()],
                        "hap": {h: {"scale": d["Scale"], "texture": d["Pref.Ori."]} for h, d in phase.data["Histograms"].items()}}
                       for p in context["phases"] for phase in [project.phase(p["name"])]]})


def relation_key(constraint):
    """Scientific identity excludes prose and term ordering."""
    import json
    if constraint['kind']=='uiso_equal':
        return json.dumps({'kind':'uiso_equal','atoms':sorted(constraint['atoms'],key=lambda t:(t['phase'],t['atom']))},sort_keys=True)
    terms=sorted(constraint['terms'],key=lambda t:(t['phase'],t['atom']));factor=float(terms[0]['coefficient'])
    return json.dumps({'kind':'occupancy_sum','terms':[{**t,'coefficient':float(t['coefficient'])/factor} for t in terms],'total':float(constraint['total'])/factor},sort_keys=True)


def check_requested_relations(project,context,constraints,*,free=None):
    """Preserving a definition is not proof that held or refined values obey it.

    With free supplied, check only newly introduced wholly held relations. A
    preview cannot predict values after an earlier, unexecuted refinement.
    Without free, verify every declared relation in the actual saved state.
    """
    phases={p['key']:project.phase(p['name']) for p in context['phases']}
    for constraint in constraints:
        field='U' if constraint['kind']=='uiso_equal' else 'F'
        targets=constraint['atoms'] if field=='U' else constraint['terms']
        if free is not None and any(t['kind']=='atom' and field in t['flags'] and any((t['phase'],t['atom'])==(a['phase'],a['atom']) for a in targets) for t in free):continue
        if field=='U':
            values=[float(phases[t['phase']].atom(t['atom']).ADP) for t in targets]
            valid=max(values)-min(values)<=1e-9
        else:
            value=sum(float(t['coefficient'])*float(phases[t['phase']].atom(t['atom']).occupancy) for t in targets)
            valid=math.isclose(value,float(constraint['total']),rel_tol=1e-7,abs_tol=1e-7)
        if not valid:
            raise ValueError('Requested '+constraint['kind']+' relation is not satisfied by '+('held values; authorize appropriate refinement freedoms or explicit starting values' if free is not None else 'the actual refined values; later stages were not executed'))


def check_normalization(project,context,normalized):
    rows=[]
    for key in sorted(normalized):
        name=next(h['name'] for h in context['histograms'] if h['key']==key)
        values=[float(p.data['Histograms'][name]['Scale'][0]) for p in project.phases() if name in p.data['Histograms'] and p.data['Histograms'][name].get('Use',True)]
        total=sum(values);rows.append({'histogram':key,'phase_scale_sum':total,'phase_count':len(values)})
        if not values or not math.isfinite(total) or not math.isclose(total,1.,rel_tol=0,abs_tol=1e-7):
            raise ValueError('Phase-scale normalization drifted in '+key+': sum='+str(total)+'; saved values do not satisfy the requested sum-to-one equation')
    return rows
