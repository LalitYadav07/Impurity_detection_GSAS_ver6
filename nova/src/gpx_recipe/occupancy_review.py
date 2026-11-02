"""Portable, source-bound permission to retain reviewed CIF rounding unchanged.

This is not a numerical tolerance for refinement. A verified importer records
the exact native sites; only those sites, held at their imported occupancies
and positions, can retain the source's explicitly reviewed small excess.
"""
from copy import deepcopy
import hashlib
import importlib
import json
import math
from numbers import Real
import re

DATA_KEY = "RADAR-PD occupancy review"
SCHEMA = "radar-gpx-occupancy-review/v2"
POLICY_SCHEMA = "radar-cif-occupancy-review/v1"
MAX_EXCESS = 0.001
DENOMINATOR = 24
REBUILD = "Rebuild this GPX from the exact reviewed CIF library before polishing; an old summary or generic warning is not an occupancy authorization."


def operator_sets(operations):
    """Canonical exact crystallographic operations shared with CIF import."""
    import numpy as np
    groups = {}
    for rotation, translation in operations:
        rotation, translation = np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
        if rotation.shape != (3, 3) or translation.shape != (3,):
            raise ValueError("A crystallographic operation needs a 3x3 rotation and three translations")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            raise ValueError("Non-finite crystallographic operation")
        rounded_rotation = np.rint(rotation).astype(int)
        rounded_translation = np.rint(translation * DENOMINATOR).astype(int)
        if not np.allclose(rotation, rounded_rotation, atol=1e-8, rtol=0) or not np.allclose(
                translation * DENOMINATOR, rounded_translation, atol=1e-6, rtol=0):
            raise ValueError("Unsupported crystallographic operation precision; manual setting review required")
        groups.setdefault(tuple(rounded_rotation.flat), set()).add(tuple(rounded_translation % DENOMINATOR))
    if not groups:
        raise ValueError("No explicit crystallographic symmetry operations")
    return groups


def matching_origin_shift(source_operations, target_operations):
    """The importer's deterministic, complete origin-equivalence proof."""
    import numpy as np
    source, target = operator_sets(source_operations), operator_sets(target_operations)
    if set(source) != set(target) or any(len(source[k]) != len(target[k]) for k in source):
        raise ValueError("CIF and GSAS-II require a basis/setting transformation, not a verified origin translation")
    if source == target:
        return np.zeros(3)
    candidates = np.indices((DENOMINATOR,) * 3).reshape(3, -1).T
    keys = sorted(source, key=lambda k: -np.linalg.matrix_rank(np.eye(3) - np.array(k).reshape(3, 3)))
    for key in keys:
        rotation = np.array(key).reshape(3, 3)
        offsets = candidates - candidates @ rotation.T
        keep = np.ones(len(candidates), dtype=bool)
        targets = np.array(sorted(target[key]))
        for translation in source[key]:
            transformed = (offsets + np.array(translation)) % DENOMINATOR
            keep &= np.any(np.all(transformed[:, None, :] == targets[None, :, :], axis=2), axis=1)
        candidates = candidates[keep]
        if not len(candidates):
            raise ValueError("No exact supported origin translation maps the supplied CIF operators to GSAS-II")
    signed = (candidates + DENOMINATOR // 2) % DENOMINATOR - DENOMINATOR // 2
    order = sorted(range(len(candidates)), key=lambda i: (int(np.dot(signed[i], signed[i])), tuple(signed[i])))
    shift = signed[order[0]] / DENOMINATOR
    translated = [(r, np.asarray(t) + shift - np.asarray(r) @ shift) for r, t in source_operations]
    if operator_sets(translated) != target:
        raise ValueError("Origin transformation failed its complete symmetry-operator check")
    return shift


def _native_operations(sg):
    import numpy as np
    return [(sign * np.asarray(rotation), sign * np.asarray(translation) + np.asarray(center))
            for center in sg["SGCen"] for sign in ([1, -1] if sg["SGInv"] else [1])
            for rotation, translation in sg["SGOps"]]


def _source_space_group(model):
    symbols = [model[key] for key in ("_space_group_name_h-m_alt", "_symmetry_space_group_name_h-m") if key in model]
    if symbols and (not isinstance(symbols[0], str) or len(symbols[0]) > 200):
        raise ValueError("Invalid source space-group label")
    try:
        # GSASIIspc imports GSASIIobj, which initializes P1SGData from spc.
        # Enter through the supported scripting API to avoid partially
        # initializing that circular dependency when this helper runs first.
        importlib.import_module("GSASII.GSASIIscriptable")
        spc = importlib.import_module("GSASII.GSASIIspc")
    except ModuleNotFoundError as exc:
        if exc.name not in {"GSASII", "GSASII.GSASIIscriptable", "GSASII.GSASIIspc"}:
            raise
        try:
            importlib.import_module("GSASIIscriptable")
            spc = importlib.import_module("GSASIIspc")
        except ModuleNotFoundError as fallback:
            if fallback.name not in {"GSASIIscriptable", "GSASIIspc"}:
                raise
            # Basic contract tests need no native installation. P1 has only
            # the identity operation; production uses GSAS's complete record.
            if symbols and re.sub(r"\s", "", symbols[0]).upper() == "P1":
                return {"SpGrp": "P 1", "SGInv": False, "SGCen": [[0, 0, 0]],
                        "SGOps": [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])]}
            raise
    error, sg = spc.SpcGroup(symbols[0]) if symbols else (1, None)
    if error:
        value = next((model[k] for k in ("_space_group_it_number", "_symmetry_int_tables_number") if k in model), None)
        try:
            number = int(value)
        except (ValueError, TypeError):
            raise ValueError("Source CIF space group cannot be resolved") from None
        if not 1 <= number <= 230:
            raise ValueError("Source CIF space group cannot be resolved")
        error, sg = spc.SpcGroup(spc.spgbyNum[number])
    if error:
        raise ValueError("Source CIF space group is unsupported")
    return sg


def _source_operations(model, canonical):
    expressions = next((model[key] for key in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz") if key in model), None)
    if expressions is None:
        return _native_operations(canonical)
    if isinstance(expressions, str):
        expressions = [expressions]
    if (not isinstance(expressions, list) or not 1 <= len(expressions) <= 384
            or any(not isinstance(value, str) or not 0 < len(value) <= 500 for value in expressions)):
        raise ValueError("Source symmetry operators must be a bounded explicit list")
    if expressions == ["x,y,z"]:
        return [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0, 0, 0])]
    from pymatgen.core.operations import SymmOp
    return [(op.rotation_matrix, op.translation_vector) for op in
            (SymmOp.from_xyz_str(value) for value in expressions)]


def _source_origin(review, model, native_symmetry):
    """Re-derive native setting from evidence covered by the original policy."""
    import numpy as np
    if (not isinstance(model, dict) or not model or len(model) > 2000
            or any(not isinstance(k, str) or not k.startswith("_")
                   or not (isinstance(v, str) or isinstance(v, list) and all(isinstance(x, str) for x in v))
                   for k, v in model.items())
            or len(json.dumps(model).encode()) > 2_000_000
            or _digest(model) != review["nuclear_model_signature"]):
        raise ValueError("Source non-cell CIF proof does not match the reviewed model signature")
    canonical = _source_space_group(model)
    target = _native_operations(canonical)
    actual = _native_operations(native_symmetry)
    if (not isinstance(native_symmetry["SGInv"], bool)
            or native_symmetry["SGInv"] != bool(canonical["SGInv"])
            or re.sub(r"\s", "", native_symmetry["SpGrp"]) != re.sub(r"\s", "", canonical["SpGrp"])
            or len(actual) != len(target)
            or operator_sets(actual) != operator_sets(target)):
        raise ValueError("Native symmetry does not match the reviewed source setting")
    for key in ("SGLaue", "SGUniq", "SGLatt", "SGSys", "SGFixed"):
        if key in canonical and native_symmetry.get(key) != _plain(canonical[key]):
            raise ValueError("Native symmetry metadata does not match the reviewed source setting")
    source = _source_operations(model, canonical)
    shift = matching_origin_shift(source, target)
    forward = [(r, np.asarray(t) + shift - np.asarray(r) @ shift) for r, t in source]
    reverse = [(r, np.asarray(t) - shift + np.asarray(r) @ shift) for r, t in target]
    if operator_sets(forward) != operator_sets(target) or operator_sets(reverse) != operator_sets(source):
        raise ValueError("Reviewed CIF origin conversion lacks complete operator equivalence")
    return shift.tolist()


def _plain(value):
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("Unsupported native occupancy-review value")


def _digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("Occupancy review requires finite native numerical values")
    return float(value)


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _element(value):
    if not isinstance(value, str):
        raise ValueError("Reviewed atom types must be strings")
    match = re.fullmatch(r"([A-Z][a-z]?)(?:(?:[+-]\d*)|(?:\d+[+-]))?", value)
    if match is None:
        raise ValueError("Unrecognized reviewed native atom type")
    return match[1]


def _checked_inventory(rows):
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError("Reviewed native atom inventory is empty or too large")
    result = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"label", "type", "coordinates", "occupancy", "multiplicity"}
                or not isinstance(row["label"], str) or not row["label"]
                or not isinstance(row["type"], str) or not row["type"]
                or not isinstance(row["coordinates"], (list, tuple)) or len(row["coordinates"]) != 3):
            raise ValueError("Malformed reviewed native atom inventory")
        multiplicity = _number(row["multiplicity"])
        occupancy = _number(row["occupancy"])
        if multiplicity <= 0 or int(multiplicity) != multiplicity or not 0 <= occupancy <= 1 + MAX_EXCESS:
            raise ValueError("Invalid reviewed native occupancy or multiplicity")
        result.append({"label": row["label"], "type": row["type"],
            "coordinates": [_number(v) % 1 for v in row["coordinates"]],
            "occupancy": occupancy, "multiplicity": int(multiplicity)})
    if len({row["label"] for row in result}) != len(result):
        raise ValueError("Reviewed native atom labels must be unique")
    return sorted(result, key=lambda row: row["label"])


def site_groups(atoms):
    """Current co-located sites, including unshared singleton sites."""
    groups = {}
    for atom in atoms:
        if "coordinates" not in atom or "multiplicity" not in atom:
            continue
        point = tuple(round(float(x) % 1, 7) % 1 for x in atom["coordinates"])
        groups.setdefault((point, atom["multiplicity"]), []).append(atom["label"])
    return list(groups.values())


def occupancy_groups(atoms):
    """Keep the existing conservative co-location rule, without orbit guessing."""
    return [labels for labels in site_groups(atoms) if len(labels) > 1]


def current_site_totals(phase):
    atoms = _inventory(phase)
    by_label = {a["label"]: a for a in atoms}
    return [(labels, math.fsum(by_label[label]["occupancy"] for label in labels))
            for labels in site_groups(atoms)]


def _inventory(phase):
    general = phase.data["General"]
    cx, ct, cs, _ = general["AtomPtrs"]
    rows = phase.data["Atoms"]
    if not rows or len(rows) > 10000:
        raise ValueError("Reviewed native atom inventory is empty or too large")
    result = []
    for row in rows:
        label, atom_type = row[ct - 1], row[ct]
        if not isinstance(label, str) or not label or not isinstance(atom_type, str) or not atom_type:
            raise ValueError("Reviewed native atoms require explicit labels and types")
        multiplicity = _number(row[cs + 1])
        if multiplicity <= 0 or int(multiplicity) != multiplicity:
            raise ValueError("Invalid reviewed native site multiplicity")
        result.append({"label": label, "type": atom_type,
            "coordinates": [_number(v) % 1 for v in row[cx:cx + 3]],
            "occupancy": _number(row[cx + 3]), "multiplicity": int(multiplicity)})
    return _checked_inventory(result)


def _symmetry(phase):
    general = phase.data["General"]
    if general.get("Type", "nuclear") != "nuclear" or general.get("Modulated"):
        raise ValueError("Reviewed occupancy retention requires an ordinary nuclear phase")
    sg = general["SGData"]
    required = {"SpGrp", "SGOps", "SGCen", "SGInv"}
    if not required <= set(sg):
        raise ValueError("Reviewed occupancy retention needs complete native symmetry operators")
    fields = required | {"SGLaue", "SGUniq", "SGLatt", "SGSys", "SGFixed", "SGGray"}
    value = {key: _plain(sg[key]) for key in sorted(fields) if key in sg}
    _digest(value)  # Reject nonfinite or nonportable operator values.
    return value


def _policy(review, source_sha256):
    if (not isinstance(review, dict) or review.get("schema") != POLICY_SCHEMA
            or review.get("reviewed") is not True or review.get("normalization_applied") is not False):
        raise ValueError("Missing explicit no-normalization CIF occupancy review")
    limit = _number(review.get("maximum_site_excess"))
    if not 0 < limit <= MAX_EXCESS or not isinstance(review.get("review_note"), str) or not review["review_note"].strip():
        raise ValueError("Invalid bounded CIF occupancy review")
    for key in ("source_sha256", "canonical_sha256", "policy_sha256", "atom_signature", "model_signature", "nuclear_model_signature"):
        if not _sha(review.get(key)):
            raise ValueError("CIF occupancy review lacks a valid source/model hash")
    for key in ("root_policy_sha256", "parent_policy_sha256", "derived_sha256"):
        if key in review and not _sha(review[key]):
            raise ValueError("CIF occupancy review has an invalid lineage hash")
    if review["policy_sha256"] != _digest({k: v for k, v in review.items() if k != "policy_sha256"}):
        raise ValueError("CIF occupancy review policy hash mismatch")
    if not _sha(source_sha256) or source_sha256 not in {
            review["source_sha256"], review["canonical_sha256"], review.get("derived_sha256")}:
        raise ValueError("Native occupancy review is bound to a different source CIF")
    affected = review.get("affected_sites")
    if not isinstance(affected, list) or not affected or len(affected) > 100000:
        raise ValueError("CIF occupancy review has no bounded affected-site evidence")
    return limit


def _reviewed_groups(review, atoms):
    limit = _number(review["maximum_site_excess"])
    by_label = {row["label"]: row for row in atoms}
    groups = {tuple(sorted(group)) for group in site_groups(atoms)}
    expected = set()
    for site in review["affected_sites"]:
        labels = site.get("labels") if isinstance(site, dict) else None
        if (not isinstance(labels, list) or not labels or any(not isinstance(v, str) for v in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError("Invalid reviewed site labels")
        xyz = site.get("fractional_coordinates")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError("Invalid reviewed site coordinates")
        for coordinate in xyz:
            _number(coordinate)
        key = tuple(sorted(labels))
        if key not in groups or not set(labels) <= set(by_label):
            raise ValueError("Reviewed source site is not an exact co-located native site")
        total = math.fsum(by_label[label]["occupancy"] for label in labels)
        if not 1 < total <= 1 + limit + 1e-12 or not math.isclose(total, _number(site.get("total")), rel_tol=0, abs_tol=1e-10):
            raise ValueError("Reviewed native site does not retain its source occupancy total")
        components = {}
        for label in labels:
            row = by_label[label]
            components.setdefault(_element(row["type"]), []).append(row["occupancy"])
        components = {key: math.fsum(values) for key, values in components.items()}
        supplied = site.get("components")
        if (not isinstance(supplied, dict) or set(components) != set(supplied)
                or any(not math.isclose(value, _number(supplied[key]), rel_tol=0, abs_tol=1e-10)
                       for key, value in components.items())):
            raise ValueError("Reviewed native site changed its source species inventory")
        expected.add(key)
    for group in groups:
        if math.fsum(by_label[label]["occupancy"] for label in group) > 1 + 1e-6 and group not in expected:
            raise ValueError("An overfilled native site lacks its exact CIF review")
    return [list(group) for group in sorted(expected)]


def _source_atoms(rows):
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError("Missing bounded source atom inventory")
    atoms = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("label"), str) or not row["label"]:
            raise ValueError("Invalid source atom label")
        xyz = row.get("xyz")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError("Invalid source atom coordinates")
        atoms.append({"label": row["label"], "symbol": _element(row["symbol"]),
                      "occupancy": _number(row["occupancy"]), "xyz": [_number(v) for v in xyz]})
    if len({row["label"] for row in atoms}) != len(atoms):
        raise ValueError("Source atom labels must be unique")
    return atoms


def _check_source_atoms(review, source_atoms, native_atoms, shift):
    if _digest(source_atoms) != review["atom_signature"]:
        raise ValueError("Native occupancy receipt changed the reviewed source atom signature")
    by_label = {row["label"]: row for row in native_atoms}
    if set(by_label) != {row["label"] for row in source_atoms}:
        raise ValueError("Native occupancy receipt changed the source atom labels")
    for source in source_atoms:
        native = by_label[source["label"]]
        delta = [abs((a - b - s + .5) % 1 - .5)
                 for a, b, s in zip(native["coordinates"], source["xyz"], shift)]
        if (_element(native["type"]) != source["symbol"]
                or not math.isclose(native["occupancy"], source["occupancy"], rel_tol=0, abs_tol=1e-10)
                or any(v > 1e-8 for v in delta)):
            raise ValueError("Native occupancy receipt does not preserve the reviewed source sites")


def make_receipt(phase, review, *, source_sha256, imported_cif_sha256, source_atoms, source_model):
    """Called only after authorized CIF parsing and exact native import checks."""
    _policy(review, source_sha256)
    if not _sha(imported_cif_sha256):
        raise ValueError("Missing imported-CIF identity for occupancy review")
    atoms = _inventory(phase)
    source_atoms = _source_atoms(source_atoms)
    symmetry = _symmetry(phase)
    shift = _source_origin(review, source_model, symmetry)
    _check_source_atoms(review, source_atoms, atoms, shift)
    receipt = {"schema": SCHEMA, "source_sha256": source_sha256,
        "imported_cif_sha256": imported_cif_sha256, "review": deepcopy(review),
        "symmetry": symmetry, "native_atoms": atoms, "source_atoms": source_atoms,
        "source_model": deepcopy(source_model),
        "reviewed_groups": _reviewed_groups(review, atoms)}
    receipt["receipt_sha256"] = _digest(receipt)
    return receipt


def validate_receipt(phase):
    """Validate a durable receipt against this actual native phase, not a note."""
    if DATA_KEY not in phase.data:
        return None
    receipt = phase.data[DATA_KEY]
    required = {"schema", "source_sha256", "imported_cif_sha256", "review",
                "symmetry", "native_atoms", "source_atoms", "source_model", "reviewed_groups", "receipt_sha256"}
    if (not isinstance(receipt, dict) or set(receipt) != required or receipt.get("schema") != SCHEMA
            or not _sha(receipt.get("receipt_sha256"))
            or receipt["receipt_sha256"] != _digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})):
        raise ValueError("Native CIF occupancy receipt is malformed or changed. " + REBUILD)
    _policy(receipt["review"], receipt["source_sha256"])
    if not _sha(receipt["imported_cif_sha256"]) or receipt["symmetry"] != _symmetry(phase):
        raise ValueError("Native CIF occupancy receipt symmetry changed. " + REBUILD)
    original = _checked_inventory(receipt["native_atoms"])
    source_atoms = _source_atoms(receipt["source_atoms"])
    shift = _source_origin(receipt["review"], receipt["source_model"], receipt["symmetry"])
    _check_source_atoms(receipt["review"], source_atoms, original, shift)
    if receipt["reviewed_groups"] != _reviewed_groups(receipt["review"], original):
        raise ValueError("Native reviewed-site membership changed. " + REBUILD)
    current = _inventory(phase)
    before = {row["label"]: row for row in original}
    after = {row["label"]: row for row in current}
    if len(before) != len(original) or set(before) != set(after):
        raise ValueError("Native reviewed-phase atom inventory changed. " + REBUILD)
    labels = reviewed_labels(receipt)
    for label, old in before.items():
        new = after[label]
        if new["type"] != old["type"] or new["multiplicity"] != old["multiplicity"]:
            raise ValueError("Native reviewed-phase atom type or multiplicity changed. " + REBUILD)
        if label in labels:
            delta = [abs((a - b + .5) % 1 - .5) for a, b in zip(new["coordinates"], old["coordinates"])]
            if new["occupancy"] != old["occupancy"] or any(v > 1e-10 for v in delta):
                raise ValueError("Reviewed occupancy/position must remain exactly held. " + REBUILD)
    actual_groups = {tuple(sorted(group)) for group in site_groups(current)}
    if any(tuple(group) not in actual_groups for group in receipt["reviewed_groups"]):
        raise ValueError("A new or moved atom changed the reviewed native site. " + REBUILD)
    retained = {tuple(group) for group in receipt["reviewed_groups"]}
    for group in actual_groups - retained:
        if math.fsum(after[label]["occupancy"] for label in group) > 1 + 1e-6:
            raise ValueError("An unreviewed current site exceeds physical occupancy capacity. " + REBUILD)
    return deepcopy(receipt)


def reviewed_labels(receipt):
    receipt = receipt or {}
    if "reviewed_groups" in receipt:
        return {label for group in receipt["reviewed_groups"] for label in group}
    return {atom["label"] for group in receipt.get("held_sites", []) for atom in group}


def retention_binding(receipt):
    """Bind reusable retention semantics, independently of scan-specific files.

    The complete receipt is still validated against each native source. Cell
    derivatives, unrelated oxygen positions and ADPs must not prevent applying
    one approved recipe to the same reviewed model in another scan.
    """
    if receipt is None:
        return None
    review = receipt["review"]
    atoms = {row["label"]: row for row in receipt["native_atoms"]}
    return {"schema": SCHEMA, "policy_sha256": review.get("root_policy_sha256", review["policy_sha256"]),
        "maximum_site_excess": review["maximum_site_excess"], "symmetry": deepcopy(receipt["symmetry"]),
        "held_sites": [[deepcopy(atoms[label]) for label in group] for group in receipt["reviewed_groups"]]}


def check_stage(stage, context):
    """Reject requested movement/normalization before any native solver call."""
    labels = {p["key"]: reviewed_labels(p.get("occupancy_review")) for p in context["phases"]}
    for target in stage["free"]:
        if (target.get("kind") == "atom" and target.get("atom") in labels.get(target.get("phase"), set())
                and set(target.get("flags", "")) & {"X", "F"}):
            raise ValueError("Reviewed CIF rounding permits retention only: affected site positions and occupancies must stay held.")
    for constraint in stage.get("constraints", []):
        if constraint.get("kind") == "occupancy_sum" and any(
                term.get("atom") in labels.get(term.get("phase"), set()) for term in constraint.get("terms", [])):
            raise ValueError("A reviewed retained site cannot be normalized or added to a new occupancy equation.")


def check_active_parameters(state, context, *, dependent=(), inverse=None):
    dependent = {str(v.varname() if hasattr(v, "varname") else v) for v in dependent}
    effective = {str(v) for v in state["parameter_space"].get("effective_vary", [])}
    for phase in context.get("phases", []):
        labels = reviewed_labels(phase.get("occupancy_review"))
        if not labels:
            continue
        ids = {a["native_id"] for a in phase["atoms"] if a["label"] in labels}
        if not ids:
            continue
        for row in state["parameter_space"]["parameters"]:
            if (row.get("phase_id") != phase["native_id"] or row.get("atom_id") not in ids
                    or row.get("group") not in {"occupancy", "atomic_coordinates"}):
                continue
            if row.get("configured_now") or row.get("effective_now"):
                raise ValueError("Native constraints or flags release a reviewed retained occupancy/position; refinement withheld.")
            variable = str(row.get("refinement_variable", row["name"]))
            if variable in dependent:
                if inverse is None:
                    raise ValueError("Reviewed dependent occupancy/position requires native constraint proof")
                variables, coefficients = inverse(variable, state["parameter_space"].get("effective_vary", []))
                if len(variables) != len(coefficients) or any(
                        str(v) in effective and _number(coefficient) != 0
                        for v, coefficient in zip(variables, coefficients)):
                    raise ValueError("Native constraints release a reviewed retained occupancy/position; refinement withheld.")
