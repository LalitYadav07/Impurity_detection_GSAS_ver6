"""Source-bound retained rounding is not permission to refine or overfill sites."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gpx_recipe import occupancy_review as review
from gpx_recipe.advanced import check_site_totals, validate_stage


def fixture(attach=True, second_oxygen=False):
    source = [
        dict(label="C1", symbol="C", occupancy=.6, xyz=[0., 0., 0.]),
        dict(label="Li1", symbol="Li", occupancy=.40066, xyz=[0., 0., 0.]),
        dict(label="O1", symbol="O", occupancy=1., xyz=[.3, .4, .5]),
    ]
    if second_oxygen:
        source.append(dict(label='O2', symbol='O', occupancy=1., xyz=[.7, .6, .5]))
    phase = SimpleNamespace(name="reviewed", data={
        "General": {"Type": "nuclear", "AtomPtrs": [3, 1, 7, 9],
            "SGData": {"SpGrp": "P 1", "SGInv": False, "SGCen": [[0., 0., 0.]],
                "SGOps": [([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0., 0., 0.])]}},
        "Atoms": [[a["label"], a["symbol"], "", *a["xyz"], a["occupancy"], "1", 1, "I", .01] for a in source],
        "Histograms": {},
    })
    phase.atom = lambda label: SimpleNamespace(occupancy=next(a[6] for a in phase.data["Atoms"] if a[0] == label),
        refinement_flags=next(a[2] for a in phase.data["Atoms"] if a[0] == label))
    model = {'_space_group_name_h-m_alt': 'P 1', '_space_group_symop_operation_xyz': ['x,y,z'],
             '_atom_site_label': [a['label'] for a in source],
             '_atom_site_type_symbol': [a['symbol'] for a in source],
             '_atom_site_occupancy': [str(a['occupancy']) for a in source]}
    phase.data['General']['SGData'] = copy.deepcopy(review._source_space_group(model))
    policy = {"schema": review.POLICY_SCHEMA, "reviewed": True, "normalization_applied": False,
        "maximum_site_excess": .001, "review_note": "Retain reviewed source rounding; no normalization",
        "source_sha256": "a" * 64, "canonical_sha256": "a" * 64,
        "atom_signature": review._digest(source), "model_signature": "b" * 64,
        "nuclear_model_signature": review._digest(model),
        "affected_sites": [{"labels": ["C1", "Li1"], "fractional_coordinates": [0., 0., 0.],
            "total": 1.00066, "components": {"C": .6, "Li": .40066}}]}
    policy["policy_sha256"] = review._digest(policy)
    if attach:
        phase.data[review.DATA_KEY] = review.make_receipt(phase, policy,
            source_sha256="a" * 64, imported_cif_sha256="a" * 64, source_atoms=source, source_model=model)
    atoms = [{"label": row[0], "native_id": i, "coordinates": row[3:6], "occupancy": row[6], "multiplicity": row[8]}
             for i, row in enumerate(phase.data["Atoms"])]
    context = {"phases": [{"key": "p0", "native_id": 7, "name": phase.name,
        "atoms": atoms, "histograms": [], "occupancy_groups": [["C1", "Li1"]]}], "histograms": []}
    if attach:
        context["phases"][0]["occupancy_review"] = copy.deepcopy(phase.data[review.DATA_KEY])
    return phase, context, policy, source


def project(phase):
    return SimpleNamespace(phase=lambda name: phase)


def test_fixed_source_rounding_survives_unrelated_oxygen_cell_and_uiso_changes():
    phase, context, _, _ = fixture()
    before = copy.deepcopy(phase.data["Atoms"])
    phase.data["Atoms"][2][3] = .315
    phase.data["Atoms"][0][2] = "U"
    phase.data["General"]["Cell"] = [True, 5., 5., 6., 90., 90., 90., 150.]
    phase.name = 'Publication display label'
    assert review.validate_receipt(phase)["reviewed_groups"] == [["C1", "Li1"]]
    check_site_totals(project(phase), context)
    assert [row[6] for row in phase.data["Atoms"]] == [row[6] for row in before]


def test_semantic_binding_ignores_lattice_derivative_receipts_and_unrelated_atoms():
    phase, _, policy, source = fixture()
    binding = review.retention_binding(review.validate_receipt(phase))
    original_policy = policy['policy_sha256']
    policy.update(root_policy_sha256=original_policy, parent_policy_sha256=original_policy,
                  derived_sha256='d' * 64)
    policy['policy_sha256'] = review._digest({k: v for k, v in policy.items() if k != 'policy_sha256'})
    phase.data[review.DATA_KEY] = review.make_receipt(phase, policy,
        source_sha256='d' * 64, imported_cif_sha256='d' * 64, source_atoms=source,
        source_model=phase.data[review.DATA_KEY]['source_model'])
    phase.data['Atoms'][2][3] += .01
    phase.data['Atoms'][2][10] += .02
    phase.data['General']['Cell'] = [False, 4.1, 4., 4., 90., 90., 90., 65.6]
    assert review.retention_binding(review.validate_receipt(phase)) == binding
    changed = copy.deepcopy(phase.data[review.DATA_KEY])
    changed['review']['maximum_site_excess'] = .0008
    assert review.retention_binding(changed) != binding


def test_unreviewed_old_gpx_fails_with_rebuild_guidance_before_any_solver(tmp_path, monkeypatch):
    from gpx_recipe import execution
    phase, context, _, _ = fixture(False)
    source = tmp_path / "source.gpx"
    source.write_bytes(b"immutable synthetic source")
    context.update(source={"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}, model_signature="test")
    p = project(phase)
    p.save = lambda filename: Path(filename).write_bytes(b"derived only")
    p.do_refinements = lambda *args: pytest.fail("A legacy unreviewed overfill reached the solver")
    monkeypatch.setattr(execution, "snapshot", lambda _: context)
    monkeypatch.setattr(execution, "validate_plan", lambda *args: None)
    monkeypatch.setattr(execution, "configure", lambda *args: None)
    monkeypatch.setattr(execution, "stage_table", lambda *args: [{}])
    monkeypatch.setattr(execution, "gsas", lambda: SimpleNamespace(G2Project=lambda **kwargs: p))
    monkeypatch.setattr(execution, "extract_gsasii_refinement_state", lambda *args, **kwargs: pytest.fail("Invalid initial sites reached parameter preflight"))
    plan = {"questions": [], "unsupported": [], "stages": [{"free": [], "constraints": []}]}
    with pytest.raises(ValueError, match="Rebuild this GPX"):
        execution.run_plan(source, tmp_path / "derived.gpx", plan, preview=True)
    assert source.read_bytes() == b"immutable synthetic source"
    report = json.loads((tmp_path / "derived.execution.json").read_text())
    assert report["status"] == "failed" and report["stages"] == []


@pytest.mark.parametrize("flag", ["F", "X", "FX", "XU"])
def test_requested_release_of_reviewed_members_is_rejected_during_plan_validation(flag):
    _, context, _, _ = fixture()
    stage = {"free": [{"kind": "atom", "phase": "p0", "atom": "Li1", "flags": flag}], "constraints": []}
    with pytest.raises(ValueError, match="retention only"):
        validate_stage(stage, context, set())


def test_unrelated_atom_and_cell_operations_are_not_blocked():
    _, context, _, _ = fixture()
    for target in [{"kind": "cell", "phase": "p0"}, {"kind": "atom", "phase": "p0", "atom": "O1", "flags": "X"},
                   {"kind": "atom", "phase": "p0", "atom": "Li1", "flags": "U"}]:
        review.check_stage({"free": [target], "constraints": []}, context)


def test_new_normalization_equation_cannot_reinterpret_the_source_review():
    _, context, _, _ = fixture()
    stage = {"free": [], "constraints": [{"kind": "occupancy_sum", "terms": [{"phase": "p0", "atom": "C1"}]}]}
    with pytest.raises(ValueError, match="normalized"):
        review.check_stage(stage, context)


@pytest.mark.parametrize("field", ["configured_now", "effective_now"])
def test_constraint_induced_release_is_blocked_even_without_an_explicit_target(field):
    _, context, _, _ = fixture()
    state = {"parameter_space": {"parameters": [{"phase_id": 7, "atom_id": 1, "group": "occupancy", field: True}]}}
    with pytest.raises(ValueError, match="constraints or flags"):
        review.check_active_parameters(state, context)


def test_inherited_constraint_cannot_move_an_inactive_dependent_reviewed_occupancy():
    _, context, _, _ = fixture()
    row = {"name": "7::Afrac:1", "phase_id": 7, "atom_id": 1, "group": "occupancy",
           "configured_now": False, "effective_now": False}
    state = {"parameter_space": {"parameters": [row], "effective_vary": ["7::Afrac:2"]}}
    with pytest.raises(ValueError, match="constraints release"):
        review.check_active_parameters(state, context, dependent=["7::Afrac:1"],
            inverse=lambda variable, active: (["7::Afrac:2"], [1e-15]))
    review.check_active_parameters(state, context, dependent=["7::Afrac:1"],
        inverse=lambda variable, active: (["7::Afrac:2"], [0.]))
    with pytest.raises(ValueError, match="constraint proof"):
        review.check_active_parameters(state, context, dependent=["7::Afrac:1"])


@pytest.mark.parametrize("mutation", ["occupancy", "redistribution", "position", "member", "label", "type", "multiplicity", "symmetry"])
def test_modified_site_or_symmetry_cannot_use_the_durable_review(mutation):
    phase, context, _, _ = fixture()
    rows = phase.data["Atoms"]
    if mutation == "occupancy": rows[0][6] += 1e-12
    elif mutation == "redistribution": rows[0][6] += .00001; rows[1][6] -= .00001
    elif mutation == "position": rows[0][3] += .00001
    elif mutation == "member": rows[2][3:6] = [0., 0., 0.]
    elif mutation == "label": rows[0][0] = "other"
    elif mutation == "type": rows[0][1] = "N"
    elif mutation == "multiplicity": rows[0][8] = 2
    elif mutation == "symmetry": phase.data["General"]["SGData"]["SGOps"][0][1][0] = .5
    with pytest.raises(ValueError):
        check_site_totals(project(phase), context)


@pytest.mark.parametrize("mutation", ["hash", "source", "limit", "normalized", "source_atoms", "native_atoms"])
def test_corrupt_or_relabelled_receipts_are_rejected(mutation):
    phase, _, _, _ = fixture()
    r = phase.data[review.DATA_KEY]
    if mutation == "hash": r["receipt_sha256"] = "0" * 64
    elif mutation == "source": r["source_sha256"] = "d" * 64
    elif mutation == "limit": r["review"]["maximum_site_excess"] = .002
    elif mutation == "normalized": r["review"]["normalization_applied"] = True
    elif mutation == "source_atoms": r["source_atoms"][0]["occupancy"] += .00001
    else:
        # Even a rewritten native inventory must still match the source's
        # independently bound raw atom signature and values.
        r["native_atoms"][0]["occupancy"] += .00001
        phase.data["Atoms"][0][6] += .00001
    if mutation != "hash":
        r["review"]["policy_sha256"] = review._digest({k: v for k, v in r["review"].items() if k != "policy_sha256"})
        r["receipt_sha256"] = review._digest({k: v for k, v in r.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError): review.validate_receipt(phase)


def test_removing_context_authority_or_adding_a_generic_warning_is_not_permission():
    phase, context, _, _ = fixture()
    context["phases"][0].pop("occupancy_review")
    with pytest.raises(ValueError, match="changed after inspection"):
        check_site_totals(project(phase), context)
    phase.data.pop(review.DATA_KEY)
    phase.data["warning"] = "Reviewed occupancy rounding <=0.001"
    with pytest.raises(ValueError, match="Rebuild this GPX"):
        check_site_totals(project(phase), context)


@pytest.mark.parametrize("mutation", ["short_coordinates", "boolean_occupancy", "boolean_multiplicity", "duplicate_atom", "bad_site_coordinates"])
def test_malformed_rehashed_inventory_is_not_native_evidence(mutation):
    phase, _, _, _ = fixture()
    r = phase.data[review.DATA_KEY]
    if mutation == "short_coordinates": r["native_atoms"][2]["coordinates"] = [.3, .4]
    elif mutation == "boolean_occupancy": r["native_atoms"][2]["occupancy"] = True
    elif mutation == "boolean_multiplicity": r["native_atoms"][2]["multiplicity"] = True
    elif mutation == "duplicate_atom": r["native_atoms"].append(copy.deepcopy(r["native_atoms"][0]))
    else:
        r["review"]["affected_sites"][0]["fractional_coordinates"] = [0., 0.]
        r["review"]["policy_sha256"] = review._digest({k: v for k, v in r["review"].items() if k != "policy_sha256"})
    r["receipt_sha256"] = review._digest({k: v for k, v in r.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError): review.validate_receipt(phase)


@pytest.mark.parametrize('change', ['inversion', 'operator', 'origin_0123', 'origin_quarter', 'source_model'])
def test_receipt_rehash_cannot_change_policy_bound_symmetry_or_origin(change):
    phase, _, _, _ = fixture()
    receipt = phase.data[review.DATA_KEY]
    if change == 'inversion':
        phase.data['General']['SGData']['SGInv'] = True
        receipt['symmetry']['SGInv'] = True
    elif change == 'operator':
        phase.data['General']['SGData']['SGOps'][0][1][0] = .5
        receipt['symmetry']['SGOps'][0][1][0] = .5
    elif change == 'source_model':
        receipt['source_model']['_space_group_name_h-m_alt'] = 'P -1'
    else:
        shift = .123 if change == 'origin_0123' else .25
        for row in phase.data['Atoms']:
            row[3] += shift
        for row in receipt['native_atoms']:
            row['coordinates'][0] += shift
    receipt['receipt_sha256'] = review._digest({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    with pytest.raises(ValueError):
        review.validate_receipt(phase)


@pytest.mark.parametrize('with_receipt', [False, True])
def test_unreviewed_singleton_cannot_borrow_reviewed_site_allowance(with_receipt):
    phase, context, _, _ = fixture(with_receipt)
    phase.data['Atoms'][2][6] = 1.0005
    if with_receipt:
        with pytest.raises(ValueError, match='unreviewed current site'):
            review.validate_receipt(phase)
    with pytest.raises(ValueError):
        check_site_totals(project(phase), context)


def test_unrelated_new_colocation_cannot_create_an_unreviewed_overfilled_site():
    phase, context, _, _ = fixture(second_oxygen=True)
    phase.data['Atoms'][3][3:6] = phase.data['Atoms'][2][3:6]
    with pytest.raises(ValueError, match='unreviewed current site'):
        review.validate_receipt(phase)
    with pytest.raises(ValueError):
        check_site_totals(project(phase), context)


def test_periodic_coordinate_identity_is_allowed_but_current_native_flags_are_held():
    phase, context, _, _ = fixture()
    phase.data["Atoms"][0][3] += 1
    check_site_totals(project(phase), context)
    phase.data["Atoms"][1][2] = "F"
    with pytest.raises(ValueError, match="held site"):
        check_site_totals(project(phase), context)


def test_local_scientific_import_fallback_needs_no_nova_web_package():
    root = Path(__file__).resolve().parents[2]
    code = '''import sys
sys.path.insert(0, sys.argv[1])
import gsas_cif_import
module = gsas_cif_import._occupancy_review_module()
assert module.SCHEMA == "radar-gpx-occupancy-review/v2"
assert "radar_pd_nova" not in sys.modules
print(module.__file__)
'''
    result = subprocess.run([sys.executable, '-I', '-c', code, str(root / 'scripts')],
                            capture_output=True, text=True, check=True, timeout=20)
    assert Path(result.stdout.strip()) == root / 'nova/src/gpx_recipe/occupancy_review.py'
