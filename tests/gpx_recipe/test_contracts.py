"""Plan/approval boundaries and one-shot planner behavior without network calls."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from gpx_recipe.contracts import validate_plan, plan_hash, stage_table
from gpx_recipe.compiler import approve, compile_script
from gpx_recipe.planner import make_plan, build_payload


@pytest.fixture
def context():
    return {"model_signature": "model-a", "source": {"sha256": "source-a"},
            "histograms": [{"key": "h0", "name": "PWDR test"}],
            "phases": [{"key": "p0", "name": "phase", "histograms": ["PWDR test"],
                        "atoms": [{"label": "O1", "element": "O", "adp_type": "I", "position_dof": 1},
                                  {"label": "Mg1", "element": "Mg", "adp_type": "I", "position_dof": 0}]}]}


@pytest.fixture
def plan():
    return {"schema": "radar-gpx-recipe/v1", "model_signature": "model-a", "request": "Refine O1 positions then only its Uiso.",
            "assumptions": [], "questions": [], "unsupported": [],
            "stages": [{"id": "positions", "title": "Positions", "cycles": 5,
                        "free": [{"kind": "atom", "phase": "p0", "atom": "O1", "flags": "X"}], "constraints": []},
                       {"id": "adp", "title": "Uiso", "cycles": 5,
                        "free": [{"kind": "atom", "phase": "p0", "atom": "O1", "flags": "U"}], "constraints": []}]}


def test_full_active_sets_show_fixed_transition(context, plan):
    validate_plan(plan, context)
    table = stage_table(plan)
    assert table[1]["free"] == ["p0:O1:U"]
    assert table[1]["newly_fixed"] == ["p0:O1:X"]


@pytest.mark.parametrize("change", ["unknown_atom", "wrong_model", "script", "unknown_kind", "duplicate", "symmetry_fixed", "boolean_cycles", "too_many_cycles", "unsupported_background"])
def test_rejects_invalid_or_unexecutable_plans(context, plan, change):
    target = plan["stages"][0]["free"][0]
    if change == "unknown_atom": target["atom"] = "O99"
    elif change == "wrong_model": plan["model_signature"] = "another-model"
    elif change == "script": plan["python"] = "import os"
    elif change == "unknown_kind": target["kind"] = "instrument"
    elif change == "duplicate": plan["stages"][0]["free"].append(copy.deepcopy(target))
    elif change == "symmetry_fixed": target["atom"] = "Mg1"
    elif change == "boolean_cycles": plan["stages"][0]["cycles"] = True
    elif change == "too_many_cycles": plan["stages"][0]["cycles"] = 500
    elif change == "unsupported_background":
        plan["stages"][0]["background"] = {"histogram": "h0", "function": "invented", "coefficients": 8, "fixed_points": "preserve"}
    with pytest.raises(ValueError): validate_plan(plan, context)


def test_approval_binds_exact_plan_and_source_preview(context, plan, tmp_path):
    preview = {"status": "preview_passed", "plan_sha256": plan_hash(plan), "model_signature": "model-a", "source_sha256": "source-a"}
    receipt = approve(plan, context, preview, approver="test-user")
    compile_script(plan, receipt, tmp_path / "refine.py")
    plan["stages"][0]["cycles"] = 6
    with pytest.raises(ValueError, match="changed or unapproved"):
        compile_script(plan, receipt, tmp_path / "other.py")


def test_no_approval_of_a_failed_or_wrong_project_preview(context, plan):
    for status, source in [("failed", "source-a"), ("preview_passed", "different")]:
        with pytest.raises(ValueError):
            approve(plan, context, {"status": status, "plan_sha256": plan_hash(plan), "model_signature": "model-a", "source_sha256": source}, approver="user")


def test_model_strings_are_inert_in_compiled_script(context, plan, tmp_path):
    plan["request"] = "PLAN_LITERAL APPROVAL_LITERAL '''; import os; raise RuntimeError('injected')"
    preview = {"status": "preview_passed", "plan_sha256": plan_hash(plan), "model_signature": "model-a", "source_sha256": "source-a"}
    receipt = approve(plan, context, preview, approver="user")
    code = compile_script(plan, receipt, tmp_path / "refine.py")
    namespace = {"__name__": "test_import"}
    exec(compile(code, "generated", "exec"), namespace)
    assert namespace["PLAN"] == plan


class FakeClient:
    def __init__(self, responses): self.responses, self.calls = responses, []
    def complete(self, messages):
        self.calls.append(copy.deepcopy(messages))
        return self.responses[len(self.calls)-1], {"model": "test"}


def test_one_shot_then_revision_preserves_previous_plan(context, plan):
    client = FakeClient([json.dumps(plan)])
    result = make_plan("refine oxygen", context, previous=plan, client=client)
    assert len(client.calls) == 1 and result["status"] == "ready_for_preflight"
    assert json.loads(client.calls[0][1]["content"])["previous_plan"] == plan


def test_only_one_bounded_repair(context, plan):
    client = FakeClient(["not JSON", json.dumps(plan)])
    assert len(make_plan("request", context, client=client)["calls"]) == 2
    client = FakeClient(["not JSON", "still invalid"])
    with pytest.raises(ValueError, match="bounded validation"):
        make_plan("request", context, client=client)
    assert len(client.calls) == 2


def test_no_silent_context_truncation(context):
    context["large"] = "x" * 350000
    with pytest.raises(ValueError, match="budget"):
        build_payload("request", context)


def test_payload_exposes_symmetry_and_adp_capabilities(context):
    caps = build_payload("refine an atom", context)["target_capabilities"][0]["atoms"]
    assert caps[0]["label"] == "O1" and caps[0]["allowed_flags"] == "FXU"
    assert caps[1]["label"] == "Mg1" and caps[1]["allowed_flags"] == "FU"


def test_first_stage_shows_preexisting_cell_becoming_fixed(context, plan):
    context["phases"][0]["cell_refined"] = True
    assert "cell:p0" in stage_table(plan, context)[0]["newly_fixed"]


def test_unanswerable_request_does_not_force_invented_stages(context, plan):
    plan["stages"] = []
    plan["unsupported"] = ["Magnetic refinement is not implemented in version 1"]
    validate_plan(plan, context)
    response = make_plan("refine magnetic moments", context, client=FakeClient([json.dumps(plan), json.dumps(plan)]))
    assert response["status"] == "needs_input"
    with pytest.raises(ValueError, match="Resolve"):
        approve(plan, context, {}, approver="user")


def test_token_limit_is_not_accepted_even_if_json_parses(context, plan):
    class TruncatedClient:
        def __init__(self): self.calls = 0
        def complete(self, messages):
            self.calls += 1
            return json.dumps(plan), {"finish_reason": "length"}
    client = TruncatedClient()
    with pytest.raises(ValueError, match="complete text plan"):
        make_plan("full recipe", context, client=client)
    assert client.calls == 2


def test_false_unsupported_claim_gets_one_capability_check(context,plan):
    refusal=copy.deepcopy(plan);refusal.update(stages=[],unsupported=['O1 positions are not supported'])
    client=FakeClient([json.dumps(refusal),json.dumps(plan)])
    result=make_plan('Refine O1 positions',context,client=client)
    assert len(client.calls)==2 and result['status']=='ready_for_preflight'


def test_joint_numbered_stage_cannot_be_silently_split(context,plan):
    joint=copy.deepcopy(plan);joint['stages']=joint['stages'][:1];joint['stages'][0]['free'][0]['flags']='XU'
    client=FakeClient([json.dumps(plan),json.dumps(joint)])
    result=make_plan('In one stage refine O1 positions and Uiso together.',context,client=client)
    assert len(client.calls)==2 and len(result['plan']['stages'])==1
    assert 'explicitly specified 1 stage' in client.calls[1][-1]['content']


def test_constraint_cannot_remove_only_requested_occupancy_freedom(context, plan):
    plan['stages'] = plan['stages'][:1]
    plan['stages'][0]['free'][0]['flags'] = 'F'
    good = copy.deepcopy(plan)
    plan['stages'][0]['constraints'] = [{
        'kind': 'occupancy_sum', 'reason': 'Invented site total', 'total': 1,
        'terms': [{'phase': 'p0', 'atom': label, 'coefficient': 1} for label in ['O1', 'Mg1']]}]
    with pytest.raises(ValueError, match='exactly one released'):
        validate_plan(plan, context)
    client = FakeClient([json.dumps(plan), json.dumps(good)])
    result = make_plan('Refine only O1 occupancy; keep all other atoms fixed.', context, client=client)
    assert len(client.calls) == 2 and result['plan']['stages'][0]['constraints'] == []
