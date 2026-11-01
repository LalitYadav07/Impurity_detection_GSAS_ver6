import copy
import json

from gpx_recipe.planner import build_payload, expand_context
from gpx_recipe.setup_context import setup_context
from gpx_recipe.snapshot import digest
from test_contracts import context


def test_actual_planner_payload_excludes_fit_and_future_diagnostic_fields(context):
    context.update(covariance_summary={"marker": "SECRET_COVARIANCE"},
                   transition_labels=["SECRET_TRANSITION"],
                   candidate_rankings=["SECRET_RANKING"],
                   future_diagnostic={"value": "SECRET_FUTURE"})
    context["histograms"][0].update(rwp="SECRET_RWP", future_fit_statistic="SECRET_HISTOGRAM")
    context["phases"][0]["atoms"][0].update(standard_uncertainty="SECRET_SIGMA",
        variables=[dict(name="0::AUiso:0", group="atomic_displacement", current_value=.01,
                        configured_now=True, status="SECRET_STATUS", effective_now="SECRET_EFFECTIVE")])
    context["parameter_records"] = [dict(name="0::AUiso:0", current_value=.01,
                                        configured_now=True, frozen_at_limit="SECRET_FROZEN")]
    original = copy.deepcopy(context)
    payload = build_payload("Refine oxygen position; hold Li fixed", context)
    assert "SECRET_" not in json.dumps(payload)
    assert context == original
    project = expand_context(payload["project"])
    assert project["phases"][0]["atoms"][0]["variables"][0]["current_value"] == .01
    assert project["parameter_records"][0]["configured_now"] is True


def test_setup_hash_binds_projection_and_ignores_fit_outcomes(context):
    projected = setup_context(context)
    fingerprint = projected.pop("context_sha256")
    assert fingerprint == digest(projected)
    context["covariance_summary"] = {"Rwp": 0.0001}
    assert setup_context(context)["context_sha256"] == fingerprint
    context["phases"][0]["atoms"][0]["occupancy"] = .9
    assert setup_context(context)["context_sha256"] != fingerprint


def test_exact_constraints_bounds_and_site_definitions_remain(context):
    context.update(binding={"source_bounds": {"Li": [.0, .951]}},
                   source_constraints={"Phase": [["native equation"]]},
                   uiso_equivalence_groups=[[{"phase": "p0", "atom": "O1"}]],
                   controls={"parmMinDict": {"Li": 0}, "parmMaxDict": {"Li": .951},
                             "last_fit_summary": "HIDDEN_OUTCOME"})
    projected = setup_context(context)
    assert projected["binding"] == context["binding"]
    assert projected["source_constraints"] == context["source_constraints"]
    assert projected["uiso_equivalence_groups"] == context["uiso_equivalence_groups"]
    assert projected["controls"]["parmMaxDict"]["Li"] == .951
    assert "HIDDEN_OUTCOME" not in json.dumps(projected)


def test_phase_histogram_model_retains_configuration_not_new_results(context):
    context["phases"][0]["histogram_phase_models"] = {"h0": {
        "Size": ["isotropic", [1, 1, 1]], "Mustrain": ["uniaxial", [1, 1]],
        "fit_quality": "PRIVATE_RESULT"}}
    projected = setup_context(context)
    assert "Size" in projected["phases"][0]["histogram_phase_models"]["h0"]
    assert "PRIVATE_RESULT" not in json.dumps(projected)
