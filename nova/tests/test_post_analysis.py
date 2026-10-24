"""Scientific edge cases for the replayable post-analysis framework."""

import copy
from datetime import datetime, timedelta, timezone
import io
import json
import math
import zipfile
from pathlib import Path

import pytest

from radar_pd_nova.post_analysis import (
    PostAnalysisSettings,
    ViewSettings,
    AxisMapping,
    calculate_views,
    _d_spacing,
    RECIPES,
)
from radar_pd_nova.post_analysis_plot import export_bundle, view_figure
from radar_pd_nova.configuration import config_from_contract
from radar_pd_nova.models import AnalysisConfig
from radar_pd_nova.powgen_watch import WatchState


def cell(phase="phase-a", a=4.0, fraction=60.0):
    covariance = [[0.0] * 7 for _ in range(7)]
    for i in [0, 1, 2, 6]:
        covariance[i][i] = 1e-6
    covariance[0][2] = covariance[2][0] = 4e-7
    return {
        "phase_id": phase,
        "phase_key": phase,
        "phase": phase,
        "basis_key": "orthorhombic",
        "a": a,
        "b": 5.0,
        "c": 6.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
        "volume": a * 30,
        "metric_states": {
            k: ("fixed" if k in {"alpha", "beta", "gamma"} else "refined")
            for k in ("a", "b", "c", "alpha", "beta", "gamma", "volume")
        },
        "cell_covariance": covariance,
        "covariance_fresh": True,
        "weight_percent": fraction,
        "fraction_state": "refined",
    }


def scan(
    run, *, a=4.0, sample="sample-1", bank=1, temperature=None, profile="profile-1", phases=None
):
    start = (datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=run)).isoformat()
    return {
        "run_id": f"NOM_{run}_BANK{bank}",
        "run_number": run,
        "bank": bank,
        "instrument": "NOM",
        "ipts": "IPTS-1",
        "metadata": {
            "sample_id": sample,
            "start_time": start,
            "end_time": start,
            "temperature": {
                "value": temperature if temperature is not None else 300 + run * 10,
                "unit": "K",
                "source": "selected sensor",
            },
        },
        "observables": {
            "$schema": "radar-pd-observables/v1",
            "source_gpx_sha256": f"hash-{run}",
            "context": {"profile_sha256": profile, "recipe_key": "full-v1", "reduction_key": "v1"},
            "phases": phases if phases is not None else [cell(a=a)],
        },
    }


def settings(kind="thermal_expansion", **kwargs):
    return PostAnalysisSettings(
        axis="run_number", views=[ViewSettings(id="view", kind=kind, **kwargs)]
    )


def view(scans, kind="thermal_expansion", **kwargs):
    return calculate_views(scans, settings(kind, **kwargs))["views"][0]


def values(result):
    return [p["value"] for s in result["series"] for p in s["points"]]


def test_configuration_and_checkpoint_round_trip_keep_instances_and_pinned_references():
    post = PostAnalysisSettings(
        views=[
            ViewSettings(id="absolute", kind="thermal_expansion", representation="absolute"),
            ViewSettings(id="relative", kind="thermal_expansion", reference_run="NOM_1_BANK1"),
        ],
        references={"group": "run"},
    )
    config = AnalysisConfig(sample_elements=["Fe"], post_analysis=post)
    assert config_from_contract(config.portable_contract()).post_analysis == post
    state = WatchState(history_id="history", post_analysis=post.model_dump(mode="json"))
    assert WatchState.from_dict(json.loads(state.to_json())).post_analysis == state.post_analysis
    assert (
        WatchState.from_dict(
            {"$schema": "radar-pd-powgen-watch-state/v1", "history_id": "history"}
        ).post_analysis
        == {}
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2},
        {"unexpected": True},
        {"views": [{"id": "v", "kind": "unknown"}]},
        {"views": [{"id": "v", "kind": "phase_evolution"}, {"id": "v", "kind": "distortion"}]},
        {"axis_mappings": {"run_number": {}}},
    ],
)
def test_unknown_contracts_are_rejected_instead_of_silently_discarded(payload):
    with pytest.raises(ValueError):
        PostAnalysisSettings.model_validate(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"metric": "arbitrary_expression"},
        {"metric": "a", "kind": "phase_evolution"},
        {"hkl": (0, 0, 0)},
        {"progress_start": 1, "progress_end": 1},
        {"fit_min": 500, "fit_max": 400},
        {"max_temperature_span": -1},
        {"banks": [0]},
        {"metric": "weight_percent", "fit_expansion": True},
        {"progress_start": float("nan")},
    ],
)
def test_invalid_scientific_definitions_fail_validation(kwargs):
    with pytest.raises(ValueError):
        ViewSettings.model_validate({"id": "v", "kind": "thermal_expansion", **kwargs})


def test_relative_change_and_conditional_uncertainty_use_the_explicit_future_reference():
    result = view([scan(1, a=4), scan(2, a=5)], reference_run="NOM_2_BANK1")
    first, second = result["series"][0]["points"]
    assert first["value"] == pytest.approx(-20)
    expected = 100**2 * (1e-6 / 5**2 + 4**2 * 1e-6 / 5**4)
    assert first["variance"] == pytest.approx(expected)
    assert second["value"] == 0 and second["variance"] == 0
    assert "physical uncertainty" in second["uncertainty_note"]


def test_reference_freezes_on_later_replay_and_is_not_replaced_when_excluded():
    post = settings()
    initial = calculate_views([scan(2, a=5)], post)
    post.references = initial["references"]
    later = calculate_views([scan(3, a=6), scan(1, a=4), scan(2, a=5)], post)
    assert later["references"] == initial["references"]
    assert values(later["views"][0]) == pytest.approx([-20, 0, 20])
    post.views[0].excluded_runs = ["NOM_2_BANK1"]
    excluded = calculate_views([scan(1, a=4), scan(2, a=5)], post)
    assert all(v is None for v in values(excluded["views"][0]))
    assert excluded["references"] == initial["references"]


def test_earlier_pending_jobs_prevent_completion_order_from_choosing_the_baseline():
    result = calculate_views([scan(2)], settings(), pending=[scan(1)])
    assert not result["references"]
    assert result["views"][0]["status"] == "needs_input"
    assert values(result["views"][0]) == [None]
    done = calculate_views([scan(2, a=5), scan(1, a=4)], settings())
    assert values(done["views"][0]) == [0, 25]


@pytest.mark.parametrize(
    "change",
    ["sample", "bank", "profile", "phase", "basis", "recipe", "reduction", "configuration"],
)
def test_incompatible_scientific_groups_never_share_a_reference(change):
    first, second = scan(1), scan(2, a=5)
    if change == "sample":
        second["metadata"]["sample_id"] = "different"
    if change == "bank":
        second["bank"] = 2
    if change == "profile":
        second["observables"]["context"]["profile_sha256"] = "new"
    if change == "phase":
        second["observables"]["phases"][0]["phase_key"] = "same-name-new-phase"
    if change == "basis":
        second["observables"]["phases"][0]["basis_key"] = "new-setting"
    if change == "recipe":
        second["observables"]["context"]["recipe_key"] = "new-recipe"
    if change == "reduction":
        second["observables"]["context"]["reduction_key"] = "new-reduction"
    if change == "configuration":
        first["configuration_id"] = "config-one"
        second["configuration_id"] = "config-two"
    result = view([first, second])
    assert len(result["series"]) == 2
    assert [v for v in values(result) if v is not None] == [0, 0]


def test_unidentified_samples_are_not_merged_by_default():
    result = view([scan(1, sample=""), scan(2, sample="")])
    assert len(result["series"]) == 2


@pytest.mark.parametrize("state", ["fixed", "unknown", "stale"])
def test_fixed_unknown_and_stale_cell_values_are_not_reported_as_refined_trends(state):
    item = scan(1)
    item["observables"]["phases"][0]["metric_states"]["a"] = state
    result = view([item], representation="absolute")
    assert values(result) == [None]


def test_missing_covariance_does_not_become_zero_uncertainty():
    items = [scan(1), scan(2, a=5)]
    for item in items:
        item["observables"]["phases"][0].pop("cell_covariance")
    result = view(items)
    assert values(result) == [0, 25]
    assert result["series"][0]["points"][1]["variance"] is None


def test_ratio_uncertainty_includes_off_diagonal_cell_covariance():
    result = view([scan(1)], kind="distortion", metric="c/a")
    point = result["series"][0]["points"][0]
    assert point["value"] == 1.5
    expected = 1e-6 / 4**2 + 6**2 * 1e-6 / 4**4 - 2 * 6 * 4e-7 / 4**3
    assert point["variance"] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "cell_values,hkl,expected",
    [
        ([4, 4, 4, 90, 90, 90], (1, 0, 0), 4),
        ([4, 4, 4, 90, 90, 90], (1, 1, 1), 4 / math.sqrt(3)),
        ([3, 3, 5, 90, 90, 120], (1, 0, 0), 3 * math.sqrt(3) / 2),
    ],
)
def test_reciprocal_metric_spacing_matches_known_cells(cell_values, hkl, expected):
    assert _d_spacing(cell_values, hkl) == pytest.approx(expected)


def test_phase_filter_preserves_original_fraction_basis_and_percentage_points():
    items = [
        scan(1, phases=[cell(fraction=60), cell("phase-b", fraction=40)]),
        scan(2, phases=[cell(fraction=50), cell("phase-b", fraction=50)]),
    ]
    filtered = view(items, kind="phase_evolution", phases=["phase-a"])
    assert values(filtered) == [60, 50]
    delta = view(items, kind="phase_evolution", phases=["phase-a"], representation="difference")
    assert values(delta) == [0, -10]
    assert values(
        view(items, kind="phase_evolution", phases=["phase-a"], representation="relative")
    ) == pytest.approx([0, -100 / 6])


def test_zero_phase_baseline_allows_difference_but_not_division():
    items = [scan(1, phases=[cell(fraction=0)]), scan(2, phases=[cell(fraction=5)])]
    assert values(view(items, kind="phase_evolution", representation="difference")) == [0, 5]
    assert values(view(items, kind="phase_evolution", representation="relative")) == [None, None]


def test_temperature_unit_conversion_does_not_mix_celsius_and_kelvin():
    first, second = scan(1, temperature=300), scan(2)
    second["metadata"]["temperature"] = {"value": 36.85, "unit": "°C", "source": "sensor"}
    result = view([first, second], axis="temperature", representation="absolute")
    assert [p["x"] for p in result["series"][0]["points"]] == pytest.approx([300, 310])
    second["metadata"]["temperature"]["unit"] = "unknown"
    result = view([first, second], axis="temperature", representation="absolute")
    assert result["series"][0]["points"][1]["value"] is None


def test_measurement_and_nominal_history_are_not_combined_as_one_axis():
    post = settings(axis="temperature")
    post.axis_mappings = {
        "temperature": {
            "NOM_2_BANK1": AxisMapping(
                value=400,
                unit="K",
                source="furnace setpoint",
                kind="preparation_history",
                verified=True,
            )
        }
    }
    post = PostAnalysisSettings.model_validate(post.model_dump())
    result = calculate_views([scan(1), scan(2)], post)["views"][0]
    assert not result["series"] and "mixed axis" in " ".join(result["notices"])


def test_reaction_progress_needs_fixed_endpoints_and_keeps_overshoot_visible():
    assert view([scan(1)], kind="reaction_progress")["status"] == "needs_input"
    result = view(
        [scan(1, phases=[cell(fraction=20)]), scan(2, phases=[cell(fraction=90)])],
        kind="reaction_progress",
        time_zero="2020-01-01T00:00:00+00:00",
        progress_start=20,
        progress_end=80,
    )
    assert values(result) == pytest.approx([0, 70 / 60])
    assert [p["x"] for p in result["series"][0]["points"]] == [1, 2]


def test_external_mapping_requires_verified_values_and_retains_their_source():
    post = settings(kind="operando", axis="voltage", representation="absolute")
    mapping = {
        "value": 4.1,
        "unit": "V",
        "source": "reviewed acquisition-window mean",
        "kind": "measured",
        "verified": False,
    }
    post.axis_mappings = {"voltage": {"NOM_1_BANK1": AxisMapping.model_validate(mapping)}}
    post = PostAnalysisSettings.model_validate(post.model_dump())
    assert calculate_views([scan(1)], post)["views"][0]["status"] == "needs_input"
    post.axis_mappings["voltage"]["NOM_1_BANK1"].verified = True
    point = calculate_views([scan(1)], post)["views"][0]["series"][0]["points"][0]
    assert point["x"] == 4.1 and "window mean" in point["axis_source"]


def test_categorical_comparisons_do_not_invent_a_measured_temperature_axis():
    post = settings(kind="sample_composition")
    post.axis_mappings = {
        "sample_state": {
            "sample-1": AxisMapping(
                value="annealed",
                unit="state",
                source="reviewed preparation sheet",
                kind="preparation_history",
                verified=True,
            )
        }
    }
    result = calculate_views([scan(1)], PostAnalysisSettings.model_validate(post.model_dump()))[
        "views"
    ][0]
    assert result["axis"] == "sample_state" and result["series"][0]["points"][0]["x"] == "annealed"


def test_advanced_recipes_do_not_treat_stored_gpx_defaults_as_measurements():
    for kind in ["broadening", "atomic_structure"]:
        result = view([scan(1)], kind=kind)
        assert result["status"] == "needs_recipe" and not result["series"]


def test_all_ten_recipes_can_be_selected_and_empty_selection_is_valid():
    post = PostAnalysisSettings(views=[ViewSettings(id=k, kind=k) for k in RECIPES])
    assert len(calculate_views([], post)["views"]) == 10
    assert calculate_views([], PostAnalysisSettings(views=[]))["views"] == []


def test_calculation_is_pure_and_export_is_independent_of_ndip_session():
    scans = [scan(1), scan(2, a=5)]
    original = copy.deepcopy(scans)
    post = settings()
    result = calculate_views(scans, post)
    assert scans == original and post.references == {}
    bundle = export_bundle(result, post, scans)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        saved = json.loads(archive.read("post-analysis.json"))
        observations = json.loads(archive.read("observations.json"))
        assert saved["references"] == result["references"]
        assert observations == scans
        assert b"plotly.js" in archive.read("view-01.html")
        assert "raw_value" in archive.read("view-01.csv").decode()


def test_plot_facets_banks_and_keeps_scans_in_hover_records():
    post = settings(representation="absolute")
    result = calculate_views([scan(1), scan(2, bank=2)], post)["views"][0]
    figure = view_figure(result, post.views[0])
    assert len(figure.data) == 2
    assert figure.data[0].customdata[0][0] == "NOM_1_BANK1"
    assert figure.layout.yaxis2 is not None


def test_missing_phase_in_an_intermediate_model_creates_a_gap_not_zero():
    items = [scan(1), scan(2, phases=[]), scan(3, a=5)]
    result = view(items, representation="absolute", connect=True)
    assert values(result) == [4, None, 5]
    figure = view_figure(result, settings(connect=True).views[0])
    assert list(figure.data[0].y) == [4, None, 5] and figure.data[0].connectgaps is False


def test_symmetry_fixed_ratio_is_not_presented_as_an_independent_distortion():
    item = scan(1)
    row = item["observables"]["phases"][0]
    row["c"] = row["a"]
    row["cell_jacobian"] = [[1.0], [0.0], [1.0], [0.0], [0.0], [0.0], [1.0]]
    assert values(view([item], kind="distortion")) == [None]
    exploratory = view([item], kind="distortion", include_fixed=True)
    assert values(exploratory) == [1]
    assert exploratory["series"][0]["points"][0]["state"] == "model-constrained"


def test_hysteresis_keeps_acquisition_order_and_candidate_brackets():
    temperatures = [300, 400, 500, 600, 500, 400, 300]
    fractions = [10, 20, 60, 80, 65, 45, 15]
    items = [
        scan(i + 1, temperature=t, phases=[cell(fraction=f)])
        for i, (t, f) in enumerate(zip(temperatures, fractions))
    ]
    result = view(
        list(reversed(items)), kind="transition_hysteresis", axis="temperature", threshold=50
    )
    points = result["series"][0]["points"]
    assert [p["x"] for p in points] == temperatures
    assert [p["branch"] for p in points] == ["increasing:0"] * 4 + ["decreasing:1"] * 3
    assert [c["interval"] for c in result["crossings"]] == [[400, 500], [500, 400]]


def test_expansion_fit_uses_raw_cells_and_retains_nonzero_covariance_uncertainty():
    temperatures = [300, 350, 400, 450, 500]
    items = [
        scan(i + 1, temperature=t, a=4 * (1 + 1e-5 * (t - 300))) for i, t in enumerate(temperatures)
    ]
    result = view(items, axis="temperature", fit_expansion=True)
    assert len(result["fits"]) == 1
    fit = result["fits"][0]
    assert fit["temperature"] == pytest.approx(400)
    assert fit["alpha"] == pytest.approx(1e-5 / (1 + 1e-5 * 100))
    assert fit["alpha_esd"] > 0
    assert "Weighted least squares" in fit["method"]
    assert len(fit["run_ids"]) == 5


def test_expansion_excludes_within_scan_ramps_and_never_fits_two_temperatures():
    items = [scan(i + 1, temperature=300 + 50 * i, a=4 + i * 0.001) for i in range(3)]
    items[1]["metadata"]["temperature"].update(minimum=340, maximum=360)
    result = view(items, axis="temperature", fit_expansion=True)
    assert not result["fits"]
    assert "three distinct qualified" in " ".join(result["notices"])
    assert len(result["series"][0]["points"]) == 3  # Ramp observations remain inspectable.


def test_expansion_never_uses_a_preparation_temperature_as_measured_thermometry():
    items = [scan(i + 1, temperature=300 + i * 100) for i in range(4)]
    for item in items:
        item["metadata"]["temperature"]["kind"] = "preparation_history"
    assert not view(items, axis="temperature", fit_expansion=True)["fits"]


def test_reaction_time_zero_does_not_change_the_elapsed_axis_of_other_views():
    post = PostAnalysisSettings(
        axis="elapsed_time",
        views=[
            ViewSettings(
                id="reaction",
                kind="reaction_progress",
                time_zero="2019-12-31T23:00:00Z",
                progress_start=0,
                progress_end=100,
            ),
            ViewSettings(id="other", kind="thermal_expansion", representation="absolute"),
        ],
    )
    first, second = calculate_views([scan(1), scan(2)], post)["views"]
    assert first["series"][0]["points"][0]["x"] == 61
    assert second["series"][0]["points"][0]["x"] == 0


def test_active_revision_conflicts_are_withheld_while_identical_retries_are_deduplicated():
    item = scan(1)
    retry = copy.deepcopy(item)
    assert len(values(view([item, retry]))) == 1
    retry["observables"]["source_gpx_sha256"] = "different-output"
    result = view([item, retry], representation="absolute")
    assert values(result) == [None]
    assert "conflicting active revisions" in " ".join(result["notices"])
    item["active_revision"] = False
    assert values(view([item, retry], representation="absolute")) == [4]


def test_unknown_observation_versions_never_fall_back_to_legacy_cell_values():
    item = scan(1)
    item["observables"]["$schema"] = "radar-pd-observables/v99"
    item["lattice_parameters"] = [cell()]
    assert not view([item], representation="absolute")["series"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_invalid_cell_values_never_enter_plots_or_normalization(value):
    assert values(view([scan(1, a=value)], representation="absolute")) == [None]


def test_explicit_plot_ranges_survive_export_and_replay():
    post = settings(axis="temperature", x_range=(300, 350), y_range=(-0.1, 1.0))
    result = calculate_views([scan(1), scan(2)], post)["views"][0]
    figure = view_figure(result, post.views[0])
    assert tuple(figure.layout.xaxis.range) == (300, 350)
    assert tuple(figure.layout.yaxis.range) == (-0.1, 1.0)


def test_recipe_order_does_not_affect_derived_values():
    scans = [scan(1), scan(2, a=4.1)]
    definitions = [
        ViewSettings(id="relative", kind="thermal_expansion"),
        ViewSettings(id="shape", kind="distortion"),
    ]
    forward = calculate_views(scans, PostAnalysisSettings(views=definitions))
    reverse = calculate_views(scans, PostAnalysisSettings(views=list(reversed(definitions))))
    assert {v["id"]: values(v) for v in forward["views"]} == {
        v["id"]: values(v) for v in reverse["views"]
    }


def test_phase_colors_are_distinct_and_stable_when_a_new_phase_arrives():
    first = calculate_views([scan(1, phases=[cell(), cell("phase-b")])], settings())
    colors = first["phase_colors"]
    assert colors["phase-a"] != colors["phase-b"]
    post = settings()
    post.phase_colors = colors
    later = calculate_views(
        [scan(0, phases=[cell("earlier-new-phase")]), scan(1, phases=[cell(), cell("phase-b")])],
        post,
    )
    assert all(later["phase_colors"][key] == value for key, value in colors.items())


def test_published_configuration_schemas_validate_defaults_and_reject_unknown_versions():
    from jsonschema import Draft202012Validator

    root = Path(__file__).resolve().parents[2] / "schemas/ndip"
    for name in [
        "radar-pd-post-analysis-v1.schema.json",
        "radar-pd-config-v1.schema.json",
        "radar-pd-observables-v1.schema.json",
    ]:
        Draft202012Validator.check_schema(json.loads((root / name).read_text()))
    post_schema = json.loads((root / "radar-pd-post-analysis-v1.schema.json").read_text())
    validator = Draft202012Validator(post_schema)
    validator.validate(PostAnalysisSettings().model_dump(mode="json"))
    assert list(validator.iter_errors({"schema_version": 999}))
    config = AnalysisConfig(sample_elements=["Fe"]).portable_contract()
    Draft202012Validator(
        json.loads((root / "radar-pd-config-v1.schema.json").read_text())
    ).validate(config)


def test_reference_key_names_cannot_overlap_between_view_instances():
    with pytest.raises(ValueError):
        ViewSettings(id="view:another", kind="thermal_expansion")


def test_expansion_coefficients_are_withheld_for_unresolved_covariance_disagreements():
    items = [scan(i + 1, temperature=300 + i * 50, a=4 + i * 0.001) for i in range(4)]
    for item in items:
        item["observables"]["phases"][0]["covariance_status"] = "gsas_esd_mismatch"
    result = view(items, axis="temperature", fit_expansion=True)
    assert not result["fits"]
    assert "coefficient is withheld" in " ".join(result["notices"])


@pytest.mark.parametrize(
    "options",
    [
        {"kind": "phase_evolution", "representation": "microstrain"},
        {"kind": "reaction_progress", "representation": "absolute"},
        {"kind": "thermal_expansion", "metric": "weight_percent"},
    ],
)
def test_recipe_and_representation_cannot_claim_a_different_physical_quantity(options):
    with pytest.raises(ValueError):
        ViewSettings(id="v", **options)


def test_reviewed_cycle_mapping_keeps_overlapping_operando_paths_separate():
    post = settings(kind="operando", axis="voltage", representation="absolute")
    post.axis_mappings = {
        "voltage": {
            f"NOM_{i}_BANK1": AxisMapping(
                value=3.8,
                unit="V",
                source="aligned window",
                kind="measured",
                verified=True,
                branch="charge",
                cycle=str(i),
            )
            for i in [1, 2]
        }
    }
    result = calculate_views([scan(1), scan(2)], post)["views"][0]
    assert [p["branch"] for p in result["series"][0]["points"]] == [
        "cycle 1 · charge",
        "cycle 2 · charge",
    ]


def test_a_cycle_reset_is_not_inferred_as_a_physical_reverse_branch():
    post = settings(kind="operando", axis="voltage", representation="absolute")
    post.axis_mappings = {
        "voltage": {
            f"NOM_{i}_BANK1": AxisMapping(
                value=value,
                unit="V",
                source="aligned window",
                kind="measured",
                verified=True,
                cycle=str(cycle),
            )
            for i, value, cycle in [(1, 3.1, 1), (2, 3.2, 1), (3, 3.1, 2), (4, 3.2, 2)]
        }
    }
    result = calculate_views([scan(i) for i in range(1, 5)], post)["views"][0]
    branches = [p["branch"] for p in result["series"][0]["points"]]
    assert branches == ["cycle 1 · increasing:0"] * 2 + ["cycle 2 · increasing:1"] * 2
