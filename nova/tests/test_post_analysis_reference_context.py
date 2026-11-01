"""Exported relative plots identify their calculated baseline without changing it."""
import copy
import io
import zipfile

import pytest

from radar_pd_nova.post_analysis import calculate_views
from radar_pd_nova.post_analysis_plot import export_bundle, view_figure
from test_post_analysis import cell, scan, settings


def tetragonal_fixture():
    rows = []
    for run, temperature, scale in [(501, 15, None), (502, 50, None),
                                     (503, 75, 1), (504, 200, 1.0006),
                                     (505, 300, 1.0008)]:
        phase = cell("Tetragonal test phase")
        phase.update(a=phase["a"] * (scale or 1), c=phase["c"] / (scale or 1))
        row = scan(run, sample="specimen-a", bank=2, temperature=temperature,
                   phases=[phase] if scale is not None else [])
        row.update(run_id=f"PG3_{run}", instrument="PG3", ipts="IPTS-99999")
        rows.append(row)
    return rows


def captions(figure):
    return [a.text for a in figure.layout.annotations if a.text.startswith("Reference")]


@pytest.mark.parametrize("representation", ["relative", "permille", "normalized", "difference"])
def test_first_eligible_reference_is_visible_in_multisignal_plot_and_export(representation):
    rows = tetragonal_fixture()
    key = rows[2]["observables"]["phases"][0]["phase_key"]
    post = settings(representation=representation, axis="temperature", lattice_series=[
        {"phase_key": key, "metric": "a"}, {"phase_key": key, "metric": "c"}])
    result = calculate_views(rows, post)
    panel = result["views"][0]
    before = copy.deepcopy(panel)
    figure = view_figure(panel, post.views[0])
    assert captions(figure) == ["Reference: PG3_503 · 75 K"]
    assert all(" · R" not in trace.name for trace in figure.data)
    assert all(s["reference_run"] == "PG3_503" for s in panel["series"])
    assert panel == before
    for trace, series in zip(figure.data, panel["series"]):
        assert list(trace.y) == [p["value"] for p in series["points"]]
    with zipfile.ZipFile(io.BytesIO(export_bundle(result, post, rows))) as archive:
        html = archive.read("view-01.html").decode()
        assert "Reference: PG3_503" in html and "75 K" in html
        assert "reference_run" in archive.read("view-01.csv").decode()


def test_explicit_reference_caption_does_not_infer_first_available_scan():
    post = settings(representation="normalized", axis="temperature", reference_run="PG3_504")
    panel = calculate_views(tetragonal_fixture(), post)["views"][0]
    figure = view_figure(panel, post.views[0])
    assert captions(figure) == ["Reference: PG3_504 · 200 K"]


def test_different_phase_references_are_keyed_to_each_legend_without_overlapping_plot():
    a, b = cell("phase-a"), cell("phase-b", a=5)
    rows = [scan(1, temperature=75, phases=[a]),
            scan(2, temperature=200, phases=[a, b]),
            scan(3, temperature=300, phases=[a, b])]
    post = settings(representation="permille", axis="temperature", lattice_series=[
        {"phase_key": "phase-a", "metric": "a"},
        {"phase_key": "phase-a", "metric": "c"},
        {"phase_key": "phase-b", "metric": "a"}])
    panel = calculate_views(rows, post)["views"][0]
    before = copy.deepcopy(panel)
    figure = view_figure(panel, post.views[0])
    text = captions(figure)[0]
    assert "R1: NOM_1_BANK1 · 75 K" in text
    assert "R2: NOM_2_BANK1 · 200 K" in text
    assert [t.name.rsplit(" · ", 1)[-1] for t in figure.data] == ["R1", "R1", "R2"]
    assert len({t.legendgroup for t in figure.data}) == 3
    assert figure.layout.margin.t > 45
    assert figure.layout.height - figure.layout.margin.t == 480 - 45
    assert panel == before


def test_same_phase_in_different_banks_has_unambiguous_reference_keys():
    post = settings(representation="normalized", axis="temperature")
    panel = calculate_views([scan(1, bank=1, temperature=75),
                             scan(1, bank=2, temperature=80)], post)["views"][0]
    figure = view_figure(panel, post.views[0])
    text = captions(figure)[0]
    assert "NOM_1_BANK1 · 75 K" in text and "NOM_1_BANK2 · 80 K" in text
    assert {t.name.rsplit(" · ", 1)[-1] for t in figure.data} == {"R1", "R2"}
    assert all(t.showlegend for t in figure.data)


def test_reference_metadata_survives_hidden_baseline_and_celsius_display():
    post = settings(representation="relative", axis="temperature", temperature_display_unit="C")
    panel = calculate_views([scan(1, temperature=300), scan(2, temperature=310)], post)["views"][0]
    panel["series"][0]["points"][0]["display_hidden"] = True
    figure = view_figure(panel, post.views[0])
    assert captions(figure) == ["Reference: NOM_1_BANK1 · 26.85 °C"]
    assert list(figure.data[0].x) == pytest.approx([36.85])


def test_missing_reference_condition_is_not_invented_from_another_scan():
    post = settings(representation="normalized")
    panel = calculate_views([scan(1), scan(2)], post)["views"][0]
    panel["series"][0]["points"][0]["temperature"] = None
    figure = view_figure(panel, post.views[0])
    assert captions(figure) == ["Reference: NOM_1_BANK1"]


def test_absolute_plot_has_no_added_caption_or_layout_changes():
    post = settings(representation="absolute", axis="temperature")
    panel = calculate_views(tetragonal_fixture(), post)["views"][0]
    figure = view_figure(panel, post.views[0])
    assert captions(figure) == []
    assert [a.text for a in figure.layout.annotations] == ["Source bank 2"]
    assert figure.layout.margin.t == 45 and figure.layout.height == 480


def test_reference_identifiers_are_escaped_in_annotations():
    post = settings(representation="relative")
    panel = calculate_views([scan(1), scan(2)], post)["views"][0]
    series = panel["series"][0]
    series["reference_run"] = series["points"][0]["run_id"] = "<b>scan</b>"
    assert "&lt;b&gt;scan&lt;/b&gt;" in captions(view_figure(panel, post.views[0]))[0]
