"""Lattice thermometry, normalization and uncertainty regression tests."""
import copy
import csv
import io
import json
import math
import zipfile

import numpy as np
import pytest

from radar_pd_nova.post_analysis import AxisMapping, PostAnalysisSettings, ViewSettings, calculate_views
from radar_pd_nova.post_analysis_plot import export_bundle, view_figure
from radar_pd_nova.thermal_reference import ReferenceCurve, infer_temperature, load_reference_csv
from test_post_analysis import cell, scan, settings, values, view


def curve(*, slope=1e-4, unit="angstrom", values=None, temperatures=None):
    temperatures = temperatures or [200., 300., 400., 500., 600.]
    return ReferenceCurve(id="test-curve", label="Test reference", source="Analytic validation fixture",
        unit=unit, temperatures=temperatures,
        values=values or [4*math.exp(slope*(t-300)) for t in temperatures], source_sha256="a"*64)


def reference_settings(output="overlay", **kwargs):
    reference = kwargs.pop("curve", curve())
    options = dict(id="thermal", kind="thermal_expansion", metric="a", phases=["phase-a"],
        representation="permille", axis="temperature", reference_run="NOM_1_BANK1",
        reference_curve_id=reference.id, reference_curve_reviewed=True, reference_output=output)
    options.update(kwargs)
    return PostAnalysisSettings(views=[ViewSettings(**options)], reference_curves={reference.id: reference})


def points(result):
    return [p for s in result["series"] for p in s["points"]]


@pytest.mark.parametrize("rep,factor,offset", [("permille", 1000., -1), ("normalized", 1., 0)])
def test_baseline_normalization_propagates_both_scan_uncertainties(rep, factor, offset):
    result = view([scan(1), scan(2, a=4.04)], representation=rep)
    baseline, changed = points(result)
    assert baseline["value"] == (1 if rep == "normalized" else 0)
    assert baseline["variance"] == 0  # normalization identity, not a perfect measurement
    assert changed["value"] == pytest.approx(factor*(4.04/4+offset))
    assert changed["variance"] == pytest.approx(factor**2*(1e-6/4**2+4.04**2*1e-6/4**4))


def test_explicit_multiphase_multiaxis_lattice_signals_and_csv_keep_their_identity():
    a, b = cell("corundum"), cell("oxide", a=5.)
    a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
    a2.update(a=4.004, c=6.012);b2["a"] = 5.015
    post = settings(representation="permille", lattice_series=[
        {"phase_key": "corundum", "metric": "a"},
        {"phase_key": "corundum", "metric": "c"},
        {"phase_key": "oxide", "metric": "a"},
    ])
    scans = [scan(1, phases=[a,b]), scan(2, phases=[a2,b2])]
    result = calculate_views(scans, post)
    panel = result["views"][0]
    assert [s["metric"] for s in panel["series"]] == ["a", "c", "a"]
    assert [s["points"][1]["value"] for s in panel["series"]] == pytest.approx([1.,2.,3.])
    assert len({s["key"] for s in panel["series"]}) == 3
    fig = view_figure(panel, post.views[0])
    assert len({s.legendgroup for s in fig.data}) == 3
    assert {s.marker.symbol for s in fig.data} == {"circle", "triangle-up"}
    with zipfile.ZipFile(io.BytesIO(export_bundle(result,post,scans))) as archive:
        rows = list(csv.DictReader(io.StringIO(archive.read("view-01.csv").decode())))
        assert [row["metric"] for row in rows] == ["a","a","c","c","a","a"]
        saved = json.loads(archive.read("post-analysis.json"))
        assert saved["settings"]["views"][0]["lattice_series"] == post.views[0].model_dump()["lattice_series"]


def paired_scan(run=1, *, cross=True):
    a, b = cell("one", a=4.), cell("two", a=5.)
    b["cell_covariance"][0][0] = 4e-6
    if cross:
        c = [[0.]*7 for _ in range(7)];c[0][0] = 1e-6
        a["cross_phase_cell_covariance"] = {"two":c}
        b["cross_phase_cell_covariance"] = {"one":c}
    return scan(run, phases=[a,b])


def ratio_options(**kwargs):
    return dict(metric="phase_ratio", representation="absolute", ratio_numerator={"phase_key":"one", "metric":"a"},
        ratio_denominator={"phase_key":"two", "metric":"a"}, **kwargs)


def test_two_phase_ratio_uses_joint_covariance_instead_of_independent_error_assumption():
    result = view([paired_scan()], **ratio_options())
    p = points(result)[0]
    expected = 1e-6/25+16*4e-6/625-2*4*1e-6/125
    assert p["value"] == .8
    assert p["variance"] == pytest.approx(expected)
    rng = np.random.default_rng(1729)
    sampled = rng.multivariate_normal([4.,5.], [[1e-6,1e-6],[1e-6,4e-6]], size=120000)
    assert np.var(sampled[:,0]/sampled[:,1]) == pytest.approx(expected, rel=.015)


@pytest.mark.parametrize("problem", ["missing", "impossible", "stale", "absent_phase", "duplicate_phase"])
def test_ratio_missing_or_invalid_evidence_does_not_invent_error_bars(problem):
    source = paired_scan(cross=problem!="missing")
    rows = source["observables"]["phases"]
    if problem == "impossible":rows[0]["cross_phase_cell_covariance"]["two"][0][0] = 1.
    if problem == "stale":rows[1]["covariance_fresh"] = False
    if problem == "absent_phase":rows.pop()
    if problem == "duplicate_phase":rows.append(copy.deepcopy(rows[0]))
    p = points(view([source], **ratio_options()))[0]
    assert p["variance"] is None
    if problem in {"stale","absent_phase","duplicate_phase"}:assert p["value"] is None
    else:assert p["value"] == .8


def test_ratio_can_use_two_directions_of_same_cell_within_phase_covariance():
    result = view([scan(1)], metric="phase_ratio", representation="absolute",
        ratio_numerator={"phase_key":"phase-a","metric":"c"},
        ratio_denominator={"phase_key":"phase-a","metric":"a"})
    p = points(result)[0]
    assert p["value"] == 1.5
    assert p["variance"] == pytest.approx(1e-6/16+36e-6/256-2*6*4e-7/64)


def test_symmetry_fixed_ratio_is_not_an_independent_zero_error_measurement():
    source=scan(1)
    row=source['observables']['phases'][0]
    row.update(a=4.,b=4.,c=4.,cell_jacobian=[[1.],[1.],[1.],[0.],[0.],[0.],[48.]])
    result=view([source],metric='phase_ratio',representation='absolute',
        ratio_numerator={'phase_key':'phase-a','metric':'c'},
        ratio_denominator={'phase_key':'phase-a','metric':'a'})
    assert points(result)[0]['value'] is None
    assert points(result)[0]['state']=='symmetry-constrained'


def test_exploratory_fixed_cells_do_not_become_lattice_thermometers():
    scans=[scan(1,temperature=300),scan(2,temperature=350)]
    for source in scans:
        source['observables']['phases'][0]['metric_states']['a']='fixed'
    result=calculate_views(scans,reference_settings('temperature',include_fixed=True))['views'][0]
    assert result['status']=='needs_input'
    assert all(p['value'] is None for p in points(result))


@pytest.mark.parametrize("output", ["overlay", "residual", "temperature", "temperature_offset"])
def test_known_expansion_and_offset_propagate_baseline_and_temperature_uncertainty(output):
    t, logged, alpha = 350., 345., 1e-4
    current = 4*math.exp(alpha*(t-300))
    scans = [scan(1,temperature=300),scan(2,temperature=logged,a=current)]
    scans[1]["metadata"]["temperature"]["standard_uncertainty"] = 1.
    post = reference_settings(output,baseline_temperature_esd=2.)
    result = calculate_views(scans,post)["views"][0]
    base,p = points(result)
    if output in {"temperature","temperature_offset"}:
        assert p["value"] == pytest.approx(t if output=="temperature" else 5.)
        expected = (1e-6/current**2+1e-6/16)/alpha**2+4
        if output=="temperature_offset":expected+=1
        assert p["variance"] == pytest.approx(expected)
        assert base["variance"] == (4 if output=="temperature" else 0)
    elif output == "overlay":
        assert p["value"] == pytest.approx(1000*(math.exp(alpha*50)-1))
        assert p["reference_value"] == pytest.approx(1000*(math.exp(alpha*45)-1))
        assert len(view_figure(result,post.views[0]).data) == 2
    else:
        assert p["value"] == pytest.approx(1000*(math.exp(alpha*50)-math.exp(alpha*45)))
        expected = 1e6*(1e-6/16+current**2*1e-6/256)+1e6*math.exp(alpha*45)**2*alpha**2*5
        assert p["variance"] == pytest.approx(expected)


def test_inverse_uncertainty_matches_finite_difference_and_joint_sampling():
    alpha=1e-4;reference=curve(slope=alpha)
    current=4*math.exp(alpha*50)
    rng=np.random.default_rng(812)
    a=rng.normal(current,.001,120000);b=rng.normal(4,.001,120000);t0=rng.normal(300,2,120000)
    inferred=t0+np.log(a/b)/alpha
    post=reference_settings("temperature",baseline_temperature_esd=2)
    result=calculate_views([scan(1,temperature=300),scan(2,temperature=350,a=current)],post)["views"][0]
    assert np.var(inferred)==pytest.approx(points(result)[1]["variance"],rel=.015)
    derivative=(infer_temperature(reference,current/4+1e-6,300)[0]-infer_temperature(reference,current/4-1e-6,300)[0])/2e-6
    assert derivative==pytest.approx(1/(alpha*current/4),rel=1e-6)


@pytest.mark.parametrize("slope", [1e-4,-1e-4])
def test_inverse_supports_expansion_or_contraction_without_extrapolation(slope):
    reference=curve(slope=slope)
    t,derivative=infer_temperature(reference,math.exp(slope*75),300)
    assert t==pytest.approx(375);assert derivative==pytest.approx(slope)
    with pytest.raises(ValueError,match="unique"):infer_temperature(reference,math.exp(slope*400),300)


def test_nonmonotonic_thermometer_is_ambiguous_and_baseline_is_only_an_anchor():
    reference=curve(temperatures=[200,300,400,500,600],values=[4.1,4.,4.1,4.2,4.3])
    with pytest.raises(ValueError,match="unique"):infer_temperature(reference,4.05/4,300)
    post=reference_settings("temperature",curve=reference,baseline_temperature_esd=2)
    result=calculate_views([scan(1,temperature=300),scan(2,temperature=350,a=4.05)],post)["views"][0]
    assert points(result)[0]["value"]==300
    assert points(result)[1]["value"] is None
    flat=curve(values=[4]*5)
    with pytest.raises(ValueError):infer_temperature(flat,1.,300)


@pytest.mark.parametrize("problem", ["not_reviewed", "no_baseline", "no_phase", "wrong_unit", "outside_range"])
def test_reference_comparison_requires_explicit_qualified_inputs(problem):
    post=reference_settings();scans=[scan(1,temperature=300),scan(2,temperature=350)]
    if problem=="not_reviewed":post.views[0].reference_curve_reviewed=False
    if problem=="no_baseline":post.views[0].reference_run=None
    if problem=="no_phase":post.views[0].phases=[]
    if problem=="wrong_unit":post.reference_curves["test-curve"].unit="dimensionless"
    if problem=="outside_range":scans[0]["metadata"]["temperature"]["value"]=100
    result=calculate_views(scans,post)["views"][0]
    assert result["status"]=="needs_input"
    assert all(p["value"] is None for p in points(result))


def test_reference_comparison_updates_from_prefix_without_changing_baseline():
    post=reference_settings("temperature_offset")
    scans=[scan(i,temperature=300+25*(i-1),a=4*math.exp(1e-4*25*(i-1))) for i in range(1,5)]
    previous=[]
    for n in range(1,5):
        result=calculate_views(scans[:n],post)
        current=points(result["views"][0])
        assert [(p["run_id"],p["value"],p["variance"]) for p in current[:len(previous)]]==previous
        previous=[(p["run_id"],p["value"],p["variance"]) for p in current]
        post=PostAnalysisSettings.model_validate(result["settings"])
    assert all(p["reference_run"]=="NOM_1_BANK1" for p in current)


def test_temperature_range_is_separate_from_sigma_and_display_conversion():
    source=scan(1,temperature=300)
    source["metadata"]["temperature"].update(minimum=299,maximum=302,standard_uncertainty=.2)
    post=settings(axis="temperature",temperature_display_unit="C",show_temperature_span=True)
    result=calculate_views([source],post)["views"][0]
    fig=view_figure(result,post.views[0])
    assert fig.data[0].x[0]==pytest.approx(26.85)
    assert fig.data[0].error_x.array[0]==2
    assert fig.data[0].error_x.arrayminus[0]==1
    assert "°C" in fig.layout.xaxis.title.text
    post.axis_mappings={"temperature":{source["run_id"]:AxisMapping(value=40,unit="C",source="reviewed",kind="measured",verified=True)}}
    result=calculate_views([source],post)["views"][0]
    assert points(result)[0]["x"]==313.15
    assert points(result)[0]["axis_minimum"] is None  # do not reuse a different sensor/unit's range


@pytest.mark.parametrize("bad", ["duplicate", "unsorted", "nan", "mixed_units", "missing", "empty_source"])
def test_reference_csv_validation(tmp_path,bad):
    rows=["temperature_C,value,unit,source", "0,4,angstrom,reviewed", "100,4.1,angstrom,reviewed", "200,4.2,angstrom,reviewed"]
    if bad=="duplicate":rows[2]="0,4.1,angstrom,reviewed"
    if bad=="unsorted":rows[1],rows[2]=rows[2],rows[1]
    if bad=="nan":rows[2]="100,nan,angstrom,reviewed"
    if bad=="mixed_units":rows[2]="100,4.1,dimensionless,reviewed"
    if bad=="missing":rows[2]="100,4.1,angstrom"
    if bad=="empty_source":rows=[r.replace("reviewed","") for r in rows]
    path=tmp_path/"reference.csv";path.write_text("\n".join(rows))
    with pytest.raises(ValueError):load_reference_csv(path)


def test_reference_csv_roundtrip_and_export_preserve_provenance_and_error_precision(tmp_path):
    path=tmp_path/"reference.csv"
    path.write_text("temperature_C,value,unit,source\n0,4,angstrom,Reviewed source\n100,4.01,angstrom,Reviewed source\n200,4.02,angstrom,Reviewed source\n")
    reference=load_reference_csv(path)
    assert reference.temperatures==[273.15,373.15,473.15]
    assert load_reference_csv(path).id==reference.id
    post=reference_settings(curve=reference)
    scans=[scan(1,temperature=300),scan(2,temperature=350,a=4.001)]
    result=calculate_views(scans,post)
    with zipfile.ZipFile(io.BytesIO(export_bundle(result,post,scans))) as archive:
        payload=json.loads(archive.read("post-analysis.json"))
        restored=PostAnalysisSettings.model_validate(payload["settings"])
        assert restored.reference_curves[reference.id]==reference
        rows=list(csv.DictReader(io.StringIO(archive.read("view-01.csv").decode())))
        assert float(rows[1]["variance"])==points(result["views"][0])[1]["variance"]
        assert rows[1]["reference_curve_id"]==reference.id


def test_lattice_series_dimension_and_duplicate_validation():
    with pytest.raises(ValueError,match="Lengths and volumes"):
        ViewSettings(id="test",kind="thermal_expansion",representation="absolute",lattice_series=[
            {"phase_key":"one","metric":"a"},{"phase_key":"two","metric":"volume"}])
    with pytest.raises(ValueError,match="each lattice signal once"):
        ViewSettings(id="test",kind="thermal_expansion",lattice_series=[{"phase_key":"one","metric":"a"}]*2)
    with pytest.raises(ValueError,match="matching dimensions"):
        ViewSettings(id="test",kind="thermal_expansion",**{**ratio_options(),"ratio_denominator":{"phase_key":"two","metric":"volume"}})
