import numpy as np
from radar_pd_nova.transition_tracking_plot import (
    heatmap_figure,
    locked_limits,
    transform,
    boundary_figure,
)
from test_pattern_transition import observed


def test_boundary_annotations_are_separate_from_intensity_and_selection():
    a = observed()
    options = dict(patterns=[a,a], labels=["15 K","50 K"],run_ids=["a","b"],limits=locked_limits(a))
    plain,_ = heatmap_figure(**options)
    marked,_ = heatmap_figure(**options,boundaries=[{"run_id":"b","boundary":True,"state_id":"new","style_index":1}])
    assert marked.data[0].z == plain.data[0].z
    assert marked.layout.shapes[0].type == "line" and marked.layout.shapes[0].y0 == .5
    assert any(s.line.color=="white" and s.line.dash=="dash" for s in marked.layout.shapes)
    selected,_=heatmap_figure(**options,selected="b")
    assert not any(s.type=="rect" for s in selected.layout.shapes)
    assert selected.layout.yaxis.ticktext[-1]==""
    assert any(a.text=="<b>50 K</b>" and a.bgcolor=="#15543c" for a in selected.layout.annotations)
    assert any(a.showarrow and a.ax<0 and a.y=="b" for a in selected.layout.annotations)
    decision={"votes":2,"required":2,"selected":["width","appearance","splitting"],"status":"transition_evidence"}
    fig = boundary_figure([{"run_id":"a","style_index":0},{"run_id":"b","style_index":1,"boundary":True,"boundary_comparison":decision}], [15,50], "Temperature (K)")
    assert fig.data[1].y==(2,)
    assert fig.layout.shapes[0].y0==fig.layout.shapes[0].y1==2
    assert all(s.type!="rect" for s in fig.layout.shapes)
    assert fig.layout.yaxis.title.text=="Indicators triggered"


def test_ribbon_shares_heatmap_zoom_and_uses_the_same_background_and_scale():
    a=observed();baseline=np.full(len(a.x),15.)
    limits=locked_limits(a,background=baseline)
    fig,_=heatmap_figure([a],labels=["scan"],run_ids=["a"],limits=limits,ribbon=a,
        ribbon_background=baseline,backgrounds=[baseline],region=[2000,3500],scale="log")
    assert fig.layout.xaxis.matches=="x2"
    assert fig.layout.yaxis2.fixedrange
    assert fig.layout.xaxis.range==fig.layout.xaxis2.range==(2000,3500)
    np.testing.assert_allclose(fig.data[1].customdata,a.intensity-baseline)
    expected=transform(a.intensity-baseline,limits,scale="log")
    valid=a.valid & np.isfinite(expected)
    np.testing.assert_allclose(np.asarray(fig.data[1].y,dtype=float)[valid],expected[valid])
    assert fig.layout.meta["palette"]=="Turbo"


def test_missing_local_baseline_does_not_poison_color_limits_or_neighboring_peak_bins():
    from radar_pd_nova.transition_tracking_plot import binned_intensity, ribbon_trace
    p = observed()
    baseline = np.full(len(p.x), 15.0)
    baseline[100:105] = np.nan
    limits = locked_limits(p, background=baseline)
    assert all(np.isfinite(limits[k]) for k in ("low", "high", "soft"))
    boundaries = np.array([p.x[99], p.x[106], p.x[110]])
    binned = binned_intensity(p, boundaries, background=baseline)
    expected = p.intensity[99:106] - baseline[99:106]
    assert binned[0] == np.max(expected[np.isfinite(expected)])
    trace = ribbon_trace(p, baseline, limits, "asinh", 50, "test")
    assert trace["y"][100:105] == [None] * 5
    assert trace["customdata"][100:105] == [None] * 5


def test_adding_rows_keeps_the_same_native_display_bins_colors_and_color_limits():
    a, b = observed(ident="a"), observed(gain=8, ident="b")
    limits = locked_limits(a)
    first, _ = heatmap_figure([a], labels=["a"], run_ids=["a"], limits=limits)
    after, stats = heatmap_figure(
        [a, b], labels=["a", "b"], run_ids=["a", "b"], limits=limits
    )
    assert first.data[0].x == after.data[0].x
    assert first.data[0].z[0] == after.data[0].z[0]
    assert (
        first.data[0].zmin == after.data[0].zmin
        and first.data[0].zmax == after.data[0].zmax
    )
    assert stats["saturated_cells"] > 0


def test_zoom_uses_native_samples_for_narrow_regions_instead_of_full_range_bins():
    a=observed()
    opts=dict(patterns=[a],labels=["a"],run_ids=["a"],limits=locked_limits(a),scale="linear")
    full,_=heatmap_figure(**opts)
    zoom,stats=heatmap_figure(**opts,region=[3950,4050])
    native=(a.x>=3950)&(a.x<=4050)
    assert len([x for x in full.data[0].x if 3950<=x<=4050])<native.sum()
    assert native.sum()+2<len(zoom.data[0].x)<=1025
    pairs=[(x,y) for x,y in zip(zoom.data[0].x,zoom.data[0].customdata[0]) if 3950<=x<=4050]
    np.testing.assert_allclose([x for x,y in pairs],a.x[native])
    np.testing.assert_allclose([y for x,y in pairs],a.intensity[native])
    assert not stats["reference_range_exceeded"]
    # Widening only the browser axis must still find real data on both sides
    # of the former zoom window, even before any server refresh completes.
    assert zoom.data[0].x[0]==full.data[0].x[0]
    assert zoom.data[0].x[-1]==full.data[0].x[-1]
    assert any(y is not None for x,y in zip(zoom.data[0].x,zoom.data[0].z[0]) if x<2000)
    assert any(y is not None for x,y in zip(zoom.data[0].x,zoom.data[0].z[0]) if x>8000)


def test_asinh_retains_negative_values_and_log_masks_them_explicitly():
    a = observed(offset=-10)
    limits = locked_limits(a)
    assert transform([-1, 0, 1], limits).tolist()[0] < 0
    assert np.isnan(transform([-1, 0, 1], limits, scale="log")[:2]).all()
    fig, stats = heatmap_figure(
        [a], labels=["a"], run_ids=["a"], limits=limits, scale="log"
    )
    assert stats["log_masked_cells"] > 0
    assert any(v is None for v in fig.data[0].z[0])


def test_masked_regions_remain_gaps_and_narrow_weak_peaks_survive_display_binning():
    a = observed(new=[(2700, 8, 0.0005)])
    mask = a.valid.copy()
    mask[(a.x > 3500) & (a.x < 4500)] = False
    from dataclasses import replace

    a = replace(a, valid=mask)
    fig, _ = heatmap_figure([a], labels=["a"], run_ids=["a"], limits=locked_limits(a))
    assert any(
        v is None for x, v in zip(fig.data[0].x, fig.data[0].z[0]) if 3600 < x < 4400
    )
    # Stored hover intensities retain a maximum, including a weak narrow line.
    assert (
        max(
            v
            for x, v in zip(fig.data[0].x, fig.data[0].customdata[0])
            if 2670 < x < 2730
        )
        > 22
    )


def test_display_cell_count_is_bounded_for_a_long_experiment():
    a = observed()
    fig, stats = heatmap_figure(
        [a] * 100,
        labels=["run"] * 100,
        run_ids=[str(i) for i in range(100)],
        limits=locked_limits(a),
    )
    assert stats["display_cells"] <= 51200
    assert len(fig.layout.yaxis.tickvals) < 40


def test_expanded_observed_overlay_retains_both_scan_and_temperature_labels():
    from radar_pd_nova.transition_tracking_plot import profile_figure
    a=observed()
    figure=profile_figure(a,current_label="Scan 8 · 3.107 K",comparison=a,comparison_label="Scan 5 · 0.2822 K")
    assert [trace.name for trace in figure.data]==["Scan 8 · 3.107 K","Scan 5 · 0.2822 K"]
    np.testing.assert_allclose(figure.data[0].y,a.intensity)
