from dataclasses import replace

import numpy as np

from radar_pd_nova.diffraction_coordinates import TofCalibration
from radar_pd_nova.transition_regions import localized_regions
from radar_pd_nova.transition_tracking_plot import heatmap_figure,locked_limits
from test_pattern_transition import observed


def evidence():
    return {"indicators":{
        "width":{"localizable":True,"detected":True,"regions":[{"interval":[2000,3000],"error_ratio":4}]},
        "shape":{"localizable":True,"detected":True,"regions":[{"interval":[2900,3500]}]},
        "rwp":{"localizable":False,"detected":True,"regions":[{"interval":[1000,10000]}]}}}


def test_localization_merges_overlap_excludes_masked_gaps_and_never_assigns_rwp_a_box():
    p=observed();mask=p.valid.copy();mask[(p.x>2600)&(p.x<2700)]=False
    p=replace(p,valid=mask)
    regions,stats=localized_regions(evidence(),["width","shape","rwp"],p,"tof_us")
    assert stats=={"total":2,"shown":2}
    assert regions[0]["interval"][1]<2600 and regions[1]["interval"][0]>2700
    assert all(r["signals"]==["shape","width"] for r in regions)
    assert localized_regions(evidence(),["rwp"],p,"tof_us")[0]==[]


def test_overlay_coordinates_follow_calibrated_q_and_the_number_of_boxes_is_bounded():
    p=observed();c=TofCalibration(22000,-4,1,-7,source_bank=2)
    native,_=localized_regions(evidence(),["width"],p,"tof_us")
    q,_=localized_regions(evidence(),["width"],p,"q_inv_angstrom",c)
    np.testing.assert_allclose(q[0]["interval"],sorted(2*np.pi/c.spacing(native[0]["interval"])))
    dense={"indicators":{"width":{"localizable":True,"detected":True,
        "regions":[{"interval":[1500+300*i,1550+300*i]} for i in range(25)]}}}
    regions,stats=localized_regions(dense,["width"],p,"tof_us")
    assert len(regions)==16 and stats["total"]==25


def test_overlay_modes_preserve_intensities_and_only_mark_the_selected_map_row():
    a=observed();opts=dict(patterns=[a,a],labels=["one","two"],run_ids=["a","b"],
        limits=locked_limits(a),selected="b",ribbon=a,change_regions=[{"interval":[2000,3000]}])
    off,_=heatmap_figure(**opts,overlay="off")
    ribbon,_=heatmap_figure(**opts,overlay="ribbon")
    both,_=heatmap_figure(**opts,overlay="both")
    assert off.data[0].z==both.data[0].z and off.data[1].y==both.data[1].y
    assert not any(s.type=="rect" for s in off.layout.shapes)
    assert len([s for s in ribbon.layout.shapes if s.type=="rect"])==1
    rows=[s for s in both.layout.shapes if s.type=="rect" and s.yref=="y2"]
    assert len(rows)==1
    np.testing.assert_allclose([rows[0].y0,rows[0].y1],[.57,1.43])


def test_pair_highlight_spans_both_endpoints_and_recognized_features_are_dotted():
    a=observed()
    fig,_=heatmap_figure([a]*3,labels=["one","two","three"],run_ids=["a","b","c"],
        limits=locked_limits(a),selected="c",comparison_run="b",ribbon=a,overlay="both",
        change_regions=[{"interval":[2000,2200],"status":"recognized_split"}])
    boxes=[s for s in fig.layout.shapes if s.type=="rect"]
    assert all(s.line.dash=="dot" for s in boxes)
    row=next(s for s in boxes if s.yref=="y2")
    np.testing.assert_allclose([row.y0,row.y1],[.57,2.43])
    assert len(boxes)==2


def test_region_support_is_masked_on_both_compared_patterns():
    p=observed();valid=p.valid.copy();valid[(p.x>2300)&(p.x<2600)]=False
    other=replace(p,valid=valid)
    regions,_=localized_regions(evidence(),["width"],p,"tof_us",comparison_pattern=other)
    assert len(regions)==2
    assert regions[0]["interval"][1]<2300 and regions[1]["interval"][0]>2600


def test_nonadjacent_reference_marks_only_the_two_compared_rows():
    a=observed()
    fig,_=heatmap_figure([a]*4,labels=list("ABCD"),run_ids=list("abcd"),
        limits=locked_limits(a),selected="d",comparison_run="a",ribbon=a,overlay="both",
        change_regions=[{"interval":[2000,2200]}])
    rows=[s for s in fig.layout.shapes if s.type=="rect" and s.yref=="y2"]
    assert sorted((s.y0,s.y1) for s in rows)==[(-.43,.43),(2.57,3.43)]


def test_a_merged_width_hit_does_not_turn_a_below_rule_doublet_into_a_split_hit():
    decision=evidence()
    decision["indicators"]["splitting"]={"localizable":True,"detected":False,"regions":[],
        "features":[{"kind":"local_component_separation_change","interval":[2100,2300],
                     "passes_rule":False,"before_profile":{"positions":[2150,2200]}}]}
    regions,_=localized_regions(decision,["width","splitting"],observed(),"tof_us")
    assert len(regions)==1 and regions[0]["status"]=="detected_change"
    assert not regions[0]["diagnostics"][0]["passes_rule"]
