"""Previewing a cursor must neither refit nor rebuild the diffraction map."""
from dataclasses import replace
import copy

import numpy as np

from radar_pd_nova.transition_cursor import cursor_frames
from radar_pd_nova.transition_tracking_plot import locked_limits
from test_pattern_transition import observed


def test_preview_uses_native_samples_masks_and_pair_regions():
    a=observed();valid=a.valid.copy();valid[22:25]=False
    b=replace(a,intensity=a.intensity*2,valid=valid)
    original=b.intensity.copy()
    frames=cursor_frames([a,b],None,["a","b"],["one","two"],selected="b",
        limits=locked_limits(a),scale="linear",contrast=50,overlay="both",
        regions=lambda run:{"comparison_run":"a","regions":[{"interval":[2000,2300],"status":"recognized_split"}]})
    assert frames[1]["trace"]["x"]==b.x.tolist()
    assert frames[1]["trace"]["y"][22:25]==[None]*3
    assert frames[1]["trace"]["customdata"]==b.intensity.tolist()
    assert all(v["line"]["dash"]=="dot" for v in frames[1]["shapes"])
    pair=next(s for s in frames[1]["shapes"] if s["yref"]=="y2")
    assert [pair["y0"],pair["y1"]]==[-.43,1.43]
    np.testing.assert_array_equal(b.intensity,original)


def test_budget_keeps_nearest_frames_and_leaves_explicit_missing_entries():
    a=observed();before=copy.deepcopy(a)
    frames=cursor_frames([a]*5,None,list("abcde"),list("ABCDE"),selected="d",
        limits=locked_limits(a),scale="linear",contrast=50,point_budget=2*len(a.x))
    assert [i for i,f in enumerate(frames) if f]==[2,3]
    assert sum(len(f["trace"]["x"]) for f in frames if f)==2*len(a.x)
    np.testing.assert_array_equal(a.x,before.x)
    np.testing.assert_array_equal(a.intensity,before.intensity)
