"""Display aggregation must never draw through an unresolved native gap."""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova.pattern_transition import ObservedPattern, PatternScope
from radar_pd_nova.transition_tracking_plot import (
    binned_intensity, display_grid, heatmap_figure, locked_limits,
)
from radar_pd_nova.unet_transition_ui import _display_points, _evolution_figure, _map_grid


def pattern(n=12000):
    x=np.arange(1.,n+1.)
    scope=PatternScope('synthetic','NOM','1','q_inv_angstrom','profile','reduction','qualified','density')
    return ObservedPattern(scope,x,np.full(n,5.),np.ones(n),np.ones(n,bool),'synthetic')


def assert_no_missing_crossings(p,x,y):
    x=np.asarray(x);y=np.asarray(y)
    for left,right,a,b in zip(x[:-1],x[1:],y[:-1],y[1:]):
        if np.isfinite(a) and np.isfinite(b):
            assert p.valid[(p.x>=left)&(p.x<=right)].all()
            assert np.isfinite(p.intensity[(p.x>=left)&(p.x<=right)]).all()


def test_mixed_native_support_blanks_whole_heatmap_pixel():
    p=pattern();valid=p.valid.copy();valid[6002:6007]=False;p=replace(p,valid=valid)
    _,edges=display_grid(p,1200)
    values=binned_intensity(p,edges)
    index=np.searchsorted(edges,p.x[6004],side='right')-1
    assert np.isnan(values[index])
    assert np.isfinite(values[index-1]) and np.isfinite(values[index+1])
    assert np.array_equal(p.intensity,np.full(len(p.x),5.))


def test_invalid_native_cell_overlap_matters_even_when_point_lies_outside_pixel():
    p=SimpleNamespace(x=np.arange(1.,9.),intensity=np.full(8,5.),valid=np.ones(8,bool))
    p.valid[3]=False
    # The unavailable x=4 sample owns [3.5,4.5]; it contaminates the first
    # display cell despite the sample center lying beyond that cell's edge.
    values=binned_intensity(p,[2.5,3.75,5.5,6.5])
    assert np.isnan(values[:2]).all() and values[2]==5.
    # Exactly touching the unavailable support edge has zero overlap.
    values=binned_intensity(p,[2.5,3.5,4.5,5.5])
    assert values[0]==values[2]==5. and np.isnan(values[1])


def test_cropped_pattern_never_fills_a_partly_unobserved_pixel():
    p=SimpleNamespace(x=np.arange(1.,9.),intensity=np.full(8,5.),valid=np.ones(8,bool))
    values=binned_intensity(p,[0.,1.5,2.5,7.5,8.5,9.5])
    assert np.isnan(values[[0,3,4]]).all()
    assert np.isfinite(values[[1,2]]).all()
    # The native rightmost display edge is not falsely classified as outside.
    _,edges=display_grid(p,8)
    assert np.isfinite(binned_intensity(p,edges)).all()


@pytest.mark.parametrize('missing',['mask','nonfinite','background'])
def test_each_unavailable_measurement_kind_blanks_partial_pixel(missing):
    p=pattern(40);background=np.zeros(40)
    if missing=='mask':
        valid=p.valid.copy();valid[12]=False;p=replace(p,valid=valid)
    elif missing=='nonfinite':
        p=SimpleNamespace(x=p.x,intensity=p.intensity.copy(),valid=p.valid)
        p.intensity[12]=np.nan
    else:background[12]=np.nan
    values=binned_intensity(p,[1.,10.5,20.5,40.],background=background)
    assert values[0]==values[2]==5. and np.isnan(values[1])


def test_heatmaps_keep_exact_nonuniform_aggregation_edges():
    p=pattern(8000)
    fig,_=heatmap_figure([p],labels=['a'],run_ids=['a'],limits=locked_limits(p),region=[2000,2070])
    _,expected=display_grid(p,512,region=[2000,2070])
    assert np.array_equal(fig.data[0].x,expected)
    assert len(fig.data[0].x)==len(fig.data[0].z[0])+1
    assert fig.data[0].connectgaps is False and fig.data[0].zsmooth is False
    assert 'partial cells blank' in fig.layout.meta['display_binning']
    # The active transition map uses the same explicit-edge convention.
    current=replace(p,x=p.x**1.02)
    _,expected=_map_grid([current])
    scans={'a':dict(run_id='a',run_number=1,instrument='NOM',ipts='IPTS-1',metadata={})}
    result,_,_=_evolution_figure([dict(run_id='a',events=[])],[current],scans,
        'a',current,current,selected_types=[])
    heat=next(row for row in result['data'] if row['type']=='heatmap')
    assert np.array_equal(heat['x'],expected)
    shape=(tuple(int(n) for n in heat['z']['shape'].split(','))
           if isinstance(heat['z'],dict) else np.asarray(heat['z']).shape)
    assert shape==(1,len(heat['x'])-1)
    assert heat['connectgaps'] is False and heat['zsmooth'] is False


def test_multiple_gaps_in_one_decimation_bucket_cannot_be_connected():
    p=pattern(30000);valid=p.valid.copy();valid[[10002,10007]]=False
    y=p.intensity.copy();y[10005]=4.;y[10009]=6.
    p=replace(p,valid=valid,intensity=y)
    x,y=_display_points(p)
    assert len(x)<=6000
    assert_no_missing_crossings(p,x,y)


@pytest.mark.parametrize('limit',[4,8,31,6000])
def test_alternating_missing_bins_keep_breaks_under_strict_point_budget(limit):
    p=pattern(30000);valid=p.valid.copy();valid[1::2]=False;p=replace(p,valid=valid)
    x,y=_display_points(p,limit)
    assert len(x)<=limit and np.isnan(y).any()
    assert_no_missing_crossings(p,x,y)


def test_fully_observed_decimation_preserves_extrema_and_endpoint_coordinates():
    p=pattern(30000);y=p.intensity.copy();y[10125]=-30.;y[25128]=40.
    p=replace(p,intensity=y);x,y=_display_points(p)
    assert len(x)<=6000 and x[0]==p.x[0] and x[-1]==p.x[-1]
    assert min(y)==-30. and max(y)==40.
    small=pattern(40);x,y=_display_points(small,40)
    assert np.array_equal(x,small.x) and np.array_equal(y,small.intensity)


def test_large_display_has_bounded_work_and_does_not_mutate_native_arrays():
    p=pattern(100000);valid=p.valid.copy();valid[::107]=False;p=replace(p,valid=valid)
    original=(p.x.tobytes(),p.intensity.tobytes(),p.valid.tobytes())
    start=perf_counter();x,y=_display_points(p);_,edges=display_grid(p,1200)
    values=binned_intensity(p,edges)
    assert len(x)<=6000 and len(values)==1200
    assert (p.x.tobytes(),p.intensity.tobytes(),p.valid.tobytes())==original
    # Broad runaway-work guard; precise timings are recorded in the audit.
    assert perf_counter()-start<5.
