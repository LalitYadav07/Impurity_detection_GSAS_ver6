"""Only identical compiled peak inputs can share a native evaluation."""
import numpy as np
import pytest

from gpx_recipe.live.native_peak_groups import group


def inputs(count=512):
    axis=np.linspace(99985.,100030.,601)
    parameters=np.tile([100000.,.5,.25,1.,.125],(count,1))
    bounds=np.tile([0,len(axis)],(count,1)).astype(np.int64)
    return parameters,bounds,axis


def grouped(parameters,bounds,axis,maximum_bytes=2**26):
    return group(parameters,bounds,axis,maximum_bytes=maximum_bytes,base_bytes=8192)


def test_identical_arguments_share_one_profile_without_mutating_inputs():
    p,b,x=inputs();before=p.copy(),b.copy(),x.copy()
    result,reason=grouped(p,b,x)
    assert reason is None and len(result.parameters)==1
    np.testing.assert_array_equal(result.mapping,np.zeros(len(p),np.int64))
    assert result.points==len(x) and result.requested_points==len(x)*len(p)
    assert result.required_bytes<2**26 and result.pointers.tolist()==[0,len(x)]
    for actual,expected in zip((p,b,x),before):np.testing.assert_array_equal(actual,expected)


def test_float64_shape_differences_that_round_identically_can_share():
    p,b,x=inputs();p[1::2,1]=np.nextafter(p[1::2,1],np.inf)
    result,reason=grouped(p,b,x)
    assert reason is None and len(result.parameters)==1


def test_equal_rounded_positions_do_not_authorize_different_translated_offsets():
    p,b,x=inputs();p[1::2,0]=np.nextafter(p[1::2,0],np.inf)
    assert np.float32(p[0,0])==np.float32(p[1,0])
    result,reason=grouped(p,b,x)
    assert reason is None and len(result.parameters)==2 and result.offset_collisions>0
    assert result.mapping.tolist()==[0,1]*256


def test_different_positions_can_share_only_when_all_compiled_offsets_match():
    p,b,x=inputs();x+=100.;p[1::2,0]=np.nextafter(p[1::2,0],np.inf)
    result,reason=grouped(p,b,x)
    assert reason is None and len(result.parameters)==1 and result.offset_comparisons>0


@pytest.mark.parametrize('change',['signed_zero','shape','window'])
def test_distinct_compiled_inputs_keep_separate_groups(change):
    p,b,x=inputs()
    if change=='signed_zero':p[::2,4]=0.;p[1::2,4]=-0.
    elif change=='shape':p[1::2,2]=np.nextafter(np.float32(.25),np.float32(1.))
    else:b[1::2,0]=1
    result,reason=grouped(p,b,x)
    assert reason is None and len(result.parameters)==2
    assert result.mapping.tolist()==[0,1]*256


def test_many_false_position_collisions_cannot_create_quadratic_offset_work():
    p,b,x=inputs();p[:,0]+=np.arange(len(p))*1e-6
    assert len(np.unique(p[:,0].astype(np.float32)))==1
    result,reason=grouped(p,b,x)
    assert result is None and reason=='offset comparison exceeds original point work'


@pytest.mark.parametrize('change,expected',[
    ('small','small workload'),('unique','few repeated native inputs'),
    ('index_budget','grouping index exceeds workspace'),
    ('raw_budget','unique native shapes exceed workspace'),
    ('position_overflow','position outside grouping precision')])
def test_unsupported_grouping_falls_back_without_truncating(change,expected):
    p,b,x=inputs();maximum=2**26
    if change=='small':p=p[:255];b=b[:255]
    elif change=='unique':p[:,1]+=np.arange(len(p))
    elif change=='index_budget':maximum=8192
    elif change=='raw_budget':maximum=8192+28*len(p)+8
    else:p[:,0]=1e100
    result,reason=grouped(p,b,x,maximum)
    assert result is None and reason==expected
