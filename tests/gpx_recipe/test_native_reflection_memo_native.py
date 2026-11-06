"""Native identities, metric/range crossings, and complete profile controls."""
import copy
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIlattice as lattice,GSASIIspc as space
from gpx_recipe.live.native_reflection_memo import NativeReflectionMemo


CASES=[('P -1',[4.1,5.2,6.3,81.,93.,102.]),('P 2/m',[4.1,5.2,6.3,90.,103.,90.]),
 ('C 2/m',[4.1,5.2,6.3,90.,103.,90.]),('P m m m',[4.1,5.2,6.3,90.,90.,90.]),
 ('P 4/m',[4.1,4.1,6.3,90.,90.,90.]),('I 4/m m m',[4.1,4.1,6.3,90.,90.,90.]),
 ('P 6/m',[4.1,4.1,6.3,90.,90.,120.]),('P 6/m m m',[4.1,4.1,6.3,90.,90.,120.]),
 ('P -3',[4.1,4.1,6.3,90.,90.,120.]),('P -3 m 1',[4.1,4.1,6.3,90.,90.,120.]),
 ('P -3 1 m',[4.1,4.1,6.3,90.,90.,120.]),('R -3 c',[4.1,4.1,6.3,90.,90.,120.]),
 ('P m -3 m',[4.1,4.1,4.1,90.,90.,90.]),('F d -3 m',[4.1,4.1,4.1,90.,90.,90.]),
 ('P a -3',[4.1,4.1,4.1,90.,90.,90.])]
PHASES={'p':{'General':dict(Type='nuclear')}}


@pytest.mark.parametrize('symbol,cell',CASES)
def test_native_metric_bits_order_and_boundary_reflections_are_preserved(symbol,cell):
    error,sg=space.SpcGroup(symbol);assert not error
    original=lattice.GenHLaue;memo=NativeReflectionMemo();rng=np.random.default_rng(822371)
    a0=np.asarray(lattice.cell2A(cell))
    for index in range(14):
        a=a0*(1+rng.uniform(-.003,.003,size=6));cut=.7*(1+rng.uniform(-.003,.003))
        expected=original(cut,sg,a)
        with memo.evaluation(lattice,PHASES):actual=lattice.GenHLaue(dmin=cut,SGData=sg,A=a)
        np.testing.assert_array_equal(actual,expected)
        assert lattice.GenHLaue is original
    for cut in [expected[0][3],expected[-1][3],np.nextafter(expected[-1][3],0.),np.nextafter(expected[-1][3],np.inf)]:
        with memo.evaluation(lattice,PHASES):actual=lattice.GenHLaue(cut,sg,a)
        np.testing.assert_array_equal(actual,original(cut,sg,a))
    assert memo.bytes<=memo.maximum_bytes
    assert memo.hits>0 or memo.bypasses['small reflection inventory']>0


def test_changed_metric_range_and_symmetry_rebuild_or_use_native_without_losing_peaks():
    _,sg=space.SpcGroup('P -1');a=np.asarray(lattice.cell2A([4.1,5.2,6.3,81.,93.,102.]))
    original=lattice.GenHLaue;memo=NativeReflectionMemo()
    for cut,metric in [(.7,a),(.7,a*1.01),(.4,a),(.4,a*.3),(.4,a*1.7)]:
        with memo.evaluation(lattice,PHASES):actual=lattice.GenHLaue(cut,sg,metric)
        np.testing.assert_array_equal(actual,original(cut,sg,metric))
    other=copy.deepcopy(sg);other['SGLatt']='I'
    with memo.evaluation(lattice,PHASES):actual=lattice.GenHLaue(.7,other,a)
    np.testing.assert_array_equal(actual,original(.7,other,a))
    assert memo.builds>=2
    for record in memo.records.values():
        assert not record['hkls'].flags.writeable and not record['monomials'].flags.writeable


def test_small_memory_and_float32_contexts_keep_original_native_results():
    _,sg=space.SpcGroup('P -1');a=np.asarray(lattice.cell2A([4.1,5.2,6.3,81.,93.,102.]))
    original=lattice.GenHLaue
    for memo,values in [(NativeReflectionMemo(1),a),(NativeReflectionMemo(),a.astype('float32'))]:
        with memo.evaluation(lattice,PHASES):actual=lattice.GenHLaue(.7,sg,values)
        np.testing.assert_array_equal(actual,original(.7,sg,values))
        assert memo.builds==0 and lattice.GenHLaue is original


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_complete_native_profiles_and_mode_derivatives_keep_the_same_values(tmp_path,kind):
    from test_native_profile_capture import native_source
    _,source=native_source(tmp_path/'input',kind);native=source.fast.native
    points=[np.zeros(len(source.radii)),np.linspace(-.03,.05,len(source.radii))]
    native.reflection_memo.enabled=False
    expected=[source.native_mode_jacobian(p*source.radii) for p in points]
    native.reflection_memo.enabled=True
    for p,(profile,derivatives) in zip(points,expected):
        actual,jac=source.native_mode_jacobian(p*source.radii)
        np.testing.assert_array_equal(actual,profile);np.testing.assert_array_equal(jac,derivatives)
