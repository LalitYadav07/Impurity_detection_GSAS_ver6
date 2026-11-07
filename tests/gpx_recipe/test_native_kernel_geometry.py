"""Preparation of pure instrument kernels cannot reject atomic model families."""
import sys

import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from test_native_tof_scalars import native_records
from gpx_recipe.preparation import native_tof_kernels as kernels
from gpx_recipe.structural_live.optimization.native_forward import NativeTOFForward
from gpx_recipe.symmetry.compiler import file_sha


@pytest.mark.parametrize('mixed',[False,True])
def test_geometry_kernels_use_no_atomic_scattering_or_intensity_and_leave_data_intact(tmp_path,mixed):
    original,s=native_source(tmp_path/'inputs','PNT');n=s.fast.native
    point=(np.linspace(-.07,.09,len(s.radii)) if mixed else np.zeros(len(s.radii)))*s.radii
    before=file_sha(original)
    data=n.histo_phases[0][s.fast.slow.hist.name]['Data']
    observations=[np.ma.array(a,copy=True) for a in data[:3]]
    atomic={n.math.StructureFactor2.__code__,n.math.GetIntensityCorr.__code__,
            s.fast.empirical_constraint.require.__func__.__code__}
    called=[]
    previous=sys.getprofile()
    def watch(frame,event,arg):
        if event=='call' and frame.f_code in atomic:called.append(frame.f_code.co_name)
    try:
        sys.setprofile(watch)
        kernel=kernels.capture(s,point)
    finally:sys.setprofile(previous)
    assert not called
    expected,curve=native_records(s,point,physical=True)
    a=kernel.data._arrays;axis=a['axis']
    assert set(map(tuple,a['hkls']))==set(expected)
    for index,hkl in enumerate(a['hkls']):
        parameters,_,lo,hi=expected[tuple(hkl)]
        np.testing.assert_array_equal(a['bounds'][index],[lo,hi])
        # The pinned primitive receives float32 shapes and translated offsets.
        # lookup requires identical bits for every actual compiled input; a
        # harmless float64 wrapper intermediate is not the native contract.
        np.testing.assert_array_equal(a['parameters'][index,1:].astype(np.float32).view(np.uint32),
                                      parameters[1:].astype(np.float32).view(np.uint32))
        raw=kernel.lookup(hkl,parameters,axis,lo,hi,function=n.math.G2pwd.getEpsVoigt)
        assert raw is not None
        np.testing.assert_array_equal(raw,n.math.G2pwd.getEpsVoigt(*parameters,axis[lo:hi])[0])
    np.testing.assert_array_equal(NativeTOFForward(s).evaluate(point,kernel=kernel),curve)
    for actual,expected in zip(data[:3],observations):np.testing.assert_array_equal(actual,expected)
    assert file_sha(original)==before


def test_chemical_rejection_does_not_block_pure_kernels_or_authorize_a_pattern(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'inputs','PNT');point=np.zeros(len(s.radii))
    expected=kernels.capture(s,point).data
    def reject(*args,**kwargs):raise ValueError('Rejected atomic configuration')
    monkeypatch.setattr(s,'check',reject)
    monkeypatch.setattr(s.fast.empirical_constraint,'require',reject)
    actual=kernels.capture(s,point).data
    for name in expected._arrays:np.testing.assert_array_equal(actual._arrays[name],expected._arrays[name])
    with pytest.raises(ValueError,match='Rejected atomic configuration'):s.native(point)
    with pytest.raises(ValueError,match='Rejected atomic configuration'):NativeTOFForward(s).evaluate(point)
    with pytest.raises(ValueError,match='Rejected atomic configuration'):s.fast.evaluate(np.zeros(s.fast.dimension))


@pytest.mark.parametrize('invalid',['outside','nan','wrong_shape'])
def test_geometry_preparation_preserves_the_complete_declared_cell_domain(tmp_path,invalid):
    _,s=native_source(tmp_path/'inputs','PNT');point=np.zeros(len(s.radii))
    if invalid=='outside':point[-1]=2*s.radii[-1]
    elif invalid=='nan':point[-1]=np.nan
    else:point=point[:-1]
    with pytest.raises(ValueError,match='reviewed structural domain'):kernels.capture(s,point)


@pytest.mark.parametrize('kind',['displacement','nuisance'])
def test_instrument_only_mapping_cannot_release_atomic_or_nuisance_parameters(tmp_path,kind):
    _,s=native_source(tmp_path/'inputs','PNT');theta=np.zeros(s.fast.dimension)
    assert s.fast.nm>0
    theta[0 if kind=='displacement' else -1]=.001
    with pytest.raises(ValueError,match='cell-only'):s.fast._peak_geometry_values(theta)
