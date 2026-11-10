"""Admission and immutable ownership checks; numerical controls run natively."""
import numpy as np
import pytest
from gpx_recipe.structural_live.optimization import native_window_response as windows


@pytest.fixture
def builder(monkeypatch):
    value=object.__new__(windows.NativeWindowBuilder)
    value.maximum_bytes=64*1024*1024
    monkeypatch.setattr(value,'check',lambda:None)
    return value


def columns():return {0:(1,4,2.,np.array([0.,1.,2.],np.float32))}


@pytest.mark.parametrize('change',[
    'index','negative_low','outside_high','wrong_size','double_precision','nonfinite','factor','boolean_count',
])
def test_invalid_native_windows_never_reach_the_helper(builder,change):
    c=columns();count=2
    if change=='index':c={2:c[0]}
    if change=='negative_low':c={0:(-1,2,2.,c[0][3])}
    if change=='outside_high':c={0:(3,6,2.,c[0][3])}
    if change=='wrong_size':c[0]=(1,3,2.,c[0][3])
    if change=='double_precision':c[0]=(1,4,2.,c[0][3].astype(float))
    if change=='nonfinite':c[0][3][1]=np.nan
    if change=='factor':c[0]=(1,4,np.inf,c[0][3])
    if change=='boolean_count':count=True
    with pytest.raises(ValueError):builder.assemble(c,count,5)


def test_native_windows_own_immutable_arrays_and_complete_column_order(builder):
    c=columns();response=builder.assemble(c,2,5);c[0][3][1]=99.
    np.testing.assert_array_equal(response._raw,np.array([0.,1.,2.],np.float32))
    for a in (response._raw,response._bounds,response._pointers,response._factors,response._mapping):
        with pytest.raises(ValueError):a.flags.writeable=True
    permuted=response.reordered(np.array([1,0]))
    assert permuted._raw is response._raw and permuted._factors is response._factors
    np.testing.assert_array_equal(permuted._mapping,[-1,0])
    assert response.numeric_bytes==12+16+16+8+16
    for order in [[0,0],[1],[True,False],[0.,1.]]:
        with pytest.raises(ValueError):response.reordered(order)


def test_response_byte_limit_precedes_mapping_allocation(builder,monkeypatch):
    monkeypatch.setattr(np,'full',lambda *_a,**_kw:pytest.fail('Unbounded mapping allocation'))
    with pytest.raises(ValueError,match='budget'):builder.assemble({},10**9,5)


@pytest.mark.parametrize('coefficients',[
    np.zeros(3),np.zeros((2,257)),np.array([1.,np.nan]),np.ones((2,1,1)),
    np.array([1+1j,2+0j]),np.array([True,False]),
])
def test_invalid_coefficients_are_refused_before_a_native_call(builder,coefficients):
    response=builder.assemble(columns(),2,5)
    with pytest.raises(ValueError):response@coefficients


def test_multiplication_byte_limit_precedes_output_allocation(builder,monkeypatch):
    response=builder.assemble(columns(),2,5);builder.maximum_bytes=1
    monkeypatch.setattr(np,'zeros',lambda *_a,**_kw:pytest.fail('Unbounded product allocation'))
    with pytest.raises(ValueError,match='workspace'):response@np.ones(2)


def test_imported_constructor_arguments_cannot_claim_native_capture():
    with pytest.raises(TypeError,match='owned native capture'):
        windows.NativeWindowResponse(None,None,None,None,None,None,4)
