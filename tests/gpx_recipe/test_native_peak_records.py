"""Packed working records retain full data while reducing Python overhead."""
import dataclasses
import numpy as np
import pytest

from gpx_recipe.live.native_peak_records import PackedPeakRecords,record_items
from gpx_recipe.live.native_peak_batch import pack_records,BatchUnavailable,MAXIMUM_BYTES


def columns():
    return [np.array([[2.,1.,2.,3.,4.],[3.,1.,2.,3.,4.]]),np.array([2.,-1.]),
        np.array([[0,4],[1,5]],np.int64),np.array([1.,2.]),np.array([[1,0,0],[0,1,0]],np.int64)]


def test_packed_transient_columns_are_readonly_and_borrowed_without_another_per_reflection_copy():
    records=PackedPeakRecords(*columns());axis=np.arange(6.)
    p,h,b,maximum,required=pack_records(records,axis,MAXIMUM_BYTES)
    assert p is records.parameters and h is records.heights and b is records.bounds
    assert maximum==4 and records.numeric_bytes==2*96
    assert required==2*axis.nbytes+8*maximum
    for array in (p,h,b,records.hkls,records.spacings):
        assert not array.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):records.parameters=np.zeros((2,5))


@pytest.mark.parametrize('case',['nan','duplicate','negative_window','short_window','bad_shape','bad_dtype','view','nonpositive_d'])
def test_invalid_packed_columns_cannot_enter_the_native_batch(case):
    a=columns()
    if case=='nan':a[0][0,0]=np.nan
    elif case=='duplicate':a[4][1]=a[4][0]
    elif case=='negative_window':a[2][0,0]=-1
    elif case=='short_window':a[2][0,1]=1
    elif case=='bad_shape':a[0]=np.zeros((2,4))
    elif case=='bad_dtype':a[2]=a[2].astype(np.int32)
    elif case=='view':a[0]=a[0][:]
    else:a[3][0]=0.
    with pytest.raises(ValueError):PackedPeakRecords(*a)


def test_packed_windows_still_need_the_actual_native_axis_and_scratch_budget():
    records=PackedPeakRecords(*columns())
    with pytest.raises(BatchUnavailable):pack_records(records,np.arange(4.),MAXIMUM_BYTES)
    with pytest.raises(BatchUnavailable):pack_records(records,np.arange(6.),1)


def test_per_peak_fallback_reads_every_packed_row_without_copying_a_dictionary():
    records=PackedPeakRecords(*columns());rows=list(record_items(records))
    assert [key for key,_ in rows]==[(1,0,0),(0,1,0)]
    assert [row[1][1] for row in rows]==[2.,-1.]
    for index,(_, (parameters,height,low,high)) in enumerate(rows):
        np.testing.assert_array_equal(parameters,records.parameters[index])
        assert np.shares_memory(parameters,records.parameters)
        assert not parameters.flags.writeable
        assert (low,high)==tuple(records.bounds[index])
    legacy={key:value for key,value in rows}
    assert [key for key,_ in record_items(legacy)]==list(legacy)
    assert next(record_items(legacy))[1] is rows[0][1]
    with pytest.raises(TypeError):list(record_items([]))
