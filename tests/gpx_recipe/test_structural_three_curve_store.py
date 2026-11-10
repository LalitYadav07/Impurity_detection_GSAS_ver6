import numpy as np
import pytest

from gpx_recipe.preparation.structural_store import DiskFeatures,digest_file
from gpx_recipe.preparation.structural_training_data import open_rows,feature_rows,channel_statistics


def test_three_curve_working_files_use_bounded_reads_and_declared_channel_identity(tmp_path):
    rng=np.random.default_rng(821);values=rng.normal(size=(12,3,128)).astype(np.float32)
    sink=DiskFeatures(tmp_path,channels=3)
    try:
        sink.reserve(12,128)
        for row in values:sink.append(row)
        sink.finish(12)
    finally:sink.close()
    checksum=digest_file(tmp_path/'features.npy')
    rows=open_rows(tmp_path/'features.npy',sha256=checksum,capacity=12,rows=12,bins=128,channels=3)
    try:
        assert rows.shape==(12,3,128) and rows._array is None
        order=np.array([9,2,11,0]);np.testing.assert_array_equal(rows.take(order),values[order])
        actual=channel_statistics(rows,order);expected=channel_statistics(feature_rows(values),order)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        assert actual[0].shape==(3,1) and abs(actual[0][2,0])>1e-5
        assert actual[1][2,0]!=1  # Residual is not treated as a coverage mask.
        with pytest.raises(ValueError,match='bounded batch'):rows.take(np.zeros(257,np.int64))
    finally:rows.close()
    # The legacy default cannot reinterpret the same bytes as eight channels.
    with pytest.raises(ValueError,match='dimensions'):
        open_rows(tmp_path/'features.npy',sha256=checksum,capacity=12,rows=12,bins=128)


def test_bad_three_curve_row_cannot_be_published_in_a_working_file(tmp_path):
    sink=DiskFeatures(tmp_path,channels=3)
    try:
        sink.reserve(2,128)
        with pytest.raises(ValueError,match='channels'):sink.append(np.zeros((8,128),np.float32))
        row=np.zeros((3,128),np.float32);row[2,4]=np.nan
        with pytest.raises(ValueError,match='channels'):sink.append(row)
        assert sink.count==0
    finally:sink.close()
