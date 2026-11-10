"""Replay preserves complete pool coverage, bounded reads and isolated holdouts."""
import numpy as np
import pytest

from gpx_recipe.preparation.structural_training_curriculum import replay_batches
from gpx_recipe.preparation.structural_training_data import feature_rows


def test_large_catalog_replay_covers_every_family_without_large_feature_batches():
    values=np.zeros((310,3,64),np.float32)
    values[:,0,0]=np.arange(310)
    rows=feature_rows(values);split=np.zeros(310,np.int8);split[-10:]=1
    representatives=np.arange(230)
    # The full data pool retains its order, including rare rows and repeats.
    order=np.tile(np.arange(300),4)
    observed=[];replayed=[]
    for selected,features in replay_batches(rows,order,representatives,split=split):
        assert len(selected)<=80 and features.shape==(len(selected),3,64)
        np.testing.assert_array_equal(features[:,0,0],selected)
        n=min(64,len(order)-len(observed))
        observed.extend(selected[:n]);replayed.extend(selected[n:])
    assert observed==order.tolist()
    assert set(replayed)==set(representatives)
    assert not set(replayed)&set(range(300,310))
    assert max(np.bincount(replayed))-min(np.bincount(replayed))<=1


def test_replay_keeps_weak_clean_and_noisy_data_rows_and_does_not_inflate_unique_counts():
    rows=feature_rows(np.zeros((13,3,64),np.float32));split=np.array([0]*10+[1,1,2],np.int8)
    order=np.array([7,8,2,3,0,9,9]);representatives=np.array([1,4,6])
    result=list(replay_batches(rows,order,representatives,split=split,size=4))
    assert result[0][0].tolist()==[7,8,2,3,1,4,6]
    assert result[1][0].tolist()==[0,9,9,1,4,6]
    assert set(np.concatenate([r[0] for r in result]))==set(order)|set(representatives)
    assert list(replay_batches(rows,np.empty(0,np.int64),representatives,split=split))==[]


@pytest.mark.parametrize('heldout',[1,2])
@pytest.mark.parametrize('where',['pool','representative'])
def test_replay_never_reads_a_heldout_curve(heldout,where):
    rows=feature_rows(np.zeros((4,3,64),np.float32));split=np.array([0,0,heldout,0],np.int8)
    indices=np.array([0,2] if where=='pool' else [0,1])
    representatives=np.array([2] if where=='representative' else [1])
    with pytest.raises(ValueError,match='validation or test'):
        list(replay_batches(rows,indices,representatives,split=split))


@pytest.mark.parametrize('case',['large_batch','negative','foreign','floating','empty_representatives'])
def test_invalid_replay_request_fails_before_reading_features(case):
    rows=feature_rows(np.zeros((5,3,64),np.float32));split=np.zeros(5,np.int8)
    indices=np.array([0,1]);representatives=np.array([2]);size=64
    if case=='large_batch':size=65
    elif case=='negative':indices=np.array([-1])
    elif case=='foreign':representatives=np.array([5])
    elif case=='floating':indices=indices.astype(float)
    else:representatives=np.empty(0,np.int64)
    with pytest.raises(ValueError,match='Bounded training'):
        list(replay_batches(rows,indices,representatives,split=split,size=size))
