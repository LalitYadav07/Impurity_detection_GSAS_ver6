"""Large data-layout controls, not scientific training or accuracy evidence."""
import json

import numpy as np
import pytest

from gpx_recipe.preparation import structural_replay_limits as limits
from gpx_recipe.preparation import structural_training_counts as counts
from gpx_recipe.preparation import structural_training_checkpoint as checkpoint
from gpx_recipe.preparation import structural_training_transfer as transfer
from gpx_recipe.preparation.structural_partitions import partition, partition_rows
from gpx_recipe.preparation.structural_training_data import MAX_ROWS, open_rows
from gpx_recipe.preparation.structural_store import DiskFeatures
from gpx_recipe.preparation.tof_kernel_arrays import array_identity
from test_structural_training_transfer import owned


def layout(rows=200000):
    return dict(schema=limits.SCHEMA, rows=rows, bins=128, channels=3,
                native_bytes=50000000, index_bytes=rows*limits.INDEX_BYTES_PER_ROW)


def grouped_rows():
    n=200000;indices=np.arange(n);group_indices=indices//1000
    names=[f'physical-trajectory-{i}' for i in range(200)]
    groups=[names[i] for i in group_indices]
    labels=np.stack([np.ones(n,bool),group_indices%3==0,group_indices%5==0],axis=1)
    return groups, labels


def test_large_logical_rows_retain_the_exact_group_level_assignment_and_old_limits(tmp_path):
    groups,labels=grouped_rows()
    actual=partition_rows(groups,labels)
    expected=partition(groups[::1000],labels[::1000])
    assert actual==expected and set(actual.values())=={'train','validation','test'}
    with pytest.raises(ValueError,match='Bounded physical'):partition(groups,labels)
    assert MAX_ROWS==131072  # Dense and NPY admission remain unchanged.
    with pytest.raises(ValueError,match='disk budget'):DiskFeatures(tmp_path,channels=3).reserve(MAX_ROWS+1,64)
    with pytest.raises(ValueError,match='dimensions'):
        open_rows(tmp_path/'not-created.npy',sha256='a'*64,capacity=MAX_ROWS+1,rows=MAX_ROWS+1,bins=64,channels=3)
    assert not list(tmp_path.iterdir())


def test_large_counts_keep_clean_pairs_simulations_alias_coverage_and_holdouts_distinct():
    groups,labels=grouped_rows();assignments=partition_rows(groups,labels)
    split=[{'train':0,'validation':1,'test':2}[assignments[g]] for g in groups]
    clean=[i%4==0 for i in range(len(groups))]
    result=counts.describe(groups,split,clean,generated_pairs=50000,simulated_observations=150000,
        classes=[1,2,5],compatible_labels=(r.tolist() for r in labels),row_layout=layout())
    assert result['schema']==counts.REPLAY_SCHEMA and result['unique_rows']==200000
    assert result['unique_clean_pairs']==50000 and result['unique_simulated_only_rows']==150000
    assert result['trajectory_groups']==200 and counts.validate(result)==result
    legacy=dict(result,schema=counts.COVERAGE_SCHEMA);del legacy['row_layout']
    with pytest.raises(ValueError):counts.validate(legacy)
    with pytest.raises(ValueError,match='native-row layout'):
        counts.describe(groups,split,clean,generated_pairs=50000,simulated_observations=150000,
            classes=[1,2,5],compatible_labels=(r.tolist() for r in labels))


@pytest.mark.parametrize('key,value', [('rows',200001),('channels',8),('native_bytes',9*1024**3),
    ('index_bytes',1),('schema','unknown'),('rows',True)])
def test_large_row_layout_cannot_claim_different_dimensions_or_unbounded_storage(key,value):
    invalid=dict(layout());invalid[key]=value
    with pytest.raises(ValueError):limits.validate(invalid,rows=200000)


def large_checkpoint(path,*,legacy=False):
    owned(path)
    with np.load(path/'training-checkpoint.npz',allow_pickle=False) as archive:
        arrays={k:archive[k] for k in archive.files}
    metadata=json.loads(arrays.pop('metadata').tobytes())
    arrays['visited']=np.zeros(200000,bool)
    if not legacy:metadata.update(schema=checkpoint.REPLAY_SCHEMA,row_layout=layout())
    metadata['arrays_sha256']=array_identity(arrays)
    arrays['metadata']=np.frombuffer(json.dumps(metadata).encode(),np.uint8)
    np.savez(path/'training-checkpoint.npz',**arrays)
    return path


def test_large_checkpoint_transfer_is_lossless_and_legacy_format_cannot_bypass_its_limit(tmp_path):
    source=large_checkpoint(tmp_path/'source')
    receipt=transfer.snapshot(source,tmp_path/'packet',identity='a'*64,epochs=60)
    result=transfer.restore(tmp_path/'packet',tmp_path/'destination',identity='a'*64,epochs=60)
    assert result['disposition']=='copied_for_complete_native_checkpoint_audit'
    assert receipt['trained_model'] is False and receipt['candidate_pruning_authorized'] is False
    assert (tmp_path/'destination/training-checkpoint.npz').read_bytes()==(source/'training-checkpoint.npz').read_bytes()
    bad=large_checkpoint(tmp_path/'legacy',legacy=True)
    with pytest.raises(ValueError,match='Legacy checkpoint'):transfer.inspect(bad/'training-checkpoint.npz')
