"""Authored contract curves exercise ambiguity, not scientific accuracy."""
import copy
import json

import numpy as np
import pytest

from gpx_recipe.preparation import structural_family_targets as targets
from gpx_recipe.preparation.structural_cohort import assemble_pairs
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_augmentation import Controls,ObservationAugmenter
from gpx_recipe.symmetry.compiler import file_sha
from test_structural_paired_blocks import block
from test_structural_mode_projection import attach,rewrite_projection


def mapped(root,*,scale=2.,irrep='GM1+',legacy=False,parent_labels=False,tiny=False):
    def change(a):
        a['parameters'][a['role']==2,0]=1e-200 if tiny else .1
        if parent_labels:a['geometry_groups'][a['role']==2]=166
    path=block(root,change=change,metadata=lambda d:d['candidate'].update(id=root.name))
    if legacy:return path
    attach(path)
    rewrite_projection(path,change=lambda d:d.update(native_mode_labels=[
        f'R-3m[0,0,0]{irrep}(a)[Ba1:c:dsp]A1(a)','unresolved origin label']),
        arrays=lambda a:a.update(physical_gram=np.array([[scale**2]],np.float64)))
    d=json.loads((path/'pairs.json').read_text())
    d['mode_projection']['metadata_sha256']=file_sha(path/'mode-projection.json')
    d['mode_projection']['arrays_sha256']=file_sha(path/'mode-projection.npz')
    (path/'pairs.json').write_text(json.dumps(d));read(path)
    return path


def prepared(paths,*,noise=False):
    def blocks():return (read(p) for p in paths)
    cohort=assemble_pairs(list(blocks()),bins=64,augmenter=(
        ObservationAugmenter(Controls(variants=1),instrument_sha256='b'*64) if noise else None))
    return blocks,cohort


def test_every_original_alias_expands_the_conditional_amplitude_interval(tmp_path):
    first=mapped(tmp_path/'first',scale=2.);second=mapped(tmp_path/'second',scale=4.)
    blocks,cohort=prepared([first,second]);before=[(p/'pairs.npz').read_bytes() for p in [first,second]]
    family=targets.build(blocks,cohort);row=next(i for i,t in enumerate(cohort['compatible_labels']) if t[0])
    batch=family.batch(np.array([row],np.int64))
    assert len(cohort['references'][row])>=2 and batch['lower'].tolist()==pytest.approx([.2])
    assert batch['upper'].tolist()==pytest.approx([.4])
    assert family.metadata['ambiguous_intervals']>0 and family.metadata['classes']==[12,166]
    assert family.metadata['opaque_native_axes']==2 and not family.metadata['absence_from_missing_family']
    assert not family.metadata['candidate_pruning_authorized']
    assert before==[(p/'pairs.npz').read_bytes() for p in [first,second]]
    family.verify(cohort)


def test_intervals_condition_on_native_class_instead_of_blending_incompatible_aliases(tmp_path):
    paths=[mapped(tmp_path/'first',scale=2.),mapped(tmp_path/'second',scale=4.,parent_labels=True)]
    blocks,cohort=prepared(paths);family=targets.build(blocks,cohort)
    row=next(i for i,t in enumerate(cohort['compatible_labels']) if t.all())
    batch=family.batch(np.array([row],np.int64))
    assert batch['space_group'].tolist()==[0,1]
    assert batch['lower'].tolist()==pytest.approx([.2,.4])
    assert np.array_equal(batch['lower'],batch['upper'])


@pytest.mark.parametrize('unknown',['legacy','other_family'])
def test_missing_family_or_map_does_not_create_zero_absence_targets(tmp_path,unknown):
    first=mapped(tmp_path/'first')
    second=mapped(tmp_path/'second',legacy=unknown=='legacy',irrep='GM3-')
    blocks,cohort=prepared([first,second]);family=targets.build(blocks,cohort)
    assert family.metadata['intervals']==0 and not family.metadata['available']
    assert family.metadata['unknown_class_family_targets']>0
    batch=family.batch(np.arange(min(len(family),4),dtype=np.int64))
    assert len(batch['lower'])==0 and not family.counts(np.arange(len(family),dtype=np.int64)).any()


def test_noise_siblings_share_sparse_targets_without_becoming_independent_examples(tmp_path):
    path=mapped(tmp_path/'source');blocks,cohort=prepared([path],noise=True)
    family=targets.build(blocks,cohort)
    assert len(family)==len(cohort['groups'])
    assert family.metadata['unique_alias_groups']<len(family)
    counts=family.counts(np.arange(len(family),dtype=np.int64))
    assert counts.sum()==len(family) and family.metadata['native_points']==17
    original={key:family.arrays[key].tobytes() for key in family.arrays}
    family.batch(np.array([0,0,1],np.int64))
    assert original=={key:family.arrays[key].tobytes() for key in family.arrays}
    for a in family.arrays.values():
        with pytest.raises(ValueError):a.flags.writeable=True


@pytest.mark.parametrize('change',['input','map','reference','labels','fold'])
def test_changed_native_context_or_alias_targets_are_rejected(tmp_path,change):
    path=mapped(tmp_path/'source');blocks,cohort=prepared([path],noise=True)
    if change=='input':cohort['input_sha256'][0]='f'*64
    elif change=='map':cohort['mode_projection_references'][0]='f'*64
    elif change=='reference':cohort['references'][0]=[[0,999]]
    elif change=='labels':cohort['compatible_labels'][0]=~cohort['compatible_labels'][0]
    else:
        groups={}
        for row,refs in enumerate(cohort['references']):groups.setdefault(tuple(map(tuple,refs)),[]).append(row)
        selected=next(rows for rows in groups.values() if len(rows)>1)
        cohort['split'][selected[0]]=(int(cohort['split'][selected[0]])+1)%3
    with pytest.raises(ValueError):targets.build(blocks,cohort)


def test_very_small_amplitudes_remain_nonzero_and_storage_limits_never_drop_targets(tmp_path,monkeypatch):
    path=mapped(tmp_path/'source',tiny=True);blocks,cohort=prepared([path]);family=targets.build(blocks,cohort)
    positive=family.arrays['upper'][family.arrays['upper']>0]
    assert len(positive) and np.allclose(positive/1e-200,2.,rtol=1e-12,atol=0)
    with pytest.raises(ValueError,match='storage'):targets.build(blocks,cohort,_maximum_bytes=1)
    with pytest.raises(ValueError):family.batch(np.arange(257,dtype=np.int64))
    monkeypatch.setattr(targets,'MAX_BATCH_BYTES',1)
    with pytest.raises(ValueError,match='Split'):family.batch(np.arange(2,dtype=np.int64))


def test_legacy_only_inputs_remain_explicitly_unavailable_for_mode_supervision(tmp_path):
    path=mapped(tmp_path/'legacy',legacy=True);blocks,cohort=prepared([path])
    family=targets.build(blocks,cohort)
    assert family.metadata['families']==[] and family.metadata['mapped_blocks']==0
    assert not family.metadata['available'] and family.metadata['native_points']==17
    assert family.counts(np.arange(len(family),dtype=np.int64)).shape==(2,0)
    with pytest.raises(TypeError):targets.TargetRows({}, {})


@pytest.mark.parametrize('change',['aliases','groups','split','classes','bounds','authority'])
def test_bound_target_tables_cannot_be_reused_after_row_or_provenance_changes(tmp_path,change):
    paths=[mapped(tmp_path/'first'),mapped(tmp_path/'second')];blocks,cohort=prepared(paths)
    family=targets.build(blocks,cohort)
    changed=dict(cohort)
    if change=='aliases':
        changed['references']=copy.deepcopy(cohort['references']);changed['references'][0].reverse()
    elif change=='groups':changed['groups']=list(cohort['groups']);changed['groups'][0]='different-physical-group'
    elif change=='split':changed['split']=cohort['split'].copy();changed['split'][0]=(int(changed['split'][0])+1)%3
    elif change=='classes':changed['classes']=cohort['classes'][::-1].copy()
    elif change=='bounds':family.arrays['upper']=family.arrays['upper']+.1
    else:family.metadata['experimental_mode_absence_qualified']=True
    with pytest.raises(ValueError):family.verify(changed)
