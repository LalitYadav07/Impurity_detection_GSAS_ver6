import json

import numpy as np
import pytest

from gpx_recipe.preparation.structural_cohort import assemble_pairs,assemble
from gpx_recipe.preparation.structural_paired_blocks import read
from gpx_recipe.preparation.structural_paired_store import build,load
from gpx_recipe.preparation.structural_augmentation import Controls,ObservationAugmenter
from test_structural_paired_blocks import block


def alias(root):
    def change(a):a['geometry_groups'][a['role']==2]=8
    def metadata(d):d['candidate']['id']='alternative';d['declared_child_group']['number']=8
    return block(root,change=change,metadata=metadata)


def test_three_curve_cohort_preserves_alias_ambiguity_and_every_class_holdout(tmp_path):
    blocks=[read(block(tmp_path/'first')),read(alias(tmp_path/'alias'))]
    result=assemble_pairs(blocks,bins=128)
    assert result['features'].shape[1:]==(3,128)
    assert result['reference_features'].shape==(3,128)
    assert result['classes'].tolist()==[8,12,166]
    assert result['original_examples']==34 and result['unique_examples']<34
    child_rows=result['compatible_labels'][:,0]
    assert result['compatible_labels'][child_rows,1].all()
    for code in range(3):assert result['compatible_labels'][result['split']==code].any(0).all()
    for group in result['groups']:
        rows=[i for i,g in enumerate(result['groups']) if g==group]
        assert len(set(result['split'][rows]))==1
    assert result['screening_qualified'] is False
    # The old eight-channel assembler must not reinterpret these pairs.
    with pytest.raises(TypeError):assemble(blocks,bins=128)


def test_file_backed_pairs_match_dense_assembly_including_noise_siblings(tmp_path):
    paths=[block(tmp_path/'first'),alias(tmp_path/'alias')]
    augmenter=ObservationAugmenter(Controls(variants=2),instrument_sha256='b'*64)
    dense=assemble_pairs([read(p) for p in paths],bins=128,augmenter=augmenter)
    stored=build(paths,tmp_path/'cache',bins=128,augmenter=augmenter)
    try:
        assert stored['features'].storage=='file_backed' and stored['channels']==3
        np.testing.assert_array_equal(stored['features'].take(np.arange(stored['unique_examples'])),dense['features'])
        for key in ('compatible_labels','classes','split','q_edges','reference_features'):
            np.testing.assert_array_equal(stored[key],dense[key])
        for key in ('groups','class_candidate_ids','candidate_groups','input_sha256','augmented_examples'):
            assert stored[key]==dense[key]
    finally:stored['features'].close()
    again=load(tmp_path/'cache');again['features'].close()


def test_changed_pair_cache_cannot_be_admitted(tmp_path):
    path=block(tmp_path/'first');stored=build([path],tmp_path/'cache',bins=128);stored['features'].close()
    raw=(tmp_path/'cache/store.json').read_bytes();metadata=json.loads(raw);metadata['channels']=8
    (tmp_path/'cache/store.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='three-curve'):load(tmp_path/'cache')
    (tmp_path/'cache/store.json').write_bytes(raw)
    with (tmp_path/'cache/features.npy').open('r+b') as out:out.seek(-4,2);out.write(b'bad!')
    with pytest.raises(ValueError,match='changed'):load(tmp_path/'cache')
