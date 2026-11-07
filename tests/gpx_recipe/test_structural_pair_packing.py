import hashlib
import json

import numpy as np
import pytest

from gpx_recipe.preparation.structural_paired_blocks import read,pack_block,MAX_BYTES
from gpx_recipe.preparation import structural_pair_arrays as codec
from test_structural_paired_blocks import block


def test_native_pair_packing_preserves_all_array_bits_controls_and_trajectories(tmp_path):
    source=block(tmp_path/'source');original=read(source)
    packed=pack_block(source,tmp_path/'packed')
    assert packed.manifest['schema']==codec.SCHEMA and packed.manifest['array_encoding']==codec.ENCODING
    assert packed.manifest['original_pairs_sha256']==original.manifest['pairs_sha256']
    for key,value in original.arrays.items():
        assert packed.arrays[key].dtype==value.dtype and packed.arrays[key].shape==value.shape
        assert packed.arrays[key].tobytes()==value.tobytes()
        with pytest.raises(ValueError):packed.arrays[key].flags.writeable=True
    assert packed.manifest['groups']==original.manifest['groups']
    with np.load(tmp_path/'packed/pairs.npz',allow_pickle=False) as a:
        assert len(a.files)==13 and all(key not in a.files for key in codec.CURVES)


def test_lossless_storage_keeps_three_curve_features_labels_and_holdouts_identical(tmp_path):
    from gpx_recipe.preparation.structural_cohort import assemble_pairs
    original=read(block(tmp_path/'source'));packed=pack_block(tmp_path/'source',tmp_path/'packed')
    before=assemble_pairs([original],bins=128);after=assemble_pairs([packed],bins=128)
    for key in ('features','compatible_labels','classes','split','q_edges','reference_features'):
        np.testing.assert_array_equal(before[key],after[key])
    assert before['groups']==after['groups'] and before['class_candidate_ids']==after['class_candidate_ids']
    assert before['input_sha256']!=after['input_sha256']  # New bytes have a new identity.


@pytest.mark.parametrize('case',['layout','dtype','missing','nonfinite','encoding','dimensions','padding'])
def test_hash_valid_corrupt_packed_arrays_cannot_override_physical_checks(tmp_path,case):
    source=block(tmp_path/'source');pack_block(source,tmp_path/'packed');root=tmp_path/'packed'
    d=json.loads((root/'pairs.json').read_text())
    with np.load(root/'pairs.npz',allow_pickle=False) as a:arrays={k:a[k] for k in a.files}
    key='theory_intensity_byte_planes'
    if case=='layout':arrays[key]=arrays[key].T.copy()
    elif case=='dtype':arrays[key]=arrays[key].astype(np.int16)
    elif case=='missing':del arrays[key]
    elif case=='nonfinite':
        decoded=codec.decode(arrays,rows=17,bins=101,maximum_bytes=MAX_BYTES)
        decoded['theory_intensity'][2,5]=np.nan;arrays=codec.encode(decoded)
    elif case=='encoding':d['array_encoding']='execute/custom-code'
    elif case=='padding':arrays[key].view(np.uint8)[:,-1]=1
    else:d['examples']=10**12
    np.savez_compressed(root/'pairs.npz',**arrays)
    d['pairs_sha256']=hashlib.sha256((root/'pairs.npz').read_bytes()).hexdigest();(root/'pairs.json').write_text(json.dumps(d))
    with pytest.raises(ValueError):read(root)


def test_packed_dimensions_are_bounded_before_reconstructing_any_curve():
    with pytest.raises(ValueError,match='bound'):
        codec.decode({},rows=36865,bins=25001,maximum_bytes=MAX_BYTES)


def test_shuffled_storage_keeps_the_generic_element_limit_for_large_complete_blocks():
    import io
    from gpx_recipe.live.array_limits import validate_npz_bytes
    value=np.zeros((2000,1001),np.float64)
    packed=codec.encode({name:value for name in codec.CURVES})
    assert sum(a.size for a in packed.values())<48_000_000<3*value.size*8
    raw=io.BytesIO();np.savez_compressed(raw,**packed)
    assert validate_npz_bytes(raw.getvalue(),maximum_bytes=MAX_BYTES)
    restored=codec.decode(packed,rows=2000,bins=1001,maximum_bytes=MAX_BYTES)
    for result in restored.values():assert result.tobytes()==value.tobytes()
