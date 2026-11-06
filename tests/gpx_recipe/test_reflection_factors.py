"""Data-only factor integrity, qualification ownership and memory limits."""
import json

import numpy as np
import pytest

from gpx_recipe.live.reflection_factors import (
    AmplitudeBlock, ScatteringFactorCache, digest_array,
)


def proof(factors):
    return dict(all_coefficient_directions_checked=True,
        dimensions=factors.shape[-1], reflections=len(factors))


def test_portable_factors_retain_data_but_never_the_process_qualification(tmp_path):
    factors = np.arange(36., dtype=float).reshape(2, 6, 3)
    cache = ScatteringFactorCache(tmp_path)
    cache._admit_native('a'*64, factors, proof(factors))
    resident, qualified = cache.lookup('a'*64)
    assert qualified and not resident.factors.flags.writeable
    factors[:] = -1
    assert resident.factors.max() == 35.
    loaded, qualified = ScatteringFactorCache(tmp_path).lookup('a'*64)
    assert not qualified
    np.testing.assert_array_equal(loaded.factors, resident.factors)
    assert not loaded.factors.flags.writeable
    with np.load(tmp_path/('a'*64+'.npz'), allow_pickle=False) as data:
        manifest = json.loads(str(data['manifest']))
        assert manifest['observations_included'] is False
        assert set(data.files) == {'manifest', 'factors'}


def test_lru_bounds_live_memory_and_keeps_qualification_with_exact_arrays():
    factors = np.ones((2, 6, 3))
    cache = ScatteringFactorCache(maximum_bytes=factors.nbytes*2)
    for key in ['a', 'b']:
        cache._admit_native(key*64, factors, proof(factors))
    cache.lookup('a'*64)
    cache._admit_native('c'*64, factors, proof(factors))
    assert cache.lookup('b'*64) == (None, False)
    assert cache.lookup('a'*64)[1] and cache.lookup('c'*64)[1]


def test_native_gate_cannot_admit_a_partially_checked_block():
    cache = ScatteringFactorCache()
    factors = np.ones((2, 6, 3))
    with pytest.raises(ValueError, match='Complete native'):
        cache._admit_native('a'*64, factors, dict(dimensions=3, reflections=2))
    assert cache.lookup('a'*64) == (None, False)


def test_altered_resident_arrays_lose_their_native_seal():
    cache = ScatteringFactorCache()
    factors = np.ones((2, 6, 3))
    block = cache._admit_native('a'*64, factors, proof(factors))
    block.factors.flags.writeable = True
    block.factors[0, 0, 0] = 2.
    with pytest.raises(ValueError, match='changed'):
        cache.lookup('a'*64)
    assert cache.lookup('a'*64) == (None, False)


@pytest.mark.parametrize('mutation', ['checksum', 'identity', 'inventory', 'nonfinite', 'dtype'])
def test_invalid_portable_blocks_are_rejected_before_native_use(tmp_path, mutation):
    factors = np.ones((2, 6, 3))
    cache = ScatteringFactorCache(tmp_path)
    cache._admit_native('a'*64, factors, proof(factors))
    path = tmp_path/('a'*64+'.npz')
    with np.load(path, allow_pickle=False) as data:
        manifest = json.loads(str(data['manifest']))
        values = dict(factors=data['factors'].copy())
    if mutation == 'checksum':
        values['factors'] *= 2.
    elif mutation == 'identity':
        manifest['identity'] = 'b'*64
    elif mutation == 'inventory':
        values['unexpected'] = np.array(1.)
    elif mutation == 'nonfinite':
        values['factors'][0, 0, 0] = np.nan
        manifest['sha256'] = digest_array(values['factors'])
    else:
        values['factors'] = values['factors'].astype('f4')
        manifest['sha256'] = digest_array(values['factors'])
    np.savez_compressed(path, manifest=np.array(json.dumps(manifest)), **values)
    with pytest.raises(ValueError):
        ScatteringFactorCache(tmp_path).lookup('a'*64)


@pytest.mark.parametrize('factors', [np.ones((2, 5, 3)), np.ones((2, 6, 25)),
    np.ones((0, 6, 3)), np.ones((2, 6, 3), dtype='f4')])
def test_unsupported_factor_shapes_and_dtypes_are_not_accepted(factors):
    with pytest.raises(ValueError):
        AmplitudeBlock('a'*64, factors, {})
