"""Portable data boundaries and likelihood binding need no native runtime."""
from dataclasses import replace
import json
import zipfile

import numpy as np
import pytest

from gpx_recipe.preparation.magnetic import MagneticResponse, load
from gpx_recipe.live.array_limits import validate_npz,validate_npz_bytes


def response():
    x = np.linspace(1., 4., 20)
    q = np.zeros((20, 2, 2))
    q[:, 0, 0] = np.exp(-(x-2.)**2)
    q[:, 1, 1] = np.exp(-(x-3.)**2)
    q[:, 0, 1] = q[:, 1, 0] = .1
    return MagneticResponse(x, np.ones(20), np.ones(20)*2., q, np.ones((20, 1)),
        np.array([[[1., 0.], [0., 1.], [0., 0.]]]), 'a'*64, {'kind': 'unit-test'})


def binding(r):
    return dict(x=r.x, y=np.ones(20)*5., weights=np.ones(20)*4., fit_mask=np.ones(20, bool),
        normalization_mask=np.ones(20, bool), physics_sha256=r.physics_sha256,
        maximum_moment=5., fixed_scale=2.)


def test_theory_bytes_do_not_authorize_a_fit(tmp_path):
    r = response(); path = tmp_path/'response.npz'; r.save(path)
    restored = load(path, physics_sha256=r.physics_sha256)
    with pytest.raises(ValueError, match='Qualify'):
        restored.bind(**binding(restored))


def test_qualified_arrays_cannot_change_before_binding():
    r = response(); r._seal_native()
    r.nuclear.flags.writeable = True
    r.nuclear[0] += 1.
    with pytest.raises(ValueError, match='Qualify'):
        r.bind(**binding(r))


@pytest.mark.parametrize('change', ['geometry', 'grid', 'fit_mask', 'zero_weight', 'nan'])
def test_invalid_current_context_cannot_reuse_preparation(change):
    r = response(); r._seal_native(); kwargs = binding(r)
    if change == 'geometry':
        kwargs['physics_sha256'] = 'b'*64
    elif change == 'grid':
        kwargs['x'] = r.x+.00001
    elif change == 'fit_mask':
        kwargs['normalization_mask'][0] = False
    elif change == 'zero_weight':
        kwargs['weights'][0] = 0.
    else:
        kwargs['y'][0] = np.nan
    with pytest.raises(ValueError):
        r.bind(**kwargs)


def test_excluded_intensity_is_retained_in_native_normalization():
    r = response(); r._seal_native(); kwargs = binding(r)
    kwargs['fit_mask'][0] = False
    kwargs['y'][0] = 100.
    model = r.bind(**kwargs)
    assert len(model.offset) == 19
    assert model.normalization == 4.*(100.**2+19*25.)


@pytest.mark.parametrize('change', ['array', 'scope', 'identity', 'inventory'])
def test_changed_bundle_fails_before_scientific_use(tmp_path, change):
    r = response(); path = tmp_path/'response.npz'; r.save(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in archive.files}
    if change == 'array':
        data['nuclear'][0] += .1
    elif change == 'inventory':
        data['foreign'] = np.ones(1)
    else:
        m = json.loads(str(data['manifest']))
        if change == 'scope':
            m['observations_included'] = True
        else:
            m['physics_sha256'] = 'b'*64
        data['manifest'] = np.array(json.dumps(m))
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError):
        load(path, physics_sha256=r.physics_sha256)


def test_float32_response_requires_a_separate_precision_qualification():
    r = response()
    with pytest.raises(ValueError, match='float64'):
        replace(r, quadratic=r.quadratic.astype('float32'))


def test_archive_headers_cannot_claim_unbounded_allocation(tmp_path):
    path = tmp_path/'oversized.npz'
    import io
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': '<f8', 'fortran_order': False,
                                                'shape': (1_000_000_000,)})
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('quadratic.npy', header.getvalue())
    with pytest.raises(ValueError, match='budget'):
        load(path, physics_sha256='a'*64)


def test_array_budget_cannot_be_increased_by_a_caller(tmp_path):
    with pytest.raises(ValueError, match='only tighten'):
        validate_npz(tmp_path/'absent.npz', maximum_bytes=1024**3)


@pytest.mark.parametrize('variant',['valid','object','expanded','observation_inventory'])
def test_immutable_byte_validation_retains_file_validator_limits(tmp_path,variant):
    path=tmp_path/'input.npz'
    values=np.array([object()]) if variant=='object' else np.zeros(1000)
    np.savez_compressed(path,values=values)
    options=dict(maximum_bytes=4096) if variant=='expanded' else dict(observation=variant=='observation_inventory')
    if variant=='valid':
        assert validate_npz(path,**options) and validate_npz_bytes(path.read_bytes(),**options)
    else:
        for call,value in ((validate_npz,path),(validate_npz_bytes,path.read_bytes())):
            with pytest.raises(ValueError):call(value,**options)


def test_immutable_validator_rejects_mutable_input_and_oversized_source():
    with pytest.raises(TypeError,match='Immutable'):
        validate_npz_bytes(bytearray(b'not immutable'))
    with pytest.raises(ValueError,match='byte budget'):
        validate_npz_bytes(b'x'*1001,maximum_bytes=1000)
