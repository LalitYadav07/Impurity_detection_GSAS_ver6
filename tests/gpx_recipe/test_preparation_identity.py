"""Preparation qualification must not cross a different numerical runtime."""
import copy
import json
from types import SimpleNamespace

import pytest

from gpx_recipe.preparation import identity, rendered_magnetic


def test_recorded_conditions_are_only_omitted_from_exact_bank_or_sample_keys():
    values={name:17. for name in identity.ACQUISITION_PARAMETERS}
    values.update({'X':.1,'Temperature_model':99.,'ranId':13})
    original=values.copy()
    assert identity.without_recorded_conditions(values)=={'X':.1,'Temperature_model':99.,'ranId':13}
    prefixed={':3:'+k:v for k,v in values.items()}
    prefixed.update({'0::Temperature':73.,':4:Temperature':81.})
    actual=identity.without_recorded_conditions(prefixed,prefix=':3:')
    assert actual['0::Temperature']==73. and actual[':4:Temperature']==81.
    assert actual[':3:X']==.1 and actual[':3:Temperature_model']==99.
    assert values==original and ':3:Temperature' in prefixed


@pytest.mark.parametrize('change', ['numpy', 'scipy', 'python', 'system', 'machine', 'byteorder', 'blas'])
def test_runtime_changes_invalidate_rendered_physics_without_changing_the_model(monkeypatch, change):
    from test_preparation_rendered_magnetic import context
    from gpx_recipe.live import validated_backend
    native_math = SimpleNamespace(__file__=identity.__file__, G2pwd=SimpleNamespace(__file__=identity.__file__),
        **{name: identity._plain for name in ('StructureFactor2', 'MagStructureFactor2',
                                             'MagStructureFactorDerv', 'getPowderProfile')})
    monkeypatch.setattr(validated_backend, 'active_identity', lambda: 'reviewed-test-runtime')
    monkeypatch.setattr(validated_backend, 'backend_identity', lambda: {})
    values = context(); values['math'] = native_math
    before = rendered_magnetic.identity(**values)
    runtime = copy.deepcopy(identity.numerical_identity())
    if change == 'python': runtime['python']['version'][2] += 1
    elif change == 'blas': runtime['reported_builds']['numpy']['blas']['version'] = 'other-build'
    else: runtime[change] = 'different-runtime'
    monkeypatch.setattr(identity, 'numerical_identity', lambda: runtime)
    assert rendered_magnetic.identity(**values) != before
    assert identity.native_identity(SimpleNamespace(math=native_math))['numerical_runtime'] == runtime


def test_build_paths_and_thread_counts_do_not_enter_portable_identity(monkeypatch):
    config = {'Build Dependencies': {
        'blas': {'name': 'openblas', 'version': '1', 'openblas configuration': 'DYNAMIC_ARCH',
                 'lib directory': '/private/build/one', 'num_threads': 1},
        'lapack': {'name': 'openblas', 'version': '1'}}}
    monkeypatch.setattr(identity.np, '__config__', SimpleNamespace(CONFIG=config))
    before = identity.numerical_identity()
    config['Build Dependencies']['blas']['lib directory'] = '/different/install'
    config['Build Dependencies']['blas']['num_threads'] = 8
    assert identity.numerical_identity() == before
    assert '/private/' not in json.dumps(before)


def test_unreported_build_metadata_is_explicit_and_versions_remain_pinned(monkeypatch):
    monkeypatch.setattr(identity.np, '__config__', SimpleNamespace())
    value = identity.numerical_identity()
    assert value['reported_builds']['numpy'] == {'blas': {}, 'lapack': {}}
    assert value['numpy'] and value['scipy'] and value['python']['version']
    json.dumps(value, allow_nan=False)
