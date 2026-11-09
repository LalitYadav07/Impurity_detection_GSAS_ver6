"""Compact analysis copies retain the complete, original training inventory.

Only completed example arrays may remain in the originating preparation.
Their hashes and all metadata travel with the compiled model. Native fitting
assets, candidates, weights and scientific coverage are never omitted.
"""
import copy
from pathlib import PurePosixPath

SCHEMA = 'radar-preparation-compiled-storage/v1'
ARRAYS = {'examples.npz', 'mode-projection.npz'}


def _eligible(document):
    entries = document['entries']
    examples = [r for r in entries if r['model']['kind'] == 'structural-examples']
    training = [r for r in entries if r['model']['kind'] == 'structural-training']
    return bool(examples and training) and all(r['state'] == 'prepared' for r in examples + training)


def _arrays(document):
    return sorted((dict(a) for r in document['entries']
                   if r['model']['kind'] == 'structural-examples'
                   for a in r['files'] if PurePosixPath(a['path']).name in ARRAYS),
                  key=lambda a: a['path'])


def validate(document):
    if 'storage' not in document:
        return
    value = document['storage']
    # Exact equality to the original manifest's example-array inventory keeps
    # this marker from hiding a damaged native bank, model or scientific input.
    expected = _arrays(document)
    if (not isinstance(value, dict) or set(value) != {'schema', 'deferred_assets'}
            or value['schema'] != SCHEMA or not _eligible(document) or not expected
            or value['deferred_assets'] != expected):
        raise ValueError('Compiled storage may defer only the exact completed training arrays')


def deferred(document):
    validate(document)
    return {a['path'] for a in document.get('storage', {}).get('deferred_assets', [])}


def snapshot(document):
    """Do not mutate the working library or change its scientific identity."""
    result = copy.deepcopy(document)
    if 'storage' not in result and _eligible(result):
        arrays = _arrays(result)
        if arrays:
            result['storage'] = dict(schema=SCHEMA, deferred_assets=arrays)
    validate(result)
    return result


def require_working_copy(document, model):
    if (document.get('storage') is not None
            and model['kind'] in {'structural-examples', 'structural-training'}):
        raise ValueError('This compiled library is ready for analysis. Recover the original '
                         'preparation History to regenerate or retrain its archived arrays.')
