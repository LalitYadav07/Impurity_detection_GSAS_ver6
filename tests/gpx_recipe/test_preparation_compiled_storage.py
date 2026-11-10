"""Cold analysis portability without copying or deleting the training corpus.

The transport fixture contains opaque weights, not a scientific accuracy test.
Native bank/fit tests independently exercise the numerical handoff.
"""
import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from gpx_recipe.preparation import library as module
from gpx_recipe.preparation import compiled_storage
from gpx_recipe.preparation.library import PreparationLibrary, import_library, validate_manifest
from gpx_recipe.preparation import hints, structural_ranking_hints
from gpx_recipe.symmetry.compiler import file_sha
from test_preparation_structural_ranking_hints import prepared
from test_structural_preparation_dependencies import library_inputs, complete_examples, TRAINING


def inventory(library):
    return {a['path']: file_sha(library.root/a['path']) for a in library._assets()}


def materialized_bytes(library, document):
    deferred = compiled_storage.deferred(document)
    return len((json.dumps(document, separators=(',', ':'))+'\n').encode()) + sum(
        (library.root/a['path']).stat().st_size for a in library._assets()
        if a['path'] not in deferred)


def test_compiled_copy_crosses_portable_budget_without_losing_source_or_scope(tmp_path, monkeypatch):
    source, protocol = prepared(tmp_path/'source', three_curve=True)
    original = copy.deepcopy(source.document)
    before = inventory(source)
    view = compiled_storage.snapshot(original)
    compact = materialized_bytes(source, view)
    full = materialized_bytes(source, original)
    assert compact < full
    # Reproduce the measured working-data/portable-size gap without allocating
    # gigabytes or weakening the archive extraction limit.
    monkeypatch.setattr(module, 'MAX_ARCHIVE_BYTES', compact)
    packet = tmp_path/'compiled.zip'
    source.export(packet)
    restored = import_library(packet, tmp_path/'cold')
    assert restored.document == view
    assert restored.document['id'] == original['id']
    assert restored.document['entries'] == original['entries']
    assert restored.document['inputs'] == original['inputs']
    assert restored.coverage() == source.coverage()
    assert source.document == original and inventory(source) == before
    source.verify()
    assert materialized_bytes(restored, restored.document) == compact
    missing = compiled_storage.deferred(view)
    assert missing
    with zipfile.ZipFile(packet) as archive:
        assert not set(archive.namelist()) & missing
        for name in set(before)-missing:
            assert file_sha(restored.root/name) == before[name]
    # Import/re-export does not need the original machine's files or paths.
    again = tmp_path/'again.zip'
    restored.export(again)
    second = import_library(again, tmp_path/'second-cold')
    assert second.document == restored.document
    assert hints.capture(restored, protocol, tmp_path/'model-packet') == hints.capture(
        source, protocol, tmp_path/'original-packet')
    captured = hints.read(tmp_path/'model-packet', protocol)
    assert captured['ranking_models']['eligible_entries'] == 1
    assert len(captured['ranking_models']['entries']) == 1
    assert protocol['candidates'] == original_protocol(source)
    row = captured['ranking_models']['entries'][0]
    structural_ranking_hints.checked_files(tmp_path/'model-packet', row, protocol)


def original_protocol(source):
    return [r['model']['settings']['candidate'] for r in source.document['entries']
            if r['model']['kind'] == 'structural-examples']


def test_compiled_copy_cannot_silently_retrain_without_the_original_arrays(tmp_path):
    source, _ = prepared(tmp_path/'source')
    packet = tmp_path/'compiled.zip'; source.export(packet)
    restored = import_library(packet, tmp_path/'cold')
    example = next(r for r in restored.document['entries']
                   if r['model']['kind'] == 'structural-examples')
    training = next(r for r in restored.document['entries'] if r['model']['kind'] == TRAINING)
    with pytest.raises(ValueError, match='Training arrays are archived'):
        restored.prepared_assets(example['id'])
    for action in (restored.start, restored.retry):
        with pytest.raises(ValueError, match='compiled library is ready for analysis'):
            action(training['id'])
    assert restored.detach_imported_work() == 0
    assert restored.prepared_assets(training['id'])
    assert source.prepared_assets(example['id'])  # original full data still usable


@pytest.mark.parametrize('original_job_active', [False, True])
def test_compiled_import_retains_native_retry_and_separate_job_ownership(tmp_path, original_job_active):
    import shutil
    source, _ = prepared(tmp_path/'source')
    native = dict(kind='magnetic', bank='bank 1', k=['0','0','0'], candidate='public-control')
    working = PreparationLibrary.create(tmp_path/'mixed', name='Mixed finite scope',
        specification=source.document['specification'],
        inputs={a['path'].removeprefix('inputs/'): source.root/a['path'] for a in source.document['inputs']},
        models=[r['model'] for r in source.document['entries']]+[native])
    for row in source.document['entries']:
        attempt=working.start(row['id'])
        for asset in row['files']:
            target=working.root/asset['path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source.root/asset['path'],target)
        working.finish(row['id'],attempt,state='prepared',files=[a['path'] for a in row['files']])
    row=working.document['entries'][-1];key=row['id']
    if original_job_active:working.start(key)
    original=copy.deepcopy(working.document)
    packet=tmp_path/'mixed.zip';working.export(packet)
    cold=import_library(packet,tmp_path/'cold')
    assert cold.detach_imported_work()==int(original_job_active)
    assert cold.row(key)['state']=='pending'
    attempt=cold.start(key)
    cold.finish(key,attempt,state='failed',reason='Public retry control')
    cold.retry(key);attempt=cold.start(key)
    path=cold.root/'entries'/key/'response.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,response=np.ones(3))
    cold.finish(key,attempt,state='prepared',files=[str(path.relative_to(cold.root))])
    assert cold.prepared_assets(key)
    assert working.document==original
    working.verify()


@pytest.mark.parametrize('change', ['weights', 'input', 'omit', 'checksum', 'duplicate', 'unfinished'])
def test_compiled_marker_cannot_hide_scientific_assets_or_incomplete_work(tmp_path, change):
    source, _ = prepared(tmp_path/'source')
    value = compiled_storage.snapshot(source.document)
    deferred = value['storage']['deferred_assets']
    if change == 'weights':
        deferred.append(next(a for r in value['entries'] if r['model']['kind'] == TRAINING
                             for a in r['files'] if a['path'].endswith('/weights.npz')))
    elif change == 'input': deferred.append(value['inputs'][0])
    elif change == 'omit': deferred.pop()
    elif change == 'checksum': deferred[0]['sha256'] = 'f'*64
    elif change == 'duplicate': deferred.append(copy.deepcopy(deferred[0]))
    else:
        row = next(r for r in value['entries'] if r['model']['kind'] == TRAINING)
        row.update(state='pending', files=[])
    with pytest.raises(ValueError, match='Compiled storage'):
        validate_manifest(value)


@pytest.mark.parametrize('marker', [None, False, [], {'schema': compiled_storage.SCHEMA}])
def test_compiled_marker_is_closed_and_typed(tmp_path, marker):
    source, _ = prepared(tmp_path/'source')
    value = copy.deepcopy(source.document); value['storage'] = marker
    with pytest.raises(ValueError, match='Compiled storage'):
        validate_manifest(value)


@pytest.mark.parametrize('change', ['unexpected_array', 'missing_weights', 'corrupt_weights'])
def test_archive_must_match_the_exact_materialized_inventory(tmp_path, change):
    source, _ = prepared(tmp_path/'source')
    packet = tmp_path/'compiled.zip'; source.export(packet)
    with zipfile.ZipFile(packet) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    value = json.loads(members['library.json'])
    weights = next(a['path'] for r in value['entries'] if r['model']['kind'] == TRAINING
                   for a in r['files'] if a['path'].endswith('/weights.npz'))
    if change == 'unexpected_array':
        name = next(iter(compiled_storage.deferred(value)))
        members[name] = (source.root/name).read_bytes()
    elif change == 'missing_weights': members.pop(weights)
    else: members[weights] += b'changed'
    with zipfile.ZipFile(tmp_path/'changed.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items(): archive.writestr(name, raw)
    with pytest.raises(ValueError): import_library(tmp_path/'changed.zip', tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
    assert not list(tmp_path.glob('invalid.*.partial'))


def test_pending_training_retains_full_arrays_and_cannot_bypass_archive_budget(tmp_path, monkeypatch):
    source, originals = library_inputs(tmp_path/'source')
    complete_examples(source, originals, tmp_path/'attempts')
    assert compiled_storage.snapshot(source.document) == source.document
    packet = tmp_path/'pending.zip'; source.export(packet)
    restored = import_library(packet, tmp_path/'cold')
    assert restored.document == source.document
    for row in restored.document['entries']:
        if row['model']['kind'] == 'structural-examples': assert restored.prepared_assets(row['id'])
    monkeypatch.setattr(module, 'MAX_ARCHIVE_BYTES', materialized_bytes(source, source.document)-1)
    with pytest.raises(ValueError, match='Complete training'):
        source.export(tmp_path/'too-large.zip')
    assert not (tmp_path/'too-large.zip').exists()
    assert not list(tmp_path.glob('too-large.zip.*.partial'))


def test_zip_framing_cannot_publish_an_archive_beyond_the_import_limit(tmp_path, monkeypatch):
    source,_=prepared(tmp_path/'source')
    compact=materialized_bytes(source,compiled_storage.snapshot(source.document))
    monkeypatch.setattr(module,'MAX_ARCHIVE_BYTES',compact)
    original=zipfile.ZipFile
    def stored(path,mode='r',*args,**kwargs):
        # Worst-case incompressible output: the stored payload fits exactly,
        # but ZIP framing also counts toward the compressed transfer limit.
        if mode=='x':args=(zipfile.ZIP_STORED,)+args[1:];kwargs['compresslevel']=None
        return original(path,mode,*args,**kwargs)
    monkeypatch.setattr(zipfile,'ZipFile',stored)
    packet=tmp_path/'previous.zip';packet.write_bytes(b'previous reviewed copy')
    with pytest.raises(ValueError,match='archive exceeds its portable byte budget'):
        source.export(packet)
    assert packet.read_bytes()==b'previous reviewed copy'
    assert not list(tmp_path.glob('previous.zip.*.partial'))
