from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from gpx_recipe.preparation.library import PreparationLibrary, import_library


def library(tmp_path):
    cif = tmp_path/'parent.cif'; cif.write_text('data_public_test\n')
    profile = tmp_path/'bank.instprm'; profile.write_text('Type:PNT\nBank:1\n')
    models = [dict(kind='magnetic', bank=bank, k=k, candidate='public:model')
              for bank in ['1', '5'] for k in [['0', '0', '0'], ['1/2', '0', '0']]]
    return PreparationLibrary.create(tmp_path/'library', name='Public test',
        specification={'scope': 'finite synthetic'}, models=models,
        inputs={'parent.cif': cif, 'profiles/bank.instprm': profile})


def prepare_one(lib, identifier):
    attempt = lib.start(identifier)
    folder = lib.root/'entries'/identifier; folder.mkdir(parents=True)
    np.savez(folder/'response.npz', coefficient=np.array([1., 2.]))
    lib.finish(identifier, attempt, state='prepared', files=['entries/'+identifier+'/response.npz'])


def test_partial_library_roundtrip_retains_exact_coverage_and_safe_inputs(tmp_path):
    lib = library(tmp_path); ids = [r['id'] for r in lib.document['entries']]
    prepare_one(lib, ids[0])
    attempt = lib.start(ids[1]); lib.finish(ids[1], attempt, state='unsupported', reason='Public test limitation')
    lib.start(ids[2])
    packet = tmp_path/'portable.zip'; lib.export(packet)
    restored = import_library(packet, tmp_path/'imported')
    assert restored.coverage() == lib.coverage()
    assert restored.coverage()['counts'] == dict(cancelled=0, failed=0, pending=1, prepared=1, preparing=1, unsupported=1)
    assert not restored.coverage()['complete'] and not restored.coverage()['all_prepared']
    assert len(restored.coverage()['banks']) == len(restored.coverage()['families']) == 2
    with zipfile.ZipFile(packet) as archive:
        assert not any(Path(n).suffix == '.gpx' or n.endswith('.lock') for n in archive.namelist())


def test_recovery_does_not_start_an_entry_already_owned_by_another_attempt(tmp_path):
    lib = library(tmp_path); identifier = lib.document['entries'][0]['id']
    first = lib.start(identifier)
    reopened = PreparationLibrary(lib.root)
    with pytest.raises(ValueError, match='pending'):
        reopened.start(identifier)
    reopened.interrupt(identifier, first)
    reopened.retry(identifier)
    second = reopened.start(identifier)
    assert second == first+1
    with pytest.raises(ValueError, match='Late'):
        lib.finish(identifier, first, state='failed', reason='Old job returned late')
    assert PreparationLibrary(lib.root).row(identifier)['attempt'] == second


def test_simultaneous_entry_admission_has_one_owner(tmp_path):
    lib = library(tmp_path); identifier = lib.document['entries'][0]['id']
    def start(_):
        try:
            return PreparationLibrary(lib.root).start(identifier)
        except ValueError:
            return 'already owned'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(start, range(2)))
    assert sorted(str(v) for v in results) == ['1', 'already owned']


@pytest.mark.parametrize('target', ['input', 'response', 'manifest'])
def test_changed_bytes_cannot_be_exported_as_a_verified_preparation(tmp_path, target):
    lib = library(tmp_path); identifier = lib.document['entries'][0]['id']
    prepare_one(lib, identifier)
    if target == 'input':
        (lib.root/'inputs/parent.cif').write_text('changed')
    elif target == 'response':
        np.savez(lib.root/'entries'/identifier/'response.npz', coefficient=np.array([4.]))
    else:
        data = json.loads(lib.path.read_text()); data['entries'][0]['state'] = 'pending'
        lib.path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        lib.export(tmp_path/'bad.zip')


@pytest.mark.parametrize('extra', ['worker.py', '../escape.cif', 'entries/unexpected.npz'])
def test_import_rejects_foreign_executable_or_escaping_members_before_use(tmp_path, extra):
    lib = library(tmp_path); packet = tmp_path/'portable.zip'; lib.export(packet)
    with zipfile.ZipFile(packet, 'a') as archive:
        archive.writestr(extra, b'not allowed')
    destination = tmp_path/'bad'
    with pytest.raises(ValueError):
        import_library(packet, destination)
    assert not destination.exists()
    assert not list(tmp_path.glob('bad.*.partial'))


def test_source_checksums_and_declared_models_are_part_of_library_identity(tmp_path):
    lib = library(tmp_path)
    data = json.loads(lib.path.read_text())
    data['entries'].pop()
    lib.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='identity'):
        PreparationLibrary(lib.root)


def test_failed_or_partial_models_never_publish_arrays_as_prepared(tmp_path):
    lib = library(tmp_path); identifier = lib.document['entries'][0]['id']
    attempt = lib.start(identifier)
    with pytest.raises(ValueError, match='Only fully prepared'):
        lib.finish(identifier, attempt, state='prepared')
    assert PreparationLibrary(lib.root).row(identifier)['state'] == 'preparing'
