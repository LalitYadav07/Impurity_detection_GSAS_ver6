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


def test_published_metadata_budget_matches_creation_mutation_and_import(tmp_path,monkeypatch):
    from gpx_recipe.preparation import library as module
    lib=library(tmp_path)
    compact=len((json.dumps(lib.document,separators=(',',':'))+'\n').encode())
    pretty=len((json.dumps(lib.document,indent=2)+'\n').encode())
    limit=(compact+pretty)//2
    assert compact<limit<pretty
    monkeypatch.setattr(module,'MAX_METADATA',limit)
    first=lib.document['entries'][0]['id'];lib.start(first)
    assert lib.path.stat().st_size<=limit
    reopened=PreparationLibrary(lib.root);assert reopened.document==lib.document
    archive=tmp_path/'compact.zip';lib.export(archive)
    restored=import_library(archive,tmp_path/'restored')
    assert restored.document==lib.document and len(restored.document['entries'])==4


def test_oversized_metadata_is_rejected_before_an_unreadable_library_is_published(tmp_path,monkeypatch):
    from gpx_recipe.preparation import library as module
    monkeypatch.setattr(module,'MAX_METADATA',1)
    with pytest.raises(ValueError,match='metadata budget'):library(tmp_path)
    assert not (tmp_path/'library').exists()


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


@pytest.mark.parametrize('file_lock_effective',[True,False])
def test_simultaneous_entry_admission_has_one_owner(tmp_path,monkeypatch,file_lock_effective):
    import fcntl
    import threading
    import time
    if not file_lock_effective:monkeypatch.setattr(fcntl,'flock',lambda *args:None)
    lib=library(tmp_path)
    refresh=PreparationLibrary.refresh
    def slow_refresh(self):
        result=refresh(self);time.sleep(.01);return result
    monkeypatch.setattr(PreparationLibrary,'refresh',slow_refresh)
    # Independent views begin from the same saved state. Widen the interval
    # after reading it so a per-instance lock cannot pass by lucky scheduling.
    for row in lib.document['entries']:
        identifier=row['id'];barrier=threading.Barrier(4)
        def start(_):
            instance=PreparationLibrary(lib.root);barrier.wait(timeout=10)
            try:return instance.start(identifier)
            except ValueError:return 'already owned'
        with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(start,range(4)))
        assert sorted(str(v) for v in results)==['1','already owned','already owned','already owned']


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


def test_state_updates_read_only_new_artifact_bytes(tmp_path, monkeypatch):
    from gpx_recipe.preparation import library as module
    lib = library(tmp_path)
    read = []
    original = module.sha
    def counted(path):
        if Path(path).suffix == '.npz':
            read.append(str(Path(path).relative_to(lib.root)))
        return original(path)
    monkeypatch.setattr(module, 'sha', counted)
    ids = [r['id'] for r in lib.document['entries']]
    for identifier in ids:
        prepare_one(lib, identifier)
    # Each new file is hashed for its published checksum and checked once.
    # Later entries do not reread that earlier file during state transitions.
    assert len(read) == 2*len(ids)
    assert all(read.count(f'entries/{identifier}/response.npz') == 2 for identifier in ids)
    lib.prepared_assets(ids[0])
    assert len(read) == 2*len(ids)+1


def test_consumption_and_export_reject_corruption_after_state_only_admission(tmp_path):
    lib = library(tmp_path);ids = [r['id'] for r in lib.document['entries']]
    prepare_one(lib, ids[0])
    np.savez(lib.root/'entries'/ids[0]/'response.npz', coefficient=np.array([5., 9.]))
    # Admission is bookkeeping. It must not consume a different entry's arrays.
    assert lib.start(ids[1]) == 1
    with pytest.raises(ValueError, match='artifact changed'):
        lib.prepared_assets(ids[0])
    with pytest.raises(ValueError, match='artifact changed'):
        lib.export(tmp_path/'corrupt.zip')
    with pytest.raises(ValueError, match='artifact changed'):
        PreparationLibrary(lib.root)


def test_new_file_budget_is_checked_before_reading_output(tmp_path, monkeypatch):
    from gpx_recipe.preparation import library as module
    lib = library(tmp_path);identifier = lib.document['entries'][0]['id']
    attempt = lib.start(identifier)
    folder = lib.root/'entries'/identifier;folder.mkdir(parents=True)
    path = folder/'response.npz';np.savez(path, coefficient=np.ones(1000))
    monkeypatch.setattr(module, 'MAX_BYTES', path.stat().st_size)
    monkeypatch.setattr(module, 'sha', lambda p:pytest.fail('Reject oversized output before hashing it'))
    with pytest.raises(ValueError, match='byte budget'):
        lib.finish(identifier, attempt, state='prepared', files=[str(path.relative_to(lib.root))])
    assert lib.coverage()['counts']['preparing'] == 1


def test_targeted_dependency_checks_reject_changed_inputs(tmp_path):
    lib = library(tmp_path)
    (lib.root/'inputs/parent.cif').write_text('changed after initial validation')
    with pytest.raises(ValueError, match='artifact changed'):
        lib.verify_assets(['inputs/parent.cif'])


def test_prepared_entry_consumption_rechecks_its_captured_parent(tmp_path):
    lib = library(tmp_path);identifier = lib.document['entries'][0]['id']
    prepare_one(lib, identifier)
    (lib.root/'inputs/parent.cif').write_text('changed after compilation')
    with pytest.raises(ValueError, match='artifact changed'):
        lib.prepared_assets(identifier)


def test_state_refresh_checks_manifest_size_before_parsing(tmp_path):
    lib = library(tmp_path)
    lib.path.write_text(' '*(8*1024*1024)+'{}')
    with pytest.raises(ValueError, match='oversized'):
        lib.refresh()


def test_portable_branch_preserves_completed_assets_and_does_not_claim_originating_jobs(tmp_path):
    lib=library(tmp_path);ids=[r['id'] for r in lib.document['entries']]
    prepare_one(lib,ids[0]);attempt=lib.start(ids[1])
    packet=tmp_path/'portable.zip';lib.export(packet)
    branch=import_library(packet,tmp_path/'branch')
    assert branch.detach_imported_work()==1
    assert branch.row(ids[0])['files']==lib.row(ids[0])['files']
    assert branch.row(ids[1])['state']=='pending' and branch.row(ids[1])['attempt']==attempt
    assert branch.start(ids[1])==attempt+1
    assert lib.row(ids[1])['state']=='preparing' and lib.row(ids[1])['attempt']==attempt
    (branch.root/'queue').mkdir();(branch.root/'queue/queue.json').write_text('{}')
    with pytest.raises(ValueError,match='recover owned jobs'):
        branch.detach_imported_work()
