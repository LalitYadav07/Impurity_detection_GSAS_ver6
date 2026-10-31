"""A Save As must publish the chosen GPX, not the original or newest filename."""
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import zipfile
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'gsasii_gui/scripts'
spec = importlib.util.spec_from_file_location('project_tracking', SCRIPTS/'project_tracking.py')
tracking = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracking)


def gpx(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        pickle.dump([['Controls', {'label': label}]], stream, protocol=2)
        pickle.dump([['Notebook', [label]]], stream, protocol=2)
    return path.read_bytes()


@pytest.fixture
def workspace(tmp_path):
    session = tmp_path/'session'
    source = session/'radar_pd_project.gpx'
    gpx(source, 'original')
    tracker = tracking.ProjectTracker(session, tmp_path/'galaxy/edited.gpx', session/'Exports')
    tracker.activate(source)
    return tracker, source


def test_save_as_primary_and_archive_follow_actual_save_even_when_original_is_newer(workspace):
    tracker, original = workspace
    named = original.parent/'my refined project.gpx'
    expected = gpx(named, 'refined')
    tracker.activate(named)
    gpx(original, 'a later unselected write')
    os.utime(original, (2_000_000_000, 2_000_000_000))
    tracker.sync()
    assert tracker.output.read_bytes() == expected
    index = json.loads(tracker.index.read_text())
    assert {entry['name'] for entry in index['projects'].values()} == {original.name, named.name}
    current = index['projects'][index['active']]
    assert current['name'] == named.name
    assert (tracker.index.parent/current['file']).read_bytes() == expected


def test_active_project_updates_after_native_refinement_without_another_save_as(workspace):
    tracker, original = workspace
    named = original.parent/'new.gpx'
    gpx(named, 'before refinement'); tracker.activate(named)
    expected = gpx(named, 'after native refinement')
    assert tracker.sync()
    assert tracker.output.read_bytes() == expected
    before = tracker.output.stat().st_mtime_ns
    assert tracker.sync() is False and tracker.output.stat().st_mtime_ns == before


def test_same_basename_saved_in_distinct_directories_keeps_both_branches(workspace):
    tracker, original = workspace
    for folder in ['first', 'second']:
        path = original.parent/folder/'result.gpx'
        gpx(path, folder); tracker.activate(path)
    entries = list(json.loads(tracker.index.read_text())['projects'].values())
    assert len(entries) == 3 and len({entry['file'] for entry in entries}) == 3
    assert all('/' not in entry['file'] for entry in entries)


def test_hidden_or_unicode_project_names_have_visible_collision_safe_archive_members(workspace):
    tracker, original = workspace
    branch = original.parent/'.refined α sample.gpx'
    expected = gpx(branch, 'named branch'); tracker.activate(branch)
    index = json.loads(tracker.index.read_text())
    item = index['projects'][index['active']]
    assert item['name'] == branch.name and not item['file'].startswith('.')
    assert (tracker.index.parent/item['file']).read_bytes() == expected


@pytest.mark.parametrize('damage', ['truncated', 'trailing', 'empty', 'symlink'])
def test_partial_or_unsafe_active_file_never_replaces_last_verified_output(workspace, damage):
    tracker, source = workspace
    before = tracker.output.read_bytes()
    if damage == 'truncated': source.write_bytes(before[:-1])
    elif damage == 'trailing': source.write_bytes(before+b'not a GPX record')
    elif damage == 'empty': source.write_bytes(b'')
    else:
        target = source.with_name('another.gpx'); gpx(target, 'another')
        source.unlink(); source.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        tracker.sync()
    assert tracker.output.read_bytes() == before
    assert not list(tracker.output.parent.glob('.edited.gpx.*'))


def test_gpx_validation_never_executes_pickle_global_operations(tmp_path):
    marker = tmp_path/'must-not-exist'
    class Payload:
        def __reduce__(self):
            return (os.system, ('touch '+str(marker),))
    path = tmp_path/'payload.gpx'
    path.write_bytes(pickle.dumps(Payload(), protocol=2))
    tracking.validate_gpx(path)
    assert not marker.exists()


@pytest.mark.parametrize('damage', ['missing', 'invalid_json', 'relative_path', 'wrong_schema'])
def test_invalid_active_pointer_does_not_fall_back_to_original_filename(workspace, damage):
    tracker, source = workspace
    before = tracker.output.read_bytes()
    gpx(source, 'new original that must not be substituted')
    if damage == 'missing': tracker.active.unlink()
    elif damage == 'invalid_json': tracker.active.write_text('{bad')
    else:
        data = json.loads(tracker.active.read_text())
        data['path' if damage == 'relative_path' else 'schema'] = 'unexpected'
        tracker.active.write_text(json.dumps(data))
    with pytest.raises((ValueError, OSError)):
        tracker.sync()
    assert tracker.output.read_bytes() == before


def test_file_change_during_copy_never_publishes_partial_output(workspace, monkeypatch):
    tracker, source = workspace; before = tracker.output.read_bytes()
    gpx(source, 'new value')
    original_copy = tracking.shutil.copyfileobj
    def change(stream, target):
        original_copy(stream, target)
        with source.open('ab') as changed: changed.write(b'changed during copy')
    monkeypatch.setattr(tracking.shutil, 'copyfileobj', change)
    with pytest.raises(ValueError, match='changed while copying'): tracker.sync()
    assert tracker.output.read_bytes() == before


def test_successful_native_save_callback_changes_active_project_only_after_save(workspace):
    tracker, source = workspace; branch = source.parent/'branch.gpx'; events = []
    def save(frame): events.append('native-save'); gpx(Path(frame.GSASprojectfile), 'branch'); return 'native-return'
    io = SimpleNamespace(ProjFileSave=save)
    tracking.install_save_tracking(io, tracker)
    tracking.install_save_tracking(io, tracker)  # No duplicate wrapper/callback.
    assert io.ProjFileSave(SimpleNamespace(GSASprojectfile=str(branch))) == 'native-return'
    assert events == ['native-save'] and tracker.output.read_bytes() == branch.read_bytes()


@pytest.mark.parametrize('failure', ['native_exception', 'native_permission_return', 'empty_tree'])
def test_failed_or_cancelled_native_save_does_not_select_a_different_project(workspace, failure):
    tracker, source = workspace; before = tracker.output.read_bytes(); pointer = tracker.active.read_bytes()
    target = source.parent/'existing.gpx'; gpx(target, 'old unrelated file')
    def save(frame):
        if failure == 'native_exception': raise PermissionError('Save failed')
        return None  # Upstream PermissionError/empty-tree paths return normally.
    io = SimpleNamespace(ProjFileSave=save)
    tracking.install_save_tracking(io, tracker)
    frame = SimpleNamespace(GSASprojectfile=str(target))
    if failure == 'native_exception':
        with pytest.raises(PermissionError): io.ProjFileSave(frame)
    else: io.ProjFileSave(frame)
    assert tracker.active.read_bytes() == pointer and tracker.output.read_bytes() == before


def test_preservation_failure_is_visible_without_changing_native_save_return(workspace, monkeypatch):
    tracker, source = workspace; messages = []
    def save(frame): gpx(Path(frame.GSASprojectfile), 'saved'); return True
    io = SimpleNamespace(ProjFileSave=save)
    def unavailable(path): raise OSError('Output volume unavailable')
    monkeypatch.setattr(tracker, 'activate', unavailable)
    tracking.install_save_tracking(io, tracker, report_error=lambda frame, message: messages.append(message))
    assert io.ProjFileSave(SimpleNamespace(GSASprojectfile=str(source))) is True
    assert len(messages) == 1 and 'could not be updated' in messages[0]


def test_corrupt_archive_index_cannot_publish_new_primary_without_its_provenance(workspace):
    tracker, source = workspace; before = tracker.output.read_bytes()
    gpx(source, 'new'); tracker.index.write_text('{broken')
    with pytest.raises(ValueError): tracker.sync()
    assert tracker.output.read_bytes() == before


def test_archive_snapshot_waits_for_consistent_active_project_and_index(workspace, monkeypatch):
    tracker, original = workspace; branch = original.parent/'branch.gpx'; gpx(branch, 'branch')
    spec = importlib.util.spec_from_file_location('save_snapshot', SCRIPTS/'snapshot_exports.py')
    snapshots = importlib.util.module_from_spec(spec); spec.loader.exec_module(snapshots)
    monkeypatch.setitem(sys.modules, 'project_tracking', tracking)
    monkeypatch.setenv('GSASII_SESSION_DIR', str(tracker.session))
    writing, release, reading, archive_entered = [threading.Event() for _ in range(4)]
    original_copy = tracking.stable_copy
    def copy(source, destination):
        if destination == tracker.output:
            writing.set(); assert release.wait(3)
        return original_copy(source, destination)
    monkeypatch.setattr(tracking, 'stable_copy', copy)
    original_manifest = snapshots._manifest
    def manifest(root): archive_entered.set(); return original_manifest(root)
    monkeypatch.setattr(snapshots, '_manifest', manifest)
    archive = tracker.session/'archive.zip'
    def read():
        reading.set()
        return snapshots.snapshot(tracker.exports, archive, tracker.session/'export-index.json')
    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(tracker.activate, branch); assert writing.wait(3)
        reader = pool.submit(read); assert reading.wait(3)
        assert not archive_entered.wait(.05)
        release.set(); writer.result(3); assert reader.result(3)
    with zipfile.ZipFile(archive) as bundle:
        index = json.loads(bundle.read('Saved_projects/index.json'))
        item = index['projects'][index['active']]
        assert item['name'] == branch.name
        assert bundle.read('Saved_projects/'+item['file']) == branch.read_bytes()
        assert not any(any(part.startswith('.') for part in Path(name).parts) for name in bundle.namelist())
