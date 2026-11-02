"""An archive review and its digest must describe the same unchanged file."""
import asyncio
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import zipfile

import pytest

from radar_pd_nova import library_setup
from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.library_setup import LibraryDraft, library_source_key
from radar_pd_nova.models import RunStatus, UtilityActionRecord


def archive(path: Path, failed_source: str) -> Path:
    with zipfile.ZipFile(path, 'w') as output:
        output.writestr('pack/manifest.json', json.dumps({'n_phases': 1, 'phase_ids': ['kept'],
            'failures': [{'source_name': failed_source, 'error': 'Source rejected'}]}))
    return path


def test_unchanged_same_name_archive_has_exact_review_hash(tmp_path):
    path = archive(tmp_path / 'library.zip', 'missing.cif')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    first = library_setup.inspect_library_review(path)
    second = library_setup.inspect_library_review(path)
    assert first == second and first['archive_sha256'] == expected
    assert first['failures'][0]['source_name'] == 'missing.cif'


@pytest.mark.parametrize('mutation', ['replace_before_hash', 'replace_after_hash', 'in_place', 'truncate', 'same_size_restore_mtime'])
def test_archive_changes_between_manifest_and_return_fail_closed(tmp_path, monkeypatch, mutation):
    path = archive(tmp_path / 'library.zip', 'old-impurity.cif')
    replacement = archive(tmp_path / 'replacement.zip', 'new-mainphase.cif')
    original = library_setup._hash_library_stream
    before = path.stat()
    def change(stream):
        if mutation == 'replace_after_hash':
            result = original(stream)
            replacement.replace(path)
            return result
        if mutation == 'replace_before_hash': replacement.replace(path)
        elif mutation == 'truncate':
            with path.open('r+b') as target: target.truncate(32)
        elif mutation == 'same_size_restore_mtime':
            raw = bytearray(path.read_bytes()); raw[-1] ^= 1
            path.write_bytes(raw)
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        else: path.write_bytes(replacement.read_bytes())
        return original(stream)
    monkeypatch.setattr(library_setup, '_hash_library_stream', change)
    with pytest.raises(ValueError, match='changed during review'):
        library_setup.inspect_library_review(path)


def test_unlink_after_hash_is_rejected(tmp_path, monkeypatch):
    path = archive(tmp_path / 'library.zip', 'missing.cif')
    original = library_setup._hash_library_stream
    def unlink(stream):
        result = original(stream); path.unlink(); return result
    monkeypatch.setattr(library_setup, '_hash_library_stream', unlink)
    with pytest.raises(ValueError, match='changed during review'):
        library_setup.inspect_library_review(path)


def test_failed_refresh_preserves_the_previous_selection_and_permission(tmp_path, monkeypatch):
    path = archive(tmp_path / 'library.zip', 'missing.cif')
    replacement = archive(tmp_path / 'replacement.zip', 'new-main.cif')
    app = RadarPdNovaApp(); draft = LibraryDraft(app.server.state)
    draft.database_source = 'archive'; draft.library_archive_source = 'computer'
    draft.database_archive_path = str(path)
    preview = library_setup.inspect_library_review(path)
    draft.library_review = {**library_setup.activation_for_review(preview), 'source_key': library_source_key(draft)}
    draft.library_review_preview = {**preview, 'source_key': library_source_key(draft)}
    saved = deepcopy(draft.library_review)
    original = library_setup._hash_library_stream
    def replace_after(stream):
        result = original(stream); replacement.replace(path); return result
    monkeypatch.setattr(library_setup, '_hash_library_stream', replace_after)
    pending = []
    app._schedule_utility = lambda work, name: pending.append(work)
    app.review_candidate_library()
    asyncio.run(pending[0])
    assert draft.database_archive_path == str(path) and draft.database_source == 'archive'
    assert draft.library_review == saved
    assert 'changed during review' in draft.library_review_message
    assert not draft.library_review_loading


def failed_build(app, context):
    draft = LibraryDraft(app.server.state, context)
    draft.library_builder_active = False
    draft.library_builder_status = 'error'
    draft.library_builder_message = 'Earlier Galaxy library build failed'
    draft.library_builder_progress = 0
    action = UtilityActionRecord(uid='failed-' + context, tool_id='builder', name='Earlier build',
                                 galaxy_job_id='old-job', status=RunStatus.ERROR,
                                 message=draft.library_builder_message)
    app.utility_actions[action.uid] = action
    return draft, action


@pytest.mark.parametrize('context', ['single', 'live'])
@pytest.mark.parametrize('mode', ['archive', 'builtin'])
def test_selecting_existing_library_clears_only_its_inactive_builder_error(context, mode):
    app = RadarPdNovaApp()
    app.server.state.ready()
    draft = LibraryDraft(app.server.state, context)
    other_context = 'live' if context == 'single' else 'single'
    other = LibraryDraft(app.server.state, other_context)
    draft.database_source = other.database_source = 'custom_mini'
    app.server.state.flush()
    draft, action = failed_build(app, context)
    other, _ = failed_build(app, other_context)
    draft.database_source = mode
    draft.flush()

    assert draft.database_source == mode
    assert draft.library_builder_status == 'idle'
    assert draft.library_builder_message == ''
    assert other.library_builder_status == 'error'
    assert other.library_builder_message == action.message
    assert app.utility_actions[action.uid].message == 'Earlier Galaxy library build failed'


@pytest.mark.parametrize('context', ['single', 'live'])
def test_successful_archive_review_drops_stale_build_error_and_keeps_job_diagnostic(tmp_path, context):
    path = tmp_path / 'portable-library.zip'
    with zipfile.ZipFile(path, 'w') as output:
        output.writestr('pack/manifest.json', json.dumps({'n_phases': 1, 'phase_ids': ['kept'], 'failures': []}))
    app = RadarPdNovaApp()
    app.server.state.ready()
    draft = LibraryDraft(app.server.state, context)
    draft.database_source = 'archive'
    draft.library_archive_source = 'computer'
    draft.database_archive_path = str(path)
    draft.flush()
    draft, action = failed_build(app, context)
    pending = []
    app._schedule_utility = lambda work, name: pending.append(work)
    app.review_candidate_library(context)
    asyncio.run(pending[0])

    assert draft.database_archive_path == str(path)
    assert not draft.library_review_preview['requires_review']
    assert 'declares no source failures' in draft.library_review_message
    assert draft.library_builder_status == 'idle'
    assert draft.library_builder_message == ''
    assert app.utility_actions[action.uid].message == 'Earlier Galaxy library build failed'


def test_mode_change_does_not_clear_a_build_that_is_still_active():
    app = RadarPdNovaApp()
    app.server.state.ready()
    draft = LibraryDraft(app.server.state)
    draft.database_source = 'custom_mini'
    draft.flush()
    draft, _ = failed_build(app, 'single')
    draft.library_builder_active = True
    draft.database_source = 'archive'
    draft.flush()
    assert draft.library_builder_status == 'error'
    assert draft.library_builder_message == 'Earlier Galaxy library build failed'
