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
