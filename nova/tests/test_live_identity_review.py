"""Full-source identity evidence must survive narrower Live eligibility checks."""
import asyncio
from copy import deepcopy
import threading

import pytest

from test_live_setup import live  # real app + local, no-submission service
from radar_pd_nova.sample_identity_review import identity_review_rows
from radar_pd_nova.powgen_samples import select_runs
from radar_pd_nova.powgen_watch import WatchedRun


def five_runs():
    rows = []
    for number in (47531, 47541, 47551, 47552, 47553):
        conflict = number < 47551
        rows.append(dict(run_number=number, status='conflict' if conflict else 'resolved',
            sample_id='' if conflict else '70730', verification='source_file',
            sources=([{'kind':'catalog', 'value':'70730'}, {'kind':'in_acquisition_log', 'value':'70780'}]
                     if conflict else [{'kind':'source_file', 'value':'70730'}]),
            actions=[{'message':'Review the conflicting recorded IDs; no automatic override.'}] if conflict else []))
    return rows


@pytest.fixture
def actual_shape(live, monkeypatch):
    state = live.server.state
    state.powgen_ipts = '25921'; state.powgen_processing_sample_ids = ['70730']
    rows = five_runs(); calls = []
    monkeypatch.setattr('radar_pd_nova.app.discover_powgen_samples', lambda *a, **k: {
        'metadata_resolution':deepcopy(rows), 'metadata_notices':['Two source identities conflict.']})
    def preflight(*args, **kwargs):
        calls.append(kwargs)
        selected = rows[2:] if kwargs['selected_sample_ids'] else rows
        return dict(ready=True, scan_count=len(selected), total_scan_count=5, excluded_count=5-len(selected),
            available_banks=[1], analysis_count=len(selected), message=f'Ready: {len(selected)} scans',
            metadata_resolution=deepcopy(selected), profile_filename='official.instprm')
    monkeypatch.setattr('radar_pd_nova.app.preflight_powgen_experiment', preflight)
    return live, rows, calls


def test_selected_cold_scans_keep_both_excluded_warm_conflicts_visible(actual_shape):
    app, original, calls = actual_shape; state = app.server.state
    assert asyncio.run(app.check_powgen_experiment()) is True
    assert calls[0]['selected_sample_ids'] == ('70730',)
    assert [r['run_number'] for r in state.powgen_metadata_resolution] == [47551,47552,47553]
    assert [r['run_number'] for r in state.powgen_identity_review_rows] == [47531,47541,47551,47552,47553]
    assert all('Excluded' in r['selection_note'] and 'not enrolled' in r['selection_note'] for r in state.powgen_identity_review_rows[:2])
    assert state.powgen_identity_review_rows[0]['sources'] == original[0]['sources']
    assert all('In checked selection' in r['selection_note'] for r in state.powgen_identity_review_rows[2:])
    assert not app.service.jobs and not app.service.saved and not app.service.uploads
    assert 'selection_note' not in original[0]


def test_all_sample_guard_still_blocks_but_full_evidence_stays_visible(actual_shape):
    app, _, _ = actual_shape; state = app.server.state
    state.powgen_processing_scope = 'all'
    assert asyncio.run(app.check_powgen_experiment()) is False
    assert not state.powgen_preflight_ready
    assert 'conflicting identities' in state.powgen_preflight_message
    assert len(state.powgen_identity_review_rows) == 5
    assert [r['status'] for r in state.powgen_identity_review_rows[:2]] == ['conflict','conflict']
    assert not app.service.jobs and not app.service.saved and not app.service.uploads


@pytest.mark.parametrize('field,value', [('powgen_ipts','25922'),('powgen_source_subfolder','shared/autoreduce/2025A'),('live_beamline','NOM')])
def test_source_change_clears_old_evidence(actual_shape, field, value):
    app, _, _ = actual_shape; state = app.server.state
    assert asyncio.run(app.check_powgen_experiment())
    setattr(state,field,value); app._powgen_processing_changed()
    assert state.powgen_identity_review_rows == []
    assert state.powgen_identity_review_notices == []


def test_sample_filter_change_keeps_source_conflicts_but_removes_old_check_annotations(actual_shape):
    app, _, _ = actual_shape; state = app.server.state
    assert asyncio.run(app.check_powgen_experiment())
    state.powgen_processing_sample_ids = ['different']; app._powgen_processing_changed()
    assert len(state.powgen_identity_review_rows) == 5
    assert all('In checked selection' not in r['selection_note'] for r in state.powgen_identity_review_rows)
    assert not state.powgen_preflight_ready


@pytest.mark.parametrize('change', ['ipts','folder','history','service'])
def test_late_inventory_cannot_publish_into_new_context(actual_shape, monkeypatch, change):
    app, rows, _ = actual_shape; state = app.server.state
    entered, release = threading.Event(), threading.Event()
    def discover(*args, **kwargs):
        entered.set(); assert release.wait(5)
        return {'metadata_resolution':rows}
    monkeypatch.setattr('radar_pd_nova.app.discover_powgen_samples', discover)
    async def run():
        task = asyncio.create_task(app.check_powgen_experiment())
        assert await asyncio.to_thread(entered.wait,5)
        if change == 'ipts': state.powgen_ipts = '99999'
        elif change == 'folder': state.powgen_source_subfolder = 'shared/autoreduce/old'
        elif change == 'history': app.service.history_id = 'another-history'
        else: app.service = type(app.service)()
        release.set(); assert await task is False
    asyncio.run(run())
    assert state.powgen_identity_review_rows == []
    assert not state.powgen_preflight_ready


def test_unknowns_prioritized_without_inventing_resolution():
    rows = [dict(run_number=i,status='resolved',sample_id='A') for i in range(150)]
    rows.append(dict(run_number=999,status='unidentified',sample_id='',sources=[{'value':'missing'}]))
    out = identity_review_rows(rows, [{'run_number':999,'status':'resolved','sample_id':'B'}], checked=True)
    assert out[0]['run_number'] == 999 and out[0]['status'] == 'unidentified' and not out[0]['sample_id']
    assert 'Excluded' in out[0]['selection_note'] and rows[-1].get('selection_note') is None


def test_old_refresh_cannot_replace_newer_same_source_checked_evidence(actual_shape, monkeypatch):
    app, rows, _ = actual_shape; state = app.server.state
    app._powgen_form_ipts = 'IPTS-25921'; app._powgen_form_source = 'shared/autoreduce'
    entered, release = threading.Event(), threading.Event(); count = 0
    def discover(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            entered.set(); assert release.wait(5)
            return {'options': [], 'run_options': [], 'scan_count':5, 'metadata_resolution': []}
        return {'metadata_resolution':deepcopy(rows)}
    monkeypatch.setattr('radar_pd_nova.app.discover_powgen_samples', discover)
    async def run():
        old = asyncio.create_task(app.refresh_powgen_samples())
        assert await asyncio.to_thread(entered.wait,5)
        assert await app.check_powgen_experiment() is True
        before = deepcopy(state.powgen_identity_review_rows)
        release.set(); assert await old is None
        assert state.powgen_identity_review_rows == before
        assert state.powgen_preflight_ready
    asyncio.run(run())


def test_failed_same_source_refresh_does_not_keep_old_evidence_or_ready_claim(actual_shape, monkeypatch):
    app, _, _ = actual_shape; state = app.server.state
    assert asyncio.run(app.check_powgen_experiment())
    app._powgen_form_ipts = 'IPTS-25921'; app._powgen_form_source = 'shared/autoreduce'
    def fail(*args, **kwargs): raise OSError('Synthetic source unavailable')
    monkeypatch.setattr('radar_pd_nova.app.discover_powgen_samples', fail)
    assert asyncio.run(app.refresh_powgen_samples()) is None
    assert not state.powgen_preflight_ready
    assert state.powgen_identity_review_rows == [] and state.powgen_identity_review_notices == []
    assert 'Synthetic source unavailable' in state.powgen_sample_message


def test_old_refresh_progress_cannot_overwrite_newer_check_message(actual_shape, monkeypatch):
    app, rows, _ = actual_shape; state = app.server.state
    app._powgen_form_ipts = 'IPTS-25921'; app._powgen_form_source = 'shared/autoreduce'
    entered, release = threading.Event(), threading.Event(); callbacks = []
    def discover(*args, **kwargs):
        if 'progress' in kwargs:
            callbacks.append(kwargs['progress']); entered.set(); assert release.wait(5)
            return {'options':[], 'run_options':[], 'scan_count':5, 'metadata_resolution':rows}
        return {'metadata_resolution':deepcopy(rows)}
    monkeypatch.setattr('radar_pd_nova.app.discover_powgen_samples', discover)
    async def run():
        task = asyncio.create_task(app.refresh_powgen_samples())
        assert await asyncio.to_thread(entered.wait,5)
        assert await app.check_powgen_experiment()
        state.powgen_sample_message = 'Newer check retained'
        callbacks[0](1,999); await asyncio.sleep(0)
        assert state.powgen_sample_message == 'Newer check retained'
        release.set(); await task
    asyncio.run(run())


def test_original_selection_still_contains_only_three_verified_cold_runs(tmp_path):
    records = five_runs(); by_number = {r['run_number']:r for r in records}; runs = []
    for row in records:
        path = tmp_path/f"PG3_{row['run_number']}.gsa";path.write_text('# sample\n')
        runs.append(WatchedRun(row['run_number'],str(path),str(row['run_number'])))
    class Index:
        def identity(self, run):
            row = by_number[run.run_number]
            return {'sample_id':row['sample_id'], 'identity_status':row['status']}
    selected = select_runs(runs, Index(), ('70730',))
    assert [r.run_number for r in selected] == [47551,47552,47553]


def test_ui_uses_full_inventory_and_marks_excluded_rows():
    from pathlib import Path
    source = (Path(__file__).parents[1]/'src/radar_pd_nova/app.py').read_text()
    assert "live_beamline === 'PG3' ? powgen_identity_review_rows : powgen_metadata_resolution" in source
    assert 'Displayed excluded runs are not enrolled' in source
    assert 'row.selection_note' in source
    assert "live_beamline !== 'PG3' && powgen_metadata_resolution.length > 100" in source
