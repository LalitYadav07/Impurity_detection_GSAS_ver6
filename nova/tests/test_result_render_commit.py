"""A canonical render must replace a fit preview in one coherent UI commit."""
import asyncio
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
from trame.app import get_server

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import ResultStatus, RunStatus
from test_fit_first_opening import setup_preview, until


def ready_preview(tmp_path, monkeypatch):
    server = get_server('render-commit-' + uuid.uuid4().hex, client_type='vue3')
    monkeypatch.setattr('radar_pd_nova.app.get_server', lambda *a, **k: server)
    app, row, preview, path, entered, release, calls = setup_preview(tmp_path)
    app._sync_runs = RadarPdNovaApp._sync_runs.__get__(app)
    # Match the real wrapper's split JSON/NPZ names and absolute producer-path
    # metadata, while retaining only the synthetic fixture's numeric values.
    for root in (preview.root, tmp_path / 'canonical'):
        old = root / 'plots' / 'stage.plotdata.json'
        payload = json.loads(old.read_text())
        arrays = payload.pop('arrays')
        image = 'Results__Plots__seq_final_joint_accepted_model.png'
        companion = image + '.plotdata_arrays.npz'
        np.savez_compressed(old.parent / companion, **arrays)
        payload.update(source_plot=image, arrays_npz=companion, array_keys=sorted(arrays),
            source_plot_path='/data/staging_test/synthetic/working/run/Results/Plots/seq_final_joint_accepted_model.png')
        (old.parent / (image + '.plotdata.json')).write_text(json.dumps(payload))
        old.unlink()
    # Full publications add CSVs that the intentionally small preview lacks.
    (tmp_path / 'canonical' / 'final_phase_fractions.csv').write_text(
        'phase,weight_fraction\nSynthetic,1.0\n')
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview', lambda *_: preview)
    server.state.ready()
    return app, row, preview, entered, release, calls


def test_ready_csv_promotion_never_flushes_an_empty_result_owner(tmp_path, monkeypatch):
    app, row, preview, entered, release, calls = ready_preview(tmp_path, monkeypatch)
    state = app.server.state
    commits = []
    previous_push = state._push_state_fn
    def capture(changes):
        commits.append(deepcopy(changes))
        if previous_push:
            previous_push(changes)
    object.__setattr__(state, '_push_state_fn', capture)

    async def run():
        app._select_record(row); app._open_record_results(row)
        try:
            await until(lambda: state.result_preview)
            assert state.result_owner_uid == row.uid and state.result_actions_pending
            commits.clear()
        finally:
            release.set()
            await asyncio.gather(*list(app._history_result_requests.values()))
        await asyncio.sleep(0)
        state.flush()
        assert commits and all(change.get('result_owner_uid', row.uid) == row.uid for change in commits)
        assert state.result_owner_uid == state.selected_run_uid == row.uid
        assert app._rendered_result_current(app.records[row.uid])
        assert not state.result_preview and not state.result_actions_pending
        assert state.plot_options and state.table_options and state.table_rows and state.file_groups
        assert state.table_rows[0]['phase'] == 'Synthetic'
        assert app._result_actions_ready(row.uid)
        assert app.records[row.uid].result_status == ResultStatus.READY
        assert calls == [row.uid]
    asyncio.run(run())


@pytest.mark.parametrize('failure', ['view', 'table', 'widget'])
def test_failed_canonical_preparation_keeps_preview_and_its_warnings(tmp_path, monkeypatch, failure):
    app, row, preview, entered, release, _ = ready_preview(tmp_path, monkeypatch)
    state = app.server.state

    async def run():
        app._select_record(row); app._open_record_results(row)
        try:
            await until(lambda: state.result_preview)
            state.result_warnings = ['Existing fit review warning']
            state.flush()
            def broken(*a, **k):
                raise ValueError('synthetic canonical ' + failure + ' failure')
            if failure == 'widget':
                app._primary_plot_widget = SimpleNamespace(update=broken)
            else:
                target = 'build_result_view' if failure == 'view' else 'read_table'
                monkeypatch.setattr('radar_pd_nova.app.' + target, broken)
        finally:
            release.set()
            await asyncio.gather(*list(app._history_result_requests.values()))
        await asyncio.sleep(0)
        assert state.result_owner_uid == row.uid and state.result_preview
        assert state.result_warnings == ['Existing fit review warning']
        assert state.plot_options and not state.file_groups and not state.checkpoint_rows
        assert state.result_actions_pending and 'incomplete' in state.result_readiness_message
        assert not app._result_actions_ready(row.uid)
        assert 'synthetic canonical' in state.selected_result_load_error
        # Scientific and canonical source authority do not roll back with a view failure.
        assert app.records[row.uid].result_status == ResultStatus.READY
        assert app.records[row.uid].output_dir.endswith('canonical')
        assert row.output_dir is None and app.records[row.uid].status == RunStatus.OK
    asyncio.run(run())


def test_widget_flush_cannot_restore_old_selection_after_canonical_commit(tmp_path, monkeypatch):
    app, row, preview, entered, release, _ = ready_preview(tmp_path, monkeypatch)
    state = app.server.state
    other = row.model_copy(deep=True, update={'uid':'B', 'galaxy_job_id':'B',
        'status':RunStatus.RUNNING, 'output_dataset_ids':{'summary':'summary-B'}})
    app.records[other.uid] = other
    updates = []
    armed = []
    class Widget:
        def update(self, figure):
            updates.append(state.result_owner_uid)
            if armed:
                # Model an event dispatched by a real widget/state flush.
                app._run_selection_changed([other.uid])
    app._primary_plot_widget = Widget()
    app._plot_widget = SimpleNamespace(update=lambda _: updates.append('gallery'))

    async def run():
        app._select_record(row); app._open_record_results(row)
        try:
            await until(lambda: state.result_preview)
            updates.clear()
            armed.append(True)
        finally:
            release.set()
            await asyncio.gather(*list(app._history_result_requests.values()))
        await asyncio.sleep(0)
        assert state.selected_run_uid == other.uid and state.workspace_view == 'monitor'
        assert state.result_owner_uid == '' and not state.result_preview
        assert updates == [row.uid]  # no late gallery from the old render
        assert app.records[row.uid].result_status == ResultStatus.READY
        assert app.records[other.uid].output_dir is None
    asyncio.run(run())


@pytest.mark.parametrize('opening', ['first-canonical', 'forced-reload'])
def test_complete_fit_renders_without_a_preview_and_on_forced_reload(tmp_path, monkeypatch, opening):
    app, row, preview, entered, release, calls = ready_preview(tmp_path, monkeypatch)
    state = app.server.state
    monkeypatch.setattr('radar_pd_nova.result_preview.collect_result_preview', lambda *_: None)
    figures = []
    app._primary_plot_widget = SimpleNamespace(update=lambda figure: figures.append(figure))
    app._plot_widget = SimpleNamespace(update=lambda figure: figures.append(figure))
    if opening == 'forced-reload':
        row = row.model_copy(deep=True, update={'output_dir':str(tmp_path/'canonical'),
                                              'result_status':ResultStatus.READY})
        app.records[row.uid] = row
        app._select_record(row)
        app._load_results(row, local_only=True)
        assert state.result_owner_uid == row.uid
        figures.clear()

    async def run():
        release.set()
        app._select_record(row)
        app._open_record_results(row, force=opening == 'forced-reload')
        await asyncio.gather(*list(app._history_result_requests.values()))
        await asyncio.sleep(0)
        assert state.result_owner_uid == state.selected_run_uid == row.uid
        assert app._rendered_result_current(app.records[row.uid])
        assert not state.result_preview and not state.result_actions_pending
        assert state.plot_options and state.table_rows and state.file_groups
        assert app._result_actions_ready(row.uid) and not state.selected_run_loading
        assert len(figures) == 2
        for figure in figures:
            assert {'Observed','Calculated','Difference'} <= {trace.name for trace in figure.data}
        assert calls == [row.uid]
    asyncio.run(run())
