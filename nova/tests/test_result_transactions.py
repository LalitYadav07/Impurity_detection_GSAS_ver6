"""Result reads, reloads and publication stay serialized across UI entry points."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import AnalysisMode, InputSelection, ResultStatus, RunRecord, RunStatus


def _app():
    app = RadarPdNovaApp()
    rows = [RunRecord(uid=uid, galaxy_job_id=uid, history_id="history",
                      name=f"Saved {uid}", mode=AnalysisMode.FULL,
                      status=RunStatus.OK, output_dataset_ids={"summary": uid})
            for uid in ("A", "B")]
    app.records = {row.uid: row for row in rows}
    app._sync_runs = lambda: None
    loaded = []
    app._load_results = lambda row: loaded.append(row.uid)
    return app, rows, loaded


def test_reselected_run_shares_inflight_publication():
    app, (a, _), loaded = _app()
    entered, second_entered, release = (threading.Event() for _ in range(3))
    publication_objects = []
    lock = threading.Lock()

    def collect(row):
        row.output_dir = "/tmp/mock-result-" + row.uid
        return row

    def publish(row, **_):
        if row.uid == a.uid:
            with lock:
                publication_objects.append(id(row))
                first = len(publication_objects) == 1
            if first:
                entered.set()
                assert release.wait(5)
            else:
                second_entered.set()
        return row

    app.service = SimpleNamespace(collect_results=collect)
    app._publish_results_if_requested = publish

    async def scenario():
        app._run_selection_changed(["A"])
        assert await asyncio.to_thread(entered.wait, 2)
        first = app._history_result_requests["A"]
        try:
            app._run_selection_changed(["B"])
            app._run_selection_changed(["A"])
            # A second publisher starts while the first one is still blocked.
            # A fixed implementation can share the first operation instead.
            await asyncio.to_thread(second_entered.wait, 1)
        finally:
            release.set()
            await asyncio.gather(first, *list(app._history_result_requests.values()))
        assert loaded == ["A"]
        assert len(publication_objects) == 1, publication_objects

    asyncio.run(scenario())


def test_saved_request_does_not_inherit_obsolete_live_controller_guard():
    app, (a, b), loaded = _app()
    app._powgen_controller = SimpleNamespace(records={"live-B": b})
    entered, release = threading.Event(), threading.Event()
    calls = []

    def collect(row):
        calls.append(row.uid)
        if row.uid == a.uid:
            entered.set()
            assert release.wait(5)
        row.output_dir = "/tmp/mock-result-" + row.uid
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 2)
        first = app._history_result_requests[a.uid]
        app.server.state.powgen_selected_run_id = "live-B"
        live = asyncio.create_task(app.open_powgen_selected_run_async())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        try:
            # Changing a stopped Live monitor's beamline clears its controller,
            # while previously recovered runs remain in saved history.
            app._powgen_controller = None
            app._run_selection_changed([a.uid])
            app._run_selection_changed([b.uid])
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert app._history_result_downloads[b.uid][4]["waiters"] == 2
        finally:
            release.set()
            await asyncio.gather(first, live, *list(app._history_result_requests.values()))
        assert loaded == [b.uid], (calls, loaded, app.server.state.selected_result_load_error)

    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["refresh_selected_run", "reload_selected_results"])
def test_refresh_and_reload_share_the_running_archive_read(method):
    app, (a, _), _ = _app()
    entered, release = threading.Event(), threading.Event()
    calls = []
    lock = threading.Lock()
    active = 0
    max_active = 0

    def collect(row, **kwargs):
        nonlocal active, max_active
        with lock:
            calls.append((row.uid, kwargs.get("force", False)))
            first = len(calls) == 1
            active += 1
            max_active = max(max_active, active)
        try:
            if first:
                entered.set()
                assert release.wait(5)
            row.output_dir = "/tmp/mock-result-" + row.uid
            row.result_status = ResultStatus.READY
            return row
        finally:
            with lock:
                active -= 1

    app.service = SimpleNamespace(collect_results=collect, refresh=lambda row: row)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 2)
        task = app._history_result_requests[a.uid]
        try:
            app.server.state.workspace_view = "monitor"
            getattr(app, method)()
        finally:
            release.set()
            await asyncio.gather(task, *list(app._history_result_requests.values()))
        assert max_active == 1, calls
        assert calls == [(a.uid, False)] + ([(a.uid, True)] if method == "reload_selected_results" else [])

    asyncio.run(scenario())


def test_same_source_monitor_refresh_preserves_pending_saved_result():
    app, (a, _), loaded = _app()
    entered, release = threading.Event(), threading.Event()
    controller = SimpleNamespace(records={"live-A": a}, refresh_errors={},
                                 state=SimpleNamespace(submitted={}, completed={"live-A": object()}))
    app._powgen_controller = controller
    app._sync_powgen_rows = lambda: None

    def refresh(**_):
        controller.records["live-A"] = controller.records["live-A"].model_copy(deep=True)
        return dict(controller.records)

    controller.refresh = refresh

    def collect(row):
        entered.set()
        assert release.wait(5)
        row.output_dir = "/tmp/mock-result-" + row.uid
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 2)
        task = app._history_result_requests[a.uid]
        try:
            await app._refresh_powgen_jobs(controller, [])
        finally:
            release.set()
            await task
        assert loaded == [a.uid], (loaded, app.server.state.selected_result_load_error)

    asyncio.run(scenario())


def test_automatic_monitor_and_saved_click_share_archive_transaction():
    app, (a, _), loaded = _app()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def collect(row):
        calls.append(row.uid)
        entered.set()
        assert release.wait(5)
        row.output_dir = "/tmp/mock-result-A"
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 2)
        monitor = asyncio.create_task(app._monitor_record(a))
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(monitor, *list(app._history_result_requests.values()))
        assert calls == [a.uid]
        assert loaded and set(loaded) == {a.uid}
        assert app.records[a.uid].output_dir == "/tmp/mock-result-A"

    asyncio.run(scenario())


@pytest.mark.parametrize("view", ["experiment", "refinement"])
def test_automatic_monitor_does_not_replace_a_newer_workspace(view):
    app, (a, _), loaded = _app()
    app._select_record(a)
    app.server.state.workspace_view = view
    a.output_dir = "/tmp/completed-result-A"
    app._monitor_update(a)
    assert loaded == [] and app.server.state.workspace_view == view


def test_saved_result_cache_survives_next_live_monitor_tick():
    app, (a, _), loaded = _app()
    controller = SimpleNamespace(records={"live-A": a}, refresh_errors={},
                                 state=SimpleNamespace(submitted={}, completed={"live-A": object()}))
    controller.refresh = lambda **_: dict(controller.records)
    app._powgen_controller = controller
    app._sync_powgen_rows = lambda: None

    def collect(row):
        row.output_dir = "/tmp/mock-result-" + row.uid
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        await asyncio.gather(*list(app._history_result_requests.values()))
        assert loaded == [a.uid]
        assert app.records[a.uid].output_dir == "/tmp/mock-result-A"
        await app._refresh_powgen_jobs(controller, [])
        assert app.records[a.uid].output_dir == "/tmp/mock-result-A"

    asyncio.run(scenario())


@pytest.mark.parametrize("conflicting_hda", [False, True])
def test_delayed_monitor_subset_keeps_new_associations_but_rejects_hda_replacement(conflicting_hda):
    app, (a, _), _ = _app()
    read_entered, read_release = threading.Event(), threading.Event()
    refresh_entered, refresh_release = threading.Event(), threading.Event()
    controller = SimpleNamespace(records={"live-A": a}, refresh_errors={},
                                 state=SimpleNamespace(submitted={}, completed={"live-A": object()}))
    app._powgen_controller = controller
    app._sync_powgen_rows = lambda: None

    def refresh(**_):
        older = a.model_copy(deep=True)
        if conflicting_hda:
            older.output_dataset_ids["summary"] = "replacement-A"
        refresh_entered.set()
        assert refresh_release.wait(5)
        return {"live-A": older}

    def collect(row):
        read_entered.set()
        assert read_release.wait(5)
        row.output_dataset_ids["results_archive"] = "archive-A"
        row.output_dir = "/tmp/expanded-result-A"
        row.publication_job_id = "export-A"
        row.publication_status = RunStatus.QUEUED
        return row

    controller.refresh = refresh
    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(read_entered.wait, 2)
        monitor = asyncio.create_task(app._refresh_powgen_jobs(controller, []))
        assert await asyncio.to_thread(refresh_entered.wait, 2)
        try:
            read_release.set()
            await asyncio.gather(*list(app._history_result_requests.values()))
            assert app.records[a.uid].output_dataset_ids["results_archive"] == "archive-A"
        finally:
            refresh_release.set()
            await monitor
        current = app.records[a.uid]
        if conflicting_hda:
            assert current.output_dataset_ids["summary"] == "replacement-A"
            assert current.output_dir is None and current.publication_job_id is None
        else:
            assert current.output_dataset_ids == {"summary": "A", "results_archive": "archive-A"}
            assert current.output_dir == "/tmp/expanded-result-A" and current.publication_job_id == "export-A"
            assert controller.records["live-A"].output_dir == current.output_dir

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["clear", "service", "source"])
def test_pending_monitor_metadata_cannot_write_into_changed_context(change):
    app, (a, _), _ = _app()
    a.status = a.analysis_status = RunStatus.RUNNING
    entered, release = threading.Event(), threading.Event()
    updates = []

    def refresh(row):
        entered.set()
        assert release.wait(5)
        row.status = row.analysis_status = RunStatus.OK
        return row

    app.service = SimpleNamespace(refresh=refresh)
    app._monitor_update = lambda row: updates.append(row)

    async def scenario():
        task = asyncio.create_task(app._monitor_record(a))
        assert await asyncio.to_thread(entered.wait, 2)
        if change == "clear":
            app._result_context = object()
            app.records.clear()
        elif change == "service":
            app.service = SimpleNamespace()
        else:
            app.records[a.uid] = a.model_copy(update={"output_dataset_ids": {"summary": "replacement-A"}})
        release.set()
        await task
        assert updates == [] and a.status == RunStatus.RUNNING
        if change == "clear":
            assert app.records == {}
        elif change == "source":
            assert app.records[a.uid].output_dataset_ids["summary"] == "replacement-A"

    asyncio.run(scenario())


def test_refresh_rpc_returns_before_metadata_read_and_preserves_active_run():
    app, (a, _), loaded = _app()
    a.status = a.analysis_status = RunStatus.RUNNING
    entered, release = threading.Event(), threading.Event()

    def refresh(row):
        entered.set()
        assert release.wait(5)
        row.stage = "Pattern scoring"
        return row

    app.service = SimpleNamespace(refresh=refresh)
    app._select_record(a)

    async def scenario():
        assert app.refresh_selected_run() is None
        assert await asyncio.to_thread(entered.wait, 2)
        try:
            assert app.server.state.selected_run_loading
            app.server.state.run_search = "UI still responds"
            await asyncio.sleep(0)
        finally:
            release.set()
            await asyncio.gather(*list(app._history_result_requests.values()))
        assert loaded == []
        assert app.records[a.uid].stage == "Pattern scoring"
        assert app.server.state.workspace_view == "monitor"
        assert not app.server.state.selected_run_loading

    asyncio.run(scenario())


def test_refresh_retains_direct_synchronous_api():
    app, (a, _), loaded = _app()
    a.status = a.analysis_status = RunStatus.RUNNING
    app._select_record(a)
    app.service = SimpleNamespace(refresh=lambda row: row.model_copy(update={"stage": "Pattern scoring"}))
    assert app.refresh_selected_run() is None
    assert app.records[a.uid].stage == "Pattern scoring"
    assert loaded == [] and not app.server.state.selected_run_loading


def test_failed_force_reload_keeps_old_cache_and_can_retry():
    app, (a, _), loaded = _app()
    a.output_dir = "/tmp/old-result"
    a.result_status = ResultStatus.READY
    calls = []

    def collect(row, *, force=False):
        calls.append(force)
        if len(calls) == 1:
            row.result_status = ResultStatus.ERROR
            row.message = "reload timed out"
        else:
            row.result_status = ResultStatus.READY
            row.output_dir = "/tmp/reloaded-result"
        return row

    app.service = SimpleNamespace(collect_results=collect)
    app._select_record(a)

    async def scenario():
        app.reload_selected_results()
        await asyncio.gather(*list(app._history_result_requests.values()))
        assert "reload timed out" in app.server.state.selected_result_load_error
        assert app.records[a.uid].output_dir == "/tmp/old-result"
        assert app.records[a.uid].status == RunStatus.OK
        app.reload_selected_results()
        await asyncio.gather(*list(app._history_result_requests.values()))
        assert calls == [True, True] and loaded == [a.uid]
        assert app.records[a.uid].output_dir == "/tmp/reloaded-result"
        assert not app.server.state.selected_result_load_error

    asyncio.run(scenario())


def test_clear_then_recover_same_source_does_not_revive_previous_transaction():
    app, (a, b), loaded = _app()
    b.status = b.analysis_status = RunStatus.RUNNING
    entered, release = threading.Event(), threading.Event()
    calls, published = [], []

    def collect(row):
        calls.append(row.uid)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
        row.output_dir = "/tmp/read-" + str(len(calls))
        return row

    def publish(row, **_):
        published.append(row.output_dir)
        return row

    async def stop():
        pass

    app.service = SimpleNamespace(collect_results=collect)
    app._publish_results_if_requested = publish
    app._stop_session_enrollment = stop

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 2)
        old = app._history_result_requests[a.uid]
        app._run_selection_changed([b.uid])
        await app.clear_workspace()
        app.records[a.uid] = a.model_copy(deep=True)
        app._run_selection_changed([a.uid])
        release.set()
        await asyncio.gather(old, *list(app._history_result_requests.values()))
        assert calls == [a.uid, a.uid] and loaded == [a.uid]
        assert published == ["/tmp/read-2"]
        assert app.records[a.uid].output_dir == "/tmp/read-2"

    asyncio.run(scenario())


@pytest.mark.parametrize("first_fails", [False, True])
def test_publication_copies_share_acknowledged_export_and_allow_failed_retry(first_fails):
    app, (a, _), _ = _app()
    a.output_dataset_ids["results_archive"] = "archive-A"
    a.inputs = InputSelection(data_path="pattern", use_builtin_cuka=True,
                              instrument="PG3", ipts="IPTS-1", publish_directory="shared",
                              publish_results_to_ipts=True)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def submit(row, **_):
        calls.append(row.uid)
        if len(calls) == 1:
            entered.set()
            assert release.wait(5)
            if first_fails:
                raise TimeoutError("export acknowledgement unavailable")
        return SimpleNamespace(galaxy_job_id="export-A", status=RunStatus.QUEUED), "shared/result.zip"

    app.service = SimpleNamespace(submit_results_export=submit, refresh_utility=lambda action: action)

    async def scenario():
        one = asyncio.create_task(asyncio.to_thread(app._publish_results_if_requested, a.model_copy(deep=True)))
        assert await asyncio.to_thread(entered.wait, 2)
        two = asyncio.create_task(asyncio.to_thread(app._publish_results_if_requested, a.model_copy(deep=True)))
        release.set()
        results = await asyncio.gather(one, two)
        assert len(calls) == (2 if first_fails else 1)
        assert results[-1].publication_job_id == "export-A"
        assert a.publication_job_id is None

    asyncio.run(scenario())
