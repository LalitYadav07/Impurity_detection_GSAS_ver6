"""Slow saved-run reads must leave navigation responsive and source-bound."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.models import AnalysisMode, ResultStatus, RunRecord, RunStatus


def history_app():
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


async def finish_requests(app):
    await asyncio.gather(*list(app._history_result_requests.values()))
    await asyncio.sleep(0)


def test_saved_run_click_returns_promptly_and_duplicate_click_reuses_slow_read(tmp_path):
    app, (a, _), loaded = history_app()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def collect(row):
        calls.append(row.uid)
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert app.server.state.selected_run_uid == a.uid
        assert "Opening Saved A" in app.server.state.selected_result_load_message
        assert app.server.state.selected_run_loading
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            app._run_selection_changed([a.uid])
            app.server.state.run_search = "navigation is responsive"
            await asyncio.sleep(0)
            assert calls == [a.uid]
        finally:
            release.set()
            await finish_requests(app)
        assert loaded == [a.uid]
        assert not app.server.state.selected_run_loading
        assert a.output_dir is None  # the worker received a private record
        assert app.records[a.uid].output_dir == str(tmp_path)
        assert not app._history_result_downloads and not app._history_result_requests

    asyncio.run(scenario())


@pytest.mark.parametrize("return_to_first", [False, True])
def test_cross_selection_uses_one_reader_and_only_displays_current_request(tmp_path, return_to_first):
    app, (a, b), loaded = history_app()
    entered = {uid: threading.Event() for uid in ("A", "B")}
    release = {uid: threading.Event() for uid in ("A", "B")}
    calls = []

    def collect(row):
        calls.append(row.uid)
        entered[row.uid].set()
        assert release[row.uid].wait(3)
        row.output_dir = str(tmp_path / row.uid)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered[a.uid].wait, 1)
        app._run_selection_changed([b.uid])
        await asyncio.sleep(0)
        assert not entered[b.uid].is_set()
        if return_to_first:
            app._run_selection_changed([a.uid])
        try:
            release[a.uid].set()
            assert await asyncio.to_thread(entered[b.uid].wait, 1)
            if not return_to_first:
                assert loaded == []
                assert app.server.state.selected_run_uid == b.uid
                assert app.server.state.selected_run_loading
        finally:
            release[a.uid].set()
            release[b.uid].set()
            await finish_requests(app)
        assert calls == [a.uid, b.uid]
        assert loaded == [a.uid if return_to_first else b.uid]
        assert not app._history_result_downloads and not app._history_result_requests

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["clear", "service", "record", "source_in_place", "view"])
def test_late_saved_result_cannot_replace_a_new_workspace_or_source(tmp_path, change):
    app, (a, _), loaded = history_app()
    entered, release = threading.Event(), threading.Event()

    def collect(row):
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 1)
        if change == "clear":
            app.records.clear()
            app.server.state.selected_run_uid = ""
        elif change == "service":
            app.service = SimpleNamespace()
        elif change == "record":
            app.records[a.uid] = a.model_copy(update={"galaxy_job_id": "new-publication"})
        elif change == "source_in_place":
            a.output_dataset_ids = {"summary": "replacement"}
        else:
            app.server.state.workspace_view = "refinement"
        release.set()
        await finish_requests(app)
        assert loaded == []
        if change == "clear":
            assert not app.records
        elif change == "view":
            assert app.server.state.workspace_view == "refinement"
            assert app.records[a.uid].output_dir == str(tmp_path)
        elif change == "record":
            assert app.records[a.uid].galaxy_job_id == "new-publication"
        elif change == "source_in_place":
            assert app.records[a.uid].output_dataset_ids == {"summary": "replacement"}
        else:
            assert app.records[a.uid] is a

    asyncio.run(scenario())


def test_selecting_active_run_immediately_clears_old_opening_message(tmp_path):
    app, (a, b), loaded = history_app()
    b.status = RunStatus.RUNNING
    entered, release = threading.Event(), threading.Event()

    def collect(row):
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 1)
        app._run_selection_changed([b.uid])
        assert app.server.state.workspace_view == "monitor"
        assert not app.server.state.selected_run_loading
        assert not app.server.state.selected_result_load_message
        release.set()
        await finish_requests(app)
        assert app.server.state.selected_run_uid == b.uid and loaded == []

    asyncio.run(scenario())


def test_failed_saved_read_keeps_successful_analysis_and_allows_explicit_retry(tmp_path):
    app, (a, _), loaded = history_app()
    calls = []

    def collect(row):
        calls.append(row.uid)
        if len(calls) == 1:
            raise TimeoutError("saved artifact read timed out")
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        await finish_requests(app)
        assert "saved artifact read timed out" in app.server.state.selected_result_load_error
        assert a.status is RunStatus.OK and loaded == []
        app._run_selection_changed([a.uid])
        await finish_requests(app)
        assert loaded == [a.uid] and calls == [a.uid, a.uid]
        assert not app.server.state.selected_result_load_error

    asyncio.run(scenario())


def test_returned_collection_error_is_not_reused_on_explicit_retry(tmp_path):
    app, (a, _), loaded = history_app()
    calls = []

    def collect(row):
        calls.append(row.uid)
        if len(calls) == 1:
            row.result_status = ResultStatus.ERROR
            row.message = "Analysis finished, but result download failed"
            return row
        row.result_status = ResultStatus.READY
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        await finish_requests(app)
        assert loaded == [] and a.status is RunStatus.OK
        assert "result download failed" in app.server.state.selected_result_load_error
        assert not app._history_result_downloads and not app._history_result_requests
        app._run_selection_changed([a.uid])
        await finish_requests(app)
        assert calls == [a.uid, a.uid] and loaded == [a.uid]
        assert app.records[a.uid].result_status is ResultStatus.READY
        assert not app._history_result_downloads and not app._history_result_requests

    asyncio.run(scenario())


def test_clear_releases_request_references_and_prunes_active_read_after_completion(tmp_path):
    app, (a, b), loaded = history_app()
    b.status = RunStatus.RUNNING
    entered, release = threading.Event(), threading.Event()

    def collect(row):
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    async def save_enrollment():
        pass

    app.service = SimpleNamespace(collect_results=collect)
    app._stop_session_enrollment = save_enrollment

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 1)
        task = app._history_result_requests[a.uid]
        app._run_selection_changed([b.uid])  # A continues as a superseded background read
        await app.clear_workspace()
        assert not app._history_result_requests
        assert len(app._history_result_downloads) == 1  # never duplicate an active read
        assert not app.records and app._result_open_owner is None
        release.set()
        await task
        await asyncio.sleep(0)
        assert not app._history_result_downloads
        assert not app.records and loaded == []

    asyncio.run(scenario())


def test_cancelled_waiter_retains_active_read_for_reselection(tmp_path):
    app, (a, _), loaded = history_app()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def collect(row):
        calls.append(row.uid)
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 1)
        task = app._history_result_requests[a.uid]
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert len(app._history_result_downloads) == 1
        app._run_selection_changed([a.uid])
        release.set()
        await finish_requests(app)
        assert calls == [a.uid] and loaded == [a.uid]
        assert not app._history_result_downloads and not app._history_result_requests

    asyncio.run(scenario())


def test_clear_discards_queued_read_before_io_but_keeps_running_read_safe(tmp_path):
    app, (a, b), loaded = history_app()
    active = b.model_copy(update={"uid": "C", "galaxy_job_id": "C", "status": RunStatus.RUNNING})
    app.records[active.uid] = active
    entered, release = threading.Event(), threading.Event()
    calls = []

    def collect(row):
        calls.append(row.uid)
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    async def save_enrollment():
        pass

    app.service = SimpleNamespace(collect_results=collect)
    app._stop_session_enrollment = save_enrollment

    async def scenario():
        app._run_selection_changed([a.uid])
        assert await asyncio.to_thread(entered.wait, 1)
        app._run_selection_changed([b.uid])
        await asyncio.sleep(0)
        pending = list(app._history_result_requests.values())
        assert calls == [a.uid] and len(app._history_result_downloads) == 2
        # The visible active run has no result load; Clear can proceed under
        # the existing UI policy while A and B are superseded reads.
        app._run_selection_changed([active.uid])
        await app.clear_workspace()
        assert not app.records and not app._history_result_requests
        release.set()
        await asyncio.gather(*pending)
        await asyncio.sleep(0)
        assert calls == [a.uid]  # B rechecks the cleared context at the lock
        assert not loaded and not app.records
        assert not app._history_result_downloads and not app._history_result_requests

    asyncio.run(scenario())


@pytest.mark.parametrize("live_first", [True, False])
def test_cross_path_selection_shares_reader_and_old_result_cannot_replace_new(tmp_path, live_first):
    app, (a, b), loaded = history_app()
    controller = app._powgen_controller = SimpleNamespace(records={"live-A": a, "live-B": b})
    entered = {uid: threading.Event() for uid in ("A", "B")}
    release = {uid: threading.Event() for uid in ("A", "B")}
    calls = []

    def collect(row):
        calls.append(row.uid)
        entered[row.uid].set()
        assert release[row.uid].wait(3)
        row.output_dir = str(tmp_path / row.uid)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        if live_first:
            app.server.state.powgen_selected_run_id = "live-A"
            first = asyncio.create_task(app.open_powgen_selected_run_async())
        else:
            app._run_selection_changed([a.uid])
            first = app._history_result_requests[a.uid]
        assert await asyncio.to_thread(entered[a.uid].wait, 1)
        if live_first:
            app._run_selection_changed([b.uid])
            second = app._history_result_requests[b.uid]
        else:
            app.server.state.powgen_selected_run_id = "live-B"
            second = asyncio.create_task(app.open_powgen_selected_run_async())
        await asyncio.sleep(0)
        assert app.server.state.selected_run_uid == b.uid
        assert not entered[b.uid].is_set()  # full archive reads never overlap
        release[a.uid].set()
        assert await asyncio.to_thread(entered[b.uid].wait, 1)
        assert loaded == [] and app.server.state.selected_run_loading
        assert "Saved B" in app.server.state.selected_result_load_message
        release[b.uid].set()
        await asyncio.gather(first, second)
        await asyncio.sleep(0)
        assert calls == [a.uid, b.uid] and loaded == [b.uid]
        assert controller.records["live-A"] is a and a.output_dir is None
        if not live_first:
            assert controller.records["live-B"].output_dir == str(tmp_path / b.uid)
        assert not app._history_result_downloads and not app._history_result_requests

    asyncio.run(scenario())


@pytest.mark.parametrize("live_first", [True, False])
def test_same_source_clicked_through_both_paths_downloads_once(tmp_path, live_first):
    app, (a, _), loaded = history_app()
    app._powgen_controller = SimpleNamespace(records={"live-A": a})
    app.server.state.powgen_selected_run_id = "live-A"
    entered, release = threading.Event(), threading.Event()
    calls = []

    def collect(row):
        calls.append(row.uid)
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        if live_first:
            task = asyncio.create_task(app.open_powgen_selected_run_async())
        else:
            app._run_selection_changed([a.uid])
            task = app._history_result_requests[a.uid]
        assert await asyncio.to_thread(entered.wait, 1)
        if live_first:
            app._run_selection_changed([a.uid])
        else:
            await app.open_powgen_selected_run_async()
        release.set()
        await task
        assert calls == [a.uid] and loaded == [a.uid]

    asyncio.run(scenario())


@pytest.mark.parametrize("change", ["controller", "source"])
def test_live_read_cannot_update_replaced_controller_or_source(tmp_path, change):
    app, (a, _), loaded = history_app()
    controller = app._powgen_controller = SimpleNamespace(records={"live-A": a})
    app.server.state.powgen_selected_run_id = "live-A"
    entered, release = threading.Event(), threading.Event()

    def collect(row):
        entered.set()
        assert release.wait(3)
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(collect_results=collect)

    async def scenario():
        task = asyncio.create_task(app.open_powgen_selected_run_async())
        assert await asyncio.to_thread(entered.wait, 1)
        replacement = a.model_copy(update={"galaxy_job_id": "replacement"})
        if change == "controller":
            app._powgen_controller = SimpleNamespace(records={"live-A": replacement})
        else:
            controller.records["live-A"] = replacement
        release.set()
        await task
        assert loaded == [] and a.output_dir is None
        assert app._powgen_controller.records["live-A"] is replacement
        assert not app._history_result_downloads

    asyncio.run(scenario())


def test_late_live_metadata_refresh_cannot_override_saved_run_selection(tmp_path):
    app, (a, b), loaded = history_app()
    a.status = RunStatus.QUEUED
    a.output_dataset_ids = {}
    controller = app._powgen_controller = SimpleNamespace(records={"live-A": a})
    app.server.state.powgen_selected_run_id = "live-A"
    entered, release = threading.Event(), threading.Event()
    calls = []

    def refresh(row):
        entered.set()
        assert release.wait(3)
        return row.model_copy(update={"status": RunStatus.OK, "output_dataset_ids": {"summary": "new-A"}})

    def collect(row):
        calls.append(row.uid)
        row.output_dir = str(tmp_path)
        return row

    app.service = SimpleNamespace(refresh=refresh, collect_results=collect)

    async def scenario():
        task = asyncio.create_task(app.open_powgen_selected_run_async())
        assert await asyncio.to_thread(entered.wait, 1)
        app._run_selection_changed([b.uid])
        await finish_requests(app)
        release.set()
        await task
        assert loaded == [b.uid] and calls == [b.uid]
        assert app.server.state.selected_run_uid == b.uid
        assert controller.records["live-A"] is a and a.status is RunStatus.QUEUED

    asyncio.run(scenario())
