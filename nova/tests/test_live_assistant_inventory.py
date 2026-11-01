"""A bank belongs to the selected sample scope, not the latest IPTS run."""

import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_live_assistant import assistant
from radar_pd_nova.live_beamlines import BEAMLINES
from radar_pd_nova.live_setup_preview import inspect_inventory


def scoped_inventory(instrument, ipts, folder, index, **kwargs):
    selected = kwargs.get("sample_ids")
    bank = 2 if selected == ("A",) else 5
    return {"instrument": instrument, "ipts": ipts, "folder": folder,
            "run_id": f"PG3_{bank}", "sample_id": "A" if bank == 2 else "B",
            "banks": [{"title": f"Bank {bank}", "value": bank}]}


def test_atomic_new_sample_and_bank_use_proposed_inventory(assistant, monkeypatch):
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", scoped_inventory)
    state = assistant.server.state

    async def run():
        await assistant.live_assistant_continue()
        assert state.powgen_bank_options[0]["value"] == 5
        await assistant._assistant_apply_reply_patch({
            "powgen_processing_scope": "selected", "powgen_processing_sample_ids": ["A"],
            "powgen_bank_scope": "selected", "powgen_processing_banks": [2],
        }, "Use sample A, bank 2")
        assert state.powgen_processing_sample_ids == ["A"]
        assert state.powgen_processing_banks == [2]
        assert state.powgen_bank_options[0]["value"] == 2
        assert "PG3_2" in state.live_assistant_inventory_note
        assert not assistant.service.jobs

    asyncio.run(run())


def test_invalid_proposed_bank_keeps_whole_draft_unchanged(assistant, monkeypatch):
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", scoped_inventory)
    state = assistant.server.state

    async def run():
        await assistant.live_assistant_continue()
        before = assistant._assistant_snapshots()
        with pytest.raises(ValueError, match="bank is no longer"):
            await assistant._assistant_apply_reply_patch({
                "powgen_processing_scope": "selected", "powgen_processing_sample_ids": ["A"],
                "powgen_bank_scope": "selected", "powgen_processing_banks": [5],
            }, "Incorrect bank")
        assert assistant._assistant_snapshots() == before
        assert state.powgen_bank_options[0]["value"] == 5

    asyncio.run(run())


def test_manual_sample_change_refreshes_before_bank_validation(assistant, monkeypatch):
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", scoped_inventory)
    state = assistant.server.state

    async def run():
        await assistant.live_assistant_continue()
        state.powgen_processing_scope = "selected"
        state.powgen_processing_sample_ids = ["A"]
        state.powgen_bank_scope = "selected"
        state.powgen_processing_banks = [2]
        state.flush()
        assistant._live_assistant_changed()
        assert state.powgen_bank_options == []
        assert "Refresh banks" in state.live_assistant_inventory_note
        await assistant.live_assistant_continue()
        assert state.live_assistant_error == ""
        assert state.live_assistant_step == "structures"
        assert state.powgen_bank_options[0]["value"] == 2

    asyncio.run(run())


def test_stale_async_inventory_cannot_overwrite_new_selection(assistant, monkeypatch):
    state = assistant.server.state

    def moved_selection(*args, **kwargs):
        # Simulate a newer browser edit arriving while the read is off-thread.
        state.powgen_processing_sample_ids = ["B"]
        return scoped_inventory(*args, **kwargs)

    async def run():
        await assistant.live_assistant_continue()
        state.powgen_processing_scope = "selected"
        state.powgen_processing_sample_ids = ["A"]
        state.flush()
        monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", moved_selection)
        with pytest.raises(ValueError, match="changed during discovery"):
            await assistant._assistant_inspect()
        assert state.powgen_processing_sample_ids == ["B"]

    asyncio.run(run())


def test_preview_prefilters_unrelated_samples_but_verifies_candidate(tmp_path, monkeypatch):
    """Large IPTS previews should open one matching source, not 3,229 sources."""
    from radar_pd_nova.powgen_watch import WatchDefinition
    path = tmp_path / "PG3_1.gsa"
    path.write_text("verified candidate")
    runs = [SimpleNamespace(run_id=f"PG3_{n}", run_number=n, source_path=str(path))
            for n in range(1, 3230)]
    inspected = []
    class Index:
        def candidate_runs(self, given, selected):
            assert len(given) == 3229 and selected == ("A",)
            return given[:1]
        def identity(self, run):
            inspected.append(run.run_number)
            return {"sample_id": "A", "sample_name": "calibrant"}
    def setup(*args):
        return SimpleNamespace(source_signature=(23, 1), as_dict=lambda: {}), [{"value": 2}]
    monkeypatch.setitem(BEAMLINES, "PG3", replace(BEAMLINES["PG3"],
        preview_runs=lambda *args: runs, preview_setup=setup))
    monkeypatch.setattr(WatchDefinition, "source_directory", property(lambda self: str(tmp_path)))
    monkeypatch.setattr("radar_pd_nova.live_setup_preview.include_run", lambda *args, **kwargs: True)
    result = inspect_inventory("PG3", "IPTS-123", "shared/autoreduce", Index(), sample_ids=("A",))
    assert result["run_id"] == "PG3_1"
    assert inspected == [1]


def test_hb2a_discovery_does_not_certify_unreviewed_sensor(assistant, monkeypatch):
    state = assistant.server.state
    state.live_beamline = "HB2A"
    state.hb2a_temperature_channel = "he3_pair"
    state.hb2a_temperature_unit = "K"
    state.hb2a_condition_reviewed = False
    state.flush()
    seen = []
    def read_inventory(*args, **kwargs):
        seen.append(kwargs["condition_bindings"])
        return {"run_id": "HB2A_exp0853_scan0014", "sample_id": "ErSSL", "banks": [{"value": 1}]}
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", read_inventory)
    asyncio.run(assistant._assistant_inspect())
    assert seen == [{}]
    assert state.hb2a_condition_reviewed is False


def test_clear_workspace_retires_pending_discovery(assistant, monkeypatch):
    async def run():
        await assistant.live_assistant_continue()
        entered, release = threading.Event(), threading.Event()
        def delayed(*args, **kwargs):
            entered.set()
            assert release.wait(5)
            return scoped_inventory(*args, **kwargs)
        monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", delayed)
        async def stop_enrollment():
            pass
        monkeypatch.setattr(assistant, "_stop_session_enrollment", stop_enrollment)
        assistant.start_live_assistant_inventory()
        old = assistant._live_assistant_action_task
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            await assistant.clear_workspace()
            assert not assistant.server.state.live_assistant_busy
            assistant.server.state.powgen_ipts = "IPTS-123"
            assistant.server.state.flush()
            assistant.start_live_assistant_inventory()
            new = assistant._live_assistant_action_task
            assert new is not old
            await asyncio.sleep(0)
        finally:
            release.set()
            await asyncio.gather(old, new, return_exceptions=True)
        assert not assistant.server.state.live_assistant_busy
    asyncio.run(run())
