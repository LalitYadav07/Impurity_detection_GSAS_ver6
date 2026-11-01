"""A delayed saved-configuration read cannot resume a superseded Live context."""
import asyncio
from copy import copy
import threading

import pytest

from test_live_setup import live, LocalGalaxy
from test_partial_library_launch_guard import pending_review


def prepared_existing(live, tmp_path):
    async def start():
        await live.start_powgen_monitoring()
        await asyncio.sleep(0)
    asyncio.run(start())
    controller = live._powgen_controller
    live.server.state.powgen_monitoring = False
    pending_review(live, tmp_path, "live")
    live._powgen_controller_configuration = None
    return controller


@pytest.mark.parametrize("change", ["ipts", "ipts_away_and_back", "beamline", "history", "service", "session", "controller", "form", "stop", "clear", "end"])
def test_changed_resume_context_never_starts_the_old_watch(live, tmp_path, change):
    controller = prepared_existing(live, tmp_path)
    service = live.service
    before = (len(service.saved), len(service.uploads), len(service.jobs))
    entered, release = threading.Event(), threading.Event()
    original = service.load_configuration_dataset
    reads = []
    def read(key):
        reads.append(key); entered.set(); assert release.wait(5)
        return original(key)
    service.load_configuration_dataset = read
    exited = []
    async def no_save(): pass
    async def exit_session(): exited.append(True)
    live._stop_session_enrollment = no_save
    live._leave_session_page = lambda: None
    live.exit_callback = exit_session

    async def scenario():
        task = asyncio.create_task(live.resume_existing_powgen_monitoring())
        assert await asyncio.to_thread(entered.wait, 2)
        state = live.server.state
        if change == "ipts":
            state.powgen_ipts = "99999"; live._powgen_ipts_changed()
        elif change == "ipts_away_and_back":
            previous = state.powgen_ipts
            state.powgen_ipts = "99999"; live._powgen_ipts_changed()
            state.powgen_ipts = previous; live._powgen_ipts_changed()
        elif change == "beamline": state.live_beamline = "NOM"
        elif change == "history": service.history_id = "new-history"
        elif change == "service": live.service = LocalGalaxy()
        elif change == "session": live._result_context = object()
        elif change == "controller": live._powgen_controller = copy(controller)
        elif change == "form": state.live_sample_elements = "Tb, Ge, O"
        elif change == "stop": live.stop_powgen_monitoring()
        elif change == "clear": await live.clear_workspace()
        elif change == "end": await live.end_session()
        release.set()
        await task
        await asyncio.sleep(0)
    asyncio.run(scenario())
    assert reads == [controller.settings.configuration_dataset_id]
    assert not live.server.state.powgen_monitoring
    assert not live.server.state.live_launch_busy and live._powgen_resume_owner is None
    assert before == (len(service.saved), len(service.uploads), len(service.jobs))
    if change == "ipts": assert live.server.state.powgen_ipts == "99999"
    if change == "end": assert exited and live.server.state.session_closing


def test_double_click_does_not_repeat_read_or_start_another_loop(live, tmp_path):
    controller = prepared_existing(live, tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = live.service.load_configuration_dataset
    reads, loops = [], []
    def read(key):
        reads.append(key); entered.set(); assert release.wait(5)
        return original(key)
    async def loop(): loops.append(True)
    live.service.load_configuration_dataset = read
    live._powgen_monitor_loop = loop
    async def scenario():
        first = asyncio.create_task(live.resume_existing_powgen_monitoring())
        assert await asyncio.to_thread(entered.wait, 2)
        await live.resume_existing_powgen_monitoring()
        release.set(); await first; await asyncio.sleep(0)
    asyncio.run(scenario())
    assert reads == [controller.settings.configuration_dataset_id] and loops == [True]
    assert live.server.state.powgen_monitoring and live._powgen_resume_owner is None


def test_previous_settings_action_allows_a_different_draft_at_click(live, tmp_path):
    controller = prepared_existing(live, tmp_path)
    state = live.server.state
    state.powgen_ipts = "99999"; live._powgen_ipts_changed()
    state.live_sample_elements = "Different draft"
    attached = []
    live.webmon_attach_monitor = lambda: attached.append(True)
    async def resume():
        await live.resume_existing_powgen_monitoring(); await asyncio.sleep(0)
    asyncio.run(resume())
    assert live.server.state.powgen_monitoring and live._powgen_controller is controller
    assert state.powgen_ipts == "99999" and state.live_sample_elements == "Different draft"
    assert controller.settings.ipts in state.powgen_message
    assert not attached  # Do not attach the new draft's WebMON source to the old watch.


@pytest.mark.parametrize("field", ["workspace_clearing", "session_closing"])
def test_resume_cannot_enter_a_closing_context(live, tmp_path, field):
    prepared_existing(live, tmp_path)
    def forbidden(*args): raise AssertionError("No configuration read in a closing view")
    live.service.load_configuration_dataset = forbidden
    setattr(live.server.state, field, True)
    asyncio.run(live.resume_existing_powgen_monitoring())
    assert not live.server.state.powgen_monitoring and live._powgen_resume_owner is None
