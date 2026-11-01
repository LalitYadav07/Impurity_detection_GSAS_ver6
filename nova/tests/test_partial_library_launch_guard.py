"""Pending library decisions block new form requests, not approved old work."""
import asyncio
from copy import deepcopy
import json

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.library_review import library_review_signature
from radar_pd_nova.library_setup import LibraryDraft
from radar_pd_nova.models import AnalysisConfig, UtilityActionRecord
from test_live_setup import live, TrialController
from test_partial_library_review import CIF


def pending_review(app, tmp_path, context):
    draft = LibraryDraft(app.server.state, context)
    draft.database_source = "builtin"; app._library_review_changed()
    path = tmp_path / "candidate.cif"; path.write_text(CIF)
    draft.database_source = "custom_mini"
    draft.library_builder_local_paths = [str(path)]
    draft.library_builder_generation = "generation"
    action = UtilityActionRecord(uid="partial", tool_id="neutrons_radar_pd_library_builder_prototype",
        name="Partial", galaxy_job_id="old-build", status="ok",
        outputs={"library_archive": "partial", "library_manifest": "manifest"})
    app._library_action_contexts[action.uid] = {"workflow": context, "generation": "generation",
        "signature": draft.signature(), "source_signature": library_review_signature(draft),
        "preflight": {"requested_count": 2, "failure_rows": [], "skipped_count": 0}}
    app.service._dataset_document = lambda _: {"phase_count": 1, "phase_ids": ["other"],
        "failures": [{"source_name": "main.cif", "error": "Invalid occupancy"}]}
    app._remember_history_dataset = lambda *_: None
    app._sync_activities = lambda: None
    asyncio.run(app._utility_completed(action))
    assert draft.database_source == "builtin" and app._library_decision_pending(context)
    return draft


def single_pending(tmp_path):
    app = RadarPdNovaApp()
    app.service = GalaxyService("https://example.invalid", "test", "local-history", output_root=tmp_path / "runs")
    app.server.state.sample_elements = "Al, O"
    app.server.state.input_source = "galaxy"; app.server.state.history_data_id = "pattern"
    app.server.state.instrument_source = "galaxy"; app.server.state.history_instrument_id = "profile"
    draft = pending_review(app, tmp_path, "single")
    return app, draft


def test_single_submit_is_blocked_before_snapshot_or_submission_and_keep_resolves(tmp_path):
    app, draft = single_pending(tmp_path)
    captured = []
    async def capture(snapshot): captured.append(snapshot)
    app._submit_pending = capture; app._sync_runs = lambda: None; app._select_record = lambda record: None
    async def submit():
        app.submit_run()
        if app._submission_tasks: await asyncio.gather(*tuple(app._submission_tasks))
    asyncio.run(submit())
    assert not captured and not app.records
    assert "partial candidate-library review" in app.server.state.error_message
    app.retain_current_candidate_library("single", draft.library_builder_decision_token)
    asyncio.run(submit())
    assert len(captured) == 1 and captured[0].inputs.database_dataset_id is None


def test_configuration_save_and_folder_export_are_blocked_before_writes(tmp_path):
    app, draft = single_pending(tmp_path)
    saved = []
    app.service.save_configuration = lambda config: saved.append(config)
    app.save_current_configuration()
    assert not saved and not app._utility_tasks
    assert "partial candidate-library review" in app.server.state.error_message
    state = app.server.state
    state.input_source = "ipts_browser"; state.use_builtin_cuka = True; state.main_cif_source = "none"
    app.facility.write_configuration = lambda *args: saved.append(args)
    app.save_watch_recipe()
    assert not saved and "partial candidate-library review" in state.watch_recipe_message


@pytest.mark.parametrize("action", ["test", "start", "configuration", "assistant"])
def test_fresh_live_routes_are_blocked_without_uploads_or_jobs(live, tmp_path, action):
    draft = pending_review(live, tmp_path, "live")
    if action == "test": asyncio.run(live.test_live_scan())
    elif action == "start": asyncio.run(live.start_powgen_monitoring())
    elif action == "configuration":
        with pytest.raises(ValueError, match="partial candidate-library review"):
            asyncio.run(live._save_live_configuration(AnalysisConfig(sample_elements=["Al", "O"])))
    else:
        with pytest.raises(ValueError, match="partial candidate-library review"):
            asyncio.run(live._assistant_configuration())
    assert not live.service.jobs and not live.service.saved and not live.service.uploads
    assert live._powgen_controller is None and draft.library_builder_decision_pending


def test_stale_review_still_needs_explicit_keep_choice(tmp_path):
    app, draft = single_pending(tmp_path)
    token = draft.library_builder_decision_token
    draft.library_builder_name = "revised selection"
    app._library_review_changed()
    assert not draft.library_builder_partial_review and app._library_decision_pending()
    with pytest.raises(ValueError, match="partial candidate-library review"): app._configuration()
    app.retain_current_candidate_library("single", "old-wrong-token")
    assert app._library_decision_pending()
    app.retain_current_candidate_library("single", token)
    assert not app._library_decision_pending() and app._configuration().sample_elements == ["Al", "O"]


def test_existing_watch_explicit_resume_keeps_original_settings_without_new_uploads(live, tmp_path):
    async def first_start():
        await live.start_powgen_monitoring(); await asyncio.sleep(0)
    asyncio.run(first_start())
    original = live._powgen_controller
    assert original is not None
    settings = original.settings
    saved_count, upload_count = len(live.service.saved), len(live.service.uploads)
    live.server.state.powgen_monitoring = False
    draft = pending_review(live, tmp_path, "live")
    async def resume():
        await live.resume_existing_powgen_monitoring(); await asyncio.sleep(0)
    asyncio.run(resume())
    assert live._powgen_controller is original and original.settings is settings
    assert len(TrialController.instances) == 1
    assert len(live.service.saved) == saved_count and len(live.service.uploads) == upload_count
    assert live.server.state.powgen_monitoring and draft.library_builder_decision_pending


def test_large_failure_review_is_paginated_and_download_retains_every_row(tmp_path):
    app, draft = single_pending(tmp_path)
    rows = [{"name": f"source-{i}.cif", "reason": f"reason-{i}"} for i in range(4999)]
    draft.library_builder_failure_rows = deepcopy(rows)
    draft.library_builder_skipped_count = len(rows)
    downloads = []
    app.download_file = lambda *args: downloads.append(args)
    app.download_candidate_library_review("single")
    payload = json.loads(downloads[0][2])
    assert payload["failures"] == rows and len(draft.library_builder_failure_rows) == 4999
    html = app.layout.html
    assert 'v-for="failure in library_builder_failure_rows"' not in html
    assert ':itemsPerPageOptions="[10, 20, 50, 100]"' in html and 'itemsPerPage="20"' in html
    assert "Search CIF names and review reasons" in html
    assert "Failed CIF sources" in html and "Download complete review" in html
    assert 'busy || library_builder_decision_pending ||' in html
    assert 'live_launch_busy || live_library_builder_decision_pending' in html
