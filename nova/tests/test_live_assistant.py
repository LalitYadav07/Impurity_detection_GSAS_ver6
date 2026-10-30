"""Live setup conversations must fill a draft without launching analyses."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.live_assistant import SetupConversation, STEPS, goal_patch, parse_elements, parse_message, signature
from radar_pd_nova.live_setup_preview import NoPreviewData, preview_pattern
from test_live_setup import LocalGalaxy
from test_transition_input import product


@pytest.fixture
def assistant(monkeypatch, tmp_path):
    app = RadarPdNovaApp()
    state = app.server.state
    assert state.live_assistant_open is False
    app.open_live_assistant()
    app.service = LocalGalaxy()
    app._powgen_sample_index = lambda ipts: None
    state.powgen_ipts = "IPTS-123"
    state.live_sample_elements = "Fe, O"
    state.sample_elements = "Single-pattern sentinel"
    state.flush()

    async def discovery(ipts=None):
        state.powgen_sample_discovery_ipts = str(ipts or state.powgen_ipts)
        state.powgen_discovered_sample_options = [
            {"title": "A — oxide", "value": "A"}, {"title": "B — reference", "value": "B"},
        ]
        state.powgen_sample_message = "Found two samples"
        state.flush()

    source = tmp_path / "PG3_1.gsa"
    source.write_text("source revision 1")

    def inventory(instrument, ipts, folder, index, **kwargs):
        stat = source.stat()
        return {"instrument": instrument, "ipts": ipts, "folder": folder,
                "run_id": "PG3_1", "sample_id": "A", "source_path": str(source),
                "source_signature": [stat.st_size, stat.st_mtime_ns],
                "banks": [{"title": "Bank 2", "value": 2}]}

    app.refresh_powgen_samples = discovery
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", inventory)
    monkeypatch.setattr("radar_pd_nova.app.preflight_powgen_experiment", lambda *a, **k: {
        "ready": True, "available_banks": [2], "scan_count": 2, "analysis_count": 2,
        "profile_filename": "verified.instprm", "message": "Ready",
    })
    return app


def test_commands_are_bounded_and_chemical_symbols_are_not_guessed():
    assert parse_message("POWGEN IPTS-123")[1] == {"live_beamline": "PG3", "powgen_ipts": "IPTS-123"}
    assert parse_message("NOMAD IPTS:123")[1]["live_beamline"] == "NOM"
    assert parse_message("elements: Fe, O; can: Al")[1] == {
        "live_sample_elements": "Fe, O", "live_environment_elements": "Al",
    }
    assert parse_message("run /SNS/../secret")[0] == "unknown"
    assert parse_message("start monitoring")[0] == "clarify"
    assert parse_elements("Co C O D") == ["Co", "C", "O", "D"]
    for value in ("CO", "Fe2O3", "Xx", "fe"):
        with pytest.raises(ValueError):
            parse_elements(value)


def test_partial_draft_invalidates_only_dependent_confirmations():
    conversation = SetupConversation()
    snapshots = {s: {"value": s} for s in STEPS[:-1]}
    conversation.confirmed = {s: signature(v) for s, v in snapshots.items()}
    snapshots["chemistry"] = {"value": "changed"}
    assert conversation.invalidate(snapshots) == "chemistry"
    assert set(conversation.confirmed) == {"experiment", "selection", "structures"}


def test_goal_edits_are_allowlisted_and_undo_preserves_newer_manual_edits():
    conversation = SetupConversation()
    state = SimpleNamespace(live_analysis_mode="rapid", live_full_profile="custom")
    conversation.apply(state, goal_patch("balanced"), "Balanced")
    assert state.live_analysis_mode == "full" and state.live_full_profile == "balanced"
    state.live_full_profile = "thorough"
    with pytest.raises(ValueError, match="edited again"):
        conversation.undo_last(state)
    assert state.live_full_profile == "thorough"
    with pytest.raises(ValueError, match="only edit Live"):
        conversation.apply(state, {"sample_elements": "Al"}, "bad")


async def complete_cards(app):
    state = app.server.state
    await app.live_assistant_continue({"live_beamline": "PG3", "powgen_ipts": "IPTS-123", "powgen_source_subfolder": "shared/autoreduce"})
    assert not state.live_assistant_error
    assert state.live_assistant_step == "selection"
    state.powgen_processing_scope = "selected"
    state.powgen_processing_sample_ids = ["A"]
    state.flush()
    await app.live_assistant_continue()
    assert state.live_assistant_step == "structures"
    await app.live_assistant_continue()
    assert state.live_assistant_step == "chemistry"
    await app.live_assistant_continue()
    assert state.live_assistant_step == "settings"
    app.live_assistant_goal("balanced")
    await app.live_assistant_continue()
    assert state.live_assistant_step == "review"


def test_complete_conversation_fills_only_live_and_never_saves_or_submits(assistant):
    state = assistant.server.state
    async def run():
        await complete_cards(assistant)
        await assistant.live_assistant_continue()
    asyncio.run(run())
    assert not state.live_assistant_error
    assert state.live_assistant_complete and state.powgen_preflight_ready
    assert not state.powgen_monitoring
    assert not assistant.service.saved and not assistant.service.uploads and not assistant.service.jobs
    assert state.sample_elements == "Single-pattern sentinel"
    assert any(row["label"] == "Full search" and "Balanced" in row["value"] for row in state.live_assistant_summary)
    assistant.finish_live_assistant()
    assert not state.live_assistant_open and state.live_setup_panels == [3]
    assert not state.powgen_monitoring


def test_changed_configuration_cannot_finish_with_an_old_check(assistant):
    state = assistant.server.state
    async def run():
        await complete_cards(assistant)
        await assistant.live_assistant_continue()
    asyncio.run(run())
    state.live_background_terms = 11
    state.flush()
    assistant.finish_live_assistant()
    assert not state.live_assistant_complete and state.live_assistant_open
    assert "changed" in state.live_assistant_error.lower()
    assert state.live_background_terms == 11


def test_typed_continue_keeps_actionable_validation_error(assistant):
    state = assistant.server.state
    assistant.live_assistant_go("chemistry")
    state.live_sample_elements = "CO"
    asyncio.run(assistant.live_assistant_send("continue"))
    assert state.live_assistant_step == "chemistry"
    assert "CO" in state.live_assistant_error
    assert not assistant.service.jobs


def test_all_samples_is_explicit_and_removed_sample_ids_are_rejected(assistant):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_continue())
    state.powgen_processing_scope = "selected"
    state.powgen_processing_sample_ids = ["foreign-sample"]
    asyncio.run(assistant.live_assistant_continue())
    assert "no longer" in state.live_assistant_error
    assert state.live_assistant_step == "selection"


def test_nomad_does_not_inherit_automatic_profiles_or_implicit_bank_mapping(assistant):
    state = assistant.server.state
    state.live_beamline = "NOM"
    assistant._live_beamline_changed()
    state.flush()
    assert state.powgen_instrument_source == "computer"
    with pytest.raises(ValueError, match="source histogram"):
        assistant._assistant_validate_selection()
    state.nomad_physical_bank = 5
    state.powgen_instrument_source = "galaxy"
    state.powgen_instrument_dataset_id = "profile"
    with pytest.raises(ValueError, match="confirm"):
        assistant._assistant_validate_selection()


def test_slow_discovery_cannot_confirm_a_replaced_experiment(assistant, monkeypatch):
    state = assistant.server.state
    def changed(instrument, ipts, folder, index, **kwargs):
        state.powgen_ipts = "IPTS-456"
        return {"run_id": "old", "banks": [], "sample_id": "A"}
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", changed)
    asyncio.run(assistant.live_assistant_continue())
    assert "changed" in state.live_assistant_error.lower()
    assert "experiment" not in assistant._live_conversation.confirmed
    assert state.live_assistant_step == "experiment"


def test_curve_preview_preserves_negatives_and_masks_and_detects_revision(tmp_path):
    scan, _, y = product(tmp_path)
    path = tmp_path / "NOM_10.gsa"
    stat = path.stat()
    inventory = {"source_path": str(path), "root": str(tmp_path), "run_id": "NOM_10",
                 "instrument": "NOM", "ipts": "IPTS-10", "sample_id": "actual-sample",
                 "identity_status": "resolved", "setup": {},
                 "source_signature": [stat.st_size, stat.st_mtime_ns]}
    before = path.read_bytes()
    preview = preview_pattern(inventory, 1, exclude_regions=[(4000, 4500)])
    assert preview["negative_points"] > 0 and np.any(y < 0)
    assert preview["y"][0] is None
    assert any(value is None for x, value in zip(preview["x"], preview["y"]) if 4000 <= x <= 4500)
    assert path.read_bytes() == before
    path.write_bytes(before + b"# revision\n")
    with pytest.raises(ValueError, match="changed"):
        preview_pattern(inventory, 1)


def test_preview_never_changes_settings_and_apply_requires_the_same_scope(assistant, monkeypatch):
    state = assistant.server.state
    state.powgen_bank_options = [{"title": "Bank 2", "value": 2}]
    original = state.live_background_terms
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.preview_pattern", lambda *a: {
        "run_id": "PG3_1", "x": [1, 2], "y": [3, 4], "background": [1, 1],
        "recommendation": {"terms": 8},
    })
    asyncio.run(assistant.live_assistant_preview())
    assert state.live_background_terms == original
    assistant.live_assistant_apply_background()
    assert state.live_background_terms == 8
    state.live_fit_start = 10
    state.live_fit_end = 100
    state.flush()
    assistant.live_assistant_apply_background()
    assert "Inspect the current" in state.live_assistant_error
    assert state.live_background_terms == 8


def test_assistant_inputs_mount_once_with_distinct_live_upload_identity(assistant):
    template = assistant.layout.html
    for key in ("assistant-live-instrument-native", "assistant-live-main-cif-native",
                "live-assistant-cif-library-upload-native", "powgen-main-cif-upload-native"):
        assert template.count(f":key=\"'{key}'\"") == 1
    assert "live_assistant_step === 'experiment'" in template
    assert "live_assistant_step === 'review'" in template
    assert "{{ message.text }}" in template  # User text is escaped, never v-html.
    assert '@keydown.enter.prevent=' in template
    assert 'maxlength="1000"' in template
    assert 'role="group" aria-label="Clarify your reply"' in template


def test_monitoring_locks_assistant_mutations(assistant):
    state = assistant.server.state
    state.powgen_monitoring = True
    before = state.live_analysis_mode
    assistant.live_assistant_goal("rapid")
    asyncio.run(assistant.live_assistant_send("elements: Al"))
    assert state.live_analysis_mode == before and state.live_sample_elements == "Fe, O"
    assert not assistant.service.jobs


def test_empty_but_readable_experiment_can_fill_a_draft_without_claiming_ready(assistant, monkeypatch):
    def waiting(*args, **kwargs):
        raise NoPreviewData("No completed reduction yet; configuration can continue.")
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", waiting)
    monkeypatch.setattr("radar_pd_nova.app.preflight_powgen_experiment", lambda *a, **k: {
        "ready": False, "available_banks": [], "scan_count": 0,
        "message": "No completed reductions match the selection.",
    })
    state = assistant.server.state
    async def run():
        for _ in range(5):
            await assistant.live_assistant_continue()
            assert not state.live_assistant_error
        assert state.live_assistant_step == "review"
        await assistant.live_assistant_continue()
    asyncio.run(run())
    assert "No completed" in state.live_assistant_error
    assert not state.live_assistant_complete and not state.powgen_preflight_ready
    assert not assistant.service.saved and not assistant.service.jobs


def test_inaccessible_storage_is_not_treated_as_waiting_for_data(assistant, monkeypatch):
    def unavailable(*args, **kwargs):
        raise PermissionError("Experiment storage is inaccessible")
    monkeypatch.setattr("radar_pd_nova.live_assistant_controller.inspect_inventory", unavailable)
    asyncio.run(assistant.live_assistant_continue())
    assert assistant.server.state.live_assistant_step == "experiment"
    assert "inaccessible" in assistant.server.state.live_assistant_error
    assert "experiment" not in assistant._live_conversation.confirmed


def test_nonfinite_fit_limits_are_rejected_before_the_final_check(assistant):
    state = assistant.server.state
    assistant.live_assistant_go("settings")
    state.live_fit_start = "NaN"
    state.live_fit_end = 1000
    asyncio.run(assistant.live_assistant_continue())
    assert "finite" in state.live_assistant_error
    assert state.live_assistant_step == "settings"
    assert not assistant.service.jobs


def test_entirely_masked_curve_withholds_background_terms(tmp_path):
    scan, _, _ = product(tmp_path)
    path = tmp_path / "NOM_10.gsa"
    stat = path.stat()
    inventory = {"source_path": str(path), "root": str(tmp_path), "run_id": "NOM_10",
                 "instrument": "NOM", "ipts": "IPTS-10", "sample_id": "actual-sample",
                 "identity_status": "resolved", "setup": {},
                 "source_signature": [stat.st_size, stat.st_mtime_ns]}
    preview = preview_pattern(inventory, 1, limits=(1, 2))
    assert preview["valid_points"] == 0
    assert preview["recommendation"]["terms"] is None
    assert all(value is None for value in preview["y"])


def test_typo_correction_retains_ipts_and_needs_explicit_confirmation(assistant):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_send("I'm on powegn, IPTS 123"))
    assert state.live_assistant_step == "experiment"
    assert assistant._live_conversation.pending
    asyncio.run(assistant.live_assistant_send("what does background mean?"))
    assert assistant._live_conversation.pending  # Help can interrupt a decision.
    asyncio.run(assistant.live_assistant_send("POWGEN"))
    assert state.powgen_ipts == "IPTS-123" and state.live_assistant_step == "selection"
    assert not assistant.service.jobs and not assistant.service.saved


def test_clickable_suggestion_and_typed_confirmation_do_not_replay_after_edits(assistant):
    state = assistant.server.state
    assistant.live_assistant_go("settings")
    asyncio.run(assistant.live_assistant_send("balnaced"))
    choice = state.live_assistant_reply_choices[0]["id"]
    state.live_background_terms = 11
    state.flush()
    before = state.live_full_profile
    asyncio.run(assistant.live_assistant_choose_reply(choice))
    assert "changed" in state.live_assistant_error
    assert state.live_background_terms == 11 and state.live_full_profile == before
    asyncio.run(assistant.live_assistant_send("balnaced"))
    choice = state.live_assistant_reply_choices[0]["id"]
    asyncio.run(assistant.live_assistant_choose_reply(choice))
    assert state.live_full_profile == "balanced"
    assert not state.live_assistant_reply_choices


def test_unknown_reply_keeps_pending_values_and_cancel_has_no_edit(assistant):
    state = assistant.server.state
    before = state.powgen_ipts
    asyncio.run(assistant.live_assistant_send("powegn IPTS-456"))
    asyncio.run(assistant.live_assistant_send("some unrelated reply"))
    assert assistant._live_conversation.pending
    asyncio.run(assistant.live_assistant_send("no"))
    assert not assistant._live_conversation.pending and state.powgen_ipts == before


def test_stale_visible_question_token_cannot_select_a_reordered_sample(assistant):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_continue())
    token = state.live_assistant_context_token
    state.powgen_discovered_sample_options = list(reversed(state.powgen_discovered_sample_options))
    state.flush()
    assistant._live_assistant_changed()  # Exercise the callback before server.ready().
    assert token != state.live_assistant_context_token
    asyncio.run(assistant.live_assistant_send("the second one", token))
    assert "changed" in state.live_assistant_error
    assert state.powgen_processing_sample_ids == []
    asyncio.run(assistant.live_assistant_send("the second one", state.live_assistant_context_token))
    assert state.powgen_processing_sample_ids == ["A"]


def test_text_bank_and_chemistry_answers_share_live_state_and_preserve_single(assistant):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_continue())
    asyncio.run(assistant.live_assistant_send("samples A and B, bank 2"))
    assert state.powgen_processing_sample_ids == ["A", "B"] and state.powgen_processing_banks == [2]
    assistant.live_assistant_go("chemistry")
    asyncio.run(assistant.live_assistant_send("my sample contains iron and oxygen; can is aluminium"))
    assert state.live_sample_elements == "Fe, O" and state.live_environment_elements == "Al"
    assert state.sample_elements == "Single-pattern sentinel"
    asyncio.run(assistant.live_assistant_send("undo"))
    assert state.live_environment_elements == ""
    assert not assistant.service.jobs and not assistant.service.uploads


def test_beamline_only_correction_requires_a_fresh_ipts_and_nomad_mapping(assistant):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_send("actually NOMAD"))
    assert state.live_beamline == "NOM" and state.powgen_ipts == ""
    assert not state.nomad_profile_acknowledged and state.live_assistant_step == "experiment"
    assert not assistant.service.jobs


def test_saved_settings_and_monitoring_cannot_be_changed_through_suggestions(assistant):
    state = assistant.server.state
    state.live_configuration_source = "history"
    state.flush()
    before = state.live_sample_elements
    asyncio.run(assistant.live_assistant_send("elements: fe, o"))
    asyncio.run(assistant.live_assistant_send("yes"))
    assert "copy" in state.live_assistant_error and state.live_sample_elements == before
    state.live_configuration_source = "inline"
    state.flush()
    asyncio.run(assistant.live_assistant_send("balnaced"))
    choice = state.live_assistant_reply_choices[0]["id"]
    state.powgen_monitoring = True
    profile = state.live_full_profile
    asyncio.run(assistant.live_assistant_choose_reply(choice))
    assert state.live_full_profile == profile
    assert not assistant.service.jobs
