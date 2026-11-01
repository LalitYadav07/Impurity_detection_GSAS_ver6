"""Actual-user Guided Live requests preserve complete, bounded draft edits."""

import asyncio
from copy import deepcopy

import pytest

from radar_pd_nova.live_assistant_language import ReplyContext, SampleOption, interpret_reply
from radar_pd_nova.setup_forms import full_profile_settings
from test_live_assistant import assistant  # noqa: F401 -- shared ordinary-app fixture


SAMPLES = (SampleOption('101', '101 — Synthetic oxide'),)
CONTEXT = ReplyContext(step='selection', samples=SAMPLES, shown_samples=SAMPLES,
                       banks=(2,), defer_sample_bank_membership=True)


@pytest.mark.parametrize('text', [
    'Full analysis; one discovery round',
    'Full analysis with one impurity discovery pass.',
    'Use Full analysis, and one discovery round.',
])
def test_explicit_full_and_discovery_budget_form_one_complete_patch(text):
    reply = interpret_reply(text, CONTEXT)
    assert reply.action == 'patch'
    assert reply.value == {'live_analysis_mode': 'full', 'live_full_max_passes': 1}


@pytest.mark.parametrize('count', ['0', '01', '-1', '1.5', '1e1', '21', 'many'])
def test_discovery_count_is_an_exact_bounded_integer(count):
    reply = interpret_reply(f'Full analysis; {count} discovery rounds', CONTEXT)
    assert reply.action == 'clarify' and reply.value is None


@pytest.mark.parametrize(('text', 'number'), [('one discovery pass', 1), ('20 discovery rounds', 20),
                                            ('twenty discovery passes', 20)])
def test_discovery_count_boundary_values_and_number_words(text, number):
    assert interpret_reply(text, CONTEXT).value == {'live_full_max_passes': number}


def test_preset_helper_returns_an_independent_copy():
    values = full_profile_settings('thorough')
    values['full_top_n_ml'] = -1
    assert full_profile_settings('thorough')['full_top_n_ml'] == 75
    assert full_profile_settings('custom') == {}


def test_mode_and_budget_do_not_swallow_unknown_scientific_instructions():
    reply = interpret_reply('Full analysis with one discovery round followed by any lattice constraints', CONTEXT)
    assert reply.action == 'clarify' and reply.value is None
    assert interpret_reply('Full analysis; one discovery round; launch analysis', CONTEXT).action == 'clarify'


def test_setting_like_sample_names_still_use_exact_discovered_identity():
    samples = (SampleOption('A', 'A — Full analysis'), SampleOption('B', 'B — one discovery round'))
    context = ReplyContext(step='selection', samples=samples, shown_samples=samples, banks=(2,))
    assert interpret_reply('Full analysis', context).value['powgen_processing_sample_ids'] == ['A']
    assert interpret_reply('one discovery round', context).value['powgen_processing_sample_ids'] == ['B']


def test_explicit_context_selection_and_settings_keep_every_clause():
    reply = interpret_reply('POWGEN IPTS-99999, sample ID 101, physical bank 2; Full analysis; one discovery round', CONTEXT)
    assert reply.action == 'patch'
    assert reply.value == {'live_beamline': 'PG3', 'powgen_ipts': 'IPTS-99999',
                           'powgen_processing_scope': 'selected', 'powgen_processing_sample_ids': ['101'],
                           'powgen_bank_scope': 'selected', 'powgen_processing_banks': [2],
                           'live_analysis_mode': 'full', 'live_full_max_passes': 1}


@pytest.mark.parametrize('text', [
    'POWGEN IPTS-99999; sample ID 101; bank 2; Full analysis; one discovery round; start monitoring',
    'POWGEN IPTS-99999; sample ID 101x; bank 2; Full analysis; one discovery round',
    'POWGEN IPTS-99999; IPTS-99998; sample ID 101; bank 2',
    'Full analysis; Rapid analysis; one discovery round',
    'one discovery round; two discovery rounds',
])
def test_ambiguous_conflicting_or_unhandled_requests_are_not_partial_patches(text):
    reply = interpret_reply(text, CONTEXT)
    assert reply.action == 'clarify' and reply.value is None


def test_mixed_scientific_narrative_remains_explicitly_unapplied():
    reply = interpret_reply(
        'POWGEN IPTS-99999, sample 101 (synthetic oxide, Al2O3). '
        'Use all six existing scans and physical bank 2. I have built the custom library synthetic_oxide_library. '
        'I want Full analysis with one impurity discovery pass, followed by lattice parameters a and c '
        'and relative changes versus measured temperature. Keep atomic positions, occupancies and '
        'displacement parameters fixed for the initial analysis.', CONTEXT)
    assert reply.action == 'clarify' and reply.value is None
    assert 'not been applied' in reply.message
    assert 'advanced Live settings' in reply.message


@pytest.mark.parametrize('profile', ['quick', 'balanced', 'thorough', 'custom'])
def test_round_override_preserves_every_other_effective_setting(assistant, profile):
    state = assistant.server.state
    state.live_full_profile = profile
    state.live_full_top_n_ml = 97
    state.live_full_nudge_candidates = 13
    state.live_full_max_passes = 7
    state.live_background_terms = 11
    state.live_fit_start, state.live_fit_end = 100, 10000
    before = assistant._live_configuration().model_dump()
    asyncio.run(assistant.live_assistant_send('Full analysis; one discovery round'))
    assert not state.live_assistant_error
    after = assistant._live_configuration().model_dump()
    assert state.live_full_profile == 'custom' and after['full_max_passes'] == 1
    assert {k: v for k, v in after.items() if k not in {'full_max_passes', 'full_profile'}} == {
        k: v for k, v in before.items() if k not in {'full_max_passes', 'full_profile'}}
    assert not assistant.service.jobs and not assistant.service.saved and not assistant.service.uploads
    asyncio.run(assistant.live_assistant_send('undo'))
    assert assistant._live_configuration().model_dump() == before


def test_budget_requires_full_mode_or_an_explicit_mode_change(assistant):
    state = assistant.server.state
    state.live_analysis_mode = 'rapid'
    before = deepcopy(assistant._assistant_snapshots())
    asyncio.run(assistant.live_assistant_send('one discovery round'))
    assert 'Full analysis' in state.live_assistant_error
    assert assistant._assistant_snapshots() == before
    asyncio.run(assistant.live_assistant_send('Full analysis; one discovery round'))
    assert not state.live_assistant_error and state.live_analysis_mode == 'full'
    assert assistant._live_configuration().full_max_passes == 1


def test_explicit_rapid_with_full_discovery_budget_is_not_partially_applied(assistant):
    state = assistant.server.state
    before = deepcopy(assistant._assistant_snapshots())
    asyncio.run(assistant.live_assistant_send('Rapid analysis; one discovery round'))
    assert 'Full analysis' in state.live_assistant_error
    assert assistant._assistant_snapshots() == before


def test_current_context_header_does_not_reset_the_discovered_experiment(assistant, monkeypatch):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_continue())
    assistant.live_assistant_go('selection')
    state.powgen_instrument_dataset_id = 'retained-profile'
    def unexpected_reset(*args, **kwargs):
        raise AssertionError('A repeated current header must not reset context or discovery')
    monkeypatch.setattr(assistant, '_live_beamline_changed', unexpected_reset)
    monkeypatch.setattr(assistant, '_powgen_ipts_changed', unexpected_reset)
    asyncio.run(assistant.live_assistant_send('POWGEN IPTS-123; sample ID A; physical bank 2; Full analysis; one discovery round'))
    assert not state.live_assistant_error
    assert state.powgen_processing_sample_ids == ['A'] and state.powgen_processing_banks == [2]
    assert state.powgen_instrument_dataset_id == 'retained-profile'
    assert assistant._live_configuration().full_max_passes == 1
    assert not assistant.service.jobs


@pytest.mark.parametrize('header', ['POWGEN IPTS-456', 'NOMAD IPTS-123'])
def test_changed_context_cannot_borrow_the_old_sample_inventory(assistant, header):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_continue())
    before = deepcopy(assistant._assistant_snapshots())
    asyncio.run(assistant.live_assistant_send(header + '; sample ID A; bank 2; Full analysis; one discovery round'))
    assert 'experiment first' in state.live_assistant_error.lower()
    assert assistant._assistant_snapshots() == before
    assert not assistant.service.jobs and not assistant.service.uploads


def test_unavailable_bank_rejects_all_compound_edits(assistant):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_continue())
    before = deepcopy(assistant._assistant_snapshots())
    asyncio.run(assistant.live_assistant_send('POWGEN IPTS-123; sample ID A; bank 9; Full analysis; one discovery round'))
    assert 'bank' in state.live_assistant_error.lower()
    assert assistant._assistant_snapshots() == before


def test_saved_configuration_still_requires_explicit_edit_copy(assistant):
    state = assistant.server.state
    state.live_configuration_source = 'history'
    before = deepcopy(assistant._assistant_snapshots())
    asyncio.run(assistant.live_assistant_send('Full analysis; one discovery round'))
    assert 'Edit a copy' in state.live_assistant_error
    assert assistant._assistant_snapshots() == before


def test_mode_only_keeps_the_chosen_preset_and_custom_budget(assistant):
    state = assistant.server.state
    state.live_analysis_mode = 'rapid'
    state.live_full_profile = 'custom'
    state.live_full_max_passes = 7
    asyncio.run(assistant.live_assistant_send('Full analysis'))
    assert state.live_analysis_mode == 'full' and state.live_full_profile == 'custom'
    assert state.live_full_max_passes == 7


def test_safe_setting_edit_does_not_require_unrelated_draft_fields_complete(assistant):
    state = assistant.server.state
    state.live_sample_elements = ''
    state.live_fit_start, state.live_fit_end = 100, None
    asyncio.run(assistant.live_assistant_send('Full analysis; one discovery round'))
    assert not state.live_assistant_error
    assert state.live_full_max_passes == 1 and state.live_full_profile == 'custom'
    assert state.live_sample_elements == '' and state.live_fit_end is None
    assert not assistant.service.jobs


def test_explicit_preset_and_round_limit_resolve_together(assistant):
    state = assistant.server.state
    state.live_full_profile = 'quick'
    asyncio.run(assistant.live_assistant_send('thorough; one discovery round'))
    assert not state.live_assistant_error
    config = assistant._live_configuration()
    assert config.full_max_passes == 1 and config.full_top_n_ml == 75
    assert config.full_compare_cycles == 8


def test_manual_edit_during_inventory_prevents_stale_compound_application(assistant, monkeypatch):
    state = assistant.server.state
    asyncio.run(assistant.live_assistant_continue())

    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def inspect(**kwargs):
            entered.set()
            await release.wait()
            return {'banks': [{'title': 'Bank 2', 'value': 2}]}
        monkeypatch.setattr(assistant, '_assistant_inspect', inspect)
        task = asyncio.create_task(assistant.live_assistant_send('sample ID A; bank 2; Full analysis; one discovery round'))
        await entered.wait()
        state.live_full_top_n_ml = 99
        state.flush()
        release.set()
        await task

    asyncio.run(run())
    assert 'setup changed' in state.live_assistant_error.lower()
    assert state.powgen_processing_sample_ids == []
    assert state.live_full_top_n_ml == 99 and state.live_full_profile == 'balanced'
    assert not state.live_assistant_busy and not assistant.service.jobs


def test_help_lists_supported_budget_command_and_scope_limits(assistant):
    template = assistant.layout.html
    assert 'Full analysis; one discovery round' in template
    assert 'maximum per-pattern' in template
    assert 'does not start' in template
