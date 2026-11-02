"""Large exclusion selections stay compact without altering selected run IDs."""
from radar_pd_nova.app import RadarPdNovaApp
from test_live_restore_polish import UiTree


def test_both_live_controls_bound_chips_and_keep_full_accessible_selection():
    app = RadarPdNovaApp()
    state = app.server.state
    state.powgen_excluded_runs = [f'NOM_{10000+i}' for i in range(55)]
    state.powgen_run_options = [
        {'value': run, 'title': run + ' / a long original source title with specimen and reduction details'}
        for run in state.powgen_excluded_runs
    ]
    state.flush()
    before = list(state.powgen_excluded_runs)
    nodes = UiTree(app.layout.html).nodes
    controls = [n for n in nodes if n['tag'] == 'vautocomplete' and n['attrs'].get('v-model') == 'powgen_excluded_runs']
    assert len(controls) == 2
    for control in controls:
        assert control['attrs']['itemvalue'] == 'value'
        assert 'powgen_processing_sample_ids.includes(r.sample_id)' in control['attrs'][':items']
        slot = next(n for n in nodes if n['parent'] is control and n['attrs'].get('v-slot:chip'))
        assert slot['attrs']['v-slot:chip'] == '{ props, item, index }'
        chip = next(n for n in nodes if n['parent'] is slot and n['tag'] == 'vchip')
        assert chip['attrs']['v-if'] == 'index < 2' and chip['attrs']['v-bind'] == 'props'
        assert chip['attrs'][':aria-label'] == 'item.title'
        count = next(n for n in nodes if n['parent'] is slot and n['tag'] == 'span')
        assert count['attrs']['v-if'] == 'index === 2'
        assert 'powgen_excluded_runs.length - 2' in count['text']
    disclosures = [n for n in nodes if n['tag'] == 'summary' and 'excluded entries' in n['text']]
    assert len(disclosures) == 2
    for summary in disclosures:
        details = summary['parent']
        assert details['tag'] == 'details' and 'open' not in details['attrs']
        region = next(n for n in nodes if n['parent'] is details and n['attrs'].get('role') == 'region')
        assert region['attrs']['tabindex'] == '0'
    full_rows = [n for n in nodes if n['attrs'].get('v-for') == 'run in powgen_excluded_runs']
    assert len(full_rows) == 2
    assert all('title || run' in n['text'] and 'v-html' not in n['attrs'] for n in full_rows)
    assert state.powgen_excluded_runs == before
    assert app._powgen_controller is None and not state.powgen_monitoring
