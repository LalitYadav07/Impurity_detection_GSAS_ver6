"""An unselected reduction family must not look like an empty experiment."""
import asyncio
from dataclasses import replace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.live_beamlines import BEAMLINES


@pytest.mark.parametrize('selected', ['', '["all",""]'])
def test_snap_variant_choice_explains_why_the_inventory_is_empty(monkeypatch,selected):
    app=RadarPdNovaApp();state=app.server.state
    state.live_beamline='SNAP';app._live_beamline_changed()
    state.powgen_ipts='1';app._powgen_ipts_changed()
    app._powgen_sample_index=lambda _:object()
    result=dict(options=[],run_options=[],scan_count=1 if selected else 0,
        unidentified_count=1 if selected else 0,
        product_family=selected,product_family_options=[
            dict(value='["all",""]',title='Standard export'),
            dict(value='["all","autoreduce"]',title='Autoreduce export')])
    monkeypatch.setitem(BEAMLINES,'SNAP',replace(BEAMLINES['SNAP'],
        discover_samples=lambda *args,**kwargs:result))
    returned=asyncio.run(app.refresh_powgen_samples('1'))
    assert returned is result and state.snap_product_family_options==result['product_family_options']
    assert not state.powgen_sample_loading
    if selected:
        assert 'across 1 completed scans' in state.powgen_sample_message
        assert 'no readable sample ID' in state.powgen_sample_message
        assert state.snap_product_family==selected
    else:
        assert state.powgen_sample_message.startswith('Found 2 SNAP export variants.')
        assert 'Choose a SNAP grouping / export variant below' in state.powgen_sample_message
        assert '0 completed scans' not in state.powgen_sample_message
        assert not state.snap_product_family


def test_empty_snap_folder_is_not_misrepresented_as_a_variant_choice(monkeypatch):
    app=RadarPdNovaApp();state=app.server.state
    state.live_beamline='SNAP';app._live_beamline_changed()
    state.powgen_ipts='1';app._powgen_ipts_changed()
    app._powgen_sample_index=lambda _:object()
    result=dict(options=[],run_options=[],scan_count=0,unidentified_count=0,
        product_family='',product_family_options=[])
    monkeypatch.setitem(BEAMLINES,'SNAP',replace(BEAMLINES['SNAP'],
        discover_samples=lambda *args,**kwargs:result))
    asyncio.run(app.refresh_powgen_samples('1'))
    assert '0 completed scans' in state.powgen_sample_message
    assert 'export variants' not in state.powgen_sample_message
