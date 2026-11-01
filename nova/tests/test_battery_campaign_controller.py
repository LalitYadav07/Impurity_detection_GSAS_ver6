import asyncio
from types import SimpleNamespace

import pytest

from radar_pd_nova import battery_campaign_controller as module
from radar_pd_nova.battery_controller import BatteryMixin
from test_battery_preparation_controller import State


@pytest.fixture
def app():
    result=BatteryMixin();result.server=SimpleNamespace(state=State());result._init_battery()
    return result


def prepared(phase='oxide',bank=2):
    return dict(banks=[dict(key='b',physical_bank=bank)],
                phases=[dict(name=phase,links=[dict(bank='b')])])


def test_width_choices_require_a_common_physical_link_not_a_local_bank_label():
    assert module.common_width_choices([prepared(),prepared(bank=5)])==[]
    assert module.common_width_choices([prepared(),prepared(phase='other')])==[]
    choices=module.common_width_choices([prepared(),prepared()])
    assert choices[0]['phase']=='oxide' and choices[0]['physical_bank']==2
    assert choices[0]['title']=='oxide · physical bank 2'


def test_changed_trial_set_removes_unshared_width_choices(app,monkeypatch):
    monkeypatch.setattr(module,'verify_prepared',lambda path:prepared(bank=2 if path=='first' else 5))
    app._battery_update_width_choices([dict(model='first')])
    key=app.server.state.battery_portfolio_width_options[0]['value']
    app.server.state.battery_portfolio_sizes=[key]
    assert app._battery_refinement_policy()['sample_widths'][0]['physical_bank']==2
    app._battery_update_width_choices([dict(model='first'),dict(model='second')])
    assert app.server.state.battery_portfolio_sizes==[]
    assert app.server.state.battery_portfolio_width_options==[]
    app.server.state.battery_portfolio_sizes=[key]
    with pytest.raises(ValueError,match='no longer shared'):app._battery_refinement_policy()


@pytest.mark.parametrize('value',[True,None,6.5,'6.5','0','37','nan'])
def test_bad_background_count_does_not_silently_choose_another_policy(app,value):
    app.server.state.battery_portfolio_background='chebyshev'
    app.server.state.battery_portfolio_bg_coefficients=value
    with pytest.raises(ValueError):app._battery_refinement_policy()


def test_late_preflight_cannot_restore_approval_after_a_policy_edit(app,monkeypatch):
    revision=app._battery_portfolio_revision
    def delayed(*args,**kwargs):
        app.server.state.battery_portfolio_bg_coefficients='12'
        app._battery_portfolio_edited()
        return 'old-token',{}
    monkeypatch.setattr(module,'prepare_portfolio',delayed)
    asyncio.run(app._review_battery_portfolio_async(None,[], 'note', .1, None,
                                                   app._battery_refinement_policy(),revision))
    assert app._battery_portfolio_token is None
    assert not app.server.state.battery_portfolio_ready
    assert not app.server.state.battery_busy


def test_failed_preflight_clears_checking_message_and_keeps_approval_unavailable(app,monkeypatch):
    def fail(*args,**kwargs):raise ValueError('Native model could not be verified')
    monkeypatch.setattr(module,'prepare_portfolio',fail)
    asyncio.run(app._review_battery_portfolio_async(None,[], 'note', .1, None,
                    app._battery_refinement_policy(),app._battery_portfolio_revision))
    assert app._battery_portfolio_token is None
    assert not app.server.state.battery_portfolio_ready
    assert not app.server.state.battery_busy
    assert app.server.state.battery_portfolio_message=='Preflight stopped: Native model could not be verified. No trials were queued.'


def test_failed_trial_removal_does_not_leave_a_partly_changed_form(app,monkeypatch):
    app._battery_portfolio_entries=[dict(model='first'),dict(model='second')]
    app._battery_portfolio_token='reviewed'
    def fail(*args,**kwargs):raise ValueError('Prepared revision changed')
    monkeypatch.setattr(module,'verify_prepared',fail)
    app.remove_battery_trial(0)
    assert app._battery_portfolio_entries==[dict(model='first'),dict(model='second')]
    assert app._battery_portfolio_token=='reviewed'
    assert app.server.state.battery_error=='Prepared revision changed'


def test_li_policy_is_explicit_and_default_does_not_release_any_atom(app):
    assert app._battery_li_profile() is None
    s=app.server.state;s.battery_li_enabled=True;s.battery_li_phase='oxide';s.battery_li_atom='Li1'
    profile=app._battery_li_profile()
    assert profile['mixing_atom'] is None and profile['refine_cells'] is True
    s.battery_li_mixing_atom='Ni_Li';s.battery_li_refine_cells=False
    assert app._battery_li_profile()['mixing_atom']=='Ni_Li'
    assert app._battery_li_profile()['refine_cells'] is False
    s.battery_li_atom=''
    with pytest.raises(ValueError):app._battery_li_profile()


def test_policy_edit_clears_a_previous_li_result_and_approval(app):
    s=app.server.state;s.battery_li_result_ready=True;s.battery_li_points=[{'occupancy':.65}]
    s.battery_li_notice='Old preference';app._battery_portfolio_token='old'
    app._battery_portfolio_edited()
    assert not s.battery_li_result_ready and s.battery_li_points==[] and not s.battery_li_notice
    assert app._battery_portfolio_token is None


def test_late_li_preflight_cannot_restore_approval_after_a_target_edit(app,monkeypatch):
    s=app.server.state;s.battery_li_enabled=True;s.battery_li_phase='oxide';s.battery_li_atom='Li1'
    profile=app._battery_li_profile();revision=app._battery_portfolio_revision
    def delayed(*args,**kwargs):
        assert kwargs['li_profile']==profile
        s.battery_li_atom='Li2';app._battery_portfolio_edited()
        return 'old-token',{}
    monkeypatch.setattr(module,'prepare_portfolio',delayed)
    asyncio.run(app._review_battery_portfolio_async(None,[], 'note',.1,None,app._battery_refinement_policy(),revision,li_profile=profile))
    assert app._battery_portfolio_token is None and not s.battery_portfolio_ready
