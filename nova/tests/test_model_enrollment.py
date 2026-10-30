"""Enrollment scope stays fixed while acquisition and viewer state change."""
import asyncio
from types import SimpleNamespace

from gpx_recipe.symmetry.campaign import Campaign
from radar_pd_nova import model_campaigns as module
from radar_pd_nova.symmetry_controller import SymmetryMixin
from radar_pd_nova.symmetry_workspace import campaign_can_resume


def setup(tmp_path,monkeypatch):
    c=Campaign(tmp_path/'campaign')
    # Contract and native compatibility have independent worker tests; this
    # fixture isolates the queue's historical/future acquisition policy.
    t=dict(scope=dict(experiment='IPTS-1',sample_key='specimen:A',bank='2'),origin='radar',
        existing_ids=['selected'],seen_ids=['selected','unselected'],maximum_scans=20)
    c.set_setting('template',t);c.set_setting('following',True)
    monkeypatch.setattr(module,'validate_template',lambda template:template)
    def source(key,**kwargs):
        return dict(id=key,experiment='IPTS-1',sample_key='specimen:A',bank='2',scan_label=key,sample_label='A',**kwargs)
    rows=[(source('selected'),'radar'),(source('unselected'),'radar'),(source('new'),'radar')]
    for field,value in [('experiment','IPTS-2'),('sample_key','specimen:B'),('bank','3')]:
        row=source('wrong-'+field);row[field]=value;rows.append((row,'radar'))
    rows.append((source('other-recipe'),'recipe:other:final'))
    monkeypatch.setattr(module,'choices',lambda *args:rows)
    return c,rows


def test_future_following_adds_only_new_matching_scans_not_old_unselected(tmp_path,monkeypatch):
    c,rows=setup(tmp_path,monkeypatch)
    assert set(module.queue_sources(c,None))=={'selected','new'}
    assert set(module.queue_sources(c,None))=={'selected','new'}
    c.set_setting('following',False)
    rows.append((dict(rows[0][0],id='later'),'radar'))
    assert set(module.queue_sources(c,None))=={'selected','new'}


def test_budget_stops_following_with_visible_reason(tmp_path,monkeypatch):
    c,rows=setup(tmp_path,monkeypatch);t=c.setting('template');t['maximum_scans']=1;c.set_setting('template',t)
    assert set(module.queue_sources(c,None))=={'selected'}
    assert c.setting('following') is False and 'budget reached' in c.setting('budget_message')


def test_recover_queued_acquisitions_before_the_first_trial_is_created(tmp_path,monkeypatch):
    c,_=setup(tmp_path,monkeypatch);c.set_setting('following',False)
    entries=module.queue_sources(c,None)
    assert not c.trials() and not c.summaries()
    assert campaign_can_resume(c.summaries(),list(entries.values()))
    entries['selected']['status']='preparing'
    assert campaign_can_resume(c.summaries(),list(entries.values()))
    entries['selected']['status']='needs_review'
    assert not campaign_can_resume(c.summaries(),list(entries.values()))


def test_session_close_prevents_dispatch_and_retains_follow_policy_for_explicit_continue(tmp_path,monkeypatch):
    c,rows=setup(tmp_path,monkeypatch)
    app=SymmetryMixin();app._model_preparations={};app._k_cancel=None;app._symmetry_task=None
    app._model_enrollment_task=None;app._model_tick_task=None;app._model_remote=lambda:None
    app._model_active_campaigns={'campaign':(c,SimpleNamespace())}
    asyncio.run(app._pause_model_campaigns())
    assert app._model_stopping and c.setting('paused') and c.setting('following')
    assert module.queue_sources(c,None)=={}


def test_switching_exploration_kind_removes_stale_candidate_choices():
    s=SimpleNamespace(sx_kind='structural',sx_parent='Parent',sx_k='0, 0, 0',sx_start_source='scan',
        sx_opd_policy='all',sx_construct_limit=50,sx_discovered=True,sx_candidates=['old'],sx_candidate_options=['old'])
    app=SymmetryMixin();app.server=SimpleNamespace(state=s)
    app._symmetry_draft={'discovery_identity':('magnetic','Parent','0, 0, 0','scan')}
    app._symmetry_unapprove()
    assert not s.sx_discovered and not s.sx_candidates and not s.sx_candidate_options
