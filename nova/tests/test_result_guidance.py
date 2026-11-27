"""Displayed guidance must reflect published artifacts and the reviewed plan."""
from copy import deepcopy
from html.parser import HTMLParser
from types import SimpleNamespace

import pytest

from radar_pd_nova.recipe_presentation import failure_policy_notice
from radar_pd_nova.refinement_controller import RefinementMixin
from radar_pd_nova.results import discover_tables


@pytest.mark.parametrize('policy,expected', [
    ('stop', 'stop this recipe'),
    ('continue_independent', 'restore the last verified checkpoint'),
])
def test_policy_notice_preserves_the_actual_plan(policy, expected):
    plan={'failure_policy':policy,'stages':[{'id':'first'}]};before=deepcopy(plan)
    text=failure_policy_notice(plan)
    assert expected in text and plan==before
    if policy=='continue_independent':
        assert 'continue independent stages' in text and 'depend on the rejected stage' in text
        assert 'Technical errors still stop' in text


def test_legacy_and_missing_draft_do_not_invent_a_new_policy():
    assert 'stop this recipe' in failure_policy_notice({'stages':[{'id':'old'}]})
    assert failure_policy_notice(None)==failure_policy_notice({})==''
    assert 'unavailable' in failure_policy_notice({'failure_policy':'unknown'})


@pytest.mark.parametrize('setup',['guided','expert'])
@pytest.mark.parametrize('policy',['stop','continue_independent'])
def test_review_uses_displayed_draft_in_both_setup_modes(setup,policy,monkeypatch):
    from radar_pd_nova import refinement_controller as controller
    plan={'failure_policy':policy,'assumptions':[],'questions':[],'unsupported':[]}
    draft={'plan':plan,'context':{},'source_id':'source','request':'same request','status':'ready','guided':setup=='guided'}
    before=deepcopy(draft)
    state=SimpleNamespace(rf_source='source',rf_request='same request',rf_setup_mode=setup,
        rf_failure_policy=('stop' if policy=='continue_independent' else 'continue_independent') if setup=='guided' else policy)
    app=RefinementMixin();app.server=SimpleNamespace(state=state)
    app._refinement_workspace=SimpleNamespace(state={'sources':{'source':{}}})
    monkeypatch.setattr(controller,'recipe_cards',lambda *a,**k:[{'id':'stage'}])
    app._refinement_sync_preview(draft)
    assert state.rf_plan_failure_notice==failure_policy_notice(plan)
    assert state.rf_cards==[{'id':'stage'}] and draft==before
    # A changed request must not retain the old plan's guidance.
    state.rf_request='new request';app._refinement_sync_preview(draft)
    assert state.rf_plan_failure_notice=='' and state.rf_cards==[]


def test_existing_csv_choices_remain_available_for_compact_results(tmp_path):
    table=tmp_path/'tables'/'Final_phase_fractions.csv';table.parent.mkdir()
    table.write_text('phase,fraction\nOxide,100\n')
    result=discover_tables(tmp_path)
    assert len(result)==1 and result[0]['path']==str(table)
    assert result[0]['primary'] is True and result[0]['stage']=='phase_fractions'


def test_template_hides_empty_controls_and_shows_policy_outside_expert_only_form():
    from radar_pd_nova.app import RadarPdNovaApp
    class Elements(HTMLParser):
        def __init__(self):super().__init__();self.stack=[];self.found={}
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if attrs.get('label')=='Published result table':self.found['table_selector']=[*self.stack,(tag,attrs)]
            if attrs.get(':headers')=='table_headers':self.found['table_grid']=[*self.stack,(tag,attrs)]
            if attrs.get('aria-label')=='Planned failure handling':self.found['policy']=[*self.stack,(tag,attrs)]
            self.stack.append((tag,attrs))
        def handle_endtag(self,tag):
            for i in range(len(self.stack)-1,-1,-1):
                if self.stack[i][0]==tag:self.stack=self.stack[:i];break
    template=RadarPdNovaApp().layout.html;parsed=Elements();parsed.feed(template)
    assert 'No technical tables were published for this compact Live result.' in template
    assert 'Published result tables are still loading.' in template
    for key in ('table_selector','table_grid'):
        assert any(a.get('v-if')=='table_options.length' for _,a in parsed.found[key])
    policy=parsed.found['policy']
    assert policy[-1][1].get('v-if')=='!!rf_plan_failure_notice'
    assert not any('rf_setup_mode' in a.get('v-if','') or 'rf_setup_mode' in a.get('v-show','') for _,a in policy)
    assert '{{ rf_plan_failure_notice }}' in template


def test_clearing_refinement_context_clears_displayed_policy():
    state=SimpleNamespace(rf_experiment='',rf_plan_failure_notice='Old plan')
    app=RefinementMixin();app.server=SimpleNamespace(state=state);app._refinement_workspace=None
    app._sync_recipe_preparation_progress=lambda:None
    app._clear_refinement_output=lambda **kwargs:None
    app._refinement_sync()
    assert state.rf_plan_failure_notice==''
