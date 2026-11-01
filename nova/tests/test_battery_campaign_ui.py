"""Catch template failures that can blank the entire NOVA application."""
from html.parser import HTMLParser

from radar_pd_nova.app import RadarPdNovaApp


def test_trial_action_expressions_survive_html_serialization():
    class Buttons(HTMLParser):
        def __init__(self):
            super().__init__();self.rows=[]
        def handle_starttag(self,tag,attrs):
            values=dict(attrs)
            if tag=='vbtn' and 'battery_portfolio_trials' in values.get('v-for',''):
                self.rows.append(values)
    template=RadarPdNovaApp().layout.html
    buttons=Buttons();buttons.feed(template)
    assert len(buttons.rows)==2
    retry=next(r for r in buttons.rows if 'status' in r['v-for'])
    assert "'failed'" in retry['v-for'] and "'interrupted'" in retry['v-for']
    assert retry[':key']=="'retry-'+trial.id"
    assert 'Inspect {{ trial.family }} / {{ trial.seed }}' in template
    assert 'Retry {{ trial.family }} / {{ trial.seed }}' in template
    assert 'Pause queued trials' in template and 'Resume paused trials' in template
    assert 'Optional held-Li sensitivity' in template
    assert 'Held-Li atom label' in template
    assert 'battery_li_result_ready' in template and 'battery_li_figure.data' in template
    assert 'does not identify Li composition' in template
    assert 'Build or revise a comparison' in template


def test_li_curve_keeps_unqualified_markers_and_gaps():
    from radar_pd_nova.refinement_plots import li_profile_figure
    points=[dict(occupancy=x,minimum_wss_per_point=y) for x,y in ((.45,2.),(.65,None),(.85,3.))]
    trials=[dict(li_occupancy=.45,wss_per_point=2.,convergence={'status':'passed'}),
            dict(li_occupancy=.65,wss_per_point=None),
            dict(li_occupancy=.85,wss_per_point=3.,convergence={'status':'needs_review'})]
    figure=li_profile_figure({'points':points},trials).to_plotly_json()
    assert [r['name'] for r in figure['data']]==['Starts passing optimization screens','Unqualified completed starts','Grid minima (guide)']
    assert list(figure['data'][-1]['y'])==[2.,None,3.]
    assert figure['data'][-1]['connectgaps'] is False


def test_li_plot_does_not_amplify_roundoff_into_a_visible_minimum():
    from radar_pd_nova.refinement_plots import li_profile_figure
    values=[1.,1.-1e-15,1.]
    points=[dict(occupancy=x,minimum_wss_per_point=y) for x,y in zip((.45,.65,.85),values)]
    trials=[dict(li_occupancy=p['occupancy'],wss_per_point=p['minimum_wss_per_point'],convergence={'status':'passed'}) for p in points]
    figure=li_profile_figure({'points':points,'screening_band':.1},trials,revision='review-a').to_plotly_json()
    lower,upper=figure['layout']['yaxis']['range']
    assert upper-lower>=.1
    assert figure['layout']['uirevision'].endswith('/review-a')
    assert figure['data'][0]['y'][1]==1.-1e-15
