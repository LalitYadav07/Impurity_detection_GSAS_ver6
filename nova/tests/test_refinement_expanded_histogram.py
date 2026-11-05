"""An expanded fit identifies its selected, verified histogram."""
from html.parser import HTMLParser
import copy
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.refinement_controller import RefinementMixin, refinement_histogram_options, figure_json
from radar_pd_nova.refinement_plots import fit_figure
from test_refinement_plots import view


def test_expanded_histogram_caption_uses_the_same_physical_bank_options_as_the_selector():
    app = RadarPdNovaApp()
    view = {'histograms':[{'name':'PWDR b2.gsa Bank 1'}, {'name':'PWDR b5.gsa Bank 1'}]}
    source = {'metadata':{'physical_banks':{'PWDR b2.gsa Bank 1':2, 'PWDR b5.gsa Bank 1':5}}}
    options = refinement_histogram_options(view, source)
    assert options == [
        {'title':'Physical bank 2', 'value':'PWDR b2.gsa Bank 1'},
        {'title':'Physical bank 5', 'value':'PWDR b5.gsa Bank 1'},
    ]
    class Expanded(HTMLParser):
        def __init__(self):
            super().__init__(); self.inside = 0; self.caption = ''; self.caption_attrs = None; self.capture = False
        def handle_starttag(self, tag, attrs):
            values = dict(attrs)
            if tag == 'div':
                if self.inside: self.inside += 1
                elif values.get('v-if') == 'rf_expanded': self.inside = 1
            if self.inside and tag == 'p':
                self.capture = True; self.current_attrs = values
        def handle_data(self, data):
            if self.capture and data.strip().startswith('Histogram:'):
                self.caption = data.strip(); self.caption_attrs = self.current_attrs
        def handle_endtag(self, tag):
            if tag == 'p': self.capture = False
            if tag == 'div' and self.inside: self.inside -= 1
    expanded = Expanded(); expanded.feed(app.layout.html)
    assert expanded.caption == ('Histogram: {{ rf_fit.layout.meta.refinement_histogram.title || rf_fit.layout.meta.refinement_histogram.name }}')
    assert expanded.caption_attrs['v-if'] == "rf_result_tab !== 'trends' && rf_fit.layout && rf_fit.layout.meta && rf_fit.layout.meta.refinement_histogram"
    assert 'rf_histograms.find' not in expanded.caption
    # A missing physical-bank mapping still uses the native identity verbatim.
    assert refinement_histogram_options(view,{})[1] == {'title':'PWDR b5.gsa Bank 1','value':'PWDR b5.gsa Bank 1'}
    # An expanded trend must use the trend rather than the selected scan's fit.
    assert ':data="rf_result_tab === \'trends\' ? rf_trend.data : rf_fit.data"' in app.layout.html
    assert ':layout="rf_result_tab === \'trends\' ? rf_trend.layout : rf_fit.layout"' in app.layout.html


def multibank_view():
    result=view()
    result['histograms'][0]['name']='PWDR b2.gsa Bank 1'
    result['histograms'].append(copy.deepcopy(result['histograms'][0]))
    result['histograms'][1].update(name='PWDR b5.gsa Bank 1',observed=[2.,40.,2.],calculated=[2.,36.,2.])
    return result


def test_selector_change_cannot_relabel_the_committed_bank2_figure_before_bank5_is_built():
    result=multibank_view()
    bank2='PWDR b2.gsa Bank 1';bank5='PWDR b5.gsa Bank 1'
    plain,_=fit_figure(result,result,histogram=bank2)
    first,_=fit_figure(result,result,histogram=bank2,histogram_label='Physical bank 2')
    assert figure_json(first)['data']==figure_json(plain)['data']
    committed=figure_json(first)
    selected=bank5
    assert selected != committed['layout']['meta']['refinement_histogram']['name']
    assert committed['layout']['meta']['refinement_histogram']=={'name':bank2,'title':'Physical bank 2'}
    second,_=fit_figure(result,result,histogram=selected,histogram_label='Physical bank 5')
    committed=figure_json(second)
    assert committed['layout']['meta']['refinement_histogram']=={'name':bank5,'title':'Physical bank 5'}
    assert committed['data'][0]['y'] != figure_json(first)['data'][0]['y']


@pytest.mark.parametrize('histogram', [None,'missing'])
def test_native_fallback_and_unknown_histogram_do_not_invent_a_caption(histogram):
    result=multibank_view()
    figure,_=fit_figure(result,result,histogram=histogram)
    metadata=figure_json(figure)['layout'].get('meta',{})
    if histogram is None:
        assert metadata['refinement_histogram']=={
            'name':'PWDR b2.gsa Bank 1','title':'PWDR b2.gsa Bank 1'}
    else:
        assert 'refinement_histogram' not in metadata


def test_failed_plot_replaces_prior_caption_with_empty_figure():
    result=multibank_view()
    first,_=fit_figure(result,result,histogram_label='Physical bank 2')
    state=SimpleNamespace(rf_has_output=True,rf_selected_job='job',rf_experiment='exp',
        rf_stage='final',rf_fit=figure_json(first))
    workspace=SimpleNamespace(state={'jobs':{'job':{'source':{'experiment':'exp'},
        'result':{'baseline_view':'before','final_view':'after'}}}})
    def unavailable(*args):raise ValueError('source unavailable')
    app=SimpleNamespace(server=SimpleNamespace(state=state),_refinement_workspace=workspace,
                        _read_refinement_view=unavailable)
    RefinementMixin._refinement_plot_changed(app)
    assert 'refinement_histogram' not in state.rf_fit['layout'].get('meta',{})
    assert state.rf_fit['data']==[]
