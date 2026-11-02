"""An expanded fit identifies its selected, verified histogram."""
from html.parser import HTMLParser

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.refinement_controller import refinement_histogram_options


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
    assert expanded.caption == ('Histogram: {{ (rf_histograms.find(item => item.value === rf_histogram) || {}).title || rf_histogram }}')
    assert expanded.caption_attrs['v-if'] == '!!rf_histogram'
    # A missing physical-bank mapping still uses the native identity verbatim.
    assert refinement_histogram_options(view,{})[1] == {'title':'PWDR b5.gsa Bank 1','value':'PWDR b5.gsa Bank 1'}
    # The caption is text-only; the expanded plot retains the existing data/layout bindings.
    assert ':data="rf_fit.data"' in app.layout.html
    assert ':layout="rf_fit.layout"' in app.layout.html
