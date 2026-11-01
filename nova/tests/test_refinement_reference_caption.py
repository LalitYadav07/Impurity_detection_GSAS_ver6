"""Reference identity must survive static Plotly image export."""
from copy import deepcopy

import pytest

from radar_pd_nova.refinement_plots import trend_figure
from test_refinement_trend_arrivals import row


def caption(figure):
    return next(item for item in figure.layout.annotations if item.name=='reference-caption')


def test_static_relative_caption_identifies_reference_without_changing_values():
    records=[row(64301,200,4.04),row(64299,75,4.)]
    before=deepcopy(records)
    figure=trend_figure(records,metric='cell|oxide|c',x_key='temperature',display='relative')
    text=caption(figure).text
    assert 'scan 64299' in text and '75 K' in text and 'value 4 Å' in text
    assert list(figure.data[0].y)==pytest.approx([0,1])
    assert list(figure.data[0].x)==[75,200]
    assert records==before
    assert figure.layout.margin.b>=85


def test_caption_uses_the_same_converted_units_as_the_plotted_axis():
    records=[row(1,114,4.),row(2,153,4.04)]
    for record,value in zip(records,[114,153]):
        record['source']['axis_overrides']['temperature'].update(value=value,unit='C')
    figure=trend_figure(records,metric='cell|oxide|c',x_key='temperature',display='relative')
    assert list(figure.data[0].x)==pytest.approx([387.15,426.15])
    assert '387.15 K' in caption(figure).text and '114 C' not in caption(figure).text
    assert figure.layout.meta['references'][0]['coordinate_value']==pytest.approx(387.15)
    assert list(figure.data[0].y)==pytest.approx([0,1])


def test_two_groups_get_distinct_static_references_and_keep_difference_values():
    records=[row(1,75,4.),row(2,200,4.04),row(3,90,8.,sample='other',bank='3',recipe='other')]
    figure=trend_figure(records,metric='cell|oxide|c',x_key='temperature',display='difference')
    text=caption(figure).text
    assert len(text.split('<br>'))==2
    assert 'sample · bank 2' in text and 'scan 1 · 75 K' in text
    assert 'other · bank 3' in text and 'scan 3 · 90 K' in text
    assert list(figure.data[0].y)==pytest.approx([0,.04])
    assert list(figure.data[1].y)==[0]


def test_many_groups_have_bounded_caption_and_explicit_complete_reference_access():
    records=[row(i,75+i,4.+i,sample='sample '+str(i)) for i in range(1,101)]
    figure=trend_figure(records,metric='cell|oxide|c',x_key='temperature',display='relative')
    text=caption(figure).text
    assert len(text.split('<br>'))==3 and '98 more reference groups' in text
    assert 'hover each series for its full reference' in text
    assert len(figure.layout.meta['references'])==100
    assert all('scan '+str(i) in trace.hovertext[0] for i,trace in enumerate(figure.data,1))
    assert len(text)<400


def test_caption_escapes_and_bounds_labels_without_truncating_hover_metadata():
    label='<b>'+('long sample '*30)+'</b>'
    figure=trend_figure([row(1,75,4.,sample=label),row(2,90,8.,sample='other')],
        metric='cell|oxide|c',x_key='temperature',display='relative')
    text=caption(figure).text
    assert '<b>' not in text and '&lt;b&gt;' in text and '…' in text
    assert label in figure.layout.meta['references'][0]['description']
    assert '&lt;b&gt;' in figure.data[0].hovertext[0]


def test_absolute_figure_has_no_reference_caption_or_extra_margin():
    figure=trend_figure([row(1,75,4.),row(2,200,4.04)],metric='cell|oxide|c',x_key='temperature')
    assert figure.layout.meta['references']==[] and not figure.layout.annotations
    assert figure.layout.margin.b==60 and list(figure.data[0].y)==[4.,4.04]


def test_voltage_caption_is_separate_from_existing_exposure_note():
    records=[row(1,75,4.)]
    records[0]['source']['metadata']=dict(sample_id='sample',cell_id='cell',electrochemistry={
        'schema':'radar-battery-model-electrochemistry/v1',
        'exposure':dict(sample_id='sample',cell_id='cell',voltage_v={'mean':3.2})})
    figure=trend_figure(records,metric='cell|oxide|c',x_key='battery_voltage',display='relative')
    assert '3.2 V' in caption(figure).text and caption(figure).yshift==-92
    assert any('Voltage means' in item.text for item in figure.layout.annotations)
    assert figure.layout.margin.b>=119
