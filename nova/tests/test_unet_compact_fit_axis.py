"""Compact fit coordinates stay visible on a populated subplot."""
from copy import deepcopy

import numpy as np
import pytest

from radar_pd_nova.scan_interpretation import Pattern
from radar_pd_nova.unet_transition_ui import _fit_figure_for_scan
from test_scan_interpretation import packet


def native_packet():
    value=packet(masked=[(2.0,2.1)])
    value['coordinate']={'kind':'tof','unit':'us'}
    value['x']=(np.asarray(value['x'])*10000.).tolist()
    value['fit_domain']['limits']=[10000.,90000.]
    value['fit_valid'][-15:]=[False]*15
    value['ycalc']=(np.asarray(value['y'])-1.5).tolist()
    return value


def test_no_ticks_puts_native_axis_on_the_populated_residual_row():
    value=native_packet();original=deepcopy(value)
    figure,note=_fit_figure_for_scan({'pattern':value},Pattern.parse(value))
    difference=next(trace for trace in figure['data'] if trace['name']=='Difference')
    axis='xaxis'+difference['xaxis'].removeprefix('x')
    assert figure['layout'][axis].get('showticklabels') is True
    assert figure['layout'][axis]['title']['text']=='Time of flight (µs)'
    assert figure['layout']['xaxis'].get('matches')==difference['xaxis']
    assert 'xaxis3' not in figure['layout'] and 'yaxis3' not in figure['layout']
    assert figure['layout']['yaxis2']['domain'][0]==0
    assert value==original
    assert 'not included' in note


@pytest.mark.parametrize('ticks',[None,{'phase':[20000.,40000.]},{'phase':[]}])
def test_layout_keeps_native_arrays_masks_and_tick_locations_exact(ticks):
    value=native_packet();original=deepcopy(value)
    if ticks is not None:
        value['phase_ticks']=ticks;value['phase_order']=['phase']
        original=deepcopy(value)
    figure,_=_fit_figure_for_scan({'pattern':value},Pattern.parse(value))
    curves={trace['name']:trace for trace in figure['data']}
    fit=np.asarray(value['fit_valid'],dtype=bool)&np.asarray(value['valid'],dtype=bool)
    obs=np.asarray([np.nan if n is None else n for n in curves['Observed']['y']])
    difference=np.asarray([np.nan if n is None else n for n in curves['Difference']['y']])
    assert curves['Observed']['x']==value['x']
    np.testing.assert_array_equal(np.isfinite(obs),fit)
    np.testing.assert_array_equal(np.isfinite(difference),fit)
    np.testing.assert_array_equal(obs[fit],np.asarray(value['y'])[fit])
    np.testing.assert_allclose(difference[fit],1.5,rtol=0,atol=1e-12)
    assert value==original
    if ticks and ticks['phase']:
        assert curves['phase']['x']==ticks['phase']
        assert curves['phase']['xaxis']=='x3'
        assert figure['layout']['xaxis3']['title']['text']=='Time of flight (µs)'
    else:
        assert not any(trace.get('xaxis')=='x3' for trace in figure['data'])
