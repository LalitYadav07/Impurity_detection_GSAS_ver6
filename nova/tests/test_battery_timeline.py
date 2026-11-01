from radar_pd_nova.battery.electrochemistry import CyclerBinding, read_cycler, summarize_exposure
from radar_pd_nova.battery.timeline import timeline_figure


def test_timeline_preserves_gaps_and_distinguishes_exposure_mean():
    binding = CyclerBinding('sample', 'cell', 'C1', 'charge', '2026-01-01T00:00:00Z', maximum_gap_seconds=15)
    series = read_cycler(b'elapsed_seconds,voltage_v,current_ma\n0,3,1\n10,4,1\n100,3,1\n110,4,1\n', binding)
    joined = dict(run=7, **summarize_exposure(series, '2026-01-01T00:00:00Z', '2026-01-01T00:00:10Z', sample_id='sample'))
    figure = timeline_figure(series, [joined])
    assert figure['data'][0]['y'] == [3., 4., None, 3., 4.]
    assert figure['data'][1]['y'] == [1., 1., None, 1., 1.]
    assert figure['data'][2]['y'] == [3.5]
    assert figure['data'][2]['error_x']['array'] == [5000.]


def test_decimation_keeps_spike_and_does_not_bridge_hidden_gap():
    raw = 'elapsed_seconds,voltage_v\n' + ''.join(f'{i*10 if i<50 else i*10+1000},{20 if i==37 else 3}\n' for i in range(100))
    series = read_cycler(raw.encode(), CyclerBinding('s','c','1','charge','2026-01-01T00:00:00Z',maximum_gap_seconds=15))
    figure = timeline_figure(series, [], max_points=12)
    assert 20 in figure['data'][0]['y'] and None in figure['data'][0]['y']
    assert len(figure['data'][0]['y']) <= 24
    assert len(figure['data']) == 1
    assert 'No measured current' in figure['layout']['annotations'][0]['text']
