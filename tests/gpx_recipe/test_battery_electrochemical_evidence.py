import pytest

from gpx_recipe.battery.electrochemical_evidence import capture_electrochemistry, reproduce_electrochemistry
from radar_pd_nova.battery.electrochemistry import CyclerBinding, read_cycler, summarize_exposure


@pytest.mark.parametrize('current',['current_ma','no_current'])
def test_saved_electrochemistry_reproduces_clock_shift_gaps_and_missing_current(tmp_path,current):
    column='current_ma' if current=='current_ma' else 'aux'
    raw=(f'elapsed_seconds,voltage_v,{column},cycle,step\n'
         '0,3,10,1,1\n10,3.2,10,1,1\n100,4,10,1,2\n110,4.1,10,1,2\n').encode()
    binding=CyclerBinding('sample','cell','C01','charge',clock_origin='2026-01-01T00:00:00Z',
                          clock_offset_seconds=5,clock_uncertainty_seconds=2,maximum_gap_seconds=30)
    series=read_cycler(raw,binding)
    acquisition=dict(sample_id='sample',cell_id='cell',start_utc='2026-01-01T00:00:00Z',end_utc='2026-01-01T00:02:00Z')
    expected=summarize_exposure(series,acquisition['start_utc'],acquisition['end_utc'],sample_id='sample')
    record=capture_electrochemistry(series,raw,tmp_path)
    restored=reproduce_electrochemistry(record,tmp_path,acquisition)
    assert restored['exposure']==expected
    assert restored['original_binding']['clock_offset_seconds']==5
    assert expected['coverage_fraction']==pytest.approx(1/6)
    assert expected['current_status']==('partial' if current=='current_ma' else 'unavailable')


def test_electrochemical_evidence_cannot_rebind_or_change_inputs(tmp_path):
    raw=b'time_s,voltage_v,current_a\n0,3,0.1\n10,4,0.1\n'
    series=read_cycler(raw,CyclerBinding('sample','cell','1','charge',clock_origin='2026-01-01T00:00:00Z'))
    record=capture_electrochemistry(series,raw,tmp_path)
    acquisition=dict(sample_id='sample',cell_id='other',start_utc='2026-01-01T00:00:00Z',end_utc='2026-01-01T00:00:10Z')
    with pytest.raises(ValueError,match='different sample or cell'):reproduce_electrochemistry(record,tmp_path,acquisition)
    acquisition['cell_id']='cell'
    (tmp_path/record['normalized']['path']).write_text('altered')
    with pytest.raises(ValueError,match='changed'):reproduce_electrochemistry(record,tmp_path,acquisition)
    with pytest.raises(ValueError,match='source is unavailable or changed'):capture_electrochemistry(series,b'different',tmp_path)
