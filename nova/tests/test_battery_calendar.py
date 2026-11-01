import pytest
from radar_pd_nova.battery.electrochemistry import CyclerBinding, read_cycler, local_calendar_time


def test_ec_lab_calendar_dates_in_time_seconds_column_require_reviewed_convention():
    raw=b'time/s\tEwe/V\t\n10/31/2024 16:32:56.0000\t3.5\t\n10/31/2024 16:33:56.0000\t3.6\t\n'
    binding=CyclerBinding('s','cell','C01','charge',local_time_format='month/day/year',timezone_name='America/New_York')
    series=read_cycler(raw,binding)
    assert series.summary()['first_utc']=='2024-10-31T20:32:56+00:00'
    assert series.points[1].timestamp-series.points[0].timestamp==60
    assert series.summary()['current_status']=='unavailable'
    with pytest.raises(ValueError,match='calendar'):
        read_cycler(raw,CyclerBinding('s','cell','C01','charge'))


@pytest.mark.parametrize('local',['11/03/2024 01:30:00','03/10/2024 02:30:00'])
def test_dst_ambiguous_and_nonexistent_wall_times_are_rejected(local):
    with pytest.raises(ValueError,match='ambiguous or nonexistent'):
        local_calendar_time(local,'month/day/year','America/New_York')


def test_date_order_is_not_guessed_from_ambiguous_numbers():
    a=local_calendar_time('01/02/2024 12:00:00','month/day/year','UTC')
    b=local_calendar_time('01/02/2024 12:00:00','day/month/year','UTC')
    assert b-a==30*86400


def test_calendar_timezone_and_elapsed_origin_cannot_be_combined():
    with pytest.raises(ValueError,match='clear the elapsed'):
        CyclerBinding('s','c','1','charge','2024-01-01T00:00:00Z',local_time_format='month/day/year',timezone_name='UTC').validate()


def test_nonempty_unnamed_column_is_not_silently_dropped():
    with pytest.raises(ValueError,match='unnamed'):
        read_cycler(b'elapsed_seconds,voltage_v,\n0,3,unexpected\n1,4,unexpected\n',
                    CyclerBinding('s','c','1','charge','2024-01-01T00:00:00Z'))
