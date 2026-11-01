from dataclasses import replace
import pytest

from radar_pd_nova.battery.electrochemistry import CyclerBinding, read_cycler, summarize_exposure


@pytest.fixture
def binding():
    return CyclerBinding("66548", "LNO-cell-3", "C05", "charge", clock_origin="2019-12-09T12:00:26-05:00")


def series(binding, body="0,3,100\n10,4,100\n20,5,100\n"):
    return read_cycler(("elapsed_seconds,voltage_v,current_ma\n"+body).encode(), binding)


def test_exact_linear_exposure_integrals_and_units(binding):
    s=series(binding)
    r=summarize_exposure(s,"2019-12-09T17:00:31Z","2019-12-09T17:00:41Z",sample_id="66548")
    assert r["status"] == "available" and r["coverage_fraction"] == 1
    assert r["voltage_v"] == {"mean":4.,"minimum":3.5,"maximum":4.5}
    assert r["current_a"]["mean"] == pytest.approx(.1)
    assert r["signed_charge_c"] == pytest.approx(1.)
    assert r["branches"] == ["charge"]
    assert r["composition_status"] == "not_evaluated"


def test_partial_exposure_does_not_extrapolate_cycler_before_start(binding):
    s=series(binding)
    r=summarize_exposure(s,"2019-12-09T17:00:05Z","2019-12-09T17:00:46Z",sample_id="66548")
    assert r["status"] == "partial"
    assert r["covered_seconds"] == 20
    assert r["coverage_fraction"] == pytest.approx(20/41)
    assert r["signed_charge_c"] == pytest.approx(2)


def test_gap_and_protocol_boundary_are_excluded(binding):
    raw=b"elapsed_seconds,voltage_v,current_a,step\n0,3,1,1\n10,4,1,1\n100,4,1,1\n110,3,-1,2\n120,2,-1,2\n"
    s=read_cycler(raw,binding)
    r=summarize_exposure(s,"2019-12-09T17:00:26Z","2019-12-09T17:02:26Z",sample_id="66548")
    assert r["covered_seconds"] == 20
    assert r["signed_charge_c"] == pytest.approx(0)
    assert r["branches"] == ["charge","discharge"]
    assert {i["reason"] for i in r["excluded_intervals"]} == {"log_gap","protocol_boundary"}


def test_separate_cells_channels_and_current_sign_are_bound(binding):
    raw=b"elapsed_seconds,voltage_v,current_a,cell_id,channel\n0,3,1,LNO-cell-3,C05\n0,9,9,other,C06\n10,4,1,LNO-cell-3,C05\n10,9,9,other,C06\n"
    s=read_cycler(raw,replace(binding,positive_current="discharge"))
    assert len(s.points)==2 and s.branch(s.points[0])=="discharge"
    with pytest.raises(ValueError,match="does not match"):
        summarize_exposure(s,"2019-12-09T17:00:26Z","2019-12-09T17:00:36Z",sample_id="66546")


def test_absolute_timestamps_and_clock_offset(binding):
    raw=b"timestamp,voltage_v,current_a\n2019-12-09T12:00:26-05:00,3,1\n2019-12-09T17:00:36Z,4,1\n"
    s=read_cycler(raw,replace(binding,clock_offset_seconds=2.,clock_uncertainty_seconds=.5))
    assert s.summary()["first_utc"] == "2019-12-09T17:00:28+00:00"
    r=summarize_exposure(s,"2019-12-09T17:00:28Z","2019-12-09T17:00:38Z",sample_id="66548")
    assert r["clock_uncertainty_seconds"]==.5


@pytest.mark.parametrize("body",["0,3,1\n0,4,1\n","10,3,1\n0,4,1\n","0,3,1\n10,nan,1\n"])
def test_conflicting_times_and_invalid_numbers_fail(binding,body):
    with pytest.raises(ValueError):series(binding,body)


def test_exact_duplicates_have_explicit_accounting(binding):
    s=series(binding,"0,3,100\n0,3,100\n10,4,100\n")
    assert len(s.points)==2 and s.duplicate_rows==1


@pytest.mark.parametrize("change",[{"positive_current":"unknown"},{"sample_id":""},{"maximum_gap_seconds":0},{"clock_uncertainty_seconds":-1},{"clock_origin":"2019-12-09T12:00:26"},{"clock_offset_seconds":"2"}])
def test_invalid_binding_does_not_become_a_series(binding,change):
    with pytest.raises(ValueError):series(replace(binding,**change))


def test_relative_time_requires_qualified_origin(binding):
    with pytest.raises(ValueError,match="clock origin"):
        series(replace(binding,clock_origin=""))


def test_biologic_text_header_and_milliamp_conversion(binding):
    raw=b"EC-Lab ASCII FILE\nNb header lines : 5\nAcquisition started on : 12/09/2019 12:00:26\n\ntime/s\tEwe/V\t<I>/mA\tcycle number\n0\t3\t100\t1\n10\t4\t100\t1\n"
    s=read_cycler(raw,binding)
    assert s.format=="biologic_text" and s.points[0].current_a==pytest.approx(.1)
    assert s.recorded_origin=="12/09/2019 12:00:26"
    assert s.points[0].cycle=="1"


def test_unknown_units_and_ambiguous_columns_are_not_guessed(binding):
    for raw in [b"time,voltage,current\n0,3,1\n1,4,1\n",b"time/s,voltage_v,current_a,current_ma\n0,3,1,1\n1,4,1,1\n"]:
        with pytest.raises(ValueError):read_cycler(raw,binding)


def test_no_overlap_has_no_zero_valued_measurement(binding):
    r=summarize_exposure(series(binding),"2020-01-01T00:00:00Z","2020-01-01T00:05:00Z",sample_id="66548")
    assert r["status"]=="unavailable" and r["voltage_v"] is None and r["signed_charge_c"] is None


def test_voltage_only_legacy_record_preserves_useful_data_without_inventing_current(binding):
    raw=b"time/s,Ewe/V,control/V/mA,P/W\n0,3,100,0.3\n10,4,100,0.4\n"
    s=read_cycler(raw,binding)
    assert s.summary()["current_status"]=="unavailable"
    r=summarize_exposure(s,"2019-12-09T17:00:26Z","2019-12-09T17:00:36Z",sample_id="66548")
    assert r["voltage_v"]["mean"]==3.5 and r["voltage_status"]=="available"
    assert r["status"]=="partial" and r["current_a"] is None and r["signed_charge_c"] is None
    assert r["branches"]==["unresolved"] and r["current_coverage_fraction"]==0
