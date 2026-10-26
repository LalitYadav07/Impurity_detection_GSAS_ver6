import json

import h5py
import numpy as np
import pytest

from radar_pd_nova.scan_conditions import Source, read_conditions, summarize_log, qualified_metadata

WINDOW = {"start": "2026-01-01T00:00:00Z", "end": "2026-01-01T00:01:40Z"}


def log(values, unit):
    return {"values": values, "attrs": {"units": unit}}


def times(values, **attrs):
    return {"values": values, "attrs": {"units": "s", "start": WINDOW["start"], **attrs}}


def test_exposure_weighted_readback_not_average_of_irregular_updates():
    result = summarize_log(
        Source("magnetic_field", "actual"), log([7, 0], "T"), times([0, 5]), WINDOW
    )
    assert result["status"] == "recorded"
    assert result["value"] == pytest.approx(0.35)
    assert (result["minimum"], result["maximum"]) == (0, 7)
    assert result["statistic"] == "time_weighted_step_hold"


def test_request_is_never_a_measured_hold_and_no_extrapolation_before_first_sample():
    result = summarize_log(
        Source("magnetic_field", "request", "requested"), log([7, 0], "T"), times([0, 5]), WINDOW
    )
    assert result["status"] == "requested"
    result = summarize_log(
        Source("temperature", "actual"), log([300, 310], "K"), times([20, 50]), WINDOW
    )
    assert result["status"] == "incomplete_coverage" and result["coverage_fraction"] == 0.8


@pytest.mark.parametrize(
    "quantity,unit", [("voltage", "K"), ("voltage", "A"), ("gas_flow", ""), ("temperature", "")]
)
def test_semantic_units_not_guessed(quantity, unit):
    result = summarize_log(Source(quantity, "channel"), log([1], unit), times([0]), WINDOW)
    assert result["status"] == "unqualified_units" and "value" not in result


def test_pressure_needs_verified_specimen_device_binding():
    source = Source("pressure", "gauge", needs_binding=True)
    result = summarize_log(source, log([1], "bar"), times([0]), WINDOW)
    assert result["status"] == "needs_mapping"
    result = summarize_log(
        source, log([1], "bar"), times([0]), WINDOW, binding={"verified": True, "channel": "gauge"}
    )
    assert result["value"] == 100000 and result["status"] == "recorded"


@pytest.mark.parametrize(
    "time",
    [
        times([0, 0]),
        times([0, -1]),
        times([0]),
        times([0, 1], scaling_factor=-1),
        times([0, 1], units="ms"),
    ],
)
def test_malformed_or_ambiguous_time_is_not_recorded(time):
    result = summarize_log(Source("temperature", "channel"), log([300, 310], "K"), time, WINDOW)
    assert result["status"] == "needs_time_alignment" and "value" not in result


def write_nexus(path):
    with h5py.File(path, "w") as h:
        h.create_dataset("entry/start_time", data=[WINDOW["start"].encode()])
        h.create_dataset("entry/end_time", data=[WINDOW["end"].encode()])
        h.create_dataset("entry/title", data=[b"field sweep"])
        for channel, values, unit in [
            ("BL11A:SE:MAG08:GetMGField", [7.0, 0.0], "T"),
            ("Mag08ActualField", [7.0, 0.0], "T"),
            ("BL11A:SE:MAG08:GetFieldSet", [7.0, 0.0], "T"),
            ("BL11A:SE:SampleVolt", [8.0, 8.0], "K"),
        ]:
            prefix = "entry/DASlogs/" + channel + "/"
            d = h.create_dataset(prefix + "value", data=values)
            d.attrs["units"] = unit
            t = h.create_dataset(prefix + "time", data=[0.0, 5.0])
            t.attrs["start"] = WINDOW["start"]
            t.attrs["units"] = "s"
        # Reading detector banks is forbidden by construction of the registry.
        h.create_dataset("entry/bank1_events/event_id", shape=(10000000,), dtype="i8")


def test_h5py_named_reads_and_returned_receipts_are_independent(tmp_path):
    path = tmp_path / "raw.h5"
    write_nexus(path)
    result = read_conditions(path, "PG3", backend="h5py", bindings={
        "magnetic_field": {"verified": True, "channel": "BL11A:SE:MAG08:GetMGField"}})
    assert result["status"] == "read"
    channels = {c["channel"]: c for c in result["channels"]}
    assert channels["BL11A:SE:MAG08:GetMGField"]["value"] == pytest.approx(0.35)
    assert channels["Mag08ActualField"]["status"] == "needs_mapping"
    assert channels["BL11A:SE:MAG08:GetMGField"]["binding_verified"]
    assert channels["BL11A:SE:MAG08:GetFieldSet"]["status"] == "requested"
    assert channels["BL11A:SE:SampleVolt"]["status"] == "unqualified_units"
    result["channels"].clear()
    assert read_conditions(path, "PG3", backend="h5py")["channels"]
    json.dumps(result, allow_nan=False)


def test_conflicting_devices_never_choose_nonzero_or_larger_value(tmp_path):
    path = tmp_path / "raw.h5"
    write_nexus(path)
    with h5py.File(path, "a") as h:
        p = "entry/DASlogs/BL11A:SE:SS:ActualField/"
        v = h.create_dataset(p + "value", data=[0.0, 0.0])
        v.attrs["units"] = "T"
        t = h.create_dataset(p + "time", data=[0.0, 5.0])
        t.attrs["start"] = WINDOW["start"]
        t.attrs["units"] = "s"
    result = read_conditions(path, "PG3", backend="h5py")
    assert all(
        c["status"] == "needs_mapping"
        for c in result["channels"]
        if c["quantity"] == "magnetic_field" and c["role"] == "measured"
    )
    result = read_conditions(
        path,
        "PG3",
        backend="h5py",
        bindings={"magnetic_field": {"verified": True, "channel": "BL11A:SE:SS:ActualField"}},
    )
    selected = [
        c
        for c in result["channels"]
        if c["quantity"] == "magnetic_field" and c["status"] == "recorded"
    ]
    assert len(selected) == 1 and selected[0]["value"] == 0


def test_legacy_zero_field_cannot_bypass_ambiguous_facility_devices(tmp_path):
    from radar_pd_nova.powgen_metadata import read_powgen_scan_metadata
    from radar_pd_nova.results import experiment_axis_options

    path = tmp_path / "raw.h5"
    write_nexus(path)
    with h5py.File(path, "a") as handle:
        prefix = "entry/DASlogs/BL11A:SE:SS:Field/"
        value = handle.create_dataset(prefix + "value", data=[0.0])
        value.attrs["units"] = "T"
        time = handle.create_dataset(prefix + "time", data=[0.0])
        time.attrs["units"] = "s"
        time.attrs["start"] = WINDOW["start"]
    metadata = read_powgen_scan_metadata("IPTS-123", 1, nexus_path=path)
    assert "magnetic_field" not in metadata
    assert metadata["magnetic_field_observation"]["value"] == 0.0
    assert "magnetic_field" not in {row["value"] for row in experiment_axis_options([{"metadata": metadata}])}
    assert any(c["status"] == "needs_mapping" for c in metadata["conditions"]["channels"])


def test_saved_metadata_conflicts_are_qualified_without_mutating_original():
    from radar_pd_nova.results import experiment_axis_options

    old = {"magnetic_field": {"value": 0.0, "unit": "T"}, "conditions": {"channels": [
        {"quantity": "magnetic_field", "role": "measured", "status": "needs_mapping", "value": 0.0},
    ]}}
    result = qualified_metadata(old)
    assert "magnetic_field" not in result and old["magnetic_field"]["value"] == 0.0
    assert "magnetic_field" not in {row["value"] for row in experiment_axis_options([{"metadata": old}])}


def test_single_logged_magnet_still_requires_specimen_association(tmp_path):
    path = tmp_path / "raw.h5"
    write_nexus(path)
    result = read_conditions(path, "PG3", backend="h5py")
    measured = [c for c in result["channels"] if c["quantity"] == "magnetic_field" and c["role"] == "measured"]
    assert measured and all(c["status"] == "needs_mapping" for c in measured)
    assert all(c["requires_binding"] and not c["binding_verified"] for c in measured)


def test_older_auto_selected_magnet_receipt_is_upgraded_for_review():
    from radar_pd_nova.scan_interpretation import _conditions

    channel = {"quantity": "magnetic_field", "channel": "BL11A:SE:MAG08:GetMGField",
               "role": "measured", "status": "recorded", "value": 0.0, "unit": "T",
               "statistic": "time_weighted_step_hold", "coverage_fraction": 1.0,
               "minimum": 0.0, "maximum": 0.0}
    old = {"magnetic_field": {"value": 0.0, "unit": "T"},
           "conditions": {"registry": "SNS-PG3-NOM-conditions/1", "metadata_sha256": "original",
                          "channels": [channel]}}
    result = qualified_metadata(old)
    assert "magnetic_field" not in result
    assert result["conditions"]["source_metadata_sha256"] == "original"
    assert result["conditions"]["channels"][0]["status"] == "needs_mapping"
    assert old["conditions"]["channels"][0]["status"] == "recorded"
    assert not _conditions({"metadata": old})


def test_window_slice_and_trajectory_retains_extrema(tmp_path):
    path = tmp_path / "raw.h5"
    write_nexus(path)
    result = read_conditions(
        path, "PG3", backend="h5py", window={**WINDOW, "start": "2026-01-01T00:00:10Z"}
    )
    actual = next(c for c in result["channels"] if c["channel"] == "BL11A:SE:MAG08:GetMGField")
    assert actual["value"] == 0 and result["window_scope"] == "explicit_product_window"
    values = np.zeros(10000)
    values[1234] = 20
    result = summarize_log(
        Source("magnetic_field", "actual"),
        log(values.tolist(), "T"),
        times((np.arange(10000) / 100).tolist()),
        WINDOW,
    )
    assert len(result["trajectory"]["seconds"]) <= 512
    assert max(result["trajectory"]["values"]) == 20
def test_condition_display_distinguishes_scan_average_and_observed_range():
    from radar_pd_nova.scan_conditions import condition_display
    metric = {"value":96.83,"unit":"K","minimum":1.452,"maximum":280.026,
              "statistic":"time_weighted_step_hold"}
    text = condition_display(metric,"Temperature")
    assert "time-weighted mean 96.83 K" in text
    assert "recorded scan range 1.452–280.026 K" in text
    assert condition_display({"value":float('nan')},"Temperature") == ""
    assert "mean" not in condition_display({"value":100.,"unit":"K"},"Temperature")
