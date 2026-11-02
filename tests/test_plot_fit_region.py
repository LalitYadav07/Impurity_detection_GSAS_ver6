from types import SimpleNamespace
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

from scripts.plot_payload import (histogram_fit_region, histogram_plot_arrays,
                                  native_plot_mask, NATIVE_FIT_CONVENTION)


def test_active_gsas_limits_are_distinct_from_original_measurement_limits():
    histogram = SimpleNamespace(data={"Limits": [(1, 100), [10, 80], [20, 25], [50, 55]]})
    assert histogram_fit_region(histogram) == ([10.0, 80.0], [[20.0, 25.0], [50.0, 55.0]])


def test_missing_or_nonfinite_limits_do_not_create_an_inferred_fit_range():
    for limits in [[], [(1, 100)], [(1, 100), [float("nan"), 80]], [(1, 100), [80, 10]]]:
        assert histogram_fit_region(SimpleNamespace(data={"Limits": limits}))[0] is None


def synthetic_histogram(instrument):
    step, origin = (20., 1000.) if instrument == "TOF" else (.05, 10.)
    x = origin + step * np.arange(20.)
    y = 20. + np.arange(20.)
    y[4:6] = [-3., 0.]  # Signed and zero observations remain valid.
    raw = np.ma.array([x, y, np.full(20, 4.), y - .25, np.zeros(20), np.full(20, .25)])
    raw.mask = np.zeros(raw.shape, dtype=bool)
    raw.mask[0, 10] = True
    raw.mask[1, 11] = True
    raw[2, 12] = 0.
    raw.mask[2, 13] = True
    limits = [(x[0], x[-1]), [x[2] + .2 * step, x[15] + .2 * step], [x[6], x[7]]]
    histogram = SimpleNamespace(data={"Limits": limits, "data": [{}, raw]}, name="PWDR synthetic")
    expected = np.zeros(20, dtype=bool)
    expected[3:17] = True
    expected[[6, 7, 10, 11, 12, 13]] = False
    return histogram, expected


@pytest.mark.parametrize("instrument", ["TOF", "CW"])
def test_plot_arrays_keep_native_grid_endpoint_weights_and_masks(instrument):
    histogram, expected = synthetic_histogram(instrument)
    before = deepcopy(histogram.data)
    arrays = histogram_plot_arrays(histogram)
    np.testing.assert_array_equal(arrays["x"], np.ma.getdata(before["data"][1][0]))
    np.testing.assert_array_equal(arrays["weights"], np.ma.getdata(before["data"][1][2]))
    np.testing.assert_array_equal(arrays["fit_valid"], expected)
    assert arrays["x"][16] > histogram.data["Limits"][1][1]
    assert arrays["yobs"][4] < 0 and arrays["yobs"][5] == 0
    np.testing.assert_array_equal(histogram.data["data"][1].data, before["data"][1].data)
    np.testing.assert_array_equal(histogram.data["data"][1].mask, before["data"][1].mask)


@pytest.mark.parametrize("change", ["reordered", "nonfinite_x", "nonfinite_fit", "bad_mask", "missing_bin",
                                    "missing_residual", "nonfinite_residual", "forged_residual"])
def test_native_plot_contract_rejects_corrupt_grid_or_masks(change):
    histogram, _ = synthetic_histogram("TOF")
    arrays = histogram_plot_arrays(histogram)
    if change == "reordered": arrays["x"] = arrays["x"][::-1]
    elif change == "nonfinite_x": arrays["x"][0] = np.nan
    elif change == "nonfinite_fit": arrays["ycalc"][4] = np.inf
    elif change == "bad_mask": arrays["fit_valid"] = arrays["fit_valid"].astype(int)
    elif change == "missing_residual": arrays.pop("resid")
    elif change == "nonfinite_residual": arrays["resid"][4] = np.nan
    elif change == "forged_residual": arrays["resid"][4] = 1e6
    else: arrays["fit_valid"][16] = False
    with pytest.raises((KeyError, ValueError)):
        native_plot_mask(arrays, *histogram_fit_region(histogram))


def native_payload():
    histogram, _ = synthetic_histogram("TOF")
    limits, exclusions = histogram_fit_region(histogram)
    return {"schema_version": "1.1", "fit_range_convention": NATIVE_FIT_CONVENTION,
            "fit_limits": limits, "excluded_ranges": exclusions,
            "arrays": histogram_plot_arrays(histogram)}


@pytest.mark.parametrize("field", ["fit_range_convention", "fit_limits", "excluded_ranges",
                                   "valid", "fit_valid", "weights", "resid"])
def test_script_viewer_requires_native_schema_fields_even_without_convention(monkeypatch, field):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.plotly_interactive import _fig_gsas_fit
    payload = native_payload()
    (payload if field in payload else payload["arrays"]).pop(field)
    assert _fig_gsas_fit(payload) is None
    payload.pop("fit_range_convention", None)
    payload["arrays"].pop("valid", None)
    payload["arrays"].pop("fit_valid", None)
    assert _fig_gsas_fit(payload) is None


def test_script_viewer_binds_residual_and_allows_only_invalid_bin_nan(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.plotly_interactive import _fig_gsas_fit
    payload = native_payload()
    payload["arrays"]["resid"][4] = 1e6
    assert _fig_gsas_fit(payload) is None
    payload["arrays"]["resid"][4] = .25 + 1e-13
    payload["arrays"]["resid"][[0, 6, 10]] = np.nan
    payload["arrays"]["yobs"][10] = payload["arrays"]["ycalc"][10] = np.nan
    figure = _fig_gsas_fit(payload)
    assert figure is not None
    assert np.isfinite(figure.data[2].y).sum() == 8


def test_script_viewer_normalizes_validated_numeric_interval_strings(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from scripts.plotly_interactive import _fig_gsas_fit
    payload = native_payload()
    payload["fit_limits"] = list(map(str, payload["fit_limits"]))
    payload["excluded_ranges"] = [list(map(str, pair)) for pair in payload["excluded_ranges"]]
    figure = _fig_gsas_fit(payload)
    assert figure is not None
    assert np.isfinite(figure.data[2].y).sum() == 8
    assert figure.layout.xaxis.range[1] == payload["arrays"]["x"][16]


def test_single_bin_exclusion_and_masked_upper_endpoint_do_not_shift_native_slice():
    histogram, expected = synthetic_histogram("TOF")
    x = histogram.data["data"][1][0].data
    histogram.data["Limits"].append([x[5], x[5]])
    histogram.data["data"][1].mask[0, 16] = True
    expected[[5, 16]] = False
    np.testing.assert_array_equal(histogram_plot_arrays(histogram)["fit_valid"], expected)
    assert not histogram_plot_arrays(histogram)["fit_valid"][17]


@pytest.mark.parametrize("instrument,downsample", [("TOF", 1), ("CW", 4)])
def test_static_plot_publishes_full_native_mask_and_keeps_display_gaps(tmp_path, monkeypatch, instrument, downsample):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    import gsas_main_phase_refiner as refiner
    import gsas_core_infrastructure
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from scripts.plot_payload import load_plot_payload
    histogram, expected = synthetic_histogram(instrument)
    raw = histogram.data["data"][1]
    histogram.getdata = lambda key: raw[{"x": 0, "Q": 0, "d": 0, "yobs": 1, "ywt": 2, "ycalc": 3, "background": 4}[key]]
    histogram.get_wR = lambda: 1.25
    histogram.get = lambda *_: 0.
    histogram.reflections = lambda: {}
    project = SimpleNamespace(histograms=lambda: [histogram], phases=lambda: [])
    monkeypatch.setitem(sys.modules, "GSASII", SimpleNamespace(GSASIIscriptable=SimpleNamespace(G2Project=lambda **_: project)))
    monkeypatch.setattr(gsas_core_infrastructure.CoordinateHandler, "from_gsas_histogram",
                        lambda *_: SimpleNamespace(instrument_type=instrument))
    captured = []
    def save(figure, path, **_):
        captured.append((figure.axes[0].get_title(), figure.axes[0].lines[0].get_ydata()))
        Path(path).write_bytes(b"synthetic static preview")
    monkeypatch.setattr(Figure, "savefig", save)
    target = tmp_path / "fit.png"
    refiner.plot_gpx_fit_with_ticks("synthetic.gpx", str(target), downsample=downsample)
    payload = load_plot_payload(str(target) + ".plotdata.json")
    assert payload["fit_range_convention"] == NATIVE_FIT_CONVENTION
    assert payload["fit_points"] == int(expected.sum())
    assert f"N (fit) = {expected.sum():,}" in captured[0][0]
    assert np.isnan(captured[0][1]).any()  # No line across excluded/invalid bins, including after decimation.
    assert len(payload["arrays"]["x"]) == 20
    np.testing.assert_array_equal(payload["arrays"]["fit_valid"], expected)
    from scripts.plotly_interactive import _fig_gsas_fit
    figure = _fig_gsas_fit(payload)
    assert figure.layout.xaxis.range[1] >= payload["arrays"]["x"][16]
    assert int(np.isfinite(figure.data[1].y).sum()) == int(expected.sum())
    plt.close("all")
