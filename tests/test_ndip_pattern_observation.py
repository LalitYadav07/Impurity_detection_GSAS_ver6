"""Trusted-GPX curve publication preserves masks and independent fit domains."""

import hashlib, json
from pathlib import Path

import numpy as np
import pytest

from scripts.ndip_pattern_observation import pattern_from_records, publish_pattern


def records():
    x = np.linspace(1000, 3000, 40)
    y = np.arange(40.0) + 20
    w = np.full(40, 4.0)
    c = y - 0.5
    data = np.ma.array([x, y, w, c, np.zeros(40), y - c])
    data.mask = np.zeros(data.shape, dtype=bool)
    data.mask[:, 7] = True
    parameters = {"Type": ["PNT", "PNT", False], "difC": [10000.0, 10001.0, False]}
    return [
        [
            ["PWDR synthetic", [{"wtFactor": 1.0}, data, {}]],
            ["Limits", [(1000, 3000), [1200, 2800], [1600, 1800]]],
            ["Instrument Parameters", [parameters, {}]],
        ]
    ]


def test_publication_has_distinct_measured_valid_and_fitted_valid_masks(tmp_path):
    source = tmp_path / "own.gpx"
    source.write_bytes(b"trusted project placeholder")
    original = records()
    packet = pattern_from_records(
        source,
        original,
        input_provenance={"instrument": "PG3", "source_bank": 2, "reduction_family": "GSAS"},
    )
    assert not packet["valid"][7] and packet["sigma"][7] is None
    assert packet["valid"][0] and not packet["fit_valid"][0]
    assert packet["ycalc"][0] is None
    for x, fit in zip(packet["x"], packet["fit_valid"]):
        if 1600 <= x <= 1800:
            assert not fit
    valid = np.asarray(packet["valid"])
    assert np.all(np.array(packet["sigma"])[valid] == 0.5)
    assert packet["context"]["source_bank"] == 2
    assert packet["coordinate"] == {"kind": "tof", "unit": "us"}
    assert packet["source_gpx_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "tables/Pattern_observation.json"
    receipt = publish_pattern(source, original, destination)
    assert receipt["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert receipt["points"] == 40
    json.loads(destination.read_text())


def test_ambiguous_histograms_are_never_silently_combined(tmp_path):
    path = tmp_path / "own.gpx"
    path.write_bytes(b"own")
    with pytest.raises(ValueError, match="one explicit histogram"):
        pattern_from_records(path, records() * 2)


def test_zero_weights_and_nonfinite_measurements_remain_missing(tmp_path):
    path = tmp_path / "own.gpx"
    path.write_bytes(b"own")
    rows = records()
    data = rows[0][0][1][1]
    data[1, 8] = np.nan
    data[2, 9] = 0
    result = pattern_from_records(path, rows)
    assert not result["valid"][8] and not result["valid"][9]
    assert result["sigma"][8] is None and result["sigma"][9] is None
    json.dumps(result, allow_nan=False)


def test_collector_registers_the_portable_curve_in_summary_and_galaxy_tables(tmp_path, monkeypatch):
    from scripts import ndip_outputs

    root = tmp_path / "run"
    folder = root / "Technical/GSAS_Projects"
    folder.mkdir(parents=True)
    project = folder / "seq_final_joint_polished.gpx"
    project.write_bytes(b"trusted generated project")
    monkeypatch.setattr(ndip_outputs, "_read_gpx_records", lambda path: records())
    portal = tmp_path / "portal"
    result = ndip_outputs.collect_outputs(root, portal, mode="full", include_archive=False)
    receipt = result["pattern_observation"]
    assert receipt["name"] == "Pattern_observation.json"
    assert (portal / "tables/Pattern_observation.json").is_file()
    assert any(
        row["name"] == receipt["name"] and row["sha256"] == receipt["sha256"]
        for row in result["artifacts"]["tables"]
    )
    assert json.loads((portal / "summary.json").read_text())["pattern_observation"] == receipt
