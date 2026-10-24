"""Republishing must update the actual JSON/NPZ pair the GUI discovers."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.ndip_outputs import collect_outputs, _copy_collections


def test_recollection_replaces_plot_payload_and_arrays_without_stale_aliases(tmp_path):
    results = pytest.importorskip("radar_pd_nova.results")
    run = tmp_path / "run"
    plots = run / "Results/Plots"
    plots.mkdir(parents=True)
    target = plots / "seq_final_joint_accepted_model.png"
    target.write_bytes(b"placeholder")
    metadata = Path(str(target) + ".plotdata.json")
    arrays = Path(str(target) + ".plotdata.npz")

    def publish(value):
        metadata.write_text(
            json.dumps(
                {
                    "plot_kind": "gsas_fit_with_ticks_v1",
                    "rwp": value,
                    "fit_limits": [1, 2],
                    "source_plot": target.name,
                    "source_plot_path": str(target),
                    "arrays_npz": arrays.name,
                }
            )
        )
        np.savez(arrays, x=[1.0, 2.0], yobs=[value, value], ycalc=[value, value], resid=[0.0, 0.0])

    portal = tmp_path / "portal"
    publish(1.0)
    first = collect_outputs(run, portal, mode="full", include_archive=False)
    initial_paths = {p.relative_to(portal) for p in portal.rglob("*") if p.is_file()}
    publish(2.0)
    second = collect_outputs(run, portal, mode="full", include_archive=False)
    assert {p.relative_to(portal) for p in portal.rglob("*") if p.is_file()} == initial_paths
    assert [r["path"] for r in first["artifacts"]["diagnostics"]] == [
        r["path"] for r in second["artifacts"]["diagnostics"]
    ]
    options = [
        p for p in results.discover_plot_payloads(portal) if p["kind"] == "gsas_fit_with_ticks_v1"
    ]
    assert len(options) == 1
    loaded = results.read_plot_payload(options[0]["path"])
    assert loaded["rwp"] == 2.0 and loaded["arrays"]["yobs"] == [2.0, 2.0]


def test_distinct_sources_with_same_friendly_name_have_stable_outputs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    for prefix, value in [("a", 1), ("b", 2)]:
        folder = run / "Results" / prefix
        folder.mkdir(parents=True)
        (folder / "summary_fractions.csv").write_text(f"phase,weight\n{prefix},{value}\n")
    portal = tmp_path / "portal"
    first = _copy_collections(run, portal)["tables"]
    second = _copy_collections(run, portal)["tables"]
    assert len({r["path"] for r in first}) == 2
    assert [r["path"] for r in first] == [r["path"] for r in second]
    assert len(list((portal / "tables").iterdir())) == 2
