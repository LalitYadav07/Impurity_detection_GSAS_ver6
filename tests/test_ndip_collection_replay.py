"""Republishing must update the actual JSON/NPZ pair the GUI discovers."""

import json
from pathlib import Path
import zipfile

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


def test_bootstrap_cif_is_archived_but_not_a_scientific_phase_output(tmp_path):
    run, portal = tmp_path / "run", tmp_path / "portal"
    sources = {
        "Technical/bootstrap/NaCl_dummy.cif": "data_bootstrap\n",
        "Models/Refined_CIFs/NaCl_dummy.cif": "data_real_sample\n",
        "Models/Refined_CIFs/other_phase.cif": "data_other_phase\n",
    }
    for name, content in sources.items():
        target = run / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    # Replaying an older result must remove its obsolete published copy too.
    stale = portal / "phases/Technical__bootstrap__NaCl_dummy.cif"
    stale.parent.mkdir(parents=True)
    stale.write_text(sources["Technical/bootstrap/NaCl_dummy.cif"])
    result = collect_outputs(run, portal, mode="full")
    phases = result["artifacts"]["phases"]
    assert {p["source_path"] for p in phases} == set(sources) - {
        "Technical/bootstrap/NaCl_dummy.cif"
    }
    assert not stale.exists()
    with zipfile.ZipFile(portal / "results.zip") as archive:
        for name, content in sources.items():
            assert archive.read(name).decode() == content


def test_bootstrap_cleanup_does_not_remove_a_current_same_name_phase(tmp_path):
    run, portal = tmp_path / "run", tmp_path / "portal"
    internal = run / "Technical/bootstrap/NaCl_dummy.cif"
    internal.parent.mkdir(parents=True)
    internal.write_text("data_bootstrap\n")
    real = run / "Technical__bootstrap__NaCl_dummy.cif"
    real.write_text("data_real_phase\n")
    phases = _copy_collections(run, portal)["phases"]
    assert len(phases) == 1 and phases[0]["source_path"] == real.name
    assert (portal / phases[0]["path"]).read_text() == real.read_text()
    assert internal.read_text() == "data_bootstrap\n"
