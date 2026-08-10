from __future__ import annotations

import csv
import json
import os
import stat
import sys
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from ndip_contracts import CONFIG_SCHEMA, GPX_INDEX_SCHEMA, RESULT_SCHEMA, atomic_write_json  # noqa: E402
from ndip_gpx_handoff import main as gpx_handoff_main  # noqa: E402
from ndip_outputs import collect_outputs  # noqa: E402
from ndip_runner import _extract_db_pack, main  # noqa: E402


def test_atomic_json_is_readable_by_galaxy_postprocessing(tmp_path: Path) -> None:
    output = atomic_write_json(tmp_path / "output.json", {"status": "ok"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "ok"}
    if os.name == "posix":
        mode = stat.S_IMODE(output.stat().st_mode)
        assert mode & stat.S_IRGRP
        assert mode & stat.S_IROTH


def test_configure_and_direct_xray_dry_run(tmp_path: Path) -> None:
    config = tmp_path / "radar_config.yaml"
    assert main(
        [
            "configure",
            "--mode",
            "rapid",
            "--radiation",
            "xray",
            "--instrument-mode",
            "cw",
            "--allowed-elements",
            "Fe, O",
            "--environment-elements",
            "Al",
            "--exclude-region",
            "20:21.5",
            "--reference-mask-preset",
            "Al_fcc",
            "--reference-mask-include-kbeta",
            "--phases-per-hypothesis",
            "4",
            "--rapid-stage-output-limit",
            "12",
            "--gsas-validation-limit",
            "6",
            "--rapid-workers",
            "3",
            "--no-rapid-family-variants",
            "--rapid-final-polish",
            "--magnetic-precheck",
            "--magnetic-q-max",
            "5.5",
            "--magnetic-denominators",
            "2,4",
            "--output",
            str(config),
        ]
    ) == 0
    portable = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert portable["$schema"] == CONFIG_SCHEMA
    assert portable["chemistry"]["sample_elements"] == ["Fe", "O"]
    assert portable["pattern"]["exclude_regions"] == [[20.0, 21.5]]
    assert portable["pattern"]["reference_phase_exclusions"]["presets"] == ["Al_fcc"]
    assert portable["rapid"] == {
        "phases_per_hypothesis": 4,
        "stage_output_limit": 12,
        "gsas_validation_limit": 6,
        "parallel_workers": 3,
        "show_family_variants": False,
        "final_polish_enabled": True,
    }
    assert portable["magnetic_precheck"]["q_max"] == 5.5
    assert portable["magnetic_precheck"]["denominators"] == [2, 4]

    data = tmp_path / "scan.xye"
    data.write_text("10 100 10\n11 120 11\n", encoding="utf-8")
    portal = tmp_path / "portal"
    work = tmp_path / "work"
    assert main(
        [
            "analyze",
            "--config",
            str(config),
            "--data",
            str(data),
            "--instrument-preset",
            "xray_lab_cuka",
            "--db-root",
            str(tmp_path / "database_xray"),
            "--work-dir",
            str(work),
            "--output-dir",
            str(portal),
            "--dry-run",
        ]
    ) == 0
    state = json.loads((portal / "state.json").read_text(encoding="utf-8"))
    resolved = yaml.safe_load((portal / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert resolved["datasets"][0]["instprm_path"].endswith("generated_CuKa_lab.instprm")
    assert resolved["rapid_hypothesis"]["beam_depth"] == 4
    assert resolved["rapid_hypothesis"]["stage_output_limit"] == 12
    assert resolved["rapid_hypothesis"]["gsas_parallel_workers"] == 3
    assert resolved["rapid_hypothesis"]["show_family_variants"] is False
    assert resolved["rapid_hypothesis"]["final_polish_enabled"] is True
    assert resolved["reference_phase_exclusions"]["presets"] == ["Al_fcc"]
    assert (work / "inputs" / "generated_CuKa_lab.instprm").is_file()


def test_full_runtime_profile_materializes_hosted_gui_budget(tmp_path: Path) -> None:
    config = tmp_path / "full_config.yaml"
    assert main(
        [
            "configure",
            "--mode",
            "full",
            "--radiation",
            "neutron",
            "--allowed-elements",
            "Tb, Ge, O",
            "--runtime-profile",
            "thorough",
            "--output",
            str(config),
        ]
    ) == 0
    data = tmp_path / "scan.dat"
    instrument = tmp_path / "scan.instprm"
    data.write_text("1 1 1\n2 2 1\n", encoding="utf-8")
    instrument.write_text("#GSAS-II\n", encoding="utf-8")
    portal = tmp_path / "portal"
    work = tmp_path / "work"
    assert main(
        [
            "analyze",
            "--config",
            str(config),
            "--data",
            str(data),
            "--instrument",
            str(instrument),
            "--db-root",
            str(tmp_path / "database_neutron"),
            "--work-dir",
            str(work),
            "--output-dir",
            str(portal),
            "--dry-run",
        ]
    ) == 0
    resolved = yaml.safe_load((portal / "resolved_config.yaml").read_text(encoding="utf-8"))
    assert resolved["max_passes"] == 3
    assert resolved["min_impurity_percent"] == 0.25
    assert resolved["hist_filter"]["topN"] == 75
    assert resolved["top_candidates"] == 12
    assert resolved["stage4"]["samples"] == 20000
    assert resolved["stage4"]["reps"] == 150
    assert resolved["stage4"]["len_tol_pct"] == 2.0
    assert resolved["stage4"]["ang_tol_deg"] == 5.0


def test_normalized_outputs_include_all_gpx_projects(tmp_path: Path) -> None:
    run = tmp_path / "run"
    rapid = run / "nested" / "rapid_results"
    rapid.mkdir(parents=True)
    (rapid / "summary.json").write_text(
        json.dumps({"live_run": {"timings": {"total_seconds": 12.5}}}), encoding="utf-8"
    )
    with (rapid / "all_gsas_validation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rank", "hypothesis", "rwp"])
        writer.writeheader()
        writer.writerow({"rank": 1, "hypothesis": "Cu + Cu2S", "rwp": 9.4})
    (run / "plots").mkdir(parents=True)
    (run / "plots" / "fit.png").write_bytes(b"not-a-real-png")
    (run / "checkpoints" / "pass1").mkdir(parents=True)
    (run / "checkpoints" / "pass1" / "accepted.gpx").write_bytes(b"GPX-1")
    (run / "checkpoints" / "rollback.gpx").write_bytes(b"GPX-2")

    portal = tmp_path / "portal"
    result = collect_outputs(run, portal, mode="rapid", run_name="demo")
    index = json.loads((portal / "gpx_index.json").read_text(encoding="utf-8"))
    assert result["$schema"] == RESULT_SCHEMA
    assert index["$schema"] == GPX_INDEX_SCHEMA
    assert index["project_count"] == 2
    assert len(list((portal / "gpx").glob("*.gpx"))) == 2
    assert all(project.get("collection_path", "").startswith("gpx/") for project in index["projects"])
    assert result["hypotheses"][0]["hypothesis"] == "Cu + Cu2S"
    assert (portal / "report.html").is_file()
    report = (portal / "report.html").read_text(encoding="utf-8")
    assert "Fit and diagnostic plots" in report
    assert "plots/plots__fit.png" in report
    assert "Continue in GSAS-II" in report
    assert "Cu + Cu2S" in report
    assert "Stage timing" in report
    assert (portal / "results.zip").is_file()


def test_ipts_resolver_copies_one_unambiguous_reduced_run(tmp_path: Path) -> None:
    shared = tmp_path / "SNS" / "HB2A" / "IPTS-123" / "shared" / "reduced"
    shared.mkdir(parents=True)
    pattern = shared / "HB2A_42_b1.gsa"
    instrument = shared / "HB2A_42_b1.instprm"
    pattern.write_text("BANK 1\n", encoding="utf-8")
    instrument.write_text("#GSAS-II\n", encoding="utf-8")
    pattern_output = tmp_path / "selected.gsa"
    instrument_output = tmp_path / "selected.instprm"
    metadata_output = tmp_path / "input.json"

    assert main(
        [
            "resolve-ipts",
            "--instrument",
            "HB2A",
            "--ipts",
            "IPTS-123",
            "--run",
            "42",
            "--bank",
            "b1",
            "--facility-root",
            str(tmp_path / "SNS"),
            "--pattern-output",
            str(pattern_output),
            "--instrument-output",
            str(instrument_output),
            "--metadata-output",
            str(metadata_output),
        ]
    ) == 0
    metadata = json.loads(metadata_output.read_text(encoding="utf-8"))
    assert metadata["source"] == "ipts"
    assert pattern_output.read_text(encoding="utf-8") == "BANK 1\n"
    assert instrument_output.is_file()


def test_custom_database_archive_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.txt", "no")
    try:
        _extract_db_pack(archive, tmp_path / "extract")
    except ValueError as exc:
        assert "Unsafe path" in str(exc)
    else:
        raise AssertionError("unsafe archive was accepted")


def test_compare_series_writes_csv_and_html(tmp_path: Path) -> None:
    summaries = []
    for index, seconds in enumerate((10.0, 20.0), start=1):
        path = tmp_path / f"summary_{index}.json"
        path.write_text(
            json.dumps(
                {
                    "$schema": RESULT_SCHEMA,
                    "run_name": f"run_{index}",
                    "analysis_mode": "rapid",
                    "status": "complete",
                    "summary": {"live_run": {"timings": {"total_seconds": seconds}}},
                    "phases": [],
                    "hypotheses": [],
                    "gpx_projects": [],
                    "provenance": {},
                    "artifacts": {},
                }
            ),
            encoding="utf-8",
        )
        summaries.append(path)
    csv_output = tmp_path / "series.csv"
    html_output = tmp_path / "series.html"
    argv = ["compare-series"]
    for path in summaries:
        argv.extend(["--summary", str(path)])
    argv.extend(["--csv-output", str(csv_output), "--html-output", str(html_output)])
    assert main(argv) == 0
    assert "run_1" in csv_output.read_text(encoding="utf-8")
    assert "RADAR-PD series comparison" in html_output.read_text(encoding="utf-8")


def test_gpx_handoff_preserves_project_and_provenance(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "accepted_hypothesis.gpx"
    project.write_bytes(b"GSAS-II project fixture")
    index = tmp_path / "gpx_index.json"
    index.write_text(
        json.dumps(
            {
                "$schema": GPX_INDEX_SCHEMA,
                "projects": [
                    {
                        "collection_name": project.name,
                        "stage": "hypothesis_refinement",
                        "status": "accepted",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "handoff.gpx"
    metadata = tmp_path / "handoff.json"
    report = tmp_path / "handoff.html"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ndip_gpx_handoff.py",
            "--project",
            str(project),
            "--index",
            str(index),
            "--output-project",
            str(output),
            "--metadata",
            str(metadata),
            "--report",
            str(report),
        ],
    )
    assert gpx_handoff_main() == 0
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert output.read_bytes() == project.read_bytes()
    assert payload["stage"] == "hypothesis_refinement"
    assert payload["checkpoint_status"] == "accepted"
    assert "interactive GSAS-II" in report.read_text(encoding="utf-8")
