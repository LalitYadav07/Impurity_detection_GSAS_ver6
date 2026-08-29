from __future__ import annotations

import csv
import json
import os
import pickle
import stat
import sys
import zipfile
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
NOVA_SRC = ROOT / "nova" / "src"
if str(NOVA_SRC) not in sys.path:
    sys.path.insert(0, str(NOVA_SRC))

from ndip_contracts import CONFIG_SCHEMA, GPX_INDEX_SCHEMA, RESULT_SCHEMA, atomic_write_json  # noqa: E402
from ndip_gpx_handoff import main as gpx_handoff_main  # noqa: E402
from ndip_outputs import _read_gpx_records, collect_outputs  # noqa: E402
from ndip_runner import (  # noqa: E402
    _extract_db_pack,
    _finalize_successful_stages,
    _instrument_mode_from_instprm,
    _resolve_db_pack_original_json,
    main,
)
from radar_pd_nova.results import build_result_view, load_plot_with_fallback  # noqa: E402


def test_atomic_json_is_readable_by_galaxy_postprocessing(tmp_path: Path) -> None:
    output = atomic_write_json(tmp_path / "output.json", {"status": "ok"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "ok"}
    if os.name == "posix":
        mode = stat.S_IMODE(output.stat().st_mode)
        assert mode & stat.S_IRGRP
        assert mode & stat.S_IROTH


def test_successful_full_run_finalizes_all_expected_stages() -> None:
    state = {
        "status": "running",
        "stages": {
            "candidate_screening": {"status": "running", "message": "Screening"},
            "refinement": {"status": "running", "message": "Refining"},
        },
    }

    finalized = _finalize_successful_stages(state, "full")

    assert state["stages"]["candidate_screening"]["status"] == "running"
    assert set(finalized["stages"]) == {
        "main_phase",
        "candidate_screening",
        "lattice_nudge",
        "refinement",
        "report",
    }
    assert all(stage["status"] == "complete" for stage in finalized["stages"].values())


def test_successful_rapid_run_finalizes_all_expected_stages() -> None:
    finalized = _finalize_successful_stages({"stages": {}}, "rapid")

    assert set(finalized["stages"]) == {
        "input_preparation",
        "coarse_search",
        "lattice_nudge",
        "pattern_scoring",
        "final_refinement",
        "report",
    }
    assert all(stage["status"] == "complete" for stage in finalized["stages"].values())


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
    instrument.write_text("#GSAS-II instrument parameter file\nType:PNT\ndifC:22597.136\n", encoding="utf-8")
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
    assert resolved["runtime_profile"] == "thorough"
    assert resolved["rwp_improve_eps"] == 0.0
    assert resolved["instrument_mode"] == "tof"
    assert resolved["datasets"][0]["mode"] == "tof"


def test_advanced_portable_controls_materialize_without_default_substitution(tmp_path: Path) -> None:
    config = tmp_path / "advanced_config.yaml"
    assert main(
        [
            "configure",
            "--mode",
            "full",
            "--radiation",
            "neutron",
            "--allowed-elements",
            "Fe, O",
            "--runtime-profile",
            "custom",
            "--output",
            str(config),
        ]
    ) == 0
    portable = yaml.safe_load(config.read_text(encoding="utf-8"))
    portable["pattern"]["reference_phase_exclusions"] = {
        "enabled": True,
        "presets": ["V_bcc"],
        "window_mode": "fixed",
        "half_width": 120.0,
        "fwhm_factor": 7.0,
        "fractional_d_tolerance": 0.004,
        "zero_tolerance": 30.0,
        "min_half_width": 80.0,
        "max_half_width": 600.0,
        "include_cu_kbeta": False,
    }
    portable["light_calibration"] = {"enabled": True}
    portable["full"].update(
        {
            "dedup_threshold": 0.91,
            "score_q_max": 9.5,
            "pearson_cell_min_r": 0.35,
            "lattice_tiebreak_score_tol": 0.0012,
            "candidate_pruning": False,
            "knee_min_points_hist": 7,
            "knee_min_relative_span": 0.08,
            "knee_keep_if_no_knee": 4,
            "knee_keep_at_most": 11,
            "excluded_space_groups": [1, 2, 15],
        }
    )
    config.write_text(yaml.safe_dump(portable, sort_keys=False), encoding="utf-8")

    data = tmp_path / "scan.dat"
    instrument = tmp_path / "scan.instprm"
    data.write_text("1 1 1\n2 2 1\n", encoding="utf-8")
    instrument.write_text("Type:PNT\ndifC:10000\n", encoding="utf-8")
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
    assert resolved["reference_phase_exclusions"]["half_width"] == 120.0
    assert resolved["corr_threshold"] == 0.91
    assert resolved["exclude_sg"] == [1, 2, 15]
    assert resolved["knee_filter"]["enable_nudge"] is False
    assert resolved["knee_filter"]["min_points_hist"] == 7
    assert resolved["knee_filter"]["min_rel_span"] == 0.08
    assert resolved["knee_filter"]["max_keep_if_no_knee"] == 4
    assert resolved["knee_filter"]["max_keep_at_most"] == 11
    assert resolved["stage4"]["score_q_max"] == 9.5
    assert resolved["stage4"]["pearson_cell_refine_min_r"] == 0.35
    assert resolved["stage4"]["lattice_tiebreak_score_tol"] == 0.0012
    assert resolved["light_calibration"]["enabled"] is True


def test_instrument_mode_is_inferred_from_real_gsasii_profiles() -> None:
    assert _instrument_mode_from_instprm(ROOT / "examples" / "tbssl" / "hb2a_si_ge113.instprm") == "cw"
    assert (
        _instrument_mode_from_instprm(
            ROOT / "examples" / "lk99" / "2023A_June_HighRes_60HzB3_CWL2p665.instprm"
        )
        == "tof"
    )


def test_explicit_instrument_mode_rejects_mismatched_profile(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    assert main(
        [
            "configure",
            "--mode",
            "full",
            "--radiation",
            "neutron",
            "--instrument-mode",
            "cw",
            "--allowed-elements",
            "Fe, O",
            "--output",
            str(config),
        ]
    ) == 0
    data = tmp_path / "pattern.dat"
    instrument = tmp_path / "tof.instprm"
    data.write_text("1 1 1\n2 2 1\n", encoding="utf-8")
    instrument.write_text("Type:PNT\ndifC:10000\n", encoding="utf-8")
    assert (
        main(
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
                str(tmp_path / "work"),
                "--output-dir",
                str(tmp_path / "portal"),
                "--dry-run",
            ]
        )
        != 0
    )


def test_normalized_outputs_publish_handoff_projects_and_archive_all_checkpoints(tmp_path: Path) -> None:
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
    (run / "checkpoints" / "accepted.temp.bak0.gpx").write_bytes(b"GPX-3")

    portal = tmp_path / "portal"
    result = collect_outputs(run, portal, mode="rapid", run_name="demo")
    index = json.loads((portal / "gpx_index.json").read_text(encoding="utf-8"))
    assert result["$schema"] == RESULT_SCHEMA
    assert index["$schema"] == GPX_INDEX_SCHEMA
    assert index["project_count"] == 1
    assert len(list((portal / "gpx").glob("*.gpx"))) == 1
    assert all(project.get("collection_path", "").startswith("gpx/") for project in index["projects"])
    assert result["hypotheses"][0]["hypothesis"] == "Cu + Cu2S"
    assert (portal / "report.html").is_file()
    report = (portal / "report.html").read_text(encoding="utf-8")
    assert "Fit and diagnostic plots" in report
    assert "plots/plots__fit.png" in report
    assert "Continue in GSAS-II" in report
    assert "Cu + Cu2S" in report
    assert "Stage timing" in report
    overview = (portal / "overview.tsv").read_text(encoding="utf-8")
    assert "Rapid Hypothesis Mode" in overview
    assert "Cu + Cu2S" in overview
    assert "9.400" in overview
    assert (portal / "results.zip").is_file()
    with zipfile.ZipFile(portal / "results.zip") as archive:
        names = set(archive.namelist())
    assert "checkpoints/rollback.gpx" in names
    assert "ndip/report.html" in names
    assert "ndip/overview.tsv" in names
    assert any(name.startswith("ndip/plots/") for name in names)


def test_full_main_only_project_is_published_as_the_main_phase_anchor(tmp_path: Path) -> None:
    run = tmp_path / "run"
    projects = run / "Technical" / "GSAS_Projects"
    projects.mkdir(parents=True)
    (projects / "IPTS-37876_PG3_63799_project.gpx").write_bytes(b"GPX")
    (projects / "IPTS-37876_PG3_63799_stage0.gpx").write_bytes(b"STAGE-0")

    result = collect_outputs(run, tmp_path / "portal", mode="full", run_name="main-only")

    assert [item["collection_name"] for item in result["gpx_projects"]] == [
        "02_Main_phase_anchor.gpx"
    ]
    assert result["gpx_projects"][0]["stage"] == "main_phase_anchor"
    assert result["gpx_projects"][0]["status"] == "accepted"


def test_full_polished_main_anchor_supersedes_the_fallback_project(tmp_path: Path) -> None:
    run = tmp_path / "run"
    projects = run / "Technical" / "GSAS_Projects"
    projects.mkdir(parents=True)
    (projects / "demo_project.gpx").write_bytes(b"FALLBACK")
    (projects / "seq_final_main_polished.gpx").write_bytes(b"POLISHED")

    result = collect_outputs(run, tmp_path / "portal", mode="full", run_name="polished-main")

    assert len(result["gpx_projects"]) == 1
    assert result["gpx_projects"][0]["collection_name"] == "02_Main_phase_anchor.gpx"
    assert result["gpx_projects"][0]["source_path"].endswith("seq_final_main_polished.gpx")


def test_published_gpx_uses_unique_scientific_phase_names(tmp_path: Path) -> None:
    run = tmp_path / "run"
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(parents=True)
    source = checkpoints / "accepted.gpx"
    phase_ids = (
        "user_00011_yourcustomfilename_collcode258024_bbac164417",
        "user_00012_another_filename_abc123",
    )
    records = [
        [["Controls", {}]],
        [
            ["PWDR test", [None]],
            ["Reflection Lists", {phase_ids[0]: {"RefList": []}, phase_ids[1]: {"RefList": []}}],
        ],
        [
            ["Phases", None],
            [phase_ids[0], {"General": {"Name": phase_ids[0]}}],
            [phase_ids[1], {"General": {"Name": phase_ids[1]}}],
        ],
        [["Restraints", {phase_ids[0]: {"Bond": []}, phase_ids[1]: {"Bond": []}}]],
    ]
    with source.open("wb") as handle:
        for record in records:
            pickle.dump(record, handle, protocol=1)

    catalog = tmp_path / "catalog_deduplicated.csv"
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "display_name", "pretty_formula", "SG_symbol", "space_group"],
        )
        writer.writeheader()
        for phase_id in phase_ids:
            writer.writerow(
                {
                    "id": phase_id,
                    "display_name": "Al Fe2 V",
                    "pretty_formula": "Al Fe2 V",
                    "SG_symbol": "F m -3 m",
                    "space_group": "225.0",
                }
            )

    portal = tmp_path / "portal"
    collect_outputs(
        run,
        portal,
        mode="full",
        run_name="friendly-gpx",
        phase_catalog_csv=catalog,
        include_archive=False,
    )

    published = next((portal / "gpx").glob("*.gpx"))
    published_records = _read_gpx_records(published)
    phase_record = next(record for record in published_records if record[0][0] == "Phases")
    names = [item[0] for item in phase_record[1:]]
    assert names == ["AlFe2V (SG Fm-3m, 225)", "AlFe2V (SG Fm-3m, 225) [2]"]
    assert [item[1]["General"]["Name"] for item in phase_record[1:]] == names
    histogram = next(record for record in published_records if record[0][0] == "PWDR test")
    assert list(histogram[1][1]) == names
    restraints = next(record for record in published_records if record[0][0] == "Restraints")
    assert list(restraints[0][1]) == names

    original_records = _read_gpx_records(source)
    original_phase_record = next(record for record in original_records if record[0][0] == "Phases")
    assert [item[0] for item in original_phase_record[1:]] == list(phase_ids)


def test_full_archive_plot_manifest_renders_in_scientific_results(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "pipeline_summary.json").write_text(
        json.dumps({"final": {"final_rwp": 8.5}}),
        encoding="utf-8",
    )
    source_plot = run / "plots" / "custom_database_output_00017.png"
    source_plot.parent.mkdir()
    source_plot.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
            "0000000d49444154789c63600000020001e221bc330000000049454e44ae426082"
        )
    )

    portal = tmp_path / "portal"
    collect_outputs(run, portal, mode="full", run_name="full-plot-contract")
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(portal / "results.zip") as archive:
        archive.extractall(extracted)
    result = json.loads((extracted / "ndip" / "summary.json").read_text(encoding="utf-8"))

    view = build_result_view(result, extracted)
    selected = load_plot_with_fallback(view.plots, view.primary_plot_path)

    assert len(result["artifacts"]["plots"]) == 1
    assert len(view.plots) == 1
    assert selected is not None
    assert len(selected[2].layout.images) == 1


def test_normalized_outputs_fall_back_to_pattern_rank_when_gsas_is_skipped(tmp_path: Path) -> None:
    run = tmp_path / "run"
    rapid = run / "rapid_results"
    nudge = rapid / "nudge" / "live_run"
    nudge.mkdir(parents=True)
    (rapid / "summary.json").write_text(json.dumps({"live_run": {}}), encoding="utf-8")
    with (rapid / "all_gsas_validation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=["rank", "hypothesis", "rwp"]).writeheader()
    with (nudge / "reranked_512_after_radar_nudge.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank512", "formulas", "space_groups", "score512", "phase_coefs512"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "rank512": 1,
                "formulas": "Cu|Cu2S",
                "space_groups": "225|14",
                "score512": 0.81,
                "phase_coefs512": "1.2|0.3",
            }
        )
    (nudge / "target_512.npz").write_bytes(b"array")

    portal = tmp_path / "portal"
    result = collect_outputs(run, portal, mode="rapid", run_name="pattern-only")
    report = (portal / "report.html").read_text(encoding="utf-8")

    assert result["hypothesis_stage"] == "pattern_scoring"
    assert result["hypotheses"][0]["rank512"] == "1"
    assert "Cu (SG 225) + Cu2S (SG 14)" in report
    assert "Pattern score" in report
    assert "comparative pattern weights" in report
    assert not list((portal / "plots").glob("*.npz"))
    assert list((portal / "diagnostics").glob("*.npz"))


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


def test_custom_database_archive_resolves_portable_pack_root(tmp_path: Path) -> None:
    archive = tmp_path / "library.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("library/catalog_deduplicated.csv", "id,pretty_formula\n1,Cu\n")
        handle.writestr("library/mp_experimental_stable.csv", "id\n1\n")
        handle.writestr("library/profiles64/profiles64.npz", b"npz")
        handle.writestr("library/profiles64/index.csv", "id,row\n1,0\n")

    extracted = _extract_db_pack(archive, tmp_path / "extract")

    assert extracted == (tmp_path / "extract" / "library").resolve()


def test_augmented_database_uses_mounted_builtin_structure_metadata(tmp_path: Path) -> None:
    pack_root = tmp_path / "augmented"
    pack_root.mkdir()
    (pack_root / "manifest.json").write_text(
        json.dumps({"kind": "augmented", "source_type": "neutron"}),
        encoding="utf-8",
    )
    builtin_root = tmp_path / "database_neutron"
    builtin_root.mkdir()
    builtin_original = builtin_root / "highsymm_metadata.json"
    builtin_original.write_text("{}", encoding="utf-8")

    resolved = _resolve_db_pack_original_json(
        pack_root,
        builtin_db_root=builtin_root,
        radiation="neutron",
    )

    assert resolved == builtin_original.resolve()


def test_augmented_database_rejects_radiation_mismatch(tmp_path: Path) -> None:
    pack_root = tmp_path / "augmented"
    pack_root.mkdir()
    (pack_root / "manifest.json").write_text(
        json.dumps({"kind": "augmented", "source_type": "xray"}),
        encoding="utf-8",
    )

    try:
        _resolve_db_pack_original_json(
            pack_root,
            builtin_db_root=tmp_path / "database_neutron",
            radiation="neutron",
        )
    except ValueError as exc:
        assert "built for xray" in str(exc)
    else:
        raise AssertionError("an X-ray augmented library was accepted for neutron data")


def test_augmented_database_dry_run_materializes_hybrid_structure_sources(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    assert main(
        [
            "configure",
            "--mode",
            "full",
            "--radiation",
            "neutron",
            "--allowed-elements",
            "Fe, V, Al",
            "--output",
            str(config),
        ]
    ) == 0
    data = tmp_path / "scan.gsa"
    data.write_text("BANK 1\n", encoding="utf-8")
    instrument = tmp_path / "instrument.instprm"
    instrument.write_text(
        "#GSAS-II instrument parameter file\nType:PNT\ndifC:22598.5\n",
        encoding="utf-8",
    )

    builtin_root = tmp_path / "database_neutron"
    builtin_root.mkdir()
    builtin_original = builtin_root / "highsymm_metadata.json"
    builtin_original.write_text("{}", encoding="utf-8")

    archive = tmp_path / "augmented.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(
            "library/catalog_deduplicated.csv",
            "id,pretty_formula,space_group,SG_symbol,elements_list,elements_mask_hi,elements_mask_lo,npz,n_reflections\n",
        )
        handle.writestr("library/mp_experimental_stable.csv", "material_id\n")
        handle.writestr("library/profiles64/profiles64.npz", b"npz")
        handle.writestr("library/profiles64/index.csv", "id,row\n")
        handle.writestr("library/cif_map.json", "{}")
        handle.writestr(
            "library/manifest.json",
            json.dumps({"kind": "augmented", "source_type": "neutron"}),
        )

    work = tmp_path / "work"
    portal = tmp_path / "portal"
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
            str(builtin_root),
            "--db-pack",
            str(archive),
            "--work-dir",
            str(work),
            "--output-dir",
            str(portal),
            "--dry-run",
        ]
    ) == 0

    resolved = yaml.safe_load((portal / "resolved_config.yaml").read_text(encoding="utf-8"))
    extracted_root = (work / "custom_database" / "library").resolve()
    assert Path(resolved["db"]["catalog_csv"]) == extracted_root / "catalog_deduplicated.csv"
    assert Path(resolved["db"]["cif_map_json"]) == extracted_root / "cif_map.json"
    assert Path(resolved["db"]["original_json"]) == builtin_original.resolve()


def test_custom_database_archive_rejects_cif_source_bundle(tmp_path: Path) -> None:
    archive = tmp_path / "sources.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("cifs/phase.cif", "data_phase\n")

    try:
        _extract_db_pack(archive, tmp_path / "extract")
    except ValueError as exc:
        assert "source CIF files" in str(exc)
        assert "portable custom library" in str(exc)
    else:
        raise AssertionError("CIF source bundle was accepted as a database")


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
