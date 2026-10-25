import asyncio, copy, hashlib, io, json, zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.galaxy_service import GalaxyService
from radar_pd_nova.models import AnalysisConfig
from radar_pd_nova.configuration import config_from_contract
from radar_pd_nova.powgen_watch import WatchState
from test_scan_interpretation import scan, packet


def app():
    application = RadarPdNovaApp()
    application.server.state.sample_elements = "Fe"
    application._refresh_post_analysis([scan(1), scan(2, pattern=packet(shift=0.02))], pending=[])
    return application


def test_changes_panel_is_compact_but_every_finding_has_accessible_evidence():
    application = app()
    state = application.server.state
    assert state.scan_change_can_plot and state.scan_change_run == "NOM_2_BANK2"
    assert len(state.scan_change_findings) <= 3
    application.open_scan_change()
    assert state.scan_change_open and state.scan_change_figure["data"]
    assert state.scan_change_regions[1:]
    before = copy.deepcopy(application._interpretation_report)
    application.select_scan_change_region("0")
    assert state.scan_change_figure["layout"]["xaxis"]["range"]
    assert application._interpretation_report == before
    application.toggle_scan_change_fit(False)
    assert len(state.scan_change_figure["data"]) == 3
    assert application._interpretation_report == before


def test_follow_latest_preserves_operator_choice_of_older_scan():
    application = app()
    state = application.server.state
    application._refresh_post_analysis([scan(1), scan(2), scan(3)])
    assert state.scan_change_run == "NOM_3_BANK2"
    application.select_scan_change("NOM_1_BANK2")
    application._refresh_post_analysis([scan(1), scan(2), scan(3), scan(4)])
    assert state.scan_change_run == "NOM_1_BANK2"


def test_display_filters_do_not_redefine_inference_history():
    application = app()
    all_scans = [scan(1), scan(2), scan(3)]
    application._post_analysis_scope = "watch:/SNS/NOM/IPTS-1/shared/autoreduce"
    application._interpretation_all_watch_scans = all_scans
    application._refresh_post_analysis([all_scans[-1]])
    assert len(application._interpretation_report["records"]) == 3
    assert application._interpretation_report["records"][-1]["comparisons"]


def test_export_contains_exact_masked_curves_settings_and_a_replayable_manifest(tmp_path):
    application = app()
    downloads = []
    application.download_file = lambda *args: downloads.append(args)
    application.download_scan_change()
    with zipfile.ZipFile(io.BytesIO(downloads[0][2])) as z:
        names = set(z.namelist())
        assert {
            "manifest.json",
            "pattern-0.json",
            "pattern-1.json",
            "settings.json",
            "comparison.json",
            "findings.json",
        } <= names
        expected = packet()
        expected["source_gpx_sha256"] = "gpx1"
        assert json.loads(z.read("pattern-0.json")) == expected
        manifest = json.loads(z.read("manifest.json"))
        assert len(manifest["scans"]) == 2
        assert all("pattern" not in s for s in manifest["scans"])
        z.extractall(tmp_path)
    from radar_pd_nova.scan_interpretation import main

    output = tmp_path / "replay.json"
    assert (
        main(
            [
                str(tmp_path / "manifest.json"),
                "--settings",
                str(tmp_path / "settings.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text())["records"][-1]["comparisons"]


def test_checkpoint_and_portable_configuration_restore_references_without_submitting_jobs():
    application = app()
    saved = []
    state = WatchState(history_id="history1", instrument="NOM")
    application._powgen_controller = SimpleNamespace(
        state=state, persist_state=lambda: saved.append(state.to_json())
    )
    application._post_analysis_scope = "watch:/SNS/NOM/IPTS-1/shared/autoreduce"
    application.pin_scan_change_reference()
    asyncio.run(application.save_scan_change())
    restored = WatchState.from_json(saved[-1])
    assert restored.scan_interpretation["references"]
    assert restored.interpretation_report["records"]
    assert '"x": [' not in saved[-1] and '"y": [' not in saved[-1]
    config = application._configuration()
    assert (
        config_from_contract(config.portable_contract()).scan_interpretation
        == config.scan_interpretation
    )


def service(tmp_path):
    result = object.__new__(GalaxyService)
    result.output_root = tmp_path
    return result


def receipt(raw):
    return {
        "name": "Pattern_observation.json",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_gpx_sha256": json.loads(raw)["source_gpx_sha256"],
    }


def test_galaxy_fetches_only_checksum_addressed_observation_and_restores_cache(tmp_path):
    raw = json.dumps(packet()).encode()
    calls = []
    s = service(tmp_path)
    s.collection_elements = lambda id: [
        {"name": "Pattern_observation.json", "id": "dataset1"},
        {"name": "large-other-file.csv", "id": "avoid"},
    ]

    def text(id, *, max_bytes):
        calls.append((id, max_bytes))
        return raw.decode()

    s.load_text_document = text
    first = s.cache_pattern_observation(receipt(raw), collection_id="tables1")
    assert calls == [("dataset1", 16000000)]
    assert Path(first["local_path"]).read_bytes() == raw
    Path(first["local_path"]).unlink()
    restored = s.cache_pattern_observation(first)
    assert restored == first and len(calls) == 2
    s.cache_pattern_observation(restored)
    assert len(calls) == 2


@pytest.mark.parametrize("case", ["checksum", "ambiguous", "project"])
def test_corrupt_or_ambiguous_galaxy_artifacts_are_not_used(tmp_path, case):
    raw = json.dumps(packet()).encode()
    s = service(tmp_path)
    desc = receipt(raw)
    s.collection_elements = lambda id: [{"name": "Pattern_observation.json", "id": "1"}] * (
        2 if case == "ambiguous" else 1
    )
    s.load_text_document = lambda *args, **kwargs: raw.decode()
    if case == "checksum":
        desc["sha256"] = "0" * 64
    if case == "project":
        desc["source_gpx_sha256"] = "other"
    with pytest.raises(ValueError):
        s.cache_pattern_observation(desc, collection_id="tables")
    assert not list(tmp_path.rglob("*.json"))


def test_background_replay_coalesces_new_arrivals_without_applying_a_stale_result():
    import threading

    application = app()
    engine = application._scan_interpreter
    evaluate = engine.evaluate
    started = threading.Event()
    release = threading.Event()
    seen = []

    def slow(scans, *args, **kwargs):
        seen.append(len(scans))
        started.set()
        release.wait(timeout=5)
        return evaluate(scans, *args, **kwargs)

    engine.evaluate = slow

    async def exercise():
        application._refresh_scan_interpretation([scan(1), scan(2), scan(3)])
        await asyncio.to_thread(started.wait, 2)
        assert application._interpretation_report["records"][-1]["run_id"] == "NOM_2_BANK2"
        application._refresh_scan_interpretation([scan(1), scan(2), scan(3), scan(4)])
        release.set()
        await application._interpretation_task

    asyncio.run(exercise())
    assert seen == [3, 4]
    assert application._interpretation_report["records"][-1]["run_id"] == "NOM_4_BANK2"


def test_conflicting_fit_revisions_have_an_actionable_ui_selection():
    application = app()
    a = scan(1)
    revision = scan(1)
    revision["run_id"] = "revision2"
    revision["observables"]["source_gpx_sha256"] = "new-gpx"
    revision["pattern"]["source_gpx_sha256"] = "new-gpx"
    application._refresh_post_analysis([a, revision, scan(2)])
    application.select_scan_change(a["run_id"])
    state = application.server.state
    assert len(state.scan_change_revision_options) == 2
    state.scan_change_revision = "new-gpx"
    application.choose_scan_change_revision()
    assert (
        application._interpretation_report["records"][-1]["comparisons"][0]["reference_run_id"]
        == "revision2"
    )


def test_device_mapping_is_explicit_scoped_and_does_not_rewrite_raw_receipts():
    application = app()
    rows = [scan(1), scan(2)]
    gauge = {
        "quantity": "pressure",
        "role": "measured",
        "status": "needs_mapping",
        "channel": "gauge",
        "value": 1000.0,
        "minimum": 1000.0,
        "maximum": 1000.0,
        "unit": "Pa",
        "coverage_fraction": 1,
        "statistic": "time_weighted_step_hold",
    }
    for row in rows:
        row["metadata"]["conditions"]["channels"].append(copy.deepcopy(gauge))
    original = copy.deepcopy(rows)
    application._refresh_post_analysis(rows)
    state = application.server.state
    assert state.scan_change_binding_options
    state.scan_change_binding = "gauge"
    application.bind_scan_change_condition()
    assert not state.scan_interpretation["condition_bindings"]
    state.scan_change_binding_confirmed = True
    application.bind_scan_change_condition()
    assert state.scan_interpretation["condition_bindings"]["NOM:IPTS-1:s1"]["pressure"] == "gauge"
    assert any(
        c["quantity"] == "pressure"
        for c in application._interpretation_report["records"][-1]["conditions"]
    )
    assert rows == original


def test_clearing_the_beamline_cannot_reintroduce_old_curve_evidence():
    application = app()
    application._interpretation_all_watch_scans = [scan(1), scan(2)]
    application._post_analysis_scope = "watch:/SNS/NOM/IPTS-1/shared/autoreduce"
    application._powgen_controller = None
    application._sync_powgen_rows()
    assert not application._interpretation_report["records"]
    assert not application.server.state.scan_change_can_plot
    assert not application.server.state.scan_change_run


def test_concurrent_restores_publish_the_same_verified_artifact_without_temporary_file_races(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    raw=json.dumps(packet()).encode();s=service(tmp_path);barrier=threading.Barrier(4)
    desc={**receipt(raw),'dataset_id':'dataset1'}
    def read(*args,**kwargs):
        barrier.wait(timeout=5)
        return raw.decode()
    s.load_text_document=read
    with ThreadPoolExecutor(max_workers=4)as pool:
        results=list(pool.map(lambda _:s.cache_pattern_observation(desc),range(4)))
    assert all(r['local_path']==results[0]['local_path']for r in results)
    assert Path(results[0]['local_path']).read_bytes()==raw
    assert not list(tmp_path.rglob('*.partial'))


def test_unchanged_cached_curves_are_not_reread_on_every_watch_poll_but_changed_bytes_are_rejected(tmp_path,monkeypatch):
    from radar_pd_nova import scan_interpretation
    s=service(tmp_path);raw=json.dumps(packet()).encode();desc={**receipt(raw),'dataset_id':'dataset1'}
    s.load_text_document=lambda *args,**kwargs:raw.decode()
    first=s.cache_pattern_observation(desc)
    calls=[];load=scan_interpretation.load_pattern
    def read(*args,**kwargs):calls.append(True);return load(*args,**kwargs)
    monkeypatch.setattr(scan_interpretation,'load_pattern',read)
    s.cache_pattern_observation(first)
    assert not calls
    Path(first['local_path']).write_text('corrupt local cache')
    with pytest.raises(ValueError,match='checksum'):s.cache_pattern_observation(first)
    assert calls==[True]
