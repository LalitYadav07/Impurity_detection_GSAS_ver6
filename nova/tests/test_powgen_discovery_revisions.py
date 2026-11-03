"""Revision/physical-bank contracts use the real parser and staged inputs."""
import os

import pytest

from radar_pd_nova import powgen_controller
from radar_pd_nova.powgen_controller import PowgenExperimentSettings, PowgenWatchController, preflight_powgen_experiment
from radar_pd_nova.powgen_watch import WatchDefinition
from test_powgen_controller import FakeGalaxyService
from test_powgen_setup import source


class SampleIndex:
    def identity(self, run):
        return {"sample_id": "sample", "identity_status": "resolved"}


def controller(monkeypatch, root, **settings):
    monkeypatch.setattr(WatchDefinition, "source_directory", property(lambda _: str(root)))
    monkeypatch.setattr(powgen_controller, "read_powgen_scan_metadata", lambda *_: {})
    return PowgenWatchController(FakeGalaxyService(),
        PowgenExperimentSettings("IPTS-2767", "history-1", "config-hda", "", **settings),
        sample_index=SampleIndex())


@pytest.mark.parametrize("target", ["data", "script"])
def test_replacement_between_discovery_and_launch_cannot_submit(monkeypatch, tmp_path, target):
    path = source(tmp_path)
    view = controller(monkeypatch, tmp_path)
    run = view.discover(checkpoint=False)[0]
    path = path if target == "data" else path.with_suffix(".py")
    path.write_text(path.read_text().replace("7.25", "9.75") if target == "data" else path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after discovery"):
        view.launch_submission(run)
    assert not view.service.snapshots
    view.discover(checkpoint=False)
    assert view.state.discovered[run.run_id].scan_metadata["source_revision_warning"]
    assert not view.due_submissions()


def test_script_content_change_with_preserved_stat_is_rejected(monkeypatch, tmp_path):
    path = source(tmp_path)
    view = controller(monkeypatch, tmp_path)
    run = view.discover(checkpoint=False)[0]
    script = path.with_suffix(".py")
    stat = script.stat()
    script.write_text(script.read_text().replace("gsas topas", "gsas other"))
    os.utime(script, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="script content changed"):
        view.launch_submission(run)
    assert not view.service.snapshots


def test_single_to_multibank_revision_does_not_admit_same_bank_twice(monkeypatch, tmp_path):
    options = dict(run=37875, banks=(3,), cycle="2017_1_2_11A_CAL", calibration="PG3_PAC_2BANKS.h5")
    source(tmp_path, **options)
    view = controller(monkeypatch, tmp_path)
    old = view.discover(checkpoint=False)[0]
    record = view.launch_submission(old)
    view.acknowledge_submission(old, record, persist=False)
    view.state.mark_completed(old, ("original-output",), galaxy_job_id=record.galaxy_job_id)
    source(tmp_path, **{**options, "banks": (3, 4)})
    found = view.discover(checkpoint=False)
    assert old.run_id == "PG3_37875"
    assert [run.run_id for run in found] == ["PG3_37875_BANK4"]
    assert view.state.completed[old.run_id].scan_metadata["source_revision_warning"]
    assert found[0].scan_metadata["source_revision_warning"]
    assert not view.due_submissions()
    assert len(view.service.snapshots) == 1


def test_latest_test_and_preflight_choose_latest_matching_bank(monkeypatch, tmp_path):
    source(tmp_path, run=43124, banks=(3,), wavelength=2.665)
    source(tmp_path, run=43125, banks=(2,), wavelength=1.5)
    view = controller(monkeypatch, tmp_path, selected_banks=(3,), initial_backfill_limit=1)
    check = preflight_powgen_experiment("IPTS-2767", selected_banks=(3,),
        initial_backfill_limit=1, sample_index=SampleIndex())
    assert check["ready"] and check["analysis_count"] == 1
    assert check["available_banks"] == [2, 3]
    found = view.discover(checkpoint=False)
    assert [(run.run_number, run.scan_metadata["source_bank"]) for run in found] == [(43124, 3)]
    assert len(view.due_submissions()) == 1


def test_manual_profile_without_conditions_still_freezes_the_input(monkeypatch, tmp_path):
    path = source(tmp_path)
    original = path.read_bytes()
    view = controller(monkeypatch, tmp_path, instrument_profile_dataset_id="manual-profile")
    run = view.discover(checkpoint=False)[0]
    record = view.launch_submission(run)
    from pathlib import Path
    staged = Path(record.inputs.data_path)
    assert staged != path and staged.read_bytes() == original
    path.write_bytes(original.replace(b"7.25", b"9.75"))
    assert staged.read_bytes() == original


def test_finite_bank_selection_stops_reading_sample_metadata_once_satisfied(monkeypatch, tmp_path):
    for number in range(43100, 43126):
        source(tmp_path, run=number, banks=(3,) if number == 43124 else (2,),
            wavelength=2.665 if number == 43124 else 1.5)
    view = controller(monkeypatch, tmp_path, selected_banks=(3,),
        selected_sample_ids=("sample",), initial_backfill_limit=1)
    read = []
    class CountingIndex(SampleIndex):
        def identity(self, run):
            read.append(run.run_number)
            return super().identity(run)
    view.sample_index = CountingIndex()
    assert view.discover(checkpoint=False)[0].run_number == 43124
    assert read == [43125, 43124]
