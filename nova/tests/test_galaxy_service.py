from pathlib import Path

from radar_pd_nova.galaxy_service import GalaxyService, normalize_status, stage_from_console
from radar_pd_nova.models import AnalysisMode, RunRecord, RunStatus


def test_progress_stage_inference() -> None:
    assert stage_from_console("starting lattice nudge", RunStatus.RUNNING) == ("Lattice nudging", 55)
    assert stage_from_console("rerank 512 patterns", RunStatus.RUNNING) == ("Pattern scoring", 72)
    assert stage_from_console("", RunStatus.OK) == ("Results ready", 100)


def test_galaxy_status_normalization() -> None:
    assert normalize_status("queued") is RunStatus.QUEUED
    assert normalize_status("running") is RunStatus.RUNNING
    assert normalize_status("ok") is RunStatus.OK
    assert normalize_status("error") is RunStatus.ERROR
    assert normalize_status("deleted") is RunStatus.CANCELLED


def test_result_collector_uses_nova_collection_api(tmp_path: Path) -> None:
    class FakeData:
        id = "output-id"

        def download(self, target: str) -> None:
            path = Path(target)
            if path.suffix:
                path.write_text("{}", encoding="utf-8")
            else:
                path.mkdir(parents=True, exist_ok=True)
                (path / "member.txt").write_text("ok", encoding="utf-8")

    class FakeOutputs:
        def __init__(self) -> None:
            self.collection_names: list[str] = []

        def get_dataset(self, name: str) -> FakeData:
            if name != "summary":
                raise RuntimeError("not published")
            return FakeData()

        def get_collection(self, name: str) -> FakeData:
            self.collection_names.append(name)
            if name != "plots":
                raise RuntimeError("not published")
            return FakeData()

    outputs = FakeOutputs()

    class FakeTool:
        def get_results(self) -> FakeOutputs:
            return outputs

    service = GalaxyService("https://example.invalid", "key", "history", output_root=tmp_path)
    service._recover_tool = lambda uid: FakeTool()  # type: ignore[method-assign]
    record = RunRecord(uid="job-1", name="run-1", mode=AnalysisMode.RAPID, history_id="history")

    result = service.collect_results(record)

    assert outputs.collection_names == ["plots", "tables", "phases", "gpx_projects", "diagnostics"]
    assert (Path(result.output_dir) / "plots" / "member.txt").read_text(encoding="utf-8") == "ok"
