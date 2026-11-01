from pathlib import Path
from types import SimpleNamespace

import pytest

from radar_pd_nova.facility import FacilityBrowser, FacilityPathError
from radar_pd_nova.webmon.session import WebmonSession
from radar_pd_nova.webmon.ui import WebmonMixin


RAW = b"# X , Y , E Distribution=true\n0\n1,10,2\n2,20,3\n3,30,4\n"


def make_source(tmp_path: Path, managed: bool):
    root = tmp_path / "SNS"
    root.mkdir()
    instrument = tmp_path / "managed-storage" / "PG3" if managed else root / "PG3"
    scope = instrument / "IPTS-2767"
    folder = scope / "shared" / "livereduce"
    folder.mkdir(parents=True)
    summary = scope / "shared" / "autoreduce" / "PG3_IPTS-2767_runsummary.csv"
    summary.parent.mkdir()
    summary.write_text("RUN,SAMPLE,ITEMS-ID,TEMP,WAVELENGTH\n7,selected specimen,18062,300,0.8\n")
    source = folder / "PG3_7.dat"
    source.write_bytes(RAW)
    if managed:
        (root / "PG3").symlink_to(instrument, target_is_directory=True)
    session = WebmonSession(dict(facility_root=str(root), archive=str(tmp_path / "archive")))
    return session, root, scope, folder, summary, source


@pytest.mark.parametrize("managed", [False, True])
def test_same_ipts_inventory_and_capture_use_facility_browser_boundary(tmp_path, managed):
    session, root, scope, _, _, source = make_source(tmp_path, managed)
    expected = FacilityBrowser(root).resolve_directory("PG3", "2767")
    assert session.scope("IPTS-2767") == expected == scope.resolve()
    rows = session.runs("2767")
    assert [(row["run"], row["sample_id"]) for row in rows] == [("7", "18062")]
    row, receipt, pattern = session.capture("2767", "7")
    assert Path(row["path"]).resolve() == source.resolve()
    assert receipt["content_sha256"] and pattern.y.tolist() == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("ipts", ["../2768", "2767/../2768", "2767\\..\\2768", "/2767", "IPTS-../2768", "2767/child"])
def test_malicious_ipts_is_rejected_before_source_discovery(tmp_path, ipts):
    session, *_ = make_source(tmp_path, True)
    with pytest.raises(ValueError):
        session.runs(ipts)


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("sibling", [False, True])
@pytest.mark.parametrize("member", ["livereduce", "summary"])
def test_source_descendants_cannot_escape_selected_ipts(tmp_path, managed, sibling, member):
    session, _, scope, folder, summary, source = make_source(tmp_path, managed)
    outside = scope.parent / "IPTS-2768" if sibling else tmp_path / "unrelated"
    outside.mkdir(parents=True)
    if member == "livereduce":
        source.unlink()
        folder.rmdir()
        (outside / "PG3_9.dat").write_bytes(RAW)
        folder.symlink_to(outside, target_is_directory=True)
    else:
        target = outside / "other.csv"
        target.write_text("RUN,SAMPLE\n7,other experiment\n")
        summary.unlink()
        summary.symlink_to(target)
    with pytest.raises(FacilityPathError, match="selected IPTS"):
        session.runs("2767")


@pytest.mark.parametrize("sibling", [False, True])
def test_individual_source_symlink_outside_selected_ipts_is_not_discovered(tmp_path, sibling):
    session, _, scope, _, _, source = make_source(tmp_path, False)
    outside = scope.parent / "IPTS-2768" if sibling else tmp_path / "unrelated"
    outside.mkdir(parents=True)
    target = outside / "source.dat"
    target.write_bytes(RAW)
    source.unlink()
    source.symlink_to(target)
    assert session.runs("2767") == []
    with pytest.raises(ValueError, match="selected IPTS"):
        session.capture("2767", "7")


def test_cached_source_is_revalidated_before_capture(tmp_path, monkeypatch):
    session, _, scope, _, _, source = make_source(tmp_path, False)
    inside = scope / 'shared' / 'source'
    inside.mkdir()
    (inside / 'source.dat').write_bytes(RAW)
    current = scope / 'shared' / 'current'
    current.symlink_to(inside, target_is_directory=True)
    source.unlink()
    source.symlink_to(current / 'source.dat')
    assert session.runs("2767")[0]["run"] == "7"
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'source.dat').write_bytes(RAW)
    # Repointing this ancestor does not change the cached inventory folder's
    # mtime. Capture must validate the source rather than trusting that cache.
    current.unlink()
    current.symlink_to(outside, target_is_directory=True)
    opened = []
    monkeypatch.setattr(session.archive, "capture", lambda *a, **kw: opened.append(a))
    with pytest.raises(FacilityPathError, match="selected IPTS"):
        session.capture("2767", "7")
    assert opened == []


def test_source_symlink_inside_selected_ipts_remains_readable(tmp_path):
    session, _, scope, _, _, source = make_source(tmp_path, True)
    target = scope / "shared" / "PG3_7.dat"
    target.write_bytes(RAW)
    source.unlink()
    source.symlink_to(target)
    _, _, pattern = session.capture("2767", "7")
    assert pattern.y.tolist() == [10.0, 20.0, 30.0]


def test_scope_failure_has_a_user_facing_context_message():
    def reject(_):
        raise FacilityPathError("Only paths inside the selected IPTS are permitted")

    state = SimpleNamespace(live_beamline="PG3", powgen_ipts="IPTS-2767", webmon_has_result=False,
                            webmon_accepted=True, webmon_context_ready=True, flush=lambda: None)
    app = WebmonMixin()
    app.server = SimpleNamespace(state=state)
    app._webmon = SimpleNamespace(config={}, runs=reject)
    app._webmon_start_scope = "IPTS-2767"
    app.webmon_refresh()
    assert "selected experiment" in state.webmon_message
    assert "selected IPTS" not in state.webmon_message
    assert state.webmon_connection == "Source unavailable — check selected experiment"
    assert not state.webmon_accepted and not state.webmon_context_ready
