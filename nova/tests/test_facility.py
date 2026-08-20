from __future__ import annotations

import json
from pathlib import Path

import pytest

from radar_pd_nova.facility import (
    FacilityBrowser,
    FacilityPathError,
    WatchRecipe,
    build_facility_export_path,
)


@pytest.fixture()
def facility(tmp_path: Path) -> tuple[FacilityBrowser, Path]:
    shared = tmp_path / "HB2A" / "IPTS-12345" / "shared"
    (shared / "reduced" / "series-a").mkdir(parents=True)
    (shared / "structures").mkdir()
    (shared / "profiles").mkdir()
    (shared / "results").mkdir()
    (shared / "reduced" / "series-a" / "scan_001.dat").write_text("1 2\n", encoding="utf-8")
    (shared / "reduced" / "series-a" / "notes.md").write_text("not data", encoding="utf-8")
    (shared / "structures" / "main.cif").write_text("data_main\n", encoding="utf-8")
    (shared / "profiles" / "HB2A.instprm").write_text("# profile\n", encoding="utf-8")
    (shared / "radar-pd-config.yaml").write_text("mode: rapid\n", encoding="utf-8")
    experiment = tmp_path / "HB2A" / "IPTS-12345" / "exp510" / "Datafiles"
    experiment.mkdir(parents=True)
    (experiment / "HB2A_exp0510_scan0001.dat").write_text("1 2\n", encoding="utf-8")
    return FacilityBrowser(tmp_path), shared


def test_lists_progressive_facility_tree(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, _ = facility
    (browser.root / "groups" / "shared").mkdir(parents=True)
    (browser.root / "software" / "IPTS-not-a-number").mkdir(parents=True)
    assert browser.list_instruments() == [{"title": "HB2A", "value": "HB2A"}]
    assert browser.list_ipts("HB2A") == [{"title": "IPTS-12345", "value": "IPTS-12345"}]
    entries = browser.list_directory("HB2A", "12345", "shared/reduced/series-a", role="data")
    assert [entry.name for entry in entries] == ["scan_001.dat"]
    assert entries[0].as_item()["detail"]


def test_ipts_suggestions_are_recent_and_bounded(tmp_path: Path) -> None:
    instrument = tmp_path / "SNAP"
    for number in (10, 300, 20):
        (instrument / f"IPTS-{number}").mkdir(parents=True)

    browser = FacilityBrowser(tmp_path)

    assert browser.list_ipts("SNAP", limit=2) == [
        {"title": "IPTS-300", "value": "IPTS-300"},
        {"title": "IPTS-20", "value": "IPTS-20"},
    ]


def test_skips_facility_entries_with_inaccessible_metadata(
    facility: tuple[FacilityBrowser, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    browser, _ = facility
    blocked = browser.root / "groups"
    blocked.mkdir()
    original_is_dir = Path.is_dir

    def guarded_is_dir(path: Path) -> bool:
        if path == blocked:
            raise PermissionError("facility metadata is restricted")
        return original_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", guarded_is_dir)
    assert browser.list_instruments() == [{"title": "HB2A", "value": "HB2A"}]


def test_browses_any_readable_folder_inside_selected_ipts(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, _ = facility
    entries = browser.list_directory("HB2A", "12345", "exp510/Datafiles", role="data")
    assert [entry.relative_path for entry in entries] == ["exp510/Datafiles/HB2A_exp0510_scan0001.dat"]
    assert browser.parent_directory("exp510/Datafiles") == "exp510"
    assert browser.parent_directory("exp510") == "."


def test_ipts_root_marker_is_a_valid_browse_directory(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, _ = facility

    entries = browser.list_directory("HB2A", "IPTS-12345", ".", role="data")

    assert {entry.name for entry in entries if entry.kind == "directory"} >= {"exp510", "shared"}


def test_selects_role_checked_file_with_provenance(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, _ = facility
    selection = browser.select_file(
        "HB2A",
        "IPTS-12345",
        "shared/reduced/series-a/scan_001.dat",
        role="data",
        checksum=True,
    )
    assert selection.relative_path == "shared/reduced/series-a/scan_001.dat"
    assert selection.facility == "SNS"
    assert selection.facility_root == str(browser.root)
    assert selection.sha256
    with pytest.raises(FacilityPathError):
        browser.select_file("HB2A", "IPTS-12345", "shared/structures/main.cif", role="data")


@pytest.mark.parametrize(
    "path",
    ["../IPTS-99999/shared", "shared/../../etc", "/etc/passwd", "nexus/run.nxs"],
)
def test_rejects_paths_outside_selected_ipts(facility: tuple[FacilityBrowser, Path], path: str) -> None:
    browser, _ = facility
    with pytest.raises(FacilityPathError):
        browser.resolve_directory("HB2A", "IPTS-12345", path)


def test_rejects_symlink_escape(facility: tuple[FacilityBrowser, Path], tmp_path: Path) -> None:
    browser, shared = facility
    outside = tmp_path / "outside"
    outside.mkdir()
    link = shared / "outside-link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is not available")
    with pytest.raises(FacilityPathError):
        browser.resolve_directory("HB2A", "IPTS-12345", "shared/outside-link")


def test_supports_managed_instrument_symlink_but_rejects_child_escape(tmp_path: Path) -> None:
    facility_root = tmp_path / "SNS"
    facility_root.mkdir()
    physical_instrument = tmp_path / "experiment-storage" / "SNAP"
    data_directory = physical_instrument / "IPTS-33088" / "shared" / "reduced"
    data_directory.mkdir(parents=True)
    (data_directory / "scan.gsa").write_text("BANK 1\n", encoding="utf-8")
    try:
        (facility_root / "SNAP").symlink_to(physical_instrument, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is not available")

    browser = FacilityBrowser(facility_root)

    assert browser.list_instruments() == [{"title": "SNAP", "value": "SNAP"}]
    assert [entry.name for entry in browser.list_directory("SNAP", "33088", "shared/reduced")] == [
        "scan.gsa"
    ]

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.dat").write_text("not experiment data\n", encoding="utf-8")
    try:
        (data_directory / "outside-link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is not available")
    with pytest.raises(FacilityPathError):
        browser.resolve_directory("SNAP", "33088", "shared/reduced/outside-link")


def test_creates_output_directory_without_overwrite(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, shared = facility
    relative = browser.create_directory("HB2A", "IPTS-12345", "shared/results", "scan 001")
    assert relative == "shared/results/scan 001"
    assert (shared / "results" / "scan 001").is_dir()
    with pytest.raises(FileExistsError):
        browser.create_directory("HB2A", "IPTS-12345", "shared/results", "scan 001")


def test_creates_named_output_parent_below_working_folder(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, shared = facility
    relative = browser.ensure_output_parent("HB2A", "IPTS-12345", "shared/results", "radar-pd-results")
    assert relative == "shared/results/radar-pd-results"
    assert (shared / "results" / "radar-pd-results").is_dir()


def test_supports_hfir_root_and_facility_metadata(tmp_path: Path) -> None:
    pattern = tmp_path / "HB2A" / "IPTS-16534" / "exp510" / "Datafiles" / "scan.dat"
    pattern.parent.mkdir(parents=True)
    pattern.write_text("1 2\n", encoding="utf-8")
    browser = FacilityBrowser(tmp_path, facility="HFIR")
    selection = browser.select_file("HB2A", "16534", "exp510/Datafiles/scan.dat", role="data")
    assert selection.facility == "HFIR"
    assert selection.absolute_path == str(pattern)


def test_builds_confined_authenticated_export_paths() -> None:
    assert build_facility_export_path(
        "/SNS",
        "NOMAD",
        "IPTS-33088",
        "shared/Lalit_radarpd",
        "results",
        "scan_001-job1234",
    ) == "/SNS/NOMAD/IPTS-33088/shared/Lalit_radarpd/results/scan_001-job1234/results.zip"
    assert build_facility_export_path(
        "/HFIR",
        "HB2A",
        "28749",
        "shared/Lalit_radarpd",
        "RADAR-PD Results",
        "scan 0003-job5678",
    ) == "/HFIR/HB2A/IPTS-28749/shared/Lalit_radarpd/RADAR-PD Results/scan 0003-job5678/results.zip"


@pytest.mark.parametrize(
    ("root", "working"),
    [
        ("/tmp/SNS", "shared/results"),
        ("/SNS", "../shared/results"),
        ("/SNS", "raw/NeXus"),
    ],
)
def test_authenticated_export_path_rejects_unmanaged_destinations(root: str, working: str) -> None:
    with pytest.raises(FacilityPathError):
        build_facility_export_path(root, "HB2A", "IPTS-12345", working, "results", "run-1")


def test_publishes_result_atomically_with_manifest(facility: tuple[FacilityBrowser, Path], tmp_path: Path) -> None:
    browser, shared = facility
    source = tmp_path / "job"
    (source / "plots").mkdir(parents=True)
    (source / "summary.json").write_text("{}\n", encoding="utf-8")
    (source / "plots" / "fit.png").write_bytes(b"png")
    destination = browser.publish_directory(source, "HB2A", "IPTS-12345", "shared/results", "scan_001")
    assert destination == shared / "results" / "scan_001"
    manifest = json.loads((destination / "published_manifest.json").read_text(encoding="utf-8"))
    assert "summary.json" in manifest["files"]
    assert "plots/fit.png" in manifest["files"]
    with pytest.raises(FileExistsError):
        browser.publish_directory(source, "HB2A", "IPTS-12345", "shared/results", "scan_001")


def test_validates_writes_and_discovers_watch_recipe(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, shared = facility
    recipe = WatchRecipe(
        instrument="HB2A",
        ipts="IPTS-12345",
        source_directory="shared/reduced/series-a",
        output_directory="shared/results",
        configuration="shared/radar-pd-config.yaml",
        instrument_profile="shared/profiles/HB2A.instprm",
        main_cif="shared/structures/main.cif",
    )
    recipe_path = browser.write_watch_recipe(recipe)
    assert recipe_path == shared / "reduced" / "series-a" / "radar-pd-watch.json"
    loaded = json.loads(recipe_path.read_text(encoding="utf-8"))
    assert loaded["$schema"] == "radar-pd-watch/v1"
    candidates = browser.discover_watch_candidates(recipe)
    assert [candidate.relative_path for candidate in candidates] == ["shared/reduced/series-a/scan_001.dat"]
    assert candidates[0].fingerprint


def test_watch_output_must_differ_from_source(facility: tuple[FacilityBrowser, Path]) -> None:
    browser, _ = facility
    recipe = WatchRecipe(
        instrument="HB2A",
        ipts="IPTS-12345",
        source_directory="shared/reduced/series-a",
        output_directory="shared/reduced/series-a",
        configuration="shared/radar-pd-config.yaml",
        instrument_profile="shared/profiles/HB2A.instprm",
    )
    with pytest.raises(ValueError, match="must differ"):
        recipe.validate(browser)
