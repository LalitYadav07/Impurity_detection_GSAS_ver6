"""Executable IPTS/NOVA smoke checks that do not require pytest."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="radar-pd-ipts-ui-") as temporary:
        root = Path(temporary)
        shared = root / "HB2A" / "IPTS-123" / "shared"
        (shared / "reduced").mkdir(parents=True)
        (shared / "results").mkdir()
        data = shared / "reduced" / "scan.dat"
        profile = shared / "reduced" / "HB2A.instprm"
        cif = shared / "reduced" / "main.cif"
        data.write_text("1 2\n", encoding="utf-8")
        profile.write_text("profile\n", encoding="utf-8")
        cif.write_text("data_main\n", encoding="utf-8")
        blocked = root / "groups"
        (blocked / "shared").mkdir(parents=True)
        (root / "software" / "IPTS-not-a-number").mkdir(parents=True)
        for number in (7, 999):
            (root / "HB2A" / f"IPTS-{number}").mkdir(parents=True)
        os.environ["RADAR_PD_FACILITY_ROOT"] = str(root)

        from radar_pd_nova.app import RadarPdNovaApp
        from radar_pd_nova.facility import FacilityBrowser, FacilityPathError

        original_is_dir = Path.is_dir

        def guarded_is_dir(path: Path) -> bool:
            if path == blocked:
                raise PermissionError("facility metadata is restricted")
            return original_is_dir(path)

        Path.is_dir = guarded_is_dir
        try:
            app = RadarPdNovaApp()
        finally:
            Path.is_dir = original_is_dir
        state = app.server.state
        template = app.layout.html
        assert state.facility_available is True
        assert state.facility_instruments == [{"title": "HB2A", "value": "HB2A"}]
        assert app.facility.list_ipts("HB2A", limit=2) == [
            {"title": "IPTS-999", "value": "IPTS-999"},
            {"title": "IPTS-123", "value": "IPTS-123"},
        ]
        assert any(option["value"] == "ipts_browser" for option in state.source_options)
        for text in (
            "Use an SNS/HFIR experiment working folder",
            "Experiment working folder",
            "Current working folder",
            "Instrument profile source",
            "Diffraction data in working folder",
            "Instrument profile in working folder",
            "Known/main-phase CIF in working folder",
            "Export completed results into this working folder",
            "A uniquely named results ZIP will be written directly into the current working folder.",
        ):
            assert text in template, text
        assert "Main-phase CIF from current IPTS auxiliary folder" not in template

        state.facility_instrument = "HB2A"
        state.facility_ipts = "IPTS-123"
        state.use_facility_workspace = True
        state.facility_working_directory = "shared"
        app.refresh_facility_browser()
        app.open_facility_working_directory("shared/reduced")
        assert state.facility_working_directory == "shared/reduced"
        state.input_source = "ipts_browser"
        state.instrument_source = "upload"
        state.facility_data_path = str(data)
        state.facility_data_relative_path = "shared/reduced/scan.dat"
        state.instrument_path = "/tmp/uploaded.instprm"
        selection = app._inputs()
        assert selection.data_relative_path == "shared/reduced/scan.dat"
        assert selection.instrument_source == "upload"
        assert selection.instrument_path == "/tmp/uploaded.instprm"

        state.instrument_source = "ipts"
        state.instrument_path = ""
        state.facility_instrument_path = str(profile)
        state.facility_instrument_relative_path = "shared/reduced/HB2A.instprm"
        state.main_cif_source = "ipts"
        state.facility_main_cif_path = str(cif)
        state.facility_main_cif_relative_path = "shared/reduced/main.cif"
        state.publish_results_to_ipts = True
        state.facility_output_subfolder = "radar-pd-results"
        selection = app._inputs()
        assert selection.instrument_relative_path == "shared/reduced/HB2A.instprm"
        assert selection.main_cif_relative_path == "shared/reduced/main.cif"
        assert selection.publish_directory == "shared/reduced"
        assert selection.publish_subfolder is None

        symlink_root = root / "SNS"
        symlink_root.mkdir()
        physical_ipts = root / "experiment-storage" / "SNAP" / "IPTS-33088"
        physical_shared = physical_ipts / "shared"
        physical_shared.mkdir(parents=True)
        (physical_shared / "scan.gsa").write_text("BANK 1\n", encoding="utf-8")
        (symlink_root / "SNAP").symlink_to(physical_ipts.parent, target_is_directory=True)
        symlink_browser = FacilityBrowser(symlink_root)
        assert symlink_browser.list_directory("SNAP", "33088", "shared")[0].name == "scan.gsa"

        outside = root / "outside"
        outside.mkdir()
        (physical_shared / "outside-link").symlink_to(outside, target_is_directory=True)
        try:
            symlink_browser.resolve_directory("SNAP", "33088", "shared/outside-link")
        except FacilityPathError:
            pass
        else:
            raise AssertionError("A child symlink escaped the selected IPTS")

    print("IPTS NOVA smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
