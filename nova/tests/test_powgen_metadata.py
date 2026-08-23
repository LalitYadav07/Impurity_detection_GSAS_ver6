from pathlib import Path

import h5py
import numpy as np

from radar_pd_nova.powgen_metadata import read_powgen_scan_metadata
from radar_pd_nova.powgen_watch import WatchedRun


def test_reads_bounded_powgen_nexus_conditions(tmp_path: Path) -> None:
    nexus = tmp_path / "PG3_63716.nxs.h5"
    with h5py.File(nexus, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("title", data=np.asarray([b"heating scan"]))
        entry.create_dataset("start_time", data=np.asarray([b"2026-08-21T10:00:00-04:00"]))
        entry.create_dataset("end_time", data=np.asarray([b"2026-08-21T10:30:00-04:00"]))
        sample = entry.create_group("sample")
        sample.create_dataset("name", data=np.asarray([b"Ga flux"]))
        sample.create_dataset("chemical_formula", data=np.asarray([b"YFeSiGa"]))
        logs = entry.create_group("DASlogs")
        temperature = logs.create_group("BL11A:SE:SampleTemp")
        temperature_value = temperature.create_dataset("value", data=np.asarray([299.0, 301.0, 300.0]))
        temperature_value.attrs["units"] = "K"
        field = logs.create_group("BL11A:SE:SS:Field")
        field_value = field.create_dataset("value", data=np.asarray([2.0]))
        field_value.attrs["units"] = "T"
        wavelength = logs.create_group("BL11A:Chop:Skf1:WavelengthUserReq")
        wavelength_value = wavelength.create_dataset("value", data=np.asarray([1.5]))
        wavelength_value.attrs["units"] = "A"

    metadata = read_powgen_scan_metadata("IPTS-38000", 63716, nexus_path=nexus)

    assert metadata["temperature"]["value"] == 300.0
    assert metadata["temperature"]["minimum"] == 299.0
    assert metadata["magnetic_field"]["value"] == 2.0
    assert metadata["wavelength"]["value"] == 1.5
    assert metadata["duration_seconds"] == 1800.0
    assert metadata["sample_formula"] == "YFeSiGa"


def test_watch_run_round_trip_preserves_scan_metadata() -> None:
    run = WatchedRun(
        run_number=63716,
        source_path="/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63716.gsa",
        fingerprint="abc",
        scan_metadata={"temperature": {"value": 1050.0, "unit": "K"}},
    )

    restored = WatchedRun.from_dict(run.as_dict())

    assert restored.scan_metadata == {"temperature": {"value": 1050.0, "unit": "K"}}
