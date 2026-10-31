from pathlib import Path

import h5py
import numpy as np

from radar_pd_nova.powgen_metadata import read_powgen_scan_metadata
from radar_pd_nova.powgen_watch import WatchedRun


def test_h5dump_numeric_snapshot_requests_roundtrip_precision(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from radar_pd_nova import powgen_metadata

    monkeypatch.setattr(powgen_metadata.shutil, "which", lambda _: "/usr/bin/h5dump")
    def run(command, **kwargs):
        assert command[command.index("-m") + 1] == "%.17g"
        return SimpleNamespace(returncode=0, stdout='DATASPACE SIMPLE { ( 1 ) / ( 1 ) }\nDATA { 29.334920630000001 }')
    monkeypatch.setattr(powgen_metadata.subprocess, "run", run)
    snapshot = powgen_metadata._h5dump_snapshot(tmp_path / "test.h5", "/entry/log")
    assert snapshot.values == (29.334920630000001,)


def test_full_metadata_prefers_bounded_native_reads_even_when_cli_is_installed(tmp_path,monkeypatch):
    from radar_pd_nova import powgen_metadata
    path=tmp_path/'raw.h5'
    with h5py.File(path,'w') as handle:
        handle.create_dataset('/entry/title',data=[b'Native metadata replay'])
        value=handle.create_dataset('/entry/DASlogs/BL11A:SE:SampleTemp/average_value',data=[143.3])
        value.attrs['units']='K'
        handle.create_dataset('/entry/bank1_events/event_id',shape=(10_000_000,),dtype='i8')
    monkeypatch.setattr(powgen_metadata.shutil,'which',lambda _: '/usr/bin/h5dump')
    def no_cli(*args,**kwargs):raise AssertionError('Installed h5py must avoid metadata subprocesses')
    monkeypatch.setattr(powgen_metadata.subprocess,'run',no_cli)
    result=read_powgen_scan_metadata('IPTS-1',42,nexus_path=path)
    assert result['run_title']=='Native metadata replay'
    assert result['temperature']['value']==143.3


def test_reader_fallback_is_only_for_unavailable_h5py_not_missing_data(tmp_path,monkeypatch):
    from radar_pd_nova import powgen_metadata
    calls=[]
    monkeypatch.setattr(powgen_metadata,'_h5dump_snapshot',lambda path,key: calls.append(key) or powgen_metadata._Snapshot(key,text='CLI'))
    monkeypatch.setattr(powgen_metadata,'_h5py_available',lambda:False)
    assert powgen_metadata._first_snapshot(tmp_path/'raw.h5',['/entry/title']).text=='CLI'
    assert calls==['/entry/title']
    monkeypatch.setattr(powgen_metadata,'_h5py_available',lambda:True)
    monkeypatch.setattr(powgen_metadata,'_h5py_snapshot',lambda *args:None)
    assert powgen_metadata._first_snapshot(tmp_path/'raw.h5',['/entry/title']) is None
    assert calls==['/entry/title']


def test_reads_bounded_powgen_nexus_conditions(tmp_path: Path) -> None:
    nexus = tmp_path / "PG3_63716.nxs.h5"
    with h5py.File(nexus, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("title", data=np.asarray([b"heating scan"]))
        entry.create_dataset("start_time", data=np.asarray([b"2026-08-21T10:00:00-04:00"]))
        entry.create_dataset("end_time", data=np.asarray([b"2026-08-21T10:30:00-04:00"]))
        sample = entry.create_group("sample")
        sample.create_dataset("identifier", data=np.asarray([b"118950"]))
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
    assert "magnetic_field" not in metadata
    assert metadata["magnetic_field_observation"]["value"] == 2.0
    assert metadata["wavelength"]["value"] == 1.5
    assert metadata["duration_seconds"] == 1800.0
    assert metadata["sample_formula"] == "YFeSiGa"
    assert metadata["sample_id"] == "118950"


def test_explicit_celsius_temperature_is_converted_to_kelvin(tmp_path: Path) -> None:
    nexus = tmp_path / "PG3_63717.nxs.h5"
    with h5py.File(nexus, "w") as handle:
        logs = handle.create_group("entry").create_group("DASlogs")
        temperature = logs.create_group("BL11A:SE:SampleTemp")
        value = temperature.create_dataset("value", data=np.asarray([24.0, 26.0]))
        value.attrs["units"] = "degC"

    metadata = read_powgen_scan_metadata("IPTS-38000", 63717, nexus_path=nexus)

    assert metadata["temperature"]["value"] == 298.15
    assert metadata["temperature"]["minimum"] == 297.15
    assert metadata["temperature"]["maximum"] == 299.15
    assert metadata["temperature"]["unit"] == "K"
    assert metadata["temperature"]["source_unit"] == "degC"
    assert metadata["temperature"]["source_value"] == 25.0


def test_watch_run_round_trip_preserves_scan_metadata() -> None:
    run = WatchedRun(
        run_number=63716,
        source_path="/SNS/PG3/IPTS-38000/shared/autoreduce/PG3_63716.gsa",
        fingerprint="abc",
        scan_metadata={"temperature": {"value": 1050.0, "unit": "K"}},
    )

    restored = WatchedRun.from_dict(run.as_dict())

    assert restored.scan_metadata == {"temperature": {"value": 1050.0, "unit": "K"}}


def test_summary_mean_and_separate_chopper_and_beam_rates(tmp_path):
    path=tmp_path/'metadata.h5'
    with h5py.File(path,'w')as h:
        for name,v in [('BL11A:SE:SampleTemp/average_value',143.3),('BL11A:SE:SampleTemp/minimum_value',134.),
            ('BL11A:SE:SampleTemp/maximum_value',152.)]:
            d=h.create_dataset('entry/DASlogs/'+name,data=[v]);d.attrs['units']='K'
        h.create_dataset('entry/DASlogs/BL11A:Chop:Skf1:SpeedUserReq/value',data=[10.])
        h.create_dataset('entry/DASlogs/Speed1/average_value',data=[9.9993])
        h.create_dataset('entry/DASlogs/frequency/average_value',data=[59.9957])
    result=read_powgen_scan_metadata('IPTS-1',42,nexus_path=path)
    assert result['temperature']['value']==143.3 and result['temperature']['statistic']=='mean'
    assert result['temperature']['quality']=='changing'
    assert result['chopper_frequency_request']['value']==10
    assert result['beam_frequency']['value']>59
