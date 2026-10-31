from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from radar_pd_nova import powgen_metadata, powgen_samples
from radar_pd_nova.powgen_samples import PowgenSampleIndex
from radar_pd_nova.powgen_watch import WatchedRun


def make_nexus(path, sample_id, name):
    with h5py.File(path, "w") as handle:
        group = handle.create_group("entry/sample")
        group.create_dataset("identifier", data=np.asarray([str(sample_id).encode()]))
        group.create_dataset("name", data=np.asarray([name.encode()]))
        # The identity reader must not visit or load event datasets.
        handle.create_dataset("entry/events/event_id", shape=(100_000_000,), dtype="i", chunks=True)


@pytest.mark.parametrize("backend", ["h5dump", "h5py"])
def test_identity_reader_handles_missing_aliases_and_unicode(tmp_path, monkeypatch, backend):
    path = tmp_path / "run.nxs.h5"
    make_nexus(path, "00123", "α sample")
    monkeypatch.setattr(powgen_metadata, "powgen_nexus_path", lambda *_: path)
    if backend == "h5dump" and not powgen_metadata.shutil.which("h5dump"):
        pytest.skip("native HDF5 tool unavailable")
    result = powgen_metadata.read_powgen_sample_identity("IPTS-1", 1, backend=backend)
    assert result["sample_id"] == "00123"
    assert result["sample_name"] == "α sample"


def test_inventory_groups_names_caches_and_refreshes_changed_files(tmp_path, monkeypatch):
    files = [tmp_path / f"PG3_{n}.nxs.h5" for n in range(1, 5)]
    make_nexus(files[0], "A", "Copper")
    make_nexus(files[1], "A", "Copper warm")
    make_nexus(files[2], "B", "Nickel")
    monkeypatch.setattr(powgen_samples, "powgen_nexus_path", lambda _, n: files[n-1])
    monkeypatch.setattr(powgen_metadata, "powgen_nexus_path", lambda _, n: files[n-1])
    calls = []
    real = powgen_samples.read_powgen_sample_identity
    def read(ipts, number):
        calls.append(number)
        return real(ipts, number)
    monkeypatch.setattr(powgen_samples, "read_powgen_sample_identity", read)
    runs = [WatchedRun(n, f"PG3_{n}.gsa", str(n)) for n in range(1, 5)]
    index = PowgenSampleIndex("IPTS-1")
    result = index.inventory(runs)
    assert result["unidentified_count"] == 1
    assert [o["value"] for o in result["options"]] == ["A", "B"]
    assert "Copper / Copper warm" in result["options"][0]["title"]
    assert result["options"][0]["scan_count"] == 2
    index.inventory(runs)
    assert sorted(calls) == [1, 2, 3]
    make_nexus(files[3], "C", "New sample")
    assert [o["value"] for o in index.inventory(runs)["options"]] == ["A", "B", "C"]
    make_nexus(files[0], "D", "Replacement sample")
    assert index.identity(runs[0])["sample_id"] == "D"


def test_missing_identity_is_not_guessed_from_name(tmp_path, monkeypatch):
    path = tmp_path / "run.nxs.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("entry/sample/name", data=[b"123"])
    monkeypatch.setattr(powgen_samples, "powgen_nexus_path", lambda *_: path)
    monkeypatch.setattr(powgen_metadata, "powgen_nexus_path", lambda *_: path)
    result = PowgenSampleIndex("IPTS-1").inventory([WatchedRun(1, "PG3_1.gsa", "f")])
    assert result["options"] == []
    assert result["unidentified_count"] == 1


@pytest.mark.parametrize("backend", ["h5dump", "h5py"])
def test_numeric_sample_id_does_not_lose_integer_precision(tmp_path, monkeypatch, backend):
    path = tmp_path / "run.nxs.h5"
    with h5py.File(path, "w") as h:
        h.create_dataset("entry/sample/id", data=np.int64(9007199254740993))
    monkeypatch.setattr(powgen_metadata, "powgen_nexus_path", lambda *_: path)
    if backend == "h5dump" and not powgen_metadata.shutil.which("h5dump"):
        pytest.skip("native HDF5 tool unavailable")
    assert powgen_metadata.read_powgen_sample_identity("IPTS-1", 1, backend=backend)["sample_id"] == "9007199254740993"


def test_identity_change_inside_run_is_not_hidden_by_matching_endpoints(tmp_path,monkeypatch):
    path=tmp_path/'run.h5';make_nexus(path,'123','sample')
    with h5py.File(path,'r+')as h:
        h.create_dataset('entry/DASlogs/SampleId/value',data=[123,124,123])
    monkeypatch.setattr(powgen_metadata.shutil,'which',lambda _:None)
    result=powgen_metadata.read_powgen_sample_identity('IPTS-1',1,nexus_path=path)
    assert result['sample_id']=='' and result['identity_status']=='conflict'


def test_discovery_prefers_one_native_file_open_without_reading_events(tmp_path, monkeypatch):
    path = tmp_path / 'run.h5'
    make_nexus(path, '00123', 'α sample')
    monkeypatch.setattr(powgen_metadata.shutil, 'which', lambda _: '/available/h5dump')
    monkeypatch.setattr(powgen_metadata.subprocess, 'run', lambda *a, **k: pytest.fail('Auto discovery started a CLI reader'))
    original = h5py.Dataset.__getitem__
    reads = []
    def guarded(dataset, key):
        assert not dataset.name.startswith('/entry/events/')
        reads.append(dataset.name)
        return original(dataset, key)
    monkeypatch.setattr(h5py.Dataset, '__getitem__', guarded)
    result = powgen_metadata.read_powgen_sample_identity('IPTS-1', 1, nexus_path=path)
    assert result['sample_id'] == '00123' and result['sample_name'] == 'α sample'
    assert set(reads) == {'/entry/sample/identifier', '/entry/sample/name'}


def test_missing_native_reader_uses_the_shape_checked_cli_fallback(tmp_path, monkeypatch):
    import sys
    path = tmp_path / 'run.h5'
    path.write_bytes(b'CLI fixture')
    monkeypatch.setitem(sys.modules, 'h5py', None)
    monkeypatch.setattr(powgen_metadata.shutil, 'which', lambda _: '/available/h5dump')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        assert kwargs['timeout'] == 20
        body = 'DATASPACE SIMPLE { ( 1 ) / ( 1 ) }' if '-H' in command else 'DATA { "A" }'
        return SimpleNamespace(returncode=0, stdout='DATASET "/entry/sample/identifier" { ' + body + ' }')
    monkeypatch.setattr(powgen_metadata.subprocess, 'run', run)
    assert powgen_metadata.read_powgen_sample_identity('IPTS-1', 1, nexus_path=path)['sample_id'] == 'A'
    assert len(commands) == 2 and '-H' in commands[0] and '-H' not in commands[1]


@pytest.mark.parametrize('field,value,error', [
    ('entry/run_number', 2, 'run identifier'),
    ('entry/instrument/name', b'NOMAD', 'instrument'),
    ('entry/DASlogs/SampleId/value', np.ones(1001, dtype=int), 'bounded identity limit'),
])
def test_invalid_source_is_not_retried_through_another_reader(tmp_path, monkeypatch, field, value, error):
    path = tmp_path / 'run.h5'
    make_nexus(path, '123', 'sample')
    with h5py.File(path, 'r+') as h:
        h.create_dataset(field, data=value)
    monkeypatch.setattr(powgen_metadata.shutil, 'which', lambda _: '/available/h5dump')
    monkeypatch.setattr(powgen_metadata.subprocess, 'run', lambda *a, **k: pytest.fail('Invalid metadata was retried through CLI'))
    with pytest.raises(ValueError, match=error):
        powgen_metadata.read_powgen_sample_identity('IPTS-1', 1, nexus_path=path)
