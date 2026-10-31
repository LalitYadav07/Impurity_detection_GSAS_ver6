"""Exercise both bounded readers in the published Linux image."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import h5py
import numpy as np

from radar_pd_nova import powgen_metadata


def verify():
    with TemporaryDirectory(prefix='radar-identity-smoke-') as directory:
        path = Path(directory) / 'NOM_42.nxs.h5'
        identifier = 9007199254740993
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('entry/sample/id', data=np.int64(identifier))
            handle.create_dataset('entry/sample/name', data=['α sample'], dtype=h5py.string_dtype())
            handle.create_dataset('entry/run_number', data=np.int64(42))
            handle.create_dataset('entry/experiment_identifier', data=[b'IPTS-1'])
            handle.create_dataset('entry/instrument/name', data=[b'NOMAD'])
            handle.create_dataset('entry/events/event_id', shape=(100_000_000,), dtype='i', chunks=True)
        def read(backend):
            return powgen_metadata.read_powgen_sample_identity('IPTS-1', 42,
                nexus_path=path, instrument='NOM', backend=backend)
        native = read('h5py')
        assert native == read('h5dump')
        assert native['sample_id'] == str(identifier) and native['sample_name'] == 'α sample'
        getitem = h5py.Dataset.__getitem__
        def guarded(dataset, key):
            assert not dataset.name.startswith('/entry/events/')
            return getitem(dataset, key)
        with patch.object(h5py.Dataset, '__getitem__', guarded), patch.object(
                powgen_metadata.subprocess, 'run', side_effect=AssertionError('Auto reader spawned a CLI process')):
            assert read('auto') == native
        with h5py.File(path, 'r+') as handle:
            handle.create_dataset('entry/DASlogs/SampleId/value',
                data=np.asarray([identifier, identifier + 1, identifier], dtype=np.int64))
        for backend in ['h5py', 'h5dump']:
            conflict = read(backend)
            assert conflict['identity_status'] == 'conflict' and not conflict['sample_id']
        with h5py.File(path, 'r+') as handle:
            del handle['entry/DASlogs/SampleId/value']
            handle.create_dataset('entry/DASlogs/SampleId/value', data=np.ones(1001, dtype=np.int64))
        for backend in ['h5py', 'h5dump']:
            try:
                read(backend)
            except ValueError as error:
                assert 'bounded identity limit' in str(error)
            else:
                raise AssertionError(backend + ' accepted an oversized sample-identity log')
    print('Packaged native/CLI identity equivalence, precision, Unicode, event exclusion and conflict/bounds checks passed')


if __name__ == '__main__':
    verify()
