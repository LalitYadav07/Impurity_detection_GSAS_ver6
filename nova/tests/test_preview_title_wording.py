"""Display-only wording never asserts that an unread title is absent at source."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from radar_pd_nova import powgen_samples
from radar_pd_nova.powgen_samples import PowgenSampleIndex, include_run
from radar_pd_nova.powgen_watch import WatchedRun
from radar_pd_nova.snap_metadata import SnapSampleIndex


@pytest.mark.parametrize('cls', [PowgenSampleIndex, SnapSampleIndex])
@pytest.mark.parametrize('catalog', [False, True])
@pytest.mark.parametrize('title', ['', 'Sample run'])
def test_preview_title_wording_preserves_identity_and_raw_title(tmp_path, monkeypatch, cls, catalog, title):
    metadata = dict(sample_id='007', sample_name='Example', identity_status='resolved', run_title=title)
    original = deepcopy(metadata)
    instrument = 'SNAP' if cls is SnapSampleIndex else 'PG3'
    index_metadata = SimpleNamespace(instrument=instrument, notices=[], revision='r', generation=1,
        refresh=lambda **_: None, enrich=lambda _run, raw, **_: dict(raw))
    index = cls('IPTS-1', metadata_index=index_metadata)
    def identity(run):
        index._cache[run.run_number] = ((), dict(metadata))
        return dict(metadata)
    index.identity = identity
    monkeypatch.setattr(powgen_samples, 'load_sample_catalog', lambda *_: {42: dict(metadata)})
    monkeypatch.setattr(powgen_samples, 'catalog_created_at', lambda *_: 123.)
    data = tmp_path / 'pattern.gsa'
    data.write_text(('# ' + title + '\n') if title else 'No retained title header\n')
    run = WatchedRun(42, str(data), 'fingerprint', instrument=instrument)

    result = index.inventory([run], catalog=catalog)
    row = result['run_options'][0]
    assert row['title'] == f'{run.run_id} — unclassified: {title or "Title unavailable in preview"}'
    assert row['run_title'] == title and row['run_role'] == 'unclassified'
    assert row['value'] == 42 and row['sample_id'] == '007'
    assert metadata == original
    assert include_run(run)
