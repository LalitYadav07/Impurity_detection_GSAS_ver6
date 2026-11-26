from types import SimpleNamespace

import pytest

from radar_pd_nova.galaxy_service import GalaxyService


def test_completed_file_uses_advertised_history_route(monkeypatch, tmp_path):
    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith('/api/datasets/result'):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'id': 'result',
                'download_url': '/api/histories/other-history/contents/result/display',
            })
        assert url == 'https://galaxy.example/api/histories/other-history/contents/result/display'
        return SimpleNamespace(raise_for_status=lambda: None,
                               iter_content=lambda **_: iter([b'complete']), close=lambda: None)

    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get', get)
    service = GalaxyService('https://galaxy.example', 'secret', 'current-history', output_root=tmp_path)
    target = tmp_path / 'result.zip'
    service._download_dataset('result', target, maximum_bytes=8)
    assert target.read_bytes() == b'complete'
    assert len(calls) == 2
    assert calls[1][1]['stream'] is True
    assert calls[1][1]['params'] == {'raw': 'true'}
    assert calls[0][1]['headers'] == calls[1][1]['headers']


@pytest.mark.parametrize('advertised', [
    '/galaxy/api/histories/h1/contents/result/display',
    'https://galaxy.example/galaxy/api/histories/h1/contents/result/display',
])
def test_same_origin_subpath_and_other_history_are_supported(tmp_path, advertised):
    service = GalaxyService('https://galaxy.example/galaxy', 'secret', 'h2', output_root=tmp_path)
    service._dataset_metadata = lambda _: {'id': 'result', 'download_url': advertised}
    assert service._dataset_download_url('result') == 'https://galaxy.example/galaxy/api/histories/h1/contents/result/display'


@pytest.mark.parametrize('advertised', [
    'https://elsewhere.example/api/histories/h1/contents/result/display',
    '//elsewhere.example/api/histories/h1/contents/result/display',
    'http://galaxy.example/api/histories/h1/contents/result/display',
    '/api/histories/h1/contents/different/display',
    '/api/histories/h1/contents/result/display?api_key=unexpected',
    '/api/histories/h1/contents/result/display#fragment',
    '/api/histories/h1/contents/result/../../display',
    17,
])
def test_unsafe_or_wrong_dataset_route_never_downloads(monkeypatch, tmp_path, advertised):
    service = GalaxyService('https://galaxy.example', 'secret', 'h1', output_root=tmp_path)
    service._dataset_metadata = lambda _: {'id': 'result', 'download_url': advertised}
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get',
                        lambda *_, **__: pytest.fail('An invalid route must not make a data request'))
    with pytest.raises(ValueError):
        service._download_dataset('result', tmp_path / 'result.zip')
    assert not (tmp_path / 'result.zip').exists()


def test_legacy_metadata_without_download_url_retains_raw_endpoint(tmp_path):
    service = GalaxyService('https://galaxy.example', 'secret', 'h1', output_root=tmp_path)
    service._dataset_metadata = lambda _: {'id': 'result'}
    assert service._dataset_download_url('result') == 'https://galaxy.example/api/datasets/result/display'


def test_mismatched_metadata_cannot_redirect_requested_file(tmp_path):
    service = GalaxyService('https://galaxy.example', 'secret', 'h1', output_root=tmp_path)
    service._dataset_metadata = lambda _: {'id': 'different'}
    with pytest.raises(ValueError, match='different dataset'):
        service._dataset_download_url('result')


def test_collection_metadata_reuses_its_history_route_without_another_lookup(monkeypatch, tmp_path):
    service = GalaxyService('https://galaxy.example/galaxy', 'secret', 'current-history', output_root=tmp_path)
    service._dataset_metadata = lambda _: pytest.fail('Collection metadata already identifies this HDA')
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url.endswith('/api/dataset_collections/collection'):
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                'collection_type': 'list', 'populated_state': 'ok', 'element_count': 1,
                'elements': [{'element_type': 'hda', 'element_identifier': 'final', 'object': {
                    'id': 'result', 'history_id': 'original-history', 'file_ext': 'gpx',
                    'state': 'ok', 'file_size': 8,
                }}],
            })
        assert url == 'https://galaxy.example/galaxy/api/histories/original-history/contents/result/display'
        return SimpleNamespace(raise_for_status=lambda: None,
                               iter_content=lambda **_: iter([b'complete']), close=lambda: None)

    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get', get)
    service._download_collection('collection', tmp_path / 'gpx_projects')
    assert (tmp_path / 'gpx_projects/final.gpx').read_bytes() == b'complete'
    assert len(calls) == 2


@pytest.mark.parametrize('metadata', [
    {'id': 'different', 'history_id': 'h1'},
    {'id': 'result', 'history_id': '../other'},
    {'id': 'result', 'history_id': '//elsewhere.example'},
    {'id': 'result', 'history_id': 'h1?api_key=unexpected'},
    {'id': 'result', 'history_id': 17},
    {'id': 'result', 'download_url': 'https://elsewhere.example/api/datasets/result/display'},
])
def test_reused_metadata_retains_identity_and_origin_validation(monkeypatch, tmp_path, metadata):
    service = GalaxyService('https://galaxy.example', 'secret', 'h1', output_root=tmp_path)
    service._dataset_metadata = lambda _: pytest.fail('A supplied invalid identity must not trigger a lookup')
    monkeypatch.setattr('radar_pd_nova.galaxy_service.requests.get',
                        lambda *_, **__: pytest.fail('Invalid metadata must not make a data request'))
    with pytest.raises(ValueError):
        service._download_dataset('result', tmp_path / 'result.zip', metadata=metadata)
    assert not (tmp_path / 'result.zip').exists()
