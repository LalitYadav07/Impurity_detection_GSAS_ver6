"""Keep portable preparation revisions separate from experimental results."""
import json
from pathlib import Path

import pytest
import numpy as np

from gpx_recipe.preparation import library as library_module
from radar_pd_nova.preparation_registry import PreparationRegistry
from gpx_recipe.preparation.operations import create_library, stage_entry, publish_entry
from gpx_recipe.symmetry.compiler import file_sha


def fixture(root):
    root.mkdir()
    parent = root/'parent.cif';parent.write_text('data_public_registry_control\n')
    profile = root/'bank.instprm';profile.write_text('# public registry profile placeholder\n')
    grid = dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1)
    settings = dict(candidate=dict(id='registry:public',magnetic_space_group='P1',
        bns_number='1.1',transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0]),
        k=['0','0','0'],sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2.)],
        maximum_moment=6.,refine_histogram_scale=True,refine_background=True)
    library = create_library(root/'library',name='Public registry control',source=parent,
        banks={'bank 1':dict(instrument=profile,grid=grid)},
        models=[dict(kind='magnetic',bank='bank 1',settings=settings)])
    identifier = library.document['entries'][0]['id'];attempt=library.start(identifier)
    request = stage_entry(library,identifier,attempt,root/'attempt')
    return library,identifier,attempt,request


def returned(request):
    root=Path(request['destination']);root.mkdir()
    np.savez(root/'response.npz',x=np.arange(4.))
    manifest=dict(schema='radar-experiment-preparation-entry/v1',kind='magnetic',
        grid=request['grid'],settings=request['settings'],
        input_sha256=[file_sha(request[k]) for k in ('source','instrument')],
        observations_included=False,scientific_result=False,
        artifacts={'response.npz':file_sha(root/'response.npz')})
    (root/'entry.json').write_text(json.dumps(manifest))
    return dict(schema='radar-experiment-preparation-worker/v1',status='prepared',kind='magnetic',
        output=str(root),observations_included=False,scientific_result=False,
        artifacts=['response.npz','entry.json'],entry_sha256=file_sha(root/'entry.json'))


def prepared(tmp_path):
    library, identifier, attempt, request = fixture(tmp_path/'input')
    publish_entry(library, identifier, attempt, request, returned(request))
    packet = tmp_path/'preparation.zip';library.export(packet)
    return library, identifier, packet


def test_import_and_selection_survive_a_new_workspace_instance(tmp_path):
    library, identifier, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    key = registry.import_archive(packet)
    reopened = PreparationRegistry(registry.root)
    assert reopened.get().document['id'] == library.document['id']
    assert reopened.get(key).coverage()['all_prepared']
    assert reopened.summaries()[0]['prepared'] == 1
    assert set(reopened.get(key).prepared_assets(identifier)) == {
        a['path'] for a in library.row(identifier)['files']}


def test_reimport_creates_a_distinct_revision_without_overwriting_progress(tmp_path):
    library, identifier, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    first = registry.import_archive(packet)
    second = registry.import_archive(packet)
    assert first != second
    assert len(registry.summaries()) == 2
    registry.select(first)
    assert PreparationRegistry(registry.root).get().root == registry.get(first).root
    assert registry.get(first).document == registry.get(second).document


def test_progress_rendering_does_not_hash_unrelated_numeric_arrays(tmp_path, monkeypatch):
    _, _, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    key = registry.import_archive(packet)
    monkeypatch.setattr(library_module, 'sha', lambda path:pytest.fail('State summaries must not consume arrays'))
    for _ in range(3):
        assert registry.summaries()[0]['prepared'] == 1
        assert registry.get(key).document['name']


def test_import_checks_each_numeric_asset_once_before_registration(tmp_path, monkeypatch):
    _, _, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    original = library_module.sha;reads=[]
    def counted(path):
        if Path(path).suffix == '.npz':reads.append(str(path))
        return original(path)
    monkeypatch.setattr(library_module, 'sha', counted)
    key=registry.import_archive(packet)
    assert registry.get(key).coverage()['all_prepared']
    assert len(reads)==1


def test_corrupted_array_is_rejected_at_consumption_after_metadata_refresh(tmp_path):
    _, identifier, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    key = registry.import_archive(packet)
    library = registry.get(key)
    path = next(p for p in library.prepared_assets(identifier).values() if p.suffix == '.npz')
    path.write_bytes(b'changed')
    assert registry.summaries()[0]['prepared'] == 1  # Historical completion, not a new native check.
    with pytest.raises(ValueError):
        registry.get(key).prepared_assets(identifier)
    with pytest.raises(ValueError):
        registry.export(key, tmp_path/'invalid.zip')


@pytest.mark.parametrize('key', ['../outside', '/outside', 'f'*32])
def test_foreign_or_escaping_selection_is_rejected(tmp_path, key):
    registry = PreparationRegistry(tmp_path/'workspace')
    with pytest.raises(ValueError):
        registry.get(key)


def test_replaced_scope_is_rejected_by_registry_identity(tmp_path):
    _, _, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    key = registry.import_archive(packet)
    library = registry.get(key)
    data = json.loads(library.path.read_text())
    data['entries'][0]['model']['grid']['bank'] = 2
    library.path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        registry.get(key)


def test_missing_revision_does_not_hide_other_preparations(tmp_path):
    _, _, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    first = registry.import_archive(packet);second = registry.import_archive(packet)
    registry.get(first).path.unlink()
    rows = {r['id']:r for r in registry.summaries()}
    assert rows[first]['available'] is False and rows[first]['total'] is None
    assert rows[second]['available'] is True and rows[second]['prepared'] == 1


def test_workspace_limit_is_checked_before_extracting_another_library(tmp_path, monkeypatch):
    from radar_pd_nova import preparation_registry as module
    _, _, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace');registry.import_archive(packet)
    monkeypatch.setattr(module, 'MAX_LIBRARIES', 1)
    monkeypatch.setattr(module, 'import_library', lambda *a:pytest.fail('Check capacity before extracting'))
    with pytest.raises(ValueError, match='limit'):
        registry.import_archive(packet)


def test_create_captures_parent_and_profile_without_a_measured_scan(tmp_path):
    library, _, _, request = fixture(tmp_path/'input')
    registry = PreparationRegistry(tmp_path/'workspace')
    key = registry.create(name='Before measurement', source=request['source'],
        banks={'bank 1':dict(instrument=request['instrument'],grid=request['grid'])},
        models=[dict(kind='magnetic',bank='bank 1',settings=request['settings'])])
    assert registry.get(key).coverage()['counts']['pending'] == 1
    assert registry.summaries()[0]['origin'] == 'created'
    assert registry.summaries()[0]['name'] == 'Before measurement'


def test_lost_index_write_acknowledgment_preserves_the_published_library(tmp_path, monkeypatch):
    from radar_pd_nova import preparation_registry as module
    _, _, packet = prepared(tmp_path)
    registry = PreparationRegistry(tmp_path/'workspace')
    original = module.write_json
    def lost(path, value):
        original(path, value)
        if Path(path) == registry.path and value.get('items'):
            raise OSError('simulated lost write acknowledgment')
    monkeypatch.setattr(module, 'write_json', lost)
    with pytest.raises(OSError, match='acknowledgment'):
        registry.import_archive(packet)
    reopened = PreparationRegistry(registry.root)
    assert len(reopened.summaries()) == 1
    assert reopened.get().coverage()['all_prepared']


def test_unchanged_summary_metadata_is_cached_but_new_progress_invalidates_it(tmp_path,monkeypatch):
    from radar_pd_nova import preparation_registry as module
    _,_,_,request=fixture(tmp_path/'input')
    registry=PreparationRegistry(tmp_path/'workspace')
    key=registry.create(name='Cached metadata',source=request['source'],
        banks={'bank 1':dict(instrument=request['instrument'],grid=request['grid'])},
        models=[dict(kind='magnetic',bank='bank 1',settings=request['settings'])])
    validate=module.validate_manifest;reads=[]
    def counted(document):reads.append('manifest');return validate(document)
    monkeypatch.setattr(module,'validate_manifest',counted)
    first=registry.summaries();first[0]['counts']['pending']=999
    assert registry.summaries()[0]['counts']['pending']==1
    assert len(reads)==1
    library=registry.get(key);library.start(library.document['entries'][0]['id'])
    assert registry.summaries()[0]['counts']['preparing']==1
    assert len(reads)==2
