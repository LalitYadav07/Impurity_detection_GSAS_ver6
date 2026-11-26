"""A measured-pair library asks for its reference before native dispatch."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.preparation.structural_attention_weights import FIXED_REFERENCE_KIND, REFERENCE_KIND
from radar_pd_nova import prepared_structural_discovery as bridge
from radar_pd_nova.prepared_reference import requires_measured_reference
from radar_pd_nova.preparation_controller import PreparationMixin
from test_preparation_structural_ranking_hints import prepared
from test_prepared_structural_discovery import setup


def library_with_reference_kinds(tmp_path, kinds):
    original, _ = prepared(tmp_path / 'original', three_curve=True)
    training = next(r for r in original.document['entries']
                    if r['model']['kind'] == 'structural-training')
    name = next(a['path'] for a in training['files'] if Path(a['path']).name == 'model.json')
    metadata = json.loads(original.verify_assets([name])[name].read_text())
    entries = []
    files = {}
    for index, kind in enumerate(kinds):
        path = tmp_path / f'case-{index}' / 'model.json'
        path.parent.mkdir()
        document = copy.deepcopy(metadata)
        if kind is not None:
            document['reference_kind'] = kind
        path.write_text(json.dumps(document))
        files[str(path)] = path
        entries.append(dict(id=str(index), state='prepared', model=dict(kind='structural-training'),
                            files=[dict(path=str(path), sha256=file_sha(path))]))
    calls = []

    def verify(names):
        calls.append(tuple(names))
        assert all(Path(n).name == 'model.json' for n in names), 'Never load weights or observations'
        for n in names:
            expected = next(a['sha256'] for r in entries for a in r['files'] if a['path'] == n)
            if file_sha(files[n]) != expected:
                raise ValueError('Changed metadata')
        return {n: files[n] for n in names}

    return SimpleNamespace(document=dict(id='library', entries=entries), verify_assets=verify), calls


@pytest.mark.parametrize('kinds,required', [
    ([FIXED_REFERENCE_KIND], True),
    ([FIXED_REFERENCE_KIND, FIXED_REFERENCE_KIND], True),
    ([REFERENCE_KIND], False),
    ([None], False),
    ([FIXED_REFERENCE_KIND, REFERENCE_KIND], False),
    ([], False),
])
def test_only_a_known_uniform_measured_pair_requirement_is_mandatory(tmp_path, kinds, required):
    library, _ = library_with_reference_kinds(tmp_path, kinds)
    assert requires_measured_reference(library) is required


def test_incomplete_metadata_is_not_presented_as_a_verified_reference_requirement(tmp_path):
    library, calls = library_with_reference_kinds(tmp_path, [FIXED_REFERENCE_KIND])
    library.document['entries'][0]['state'] = 'preparing'
    assert not requires_measured_reference(library)
    assert not calls


def test_changed_model_metadata_cannot_supply_a_reference_hint(tmp_path):
    library, _ = library_with_reference_kinds(tmp_path, [FIXED_REFERENCE_KIND])
    Path(library.document['entries'][0]['files'][0]['path']).write_text('{}')
    with pytest.raises(ValueError, match='Changed metadata'):
        requires_measured_reference(library)


def test_progress_refresh_caches_metadata_and_deselection_preserves_the_reference(tmp_path):
    library, calls = library_with_reference_kinds(tmp_path, [FIXED_REFERENCE_KIND])
    app = PreparationMixin()
    state = SimpleNamespace(sx_reference_required=False, sx_parent_reference='reviewed-warm-scan')
    app.server = SimpleNamespace(state=state)
    app._sync_prepared_reference_requirement(library)
    assert state.sx_reference_required and len(calls) == 1
    app._sync_prepared_reference_requirement(library)
    assert len(calls) == 1
    app._sync_prepared_reference_requirement()
    assert not state.sx_reference_required
    assert state.sx_parent_reference == 'reviewed-warm-scan'


def test_measured_pair_library_stops_before_a_native_job_without_reference(setup, monkeypatch):
    app, draft, library, _, calls = setup
    stage = bridge.stage_models

    def measured(*args):
        models = stage(*args)
        for model in models:
            model['reference_kind'] = FIXED_REFERENCE_KIND
        return models

    monkeypatch.setattr(bridge, 'stage_models', measured)
    with pytest.raises(bridge.PreparedLibraryUnavailable, match='No native job was submitted'):
        bridge.discover(app, draft, dict(setting_sha256='a' * 64), ['0', '0', '0'], lambda _: None)
    assert not calls


def test_supplied_reference_still_gets_native_admission(setup, monkeypatch):
    app, draft, _, _, calls = setup
    path = draft['root'] / 'reference.gpx'
    path.write_bytes(b'Checked warm-reference fixture')
    draft['reference_asset'] = dict(path=path.name, sha256=file_sha(path))
    stage = bridge.stage_models

    def measured(*args):
        models = stage(*args)
        for model in models:
            model['reference_kind'] = FIXED_REFERENCE_KIND
        return models

    monkeypatch.setattr(bridge, 'stage_models', measured)
    result = bridge.discover(app, draft, dict(setting_sha256='a' * 64), ['0', '0', '0'], lambda _: None)
    assert len(calls) == 1 and calls[0]['reference'] == str(path)
    assert result['provenance']['prepared_library']['reference_kind'] == FIXED_REFERENCE_KIND
