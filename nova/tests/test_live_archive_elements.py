"""Archive chemistry comes from complete mapped CIF loops and stays source-bound."""
import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace
import zipfile

import pytest

from radar_pd_nova.app import RadarPdNovaApp
from radar_pd_nova.library_setup import (
    activation_for_review, inspect_library_review, inspect_library_review_bytes,
)
from radar_pd_nova.uploads import cif_element_suggestion, store_browser_upload


def source_cif(symbols=('D', 'O')):
    rows = '\n'.join(f'site{i} {symbol} 0 0 0' for i, symbol in enumerate(symbols))
    return ("data_candidate\n_chemical_formula_sum 'Au99'\n"
            "_cell_length_a 4\n_cell_length_b 4\n_cell_length_c 4\n"
            "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
            "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
            "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n" + rows + '\n').encode()


def library(tmp_path, *, raw=None, mapping=None, extras=None, occupancy=False):
    raw = source_cif() if raw is None else raw
    manifest = {'n_phases': 1, 'phase_ids': ['phase'], 'elements': ['Xe'], **(extras or {})}
    if occupancy:
        review = {'schema': 'radar-cif-occupancy-review/v1', 'reviewed': True,
                  'source_sha256': hashlib.sha256(raw).hexdigest(),
                  'canonical_sha256': hashlib.sha256(raw).hexdigest(),
                  'maximum_site_excess': .001, 'normalization_applied': False,
                  'review_note': 'Synthetic value-preserving review',
                  'composition': {'D': .60066, 'O': .4},
                  'affected_sites': [{'labels': ['site0', 'site1'], 'fractional_coordinates': [0, 0, 0],
                                      'total': 1.00066, 'components': {'D': .60066, 'O': .4}}]}
        review['policy_sha256'] = hashlib.sha256(json.dumps(
            review, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        manifest['occupancy_reviews'] = {'phase': review}
    path = tmp_path / 'library.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('pack/manifest.json', json.dumps(manifest))
        archive.writestr('pack/cif_map.json', mapping if isinstance(mapping, str) else
                         json.dumps({'phase': 'cifs/source.cif'} if mapping is None else mapping))
        archive.writestr('pack/cifs/source.cif', raw)
    return path


def test_archive_uses_atom_types_not_formula_manifest_or_filename(tmp_path):
    path = library(tmp_path, raw=source_cif(('D+1', 'O-2', 'Fe3+', 'Li+1')))
    original = path.read_bytes()
    preview = inspect_library_review(path)
    assert preview['element_suggestion'] == {
        'available': True, 'elements': ['D', 'Fe', 'Li', 'O'], 'source_count': 1,
        'evidence': 'mapped CIF atom-type loops',
    }
    assert preview['archive_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert inspect_library_review_bytes(path.read_bytes()) == preview
    assert path.read_bytes() == original


def test_unused_type_table_does_not_identify_actual_sites(tmp_path):
    raw = (b"data_candidate\nloop_\n_atom_type_symbol\nFe\n"
           b"loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nCu1 0 0 0\n")
    suggestion = inspect_library_review(library(tmp_path, raw=raw))['element_suggestion']
    assert suggestion['available'] is False
    assert suggestion['elements'] == []


def test_unused_type_table_cannot_add_species_to_explicit_sites(tmp_path):
    raw = source_cif(('Cu',)) + b"loop_\n_atom_type_symbol\nFe\n"
    suggestion = inspect_library_review(library(tmp_path, raw=raw))['element_suggestion']
    assert suggestion['available'] is True
    assert suggestion['elements'] == ['Cu']


def test_one_readable_site_loop_does_not_hide_an_untyped_site_loop(tmp_path):
    raw = source_cif(('O',)) + (b"loop_\n_atom_site_label\n_atom_site_fract_x\n"
                                  b"_atom_site_fract_y\n_atom_site_fract_z\nCu1 0 0 0\n")
    suggestion = inspect_library_review(library(tmp_path, raw=raw))['element_suggestion']
    assert suggestion['available'] is False
    assert suggestion['elements'] == []


@pytest.mark.parametrize('raw', [
    b"data_bad\n_chemical_formula_sum 'Fe O'\n",
    source_cif(('D', 'Xx')),
    source_cif(('D', '?')),
    source_cif().replace(b'site1 O 0 0 0', b'site1 O 0'),
])
def test_incomplete_source_does_not_make_a_partial_element_filter(tmp_path, raw):
    result = inspect_library_review(library(tmp_path, raw=raw))['element_suggestion']
    assert not result['available'] and result['elements'] == []


@pytest.mark.parametrize('mapping', [
    {'wrong_phase': 'cifs/source.cif'}, {}, {'phase': '../outside.cif'},
    {'phase': '/outside.cif'}, {'phase': 'cifs/../../outside.cif'},
    {'phase': r'C:\outside.cif'}, {'phase': 'cifs//source.cif'},
    {'phase': 'cifs/missing.cif'}, {'phase': 'cifs/source.txt'},
    '{"phase":"cifs/source.cif","phase":"cifs/source.cif"}',
])
def test_bad_source_mapping_is_unavailable(tmp_path, mapping):
    result = inspect_library_review(library(tmp_path, mapping=mapping))['element_suggestion']
    assert not result['available'] and result['elements'] == []


def test_cif_size_bound_is_checked_before_parsing(tmp_path):
    result = inspect_library_review(library(tmp_path, raw=b'x' * 2_000_001))['element_suggestion']
    assert not result['available'] and 'limit' in result['reason']


def test_symlink_member_is_not_a_source(tmp_path):
    path = library(tmp_path, mapping={'phase': 'cifs/link.cif'})
    with zipfile.ZipFile(path, 'a') as archive:
        member = zipfile.ZipInfo('pack/cifs/link.cif')
        member.external_attr = 0o120777 << 16
        archive.writestr(member, 'source.cif')
    assert not inspect_library_review(path)['element_suggestion']['available']


def test_failed_sources_do_not_silently_narrow_suggestion(tmp_path):
    path = library(tmp_path, extras={'failures': [{'name': 'missing.cif', 'reason': 'Unreadable source'}]})
    assert not inspect_library_review(path)['element_suggestion']['available']


def test_strict_archive_loop_mode_does_not_change_legacy_formula_suggestion():
    text = "data_formula\n_chemical_formula_sum 'Fe O'\n"
    assert cif_element_suggestion(text) == ['Fe', 'O']
    with pytest.raises(ValueError, match='atom-type loop'):
        cif_element_suggestion(text, require_atom_types=True)


@pytest.fixture
def app(tmp_path):
    result = RadarPdNovaApp()
    state = result.server.state
    state.live_database_source = 'archive'
    state.live_library_archive_source = 'galaxy'
    state.powgen_database_dataset_id = 'selected-archive'
    state.live_sample_elements = 'C'
    state.live_environment_elements = 'Al'
    state.live_library_builder_elements = ['Xe']  # Old builder source must not leak in.
    state.flush()
    return result


def test_existing_reviewed_archive_button_uses_server_inspection(app, tmp_path):
    preview = inspect_library_review(library(tmp_path, occupancy=True))
    state = app.server.state
    state.live_library_review = activation_for_review(preview)
    state.live_library_review_preview = {**preview, 'element_suggestion': {'available': True, 'elements': ['Xe']}}
    state.flush()
    calls = []
    def inspect(identifier):
        calls.append(identifier)
        return preview
    app.service = SimpleNamespace(inspect_library_review=inspect)
    asyncio.run(app.live_assistant_suggest_elements())
    assert calls == ['selected-archive']
    assert state.live_sample_elements == 'D, O' and state.live_environment_elements == 'Al'
    assert state.live_library_builder_elements == ['Xe']
    assert not state.live_assistant_error and not state.live_assistant_busy
    assert state.live_library_review == activation_for_review(preview)


def test_occupancy_permission_is_still_required(app, tmp_path):
    preview = inspect_library_review(library(tmp_path, occupancy=True))
    app.service = SimpleNamespace(inspect_library_review=lambda _: preview)
    asyncio.run(app.live_assistant_suggest_elements())
    assert app.server.state.live_sample_elements == 'C'
    assert 'Review the selected' in app.server.state.live_assistant_error


def test_unreadable_archive_never_falls_back_to_stale_builder_elements(app, tmp_path):
    preview = inspect_library_review(library(tmp_path, raw=b'data_empty\n'))
    app.service = SimpleNamespace(inspect_library_review=lambda _: preview)
    asyncio.run(app.live_assistant_suggest_elements())
    assert app.server.state.live_sample_elements == 'C'
    assert 'no complete CIF element suggestion' in app.server.state.live_assistant_error


@pytest.mark.parametrize('change', ['source', 'chemistry', 'monitoring', 'reset', 'connection'])
def test_late_suggestion_cannot_overwrite_changed_context(app, tmp_path, change):
    preview = inspect_library_review(library(tmp_path))
    entered, release = threading.Event(), threading.Event()
    def inspect(_):
        entered.set()
        assert release.wait(3)
        return preview
    app.service = SimpleNamespace(inspect_library_review=inspect)
    async def exercise():
        task = asyncio.create_task(app.live_assistant_suggest_elements())
        assert await asyncio.to_thread(entered.wait, 2)
        state = app.server.state
        if change == 'source':
            state.powgen_database_dataset_id = 'other-archive'
        elif change == 'chemistry':
            state.live_sample_elements = 'Na, Cl'
        elif change == 'monitoring':
            state.powgen_monitoring = True
        elif change == 'connection':
            app.service = SimpleNamespace(inspect_library_review=lambda _: preview)
        else:
            app._reset_live_assistant()
        state.flush()
        release.set()
        await task
    asyncio.run(exercise())
    assert app.server.state.live_sample_elements == ('Na, Cl' if change == 'chemistry' else 'C')
    assert app.server.state.live_environment_elements == 'Al'
    assert not app.server.state.live_assistant_busy


def test_registered_local_archive_uses_same_source_validation(app, tmp_path):
    path = library(tmp_path)
    state = app.server.state
    state.live_library_archive_source = 'computer'
    state.live_database_archive_path = str(store_browser_upload(path.read_bytes(), 'library.zip'))
    state.flush()
    asyncio.run(app.live_assistant_suggest_elements())
    assert state.live_sample_elements == 'D, O' and not state.live_assistant_error


def test_client_path_is_not_authority_to_read_a_local_archive(app, tmp_path):
    state = app.server.state
    state.live_library_archive_source = 'computer'
    state.live_database_archive_path = str(library(tmp_path))
    state.flush()
    asyncio.run(app.live_assistant_suggest_elements())
    assert state.live_sample_elements == 'C'
    assert 'Upload the requested file' in state.live_assistant_error


def test_saved_configuration_remains_unchanged(app):
    app.server.state.live_configuration_source = 'history'
    app.service = SimpleNamespace(inspect_library_review=lambda _: pytest.fail('Must not read archive'))
    asyncio.run(app.live_assistant_suggest_elements())
    assert app.server.state.live_sample_elements == 'C'
    assert 'Edit a copy' in app.server.state.live_assistant_error


def test_one_bad_member_prevents_a_partial_union(tmp_path):
    path = tmp_path / 'mixed.zip'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('pack/manifest.json', json.dumps({'n_phases': 2, 'phase_ids': ['a', 'b']}))
        archive.writestr('pack/cif_map.json', json.dumps({'a': 'a.cif', 'b': 'b.cif'}))
        archive.writestr('pack/a.cif', source_cif(('Fe',)))
        archive.writestr('pack/b.cif', source_cif(('O', '?')))
    result = inspect_library_review(path)['element_suggestion']
    assert not result['available'] and result['elements'] == []


def test_async_button_releases_event_and_refuses_duplicate_inspections(app, tmp_path):
    preview = inspect_library_review(library(tmp_path))
    entered, release = threading.Event(), threading.Event()
    calls = []
    def inspect(identifier):
        calls.append(identifier)
        entered.set()
        assert release.wait(3)
        return preview
    app.service = SimpleNamespace(inspect_library_review=inspect)
    async def exercise():
        assert app.start_live_assistant_suggest_elements() is None
        task = app._live_assistant_action_task
        app.start_live_assistant_suggest_elements()
        assert app._live_assistant_action_task is task
        assert await asyncio.to_thread(entered.wait, 2)
        assert app.server.state.live_assistant_busy
        release.set()
        await task
    asyncio.run(exercise())
    assert calls == ['selected-archive']
    assert app.server.state.live_sample_elements == 'D, O'
