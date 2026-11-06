"""Do not publish late, substituted or corrupted preparation calculations."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from gpx_recipe.preparation.operations import create_library, stage_entry, publish_entry, pending_entries
from gpx_recipe.preparation.library import import_library
from gpx_recipe.symmetry.compiler import file_sha


def fixture(root):
    root.mkdir()
    cif = root/'parent.cif';cif.write_text('data_public\n')
    profile = root/'bank.instprm';profile.write_text('#GSAS-II profile\n')
    settings = dict(candidate=dict(id='mag:test', magnetic_space_group='P1',
        bns_number='1.1', transformation_matrix=np.eye(3).tolist(), origin_shift=[0, 0, 0]),
        k=['0', '0', '0'], sites=[dict(label='Fe1', form_factor='Fe+3', lande_g=2.)],
        maximum_moment=6., refine_histogram_scale=True, refine_background=True)
    library = create_library(root/'library', name='Public test', source=cif,
        banks={'bank 1': dict(instrument=profile,
            grid=dict(kind='PNT', lower=3000., upper=40000., bins=701, bank=1))},
        models=[dict(kind='magnetic', bank='bank 1', settings=settings)])
    identifier = library.document['entries'][0]['id']
    attempt = library.start(identifier)
    request = stage_entry(library, identifier, attempt, root/'attempt')
    return library, identifier, attempt, request


def returned(request):
    output = Path(request['destination']);output.mkdir()
    np.savez_compressed(output/'response.npz', x=np.arange(4., dtype=float))
    manifest = dict(schema='radar-experiment-preparation-entry/v1', kind='magnetic',
        grid=request['grid'], settings=request['settings'],
        input_sha256=[file_sha(request[k]) for k in ('source', 'instrument')],
        observations_included=False, scientific_result=False,
        artifacts={'response.npz': file_sha(output/'response.npz')})
    (output/'entry.json').write_text(json.dumps(manifest))
    return dict(schema='radar-experiment-preparation-worker/v1', status='prepared', kind='magnetic',
        output=str(output), observations_included=False, scientific_result=False,
        artifacts=['response.npz', 'entry.json'], entry_sha256=file_sha(output/'entry.json'))


def test_stage_resume_and_roundtrip_preserve_exact_scope(tmp_path):
    library, identifier, attempt, request = fixture(tmp_path/'input')
    assert stage_entry(library, identifier, attempt, tmp_path/'input/attempt') == request
    result = returned(request)
    coverage = publish_entry(library, identifier, attempt, request, result)
    assert coverage['all_prepared'] and coverage['total'] == 1
    assert all('/attempt-1/' in f['path'] for f in library.row(identifier)['files'])
    library.export(tmp_path/'library.zip')
    restored = import_library(tmp_path/'library.zip', tmp_path/'restored')
    assert restored.document == library.document
    assert restored.coverage()['all_prepared']


@pytest.mark.parametrize('changed', ['input', 'settings', 'grid', 'artifact', 'manifest', 'gpx', 'result'])
def test_substituted_preparation_is_not_counted_as_ready(tmp_path, changed):
    library, identifier, attempt, request = fixture(tmp_path/'input')
    request = copy.deepcopy(request)
    if changed == 'input':
        Path(request['source']).write_text('data_different_parent\n')
    if changed == 'settings':
        request['settings']['maximum_moment'] = 2.
    if changed == 'grid':
        request['grid']['bank'] = 2
    result = returned(request)
    output = Path(request['destination'])
    if changed == 'artifact':
        np.savez_compressed(output/'response.npz', x=np.arange(7., dtype=float))
    if changed == 'manifest':
        (output/'entry.json').write_text('{}')
    if changed == 'gpx':
        result['artifacts'].append('measurement.gpx')
    if changed == 'result':
        result['scientific_result'] = True
    with pytest.raises(ValueError):
        publish_entry(library, identifier, attempt, request, result)
    assert library.coverage()['counts']['preparing'] == 1
    assert not library.row(identifier)['files']


def test_late_attempt_cannot_publish_after_a_verified_terminal_retry(tmp_path):
    library, identifier, attempt, request = fixture(tmp_path/'input')
    result = returned(request)
    library.finish(identifier, attempt, state='failed', reason='Owned job is terminal in this test')
    library.retry(identifier)
    newer = library.start(identifier)
    with pytest.raises(ValueError, match='another attempt'):
        publish_entry(library, identifier, attempt, request, result)
    assert library.row(identifier)['attempt'] == newer
    assert not (library.root/'entries').exists()


def test_changed_staging_file_is_not_overwritten_on_recovery(tmp_path):
    library, identifier, attempt, request = fixture(tmp_path/'input')
    Path(request['instrument']).write_text('different profile')
    with pytest.raises(ValueError, match='attempt input changed'):
        stage_entry(library, identifier, attempt, tmp_path/'input/attempt')


def test_repeated_admission_covers_families_before_deepening_one(tmp_path):
    from gpx_recipe.preparation.library import PreparationLibrary
    cif = tmp_path/'parent.cif';cif.write_text('data_public\n')
    models = []
    for k in ['0', '1/2', '1/3']:
        for i in range(4):
            models.append(dict(kind='magnetic', bank='bank1',
                settings=dict(k=[k, '0', '0'], candidate=dict(id=f'{k}:{i}', irrep=None))))
    library = PreparationLibrary.create(tmp_path/'library', name='Coverage',
        specification={'schema': 'radar-experiment-preparation-plan/v1', 'source': 'inputs/parent.cif'},
        models=models, inputs={'parent.cif': cif})
    admitted = []
    for i in range(9):
        identifier = pending_entries(library)[0]
        attempt = library.start(identifier)
        row = library.row(identifier)
        admitted.append(row['model']['settings']['k'][0])
        # Both active and terminal unsupported work count as admitted. The
        # remaining denominator stays intact after each scheduling pass.
        if i % 2:
            library.finish(identifier, attempt, state='unsupported', reason='Retained test coverage')
        if (i+1) % 3 == 0:
            assert all(admitted.count(k) == (i+1)//3 for k in ['0', '1/2', '1/3'])
    assert len(set(pending_entries(library))) == 3
    assert library.coverage()['total'] == 12
