"""Real worker operations start with a CIF, never a measured GPX."""
import json
from pathlib import Path

import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from test_preparation_theory_native import inputs
from gpx_recipe.preparation.worker import execute
from gpx_recipe.preparation.magnetic import load
from gpx_recipe.symmetry.compiler import file_sha


def request(root, kind):
    root.mkdir()
    _, protocol, candidate, _ = inputs(root, kind)
    grid = dict(kind=kind, lower=12. if kind == 'PNC' else 3000.,
        upper=130. if kind == 'PNC' else 40000., bins=701, bank=1)
    payload = dict(source=str(root/'parent.cif'), instrument=str(root/'bank.instprm'),
        grid=grid, settings={}, destination=str(root/'worker'))
    return payload, protocol, candidate


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_reference_worker_returns_pure_parent_and_k_scope(tmp_path, kind):
    payload, _, _ = request(tmp_path/'input', kind)
    payload['settings'] = dict(maximum_denominator=8)
    result = execute('reference', payload)
    manifest = json.loads((Path(result['output'])/'entry.json').read_text())
    assert result['status'] == 'prepared'
    assert result['artifacts'] == ['entry.json']
    assert result['observations_included'] is False and result['scientific_result'] is False
    assert manifest['detail']['parent']['space_group'] == 'P 1'
    assert manifest['detail']['points']
    assert not manifest['artifacts']


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_magnetic_worker_prepares_native_response_without_a_measurement(tmp_path, kind):
    payload, protocol, candidate = request(tmp_path/'input', kind)
    before = [file_sha(payload[k]) for k in ('source', 'instrument')]
    payload['settings'] = dict(candidate=candidate, k=['0', '0', '0'],
        sites=protocol['sites'], maximum_moment=6., refine_histogram_scale=True,
        refine_background=True)
    result = execute('magnetic', payload)
    output = Path(result['output'])
    assert result['artifacts'] == ['response.npz', 'scattering.npz', 'entry.json']
    manifest = json.loads((output/'entry.json').read_text())
    assert manifest['detail']['operator_audit']['passed']
    assert manifest['detail']['periodicity']['passed']
    assert manifest['detail']['scattering']['status']=='prepared'
    response = load(output/'response.npz', physics_sha256=manifest['detail']['physics_sha256'])
    assert response._native_seal is None  # Import never trusts an old native qualification.
    assert response.quadratic.shape[1] == 3 and np.any(response.quadratic)
    assert not manifest['observations_included'] and not manifest['scientific_result']
    assert [file_sha(payload[k]) for k in ('source', 'instrument')] == before
    project = G.G2Project(gpxfile=str(output/'native-reference/reference.gpx'))
    assert not np.any(project.histograms()[0].data['data'][1][1])


def test_structural_worker_preserves_full_displacements_and_independent_cells(tmp_path):
    from test_native_profile_capture import native_source
    source, _ = native_source(tmp_path/'source', 'PNC')
    project = G.G2Project(gpxfile=str(source))
    cif = tmp_path/'parent.cif';project.phase(0).export_CIF(str(cif))
    child = Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    payload = dict(source=str(cif), instrument=str(tmp_path/'source/profile.instprm'),
        child=str(child), grid=dict(kind='PNC', lower=12., upper=125., bins=701, bank=1),
        settings=dict(candidate=dict(id='structural:test',
            transformation_matrix=[[-1., 0, 0], [0, -1., 0], [0, 0, 1.]], origin_shift=[0, 0, 0]),
            mode_radius=.05, cell_radius=.005), destination=str(tmp_path/'prepared'))
    result = execute('structural', payload)
    manifest = json.loads((Path(result['output'])/'entry.json').read_text())
    detail = manifest['detail']
    assert detail['mode_dimensions'] == 2 and detail['cell_dimensions'] == 3
    assert detail['starting_curves'] > 100
    assert detail['proposed_starts'] == 185
    assert set(result['artifacts']) == {'starts.npz', 'starts.json', 'entry.json'}
    assert not result['scientific_result'] and not result['observations_included']
    assert all(not name.endswith('.gpx') for name in result['artifacts'])


def test_galaxy_worker_roundtrip_accepts_cif_and_bank_without_a_gpx(tmp_path):
    import subprocess
    import sys
    from gpx_recipe.execution import write_json
    from gpx_recipe.symmetry.bundle import pack, unpack
    from gpx_recipe.symmetry.compiler import package_signature
    payload, _, _ = request(tmp_path/'captured', 'PNC')
    root = tmp_path/'captured'
    payload.update(source='parent.cif', instrument='bank.instprm',
        settings={'maximum_denominator': 8}, destination='compiled')
    assets = ['parent.cif', 'bank.instprm']
    pack(root, tmp_path/'inputs.zip', assets)
    message = dict(schema='radar-model-native/v1', operation='preparation-reference', payload=payload,
        implementation_sha256=package_signature(), expected_source_sha256=file_sha(root/'parent.cif'),
        dispatch_key='f'*24, timeout=120, input_hashes={name: file_sha(root/name) for name in assets})
    write_json(tmp_path/'request.json', message)
    script = Path(__file__).resolve().parents[2]/'scripts/ndip_model_campaign.py'
    with (tmp_path/'galaxy-wrapper.log').open('w') as stream:
        subprocess.run([sys.executable, str(script), 'native', str(root/'parent.cif'),
            str(tmp_path/'inputs.zip'), str(tmp_path/'request.json'), str(tmp_path/'job'),
            str(tmp_path/'results.zip'), '--preparation-certificate',str(tmp_path/'certificate.json')], check=True, timeout=150, stdout=stream,
            stderr=subprocess.STDOUT)
    received = unpack(tmp_path/'results.zip', tmp_path/'received')
    response = json.loads((received/'response.json').read_text())
    assert response['source_sha256'] == message['expected_source_sha256']
    assert response['result']['status'] == 'prepared'
    entry = json.loads((received/'native/compiled/entry.json').read_text())
    assert entry['kind'] == 'reference' and entry['observations_included'] is False
    assert entry['scientific_result'] is False
    assert not (tmp_path/'job/verified_projects').exists()
    from gpx_recipe.preparation.certificates import read
    assert read(tmp_path/'certificate.json')['entries']==[]


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_prepared_magnetic_library_reconstructs_without_the_original_gpx_ids(tmp_path, kind):
    from gpx_recipe.preparation.operations import create_library, stage_entry, publish_entry
    from gpx_recipe.preparation.library import import_library
    from gpx_recipe.preparation.native_magnetic import physics_identity, qualify_response, _physics_record
    from gpx_recipe.live.native_profile import NativeReducedProfile
    from gpx_recipe.symmetry.native_magnetic import construct_theory
    from gpx_recipe.symmetry.worker import Artifacts
    payload, protocol, candidate = request(tmp_path/'original', kind)
    settings = dict(candidate=candidate, k=['0', '0', '0'], sites=protocol['sites'],
        maximum_moment=6., refine_histogram_scale=True, refine_background=True)
    library = create_library(tmp_path/'library', name='Public portable physics', source=payload['source'],
        banks={'bank 1': dict(instrument=payload['instrument'], grid=payload['grid'])},
        models=[dict(kind='magnetic', bank='bank 1', settings=settings)])
    identifier = library.document['entries'][0]['id'];attempt = library.start(identifier)
    staged = stage_entry(library, identifier, attempt, tmp_path/'attempt')
    result = execute('magnetic', staged)
    publish_entry(library, identifier, attempt, staged, result)
    library.export(tmp_path/'portable.zip')
    restored = import_library(tmp_path/'portable.zip', tmp_path/'restored')
    assets = {Path(a['path']).name: restored.root/a['path'] for a in restored.row(identifier)['files']}
    manifest = json.loads(assets['entry.json'].read_text())
    from gpx_recipe.preparation.consume import restore_magnetic_entry
    response,cache,imported=restore_magnetic_entry(assets['entry.json'].parent)
    assert imported['scattering_available'] and imported['native_qualification_required']
    assert not cache.lookup(manifest['detail']['scattering']['identity'])[1]
    first = NativeReducedProfile(Path(result['output'])/'native-model/magnetic-refinement.gpx')
    first_record = _physics_record(first)
    # A new reference and audited constructor produce new native random IDs.
    _, new_protocol, new_candidate = request(tmp_path/'reconstructed', kind)
    output = tmp_path/'reconstructed-model';output.mkdir()
    built = construct_theory(tmp_path/'reconstructed/reference/reference.gpx', Artifacts(output),
        new_protocol, new_candidate, dict(axis='x', magnitude=1., independent_mode_seed=17,
            parent_direction=None, site_signs={}))
    native = NativeReducedProfile(output/built['gpx'])
    (tmp_path/'first-magnetic.json').write_text(json.dumps(first_record, indent=2))
    (tmp_path/'second-magnetic.json').write_text(json.dumps(_physics_record(native), indent=2))
    assert physics_identity(native) == response.physics_sha256
    validation=qualify_response(response,native,factor_cache=cache)
    assert max(validation['independent_magnetic_errors']) < 1e-8
    proof=validation['coefficient_validation']
    assert proof['scattering_factors_reused'] and not proof['scattering_qualification_reused']
    assert proof['native_structure_factor_calls']==19
    assert response._native_seal is not None
