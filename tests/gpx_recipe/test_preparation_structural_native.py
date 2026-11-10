"""Structural physics must be reusable without retaining the observation."""
import json
from pathlib import Path

import numpy as np
import pytest

G = pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation.structural_identity import structural_physics_identity, _physics_record


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_structural_identity_is_anchored_and_observation_independent(tmp_path, kind):
    _, source = native_source(tmp_path/'first', kind)
    before = structural_physics_identity(source)
    histogram = source.fast.native.histo_phases[0][source.fast.slow.hist.name]
    histogram['Data'][1][:] *= 3.
    histogram['Data'][2][:] *= 7.
    assert structural_physics_identity(source) == before
    point = np.zeros(len(source.radii))
    point[-source.cell_dimensions:] = source.radii[-source.cell_dimensions:]*.07
    point[0] = source.radii[0]*.06
    source.native(point)
    assert structural_physics_identity(source) == before
    first_record = _physics_record(source)
    _, second = native_source(tmp_path/'second', kind)
    second_record = _physics_record(second)
    # Independent imports produce new random native IDs and private paths.
    # They describe the same physical reference, not another observation.
    (tmp_path/'first-record.json').write_text(json.dumps(first_record, indent=2))
    (tmp_path/'second-record.json').write_text(json.dumps(second_record, indent=2))
    assert structural_physics_identity(second) == before
    native = second.fast.native
    histogram = native.histo_phases[0][second.fast.slow.hist.name]
    hfx = f":{histogram['hId']}:"
    original = list(native.controls.get(hfx+'Exclude', []))
    native.controls[hfx+'Exclude'] = [[float(histogram['Data'][0][40]), float(histogram['Data'][0][80])]]
    assert structural_physics_identity(second) != before
    native.controls[hfx+'Exclude'] = original
    native.parms[f":{histogram['hId']}:X"] += .1
    assert structural_physics_identity(second) != before


@pytest.mark.parametrize('kind', ['PNC', 'PNT'])
def test_structural_theory_forward_preserves_empty_observations(tmp_path, kind):
    from gpx_recipe.preparation.reference import create_reference
    from gpx_recipe.structural_live.native import prepare
    from gpx_recipe.structural_live.service import make_source, SearchSettings
    # Reuse only the public test's input structures and instrument. No
    # measured/source GPX is provided to create_reference.
    source_gpx, source = native_source(tmp_path/'inputs', kind)
    source_project = G.G2Project(gpxfile=str(source_gpx))
    cif = tmp_path/'parent.cif'
    source_project.phase(0).export_CIF(str(cif))
    instrument = tmp_path/'inputs/profile.instprm'
    limits = (12., 125.) if kind == 'PNC' else (3000., 40000.)
    reference = create_reference(cif, instrument, tmp_path/'reference',
        kind=kind, lower=limits[0], upper=limits[1], bins=1501)
    checksum = Path(reference['path']).read_bytes()
    child = Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    metadata = prepare(reference['path'], child, [[-1., 0, 0], [0, -1., 0], [0, 0, 1.]],
        [0, 0, 0], tmp_path/'theory-prepared')
    assert metadata['observations_included'] is False
    assert metadata['scientific_result'] is False
    prepared = make_source(tmp_path/'theory-prepared', tmp_path/'evaluation', SearchSettings())
    prediction = prepared.native(np.zeros(len(prepared.radii)))
    assert np.isfinite(prediction).all() and np.any(prediction)
    assert structural_physics_identity(prepared)
    project = G.G2Project(gpxfile=str(tmp_path/'theory-prepared/prepared.gpx'))
    assert not np.any(project.histograms()[0].data['data'][1][1])
    assert not np.any(project.data['Covariance']['data'])
    assert Path(reference['path']).read_bytes() == checksum
