"""Independent GSAS-II atom-displacement check, including orbit multiplicity."""
import copy
from pathlib import Path

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_family_rms_matches_native_asymmetric_sites_with_full_orbit_weights(tmp_path,kind):
    from test_native_profile_capture import native_source
    from gpx_recipe.preparation.reference import create_reference
    from gpx_recipe.preparation import structural_mode_projection as projection
    from gpx_recipe.preparation.structural_mode_families import FamilyProjection
    from gpx_recipe.structural_live.native import prepare,apply_modes
    from gpx_recipe.structural_live.service import make_source,SearchSettings
    from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix
    from gpx_recipe.structural_live.optimization.chemical_geometry import direct_metric_from_reciprocal
    from gpx_recipe.symmetry.compiler import file_sha
    original,_=native_source(tmp_path/'input',kind)
    parent=tmp_path/'parent.cif';G.G2Project(gpxfile=str(original)).phase(0).export_CIF(str(parent))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    request=dict(source=str(parent),instrument=str(tmp_path/'input/profile.instprm'),child=str(child),
        grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,upper=125. if kind=='PNC' else 40000.,bins=701,bank=1),
        settings=dict(candidate=dict(id='public-control',transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],
            origin_shift=[0,0,0],child=dict(path='child.cif',sha256=file_sha(child))),mode_radius=.05,cell_radius=.01))
    ref=create_reference(parent,request['instrument'],tmp_path/'reference',**request['grid'])
    prepare(ref['path'],child,request['settings']['candidate']['transformation_matrix'],[0,0,0],tmp_path/'prepared')
    source=make_source(tmp_path/'prepared',tmp_path/'source',SearchSettings())
    stamp=projection.write(source,request,tmp_path/'map');value=projection.read(tmp_path/'map',stamp=stamp)
    families=FamilyProjection(value);assert not families.opaque_native_axes
    point=np.full(source.mode_dimensions,.003)
    source.check(np.r_[point,np.zeros(source.cell_dimensions)])
    amplitudes=families.project(point[None])['amplitudes'][0]
    slow=source.fast.slow;geometry=source.fast.empirical_constraint.geometry
    phase=copy.deepcopy(slow.phase.data);cx=phase['General']['AtomPtrs'][0]
    weights=np.array([slow.s.SytSym(a[cx:cx+3],phase['General']['SGData'])[1]*a[cx+3] for a in phase['Atoms']],float)
    assert len(weights)<len(geometry.occupancy)
    np.testing.assert_allclose(weights.sum(),geometry.occupancy.sum(),rtol=1e-12,atol=1e-12)
    weights/=weights.sum()
    mapping=value.arrays['physical_to_native'];basis=np.c_[mapping,value.arrays['gauge_basis']]
    apply_modes(phase,np.zeros(len(mapping)),slow.m)
    base=np.array([a[cx:cx+3] for a in phase['Atoms']])
    metric=direct_metric_from_reciprocal(native_reciprocal_matrix(geometry.a0))
    for family,amplitude in zip(families.families,amplitudes):
        native=mapping@point;selected=np.zeros_like(native)
        selected[family['native_indices']]=native[family['native_indices']]
        physical=np.linalg.solve(basis,selected)[:source.mode_dimensions]
        apply_modes(phase,mapping@physical,slow.m)
        displacement=(np.array([a[cx:cx+3] for a in phase['Atoms']])-base+.5)%1-.5
        expected=np.sqrt(np.einsum('a,ai,ij,aj->',weights,displacement,metric,displacement))
        np.testing.assert_allclose(amplitude,expected,rtol=1e-9,atol=1e-12)
