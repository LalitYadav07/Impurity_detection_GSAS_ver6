"""Native parent reference versus independently saved-project calculation."""
from pathlib import Path
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN,GSASIIlattice as L
from gpx_recipe.preparation.structural_parent_reference import (
    NativeParentReference,coefficients,parent_metric,child_metric,
)
from gpx_recipe.structural_live.optimization.physical_prior_geometry import native_reciprocal_matrix
from gpx_recipe.symmetry.compiler import file_sha
from test_native_profile_capture import native_source


def reference(root,kind):
    root.mkdir();source,_=native_source(root/'input',kind)
    cif=root/'parent.cif';G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(cif))
    profile=root/'input/profile.instprm'
    grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,upper=125. if kind=='PNC' else 40000.,bins=701,bank=1)
    owned=NativeParentReference(cif,profile,root/'parent-forward',grid=grid,maximum_relative_metric=.005)
    return source,cif,profile,owned,grid


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_reference_matches_independent_native_calculation_at_anisotropic_parent_cells(tmp_path,kind):
    source,cif,profile,owned,_=reference(tmp_path/'control',kind)
    hashes=[file_sha(p) for p in (source,cif,profile)]
    baseline=coefficients(owned.space.baseline)
    shifted=baseline.copy();shifted[:3]*=[1.0005,.9997,1.0009]
    curves=[owned.evaluate(a) for a in (baseline,shifted)]
    assert owned.native_calls==2
    first=owned.evaluate(shifted);first[0][0]=99999.
    np.testing.assert_array_equal(owned.evaluate(shifted)[0],curves[1][0])
    assert owned.native_calls==2
    # DoNoFit is deliberately performed after the owned forward evaluations:
    # it replaces native global maps, so this context must not run afterward.
    for i,a in enumerate((baseline,shifted)):
        project=G.G2Project(gpxfile=owned.project.filename);phase=project.phase(0);hist=project.histogram(0)
        phase.data['General']['Cell'][1:]=list(L.A2cell(a))+[float(L.calc_V(a))]
        output=tmp_path/f'oracle-{i}.gpx';project.save(str(output))
        expected=np.asarray(MAIN.DoNoFit(str(output),hist.name),float)[owned.order]
        np.testing.assert_allclose(curves[i][0],expected,rtol=1e-11,atol=1e-8)
    assert hashes==[file_sha(p) for p in (source,cif,profile)]
    with pytest.raises(RuntimeError,match='constraint'):owned.evaluate(shifted)


def test_reference_cell_transform_matches_native_nonstandard_child_basis():
    cell=[4.1,5.2,7.3,90.,103.,90.]
    g=native_reciprocal_matrix(L.cell2A(cell));t=np.array([[0,1,0],[1,0,1],[1,0,0.]])
    native=native_reciprocal_matrix(L.cell2A(L.TransformCell(cell,t)[:6]))
    np.testing.assert_allclose(child_metric(g,t),native,rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(parent_metric(native,t),g,rtol=1e-12,atol=1e-15)


def test_reference_rejects_unreviewed_metric_and_bounds_cache(tmp_path):
    _,_,_,owned,_=reference(tmp_path/'control','PNT')
    baseline=coefficients(owned.space.baseline)
    with pytest.raises(ValueError,match='envelope'):owned.evaluate(baseline*1.1)
    owned.maximum_cache_bytes=4*len(owned.x)*8  # two curve/background pairs
    for factor in (1.,1.0001,1.0002):owned.evaluate(baseline*factor)
    assert len(owned.cache)==2 and owned.cache_bytes<=owned.maximum_cache_bytes
