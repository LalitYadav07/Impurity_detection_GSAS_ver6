"""Qualify cell-free TOF shapes against the original installed native driver."""
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from test_native_tof_scalars import native_records
from gpx_recipe.structural_live.optimization.tof_shape import IsotropicTOFShape
from gpx_recipe.structural_live.optimization.native_cell_proposals import ProposalUnavailable


@pytest.mark.parametrize('change',['parent','cell','XYZ','size','strain'])
def test_same_native_peak_parameters_without_cell_or_atom_inputs(tmp_path,change):
    _,s=native_source(tmp_path/'input','PNT');n=s.fast.native
    h=n.histo_phases[0][s.fast.slow.hist.name];phase=n.histo_phases[1][s.fast.slow.phase.name]
    hfx=f":{h['hId']}:";phfx=f"{phase['pId']}:{h['hId']}:"
    point=np.zeros(len(s.radii))
    if change=='cell':point[s.mode_dimensions:]=np.linspace(-.6,.7,s.cell_dimensions)
    elif change=='XYZ':
        for name,value in [('X',.02),('Y',.04),('Z',.06)]:n.parms[hfx+name]=value
    elif change=='size':n.parms[phfx+'Size;i']=.8;n.parms[phfx+'Size;mx']=.3
    elif change=='strain':n.parms[phfx+'Mustrain;i']=700.;n.parms[phfx+'Mustrain;mx']=.4
    original,_=native_records(s,point)
    rows=h['Reflection Lists'][s.fast.slow.phase.name]['RefList']
    ds={tuple(map(int,r[:3])):r[4] for r in rows}
    shape=IsotropicTOFShape(n,s.fast.slow.hist.name,s.fast.slow.phase.name)
    actual=shape.at(np.asarray([ds[k] for k in original]))
    expected=np.asarray([r[0][1:] for r in original.values()])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    restored=IsotropicTOFShape.from_specification(shape.specification,n.math)
    np.testing.assert_array_equal(restored.at(np.asarray([ds[k] for k in original])),actual)
    assert restored.identity==shape.identity


def test_specification_survives_different_lattice_atoms_space_group_and_grid(tmp_path,monkeypatch):
    _,s=native_source(tmp_path/'first','PNT');n=s.fast.native
    first=IsotropicTOFShape(n,s.fast.slow.hist.name,s.fast.slow.phase.name)
    from gpx_recipe.execution import freeze
    from gpx_recipe.live.native_profile import NativeReducedProfile
    p=G.G2Project(newgpx=str(tmp_path/'other.gpx'))
    phase=p.add_phase(phasename='Different public structure',spacegroup='P 1',
        cell=[5.1,6.2,7.3,80.,90.,100.]);phase.add_atom(.12,.23,.34,'Si','Other1')
    hist=p.add_simulated_powder_histogram('Other grid',str(tmp_path/'first/profile.instprm'),
        4.,35.,Npoints=1111,scale=2.,phases=[phase]);freeze(p);p.save()
    native=NativeReducedProfile(tmp_path/'other.gpx')
    second=IsotropicTOFShape(native,hist.name,phase.name)
    assert first.identity==second.identity
    ds=np.geomspace(.3,8.,173)
    # Evaluation uses the copied native shape specification, not the old
    # global constraint owner, a structure factor or measured arrays.
    monkeypatch.setattr(native.math,'StructureFactor2',lambda *a,**k:pytest.fail('No atomic scattering belongs in a peak shape'))
    np.testing.assert_array_equal(first.at(ds),second.at(ds))
    spec=first.specification;spec['bank']['X']=123.
    assert first.specification['bank']['X']!=123.


@pytest.mark.parametrize('change',['cw','anisotropic','lookup','function'])
def test_unsupported_native_shape_context_does_not_claim_reuse(tmp_path,monkeypatch,change):
    _,s=native_source(tmp_path/'input','PNC' if change=='cw' else 'PNT');n=s.fast.native
    h=n.histo_phases[0][s.fast.slow.hist.name];phase=n.histo_phases[1][s.fast.slow.phase.name]
    hfx=f":{h['hId']}:";phfx=f"{phase['pId']}:{h['hId']}:"
    if change=='anisotropic':n.controls[phfx+'SizeType']='uniaxial'
    elif change=='lookup':n.histdict[hfx+'pdabc']={'d':[1,2],'sig':[1,2]}
    elif change=='function':monkeypatch.setattr(n.math,'GetSampleSigGam',lambda *a,**k:None)
    with pytest.raises(ProposalUnavailable):IsotropicTOFShape(n,s.fast.slow.hist.name,s.fast.slow.phase.name)


@pytest.mark.parametrize('d',[[],[0.],[-1.],[float('nan')],[[1.,2.]],np.ones(25001)])
def test_shape_grid_bound_precedes_native_work(tmp_path,d):
    _,s=native_source(tmp_path/'input','PNT')
    shape=IsotropicTOFShape(s.fast.native,s.fast.slow.hist.name,s.fast.slow.phase.name)
    with pytest.raises(ValueError):shape.at(d)
