"""Independent native controls generated from public artificial P1 models."""
import copy
import hashlib
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIlattice as L,GSASIIstrMain as MAIN,GSASIIobj as OBJ
from gpx_recipe.execution import freeze
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint


def artificial_project(tmp_path,*,sites=1,constrained=False,masked=False,scale_free=True,background_free=True):
    path=tmp_path/'artificial.gpx'
    inst=tmp_path/'artificial.instprm'
    inst.write_text('#GSAS-II instrument parameter file\nType:PNC\nBank:1.0\nLam:1.54\nZero:0.0\nU:0.12\nV:0.0\nW:0.08\nX:0.0\nY:0.0\nZ:0.0\nSH/L:0.002\nPolariz.:0.0\nAzimuth:0.0\n')
    p=G.G2Project(newgpx=str(path))
    cell=[4.7,5.6,6.8,79.,93.,104.]
    nuclear=p.add_phase(phasename='Public artificial nuclear',cell=cell)
    nuclear.add_atom(.13,.21,.34,'Fe','Fe1');nuclear.add_atom(.37,.29,.19,'O','O1')
    if sites==2:nuclear.add_atom(.41,.36,.71,'Fe','Fe2')
    mag=p.add_phase(phasename='Public artificial magnet',cell=cell)
    general=mag.data['General'];general['Type']='magnetic';general['AtomPtrs']=[3,1,10,12]
    general['SGData']=L.makeBilbaoPhase(['P1',''],[0.,0.,0.],np.eye(3),True)['SGData']
    general['MagDmin']=.6
    G.SetupGeneral(mag.data,None)
    for i in range(sites):
        xyz=[.13,.21,.34] if i==0 else [.41,.36,.71]
        mag.add_atom(*xyz,'Fe+3','Fe'+str(i+1))
        mag.data['Atoms'][i][7:10]=([1.7,2.1,1.2] if i==0 else [-1.7,-2.1,-1.2])
    general['Lande g']=[2.]*len(general['AtomTypes'])
    hist=p.add_simulated_powder_histogram('public artificial',str(inst),12.,130.,Npoints=701,scale=120.,phases=p.phases())
    hist.data['Background'][0][2]=3;hist.data['Background'][0][3:]=[3.,.2,.1]
    freeze(p)
    hist.data['data'][1][1][:]=1.
    p.save();truth=np.asarray(MAIN.DoNoFit(str(path),hist.name),float)
    rng=np.random.default_rng(77017)
    arrays=hist.data['data'][1];arrays[1]=truth+rng.normal(0.,.05,len(truth));arrays[2]=np.ones(len(truth))*400.
    if masked:
        hist.data['Limits'].append([45.,48.])
        for j in [0,1,2]:arrays[j]=np.ma.array(arrays[j],mask=np.arange(len(truth))%73==0)
    # Bias every seed. Nuclear coordinates, cell and instrument stay fixed.
    for i,atom in enumerate(mag.data['Atoms']):
        atom[2]='M';atom[7:10]=([.7,1.1,1.3] if i==0 else [-.7,-1.1,-1.3])
    if constrained:
        p.index_ids()
        for axis in 'xyz':
            variables=[OBJ.G2VarObj([mag.data['ranId'],None,'AM'+axis,a[-1]]) for a in mag.data['Atoms']]
            p.data['Constraints']['data']['Phase'].append([[1.,variables[0]],[1.,variables[1]],0.,None,'c'])
    hist.data['Sample Parameters']['Scale'][1]=scale_free
    hist.data['Background'][0][1]=background_free
    p.save();p.index_ids()
    return p,path


@pytest.mark.parametrize('sites,constrained,masked,scale_free,background_free',[
    (1,False,False,True,True),(1,False,True,False,False),
    (2,False,True,True,True),(2,True,False,True,True)])
def test_native_full_constraint_basis_matches_independent_profiles(tmp_path,sites,constrained,masked,scale_free,background_free):
    project,path=artificial_project(tmp_path,sites=sites,constrained=constrained,masked=masked,
        scale_free=scale_free,background_free=background_free)
    original=hashlib.sha256(path.read_bytes()).hexdigest()
    adapter=NativeMagneticStartingPoint(path,refine_scale=scale_free,refine_background=background_free,maximum_moment=6.)
    model,receipt=adapter.build()
    assert max(receipt['heldout_relative_errors'])<1e-8
    assert receipt['linear_identity_error']<1e-8
    expected=3 if constrained else sites*3
    assert model.dimension==expected
    if constrained:assert any(key.startswith('::constr') for key in adapter.moment_names)
    seed=adapter.initial[adapter.moment_indices]
    fits=[model.fit(seed),model.fit(-seed*1.2)]
    usable=[fit for fit in fits if fit.success]
    assert usable,[fit.message for fit in fits]
    best=min(usable,key=lambda fit:fit.weighted_cost)
    assert best.weighted_cost<.5*np.dot(adapter.residual(seed,adapter.source_scale,adapter.initial[adapter.background_indices]),
                                      adapter.residual(seed,adapter.source_scale,adapter.initial[adapter.background_indices]))
    applied=adapter.apply_to_project(project,best)
    assert applied['final_native_relative_error']<1e-8
    assert not applied['native_convergence_declared']
    assert not applied['covariance_supplied']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original


def test_declared_freedom_and_process_global_ownership_fail_closed(tmp_path):
    project,path=artificial_project(tmp_path)
    with pytest.raises(ValueError,match='scale freedom'):
        NativeMagneticStartingPoint(path,refine_scale=False,refine_background=True,maximum_moment=6.)
    first=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    with pytest.raises(RuntimeError,match='process-global'):
        first.residual(first.initial[first.moment_indices],first.source_scale,first.initial[first.background_indices])
