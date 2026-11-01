"""Reopen public artificial GPXs and verify exact posterior/observation export."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIlattice as L, GSASIIstrMain as MAIN, GSASIIobj as OBJ
from gpx_recipe.execution import freeze
from gpx_recipe.live.validated_backend import corrected_backend
from gpx_recipe.live.native_profile import NativeReducedProfile
from gpx_recipe.live.central_quadratic import compile_even_quadratic
from gpx_recipe.live.gpx_export import export_posterior_gpx
from gpx_recipe.live.contracts import file_hash


def artificial(tmp_path, sites):
    inst=tmp_path/'public.instprm'
    inst.write_text('#GSAS-II instrument parameter file\nType:PNC\nBank:1.0\nLam:1.54\nZero:0.0\nU:0.12\nV:0.0\nW:0.08\nX:0.0\nY:0.0\nZ:0.0\nSH/L:0.002\nPolariz.:0.0\nAzimuth:0.0\n')
    path=tmp_path/'magnetic-refinement.gpx';p=G.G2Project(newgpx=str(path));cell=[4.7,5.6,6.8,79.,93.,104.]
    nuclear=p.add_phase(phasename='Nuclear',cell=cell);nuclear.add_atom(.13,.21,.34,'Fe','Fe1');nuclear.add_atom(.37,.29,.19,'O','O1')
    mag=p.add_phase(phasename='Magnetic',cell=cell);general=mag.data['General'];general['Type']='magnetic';general['AtomPtrs']=[3,1,10,12]
    general['SGData']=L.makeBilbaoPhase(['P1',''],[0.,0.,0.],np.eye(3),True)['SGData'];general['MagDmin']=.6;G.SetupGeneral(mag.data,None)
    for i in range(sites):
        mag.add_atom(.13+i*.2,.21,.34,'Fe+3','Fe'+str(i));mag.data['Atoms'][i][7:10]=[1.,1.2,.8]
    general['Lande g']=[2.]*len(general['AtomTypes'])
    hist=p.add_simulated_powder_histogram('Public',str(inst),12.,130.,Npoints=301,scale=120.,phases=p.phases())
    freeze(p);hist.data['data'][1][1][:]=1.;hist.data['data'][1][2][:]=4.;hist.data['data'][1][2][12]=0.;hist.data['Limits'].append([45.,48.])
    for atom in mag.data['Atoms']:atom[2]='M'
    if sites==2:
        p.index_ids()
        for axis in 'xyz':
            variables=[OBJ.G2VarObj([mag.data['ranId'],None,'AM'+axis,a[-1]]) for a in mag.data['Atoms']]
            p.data['Constraints']['data']['Phase'].append([[1.,variables[0]],[1.,variables[1]],0.,None,'c'])
    p.save();return p,path


@pytest.mark.parametrize('sites,stream',[(1,False),(2,True)])
def test_posterior_checkpoint_reopens_at_plotted_parameters_and_current_observations(tmp_path,sites,stream):
    with corrected_backend():
        project,source=artificial(tmp_path,sites);before=file_hash(source)
        native=NativeReducedProfile(source);n=len(native.x0);best=np.linspace(.65,1.4,n)
        base=native.profile(np.zeros(n));q=compile_even_quadratic(native.profile,n)
        x=np.concatenate([native.histo_phases[0][name]['Data'][0][sl] for name,sl in native.parts])
        y=np.concatenate([native.histo_phases[0][name]['Data'][1][sl] for name,sl in native.parts])
        w=np.concatenate([native.original_weights[name][sl] for name,sl in native.parts]);valid=w>0
        request={'source_revision':'PG3_1:public','reference_revision':'PG3_2:public'}
        if stream:
            arrays=project.histogram(0).data['data'][1];obs=tmp_path/'observation.npz'
            np.savez(obs,x=np.asarray(arrays[0]),y=np.asarray(arrays[1])*2,sigma=np.full(len(arrays[0]),.25),mask=np.zeros(len(arrays[0]),bool))
            request.update(observation=str(obs),observation_sha256=file_hash(obs));y=y*2;w[valid]=16.
        np.savez(tmp_path/'kernel.npz',x=x[valid],y=y[valid],sigma=1/np.sqrt(w[valid]),base=base[valid],quadratic=q[:,:,valid])
        np.savez(tmp_path/'posterior.npz',samples=best[None,:],loglike=np.array([0.]))
        exported=export_posterior_gpx(tmp_path,request,backend_active=True)
        assert exported['native_profile_relative_error']<1e-7
        assert exported['native_profile_max_error_sigma']<1e-5
        assert exported['native_convergence_declared'] is False
        assert file_hash(source)==before
        saved=G.G2Project(str(tmp_path/exported['fitted_gpx']))
        assert saved.data['Covariance']['data']=={}
        assert file_hash(tmp_path/exported['fitted_gpx'])==exported['fitted_gpx_sha256']
        assert saved.histogram(0).data['Limits']==project.histogram(0).data['Limits']
        np.testing.assert_array_equal(saved.phase('Nuclear').data['Atoms'],project.phase('Nuclear').data['Atoms'])
        assert saved.phase('Magnetic').data['Atoms']!=project.phase('Magnetic').data['Atoms']
