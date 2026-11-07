"""Independent native controls generated from public artificial P1 models."""
import copy
import hashlib
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIlattice as L,GSASIIstrMain as MAIN,GSASIIobj as OBJ
from gpx_recipe.execution import freeze
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint


def test_reuses_only_checked_objective_and_rechecks_native_on_each_load(tmp_path):
    project,path=artificial_project(tmp_path,sites=1)
    cache=tmp_path/'objective.npz'
    first=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,proof=first.build(cache_path=cache,cache_identity='same-sealed-protocol')
    assert not proof['cache_hit'] and cache.is_file()
    fresh=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    reused,check=fresh.build(cache_path=cache,cache_identity='same-sealed-protocol')
    differences=[k for k,v in getattr(first,'_qualification_components',{}).items()
        if getattr(fresh,'_qualification_components',{}).get(k)!=v]
    assert check['cache_hit'] and check['native_calls']==2,(check['qualification_reuse'],differences)
    assert check['native_calls']<proof['native_calls']
    assert check['qualification_reuse']==dict(owned_process=True,complete_prior_native_probes=4,
        current_native_probes=1,requested_native_probes=4,imported_qualification_accepted=False,
        qualification_persisted=False,unavailable_reason='')
    np.testing.assert_array_equal(reused.quadratic,model.quadratic)
    assert max(check['heldout_relative_errors'])<1e-8
    hist=project.histograms()[0]
    hist.data['data'][1][1][10]+=1.;project.save()
    changed=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    _,changed_proof=changed.build(cache_path=cache,cache_identity='same-sealed-protocol')
    assert not changed_proof['cache_hit']


def test_new_native_context_or_stronger_probe_count_requires_complete_checks(tmp_path):
    from gpx_recipe.symmetry import profiled_qualification
    project,path=artificial_project(tmp_path)
    cache=tmp_path/'objective.npz'
    first=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    first.build(cache_path=cache,cache_identity='same-protocol')
    stronger=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    _,proof=stronger.build(heldouts=6,cache_path=cache,cache_identity='same-protocol')
    assert proof['cache_hit'] and not proof['qualification_reuse']['owned_process']
    assert proof['qualification_reuse']['current_native_probes']==6
    # A profile edit leaves the scale-zero offset unchanged. It must never
    # inherit qualification solely from cache identity, offset or moment maps.
    project.histograms()[0].data['Instrument Parameters'][0]['W'][1]*=1.2
    project.save()
    changed=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    _,proof=changed.build(cache_path=cache,cache_identity='same-protocol')
    assert not proof['cache_hit'] and not proof['qualification_reuse']['owned_process']
    assert proof['qualification_reuse']['current_native_probes']==4
    # An independently started process has no ledger even when it sees the
    # same valid data-only cache. Simulate only that ledger boundary here.
    profiled_qualification._entries.clear()
    cold=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    _,proof=cold.build(cache_path=cache,cache_identity='same-protocol')
    assert proof['cache_hit'] and not proof['qualification_reuse']['owned_process']
    assert proof['qualification_reuse']['current_native_probes']==4


def test_imported_cache_never_inherits_process_qualification(tmp_path):
    import json
    import subprocess
    import sys
    from gpx_recipe.symmetry.compiler import native_environment,package_signature
    _,path=artificial_project(tmp_path)
    cache=tmp_path/'objective.npz';receipt=tmp_path/'fresh-worker.json'
    context=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    context.build(cache_path=cache,cache_identity='same-protocol')
    code='''import json,sys
from pathlib import Path
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.symmetry.compiler import package_signature
n=NativeMagneticStartingPoint(sys.argv[1],refine_scale=True,refine_background=True,maximum_moment=6.)
_,proof=n.build(cache_path=sys.argv[2],cache_identity='same-protocol')
second=NativeMagneticStartingPoint(sys.argv[1],refine_scale=True,refine_background=True,maximum_moment=6.)
_,again=second.build(cache_path=sys.argv[2],cache_identity='same-protocol')
Path(sys.argv[3]).write_text(json.dumps(dict(proof=proof,again=again,implementation=package_signature())))
'''
    with (tmp_path/'fresh-worker.log').open('w') as log:
        subprocess.run([sys.executable,'-c',code,str(path),str(cache),str(receipt)],
            env=native_environment(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    document=json.loads(receipt.read_text());proof=document['proof']
    assert document['implementation']==package_signature()
    assert proof['cache_hit'] and not proof['qualification_reuse']['owned_process']
    assert proof['qualification_reuse']['current_native_probes']==4
    assert proof['native_calls']==5
    assert max(proof['heldout_relative_errors'])<=1e-8
    # Four matching imported-cache probes are not an owned full coefficient
    # construction. Reopening it again must not promote the imported matrix.
    assert document['again']['cache_hit']
    assert not document['again']['qualification_reuse']['owned_process']
    assert document['again']['qualification_reuse']['current_native_probes']==4


def test_wrong_quadratic_hint_rebuilds_from_native_instead_of_rejecting_model(tmp_path):
    _,path=artificial_project(tmp_path,sites=1)
    cache=tmp_path/'objective.npz'
    context=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    correct,_=context.build(cache_path=cache,cache_identity='same-protocol')
    with np.load(cache,allow_pickle=False) as data:values={k:data[k] for k in data.files}
    values['quadratic']=values['quadratic']*2.
    np.savez_compressed(cache,**values)
    fresh=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    rebuilt,proof=fresh.build(cache_path=cache,cache_identity='same-protocol')
    assert not proof['cache_hit']
    np.testing.assert_array_equal(rebuilt.quadratic,correct.quadratic)
    assert max(proof['heldout_relative_errors'])<1e-8


def artificial_project(tmp_path,*,sites=1,constrained=False,masked=False,scale_free=True,background_free=True,npoints=701):
    path=tmp_path/'artificial.gpx'
    inst=tmp_path/'artificial.instprm'
    inst.write_text('#GSAS-II instrument parameter file\nType:PNC\nBank:1.0\nLam:1.54\nZero:0.0\nU:0.12\nV:0.0\nW:0.08\nX:0.0\nY:0.0\nZ:0.0\nSH/L:0.002\nPolariz.:0.0\nAzimuth:0.0\n')
    p=G.G2Project(newgpx=str(path))
    cell=[4.7,5.6,6.8,79.,93.,104.]
    nuclear=p.add_phase(phasename='Public artificial nuclear',cell=cell)
    positions=[(.13,.21,.34),(.41,.36,.71),(.23,.67,.46),(.76,.14,.58)]
    assert 1<=sites<=len(positions)
    nuclear.add_atom(.37,.29,.19,'O','O1')
    for i,xyz in enumerate(positions[:sites]):nuclear.add_atom(*xyz,'Fe','Fe'+str(i+1))
    mag=p.add_phase(phasename='Public artificial magnet',cell=cell)
    general=mag.data['General'];general['Type']='magnetic';general['AtomPtrs']=[3,1,10,12]
    general['SGData']=L.makeBilbaoPhase(['P1',''],[0.,0.,0.],np.eye(3),True)['SGData']
    general['MagDmin']=.6
    G.SetupGeneral(mag.data,None)
    for i in range(sites):
        xyz=positions[i]
        mag.add_atom(*xyz,'Fe+3','Fe'+str(i+1))
        mag.data['Atoms'][i][7:10]=([1.7,2.1,1.2] if i==0 else [-1.7,-2.1,-1.2])
    general['Lande g']=[2.]*len(general['AtomTypes'])
    hist=p.add_simulated_powder_histogram('public artificial',str(inst),12.,130.,Npoints=npoints,scale=120.,phases=p.phases())
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
    assert receipt['compiler']=='native_reflection_amplitudes'
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


@pytest.mark.parametrize('masked',[False,True])
def test_shared_derivative_compiler_matches_forward_only_oracle(tmp_path,masked):
    project,path=artificial_project(tmp_path,sites=2,masked=masked)
    project.histograms()[0].data['data'][0]['wtFactor']=2.5
    project.save()
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    fast=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,proof=fast._build(use_derivative=True)
    assert proof['compiler']=='native_derivatives',proof
    assert proof['native_derivative_calls']>=12
    direct=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    reference,control=direct._build(use_derivative=False)
    assert control['compiler']=='native_profiles'
    for name in ['offset','nuclear','quadratic','background']:
        np.testing.assert_allclose(getattr(model,name),getattr(reference,name),rtol=1e-8,atol=1e-6)
    assert max(proof['heldout_relative_errors'])<1e-8
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before


def test_bad_native_magnetic_derivatives_fall_back_to_checked_profiles(tmp_path,monkeypatch):
    _,path=artificial_project(tmp_path,sites=2)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    original=native.math.dervRefine
    monkeypatch.setattr(native.math,'dervRefine',lambda *args:2.*original(*args))
    _,proof=native._build(use_derivative=True)
    assert proof['compiler']=='native_profiles'
    assert proof['compiler_note']
    assert max(proof['heldout_relative_errors'])<1e-8


def test_large_basis_automatically_uses_shared_reflection_compiler(tmp_path):
    _,path=artificial_project(tmp_path,sites=4,masked=True)
    native=NativeMagneticStartingPoint(path,refine_scale=True,refine_background=True,maximum_moment=6.)
    model,proof=native.build()
    assert model.dimension==12
    assert proof['compiler']=='native_reflection_amplitudes',proof
    assert max(proof['heldout_relative_errors'])<1e-8
