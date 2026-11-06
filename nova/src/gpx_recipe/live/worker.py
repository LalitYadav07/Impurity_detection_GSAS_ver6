"""Local scientific worker for a sealed reduced-snapshot experiment.

Initial integration: exact native per-subgroup kernels and conditional diagonal
Gaussian posteriors. No calibrated physical exclusion or exposure forecasting.
"""
import copy,hashlib,json,time,sys,traceback
from pathlib import Path
import numpy as np
from .contracts import validate_request,seal_posterior,verify_posterior

def _read_code_identity():
    import inspect
    from ..symmetry.compiler import package_signature
    result={name:sha(Path(__file__).parent/name) for name in ["statistics.py","native_profile.py","native_peak_memo.py","validated_backend.py","native_quadratic.py","native_kernel_policy.py","native_reflection.py","reflection_factors.py","quadratic_qualification.py","moment_basis.py","central_quadratic.py","gpx_index.py","native_validation.py","contracts.py","worker.py","observations.py","gpx_export.py","resources.py"]}
    import importlib.util
    for name in ['gsas_magnetic_preflight','gpx_recipe.symmetry.native_magnetic','gpx_recipe.symmetry.native_preparation','gpx_recipe.symmetry.worker','gpx_recipe.execution','gpx_recipe.preparation.identity','gpx_recipe.preparation.hints','gpx_recipe.structural_live.optimization.native_profile_capture']:
        spec=importlib.util.find_spec(name)
        result[name]=sha(Path(spec.origin)) if spec and spec.origin else 'unavailable'
    result.update({fn.__name__:hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() for fn in [compile_candidate,infer]})
    # The explicit inventory is useful diagnostics but does not cover every
    # imported scientific helper or policy table. Bind the complete package
    # too, so a shared construction/helper change cannot retain an old kernel.
    result['gpx_recipe_package_sha256']=package_signature()
    return result

def code_identity():return dict(_CODE_IDENTITY)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));tmp.replace(path)

def compile_candidate(source,candidate,k,sites,out,*,particles=256,factor_cache=None):
    from GSASII import GSASIIscriptable as G,GSASIIlattice as L
    from gpx_recipe.symmetry.native_magnetic import simulate
    from gpx_recipe.symmetry.worker import Artifacts
    from .native_profile import NativeReducedProfile
    from .gpx_index import install_fresh_gpx_index
    install_fresh_gpx_index();g=G.G2Project(str(source));hist=g.histogram(0)
    parents=[p for p in g.phases() if p.data['General']['Type']=='nuclear' and all(any(a.label==s['label'] for a in p.atoms()) for s in sites)]
    if len(parents)!=1:raise ValueError('Magnetic parent is missing or ambiguous; select an unambiguous parent phase')
    if len(g.histograms())!=1:raise ValueError('The live adapter currently requires one verified histogram')
    parent=parents[0]
    from gsas_magnetic_preflight import setting_fingerprint
    protocol=dict(source={'sha256':sha(source)},parent_setting_sha256=setting_fingerprint(parent.data['General']['SGData']),parent_phase=parent.name,histogram=hist.name,k=k,sites=sites,policy=dict(maximum_cell_multiplier=16,magnetic_d_min=0,cycles=1,maximum_moment=9.,site_policy='all',refine_histogram_scale=False,refine_background=False,moment_model='symmetry_allowed'))
    seed=dict(axis='x',magnitude=6.,parent_direction=None,independent_mode_seed=17,site_signs={})
    out.mkdir(parents=True,exist_ok=True)
    result=simulate(source,Artifacts(out),protocol,candidate,seed)
    project=out/result['gpx'];native=NativeReducedProfile(project)
    n=len(native.x0)
    if not n:raise ValueError('No independent magnetic parameters')
    from .resources import kernel_budget
    memory_budget=kernel_budget(n,len(native.counts),particles)
    # The native candidate constructor freezes all nonmagnetic parameters.
    from .native_kernel_policy import compile_native_magnetic_kernel
    compiled=compile_native_magnetic_kernel(native,np.zeros(n),list(range(n)),checks=5,
        factor_cache=factor_cache)
    zero=np.zeros(n);base=compiled['base'];q=compiled['quadratic']
    histograms,phases=native.histo_phases[:2]
    from .moment_basis import from_native
    basis=from_native(native,zero,list(range(n)))
    forms=np.einsum('aki,akj->aij',basis.cartesian,basis.cartesian)
    labels=[str(phases[name]['Atoms'][atom][0]) for name,atom,_ in basis.atoms]
    arrays=[histograms[name]['Data'] for name,sl in native.parts]
    x=np.concatenate([histograms[name]['Data'][0][sl] for name,sl in native.parts]);y=np.concatenate([histograms[name]['Data'][1][sl] for name,sl in native.parts]);weight=np.concatenate([native.original_weights[name][sl] for name,sl in native.parts])
    mask=np.isfinite(x)&np.isfinite(y)&np.isfinite(weight)&(weight>0)
    errors=[];rng=np.random.default_rng(83)
    for _ in range(3):
        p=rng.normal(0,2,n);direct=native.profile(p);pred=base+np.einsum('i,ijb,j->b',p,q,p)
        e=float(np.max(abs(pred-direct))/max(np.max(abs(direct)),1e-12));errors.append(e)
        if e>1e-7:raise ValueError('Independent native kernel validation failed')
    np.savez_compressed(out/'kernel.npz',base=base[mask],quadratic=q[:,:,mask],forms=np.array(forms),x=x[mask],y=y[mask],sigma=1/np.sqrt(weight[mask]),labels=np.array(labels))
    from .validated_backend import active_identity
    meta=dict(native_gpx_sha256=sha(project),kernel_sha256=sha(out/'kernel.npz'),compiler=compiled['diagnostics'],exact_native_peak_reuse=native.peak_memo.diagnostics(),code_identity=code_identity(),backend=active_identity(),id=candidate['id'],name=candidate['magnetic_space_group'],dimensions=n,bin_count=len(native.counts),memory_budget=memory_budget,labels=labels,source_sha256=sha(source),candidate=candidate,k=k,sites=sites,native_errors=errors,validated=True,fixed_nuclear=True)
    write(out/'kernel.json',meta);return meta

def infer(kernel,destination,*,lower,upper,particles=256,seconds=45.,seed=0):
    from .statistics import QuadraticObservation,sample_posterior,compare_replicas
    with np.load(kernel,allow_pickle=False) as a:data={k:a[k].copy() for k in a.files}
    model=QuadraticObservation(data['base'],data['quadratic'],data['y'],data['sigma'],data['forms'],lower,upper)
    a=sample_posterior(model,particles=particles,seed=seed,max_seconds=seconds)
    b=sample_posterior(model,particles=particles,seed=seed+7741,max_seconds=seconds)
    health=compare_replicas(model,a,b);samples=np.concatenate([a['samples'],b['samples']]);ll=model.loglike(samples);best=samples[np.argmax(ll)]
    # Plot payload is bounded; full bins and draws stay in the archive.
    take=np.unique(np.linspace(0,len(data['x'])-1,min(1800,len(data['x']))).astype(int));draws=model.predict(samples[::max(1,len(samples)//48)])
    prior_moments=model.moments(np.concatenate([a['prior'],b['prior']]))
    moments=model.moments(samples);intervals=np.quantile(moments,[.025,.5,.975],axis=0).T
    denominator=float(np.sum((data['y']/data['sigma'])**2))
    if denominator<=0:raise ValueError('Rwp undefined for zero weighted observation norm')
    calc=model.predict(best)[0];rwp=100*np.sqrt(np.sum(((data['y']-calc)/data['sigma'])**2)/np.sum((data['y']/data['sigma'])**2))
    np.savez_compressed(Path(destination)/'posterior.npz',samples=samples,prior=np.concatenate([a['prior'],b['prior']]),moments=moments,loglike=ll)
    result=dict(prior_moment_draws=prior_moments[::max(1,len(prior_moments)//256)].tolist(),relative_half_width=((intervals[:,2]-intervals[:,0])/(2*np.maximum(intervals[:,1],1e-12))).tolist(),prior_edge_mass=float(np.mean(np.any((moments<lower+.01*(upper-lower))|(moments>upper-.01*(upper-lower)),axis=1))),status='conditional_posterior' if health['screen_passed'] else 'sampling_unresolved',rwp=float(rwp),reduced_chi2=float(-2*ll.max()/max(1,len(data['y'])-model.dimension)),health=health,
      seconds=a['seconds']+b['seconds'],intervals=intervals.tolist(),labels=data['labels'].tolist(),x=data['x'][take].tolist(),y=data['y'][take].tolist(),sigma=data['sigma'][take].tolist(),fit=calc[take].tolist(),
      mean_band=np.quantile(draws,[.025,.5,.975],axis=0)[:,take].tolist(),ensemble=draws[:8,take].tolist(),moment_draws=moments[::max(1,len(moments)//256)].tolist(),parameter_draws=samples[::max(1,len(samples)//256)].tolist(),
      sampler_history=a['history'],sampler_histories=[a['history'],b['history']],
      prior_diagnostics=[a['prior_diagnostics'],b['prior_diagnostics']],
      particles=particles,exclusion_authorized=False,forecast_enabled=False,scope='Conditional on fixed nuclear baseline and supplied diagonal reduced variances; reference/cross-bin covariance unqualified')
    write(Path(destination)/'posterior.json',result);return result

def execute(request_path,candidate_id=None):
    request_path=Path(request_path).resolve();req=json.loads(request_path.read_text());out=request_path.parent
    report_dir=out if candidate_id is None else out/'items'/candidate_id.replace(':','_')
    report_dir.mkdir(parents=True,exist_ok=True)
    source=Path(req['source_gpx']);catalog=json.loads(Path(req['catalog']).read_text());preflight=json.loads(Path(req['preflight']).read_text())
    if sha(source)!=req['source_sha256'] or sha(req['catalog'])!=req['catalog_sha256'] or sha(req['preflight'])!=req['preflight_sha256']:raise ValueError('Scientific source changed')
    if catalog['k']!=req['k']:raise ValueError('Catalogue k differs from request')
    rows=[];started=time.monotonic()
    report=dict(code_identity=code_identity(),schema='radar-webmon-result/v1',status='running',request_sha256=sha(request_path),source_revision=req['source_revision'],reference_revision=req['reference_revision'],rows=rows,forecast_enabled=False,exclusion_authorized=False)
    for index,c in enumerate(catalog['candidates']):
        if candidate_id is not None and c['id']!=candidate_id:continue
        r=dict(id=c['id'],name=c['magnetic_space_group'],bns=c['bns_number'],status='pending');rows.append(r)
        info=next(v for v in preflight if v['id']==c['id']);dest=out/'models'/c['id'].replace(':','_')
        try:
            if int(str(c['bns_number']).split('.')[0]) in req.get('excluded_parent_space_groups',[]):r.update(status='policy_omitted',reason='Omitted by the explicitly reviewed parent-space-group policy');continue
            if info.get('error'):r.update(status='construction_unresolved',reason=info['error']);continue
            if not info['compatible']:r.update(status='constraint_incompatible',reason='Selected sites do not satisfy the reviewed magnetic-site policy');continue
            # Explicit finite initial integration budget; never label deferred models excluded.
            if info['dimensions']>req.get('maximum_dimensions',32):r.update(status='deferred',reason='Exceeds declared preparation dimension budget');continue
            r['status']='preparing';write(report_dir/'result.json',report)
            meta=json.loads((dest/'kernel.json').read_text()) if (dest/'kernel.json').exists() else None
            preparation_started=time.monotonic()
            if meta and not meta.get('bin_count'):meta=None
            if meta and meta.get('observation_revision') and not req.get('observation'):meta=None
            from .validated_backend import active_identity
            if meta and (meta.get('backend')!=active_identity() or meta.get('sites')!=req['sites'] or meta.get('k')!=req['k']):meta=None
            cache_hit=not (meta is None or not (dest/'magnetic-refinement.gpx').is_file() or meta.get('native_gpx_sha256')!=sha(dest/'magnetic-refinement.gpx') or meta.get('code_identity')!=code_identity() or meta['source_sha256']!=sha(source) or meta['candidate']!=c or meta.get('kernel_sha256')!=sha(dest/'kernel.npz'))
            if not cache_hit:
                factor_cache=None;prepared_scattering=None
                if req.get('prepared_scattering') is not None:
                    from ..preparation.hints import cache_for,live_context
                    factor_cache,prepared_scattering=cache_for(Path(req['catalog']).parent.parent,
                        live_context(req,catalog),candidate=c['id'])
                meta=compile_candidate(source,c,req['k'],req['sites'],dest,particles=req['particles'],
                    factor_cache=factor_cache)
                if prepared_scattering is not None:
                    meta['prepared_scattering']=prepared_scattering;write(dest/'kernel.json',meta)
            from .resources import kernel_budget
            kernel_budget(meta['dimensions'],meta['bin_count'],req['particles'])
            from .observations import replace_observations
            if replace_observations(dest/'kernel.npz',req):
                meta['kernel_sha256']=sha(dest/'kernel.npz');meta['observation_revision']=req['source_revision'];write(dest/'kernel.json',meta)
            r.update(status='sampling',dimensions=meta['dimensions'],native_errors=meta['native_errors'],preparation_seconds=time.monotonic()-preparation_started,response_cache_hit=cache_hit);write(report_dir/'result.json',report)
            if meta.get('prepared_scattering') is not None:r['prepared_scattering']=meta['prepared_scattering']
            inference_key=dict(lower=req['lower'],upper=req['upper'],particles=req['particles'],code_identity=code_identity(),kernel_sha256=sha(dest/'kernel.npz'))
            existing_key=json.loads((dest/'inference-request.json').read_text()) if (dest/'inference-request.json').exists() else None
            if (dest/'posterior.json').exists() and existing_key==inference_key:
                post=json.loads((dest/'posterior.json').read_text())
            else:
                post=infer(dest/'kernel.npz',dest,lower=req['lower'],upper=req['upper'],particles=req['particles'],seconds=req.get('candidate_seconds',45),seed=180+index)
                write(dest/'inference-request.json',inference_key)
            # The construction GPX is a starting model, never the inferred fit.
            # Expose a separate checkpoint only after reopening it and verifying
            # its parameters, observations and native curve against this posterior.
            try:
                from .gpx_export import export_posterior_gpx
                exported = export_posterior_gpx(dest, req, backend_active=True)
                r['fitted_gpx'] = str(dest / exported['fitted_gpx'])
                r['fitted_gpx_sha256'] = exported['fitted_gpx_sha256']
                r['gpx_status'] = exported['status']
            except Exception as exc:
                r['gpx_status'] = 'unavailable'
                r['gpx_warning'] = f'No matching inferred GPX: {type(exc).__name__}: {exc}'
            # Presentation metadata must never turn an otherwise completed fit into a failure.
            try:
                from .reflections import export_reflections
                if (dest/'magnetic-refinement.gpx').exists():
                    tick_path=dest/'reflections.json'
                    ticks=json.loads(tick_path.read_text()) if tick_path.exists() else {}
                    if ticks.get('posterior_sha256')!=sha(dest/'posterior.npz') or ticks.get('kernel_sha256')!=sha(dest/'kernel.npz'):
                        export_reflections(dest,dest,backend_active=True)
            except Exception as exc:
                r['reflection_warning']=f'Native ticks unavailable: {type(exc).__name__}: {exc}'
            r['artifacts']=seal_posterior(dest)
            r.update({k:post[k] for k in ['status','rwp','reduced_chi2','health','seconds','intervals','labels']});r['posterior']=str(dest/'posterior.json')
        except TimeoutError as exc:
            if r.get('status')!='sampling':
                r.update(status='numerical_failure',reason=type(exc).__name__+': '+str(exc)[:600])
                (report_dir/'failures.log').open('a').write(traceback.format_exc()+'\n')
            else:
                r.update(status='sampling_unresolved',sampling_incomplete=True,
                    reason=str(exc)[:600],kernel_retained=True)
        except Exception as exc:
            r.update(status='numerical_failure',reason=type(exc).__name__+': '+str(exc)[:600]);(report_dir/'failures.log').open('a').write(traceback.format_exc()+'\n')
        finally:
            report['elapsed_seconds']=time.monotonic()-started;write(report_dir/'result.json',report)
            print(index+1,len(catalog['candidates']),r['name'],r['status'],r.get('reason',''),flush=True)
    report['status']='complete';write(report_dir/'result.json',report)

def parallel(request_path):
    import os,subprocess,signal
    req=json.loads(Path(request_path).read_text());out=Path(request_path).parent
    catalog=json.loads(Path(req['catalog']).read_text());rows=[dict(id=c['id'],name=c['magnetic_space_group'],bns=c['bns_number'],status='pending') for c in catalog['candidates']]
    report=dict(code_identity=code_identity(),schema='radar-webmon-result/v1',status='running',request_sha256=sha(request_path),source_revision=req['source_revision'],reference_revision=req['reference_revision'],rows=rows,forecast_enabled=False,exclusion_authorized=False,workers=req['workers'])
    from gpx_recipe.symmetry.resources import allocation
    resources=allocation(req['workers']);report['resources']=resources;report['workers']=resources['workers']
    active={};waiting=list(range(len(rows)));started=time.monotonic();env=dict(os.environ)
    env['RADAR_PD_LIVE_WORKERS']=str(resources['workers'])
    env['PYTHONPATH']=os.pathsep.join(sys.path)
    def cleanup(*_):
        for process,log,_ in active.values():
            if process.poll() is None:process.kill()
            process.wait();log.close()
        if _:
            report['status']='cancelled';write(out/'result.json',report)
            raise SystemExit(143)
    previous=signal.signal(signal.SIGTERM,cleanup)
    try:
        while waiting or active:
            while waiting and len(active)<resources['workers']:
                i=waiting.pop(0);item=out/'items'/rows[i]['id'].replace(':','_');item.mkdir(parents=True,exist_ok=True)
                previous_result=item/'result.json'
                if previous_result.exists():previous_result.rename(item/f'prior-result-{time.time_ns()}.json')
                log=(item/'worker.log').open('ab');process=subprocess.Popen([sys.executable,'-m','gpx_recipe.live.worker',str(request_path),'--candidate',rows[i]['id']],env=env,stdout=log,stderr=log)
                active[i]=(process,log,time.monotonic())
            for i,(process,log,t0) in list(active.items()):
                result=out/'items'/rows[i]['id'].replace(':','_')/'result.json'
                if result.exists():
                    value=json.loads(result.read_text())
                    if value.get('request_sha256')==sha(request_path) and value.get('code_identity')==code_identity() and value['rows']:rows[i]=value['rows'][0]
                if process.poll() is None and time.monotonic()-t0>req.get('preparation_timeout_seconds',600):
                    process.kill();process.wait();rows[i].update(status='numerical_failure',reason='Candidate wall-time budget exceeded; unresolved')
                if process.poll() is not None:
                    if (process.returncode or rows[i]['status'] in ('pending','preparing','sampling')) and rows[i]['status'] not in ['numerical_failure']:
                        rows[i].update(status='numerical_failure',reason='Native worker exited without a complete result; unresolved')
                    log.close();del active[i]
            report['elapsed_seconds']=time.monotonic()-started;write(out/'result.json',report)
            time.sleep(.25)
        report['status']='complete';write(out/'result.json',report)
    finally:cleanup();signal.signal(signal.SIGTERM,previous)

def run(request_path,candidate_id=None):
    import fcntl
    request_path=Path(request_path).resolve()
    suffix='run' if candidate_id is None else 'candidate-'+hashlib.sha256(candidate_id.encode()).hexdigest()[:24]
    with (request_path.parent/('.'+suffix+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('This native job already has an active writer') from None
        req=json.loads(request_path.read_text());validate_request(req)
        if candidate_id is None and req.get('workers',1)>1:return parallel(request_path)
        from .validated_backend import corrected_backend
        with corrected_backend():execute(request_path,candidate_id)

_CODE_IDENTITY=_read_code_identity()

if __name__=='__main__':run(sys.argv[1],sys.argv[3] if len(sys.argv)>3 and sys.argv[2]=='--candidate' else None)
