"""Native re-preparation of a verified refinement output for the local TOF adapter.

Input is a server-created receipt, never a browser-supplied path. This adapter
preserves nuclear parameters and exclusions; its scale/background transfer is
explicitly provisional and cannot authorize counting advice.
"""
import copy,json,sys
from pathlib import Path
import numpy as np
from .contracts import file_hash as digest


def prepare(receipt_path):
    from GSASII import GSASIIscriptable as G
    from .gpx_index import install_fresh_gpx_index
    from .native_profile import NativeReducedProfile
    from .snapshot import parse_live_ascii,density_to_tof,stable_read
    from gpx_recipe.execution import freeze
    from gpx_recipe.symmetry.native_preparation import normalize_flags
    from gsas_magnetic_preflight import preflight_magnetic_sites,setting_fingerprint
    from .native_validation import checked_native_refine,refresh_saved_profiles
    install_fresh_gpx_index()
    receipt_path=Path(receipt_path).resolve();r=json.loads(receipt_path.read_text());out=receipt_path.parent
    choice=copy.deepcopy(r['base_option']);binding=choice['binding'];source=Path(r['source_path']);source_sha=r['source_sha256']
    for path,sha in [(source,source_sha),(choice['source_path'],choice['source_sha256']),(choice['view_path'],choice['view_sha256']),(binding['source_dat'],binding['source_dat_sha256'])]:
        if digest(path)!=sha:raise ValueError('Reference preparation input changed')
    req=json.loads(Path(binding['request']).read_text())
    if digest(req['catalog'])!=req['catalog_sha256']:raise ValueError('Magnetic catalogue changed')
    g=G.G2Project(str(source));h=g.histogram(0)
    if len(g.histograms())!=1:raise ValueError('This local transfer adapter supports a single verified TOF histogram')
    prior=G.G2Project(choice['source_path']);old=prior.histogram(0)
    # Identical observations establish which archived reference is being polished.
    for index in (0,1,2):
        if not np.array_equal(h.data['data'][1][index],old.data['data'][1][index]):raise ValueError('Refinement output no longer contains the original reference observations')
    labels=[s['label'] for s in req['sites']]
    def parent(project):
        matches=[p for p in project.phases() if p.data['General']['Type']=='nuclear' and all(label in {a.label for a in p.atoms()} for label in labels)]
        if len(matches)!=1:raise ValueError('Magnetic parent was removed or is ambiguous; create a new catalogue')
        return matches[0]
    phase=parent(g);fingerprint=setting_fingerprint(phase.data['General']['SGData'])
    if fingerprint!=setting_fingerprint(parent(prior).data['General']['SGData']):raise ValueError('Parent symmetry changed; create a new magnetic catalogue')
    difc=binding['display_export_difc']
    x,y,w,calc,bg,*_=[np.asarray(a).copy() for a in h.data['data'][1]]
    valid=(w>0)&np.isfinite(y)&(x>=h.data['Limits'][1][0])&(x<=h.data['Limits'][1][1])
    for lo,hi in h.data['Limits'][2:]:valid&=~((x>=lo)&(x<=hi))
    if not valid.any() or not np.isfinite(x[valid]).all() or not np.isfinite(w[valid]).all():raise ValueError('Reference has no finite valid fitted observations')
    if float(np.sum(w[valid]*y[valid]**2))<=0:raise ValueError('Reference Rwp is undefined for zero weighted intensity')
    for ph in g.phases():
        hap=ph.data['Histograms'][h.name]
        for key in ('Size','Mustrain'):
            model=hap.get(key)
            if model and model[0] in ('isotropic','uniaxial'):
                values=np.asarray(model[1][:1 if model[0]=='isotropic' else 2],float)
                if not np.isfinite(values).all() or np.any(values<=0 if key=='Size' else values<0):raise ValueError('Unphysical '+key+' parameters in selected reference')
    native=NativeReducedProfile(source);direct=native.profile(native.x0)
    if len(native.parts)!=1:raise ValueError('Expected one reference histogram')
    _,sl=native.parts[0];error=float(np.max(abs(direct-calc[sl]))/max(np.max(abs(calc[sl])),1e-12))
    saved_curve_error_sigma=float(np.max(abs(direct-calc[sl])*np.sqrt(np.maximum(w[sl],0))))
    # Native refinement can leave last-cycle arrays slightly behind the saved
    # parameters. Recompute rather than plotting those arrays as the new fit.
    # A material discrepancy requires investigation, not automatic repair.
    if saved_curve_error_sigma>.01:raise ValueError(f'Saved refinement curve differs from current native parameters by {saved_curve_error_sigma:.4g} sigma; inspect the output')
    calc[sl]=direct
    rwp=float(100*np.sqrt(np.sum(w[valid]*(y[valid]-calc[valid])**2)/np.sum(w[valid]*y[valid]**2)))
    with np.load(choice['view_path'],allow_pickle=False) as prior_view:
        view={k:prior_view[k].copy() for k in prior_view.files}
    view.update(x=x,y=y,sigma=np.where(w>0,1/np.sqrt(np.maximum(w,1e-300)),0),calc=calc,background=bg,valid=valid)
    from .profile_diagnostics import compare_widths
    centers=[p['d_A'] for p in choice['verification'].get('peaks',[])]
    if not centers:
        from scipy.signal import find_peaks
        peaks,_=find_peaks(np.where(valid,calc-bg,0),distance=30)
        centers=[float(x[i]/difc) for i in sorted(peaks,key=lambda i:calc[i]-bg[i],reverse=True)[:3]]
    width_review=compare_widths(view,difc,centers)
    phase_review=[dict(name=ph.name,cell=copy.deepcopy(ph.data['General']['Cell']),profile={k:copy.deepcopy(ph.data['Histograms'][h.name][k]) for k in ('Size','Mustrain')}) for ph in g.phases()]
    view_path=out/'verified.npz';np.savez_compressed(view_path,**view)
    # Freeze every nonmagnetic parameter before the explicit target transfer.
    g=G.G2Project(str(source),newgpx=str(out/'ordered-live.gpx'));h=g.histogram(0);phase=parent(g)
    normalize_flags(g);freeze(g);instrument=copy.deepcopy(h.data['Instrument Parameters'])
    raw=stable_read(binding['source_dat'])
    import hashlib
    if hashlib.sha256(raw).hexdigest()!=binding['source_dat_sha256']:raise ValueError('Target snapshot changed during preparation')
    live=parse_live_ascii(raw);spacing=np.diff(live.x)
    if not len(spacing) or not np.allclose(spacing,spacing[0],rtol=1e-6,atol=1e-10):raise ValueError('This archive adapter requires verified uniform d bins')
    step=spacing[0];a=density_to_tof(live,d_edges=np.r_[live.x[0]-step/2,live.x+step/2],export_difc=difc)
    limits=copy.deepcopy(h.data['Limits']);limits[0]=[float(a['x'][0]),float(a['x'][-1])]
    limits[1]=[max(limits[1][0],limits[0][0]),min(limits[1][1],limits[0][1])]
    transfer_limits=[max(limits[1][0],.7*difc),min(limits[1][1],1.65*difc)]
    if transfer_limits[0]>=transfer_limits[1]:raise ValueError('Edited fit limits exclude the reviewed transfer interval d 0.7–1.65 Å')
    h.data['data'][1]=[a['x'],a['y'],a['weights'],np.zeros_like(a['x']),np.zeros_like(a['x']),np.zeros_like(a['x'])]
    h.data['data'][0]['Dummy']=False;h.data['Limits']=copy.deepcopy(limits);h.data['Limits'][1]=transfer_limits
    for ph in g.phases():ph.set_HAP_refinements({'Scale':True})
    h.set_refinements({'Background':{'refine':True}});g.set_Controls('cycles',12)
    transfer=checked_native_refine(g)
    if not transfer['qualified']:raise ValueError('Native scale/background transfer failed its numerical checks')
    if any(not np.isfinite(ph.data['Histograms'][h.name]['Scale'][0]) or ph.data['Histograms'][h.name]['Scale'][0]<0 for ph in g.phases()):raise ValueError('Native transfer produced a negative or nonfinite phase scale')
    normalize_flags(g);freeze(g);h.data['Limits']=limits;g.save();refresh_saved_profiles(g);g.save()
    if h.data['Instrument Parameters']!=instrument:raise ValueError('Instrument parameters changed during transfer')
    rows=[];catalog=json.loads(Path(req['catalog']).read_text())
    for c in catalog['candidates']:
        try:
            _,audit=preflight_magnetic_sites(phase.data,dict(c,candidate_id=c['id']),magnetic_sites=labels,site_policy='all',expected_parent_setting=fingerprint)
            rows.append(dict(id=c['id'],name=c['magnetic_space_group'],compatible=audit['compatible'],dimensions=sum(s['independent_moment_count'] for s in audit['sites']),sites=audit['sites']))
        except Exception as exc:rows.append(dict(id=c['id'],error=str(exc)))
    preflight=out/'candidate-preflight.json';preflight.write_text(json.dumps(rows,indent=2))
    req.update(source_gpx=str(out/'ordered-live.gpx'),source_sha256=digest(out/'ordered-live.gpx'),preflight=str(preflight),preflight_sha256=digest(preflight),reference_revision=f"PG3_{choice['run']}:{source_sha}")
    request=out/'request.json';request.write_text(json.dumps(req,indent=2))
    binding.update(reference_gpx=str(source),reference_sha256=source_sha,source_gpx=req['source_gpx'],source_sha256=req['source_sha256'],request=str(request),description=f"Refined nuclear checkpoint from run {choice['run']}; original-data Rwp {rwp:.2f}%. Native preparation complete; scientific adequacy and transfer remain unqualified. No counting forecasts.")
    binding['review']=[dict(label='Origin',value=f"Refinement job {r['job_id']} of {choice['id']}"),dict(label='Original fit',value=f'Rwp {rwp:.4f}%; inspect the residuals below'),dict(label='Transfer',value='Cell, atoms, instrument and sample profile retained; only scale/background refit in d 0.7–1.65 Å. Limits and exclusions preserved.'),dict(label='Checkpoint hash',value=source_sha)]
    vfy=copy.deepcopy(choice['verification']);vfy.update(path=str(source),sha256=source_sha,rwp_recomputed=rwp,native_saved_curve_relative_error=error,saved_curve_error_sigma=saved_curve_error_sigma,curve_source='Fresh native evaluation at saved parameters',peaks=width_review,phases=phase_review)
    rs=copy.deepcopy(choice['refinement_source']);rs.update(id='polished-'+source_sha[:24],source_path=str(source),source_sha256=source_sha,record_uid='polished-'+source_sha[:24],collection_name=source.name,base_rwp=rwp,published_sha256=source_sha,internal_source_sha256=source_sha)
    choice.update(id='refined-'+source_sha[:12]+'-'+req['source_sha256'][:12],title=f"Refined {choice['role']} · run {choice['run']} · Rwp {rwp:.2f}%",source_path=str(source),source_sha256=source_sha,view_path=str(view_path),view_sha256=digest(view_path),verification=vfy,refinement_source=rs,parent_checkpoint=r['base_option']['id'],refinement_job=r['job_id'])
    if digest(source)!=source_sha:raise ValueError('Original refinement output was modified')
    (out/'transfer-receipt.json').write_text(json.dumps(dict(qualified=False,numerical_checks=transfer,source_unchanged=True,instrument_preserved=True,forecast_enabled=False),indent=2))
    (out/'option.json').write_text(json.dumps(choice,indent=2));return choice

if __name__=='__main__':prepare(sys.argv[1])
