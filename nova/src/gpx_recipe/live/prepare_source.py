"""Prepare one reviewed POWGEN snapshot from an immutable nuclear checkpoint.

Discovery belongs to the existing model workflow. This worker only validates its
catalogue/site choices, transfers the declared observation grid, and freezes the
reviewed response. Nothing here establishes empirical counting uncertainty.
"""
from __future__ import annotations
import copy,json
from pathlib import Path
import numpy as np
from .contracts import file_hash
from .io import atomic_json
from .snapshot import parse_live_ascii,density_to_tof


def validate_preparation(request):
    for field in ('source_gpx','catalog','observation_source'):
        expected=request.get(field.replace('_gpx','')+'_sha256')
        if not expected or file_hash(request[field])!=expected:raise ValueError('Live preparation input changed: '+field)
    mapping=request.get('mapping') or {}
    if mapping.get('mapping_kind')!='user_reviewed' or mapping.get('snapshot_sha256')!=request['observation_source_sha256']:
        raise ValueError('Review the producer mapping for this snapshot first')
    if mapping.get('source_gpx_sha256')!=request['source_sha256'] or not mapping.get('producer_config_sha256'):
        raise ValueError('Producer mapping and nuclear source disagree')
    if request.get('source_revision','').split(':')[-1]!=request['observation_source_sha256'] or request.get('reference_revision','').split(':')[-1]!=request['source_sha256']:
        raise ValueError('Preparation revision does not match the verified source files')
    if not request.get('parent_phase') or not request.get('sites'):raise ValueError('Review a magnetic parent and sites in Model exploration')
    if type(request.get('refine_transfer',False)) is not bool:raise ValueError('Transfer freedom must be explicit')
    return request


def prepare(request,root):
    from GSASII import GSASIIscriptable as G
    from gpx_recipe.execution import freeze
    from gpx_recipe.symmetry.native_preparation import normalize_flags
    from gsas_magnetic_preflight import setting_fingerprint,preflight_magnetic_sites
    from .native_validation import checked_native_refine,refresh_saved_profiles
    from .gpx_index import install_fresh_gpx_index
    validate_preparation(request);root=Path(root);dest=root/'prepared';dest.mkdir()
    install_fresh_gpx_index();g=G.G2Project(request['source_gpx'],newgpx=str(dest/'source.gpx'))
    if len(g.histograms())!=1:raise ValueError('Select a single independently calibrated POWGEN histogram')
    h=g.histogram(0)
    if h.InstrumentParameters['Type'][0]!='PNT':raise ValueError('A TOF powder reference is required for this POWGEN exporter')
    if any(p.data['General']['Type']!='nuclear' for p in g.phases()):raise ValueError('The reviewed reference must contain nuclear phases only')
    parent=g.phase(request['parent_phase'])
    catalog=json.loads(Path(request['catalog']).read_text())
    from gpx_recipe.symmetry.chemistry import validate_sites
    cx,ct,_,_=parent.data['General']['AtomPtrs']
    validate_sites([dict(label=a[ct-1],type=a[ct],occupancy=a[cx+3]) for a in parent.data['Atoms']],request['sites'])
    fingerprint=setting_fingerprint(parent.data['General']['SGData'])
    if catalog.get('parent_setting_sha256')!=fingerprint:raise ValueError('Magnetic catalogue belongs to a different nuclear setting')
    if catalog.get('k')!=request.get('k'):raise ValueError('Magnetic catalogue has a different propagation vector')
    source_sha=file_hash(request['source_gpx']);instrument=copy.deepcopy(h.data['Instrument Parameters'])
    original_limits=copy.deepcopy(h.data['Limits']);normalize_flags(g);freeze(g)
    # Compare, do not assume, the source's saved curve and recomputed model.
    saved_curve=np.asarray(h.data['data'][1][3],float).copy()
    refresh_saved_profiles(g)
    x,y,w,calc,bg,*_=[np.asarray(np.ma.filled(a,0.)).copy() for a in h.data['data'][1]]
    valid=(w>0)&(x>=original_limits[1][0])&(x<=original_limits[1][1])
    for lo,hi in original_limits[2:]:valid&=~((x>=lo)&(x<=hi))
    if not valid.any() or not np.sum(w[valid]*y[valid]**2)>0:raise ValueError('Reference has no fitted positive-weight observations')
    saved_curve_relative_error=float(np.max(abs(saved_curve-calc))/max(float(np.max(abs(calc))),1e-12))
    saved_curve_error_sigma=float(np.max(abs(saved_curve[valid]-calc[valid])*np.sqrt(w[valid])))
    original_rwp=float(100*np.sqrt(np.sum(w[valid]*(y[valid]-calc[valid])**2)/np.sum(w[valid]*y[valid]**2)))
    live=parse_live_ascii(Path(request['observation_source']).read_bytes());spacing=np.diff(live.x)
    if not np.allclose(spacing,spacing[0],rtol=1e-6,atol=1e-10):raise ValueError('The reviewed exporter requires uniform d bins')
    factor=float(request['mapping']['export_geometry']['microseconds_per_angstrom'])
    transformed=density_to_tof(live,d_edges=np.r_[live.x[0]-spacing[0]/2,live.x+spacing[0]/2],export_difc=factor)
    np.savez_compressed(dest/'reference-view.npz',x=x,y=y,sigma=np.where(w>0,1/np.sqrt(np.maximum(w,1e-300)),0),calc=calc,background=bg,valid=valid,
                        live_x=transformed['x'],live_scaled=transformed['y'],reviewed_live_conversion=np.array(True))
    limits=copy.deepcopy(original_limits);limits[0]=[float(transformed['x'][0]),float(transformed['x'][-1])]
    limits[1]=[max(limits[0][0],original_limits[1][0]),min(limits[0][1],original_limits[1][1])]
    if limits[1][0]>=limits[1][1]:raise ValueError('Nuclear reference and live snapshot have no common fitted range')
    h.data['data'][1]=[transformed['x'],transformed['y'],transformed['weights'],np.zeros_like(live.x),np.zeros_like(live.x),np.zeros_like(live.x)]
    h.data['data'][0]['Dummy']=False;h.data['Limits']=limits
    transfer={'performed':False,'qualified':False}
    if request.get('refine_transfer'):
        for phase in g.phases():phase.set_HAP_refinements({'Scale':True})
        h.set_refinements({'Background':{'refine':True}});g.set_Controls('cycles',12)
        transfer=checked_native_refine(g)
        if not transfer['qualified']:raise ValueError('Declared scale/background transfer did not converge')
    normalize_flags(g);freeze(g);g.data.setdefault('Covariance',{})['data']={};g.save();refresh_saved_profiles(g);g.save()
    if h.data['Instrument Parameters']!=instrument:raise ValueError('Instrument changed while preparing the fixed response')
    if file_hash(request['source_gpx'])!=source_sha:raise ValueError('Original nuclear checkpoint changed')
    preflight=[]
    for candidate in catalog['candidates']:
        try:
            _,audit=preflight_magnetic_sites(parent.data,dict(candidate,candidate_id=candidate['id']),
                magnetic_sites=[s['label'] for s in request['sites']],site_policy='all',expected_parent_setting=fingerprint)
            preflight.append(dict(id=candidate['id'],compatible=audit['compatible'],dimensions=sum(s['independent_moment_count'] for s in audit['sites']),sites=audit['sites']))
        except Exception as exc:preflight.append(dict(id=candidate['id'],error=str(exc)))
    atomic_json(dest/'preflight.json',preflight)
    inference={key:request[key] for key in ('k','sites','source_revision','reference_revision')}
    inference.update(source_gpx='prepared/source.gpx',source_sha256=file_hash(dest/'source.gpx'),catalog='inputs/catalog.json',catalog_sha256=request['catalog_sha256'],
        preflight='prepared/preflight.json',preflight_sha256=file_hash(dest/'preflight.json'),lower=float(request.get('lower',0)),upper=float(request.get('upper',9)),
        particles=int(request.get('particles',256)),workers=int(request.get('workers',1)),maximum_dimensions=int(request.get('maximum_dimensions',32)),
        candidate_seconds=float(request.get('candidate_seconds',45)),preparation_timeout_seconds=float(request.get('preparation_timeout_seconds',600)),
        excluded_parent_space_groups=request.get('excluded_parent_space_groups',[]))
    atomic_json(dest/'inference-request.json',inference)
    return dict(request='prepared/inference-request.json',reference_view='prepared/reference-view.npz',
        source_gpx='prepared/source.gpx',source_sha256=inference['source_sha256'],reference_sha256=source_sha,
        source_dat='inputs/source.dat',source_dat_sha256=request['observation_source_sha256'],display_export_difc=factor,
        original_reference_rwp=original_rwp,native_saved_curve_relative_error=saved_curve_relative_error,saved_curve_error_sigma=saved_curve_error_sigma,transfer_limits_tof=limits[1],transfer=transfer,
        scientifically_qualified=False,forecast_enabled=False)
