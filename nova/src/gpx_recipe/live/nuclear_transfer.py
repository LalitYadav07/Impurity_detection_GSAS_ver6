"""Shared nuclear-reference transfer to a reviewed POWGEN snapshot grid.

The model-specific caller reviews its own symmetry/site/candidate assumptions.
This operation preserves the supplied reduced observations and instrument and
performs only the explicitly declared scale/background transfer. It creates no
posterior, candidate ranking, structural solution or counting recommendation.
"""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import numpy as np
from .contracts import file_hash
from .snapshot import parse_live_ascii, stable_read, uniform_density_to_tof


def validate_transfer(request):
    for field, checksum in [('source_gpx','source_sha256'),
                            ('observation_source','observation_source_sha256')]:
        if not request.get(checksum) or file_hash(request[field])!=request[checksum]:
            raise ValueError('Live preparation input changed: '+field)
    mapping=request.get('mapping') or {}
    if mapping.get('mapping_kind')!='user_reviewed' or mapping.get('snapshot_sha256')!=request['observation_source_sha256']:
        raise ValueError('Review the producer mapping for this snapshot first')
    if mapping.get('source_gpx_sha256')!=request['source_sha256'] or not mapping.get('producer_config_sha256'):
        raise ValueError('Producer mapping and nuclear source disagree')
    if request.get('source_revision','').split(':')[-1]!=request['observation_source_sha256'] or request.get('reference_revision','').split(':')[-1]!=request['source_sha256']:
        raise ValueError('Preparation revision does not match the verified source files')
    if not request.get('parent_phase'):raise ValueError('Review a nuclear parent in Model exploration')
    if type(request.get('refine_transfer',False)) is not bool:raise ValueError('Transfer freedom must be explicit')
    return request


def prepare_nuclear_snapshot(request,root,*,review=None):
    """One native transfer for magnetic and structural acquisition callers.

    ``review`` is a fixed in-process model validator, never uploaded code. The
    source GPX remains immutable; the returned GPX contains the actual captured
    snapshot on its producer-calibrated grid with the original fit exclusions.
    """
    from GSASII import GSASIIscriptable as G
    from gpx_recipe.execution import freeze
    from gpx_recipe.symmetry.native_preparation import normalize_flags
    from .native_validation import checked_native_refine,refresh_saved_profiles
    from .gpx_index import install_fresh_gpx_index
    validate_transfer(request);root=Path(root);dest=root/'prepared';dest.mkdir()
    install_fresh_gpx_index();g=G.G2Project(request['source_gpx'],newgpx=str(dest/'source.gpx'))
    if len(g.histograms())!=1:raise ValueError('Select a single independently calibrated POWGEN histogram')
    h=g.histogram(0)
    if h.InstrumentParameters['Type'][0]!='PNT':raise ValueError('A TOF powder reference is required for this POWGEN exporter')
    if any(p.data['General']['Type']!='nuclear' for p in g.phases()):raise ValueError('The reviewed reference must contain nuclear phases only')
    g.phase(request['parent_phase'])
    if review is not None:review(g)
    source_sha=request['source_sha256']
    original_limits=copy.deepcopy(h.data['Limits']);normalize_flags(g);freeze(g)
    # Recipe outputs can retain released X/Y/Z flags. Freeze those flags as
    # declared, then guard every value against changes during transfer. A flag
    # reset is not an instrument calibration change.
    instrument=copy.deepcopy(h.data['Instrument Parameters'])
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
    raw=stable_read(request['observation_source'])
    if hashlib.sha256(raw).hexdigest()!=request['observation_source_sha256']:
        raise ValueError('Live snapshot changed before native transfer')
    inputs=root/'inputs'
    if inputs.is_symlink():raise ValueError('The captured snapshot directory is not owned')
    inputs.mkdir(exist_ok=True)
    captured=inputs/'source.dat'
    if captured.exists() or captured.is_symlink():
        if (captured.is_symlink() or not captured.is_file()
                or file_hash(captured)!=request['observation_source_sha256']):
            raise ValueError('The captured snapshot path already contains different input')
    else:
        with captured.open('xb') as stream:stream.write(raw)
    live=parse_live_ascii(raw)
    factor=float(request['mapping']['export_geometry']['microseconds_per_angstrom'])
    transformed=uniform_density_to_tof(live,export_difc=factor)
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
    if file_hash(request['observation_source'])!=request['observation_source_sha256']:
        raise ValueError('Live snapshot changed during native transfer')
    return dict(reference_view='prepared/reference-view.npz',source_gpx='prepared/source.gpx',
        source_sha256=file_hash(dest/'source.gpx'),reference_sha256=source_sha,
        source_dat='inputs/source.dat',source_dat_sha256=request['observation_source_sha256'],
        display_export_difc=factor,original_reference_rwp=original_rwp,
        native_saved_curve_relative_error=saved_curve_relative_error,
        saved_curve_error_sigma=saved_curve_error_sigma,transfer_limits_tof=limits[1],transfer=transfer,
        scientifically_qualified=False,forecast_enabled=False)
