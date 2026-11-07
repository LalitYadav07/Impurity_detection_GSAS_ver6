"""Exact affine rebinding of qualified native structural starting curves.

Recorded acquisition metadata is excluded by the common forward identity.
Only the single-phase intensity normalization and existing additive polynomial
background values are then removed. Cells, modes, chemistry, masks,
instrument profiles and background definitions remain dependencies. This is
not permission to reuse arbitrary uploaded arrays or a changed geometry.
"""
import copy
import hashlib
import json
import re

import numpy as np

from .structural_identity import _physics_record

SCHEMA='radar-native-structural-affine-nuisance/v1'


def validate_context(value):
    if (not isinstance(value,dict) or set(value)!={'schema','shape_sha256','scale'} or value['schema']!=SCHEMA
            or not isinstance(value['shape_sha256'],str) or not re.fullmatch('[a-f0-9]{64}',value['shape_sha256'])
            or type(value['scale']) not in (int,float) or not np.isfinite(value['scale']) or value['scale']<=1e-10):
        raise ValueError('Invalid native structural nuisance context')
    return value


def affine_context(source):
    """Return a conservative shape identity and the positive native scale.

    Missing/unsupported affine structure disables this accelerator. Existing
    full-context native calculations remain available without narrowing the
    scientific model. Never mutate the live native dictionaries to normalize.
    """
    native=getattr(getattr(source,'fast',None),'native',None)
    if native is None:return None
    native._check()
    histograms,phases=native.histo_phases[:2]
    # errRefine adds scalar diagnostics to the histogram dictionary. Count
    # actual selected powder parts, never its mutable metadata keys.
    if len(native.parts)!=1 or len(phases)!=1:return None
    histogram=histograms[source.fast.slow.hist.name];phase=phases[source.fast.slow.phase.name]
    hfx=f":{histogram['hId']}:";phfx=f"{phase['pId']}:{histogram['hId']}:"
    general=phase['General']
    if (general.get('Type')!='nuclear' or general.get('Modulated') or general.get('doPawley')
            or any(v.get('LeBail') for v in phase['Histograms'].values())
            or native.controls.get(hfx+'histType') not in {'PNC','PNT'}
            or float(native.parms.get(phfx+'Extinction',float('nan')))!=0.
            or native.controls.get(hfx+'Exclude')
            or any(np.ma.getmaskarray(a).any() for a in histogram['Data'][:3])):
        return None
    record=copy.deepcopy(_physics_record(source))
    parameters=record['parameters'];key=':bank:Scale'
    scale=parameters.get(key);phase_scale=parameters.get('primary:bank:Scale')
    if any(type(v) not in (int,float) or not np.isfinite(v) or v<=1e-10 for v in (scale,phase_scale)):return None
    background=record['background']
    if (not isinstance(background,list) or len(background)!=2 or not isinstance(background[0],list)
            or len(background[0])<4 or type(background[0][2]) is not int
            or len(background[0])!=background[0][2]+3 or not isinstance(background[1],dict)
            or background[1].get('nDebye',0) or background[1].get('nPeaks',0)):
        return None
    sample_scale=record['sample'].get('Scale')
    if not isinstance(sample_scale,list) or len(sample_scale)!=2:return None
    phase_hist=record['phases'][0].get('histograms',{}).get('bank',{})
    hap_scale=phase_hist.get('Scale')
    if not isinstance(hap_scale,list) or len(hap_scale)!=2:return None
    parameters[key]=1.;sample_scale[0]=1.
    parameters['primary:bank:Scale']=1.;hap_scale[0]=1.
    for index in range(background[0][2]):
        key=f':bank:Back;{index}'
        if key not in parameters:return None
        parameters[key]=0.;background[0][index+3]=0.
    encoded=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    combined=float(scale)*float(phase_scale)
    if not np.isfinite(combined) or combined<=1e-10:return None
    return validate_context(dict(schema=SCHEMA,shape_sha256=hashlib.sha256(encoded).hexdigest(),scale=combined))


def reference(source):
    context=affine_context(source)
    if context is None:return None
    _,_,background,_,_=source.snapshot(np.zeros(len(source.radii)))
    background=np.asarray(background,float).copy()
    if background.shape!=source.fast.x.shape or not np.isfinite(background).all():
        raise ValueError('Native background differs from the complete structural grid')
    return context,background


def rebind(curves,background,stored,current,current_background):
    """Rebind in place under a conservative floating-point error check."""
    validate_context(stored);validate_context(current)
    if (stored['shape_sha256']!=current['shape_sha256']
            or background.shape!=current_background.shape or background.shape!=(curves.shape[1],)
            or not np.isfinite(background).all()):
        raise ValueError('Prepared structural nuisance factorization is incompatible')
    ratio=current['scale']/stored['scale'];maximum=0.
    old_norm=float(np.linalg.norm(background));new_norm=float(np.linalg.norm(current_background))
    for row in curves:
        original_norm=float(np.linalg.norm(row))
        row-=background;row*=ratio
        signal_norm=float(np.linalg.norm(row));row+=current_background
        # Cancellation against a large original background must not conceal
        # an inaccurate foreground. Reject unsafe rebinding rather than clip
        # intensities, change tolerances or accept a few agreeing strong peaks.
        error=32*np.finfo(float).eps*((original_norm+old_norm)*abs(ratio)+float(np.linalg.norm(row))+new_norm)
        relative=error/max(signal_norm,1e-30);maximum=max(maximum,relative)
        if not np.isfinite(row).all() or not np.isfinite(relative) or relative>1e-11:
            raise ValueError('Prepared structural foreground loses precision during nuisance rebinding')
    return dict(method='native single-phase intensity scale and additive polynomial background',
        scale_ratio=ratio,maximum_relative_roundoff_bound=maximum,
        stored_shape_sha256=stored['shape_sha256'],current_shape_sha256=current['shape_sha256'],
        current_scale=current['scale'],reference_scale=stored['scale'],observations_used=False)
