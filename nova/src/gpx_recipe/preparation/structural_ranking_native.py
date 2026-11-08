"""Native GPX admission for prepared structural ranking; no candidate pruning.

One isolated calculation per acquisition can prioritize the remaining native
work. It imports the exact prepared parent and bank, preserves measured data,
and recalculates background without running another diffraction refinement.
"""
import json
from functools import partial
from pathlib import Path
import time

import numpy as np

from ..execution import write_json
from ..fit_domain import native_powder_fit_mask
from ..symmetry.compiler import confined,file_sha,package_signature,verify_inputs
from ..symmetry.contracts import validate_protocol
from .structural_ranking_result import unavailable,validate


def _sites(phase):
    cx,ct,_,cia=phase.data['General']['AtomPtrs']
    sites={a[ct-1]:dict(type=a[ct],xyz=np.asarray(a[cx:cx+3],float),occupancy=float(a[cx+3]),
        adp_type=a[cia],adp=np.asarray(a[cia+1:cia+8],float)) for a in phase.data['Atoms']}
    if len(sites)!=len(phase.data['Atoms']):raise ValueError('Native parent atom labels must be unique')
    return sites


def require_unconstrained_project(project):
    """CIF-only preparation does not cover extra user parameter mappings."""
    constraints=project.data.get('Constraints',{})
    constraints=constraints.get('data',constraints)
    if any(constraints.get(key) for key in ('Hist','HAP','Phase','Global')):
        raise ValueError('Prepared ranking is not qualified for additional user constraints')
    if any(any(phase.data.get('RBModels',{}).values()) for phase in project.phases()):
        raise ValueError('Prepared ranking is not qualified for assigned rigid-body models')


def _sample_values(parameters):
    ignored={'Scale','ranId','InstrName','Temperature','Pressure','Time'}
    result={k:v for k,v in parameters.items() if k not in ignored}
    if 'Contrast' in result:
        # Contrast is a pair of numerical scattering contrasts, not [value,
        # flag]. Older transfer code encoded a numerical zero/one as bool.
        # Compare both physical values, including that legacy encoding;
        # the generic native audit removes boolean refinement switches.
        contrast=np.asarray(result['Contrast'],dtype=float)
        if contrast.shape!=(2,) or not np.isfinite(contrast).all():
            raise ValueError('Finite normal and anomalous sample contrasts required')
        result['Contrast']=contrast.tolist()
    return result


def require_compatible(parent,histogram,reference,reference_histogram,*,maximum_metric):
    """Compare native physical context, allowing only sampled parent strain.

    Scales and background are observation-specific nuisances. Acquisition
    temperature/pressure labels do not alter the native scattering context.
    Geometry/ADP and profile changes need their own training coverage.
    """
    from gsas_magnetic_preflight import setting_fingerprint
    from ..symmetry.native_audit import same
    from .structural_domain import relative_metric_norm
    if (type(maximum_metric) not in (int,float) or not np.isfinite(maximum_metric)
            or not 0<=maximum_metric<=10):
        raise ValueError('A finite nonnegative training-only parent metric bound is required')
    a,b=parent.data['General'],reference.data['General']
    if (a['Type']!='nuclear' or a.get('Modulated') or a.get('doPawley')
            or setting_fingerprint(a['SGData'])!=setting_fingerprint(b['SGData'])):
        raise ValueError('Prepared parent symmetry or setting differs from this native GPX')
    if not same(a.get('Isotope'),b.get('Isotope')):
        raise ValueError('Prepared neutron isotope selections differ from this native GPX')
    left,right=_sites(parent),_sites(reference)
    if set(left)!=set(right):raise ValueError('Prepared parent atom labels differ from this native GPX')
    for key,x in left.items():
        y=right[key];delta=x['xyz']-y['xyz'];delta-=np.rint(delta)
        if (x['type']!=y['type'] or x['adp_type']!=y['adp_type']
                or not np.isclose(x['occupancy'],y['occupancy'],rtol=0,atol=1e-8)
                or not np.allclose(delta,0,rtol=0,atol=2e-6)
                or not np.allclose(x['adp'],y['adp'],rtol=0,atol=1e-6)):
            raise ValueError('Prepared chemistry, parent positions or ADPs differ at '+str(key))
    distance=relative_metric_norm(b['Cell'][1:7],a['Cell'][1:7])
    if distance>maximum_metric+1e-9:
        raise ValueError('Current parent lattice exceeds the training-only reciprocal-metric envelope '
            f'(current change {distance:.6g}; trained maximum {maximum_metric:.6g}). '
            'Generate training examples over the wider parent-cell range before using this library.')
    if not same(histogram.InstrumentParameters,reference_histogram.InstrumentParameters):
        raise ValueError('Prepared instrument bank or profile coefficients differ from this native histogram')
    hap=parent.data['Histograms'][histogram.name]
    if hap.get('LeBail'):raise ValueError('Le Bail intensities do not define a structural ranking parent')
    ref=reference.data['Histograms'][reference_histogram.name]
    ignore={'Scale','Use','hId','Histogram','Fix FXU'}
    if not same({k:v for k,v in hap.items() if k not in ignore},{k:v for k,v in ref.items() if k not in ignore}):
        raise ValueError('Sample broadening, texture or histogram-phase corrections differ from preparation')
    if not same(_sample_values(histogram.data['Sample Parameters']),
                _sample_values(reference_histogram.data['Sample Parameters'])):
        raise ValueError('Physical sample corrections differ from preparation')
    if (not np.isfinite(hap['Scale'][0]) or hap['Scale'][0]<=0
            or not np.isfinite(histogram.data['Sample Parameters']['Scale'][0])
            or histogram.data['Sample Parameters']['Scale'][0]<=0):
        raise ValueError('Structural ranking needs positive current phase and histogram scales')
    return distance


def observed_features(source,histogram,*,include_reference=False):
    """Original observations and mask; freshly evaluated native background.

    getBackground uses the same native histogram slice as errRefine. It does
    not evaluate structure factors, overwrite a GPX, invent counts or trust a
    potentially stale saved calculated/background array.
    """
    from ..live.native_profile import NativeReducedProfile
    if type(include_reference) is not bool:raise ValueError('Explicit native parent-reference request required')
    forward=NativeReducedProfile(source)
    native=forward.histo_phases[0][histogram.name]
    sl=next(sl for name,sl in forward.parts if name==histogram.name)
    prefix=f":{native['hId']}:"
    arrays=histogram.data['data'][1]
    x,y,w=(np.asarray(np.ma.getdata(a),float) for a in arrays[:3])
    valid=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0)
    for a in arrays[:3]:valid &= ~np.ma.getmaskarray(a)
    valid &= native_powder_fit_mask(x,histogram.data['Limits'][1],histogram.data['Limits'][2:])
    native_x=native['Data'][0][sl]
    if not np.array_equal(np.asarray(np.ma.getdata(native_x)),x[sl]):
        raise ValueError('Native background and observation grids differ')
    background,_=forward.math.G2pwd.getBackground(prefix,forward.parms,forward.controls[prefix+'bakType'],
        forward.controls[prefix+'histType'],native_x,native['Background'][1].get('fixback',None))
    background=np.asarray(np.ma.getdata(background),float)
    d=np.asarray(forward.lattice.Pos2dsp(histogram.InstrumentParameters,x[sl]),float)
    if (background.shape!=x[sl].shape or not np.isfinite(d).all() or np.any(d<=0)):
        raise ValueError('Native background or Q conversion is unavailable on this fit range')
    q=2*np.pi/d;order=np.argsort(q,kind='stable')
    if np.any(np.diff(q[order])<=0):raise ValueError('Native calibration has no unique increasing Q grid')
    result=dict(q=q[order],intensity=y[sl][order],background=background[order],valid=valid[sl][order])
    if include_reference:
        # Recalculate on the actual acquisition grid and current checked
        # parent cell. Do not interpolate an old prepared curve or trust the
        # GPX's saved calculated/background arrays. This runs no refinement
        # and writes neither observations nor the input GPX.
        calculated=forward.profile(forward.x0)
        start=0
        for name,part in forward.parts:
            size=len(forward.histo_phases[0][name]['Data'][0][part])
            if name==histogram.name:
                current=calculated[start:start+size]
                if current.shape!=background.shape or not np.isfinite(current).all():
                    raise ValueError('The native parent reference has an invalid acquisition profile')
                result['reference']=dict(q=q[order].copy(),intensity=current[order].copy(),
                    background=background[order].copy(),valid=valid[sl][order].copy())
                break
            start+=size
        else:raise ValueError('The native parent histogram is absent from its calculated profile')
    return result


def rank_gpx(input_root,protocol,destination,*,source_override=None,observation=None):
    """Read one exact source revision and attached data-only training context."""
    from .structural_hints import read
    from . import structural_ranking_hints as hints
    from .reference import create_reference
    from .structural_domain import maximum_parent_metric
    from ..structural_live.native import modules,require_nuclear_runtime
    from ..symmetry.worker import require_runtime,runtime_identity
    from gsas_magnetic_preflight import setting_fingerprint
    protocol=validate_protocol(protocol);verify_inputs(protocol,input_root)
    if protocol['kind']!='structural' or protocol['implementation_sha256']!=package_signature():
        raise ValueError('Structural ranking needs the exact captured structural implementation')
    source=confined(input_root,protocol['source']['path'])
    if (source_override is None)!=(observation is None):
        raise ValueError('A live ranking source needs its exact observation binding')
    if observation is not None:
        from .structural_ranking_result import observation_binding
        observation=observation_binding(observation);source=Path(source_override)
        if source.is_symlink() or file_sha(source)!=observation['calculation_source_sha256']:
            raise ValueError('The native structural ranking observation changed')
    fallback=partial(unavailable,protocol,observation=observation)
    began=time.monotonic();out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False)
    checks=[];snapshot=None;features=None;support=None
    result=fallback('No compatible prepared structural ranking is attached')
    try:
        document=read(input_root,protocol)
        if document is None:return result
        snapshot=document['snapshot_sha256']
        entries=document.get('ranking_models',{}).get('entries',[])
        if not entries:
            result=fallback('No prepared structural ranking model is attached',snapshot=snapshot)
            return result
        G,*_=modules();require_nuclear_runtime()
        require_runtime(runtime_identity(),protocol['magnetic_function_sha256'])
        project=G.G2Project(gpxfile=str(source))
        require_unconstrained_project(project)
        parent=project.phase(protocol['parent_phase']);histogram=project.histogram(protocol['histogram'])
        if setting_fingerprint(parent.data['General']['SGData'])!=protocol['parent_setting_sha256']:
            raise ValueError('Current nuclear parent setting differs from the reviewed structural protocol')
        used=[p for p in project.phases() if p.data.get('Histograms',{}).get(histogram.name,{}).get('Use')]
        if len(used)!=1 or used[0].name!=parent.name:
            raise ValueError('Prepared structural ranking is not qualified for additional active phases')
        for row in entries:
            try:
                files=hints.checked_files(input_root,row,protocol)
                metadata=json.loads(files['model'].read_text());maximum=maximum_parent_metric(metadata)
                prepared=create_reference(files['parent'],files['instrument'],out/'reference'/row['entry'],**row['grid'])
                reference=G.G2Project(gpxfile=prepared['path'])
                require_compatible(parent,histogram,reference.phase(prepared['parent_phase']),
                    reference.histogram(prepared['histogram']),maximum_metric=maximum)
                from .structural_attention_weights import SCHEMA as THREE_CURVE_SCHEMA
                paired=metadata.get('schema')==THREE_CURVE_SCHEMA
                if features is None:features={}
                if paired not in features:features[paired]=observed_features(source,histogram,include_reference=paired)
                ranked=hints.rank(input_root,protocol,**features[paired],entry_id=row['entry'],
                    parent_sha256=row['files']['parent']['sha256'],instrument_sha256=row['files']['instrument']['sha256'])
                if not ranked['ranking_available']:
                    support=ranked.get('support')
                    checks.append(dict(entry=row['entry'],status='inference_unavailable',
                        reason=ranked.get('reason','Prepared model inference is unavailable')[:600]));continue
                result=dict(fallback('Prepared simulations prioritize native work; every declared candidate remains available',snapshot=snapshot),
                    ranking_available=True,candidate_order=ranked['candidate_order'],families=ranked['families'],
                    entry_id=row['entry'],bank=row['bank'],support=ranked['support'])
                if 'adaptive_work_set' in ranked:
                    from .structural_ranking_result import attach_adaptive
                    result=attach_adaptive(result,ranked['adaptive_work_set'])
                if 'mode_estimates' in ranked:
                    from .structural_ranking_result import attach_modes
                    result=attach_modes(result,ranked['mode_estimates'])
                checks.append(dict(entry=row['entry'],status='compatible',
                    reason='Native parent, bank profile, sites/ADPs and sample corrections matched within the trained lattice envelope'))
                break
            except (OSError,ValueError,ImportError,KeyError,TypeError,IndexError) as error:
                checks.append(dict(entry=row['entry'],status='incompatible',reason=str(error)[:600]))
        else:
            result=fallback('Prepared ranking is unavailable for this scan; the complete native search is retained',
                               snapshot=snapshot,support=support)
    except (OSError,ValueError,ImportError,KeyError,TypeError,IndexError) as error:
        result=fallback('Native ranking admission did not complete: '+str(error)[:600],snapshot=snapshot)
    finally:
        result.update(checks=checks,seconds=time.monotonic()-began)
        validate(result,protocol,observation=observation)
        write_json(out/'ranking.json',result)
        verify_inputs(protocol,input_root)
        if observation is not None and file_sha(source)!=observation['calculation_source_sha256']:
            raise ValueError('The native structural ranking observation changed during inference')
    return result


if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]).resolve()
    protocol=json.loads((root/'protocol.json').read_text())
    rank_gpx(root,protocol,Path(sys.argv[2]))
