"""Native, copy-isolated plot/parameter exports for the integrated NOVA view."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from numbers import Real

import numpy as np

from .snapshot import gsas, snapshot, json_safe
from .advanced import lattice
from .studio_state import extract_gsasii_refinement_state
from .fit_domain import canonical_observations, native_powder_fit_mask, NATIVE_FIT_CONVENTION
from .convergence import convergence_evidence
from .cell_covariance import cell_fit_information


def _finite_array(values):
    a = np.asarray(np.ma.getdata(values), dtype=float)
    mask = np.ma.getmaskarray(values)
    return [float(v) if np.isfinite(v) and not m else None for v, m in zip(a, mask)]


def _optional_rwp(value):
    if isinstance(value,(bool,np.bool_)):return None
    try:number=float(value)
    except (TypeError,ValueError):return None
    return number if np.isfinite(number) and number>=0 else None
def finite_scalar(value):
    """Unrefined native histograms legitimately have no Rwp yet."""
    return float(value) if isinstance(value, Real) and not isinstance(value, bool) and np.isfinite(value) else None


def project_view(path):
    """Keep complete measured samples; the browser never performs GPX extraction."""
    path = Path(path); before = hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='radar-polish-view-') as folder:
        local = Path(folder) / 'project.gpx'; shutil.copy2(path, local)
        context = snapshot(local)
        G = gsas(); project = G.G2Project(gpxfile=str(local)); project.index_ids()
        native=extract_gsasii_refinement_state(local,project,G,max_correlations=10)
        parameters=native["parameter_space"]["parameters"]
        histogram_keys={h['name']:h['key'] for h in context['histograms']}
        histograms = []
        for h in project.histograms():
            data = h.data['data'][1]
            x = np.asarray(np.ma.getdata(data[0]), dtype=float)
            d = np.asarray(lattice(project).Pos2dsp(h.InstrumentParameters, x), dtype=float)
            weights = np.asarray(np.ma.getdata(data[2]), dtype=float)
            y = np.asarray(np.ma.getdata(data[1]), dtype=float)
            calc = np.asarray(np.ma.getdata(data[3]), dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
            for array in data[:3]:
                valid &= ~np.ma.getmaskarray(array)
            lo, hi = map(float, h.data['Limits'][1])
            exclusions = [list(map(float, row)) for row in h.Excluded()]
            fitted = valid & native_powder_fit_mask(x, [lo,hi], exclusions)
            ticks = []
            for name, record in h.reflections().items():
                refs = np.asarray(record.get('RefList', []), dtype=float)
                if refs.ndim != 2 or not len(refs):
                    continue
                keep=np.isfinite(refs[:,4])&(refs[:,4]>0)&np.isfinite(refs[:,5])
                ticks.append(dict(phase=name,d=refs[keep,4].tolist(),native=refs[keep,5].tolist()))
            instrument_type = str(h.InstrumentParameters.get('Type', [''])[0])
            native_rwp = _optional_rwp(h.get_wR())
            weight_factor = finite_scalar(h.data['data'][0].get('wtFactor',1.))
            usable_weights = weight_factor is not None and weight_factor>0
            residuals = y[fitted]-calc[fitted]
            fit_statistics = dict(point_count=int(np.count_nonzero(fitted)),
                weighted_residual_sum=weight_factor*float(np.dot(weights[fitted], residuals**2)) if native_rwp is not None and usable_weights else None,
                weighted_observed_sum=weight_factor*float(np.dot(weights[fitted], y[fitted]**2)) if usable_weights else None,
                calculation_status="evaluated" if native_rwp is not None else "not_yet_evaluated")
            histograms.append(dict(key=histogram_keys[h.name],name=h.name, coordinate='tof' if 'T' in instrument_type else 'two_theta',
                x=_finite_array(data[0]), d=[float(v) if np.isfinite(v) and v > 0 else None for v in d],
                observed=_finite_array(data[1]), weights=_finite_array(data[2]), weight_factor=weight_factor, calculated=_finite_array(data[3]),
                background=_finite_array(data[4]) if len(data) > 4 else [],
                valid=valid.tolist(), fitted=fitted.tolist(), limits=[lo, hi], exclusions=exclusions,
                fit_range_convention=NATIVE_FIT_CONVENTION,
                rwp=native_rwp, fit_statistics=fit_statistics, ticks=ticks,
                instrument_parameters=json_safe(h.InstrumentParameters)))
            histograms[-1]['observed'] = canonical_observations(histograms[-1])
        phases = []; metrics=[]
        for record in context['phases']:
            p = project.phase(record['name'])
            hap = {name:float(values['Scale'][0]) for name,values in p.data['Histograms'].items() if values.get('Use', True)}
            cell=list(map(float,p.data['General']['Cell'][1:]))
            cell_info=cell_fit_information(p,project.data.get('Covariance',{}).get('data',{}))
            covariance=cell_info.get('cell_covariance')
            for index,name in enumerate(['a','b','c','alpha','beta','gamma','volume']):
                sigma=float(np.sqrt(max(0,covariance[index][index]))) if covariance is not None and cell_info.get('metric_states',{}).get(name)=='refined' else None
                metrics.append(dict(key='cell|'+p.name+'|'+name,phase=p.name,kind='cell',parameter=name,label=p.name+' · '+name,unit='Å³' if name=='volume' else 'deg' if index in [3,4,5] else 'Å',value=cell[index],sigma=sigma,status=cell_info.get('metric_states',{}).get(name,'unknown')))
            for atom_index,a in enumerate(p.atoms()):
                values={'x':float(a.coordinates[0]),'y':float(a.coordinates[1]),'z':float(a.coordinates[2]),'occupancy':float(a.occupancy)}
                if a.adp_flag=='I':values['Uiso']=float(a.ADP)
                for name,value in values.items():
                    family={'x':'Ax','y':'Ay','z':'Az','occupancy':'Afrac','Uiso':'AUiso'}[name]
                    native_row=next((r for r in parameters if r.get('phase_id')==int(p.id) and r.get('atom_id')==atom_index and r.get('family')==family),{})
                    flag={'x':'X','y':'X','z':'X','occupancy':'F','Uiso':'U'}[name]
                    released=flag in a.refinement_flags
                    native_status=native_row.get('status','fixed')
                    fitted=native_status in {'refined_independent','refined_dependent'} or (native_status=='derived' and native_row.get('sigma') is not None)
                    sigma=native_row.get('sigma') if released and fitted else None
                    status='fixed' if not released else 'refined' if fitted else 'configured' if native_status.startswith('configured') else 'fixed_by_model'
                    metrics.append(dict(key='atom|'+p.name+'|'+a.label+'|'+name,phase=p.name,atom=a.label,kind='atom',parameter=name,label=p.name+' · '+a.label+' '+name,unit='Å²' if name=='Uiso' else '',value=value,sigma=float(sigma) if sigma is not None and np.isfinite(sigma) else None,status=status,native_status=native_status))
            phases.append(dict(key=record['key'], native_id=int(p.id), name=p.name, space_group=record['space_group'],
                cell=[float(x) for x in p.data['General']['Cell'][1:]], hap_scales=hap,
                atoms=[dict(label=a.label, native_id=i, element=a.element, coordinates=list(map(float,a.coordinates)),
                            occupancy=float(a.occupancy), adp=json_safe(a.ADP),
                            adp_type=next(x['adp_type'] for x in record['atoms'] if x['label']==a.label),
                            flags=a.refinement_flags.strip()) for i,a in enumerate(p.atoms())]))
        # Use GSAS-II's derived weight-fraction records, never equate raw HAP scales with mass fractions.
        for row in parameters:
            if row.get('family') not in {'Size;i', 'Mustrain;i'}:
                continue
            phase=next((p for p in project.phases() if int(p.id)==row.get('phase_id')),None)
            hist=next((h for h in project.histograms() if int(h.id)==row.get('histogram_id')),None)
            if phase is None or hist is None or finite_scalar(row.get('current_value')) is None:
                continue
            family=row['family'];kind='sample_size' if family=='Size;i' else 'sample_microstrain'
            configured=bool(row.get('configured_now'));fitted=row.get('status') in {'refined_independent','refined_dependent'}
            metrics.append(dict(key=kind+'|'+phase.name+'|'+histogram_keys[hist.name],kind=kind,
                phase=phase.name,histogram=hist.name,histogram_key=histogram_keys[hist.name],parameter=family,
                label=phase.name+' · '+kind.replace('_',' '),unit='µm' if kind=='sample_size' else 'µstrain',
                value=finite_scalar(row['current_value']),sigma=finite_scalar(row.get('sigma')) if configured and fitted else None,
                status='refined' if configured and fitted else 'configured' if configured else 'fixed'))
        for hist in project.histograms():
            rows=[r for r in parameters if r.get('histogram_id')==int(hist.id) and r.get('family')=='WgtFrac' and r.get('current_value') is not None]
            total=sum(float(r['current_value']) for r in rows)
            if total>0 and all(float(r['current_value'])>=0 for r in rows):
                for row in rows:
                    p=next((p for p in project.phases() if int(p.id)==row.get('phase_id')),None)
                    if p:metrics.append(dict(key='weight|'+p.name+'|'+histogram_keys[hist.name],phase=p.name,kind='weight',parameter='weight_percent',histogram=hist.name,histogram_key=histogram_keys[hist.name],label=p.name+' · modeled crystalline weight',unit='%',value=100*float(row['current_value'])/total,sigma=None,status='derived'))
        result = dict(schema='radar-polish-view/v1', source_sha256=before, histograms=histograms, phases=phases,parameters=metrics,
                      convergence=convergence_evidence(native))
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError('Plot extraction changed the original project')
    return result


def write_project_view(path, destination):
    view=project_view(path)
    Path(destination).write_text(json.dumps(view, allow_nan=False, separators=(',',':'))+'\n')
    return view
