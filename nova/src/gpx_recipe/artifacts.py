"""Native, copy-isolated plot/parameter exports for the integrated NOVA view."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .snapshot import gsas, snapshot, json_safe
from .advanced import lattice
from .studio_state import extract_gsasii_refinement_state


def _finite_array(values):
    a = np.asarray(np.ma.getdata(values), dtype=float)
    mask = np.ma.getmaskarray(values)
    return [float(v) if np.isfinite(v) and not m else None for v, m in zip(a, mask)]


def project_view(path):
    """Keep complete measured samples; the browser never performs GPX extraction."""
    path = Path(path); before = hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='radar-polish-view-') as folder:
        local = Path(folder) / 'project.gpx'; shutil.copy2(path, local)
        context = snapshot(local)
        G = gsas(); project = G.G2Project(gpxfile=str(local)); project.index_ids()
        native=extract_gsasii_refinement_state(local,project,G,max_correlations=0)
        parameters=native["parameter_space"]["parameters"]
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
            fitted = valid & (x >= lo) & (x <= hi)
            exclusions = [list(map(float, row)) for row in h.Excluded()]
            for a, b in exclusions:
                fitted &= ~((x >= a) & (x <= b))
            ticks = []
            for name, record in h.reflections().items():
                refs = np.asarray(record.get('RefList', []), dtype=float)
                if refs.ndim != 2 or not len(refs):
                    continue
                keep=np.isfinite(refs[:,4])&(refs[:,4]>0)&np.isfinite(refs[:,5])
                ticks.append(dict(phase=name,d=refs[keep,4].tolist(),native=refs[keep,5].tolist()))
            instrument_type = str(h.InstrumentParameters.get('Type', [''])[0])
            histograms.append(dict(name=h.name, coordinate='tof' if 'T' in instrument_type else 'two_theta',
                x=x.tolist(), d=[float(v) if np.isfinite(v) and v > 0 else None for v in d],
                observed=_finite_array(data[1]), calculated=_finite_array(data[3]),
                background=_finite_array(data[4]) if len(data) > 4 else [],
                valid=valid.tolist(), fitted=fitted.tolist(), limits=[lo, hi], exclusions=exclusions,
                rwp=float(h.get_wR()) if np.isfinite(h.get_wR()) else None, ticks=ticks,
                instrument_parameters=json_safe(h.InstrumentParameters)))
        phases = []; metrics=[]
        for record in context['phases']:
            p = project.phase(record['name'])
            hap = {name:float(values['Scale'][0]) for name,values in p.data['Histograms'].items() if values.get('Use', True)}
            cell=list(map(float,p.data['General']['Cell'][1:]))
            try:
                from ndip_lattice import cell_fit_information
                cell_info=cell_fit_information(p,project.data.get('Covariance',{}).get('data',{}))
            except ImportError:cell_info={}
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
                    sigma=native_row.get('sigma') if flag in a.refinement_flags else None
                    metrics.append(dict(key='atom|'+p.name+'|'+a.label+'|'+name,phase=p.name,atom=a.label,kind='atom',parameter=name,label=p.name+' · '+a.label+' '+name,unit='Å²' if name=='Uiso' else '',value=value,sigma=float(sigma) if sigma is not None and np.isfinite(sigma) else None,status='refined' if flag in a.refinement_flags else 'fixed'))
            phases.append(dict(key=record['key'], name=p.name, space_group=record['space_group'],
                cell=[float(x) for x in p.data['General']['Cell'][1:]], hap_scales=hap,
                atoms=[dict(label=a.label, element=a.element, coordinates=list(map(float,a.coordinates)),
                            occupancy=float(a.occupancy), adp=json_safe(a.ADP),
                            adp_type=next(x['adp_type'] for x in record['atoms'] if x['label']==a.label),
                            flags=a.refinement_flags.strip()) for a in p.atoms()]))
        # Use GSAS-II's derived weight-fraction records, never equate raw HAP scales with mass fractions.
        for hist in project.histograms():
            rows=[r for r in parameters if r.get('histogram_id')==int(hist.id) and r.get('family')=='WgtFrac' and r.get('current_value') is not None]
            total=sum(float(r['current_value']) for r in rows)
            if total>0 and all(float(r['current_value'])>=0 for r in rows):
                for row in rows:
                    p=next((p for p in project.phases() if int(p.id)==row.get('phase_id')),None)
                    if p:metrics.append(dict(key='weight|'+p.name+'|'+hist.name,phase=p.name,kind='weight',parameter='weight_percent',histogram=hist.name,label=p.name+' · modeled crystalline weight',unit='%',value=100*float(row['current_value'])/total,sigma=None,status='derived'))
        result = dict(schema='radar-polish-view/v1', source_sha256=before, histograms=histograms, phases=phases,parameters=metrics)
    if hashlib.sha256(path.read_bytes()).hexdigest() != before:
        raise ValueError('Plot extraction changed the original project')
    return result


def write_project_view(path, destination):
    view=project_view(path)
    Path(destination).write_text(json.dumps(view, allow_nan=False, separators=(',',':'))+'\n')
    return view
