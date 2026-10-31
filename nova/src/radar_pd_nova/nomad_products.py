"""Identities and bounded provenance for normalized NOMAD derived products.

A filename token identifies a product; it is never a measured temperature.
Summed membership comes from the captured reduction input, not a guessed run.
"""
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from datetime import datetime, timedelta, timezone
import json
import math
import re

SLICED_FOLDER='shared/sliced/reduced/GSAS'
SUMMED_FOLDER='shared/autoreduce/multi_banks_summed/GSAS'
_INDIVIDUAL=re.compile(r'(?:NOM_(\d+)|NOM(\d+)tof)\.gsa',re.I)
_SLICE=re.compile(r'nom_(\d+)_(\d+(?:\.\d+)?)\.h5\.gsa',re.I)


@dataclass(frozen=True)
class Product:
    run_number: int
    kind: str='individual'
    product_id: str | None=None
    members: tuple[int,...]=()


def member_runs(value):
    if not isinstance(value,str) or len(value)>12000:
        raise ValueError('Summed NOMAD products require an explicit bounded run list')
    result=[]
    for token in re.split(r'[,\s]+',value.strip()):
        match=re.fullmatch(r'([1-9]\d*)(?:-([1-9]\d*))?',token)
        if not match:raise ValueError('Unsupported NOMAD summed-run membership')
        first=int(match[1]);last=int(match[2] or first)
        if last<first or last-first>=2000:raise ValueError('Summed NOMAD run range exceeds its bound')
        result.extend(range(first,last+1))
        if len(result)>2000:raise ValueError('Summed NOMAD product exceeds 2000 members')
    if not result or len(set(result))!=len(result):raise ValueError('Summed NOMAD membership is empty or duplicated')
    return tuple(sorted(result))


def product_from_path(path):
    path=Path(path);name=path.name
    match=_SLICE.fullmatch(name)
    if match:return Product(int(match[1]),'slice','SLICE_'+match[2].replace('.','P'),(int(match[1]),))
    # Sum names alone cannot establish either ownership or membership.
    if (path.parent.name=='GSAS' and path.parent.parent.name=='multi_banks_summed'
            and re.fullmatch(r'NOM_[A-Za-z0-9_.+-]+\.gsa',name,re.I)):
        sidecar=path.parent.parent/'Input'/(path.stem+'.json')
        if not sidecar.is_file() or sidecar.stat().st_size>1048576:
            raise ValueError('Summed NOMAD product is missing its bounded reduction input')
        config=json.loads(sidecar.read_text())
        if not isinstance(config,dict) or not isinstance(config.get('Sample'),dict):
            raise ValueError('Invalid summed NOMAD reduction input')
        members=member_runs(config['Sample'].get('Runs'))
        return Product(max(members),'sum','SUM_'+sha256(name.encode()).hexdigest()[:12].upper(),members)
    match=_INDIVIDUAL.fullmatch(name)
    if match:return Product(next(int(x) for x in match.groups() if x is not None))
    return None


def slice_metadata_path(path):
    path=Path(path)
    if not _SLICE.fullmatch(path.name):raise ValueError('Not a supported NOMAD slice')
    if path.parent.name!='GSAS' or path.parent.parent.name!='reduced':
        raise ValueError('NOMAD slices must retain the processed-HDF5/reduction folder layout')
    return path.parent.parent.parent/'h5'/path.name.removesuffix('.gsa')


def slice_window(path):
    """Use filtered pulse logs and duration, never the retained parent end_time.

    Mantid NXlog relative times are seconds; timezone-free Mantid DateAndTime
    strings denote UTC. Parent NeXus bounds are checked by the caller.
    """
    path=Path(path);stat=path.stat()
    return json.loads(json.dumps(_slice_window(str(path),stat.st_size,stat.st_mtime_ns)))


@lru_cache(maxsize=512)
def _slice_window(path,size,mtime):
    import h5py
    import numpy as np
    prefix='mantid_workspace_1/logs/'
    with h5py.File(path,'r') as f:
        def scalar(name,default=None):
            if prefix+name+'/value' not in f:
                if default is not None:return default
                raise ValueError(f'Processed NOMAD metadata is missing {name}')
            item=f[prefix+name+'/value']
            if item.size!=1 or item.nbytes>16384:raise ValueError('Invalid scalar in processed NOMAD metadata')
            value=np.asarray(item[()]).reshape(-1)[0]
            return value.decode() if isinstance(value,bytes) else value.item() if hasattr(value,'item') else value
        run=str(scalar('run_number')).strip()
        if not re.fullmatch(r'[1-9]\d*',run):raise ValueError('Processed NOMAD product has no exact parent run')
        duration=float(scalar('duration'))
        charge=f[prefix+'proton_charge/time']
        if charge.ndim!=1 or not 2<=charge.size<=200000:raise ValueError('Processed NOMAD interval requires at least two pulses and at most 200000 pulse-log entries')
        start=charge.attrs.get('start')
        start=start.decode() if isinstance(start,bytes) else str(start or '')
        origin=datetime.fromisoformat(start.replace('Z','+00:00'))
        if origin.tzinfo is None:origin=origin.replace(tzinfo=timezone.utc)
        times=np.asarray(charge[()],dtype=float)
        differences=np.diff(times)
        if not np.isfinite(times).all() or not math.isfinite(duration) or duration<=0 or duration>86400:
            raise ValueError('Invalid processed NOMAD acquisition duration or pulse times')
        if np.any(differences<=0):raise ValueError('Processed NOMAD pulse times are not strictly ordered')
        period=float(np.median(differences));span=float(times[-1]-times[0])
        if not 0<period<=1 or float(differences.max())>max(5*period,.1):
            raise ValueError('Processed NOMAD pulse log is discontiguous; review a multi-interval product')
        if abs(span-duration)>max(3*period,.1):
            raise ValueError('Processed NOMAD duration and filtered pulse interval disagree')
        first=origin+timedelta(seconds=float(times[0]));last=origin+timedelta(seconds=float(times[-1]))
        result=dict(run_number=int(run),start=first.astimezone(timezone.utc).isoformat(),
            end=last.astimezone(timezone.utc).isoformat(),duration_seconds=span,recorded_duration_seconds=duration,
            basis='Contiguous filtered proton-pulse interval, cross-checked against Mantid processed-product duration',
            pulse_count=int(len(times)),pulse_period_seconds=period,
            metadata_path=str(path),metadata_signature=[size,mtime],retained_parent_end_time=str(scalar('end_time','unavailable')))
    return result
