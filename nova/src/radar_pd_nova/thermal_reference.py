"""Reviewed, bounded reference curves for lattice thermometry.

Curves are supplied explicitly; chemistry, composition, pressure and calibration
validity are not inferred from a phase name. No extrapolation or nearest root.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from hashlib import sha256
import io
import math
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceCurve(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    id: str = Field(min_length=1,max_length=80,pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]*$')
    label: str = Field(min_length=1,max_length=200)
    source: str = Field(min_length=1,max_length=1000)
    unit: str
    temperatures: list[float] = Field(min_length=3,max_length=2000)
    values: list[float] = Field(min_length=3,max_length=2000)
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def valid_table(self):
        if self.unit not in {'angstrom','angstrom^3','dimensionless'}:
            raise ValueError('Reference values must use angstrom, angstrom^3 or dimensionless units')
        if len(self.temperatures)!=len(self.values):
            raise ValueError('Reference temperatures and values must have equal length')
        if any(v<=0 for v in [*self.temperatures,*self.values]):
            raise ValueError('Reference temperatures in K and lattice values must be positive')
        if any(b<=a for a,b in zip(self.temperatures,self.temperatures[1:])):
            raise ValueError('Reference temperatures must increase strictly, without duplicate knots')
        return self


def load_reference_csv(path):
    path=Path(path)
    before=path.stat()
    if before.st_size>512*1024:
        raise ValueError('Reference CSV exceeds 512 KiB')
    raw=path.read_bytes()
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Reference CSV changed while reading')
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    names=reader.fieldnames or []
    temperature_fields=set(names)&{'temperature_K','temperature_C'}
    if len(temperature_fields)!=1 or not {'value','unit','source'}<=set(names):
        raise ValueError('Reference CSV needs temperature_K (or temperature_C), value, unit and source columns')
    if len(set(names))!=len(names):
        raise ValueError('Reference CSV has duplicate columns')
    key=next(iter(temperature_fields));rows=list(reader)
    if not 3<=len(rows)<=2000 or any(None in row or any(row.get(k) is None for k in names) for row in rows):
        raise ValueError('Reference CSV needs 3–2000 complete rows')
    units={str(r['unit']).strip() for r in rows};sources={str(r['source']).strip() for r in rows}
    if len(units)!=1 or len(sources)!=1 or not next(iter(sources)):
        raise ValueError('Reference CSV must have one explicit unit and one provenance source')
    digest=sha256(raw).hexdigest()
    return ReferenceCurve(id='curve-'+digest,label=path.name,source=next(iter(sources)),
        unit=next(iter(units)),source_sha256=digest,
        temperatures=[float(r[key])+(273.15 if key=='temperature_C' else 0) for r in rows],
        values=[float(r['value']) for r in rows])


@lru_cache(maxsize=32)
def _interpolator(temperatures,values):
    from scipy.interpolate import PchipInterpolator
    return PchipInterpolator(temperatures,np.log(values),extrapolate=False)


def interpolator(curve):
    return _interpolator(tuple(curve.temperatures),tuple(curve.values))


def infer_temperature(curve,ratio,baseline_temperature):
    """Return unique T and d ln F/dT. A multivalued thermometer is unavailable."""
    model=interpolator(curve)
    if not curve.temperatures[0]<=baseline_temperature<=curve.temperatures[-1]:
        raise ValueError('Baseline temperature is outside the reviewed reference table')
    if not math.isfinite(ratio) or ratio<=0:
        raise ValueError('A positive normalized lattice signal is required')
    target=float(model(baseline_temperature))+math.log(ratio)
    roots=model.solve(target,extrapolate=False)
    roots=sorted(float(t) for t in roots if np.isfinite(t)
        and curve.temperatures[0]-1e-8<=t<=curve.temperatures[-1]+1e-8)
    unique=[]
    for t in roots:
        if not unique or abs(t-unique[-1])>1e-5:unique.append(t)
    if len(unique)!=1:
        raise ValueError('Reference inversion has no unique temperature in its supported range')
    t=unique[0];slope=float(model.derivative()(t))
    if not math.isfinite(slope) or abs(slope)<1e-12:
        raise ValueError('Reference curve is too insensitive for lattice thermometry here')
    return t,slope


def apply_reference(result,view,curve):
    """Transform already grouped, baseline-normalized observations in place."""
    expected_unit='dimensionless' if view.metric=='phase_ratio' else 'angstrom^3' if view.metric=='volume' else 'angstrom'
    if curve.unit!=expected_unit:
        result.update(status='needs_input')
        result['notices'].append('Reference units do not match the selected lattice signal')
        for s in result['series']:
            for p in s['points']:p.update(value=None,variance=None,reason='incompatible reference-curve units')
        return
    model=interpolator(curve);factor={'relative':100.,'permille':1000.,'normalized':1.}[result['representation']]
    output=view.reference_output
    result.update(reference_output=output,reference_curve={'id':curve.id,'label':curve.label,
        'source':curve.source,'source_sha256':curve.source_sha256,'unit':curve.unit,
        'range_K':[curve.temperatures[0],curve.temperatures[-1]]})
    result['reference_traces']=[]
    problems={}
    for series in result['series']:
        baseline=next((p for p in series['points'] if p['run_id']==series.get('reference_run')
            and p.get('raw_value') is not None and p.get('state') in {'refined','constraint-derived'}
            and not p.get('excluded') and not p.get('reason')),None)
        base_axis=(baseline or {}).get('temperature') or {}
        t0=base_axis.get('value')
        def qualified_temperature(axis):
            return axis.get('kind')=='measured' or (
                axis.get('kind')=='user_supplied' and axis.get('reviewed_mapping') is True)
        if (baseline is None or not isinstance(t0,(float,int)) or not qualified_temperature(base_axis)
                or not curve.temperatures[0]<=t0<=curve.temperatures[-1]):
            for p in series['points']:p.update(value=None,variance=None,reason='a qualified, supported baseline temperature is required')
            problems['A baseline with measured or reviewed mapped temperature inside the reference range is required']=1
            continue
        base=baseline['raw_value'];vb=baseline['raw_variance'];g0=float(model(t0));dg0=float(model.derivative()(t0))
        base_sigma=view.baseline_temperature_esd
        if base_sigma is None:base_sigma=base_axis.get('standard_uncertainty')
        reference_points=[]
        for p in series['points']:
            if p['value'] is None or p.get('reason') or p.get('excluded'):continue
            if p.get('state') not in {'refined','constraint-derived'}:
                reason='Reference comparison requires a refined lattice signal, not a fixed or unverified model value'
                p.update(value=None,variance=None,reason=reason)
                problems[reason]=problems.get(reason,0)+1
                continue
            raw=p['raw_value'];va=p['raw_variance'];q=raw/base
            self_reference=p['run_id']==baseline['run_id']
            log_var=0. if self_reference else (va/raw**2+vb/base**2 if va is not None and vb is not None else None)
            p['reference_curve_id']=curve.id
            p['uncertainty_note']='; '.join(filter(None,[p.get('uncertainty_note'),
                'Conditional refinement uncertainty, including the baseline; shared-baseline correlations and reference/calibration systematics are not independent point errors']))
            axis=p.get('temperature') or {};logged=axis.get('value') if qualified_temperature(axis) else None
            sigma_log=axis.get('standard_uncertainty')
            try:
                if output in {'overlay','residual','temperature_offset'} and logged is None:
                    raise ValueError('Measured or reviewed mapped logged temperature is required for this comparison')
                if output in {'overlay','residual'}:
                    if not curve.temperatures[0]<=logged<=curve.temperatures[-1]:
                        raise ValueError('Logged temperature is outside the reviewed reference table; no extrapolation')
                    expected=math.exp(float(model(logged))-g0)
                    reference_value=expected if result['representation']=='normalized' else factor*(expected-1)
                    reference_points.append({'x':p['x'],'value':reference_value,'run_id':p['run_id'],'branch':p['branch']})
                    p['reference_value']=reference_value
                    if output=='residual':
                        variance=p['variance']
                        if variance is not None and not self_reference:
                            variance+=factor**2*expected**2*((float(model.derivative()(logged))*(sigma_log or 0))**2+(dg0*(base_sigma or 0))**2)
                        p.update(value=p['value']-reference_value,variance=variance)
                else:
                    if self_reference:
                        # This point is defined by the anchor, not by inverting a
                        # possibly multivalued or locally flat thermometer.
                        t=t0;slope=dg0
                        variance=base_sigma**2 if base_sigma is not None else None
                    else:
                        t,slope=infer_temperature(curve,q,t0)
                        variance=(log_var+(dg0*(base_sigma or 0))**2)/slope**2 if log_var is not None else None
                    if output=='temperature_offset':
                        if self_reference:
                            variance=0.
                            p['uncertainty_note']='Zero offset at the baseline is a normalization identity, not an independent temperature validation'
                        elif variance is not None and sigma_log is not None:
                            variance+=sigma_log**2
                    p.update(value=t-logged if output=='temperature_offset' else t,variance=variance,
                        inferred_temperature_K=t,logged_temperature_K=logged,reference_sensitivity_per_K=slope)
            except ValueError as exc:
                reason=str(exc);problems[reason]=problems.get(reason,0)+1
                p.update(value=None,variance=None,reason=reason)
        if output=='overlay':
            result['reference_traces'].append({'series':series['key'],'label':series['phase'],'points':reference_points})
    if output in {'temperature','temperature_offset'}:result['value_unit']='K'
    result['notices'].extend(f'{count} point(s): {reason}.' for reason,count in problems.items())
    result['notices'].append('Reference-curve comparison is conditional on the reviewed material, pressure and phase range, and on the baseline temperature being correct. Literature/model systematics and temperature uncertainty not explicitly supplied are excluded from these bars; a recorded temperature range is not a standard uncertainty.')
