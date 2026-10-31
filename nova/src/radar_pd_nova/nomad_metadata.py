"""NOMAD scalar NeXus metadata; event banks and inactive sensors are never read."""
from pathlib import Path
from typing import Any
import re
import math
from datetime import timezone

from .powgen_metadata import (_first_snapshot, _text_value, _identifier_value,
    _numeric_log, _temperature_in_kelvin, _duration_seconds, read_powgen_sample_identity)
from .powgen_samples import PowgenSampleIndex
from .facility_metadata import resolve_nexus_path, qualify_temperature


def nomad_nexus_path(ipts: str, run: int) -> Path:
    proposal = str(ipts).upper()
    if not re.fullmatch(r"IPTS-\d+", proposal) or isinstance(run, bool) or int(run) < 1:
        raise ValueError("NOMAD requires an IPTS identifier and a positive run number")
    return resolve_nexus_path('NOM', proposal, run)


def read_nomad_sample_identity(ipts: str, run: int, *, nexus_path=None) -> dict[str, Any]:
    path = Path(nexus_path) if nexus_path else nomad_nexus_path(ipts, run)
    return read_powgen_sample_identity(ipts, run, nexus_path=path, instrument='NOM')


class NomadSampleIndex(PowgenSampleIndex):
    def __init__(self, ipts: str):
        super().__init__(ipts, nexus_path=nomad_nexus_path, identity_reader=read_nomad_sample_identity, instrument='NOM')

    def identity(self,run):
        from dataclasses import replace
        from .nomad_products import product_from_path
        product=product_from_path(run.source_path)
        if not product or product.kind!='sum':return super().identity(run)
        parents=[super(NomadSampleIndex,self).identity(replace(run,run_number=number,product_id=None)) for number in product.members]
        ids={r.get('sample_id') for r in parents}
        if None in ids or '' in ids or len(ids)!=1 or any(r.get('identity_status')=='conflict' for r in parents):
            return dict(sample_id='',sample_name='',identity_status='conflict',
                identity_warning='Summed product members have missing or conflicting sample identities',
                metadata_actions=['Review every member run before analyzing this sum'])
        result=dict(parents[0]);result['sample_name']=' / '.join(sorted({r.get('sample_name','') for r in parents}))
        result['member_run_numbers']=list(product.members)
        return result

    def candidate_runs(self,runs,selected_ids):
        runs=list(runs)
        return runs if any(r.product_id for r in runs) else super().candidate_runs(runs,selected_ids)

    def inventory(self,runs,*,catalog=False,progress=None):
        runs=list(runs)
        # A catalog row for the last member cannot qualify an entire sum.
        derived=any(r.product_id for r in runs)
        result=super().inventory(runs,catalog=catalog and not derived,progress=progress)
        if derived:
            for option,run in zip(result['run_options'],runs):
                option['value']=run.run_id
                option['title']=Path(run.source_path).name+' · '+option['title']
        return result


def read_nomad_product_metadata(ipts,run,source):
    from .nomad_products import product_from_path,slice_metadata_path,slice_window
    from .scan_conditions import attach_conditions,qualified_metadata,SCHEMA,REGISTRY
    from .facility_metadata import parse_nexus_timestamp
    from hashlib import sha256
    import json
    product=product_from_path(source)
    if not product or product.run_number!=run:raise ValueError('NOMAD product identity changed before metadata inspection')
    if product.kind=='individual':return read_nomad_scan_metadata(ipts,run)
    if product.kind=='slice':
        window=slice_window(slice_metadata_path(source))
        if window['run_number']!=run:raise ValueError('Processed NOMAD slice names another parent run')
        metadata=read_nomad_scan_metadata(ipts,run)
        if not metadata.get('available'):raise ValueError('The slice parent NeXus is unavailable for interval verification')
        start,end=(parse_nexus_timestamp(window[k]) for k in ('start','end'))
        parent_start,parent_end=(parse_nexus_timestamp(metadata[k]) for k in ('start_time','end_time'))
        if start<parent_start or end>parent_end:
            raise ValueError('Processed NOMAD slice interval lies outside its parent acquisition')
        interval={k:window[k] for k in ('start','end')}
        # A missing/unusable slice readback must never resurrect the parent
        # run's full-temperature-sweep mean through the legacy scalar fallback.
        metadata={k:v for k,v in metadata.items() if k not in
                  ['temperature','magnetic_field','pressure','voltage','temperature_observation',
                   'magnetic_field_observation','pressure_observation','voltage_observation','conditions']}
        metadata=attach_conditions(metadata,nomad_nexus_path(ipts,run),'NOM',window=interval)
        metadata.update(schema_version=3,product_id=product.product_id,product_kind='slice',
            product_metadata=window,start_time=interval['start'],end_time=interval['end'],duration_seconds=window['duration_seconds'])
        return metadata
    members=[read_nomad_scan_metadata(ipts,number) for number in product.members]
    ids={m.get('sample_id') for m in members}
    if len(ids)!=1 or None in ids or '' in ids or any(m.get('identity_status')=='conflict' for m in members):
        raise ValueError('Summed NOMAD product has mixed or unresolved sample identities')
    if any(not m.get('available') for m in members):raise ValueError('A summed NOMAD parent acquisition is unavailable')
    interval=dict(start=min(parse_nexus_timestamp(m['start_time']) for m in members).astimezone(timezone.utc).isoformat(),
                  end=max(parse_nexus_timestamp(m['end_time']) for m in members).astimezone(timezone.utc).isoformat())
    channels=[]
    for quantity in ['temperature','magnetic_field','pressure','voltage']:
        metrics=[m.get(quantity) or {} for m in members]
        if any(m.get('role')!='measured' or m.get('status')!='recorded' or m.get('coverage_fraction',0)<1-1e-6 for m in metrics):continue
        units={m.get('unit') for m in metrics}
        charges=[m.get('proton_charge') or {} for m in members]
        if len(units)!=1 or None in units or '' in units or len({c.get('unit') for c in charges})!=1 or not charges[0].get('unit'):continue
        weights=[c.get('value',0) for c in charges]
        def finite(value):return type(value) in (int,float) and math.isfinite(value)
        if any(not finite(w) or w<=0 for w in weights):continue
        if any(not finite(m.get('value')) or not finite(m.get('minimum',m['value'])) or not finite(m.get('maximum',m['value'])) for m in metrics):continue
        channels.append(dict(quantity=quantity,value=sum(m['value']*w for m,w in zip(metrics,weights))/sum(weights),
            unit=metrics[0]['unit'],minimum=min(m.get('minimum',m['value']) for m in metrics),
            maximum=max(m.get('maximum',m['value']) for m in metrics),role='measured',status='recorded',
            coverage_fraction=1.,statistic='parent_run_charge_weighted_mean',
            source='NOMAD member-run NXlogs; membership retained in product metadata',acquisition_window=interval))
        for flag in ['requires_binding','binding_verified']:
            if any(flag in m for m in metrics):channels[-1][flag]=all(m.get(flag,False) for m in metrics)
    packet={'$schema':SCHEMA,'registry':REGISTRY,'acquisition_window':interval,
        'window_scope':'summed_parent_runs; individual reduced-product intervals unverified','channels':channels,
        'notices':['Conditions summarize multiple member acquisitions; do not interpret the interval as one continuous measurement'],
        'member_runs':list(product.members)}
    packet['metadata_sha256']=sha256(json.dumps(packet,sort_keys=True,allow_nan=False).encode()).hexdigest()
    metadata={k:v for k,v in members[0].items() if k not in ['conditions','temperature','magnetic_field','pressure','voltage','proton_charge']}
    durations=[m.get('duration_seconds') for m in members]
    if any(type(d) not in (int,float) or not math.isfinite(d) or d<=0 for d in durations):
        raise ValueError('Summed NOMAD parent duration is unavailable')
    metadata.update(schema_version=3,product_kind='sum',product_id=product.product_id,
        member_run_numbers=list(product.members),start_time=interval['start'],end_time=interval['end'],
        duration_seconds=sum(durations),conditions=packet)
    return qualified_metadata(metadata)


def read_nomad_scan_metadata(ipts: str, run: int, *, nexus_path=None) -> dict[str, Any]:
    path = Path(nexus_path) if nexus_path else nomad_nexus_path(ipts, run)
    result: dict[str, Any] = {'schema_version': 1, 'instrument': 'NOM',
                             'nexus_path': str(path), 'available': path.is_file()}
    if not result['available']:
        return result
    stored_run = _identifier_value(path, ('/entry/run_number',))
    stored_proposal = _text_value(path, ('/entry/experiment_identifier',)).upper()
    if stored_run and stored_run != str(run):
        raise ValueError('NOMAD NeXus run identifier disagrees with the selected run')
    if stored_proposal and stored_proposal != str(ipts).upper():
        raise ValueError('NOMAD NeXus proposal disagrees with the selected IPTS')
    result.update(read_nomad_sample_identity(ipts, run, nexus_path=path))
    for key, field in [('run_title','/entry/title'), ('sample_formula','/entry/sample/chemical_formula'),
                       ('start_time','/entry/start_time'), ('end_time','/entry/end_time')]:
        result[key] = _text_value(path, (field,))
    result['duration_seconds'] = _duration_seconds(result['start_time'], result['end_time'])
    result['temperature_sensor'] = _text_value(path, ('/entry/DASlogs/BL1B:SE:SelectSampleTemp/value_strings',))
    # Only the selected SampleTemp summary is meaningful; never fall back to an
    # inactive levitator/furnace channel or infer units from a title/magnitude.
    prefix = '/entry/DASlogs/BL1B:SE:SampleTemp/'
    temp = _numeric_log(_first_snapshot(path, (prefix+'average_value',)))
    if temp:
        temp['statistic'] = 'mean'
        for key, field in [('minimum','minimum_value'), ('maximum','maximum_value')]:
            value = _numeric_log(_first_snapshot(path, (prefix+field,)))
            if value and value['unit'] == temp['unit']:
                temp[key] = value['value']
        result['temperature_observation'] = _temperature_in_kelvin(temp)
        qualified, notices = qualify_temperature(result['temperature_observation'], result.get('run_title',''))
        result['metadata_notices'] = notices
        if qualified:
            result['temperature'] = qualified
    else:
        result['metadata_notices'] = ['Selected sample temperature is unavailable']
    charge = _numeric_log(_first_snapshot(path, ('/entry/proton_charge',)))
    if charge:
        result['proton_charge'] = charge
    # A chopper request is diagnostic metadata, not a monochromatic wavelength.
    request = _numeric_log(_first_snapshot(path, ('/entry/DASlogs/BL1B:Chop:Skf1:WavelengthUserReq/value',)))
    if request:
        result['chopper_wavelength_request'] = request
    from .scan_conditions import attach_conditions
    return attach_conditions(result, path, "NOM")
