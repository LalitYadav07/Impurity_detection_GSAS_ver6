"""Freeze reviewed cycler input and reproduce its exposure join in a worker.

The original bytes remain available for audit. A canonical UTC CSV permits a
native worker to reproduce the interval calculation without a binary-file
adapter. Its clock correction has already been applied and is never applied
twice. Both sources and the original binding are part of the preparation hash.
"""
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import io

from ..contracts import exact_keys
from radar_pd_nova.battery.electrochemistry import (
    CyclerBinding, MAX_BYTES, read_cycler, summarize_exposure,
)


def capture_electrochemistry(series, raw, root):
    if not isinstance(raw, bytes) or len(raw)>MAX_BYTES or sha256(raw).hexdigest()!=series.source_sha256:
        raise ValueError('The reviewed cycler source is unavailable or changed; bind it again')
    text=io.StringIO(newline='')
    writer=csv.writer(text)
    writer.writerow(['timestamp','voltage_v','current_a','cycle','step'])
    for point in series.points:
        writer.writerow([datetime.fromtimestamp(point.timestamp,timezone.utc).isoformat(),
                         point.voltage_v,point.current_a,point.cycle,point.step])
        if text.tell()>MAX_BYTES:
            raise ValueError('Canonical cycler evidence exceeds 64 MiB; use a shorter record')
    normalized=text.getvalue().encode('utf-8')
    if len(normalized)>MAX_BYTES:
        raise ValueError('Canonical cycler evidence exceeds 64 MiB; use a shorter record')
    references=[]
    for name,content in [('cycler-original.bin',raw),('cycler-utc.csv',normalized)]:
        path=root/name
        with path.open('xb') as stream:stream.write(content)
        references.append(dict(path=name,sha256=sha256(content).hexdigest()))
    return dict(schema='radar-battery-cycler-input/v1',raw=references[0],normalized=references[1],
                source_name=series.source_name,source_format=series.format,binding=asdict(series.binding))


def validate_electrochemistry(record, root, acquisition):
    from .preparation import input_path
    fields={'schema','raw','normalized','source_name','source_format','binding'}
    exact_keys(record,fields,fields)
    if record['schema']!='radar-battery-cycler-input/v1':
        raise ValueError('Unsupported cycler evidence schema')
    for name in ('source_name','source_format'):
        if not isinstance(record[name],str) or not 0<len(record[name])<=1024:
            raise ValueError('Record a bounded cycler source name and format')
    try:binding=CyclerBinding(**record['binding'])
    except TypeError as exc:raise ValueError('Invalid cycler binding fields') from exc
    binding.validate()
    if binding.sample_id!=acquisition['sample_id'] or binding.cell_id!=acquisition['cell_id']:
        raise ValueError('Cycler evidence belongs to a different sample or cell')
    input_path(root,record['raw']);input_path(root,record['normalized'])
    return binding


def reproduce_electrochemistry(record, root, acquisition):
    from .preparation import input_path
    original=validate_electrochemistry(record,root,acquisition)
    canonical=replace(original,clock_origin='',clock_offset_seconds=0.,local_time_format='',timezone_name='')
    series=read_cycler(input_path(root,record['normalized']).read_bytes(),canonical,source_name='cycler-utc.csv')
    series=replace(series,source_sha256=record['raw']['sha256'])
    joined=summarize_exposure(series,acquisition['start_utc'],acquisition['end_utc'],sample_id=acquisition['sample_id'])
    return dict(schema='radar-battery-model-electrochemistry/v1',original_binding=asdict(original),
                original_source=record['raw'],canonical_source=record['normalized'],
                normalization='UTC timestamps, V and A; original clock mapping already applied',
                source_name=record['source_name'],source_format=record['source_format'],exposure=joined,
                scientific_qualification='not_evaluated')
