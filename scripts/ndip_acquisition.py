"""Validate a small acquisition handoff and annotate published GPX copies."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import math

SCHEMA = 'radar-pd-acquisition/v1'


def validate_acquisition(payload):
    if not isinstance(payload, dict) or payload.get('$schema') != SCHEMA:
        raise ValueError('Unsupported acquisition metadata')
    if len(json.dumps(payload, allow_nan=False).encode('ascii')) > 6000:
        raise ValueError('Acquisition metadata exceeds its size limit')
    if not isinstance(payload.get('product_window_verified'), bool):
        raise ValueError('Acquisition interval qualification is missing')
    if (not isinstance(payload.get('product_window_basis'), str) or not payload['product_window_basis']
            or isinstance(payload.get('run_number'), bool) or not isinstance(payload.get('run_number'), int)
            or payload['run_number'] < 1):
        raise ValueError('Acquisition identity/interval provenance is missing')
    window = payload.get('acquisition_window') or {}
    try:
        start, end = (datetime.fromisoformat(window[k].replace('Z', '+00:00')) for k in ('start', 'end'))
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValueError
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError('Acquisition metadata needs a dated, ordered interval') from exc
    seen = set()
    channels = payload.get('channels')
    if not isinstance(channels, list) or len(channels) > 4:
        raise ValueError('Invalid acquisition channels')
    for channel in channels:
        quantity = channel.get('quantity') if isinstance(channel, dict) else None
        if quantity not in {'temperature', 'magnetic_field', 'pressure', 'voltage'} or quantity in seen:
            raise ValueError('Ambiguous acquisition channels')
        seen.add(quantity)
        if channel.get('role') != 'measured' or channel.get('status') != 'recorded':
            raise ValueError('Only recorded readbacks can be handed to GPX')
        for field in ('value', 'coverage_fraction'):
            value = channel.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Nonfinite or unqualified acquisition value')
        if not 1 - 1e-6 <= channel['coverage_fraction'] <= 1 + 1e-6:
            raise ValueError('Acquisition log does not cover the interval')
        if not isinstance(channel.get('source'), str) or not channel['source'] or not channel.get('unit'):
            raise ValueError('Acquisition units/source are missing')
        if channel.get('acquisition_window', window) != window:
            raise ValueError('Acquisition channel belongs to a different interval')
        low, high = channel.get('minimum'), channel.get('maximum')
        if low is not None or high is not None:
            if (any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (low, high))
                    or not low <= channel['value'] <= high):
                raise ValueError('Acquisition range does not contain the reported value')
    return deepcopy(payload)


def annotate_gpx_records(records, payload):
    """Preserve every fitted array/parameter; only sample metadata is changed."""
    acquisition = validate_acquisition(payload)
    histograms = [r for r in records if r and str(r[0][0]).startswith('PWDR')]
    if len(histograms) != 1:
        return {'status':'multiple_histograms_need_mapping', 'temperature_written':False}
    record = histograms[0]
    sample = next((item[1] for item in record[1:] if item[0] == 'Sample Parameters'), None)
    temperature = next((c for c in acquisition['channels'] if c['quantity'] == 'temperature'), None)
    status = 'recorded_context_only'
    written = False
    if not acquisition['product_window_verified']:
        status = 'product_interval_needs_review'
    elif temperature is not None:
        if temperature['unit'] != 'K' or temperature['value'] < 0:
            status = 'temperature_units_need_review'
        elif not isinstance(sample, dict):
            status = 'sample_parameters_missing'
        else:
            sample['Temperature'] = float(temperature['value'])
            status, written = 'applied', True
    receipt = {'status':status, 'temperature_written':written, 'metadata':acquisition}
    record[:] = [item for item in record if item[0] != 'RADAR-PD acquisition']
    record.append(['RADAR-PD acquisition', receipt])
    comments = next((item[1] for item in record[1:] if item[0] == 'Comments'), None)
    if isinstance(comments, list):
        prefix = 'RADAR-PD acquisition: '
        comments[:] = [line for line in comments if not isinstance(line, str) or not line.startswith(prefix)]
        detail = (f"measured temperature {temperature['value']:.8g} K; "
                  f"recorded range {temperature.get('minimum')}–{temperature.get('maximum')} K; "
                  f"{temperature.get('statistic', 'recorded readback')}; source {temperature['source']}; "
                  if written else 'sample temperature not assigned; ')
        comments.append(prefix + detail + acquisition['product_window_basis'])
    return {'status':status, 'temperature_written':written,
            'temperature_K':float(temperature['value']) if written else None}
