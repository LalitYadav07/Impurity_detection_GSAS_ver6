"""Reviewed specimen sensor choices, scoped to a SNAP acquisition setup.

The selection is part of the immutable watch, not a plotting preference.
Hydraulic pressure is deliberately absent: it does not measure DAC pressure.
"""
from hashlib import sha256
import json
import re

from .scan_conditions import sources


def validate_bindings_json(value):
    if not isinstance(value, str) or len(value) > 16384:
        raise ValueError('SNAP condition mappings exceed the supported metadata bound')
    try:
        bindings = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid SNAP condition mapping') from exc
    if not isinstance(bindings, dict) or set(bindings) - {'temperature'}:
        raise ValueError('SNAP supports a reviewed specimen-temperature sensor; use a reviewed mapping for DAC pressure')
    allowed = {s.channel for s in sources('SNAP') if s.quantity == 'temperature'}
    for binding in bindings.values():
        if (not isinstance(binding, dict)
                or set(binding) != {'channel', 'unit', 'verified', 'scope_sha256', 'schema_sha256','product_window_reviewed'}
                or binding.get('channel') not in allowed
                or binding.get('unit') not in {'K', 'C'}
                or binding.get('verified') is not True
                or binding.get('product_window_reviewed') is not True
                or any(not isinstance(binding.get(k), str) or not re.fullmatch(r'[0-9a-f]{64}', binding[k])
                       for k in ('scope_sha256', 'schema_sha256'))):
            raise ValueError('Check the SNAP specimen sensor, units and full-acquisition export interval, then explicitly review the mapping')
    return bindings


def binding_scope(metadata):
    """No values or run number: changing a readback must not break a sweep."""
    packet = metadata.get('conditions') or {}
    document = {
        'instrument': 'SNAP', 'ipts': metadata.get('ipts', ''),
        'sample_id': metadata.get('sample_id', ''),
        'identity_status': metadata.get('identity_status', ''),
        'sample_environment_selector': metadata.get('sample_environment_selector', ''),
        'schema_sha256': packet.get('channel_schema_sha256', ''),
    }
    return {**document, 'scope_sha256': sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()}


def condition_options(metadata):
    result = []
    for c in (metadata.get('conditions') or {}).get('channels', []):
        if c.get('quantity') != 'temperature' or c.get('role') != 'measured':
            continue
        unit = c.get('source_unit') or 'units not recorded'
        low, high = c.get('minimum'), c.get('maximum')
        observed = f' · {low:.5g}–{high:.5g} {c.get("unit", "")}' if low is not None and high is not None else ''
        result.append({'value': c['channel'],
                       'title': f'{c["channel"]} · source: {unit}{observed} · {c["status"]}'})
    return result
