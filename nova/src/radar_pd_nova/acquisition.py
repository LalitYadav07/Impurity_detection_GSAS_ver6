"""Bounded acquisition context carried with the staged powder file.

Only recorded specimen readbacks qualify. Requested values, unbound devices
and incomplete log coverage never become GSAS-II sample parameters.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

from .scan_conditions import qualified_metadata
from .facility_metadata import parse_nexus_timestamp

PREFIX = b"# RADAR-PD input provenance: "
SCHEMA = "radar-pd-acquisition/v1"


def _full_powgen_run(source, metadata, expected_script_hash):
    """Recognize the unfiltered, one-file SNSPowderReduction recipe only."""
    path = Path(source).with_suffix('.py')
    if not expected_script_hash or not path.is_file() or path.stat().st_size > 262144:
        return False
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_script_hash:
        return False
    try:
        tree = ast.parse(raw)
    except (SyntaxError, ValueError):
        return False
    reducers = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ''
        if name in {'FilterEvents', 'FilterByTime', 'GenerateEventsFilter'}:
            return False
        if name != 'SNSPowderReduction':
            continue
        if not any(isinstance(top, ast.Expr) and top.value is node for top in tree.body):
            return False
        keys = {kw.arg for kw in node.keywords}
        if None in keys or 'InputWorkspace' in keys or any('time' in k.lower() or 'split' in k.lower() for k in keys):
            return False
        kwargs = {kw.arg: kw.value.value for kw in node.keywords if isinstance(kw.value, ast.Constant)}
        if 'SaveAs' in kwargs and 'gsas' not in str(kwargs['SaveAs']).lower():
            continue
        reducers.append(kwargs)
    return len(reducers) == 1 and reducers[0].get('Filename') == metadata.get('nexus_path')


def acquisition_context(metadata, *, instrument, run_number, source, script_sha256=''):
    metadata = qualified_metadata(metadata)
    conditions = metadata.get('conditions') or {}
    if conditions.get('$schema') != 'radar-pd-conditions/v1':
        return None
    window = conditions.get('acquisition_window') or {}
    try:
        start, end = (parse_nexus_timestamp(window[k]) for k in ('start', 'end'))
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            return None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    window_verified = conditions.get('window_scope') == 'explicit_product_window'
    basis = 'explicit reduced-product interval' if window_verified else 'parent-run log; reduced-product interval unverified'
    if instrument == 'SNAP' and conditions.get('window_scope') == 'reviewed_full_parent_run':
        window_verified = True
        basis = conditions['product_window_basis']
    if instrument == 'PG3' and _full_powgen_run(source, metadata, script_sha256):
        window_verified = True
        basis = 'one matching NeXus file in hashed SNSPowderReduction script; no time/split selection'
    channels = []
    for quantity in ('temperature', 'magnetic_field', 'pressure', 'voltage'):
        metric = metadata.get(quantity)
        if not isinstance(metric, dict) or metric.get('role') != 'measured' or metric.get('status') != 'recorded':
            continue
        if metric.get('requires_binding') and not metric.get('binding_verified'):
            continue
        value, coverage = metric.get('value'), metric.get('coverage_fraction')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (value, coverage)):
            continue
        if coverage < 1 - 1e-6 or not metric.get('source') or not metric.get('unit'):
            continue
        allowed = ('quantity', 'value', 'unit', 'source', 'minimum', 'maximum',
                   'statistic', 'coverage_fraction', 'role', 'status', 'acquisition_window')
        channel = {k: metric[k] for k in allowed if k in metric}
        channel['quantity'] = quantity
        channels.append(channel)
    payload = {'$schema': SCHEMA, 'instrument': instrument, 'run_number': int(run_number),
        'sample_id': str(metadata.get('sample_id') or '')[:160],
        'sample_name': str(metadata.get('sample_name') or '')[:240],
        'acquisition_window': conditions.get('acquisition_window') or {},
        'product_window_verified': window_verified, 'product_window_basis': basis,
        'source_metadata_sha256': conditions.get('metadata_sha256'), 'channels': channels}
    # No trajectories or arbitrary metadata objects may spill into this header.
    if len(json.dumps(payload, allow_nan=False).encode('ascii')) > 6000:
        raise ValueError('Acquisition context exceeds the powder-header limit')
    return payload


def stage_manual_acquisition(source, destination, acquisition, provenance):
    """Copy a manual-profile input; the facility source is never edited."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Acquisition staging must not overwrite the source')
    raw = source.read_bytes()
    lines = raw.splitlines(keepends=True)
    existing = [line for line in lines[:64] if line.startswith(PREFIX)]
    payload = json.loads(existing[0][len(PREFIX):]) if existing else dict(provenance)
    payload['acquisition'] = acquisition
    marker = PREFIX + json.dumps(payload, separators=(',', ':'), allow_nan=False).encode('ascii') + b'\n'
    if len(marker) > 8192:
        raise ValueError('Input provenance exceeds the powder-header limit')
    lines = [line for line in lines if not line.startswith(PREFIX)]
    lines.insert(1 if lines and not lines[0].startswith(b'BANK') else 0, marker)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b''.join(lines))
    return destination
