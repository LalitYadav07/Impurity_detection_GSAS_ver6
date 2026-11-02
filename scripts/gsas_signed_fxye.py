"""Preserve supplied-uncertainty observations in native neutron TOF imports.

GSAS-II's general FXYE reader zeros both y and weight when y <= 0. That is
not a validity rule for reduced data with supplied positive uncertainty.
Only exact native SLOG/FXYE sources and a loaded PNT profile enter this path.
An acquisition/provenance claim is not required to interpret supplied sigma.
No source file, positive observation, or physical model is changed.
"""
import json
from hashlib import sha256
import math
from pathlib import Path
import re

import numpy as np


_MAX_BYTES = 64 * 1024 * 1024
_MAX_BANKS = 256
_PROVENANCE_PREFIX = b'# RADAR-PD input provenance: '


def read_staged_tof_source(path, *, instruments=None, defer_unmarked_errors=False):
    """Defer unmarked FXYE validation until the native profile is known.

    A CW FXYE file can legitimately contain coordinates that are invalid for
    TOF. Deferred errors are raised only if its loaded profile is PNT. Marked
    sources retain their existing strict contract, and this never authorizes
    a different reader to bypass a malformed neutron TOF source.
    """
    try:
        return _read_tof_source(path, instruments=instruments)
    except ValueError as exc:
        if not defer_unmarked_errors:
            raise
        path = Path(path)
        with path.open('rb') as stream:
            probe = stream.read(min(path.stat().st_size, max(_MAX_BYTES, 512 * 1024)) + 1)
        lines = probe.splitlines()
        if (any(line.startswith(_PROVENANCE_PREFIX) for line in lines)
                or not any(line.startswith(b'BANK') and line.split()[-1:] == [b'FXYE'] for line in lines)):
            raise
        return {'path': path.resolve(), 'marked': False, 'deferred_error': str(exc)}


def _read_tof_source(path, *, instruments=None):
    """Read bounded GSAS banks without deriving any acquisition metadata.

    The historical name is retained for callers. ``instruments`` is used only
    by the legacy SNAP-specific entry point; the native import path does not
    require or trust an instrument claim. Existing explicit invalid indices
    can restrict validity, but can never make an observation valid.
    """
    path = Path(path)
    size = path.stat().st_size
    provenance = None
    if size > _MAX_BYTES:
        with path.open('rb') as stream:
            probe = stream.read(512 * 1024)
        if any(line.startswith(_PROVENANCE_PREFIX) or
               (line.startswith(b'BANK') and line.split()[-1:] == [b'FXYE'])
               for line in probe.splitlines()):
            raise ValueError('TOF file exceeds the import limit')
        return None
    with path.open('rb') as stream:
        raw = stream.read(_MAX_BYTES + 1)
    lines = raw.decode('latin-1').splitlines()  # Match the native text reader.
    first_bank = next((i for i,line in enumerate(lines) if line.startswith('BANK')), None)
    header = lines if first_bank is None else lines[:first_bank]
    for i,line in enumerate(header):
        if not line.startswith(_PROVENANCE_PREFIX.decode('ascii')):
            continue
        if i >= 64 or len(line.encode('latin-1')) > 8192:
            raise ValueError('TOF provenance exceeds the header limit')
        if provenance is not None:
            raise ValueError('Duplicate observation provenance')
        provenance = json.loads(line[len(_PROVENANCE_PREFIX):])
        if not isinstance(provenance, dict) or provenance.get('schema') != 'radar-pd-input-provenance/v1':
            raise ValueError('Unsupported observation provenance')
    instrument = (provenance or {}).get('instrument')
    if instruments is not None and instrument not in instruments:
        return None
    if first_bank is None:
        if provenance is not None:
            raise ValueError('Marked TOF source has no GSAS bank header')
        return None  # Unbanked XY/XYE/XML input retains its native reader.
    if not any(line.startswith('BANK') and line.split()[-1:] == ['FXYE'] for line in lines):
        if provenance is not None:
            raise ValueError('Marked TOF import requires SLOG/FXYE data')
        return None  # Do not validate or reinterpret unrelated native formats.
    if size > _MAX_BYTES or len(raw) > _MAX_BYTES:
        raise ValueError('TOF file exceeds the import limit')
    banks = {}
    bank = None
    for line in lines:
        if line.startswith('BANK'):
            fields = line.split()
            if len(fields) < 6 or not re.fullmatch(r'[+-]?[0-9]+', fields[1]):
                raise ValueError('GSAS bank requires an unambiguous integer label')
            label = int(fields[1])
            if label in banks:
                raise ValueError('Duplicate GSAS bank label')
            if len(banks) >= _MAX_BANKS:
                raise ValueError('TOF file exceeds the bank limit')
            bank = {'label': label, 'kind': fields[4], 'fxye': fields[-1] == 'FXYE',
                    'rows': [], 'closed': False}
            banks[label] = bank
            if not bank['fxye']:
                continue
            if (not all(re.fullmatch(r'\+?[0-9]+', f) and int(f) > 0 for f in fields[2:4])
                    or (bank['kind'] == 'SLOG' and (len(fields) != 10 or
                        not all(math.isfinite(float(f)) for f in fields[5:9])))):
                raise ValueError('Malformed SLOG/FXYE bank declaration')
            bank['declared'] = tuple(map(int, fields[2:4]))
            continue
        if bank is None or not bank['fxye']:
            continue
        if not line.strip():
            if bank['rows']:
                bank['closed'] = True
            continue
        if line.startswith('#'):
            bank['closed'] = True  # Native GetFXYEdata stops at a comment.
            continue
        if bank['closed']:
            raise ValueError('FXYE rows resume after the native end of a bank')
        values = line.split()
        if len(values) != 3:
            raise ValueError('Malformed FXYE observation')
        tof, intensity, sigma = map(float, values)
        if not math.isfinite(tof) or tof <= 0 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError('Invalid FXYE coordinate or uncertainty')
        bank['rows'].append((tof, intensity, sigma))
    candidates = [b for b in banks.values() if b['fxye']]
    if not candidates:
        if provenance is not None:
            raise ValueError('Marked TOF import requires SLOG/FXYE data')
        return None
    if provenance is not None and any(b['kind'] != 'SLOG' for b in candidates):
        raise ValueError('Marked TOF import requires SLOG/FXYE data')
    for bank in candidates:
        rows = bank['rows']
        if len(rows) < 2 or bank['declared'] != (len(rows), len(rows)):
            raise ValueError('TOF source count disagrees with its bank declaration')
    invalid = (provenance or {}).get('invalid_count_bin_indices', [])
    if (not isinstance(invalid, list) or any(type(i) is not int or i < 0 for i in invalid)
            or len(set(invalid)) != len(invalid)):
        raise ValueError('Invalid source observation mask indices')
    if invalid and len(banks) != 1:
        raise ValueError('Source observation mask is ambiguous across multiple banks')
    invalid = set(invalid)
    if invalid and max(invalid) >= len(candidates[0]['rows']):
        raise ValueError('Invalid source observation mask indices')
    for bank in candidates:
        for i, (_, intensity, sigma) in enumerate(bank['rows']):
            if sigma > 0 and i not in invalid and not math.isfinite(intensity):
                raise ValueError('Nonfinite positive-weight TOF observation')
    return {'path': path.resolve(), 'banks': banks, 'invalid': invalid, 'instrument': instrument,
            'marked': provenance is not None, 'sha256': sha256(raw).hexdigest()}


def restore_staged_tof_observations(reader, source, instrument_parameters):
    """Bind the selected native bank, then restore only nonpositive-y suppression."""
    if source is None:
        return False
    parameters = instrument_parameters[0] if isinstance(instrument_parameters, (list, tuple)) else instrument_parameters
    token = parameters.get('Type') if isinstance(parameters, dict) else None
    if isinstance(token, (list, tuple, np.ndarray)):
        token = token[0] if len(token) else None
    if token != 'PNT':
        if source['marked']:
            raise ValueError('Marked TOF observations require native PNT instrument parameters')
        return False  # Leave all non-neutron-TOF reader semantics unchanged.
    if source.get('deferred_error'):
        raise ValueError(source['deferred_error'])
    with source['path'].open('rb') as stream:
        current = stream.read(_MAX_BYTES + 1)
    if len(current) > _MAX_BYTES or sha256(current).hexdigest() != source['sha256']:
        raise ValueError('TOF source changed during native import')
    entry = getattr(reader, 'powderentry', [])
    if (getattr(reader, 'formatName', '') != 'GSAS powder data'
            or not getattr(reader, 'GSAS', False) or getattr(reader, 'clockWd', {})
            or getattr(reader, 'numbanks', None) != len(source['banks']) or len(entry) < 3
            or Path(entry[0]).resolve() != source['path']):
        raise ValueError('TOF source does not match the native reader file and banks')
    bank = source['banks'].get(entry[2])
    if bank is None:
        raise ValueError('Selected native bank label does not match the TOF source')
    if not bank['fxye']:
        return False
    if bank['kind'] != 'SLOG':
        if any(sigma > 0 and intensity <= 0 for _,intensity,sigma in bank['rows']):
            raise ValueError('Selected native neutron TOF bank requires supported SLOG/FXYE observations')
        return False  # No signed observations to preserve; keep the native format contract.
    original = reader.powderdata[:3]
    actual = [np.asarray(np.ma.getdata(a)) for a in original]
    rows = bank['rows']
    x = np.array([tof / 100. for tof, _, _ in rows])
    if (any(a.ndim != 1 or a.shape != x.shape for a in actual)
            or not np.array_equal(actual[0], x) or not np.all(np.diff(x) > 0)):
        raise ValueError('TOF observation grid disagrees with the GSAS-II import')
    y = np.array([intensity for _, intensity, _ in rows])
    sigma = np.array([sigma for _, _, sigma in rows])
    masks = [np.ma.getmaskarray(a).copy() for a in original]
    masked = masks[0] | masks[1] | masks[2]
    inactive = masked | (sigma == 0)
    if source['invalid']:
        inactive[list(source['invalid'])] = True
    try:
        # Use the same Python-float operation as the GSAS FXYE reader. A
        # vectorized reciprocal can round differently on valid positive bins.
        weights = np.array([1. / s ** 2 if s > 0 else 0. for _, _, s in rows])
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError('Staged TOF uncertainties cannot produce finite positive weights') from exc
    if not np.all(np.isfinite(weights)) or np.any((weights == 0) & (sigma > 0)):
        raise ValueError('Staged TOF uncertainties cannot produce finite positive weights')
    positive = (y > 0) & ~inactive
    if (not np.array_equal(actual[1][positive], y[positive])
            or not np.array_equal(actual[2][positive], weights[positive])):
        raise ValueError('Native positive observations changed before TOF restoration')
    restore = (y <= 0) & ~inactive
    native_suppression = (actual[1] == 0) & (actual[2] == 0)
    already_restored = (actual[1] == y) & (actual[2] == weights)
    if np.any(restore & ~(native_suppression | already_restored)):
        raise ValueError('Unexpected native signed-observation state')
    # Copy only y/w. Existing positive values retain their original bits;
    # every preexisting mask remains a mask, including masks on x or y.
    new_y, new_w = actual[1].copy(), actual[2].copy()
    new_y[restore], new_w[restore] = y[restore], weights[restore]
    new_w[inactive] = 0.
    new_y[inactive & ~np.isfinite(new_y)] = 0.
    weight_mask = masks[2] | masked
    if source['invalid']:
        weight_mask[list(source['invalid'])] = True
    reader.powderdata[1] = (np.ma.array(new_y, mask=masks[1]) if np.ma.isMaskedArray(original[1]) else new_y)
    reader.powderdata[2] = (np.ma.array(new_w, mask=weight_mask)
                           if np.any(weight_mask) or np.ma.isMaskedArray(original[2]) else new_w)
    return True


def restore_snap_signed_observations(reader, path, instrument_parameters=None):
    """Compatibility entry point; a native PNT profile is now required."""
    source = read_staged_tof_source(path, instruments={'SNAP'})
    if source is None:
        return False
    return restore_staged_tof_observations(reader, source, instrument_parameters)
