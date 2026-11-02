"""Preserve finite signed observations from provenance-bound staged TOF files.

GSAS-II's general FXYE reader zeros both y and weight when y <= 0. That is
not a validity rule for reduced data with supplied positive uncertainty.
Only recognized single-bank SLOG/FXYE sources and a native PNT profile enter
this path. No source file, positive observation, or physical model is changed.
"""
import json
from hashlib import sha256
import math
from pathlib import Path
import re

import numpy as np


_MAX_BYTES = 64 * 1024 * 1024
_FAMILIES = {
    'NOM': {'NOMAD individual GSAS/SLOG', 'NOMAD IDL individual GSAS/SLOG'},
    'PG3': {'POWGEN individual GSAS/FXYE'},
    'SNAP': {'SNAP individual GSAS/SLOG'},
}


def read_staged_tof_source(path, *, instruments=None):
    """Read one bounded source once; unrelated inputs retain native behavior."""
    path = Path(path)
    size = path.stat().st_size
    provenance = None
    with path.open('rb') as stream:
        for _ in range(64):
            line = stream.readline(8193)
            if not line or line.startswith(b'BANK'):
                break
            if len(line) > 8192:
                if provenance is not None:
                    raise ValueError('Staged TOF provenance exceeds the header limit')
                return None
            if line.startswith(b'# RADAR-PD input provenance: '):
                if provenance is not None:
                    raise ValueError('Duplicate staged observation provenance')
                provenance = json.loads(line.split(b': ', 1)[1])
        if not isinstance(provenance, dict) or provenance.get('schema') != 'radar-pd-input-provenance/v1':
            return None
        instrument = provenance.get('instrument')
        if not isinstance(instrument, str) or instrument not in _FAMILIES or (instruments is not None and instrument not in instruments):
            return None
        # Early SNAP staging predates the explicit family field. Keep that
        # already supported path, without extending it to NOMAD or POWGEN.
        family = provenance.get('reduction_family')
        if (not isinstance(family, str) or family not in _FAMILIES[instrument]) and not (instrument == 'SNAP' and family is None):
            return None
        if size > _MAX_BYTES:
            raise ValueError('Staged TOF file exceeds the import limit')
        stream.seek(0)
        raw = stream.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError('Staged TOF file exceeds the import limit')
    rows, banks, declared = [], 0, None
    for line in raw.decode('latin-1').splitlines():  # Match the native GSAS text reader.
        if line.startswith('BANK'):
            banks += 1
            match = re.fullmatch(r'BANK\s+1\s+(\d+)\s+(\d+)\s+SLOG\s+.*\s+FXYE\s*', line)
            if banks != 1 or match is None:
                raise ValueError('Marked TOF import requires one staged bank-1 SLOG/FXYE histogram')
            declared = tuple(map(int, match.groups()))
            continue
        if not banks or not line.strip() or line.lstrip().startswith('#'):
            continue
        values = line.split()
        if len(values) != 3:
            raise ValueError('Malformed marked TOF observation')
        tof, intensity, sigma = map(float, values)
        if not math.isfinite(tof) or tof <= 0 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError('Invalid marked TOF coordinate or uncertainty')
        rows.append((tof, intensity, sigma))
    if banks != 1 or len(rows) < 2 or declared != (len(rows), len(rows)):
        raise ValueError('Marked TOF source count disagrees with its bank declaration')
    invalid = provenance.get('invalid_count_bin_indices', [])
    if (not isinstance(invalid, list) or any(type(i) is not int or not 0 <= i < len(rows) for i in invalid)
            or len(set(invalid)) != len(invalid)):
        raise ValueError('Invalid source observation mask indices')
    invalid = set(invalid)
    for i, (_, intensity, sigma) in enumerate(rows):
        if sigma > 0 and i not in invalid and not math.isfinite(intensity):
            raise ValueError('Nonfinite positive-weight TOF observation')
    return {'path': path.resolve(), 'rows': rows, 'invalid': invalid, 'instrument': instrument,
            'sha256': sha256(raw).hexdigest()}


def restore_staged_tof_observations(reader, source, instrument_parameters):
    """Restore only native nonpositive-y suppression; keep real masks invalid."""
    if source is None:
        return False
    with source['path'].open('rb') as stream:
        current = stream.read(_MAX_BYTES + 1)
    if len(current) > _MAX_BYTES or sha256(current).hexdigest() != source['sha256']:
        raise ValueError('Staged TOF source changed during native import')
    parameters = instrument_parameters[0] if isinstance(instrument_parameters, (list, tuple)) else instrument_parameters
    token = parameters.get('Type') if isinstance(parameters, dict) else None
    if isinstance(token, (list, tuple, np.ndarray)):
        token = token[0] if len(token) else None
    if token != 'PNT':
        raise ValueError('Marked staged TOF observations require native PNT instrument parameters')
    entry = getattr(reader, 'powderentry', [])
    if (getattr(reader, 'formatName', '') != 'GSAS powder data'
            or not getattr(reader, 'GSAS', False) or getattr(reader, 'clockWd', {})
            or getattr(reader, 'numbanks', None) != 1 or len(entry) < 3 or entry[2] != 1
            or Path(entry[0]).resolve() != source['path']):
        raise ValueError('Staged TOF source does not match the native reader file and bank')
    original = reader.powderdata[:3]
    actual = [np.asarray(np.ma.getdata(a)) for a in original]
    rows = source['rows']
    x = np.array([tof / 100. for tof, _, _ in rows])
    if (any(a.ndim != 1 or a.shape != x.shape for a in actual)
            or not np.array_equal(actual[0], x) or not np.all(np.diff(x) > 0)):
        raise ValueError('Marked TOF observation grid disagrees with the GSAS-II import')
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
