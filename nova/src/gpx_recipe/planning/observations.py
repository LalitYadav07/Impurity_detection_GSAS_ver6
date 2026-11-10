"""Physical histogram observations from an explicitly declared producer contract.

Mantid 6.16.1 SaveGSS SLOG/FXYE writes centers and bin-integrated density/error.
Those records are not the native GSAS-II left-edge/terminal-record convention.
This reader preserves every observation and never infers raw Poisson counts
from a normalized diffraction curve.
"""
from __future__ import annotations

import hashlib
import math
from decimal import Decimal
from pathlib import Path
import numpy as np
from .binning import LOG_CENTER_CONVENTION, log_edges

MANTID_SLOG_CONTRACT = 'mantid-6.16.1-slog-distribution/v1'


def read_mantid_slog(path, *, producer, bank=None, workspace_edges=None):
    """Return physical density/integrals under a hash-bound producer declaration.

    The caller must establish the producer evidence. Merely supplying this
    declaration does not certify a file's historical origin or profile mapping.
    Prefer actual workspace edges; explicit constant-log reconstruction remains
    recorded separately. Only the tested distributed-histogram writer is accepted.
    """
    path = Path(path)
    before = path.stat()
    if not 0 < before.st_size <= 20 * 1024 * 1024:
        raise ValueError('A nonempty reduced SLOG file at most 20 MiB is required')
    if not isinstance(producer, dict) or producer.get('contract') != MANTID_SLOG_CONTRACT:
        raise ValueError('Declare the tested Mantid SLOG producer contract')
    if producer.get('mantid_version') != '6.16.1':
        raise ValueError('This producer version needs its own writer-contract verification')
    if producer.get('multiply_by_bin_width') is not True or producer.get('distribution') is not True:
        raise ValueError('Only explicitly distributed, bin-width-multiplied observations are supported')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if producer.get('file_sha256') != digest:
        raise ValueError('Observation differs from the hash-bound producer declaration')
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Reduced observation changed during the read')
    records, current = {}, None
    for line in raw.decode('utf-8').splitlines():
        text = line.strip()
        if text.startswith('BANK '):
            fields = text.split()
            if len(fields) != 10 or fields[4] != 'SLOG' or fields[-1] != 'FXYE':
                raise ValueError('Expected explicit SLOG FXYE bank records')
            number, bins, lines = map(int, fields[1:4])
            step = float(fields[7])
            if number < 1 or bins != lines or not 3 <= bins <= 250000 or not math.isfinite(step) or step <= 0:
                raise ValueError('Invalid bounded SLOG bank declaration')
            if number in records:
                raise ValueError('Duplicate reduced bank identifier')
            current = {'bins': bins, 'relative_step_text': fields[7], 'rows': []}
            records[number] = current
        elif current is not None and text and not text.startswith(('#', "'")):
            values = text.split()
            if len(values) != 3:
                raise ValueError('Every selected FXYE record must contain exactly three numeric columns')
            current['rows'].append([float(v) for v in values])
            if len(current['rows']) > current['bins']:
                raise ValueError('FXYE row count exceeds its declaration')
    if any(len(r['rows']) != r['bins'] for r in records.values()):
        raise ValueError('Incomplete physical FXYE observations')
    if bank is None:
        if len(records) != 1:
            raise ValueError('Select an explicit bank for a multi-bank or empty reduced file')
        bank = next(iter(records))
    if type(bank) is not int or bank not in records:
        raise ValueError('Selected physical bank is unavailable')
    record = records[bank]
    data = np.asarray(record['rows'], float)
    if not np.isfinite(data).all() or np.any(data[:, 0] <= 0) or np.any(np.diff(data[:, 0]) <= 0) or np.any(data[:, 2] < 0):
        raise ValueError('Invalid physical coordinates, intensity or supplied error')
    centers, integrals, errors = data.T
    if workspace_edges is None:
        if producer.get('coordinate_convention') != LOG_CENTER_CONVENTION:
            raise ValueError('Declare the coordinate convention or provide producer workspace edges')
        edges, edge_receipt = log_edges(centers, convention=LOG_CENTER_CONVENTION)
        nominal = Decimal(record['relative_step_text'])
        rounding = .5 * 10. ** nominal.as_tuple().exponent
        if abs(edge_receipt['ratio'] - 1 - float(nominal)) > rounding + 2e-11:
            raise ValueError('Recorded coordinates disagree with the declared SLOG step')
        edge_receipt['source'] = 'Reconstructed under declared constant-log center convention'
    else:
        edges = np.asarray(workspace_edges, float)
        if edges.shape != (len(centers) + 1,) or not np.isfinite(edges).all() or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
            raise ValueError('Matching positive finite increasing workspace edges required')
        error = float(np.max(abs((edges[:-1] + edges[1:]) / 2 / centers - 1)))
        if error > 1e-8:
            raise ValueError('Workspace edges do not match the recorded center coordinates')
        edge_receipt = {'source': 'Caller-supplied producer workspace edges',
                        'maximum_relative_center_reconstruction_error': error,
                        'edge_array_sha256': hashlib.sha256(edges.astype('<f8').tobytes()).hexdigest()}
    widths = np.diff(edges)
    arrays = {'x': centers.copy(), 'edges': edges.copy(), 'y': integrals / widths,
              'sigma': errors / widths, 'integrated_intensity': integrals.copy(),
              'integrated_sigma': errors.copy()}
    receipt = {'schema': 'powgen-physical-histogram-observation/v1',
               'source': str(path.resolve()), 'sha256': digest, 'bytes': len(raw),
               'contract': MANTID_SLOG_CONTRACT, 'bank': bank,
               'bins': len(centers), 'negative_intensity_bins': int((integrals < 0).sum()),
               'zero_supplied_error_bins': int((errors == 0).sum()),
               'edge_provenance': edge_receipt, 'all_physical_observations_retained': True,
               'producer_identity_independently_verified_by_reader': False,
               'raw_counts_recovered': False, 'covariance_available_from_file': False,
               'uncertainty_semantics': 'Supplied reduced marginal errors; no independent-Poisson interpretation established',
               'experimental_accuracy_certified': False,
               'scope': 'Physical data contract only; instrument-profile coordinate compatibility requires separate calibration evidence'}
    return arrays, receipt
