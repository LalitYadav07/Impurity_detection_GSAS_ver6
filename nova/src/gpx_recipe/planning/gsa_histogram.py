"""Recover physical TOF bins under an explicitly established SaveGSS contract.

Mantid SaveGSS SLOG writes arithmetic histogram centres and bin-width-scaled
Y/E. A native GSAS import can instead treat these coordinates as edges. SLOG
alone does not identify the producer convention. Callers must establish it and
the TOF histogram input independently; this reader never guesses a contract.

The supported grid starts with full logarithmic bins and may have a terminal
bin between 25% and 125% of the full width, as documented by Mantid Rebin.
Other grids need the original physical histogram edges.
"""
import hashlib
from pathlib import Path

import numpy as np

VERSION = 'powgen-Mantid-SLOG-physical-histogram/1'
PRODUCER_CONTRACT = 'SaveGSS-TOF-histogram-centres-and-binwidth-scaled-YE'
_MAX_BYTES = 64*1024**2


def _printing_half_unit(token):
    parts = token.lower().split('e')
    mantissa = parts[0]
    exponent = int(parts[1]) if len(parts) == 2 else 0
    decimals = len(mantissa.split('.')[1]) if '.' in mantissa else 0
    return .5*10.**(exponent-decimals)


def read_mantid_slog_histogram(path, *, producer_contract):
    """Return signed physical density/error arrays and a source/contract receipt.

    The first full logarithmic bin fixes the starting edge. Arithmetic centres
    then fix every subsequent edge by recurrence, including the terminal bin.
    Encoded error columns remain supplied marginal errors; no Poisson law,
    independent-bin covariance or experimental qualification is inferred.
    """
    if producer_contract != PRODUCER_CONTRACT:
        raise ValueError('An independently established SaveGSS TOF histogram-centre contract is required')
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(_MAX_BYTES+1)
    if not 0 < len(raw) <= _MAX_BYTES:
        raise ValueError('Bounded nonempty GSA input required')
    lines = raw.decode('utf-8-sig').splitlines()
    banks = [i for i, line in enumerate(lines) if line.startswith('BANK')]
    if len(banks) != 1:
        raise ValueError('Exactly one explicit bank is required')
    index = banks[0]
    descriptor = lines[index].split()
    if len(descriptor) < 10 or descriptor[4] != 'SLOG' or descriptor[-1] != 'FXYE':
        raise ValueError('Only declared SLOG FXYE histogram input is supported')
    count = int(descriptor[2])
    bank = int(descriptor[1])
    first, last, declared_step = map(float, descriptor[5:8])
    if (bank <= 0 or count != int(descriptor[3]) or not 4 <= count <= 250000
            or not np.isfinite([first, last, declared_step]).all() or min(first, last, declared_step) <= 0):
        raise ValueError('Finite positive header coordinates/step and a bounded row count are required')
    data_lines = [line for line in lines[index+1:] if line.strip()]
    if len(data_lines) != count:
        raise ValueError('The declared FXYE row count differs from the file')
    rows = np.array([[float(v) for v in line.split()] for line in data_lines], float)
    if rows.shape != (count, 3) or not np.isfinite(rows).all():
        raise ValueError('Finite complete FXYE rows required')
    centers, encoded_y, encoded_e = rows.T
    if np.any(centers <= 0) or np.any(np.diff(centers) <= 0) or np.any(encoded_e < 0):
        raise ValueError('Positive increasing centres and nonnegative encoded errors required')
    ratios = np.diff(centers)/centers[:-1]
    step = float(np.median(ratios[:-1]))
    tolerance = min(5e-10, 1e-5*step)
    # The last centre interval includes the possibly adjusted terminal bin.
    if step <= 0 or np.max(abs(ratios[:-1]-step)) > tolerance:
        raise ValueError('Nonlogarithmic interior requires original histogram edges')
    if abs(step-declared_step) > _printing_half_unit(descriptor[7])+2e-12:
        raise ValueError('Header logarithmic step contradicts the encoded centres')
    edges = np.empty(count+1)
    edges[0] = 2*centers[0]/(2+step)
    for i, center in enumerate(centers):
        edges[i+1] = 2*center-edges[i]
    widths = np.diff(edges)
    if np.any(edges <= 0) or np.any(widths <= 0) or not np.isfinite(edges).all():
        raise ValueError('Invalid reconstructed physical bins')
    relative = widths/edges[:-1]
    if (np.max(abs(relative[:-1]-step)) > tolerance
            or not .25*step-tolerance <= relative[-1] <= 1.25*step+tolerance):
        raise ValueError('Bins violate the logarithmic/documented-terminal-adjustment contract')
    if (abs(edges[0]-first) > _printing_half_unit(descriptor[5])+1e-8
            or abs(centers[-1]-last) > _printing_half_unit(descriptor[6])+1e-8):
        raise ValueError('Header endpoints contradict the declared SaveGSS producer convention')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        density, error = encoded_y/widths, encoded_e/widths
    if not np.isfinite(density).all() or not np.isfinite(error).all():
        raise ValueError('Unrepresentable physical histogram density or error')
    metadata = {
        'schema': VERSION, 'source': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'producer_contract': producer_contract, 'source_bank': bank, 'x_unit': 'microsecond',
        'encoded_rows': count, 'physical_histogram_bins': count,
        'encoded_coordinates_are_arithmetic_bin_centres': True,
        'regular_log_step_from_interior_centres': step, 'header_nominal_log_step': declared_step,
        'first_full_log_bin_assumed_under_witnessed_producer_contract': True,
        'maximum_interior_centre_ratio_residual': float(np.max(abs(ratios[:-1]-step))),
        'last_bin_fraction_of_full_log_bin': float(relative[-1]/step),
        'documented_Rebin_terminal_fraction_bounds': [.25, 1.25],
        'signed_intensity_and_zero_error_rows_preserved': True,
        'native_extra_half_bin_shift_or_nonpositive_row_coercion_applied': False,
        'all_encoded_YE_rows_have_reconstructed_bins': True,
        'supplied_errors_are_marginal_not_an_independent_count_law': True,
        'experimental_accuracy_certified': False,
    }
    return {'edges': edges, 'centers': centers, 'y': density, 'e': error,
            'encoded_y': encoded_y, 'encoded_e': encoded_e}, metadata
