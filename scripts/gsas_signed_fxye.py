"""Preserve signed SNAP observations in the standard GSAS-II FXYE reader.

The GSAS-II reader masks all nonpositive intensities, even when a reduced,
background-subtracted observation has a valid positive uncertainty. For an
explicitly marked, staged SNAP histogram, restore observations/weights before
GSAS-II's ordinary TOF/bin-width conversion. Invalid bins keep zero weights.
Unmarked data and other importer paths retain their established behavior.
"""
import json
import math
from pathlib import Path
import re

import numpy as np


def restore_snap_signed_observations(reader, path):
    if getattr(reader, 'formatName', '') != 'GSAS powder data':
        return False
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        return False
    provenance = None
    with path.open('rb') as stream:
        for _ in range(64):
            line = stream.readline(8193)
            if not line or line.startswith(b'BANK'):
                break
            if len(line) > 8192:
                return False
            if line.startswith(b'# RADAR-PD input provenance: '):
                provenance = json.loads(line.split(b': ',1)[1])
                break
    if not isinstance(provenance,dict) or provenance.get('schema') != 'radar-pd-input-provenance/v1' or provenance.get('instrument') != 'SNAP':
        return False
    x, y, weights = [], [], []
    banks = 0
    with path.open() as stream:
        for line in stream:
            if line.startswith('BANK'):
                banks += 1
                if banks != 1 or not re.fullmatch(r'BANK\s+1\s+\d+\s+\d+\s+SLOG\s+.*\s+FXYE\s*',line):
                    raise ValueError('Marked SNAP signed import requires one staged SLOG/FXYE histogram')
                continue
            if not banks or not line.strip() or line.lstrip().startswith('#'):
                continue
            values = line.split()
            if len(values) != 3:
                raise ValueError('Malformed marked SNAP signed observation')
            tof, intensity, sigma = map(float,values)
            if not math.isfinite(tof) or tof <= 0 or not math.isfinite(sigma) or sigma < 0:
                raise ValueError('Invalid marked SNAP coordinate or uncertainty')
            if sigma > 0 and not math.isfinite(intensity):
                raise ValueError('Nonfinite positive-weight SNAP observation')
            x.append(tof/100.)  # GSAS reader convention before TOF conversion.
            y.append(intensity if sigma > 0 else 0.)
            weights.append(1./sigma**2 if sigma > 0 else 0.)
    imported = reader.powderdata
    if not banks or len(x) != len(imported[0]) or not np.allclose(x,imported[0],rtol=1e-12,atol=1e-10):
        raise ValueError('Marked SNAP signed observation grid disagrees with the GSAS-II import')
    if not np.all(np.diff(x) > 0) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(weights)):
        raise ValueError('Marked SNAP observation arrays are nonfinite or unordered')
    imported[1] = np.asarray(y,dtype=float)
    imported[2] = np.asarray(weights,dtype=float)
    return True
