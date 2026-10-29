"""Conservative reduction of a finite, declared symmetry-search portfolio.

This does not choose a physical structure. Failed optimizations remain evidence:
a lower-residual unresolved checkpoint must not vanish behind a worse converged
local minimum. Hard physical rejections are distinct from unresolved numerics.
"""
from __future__ import annotations

import math

TERMINAL = {'valid', 'physical_rejection', 'site_incompatible', 'optimization_unresolved',
            'technical_failure', 'deferred'}


def summarize_portfolio(expected_trial_ids, results, *, fit_band_percent):
    """Return coverage, competitive qualified fits and unresolved competition.

    Expected IDs and the adequacy band come from the approved protocol. A `valid`
    row must have its parameter effects and input-data contract verified by the
    native worker/auditor. Rwp is a percentage; this function never changes a
    worker's scientific disposition or retries a refinement.
    """
    expected = list(expected_trial_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('A unique, nonempty declared trial set is required')
    if not math.isfinite(fit_band_percent) or fit_band_percent < 0:
        raise ValueError('An explicit nonnegative Rwp adequacy band is required')
    by_id = {}
    for row in results:
        uid = row['id']
        if uid not in expected or uid in by_id or row['status'] not in TERMINAL:
            raise ValueError('Unexpected, duplicated or unterminated trial receipt')
        if row.get('rwp') is not None and (not math.isfinite(row['rwp']) or row['rwp'] < 0):
            raise ValueError('A finite nonnegative Rwp is required when present')
        if row['status'] == 'valid' and (row.get('rwp') is None or row.get('effects_verified') is not True
                                       or row.get('inputs_verified') is not True):
            raise ValueError('An unverified native output cannot be a qualified fit')
        by_id[uid] = row
    missing = [uid for uid in expected if uid not in by_id]
    incomplete = [r['id'] for r in by_id.values() if r['status'] in {'technical_failure', 'deferred'}]
    valid = sorted((r for r in by_id.values() if r['status'] == 'valid'), key=lambda r:r['rwp'])
    best = valid[0]['rwp'] if valid else None
    competitive = [r['id'] for r in valid if r['rwp'] <= best+fit_band_percent] if valid else []
    unresolved = [r['id'] for r in by_id.values()
                  if r['status'] == 'optimization_unresolved'
                  and (best is None or r.get('rwp') is None or r['rwp'] <= best+fit_band_percent)]
    status = ('incomplete' if missing or incomplete else 'optimization_unresolved' if unresolved
              else 'no_qualified_fit' if not valid else 'reviewable_portfolio')
    return dict(schema='radar-symmetry-portfolio-summary/v1', status=status,
                declared_count=len(expected), received_count=len(by_id), missing=missing,
                incomplete=incomplete, unresolved_competitors=unresolved,
                competitive_trial_ids=competitive, best_qualified_rwp=best,
                scientific_assignment='not_established')
