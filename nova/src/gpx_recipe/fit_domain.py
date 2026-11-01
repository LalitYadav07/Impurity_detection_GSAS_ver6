"""Identity of the observations used in a weighted least-squares comparison."""
import hashlib
import json

DOMAIN_SCHEMA = "weighted-observations/v4"
PREVIOUS_DOMAIN_SCHEMA = "weighted-observations/v3"
LEGACY_DOMAIN_SCHEMA = "weighted-observations/v2"
DOMAIN_SCHEMAS = (DOMAIN_SCHEMA, PREVIOUS_DOMAIN_SCHEMA, LEGACY_DOMAIN_SCHEMA)
NATIVE_FIT_CONVENTION = "GSAS-II errRefine searchsorted upper bin inclusive/v1"
DOMAIN_FIELDS = ("coordinate", "x", "observed", "weights", "weight_factor", "valid", "fitted", "limits", "exclusions")


def native_powder_fit_mask(coordinates, limits, exclusions=()):
    """Match errRefine's index slice, including the first bin at/above hi."""
    import numpy as np
    x = np.asarray(np.ma.getdata(coordinates), dtype=float)
    mask = np.zeros(len(x), dtype=bool)
    lo, hi = limits
    mask[np.searchsorted(x, lo):np.searchsorted(x, hi)+1] = True
    for start, end in exclusions:
        mask &= (x < start) | (x > end)
    return mask


def histogram_domain_schema(histogram):
    return histogram.get('fit_domain_schema') or (
        DOMAIN_SCHEMA if histogram.get('fit_range_convention') == NATIVE_FIT_CONVENTION else PREVIOUS_DOMAIN_SCHEMA)


def canonical_observations(histogram):
    """Keep invalid, unused zero-weight placeholders distinct from measurements.

    GSASIIstrIO.GetHistograms clears array masks when reading a refinement.
    An invalid bin can consequently export as None before refinement and as
    its finite placeholder afterward. Its zero weight and invalid/unfitted
    state survive. Only that state is equivalent; fitted or usable samples,
    their coordinates, weights and masks still participate in comparisons.
    Original numeric records remain in the immutable input/GPX artifacts.
    """
    observed = list(histogram['observed'])
    arrays = [histogram.get(key) for key in ('weights', 'valid', 'fitted')]
    if any(not isinstance(values, list) or len(values) != len(observed) for values in arrays):
        return observed
    weights, valid, fitted = arrays
    return [None if v is False and f is False and type(w) in (int, float) and w == 0 else y
            for y, w, v, f in zip(observed, weights, valid, fitted)]


def fit_domain_digest(histogram, *, schema=None):
    """Legacy exports without weights cannot establish comparable Rwp values.

    Calculated intensities deliberately do not participate: a comparison is
    meant to evaluate alternative calculations against the same observations.
    """
    schema = schema or histogram_domain_schema(histogram)
    if schema not in DOMAIN_SCHEMAS:
        raise ValueError('Unknown weighted-observation contract')
    if not all(key in histogram for key in DOMAIN_FIELDS):
        return None
    domain = {key: histogram[key] for key in DOMAIN_FIELDS}
    if schema == DOMAIN_SCHEMA:
        if histogram.get('fit_range_convention') != NATIVE_FIT_CONVENTION:
            return None
        domain['fit_range_convention'] = NATIVE_FIT_CONVENTION
    elif histogram.get('fit_range_convention') == NATIVE_FIT_CONVENTION:
        # Previously sealed reviews retain their original coordinate-filter
        # contract. New artifacts carry the true native domain; reconstruct
        # only the old hash projection when verifying an old review.
        lo, hi = histogram['limits']
        domain['fitted'] = [bool(valid and lo <= x <= hi and not any(a <= x <= b for a,b in histogram['exclusions']))
                            for x,valid in zip(histogram['x'],histogram['valid'])]
    if schema != LEGACY_DOMAIN_SCHEMA:
        domain['observed'] = canonical_observations(domain)
    domain["schema"] = schema
    return hashlib.sha256(json.dumps(domain, sort_keys=True, allow_nan=False).encode()).hexdigest()
