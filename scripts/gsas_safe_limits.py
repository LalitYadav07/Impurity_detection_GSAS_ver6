"""Apply the shared continuous TOF-domain guard to native GSAS-II histograms."""
import copy
import numpy as np

from gpx_recipe.tof_domain import assess_tof_domain, calculate_safe_d_min


def apply_safe_limits(proj):
    """Validate every bank before applying only the established low-tail repair."""
    changes = []
    for hist in proj.histograms():
        if 'Instrument Parameters' not in hist.data:
            continue
        inst = hist.data['Instrument Parameters'][0]
        token = inst.get('Type', '')
        if isinstance(token, (list, tuple, np.ndarray)):
            token = token[0] if len(token) else ''
        if 'T' not in str(token):
            continue
        try:
            limits = hist.data['Limits']
            domain = assess_tof_domain(inst, limits[1])
            if domain['trimmed_lower']:
                changes.append((hist, limits, float(limits[1][0]),
                                domain['native_limits'][0], domain['fitted_d'][0]))
        except (ValueError, TypeError, KeyError, IndexError, FloatingPointError) as exc:
            raise ValueError(f'[SafeLimits] Histogram {hist.name}: {exc}') from exc
    for hist, limits, old, new, safe_d in changes:
        updated = copy.deepcopy(limits)
        updated[1] = list(updated[1])
        updated[1][0] = float(new)
        hist.data['Limits'] = updated
        print(f'[SafeLimits] Histogram {hist.name}: Raising min limit {old:.2f} -> {new:.2f} (d_min={safe_d:.5g} Å)')
    return bool(changes)
