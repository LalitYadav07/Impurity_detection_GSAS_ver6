"""Exact histogram-scoped TOF width release and post-fit qualification."""
import math
import numpy as np

TERMS = ('X', 'Y', 'Z')


def capabilities(histogram):
    inst = histogram.get('instrument', {})
    if 'T' not in str(inst.get('Type', '')):
        return {}
    return {name: float(inst[name]) for name in TERMS
            if type(inst.get(name)) in (int, float) and math.isfinite(inst[name])}


def validate_targets(stage, context):
    hists = {h['key']: h for h in context['histograms']}
    for target in stage['free']:
        if target['kind'] != 'instrument_profile':
            continue
        if not isinstance(target['parameter'], str) or target['parameter'] not in capabilities(hists[target['histogram']]):
            raise ValueError('Release only X, Y or Z present in the selected native TOF histogram; position calibration stays fixed.')
        if any(t['kind'] in {'sample_size', 'sample_microstrain'} and
               t['histogram'] == target['histogram'] for t in stage['free']):
            raise ValueError('Hold sample size/microstrain while adjusting the instrument profile; their width contributions are confounded.')


def gamma_domain(values, lower, upper):
    """Exact quadratic extrema: signed coefficients are allowed."""
    z, x, y = (float(values.get(k, 0)) for k in ('Z', 'X', 'Y'))
    if not all(math.isfinite(v) for v in (z, x, y, lower, upper)) or not 0 < lower < upper:
        raise ValueError('A finite, increasing positive d-spacing domain is required.')
    points = [lower, upper]
    if y and lower < -x / (2*y) < upper:
        points.append(-x / (2*y))
    widths = [z + x*d + y*d*d for d in points]
    if min(widths) < -1e-7:
        raise ValueError('The adjusted TOF Lorentzian width becomes negative in the fitted d-spacing domain.')
    return {'d_min': lower, 'd_max': upper, 'gamma_min': min(widths),
            'gamma_max': max(widths), 'formula': 'gamma(d) = Z + X*d + Y*d^2',
            'interpretation': 'Empirical profile adjustment; not an independently calibrated instrument standard.'}


def audit(project, stage, context, covariance):
    from .advanced import lattice
    selected = [t for t in stage['free'] if t['kind'] == 'instrument_profile']
    if not selected:
        return []
    native = {h['key']: project.histogram(h['name']) for h in context['histograms']}
    rows = []
    for key in sorted({t['histogram'] for t in selected}):
        hist = native[key]
        inst = hist.InstrumentParameters
        if 'T' not in str(inst['Type'][0]):
            raise ValueError('The histogram is no longer a TOF measurement.')
        x = np.asarray(hist.getdata('X'), dtype=float)
        lo, hi = hist.data['Limits'][1]
        used = np.isfinite(x) & (x >= lo) & (x <= hi)
        for a, b in hist.data['Limits'][2:]:
            used &= (x < a) | (x > b)
        d = np.asarray(lattice(project).Pos2dsp(inst, x[used]), dtype=float)
        if len(d) < 20 or not np.all(np.isfinite(d)) or np.any(d <= 0):
            raise ValueError('The adjusted profile has insufficient qualified fitted points.')
        values = {name: float(inst[name][1]) for name in TERMS if name in inst}
        check = gamma_domain(values, float(min(d)), float(max(d)))
        rows.append({'histogram': key, 'coefficients': values, **check})
    vary = [str(v) for v in covariance.get('varyList', [])]
    cov = np.asarray(covariance.get('covMatrix', []), dtype=float)
    names = {f':{native[t["histogram"]].id}:{t["parameter"]}' for t in selected}
    if cov.shape != (len(vary), len(vary)):
        raise ValueError('Profile identifiability needs fresh native covariance.')
    if (not names.issubset(set(vary)) or not np.all(np.isfinite(cov)) or
            any(cov[vary.index(name), vary.index(name)] <= 0 for name in names)):
        raise ValueError('Profile identifiability needs finite positive variances for every released coefficient.')
    for i, name in enumerate(vary):
        if name not in names:
            continue
        for j in range(len(vary)):
            if i == j or cov[i,i] <= 0 or cov[j,j] <= 0:
                continue
            correlation = float(cov[i,j] / math.sqrt(cov[i,i]*cov[j,j]))
            if abs(correlation) > .995:
                raise ValueError('Adjusted profile is not independently identifiable: '+name+
                                 ' and '+vary[j]+f' have correlation {correlation:.5f}. Release fewer terms or use a better reference.')
    return rows
