"""Experimental commensurate-k survey, independent of LLMs and refinement.

SeeK-path proposes points in the *original* reciprocal basis. This is a finite
high-symmetry survey, not an enumeration of all commensurate propagation vectors.
Profile ranking consumes calibrated, native-simulated magnetic-only templates;
it does not construct magnetic phases or infer that a residual is magnetic.
"""
from __future__ import annotations

from fractions import Fraction
import math
import numpy as np


def _vector(value, *, denominator=96, tolerance=1e-8):
    if len(value) != 3:
        raise ValueError('A propagation vector needs three components')
    result = []
    for item in value:
        raw = Fraction(str(item))
        limited = raw.limit_denominator(denominator)
        if not math.isfinite(float(raw)) or abs(float(raw - limited)) > tolerance:
            raise ValueError('A coordinate is outside the declared rational search budget')
        result.append(limited)
    return tuple(result)


def _operations(operations):
    rotations, translations = [], []
    identity = np.eye(3, dtype=int)
    for operation in operations:
        matrix = np.asarray(operation['rotation'], dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix, np.rint(matrix), atol=1e-10, rtol=0):
            raise ValueError('Parent rotations must be integral finite 3x3 matrices')
        matrix = np.rint(matrix).astype(int)
        if abs(round(np.linalg.det(matrix))) != 1:
            raise ValueError('Parent rotations must be unimodular')
        inverse_transpose = np.rint(np.linalg.inv(matrix).T).astype(int)
        if not any(np.array_equal(inverse_transpose, r) for r in rotations):
            rotations.append(inverse_transpose)
        if np.array_equal(matrix, identity):
            translation = _vector(operation['translation'])
            translations.append(translation)
    if not translations or not any(all(v.denominator == 1 for v in t) for t in translations):
        raise ValueError('A complete parent inventory including the identity is required')
    return rotations, translations


def reciprocal_equivalent(left, right, operations):
    """Same parent k-star, including +/-k and *actual* reciprocal translations.

    In centred conventional cells an integral coefficient is not sufficient to
    be a reciprocal-lattice vector. For example FCC X=(0,1,0) is not Gamma.
    Screw/glide translations are not centring translations.
    """
    a, b = _vector(left), _vector(right)
    rotations, translations = _operations(operations)
    for rotation in rotations:
        transformed = tuple(sum(int(rotation[i, j]) * b[j] for j in range(3)) for i in range(3))
        for sign in (1, -1):
            delta = tuple(a[i] - sign * transformed[i] for i in range(3))
            if all(x.denominator == 1 for x in delta) and all(
                sum(delta[i] * t[i] for i in range(3)).denominator == 1 for t in translations
            ):
                return True
    return False


def point_candidates(output, operations, *, maximum_denominator=8, tolerance=1e-8):
    """Interpret get_path_orig_cell output, preserving rational parent-basis k."""
    _operations(operations)
    if type(maximum_denominator) is not int or not 1 <= maximum_denominator <= 96:
        raise ValueError('Choose a denominator budget between 1 and 96')
    rows, excluded = [], []
    points = {'GAMMA': [0, 0, 0], **output['point_coords']}
    for label, coordinates in points.items():
        try:
            k = _vector(coordinates, denominator=maximum_denominator, tolerance=tolerance)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            excluded.append(dict(label=label, coordinates=list(coordinates), reason='Not rational within the declared denominator/tolerance budget'))
            continue
        existing = next((r for r in rows if reciprocal_equivalent(r['k'], k, operations)), None)
        if existing is not None:
            existing['labels'].append(label)
        else:
            rows.append(dict(k=[str(x) for x in k], labels=[label]))
    return dict(schema='radar-commensurate-k-points/v1', candidates=rows, excluded=excluded,
                basis='original GPX parent reciprocal basis', scope='rational SeeK-path high-symmetry points only',
                exhaustive_commensurate_search=False, maximum_denominator=maximum_denominator,
                input_nonprimitive=bool(output.get('is_supercell')))


def discover_points(structure, operations, *, expected_space_group, maximum_denominator=8,
                    symprec=1e-5):
    """Local SeeK-path/spglib only; never upload the crystal structure."""
    import seekpath
    import spglib
    dataset = spglib.get_symmetry_dataset(structure, symprec=symprec)
    if dataset is None or dataset.number != expected_space_group:
        raise ValueError('Expanded GPX structure does not reproduce the declared parent space group at this symmetry tolerance')
    def operation_key(rotation, translation):
        t = np.asarray(translation, dtype=float) % 1
        t[np.isclose(t, 1, atol=1e-7, rtol=0)] = 0
        return tuple(np.asarray(rotation, dtype=int).ravel()) + tuple(np.round(t, 7))
    supplied = {operation_key(o['rotation'], o['translation']) for o in operations}
    recovered = {operation_key(r, t) for r, t in zip(dataset.rotations, dataset.translations)}
    if supplied != recovered:
        raise ValueError('Recovered translations/rotations differ from the exact GPX parent inventory; review setting or symmetry tolerance')
    output = seekpath.get_path_orig_cell(structure, symprec=symprec)
    result = point_candidates(output, operations, maximum_denominator=maximum_denominator)
    result.update(seekpath_version=seekpath.__version__, spglib_version=spglib.__version__,
                  parent_space_group=dataset.number, symprec=symprec)
    return result


def recover_points(structure,operations,*,expected_space_group=None,maximum_denominator=8):
    """Retain close split sites by tightening, never broadening, precision.

    Every accepted result must reproduce the complete supplied GPX operation
    inventory. Trying a smaller distance tolerance does not merge, move or
    symmetrize any input atom and cannot silently promote the parent symmetry.
    """
    import spglib
    for precision in (1e-5,1e-6,1e-7):
        dataset=spglib.get_symmetry_dataset(structure,symprec=precision)
        if dataset is None:continue
        try:return discover_points(structure,operations,expected_space_group=expected_space_group or dataset.number,
                                   maximum_denominator=maximum_denominator,symprec=precision)
        except ValueError:continue
    raise ValueError('No tested coordinate precision reproduces the exact GPX parent operations; review close/split sites or the parent setting')


def rank_profiles(observed, nuclear, weights, templates, *, mask=None,
                  minimum_delta_chi2=25., minimum_score_gap=9., complete=True):
    """Fit one nonnegative intensity amplitude per native magnetic template.

    All templates use the same native bins, weights, limits and exclusions.
    The signed nuclear residual is retained: predicted peaks where no excess
    exists worsen the fit. The screening score is delta-chi2 minus log(N) and a
    2log(number-of-tried-templates) family penalty. It is not a probability or a
    magnetic-order test. Full refinement and experiment-specific validation are
    still required. More flexible/per-peak NNLS is deliberately not used.
    """
    observed, nuclear, weights = [np.ma.asarray(v, dtype=float) for v in (observed, nuclear, weights)]
    if any(not math.isfinite(v) or v <= 0 for v in [minimum_delta_chi2, minimum_score_gap]):
        raise ValueError('Screening thresholds must be finite and positive')
    if observed.ndim != 1 or nuclear.shape != observed.shape or weights.shape != observed.shape:
        raise ValueError('Observed, nuclear and weight arrays must share one native grid')
    valid = np.ones(observed.shape, bool) if mask is None else np.asarray(mask, dtype=bool).copy()
    if valid.shape != observed.shape:
        raise ValueError('The fit-domain mask must match the native grid')
    for values in (observed, nuclear, weights):
        valid &= ~np.ma.getmaskarray(values) & np.isfinite(np.ma.getdata(values))
    valid &= np.ma.getdata(weights) > 0
    if valid.sum() < 8:
        raise ValueError('At least eight valid, weighted native bins are required')
    residual = np.ma.getdata(observed - nuclear)[valid]
    w = np.ma.getdata(weights)[valid]
    null_chi2 = float(np.dot(w, residual * residual))
    rows = []
    seen = set()
    for template in templates:
        key = str(template['id'])
        if key in seen:
            raise ValueError('Template IDs must be unique')
        seen.add(key)
        values = np.ma.asarray(template['intensity'], dtype=float)
        if values.shape != observed.shape or np.ma.getmaskarray(values)[valid].any() or not np.isfinite(np.ma.getdata(values)[valid]).all():
            raise ValueError('Every template must cover the same valid native bins')
        vector = np.ma.getdata(values)[valid]
        if np.any(vector < -1e-10 * max(1., float(np.max(np.abs(vector))))):
            raise ValueError('A template must be a magnetic-only nonnegative intensity profile')
        vector = np.maximum(vector, 0.)
        denominator = float(np.dot(w, vector * vector))
        numerator = float(np.dot(w, vector * residual))
        maximum=template.get('maximum_amplitude',float('inf'))
        if maximum<=0 or math.isnan(maximum):raise ValueError('A template amplitude bound must be positive')
        amplitude = min(maximum,max(0., numerator / denominator)) if denominator > 0 else 0.
        chi2 = float(np.dot(w, (residual - amplitude * vector) ** 2))
        rows.append(dict(id=key, k=[str(x) for x in _vector(template['k'])], amplitude=amplitude,
                         delta_chi2=max(0., null_chi2 - chi2), chi2=chi2,
                         evaluable=denominator > 0))
    families = {}
    for row in rows:
        families.setdefault(tuple(row['k']), []).append(row)
    ranked = []
    for k, members in families.items():
        best = max(members, key=lambda r: r['delta_chi2'])
        ranked.append(dict(k=list(k), best_template=best['id'], delta_chi2=best['delta_chi2'],
                           score=best['delta_chi2'] - math.log(int(valid.sum())) - 2 * math.log(len(members)),
                           evaluated_templates=len(members), evaluable_templates=sum(r['evaluable'] for r in members)))
    ranked.sort(key=lambda r: (-r['score'], r['k']))
    status = 'no_support'
    if ranked and ranked[0]['delta_chi2'] >= minimum_delta_chi2 and ranked[0]['score'] > 0:
        status = 'shortlist' if len(ranked) == 1 or ranked[0]['score'] - ranked[1]['score'] >= minimum_score_gap else 'ambiguous'
    if not complete:
        status = 'incomplete'
    return dict(schema='radar-commensurate-k-survey/v1', status=status, complete=bool(complete),
                null_chi2=null_chi2, valid_points=int(valid.sum()), families=ranked, templates=rows,
                magnetic_order_established=False, unique_k_established=False,
                notes=['Screening scores are not probabilities.',
                       'Coverage and sampled magnetic directions/sites must accompany the ranking.',
                       'A correct k can be missed by an incomplete magnetic-template sample.',
                       'Nuclear-model errors, impurity peaks and background errors can mimic support.'])
