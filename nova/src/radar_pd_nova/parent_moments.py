"""Display verified native moment components in their parent-cell frame.

This module only converts coordinates. It never refits, changes a receipt,
expands a magnetic orbit, or propagates incomplete component uncertainties.
"""
from __future__ import annotations

import numpy as np


def _numbers(value, shape):
    values = np.asarray(value, dtype=object)
    if values.shape != shape or any(isinstance(v, (bool, np.bool_)) or not isinstance(
            v, (int, float, np.integer, np.floating)) for v in values.flat):
        raise ValueError('expected finite numeric cell, embedding and moment values')
    result = values.astype(float)
    if not np.isfinite(result).all():
        raise ValueError('cell, embedding or moment values are nonfinite')
    return result


def _cell_metric(value):
    cell = _numbers(value, (6,))
    if np.any(cell[:3] <= 0) or np.any(cell[3:] <= 0) or np.any(cell[3:] >= 180):
        raise ValueError('cell lengths or angles are invalid')
    alpha, beta, gamma = np.deg2rad(cell[3:])
    unit = np.array([[1., np.cos(gamma), np.cos(beta)],
                     [np.cos(gamma), 1., np.cos(alpha)],
                     [np.cos(beta), np.cos(alpha), 1.]])
    if np.linalg.eigvalsh(unit).min() <= 1e-10:
        raise ValueError('cell metric is singular or nonphysical')
    return np.diag(cell[:3]) @ unit @ np.diag(cell[:3])


def _parent_frame(row, view):
    names = row.get('moment_phase_names') or {}
    if not row.get('parent') or names.get('parent') != row['parent'] or not names.get('magnetic'):
        raise ValueError('exact parent and magnetic phase names are unavailable')
    phases = view.get('phases') or []
    parents = [p for p in phases if p.get('name') == names['parent'] and p.get('type') == 'nuclear']
    magnetic = [p for p in phases if p.get('type') == 'magnetic']
    if len(parents) != 1 or len(magnetic) != 1 or magnetic[0].get('name') != names['magnetic']:
        raise ValueError('the view does not identify the exact parent and magnetic phases')
    if view.get('histogram') != row.get('histogram') or not row.get('histogram'):
        raise ValueError('the view belongs to another or unspecified histogram')
    parent_metric = _cell_metric(parents[0].get('cell'))
    child_metric = _cell_metric(magnetic[0].get('cell'))
    transform = _numbers((row.get('embedding') or {}).get('basis'), (3, 3))
    determinant = float(np.linalg.det(transform))
    if not np.isfinite(determinant) or abs(determinant) < 1e-10:
        raise ValueError('the embedding is singular')
    expected = transform @ parent_metric @ transform.T
    if not np.isfinite(expected).all() or not np.allclose(expected, child_metric, rtol=1e-7, atol=1e-7):
        raise ValueError('the complete magnetic cell metric disagrees with the embedding')
    # Columns of A are parent direct axes: X || a, Z || a cross b.
    # Child basis rows obey a' = T a. Native AM components are coefficients
    # along unit child axes, not Cartesian values. Preserve axial parity.
    parent_axes = np.linalg.cholesky(parent_metric).T
    child_lengths = np.sqrt(np.diag(expected))
    convert = np.sign(determinant) * parent_axes @ transform.T @ np.diag(1 / child_lengths)
    unit_axes = parent_axes / np.linalg.norm(parent_axes, axis=0)
    return magnetic[0], convert, unit_axes


def parent_frame_moments(row, view, axis='c'):
    """Return display rows and an availability note; inputs remain untouched.

    ``moment_phase_names`` and ``refined_moments`` come from the verified
    native receipt; the view and embedding belong to that same sealed trial.
    Angle means the acute angle to the selected parent direct axis, so a global magnetic
    time reversal changes the signed projection but not this angle.
    """
    if not row or row.get('kind') != 'magnetic':
        return [], ''
    try:
        if axis not in ('a','b','c'):
            raise ValueError('select parent axis a, b or c')
        magnetic, convert, unit_axes = _parent_frame(row, view)
        reference_axis = unit_axes[:,('a','b','c').index(axis)]
        moments = magnetic.get('moments') or []
        estimates = row.get('refined_moments') or []
        labels = [value.get('label') for value in moments]
        native_labels = [value.get('label') for value in estimates]
        if (not labels or any(not isinstance(label, str) or not label for label in labels)
                or len(set(labels)) != len(labels) or len(set(native_labels)) != len(native_labels)
                or set(labels) != set(native_labels)):
            raise ValueError('unique matching representative labels are unavailable')
        native = dict(zip(native_labels, estimates))
    except (ValueError, TypeError, KeyError, AttributeError, np.linalg.LinAlgError) as exc:
        return [], 'Parent-frame directions unavailable: ' + str(exc) + '.'

    rows, unavailable = [], []
    for value in moments:
        label = value['label']
        try:
            components = _numbers(value.get('components'), (3,))
            reference = native[label]
            if not np.allclose(components, _numbers(reference.get('moment'), (3,)), rtol=1e-10, atol=1e-10):
                raise ValueError('view and receipt moment components disagree')
            cartesian = convert @ components
            magnitude = float(np.linalg.norm(cartesian))
            for declared in (value.get('magnitude'), reference.get('magnitude')):
                if not np.isclose(_numbers(declared, ()), magnitude, rtol=1e-8, atol=1e-8):
                    raise ValueError('view, receipt and converted magnitudes disagree')
            if magnitude <= 1e-12:
                raise ValueError('a zero moment has no direction')
            parallel = float(np.dot(cartesian, reference_axis))
            transverse = float(np.linalg.norm(cartesian - parallel * reference_axis))
            angle = float(np.rad2deg(np.arctan2(transverse, abs(parallel))))
            rows.append(dict(label=label, magnitude=round(magnitude, 6),
                **{'parallel_'+axis:round(parallel, 6),'transverse_'+axis:round(transverse, 6),
                   'angle_'+axis:round(angle, 6)}))
        except (ValueError, TypeError, KeyError) as exc:
            unavailable.append(label + ': ' + str(exc))
    note = ('Unavailable for ' + '; '.join(unavailable) + '.') if unavailable else ''
    return rows, note
