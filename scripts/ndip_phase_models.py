"""Flag interpretation limits of concurrently fitted nuclear phase models."""
from __future__ import annotations

from collections import defaultdict
import math
import re


def _composition(phase):
    general = phase.get('General') or {}
    if general.get('Type', 'nuclear') != 'nuclear' or phase.get('Pawley ref'):
        return None
    pointers = general.get('AtomPtrs')
    if not isinstance(pointers, (list, tuple)) or len(pointers) != 4:
        return None
    cx, ct, cs, _ = pointers
    if any(type(p) is not int or p < 0 for p in (cx, ct, cs)):
        return None
    amounts = defaultdict(list)
    isotopes = general.get('Isotopes') or {}
    if not isinstance(isotopes, dict):
        return None
    for atom in phase.get('Atoms') or []:
        try:
            kind = str(atom[ct]).strip()
            match = re.fullmatch(r'([A-Z][a-z]?)(?:[+-]\d*|\d+[+-])?', kind)
            if not match:
                return None
            element = match[1]
            occupancy, multiplicity = float(atom[cx+3]), float(atom[cs+1])
            if (not math.isfinite(occupancy) or not 0 <= occupancy <= 1
                    or not math.isfinite(multiplicity) or multiplicity <= 0):
                return None
            isotope = str(isotopes.get(kind, isotopes.get(element, '')))
            amounts[(element, isotope)].append(occupancy*multiplicity)
        except (IndexError, TypeError, ValueError):
            return None
    counts = {key: math.fsum(values) for key, values in amounts.items() if math.fsum(values) > 0}
    total = math.fsum(counts.values())
    return {key: value/total for key, value in counts.items()} if total else None


def same_composition_warnings(records, *, labels=None):
    """A fraction fit cannot distinguish coexistence from competing model errors.

    Compare site-occupancy × multiplicity, not names, cell size, or formula
    strings. Do not merge, remove, or relabel phases, and do not include a
    magnetic counterpart or phases that occur only in different histograms.
    """
    by_histogram = defaultdict(list)
    for record in records:
        if not record or record[0][0] != 'Phases':
            continue
        for name, phase in record[1:]:
            composition = _composition(phase)
            if composition is None:
                continue
            for histogram, hap in (phase.get('Histograms') or {}).items():
                try:
                    scale = float(hap.get('Scale', [0])[0])
                except (ValueError, TypeError, IndexError):
                    continue
                if hap.get('Use', True) and math.isfinite(scale) and scale > 0:
                    by_histogram[histogram].append((str(name), composition))
    groups = []
    for histogram, phases in sorted(by_histogram.items()):
        pending = list(phases)
        while pending:
            name, composition = pending.pop(0)
            matches = [name]
            remaining = []
            for candidate, other in pending:
                if (composition.keys() == other.keys()
                        and all(math.isclose(value, other[key], rel_tol=0, abs_tol=1e-10)
                                for key, value in composition.items())):
                    matches.append(candidate)
                else:
                    remaining.append((candidate, other))
            pending = remaining
            if len(matches) > 1:
                groups.append({'histogram': str(histogram), 'phase_ids': matches,
                               'phase_labels': [(labels or {}).get(name, name) for name in matches]})
    if not groups:
        return []
    names = '; '.join(', '.join(group['phase_labels']) for group in groups)
    return [{'code': 'same_composition_phase_models',
             'message': 'Several fitted nuclear models have the same normalized chemical composition: '
                        + names + '. Their fitted fractions do not establish phase coexistence. '
                        'Competing structure models can also compensate for profile or model errors; '
                        'compare single-structure fits or Model exploration before interpreting these fractions.',
             'groups': groups, 'basis': 'Native site occupancies and multiplicities in the same histogram; isotope settings retained'}]
