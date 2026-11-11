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


def _element_set(values):
    if not isinstance(values, (list, tuple)) or any(
            not isinstance(value, str) or not re.fullmatch(r'[A-Z][a-z]?', value) for value in values):
        return None
    return set(values)


def resolved_component_chemistry(config):
    """Retain only an unambiguous declared chemistry from the executed config."""
    choices = []
    for dataset in config.get('datasets') or [{}]:
        sample = _element_set(dataset.get('allowed_elements', config.get('allowed_elements')))
        filters = {**(config.get('element_filter') or {}), **(dataset.get('element_filter') or {})}
        environment = _element_set((filters.get('sample_env') or {}).get('elements', []))
        if not sample or environment is None:
            return None
        choices.append((tuple(sorted(sample)), tuple(sorted(environment))))
    if len(set(choices)) != 1:
        return None
    sample, environment = choices[0]
    return dict(sample_elements=list(sample), environment_elements=list(environment))


def component_chemistry_context(records, chemistry, *, labels=None):
    """Describe declared component roles, without removing or renormalizing fits.

    Element assignment is an analysis input, not proof of physical origin.
    Overlapping declarations and mixed chemistry must remain distinguishable
    from a phase made exclusively of declared environment elements.
    """
    chemistry = chemistry if isinstance(chemistry, dict) else {}
    sample = _element_set(chemistry.get('sample_elements'))
    environment = _element_set(chemistry.get('environment_elements', []))
    if not sample or environment is None:
        return {}, []
    phases, masked_histograms = [], set()
    for record in records:
        if not record:
            continue
        if str(record[0][0]).startswith('PWDR'):
            for key, value in record[1:]:
                if key == 'Limits' and isinstance(value, (list, tuple)):
                    for span in value[2:]:
                        try:
                            valid = (isinstance(span, (list, tuple)) and len(span)==2
                                     and all(math.isfinite(float(x)) for x in span)
                                     and float(span[0]) <= float(span[1]))
                        except (TypeError, ValueError):
                            valid = False
                        if valid:
                            masked_histograms.add(str(record[0][0]))
        if record[0][0] != 'Phases':
            continue
        for name, phase in record[1:]:
            composition = _composition(phase)
            if composition is None:
                continue
            elements = {element for element, isotope in composition}
            is_sample, is_environment = elements <= sample, bool(environment) and elements <= environment
            role = ('overlapping_chemistry' if is_sample and is_environment else
                    'environment_only_chemistry' if is_environment else
                    'sample_chemistry' if is_sample else 'mixed_or_unassigned_chemistry')
            active = []
            for histogram, hap in (phase.get('Histograms') or {}).items():
                try:
                    scale = float(hap.get('Scale', [0])[0])
                except (TypeError, ValueError, IndexError):
                    continue
                if hap.get('Use', True) and math.isfinite(scale) and scale > 0:
                    active.append(str(histogram))
            phases.append(dict(phase_id=str(name), phase_label=(labels or {}).get(str(name), str(name)),
                elements=sorted(elements), component_role=role, active_histograms=active))
    context = dict(schema='radar-component-chemistry/v1', sample_elements=sorted(sample),
        environment_elements=sorted(environment), phase_roles=phases,
        assignment_basis='Executed analysis chemistry and native occupied atomic sites; declared role, not measured physical origin')
    environment_phases = [p for p in phases if p['component_role']=='environment_only_chemistry' and p['active_histograms']]
    if not environment_phases:
        return context, []
    names = ', '.join(p['phase_label'] for p in environment_phases)
    masked = any(masked_histograms.intersection(p['active_histograms']) for p in environment_phases)
    message = ('Normalized crystalline fractions include components made only of elements assigned to the sample environment: '
               + names + '. These shares do not measure bulk sample composition or container mass.')
    if masked:
        message += ' The fit excludes pattern regions; check the remaining component reflections before interpreting changes in these shares.'
    return context, [dict(code='environment_component_fractions', message=message,
        phase_ids=[p['phase_id'] for p in environment_phases], excluded_regions_present=masked,
        basis=context['assignment_basis'])]
