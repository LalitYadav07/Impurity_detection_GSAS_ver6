"""Unpolarized, randomly distributed isotope abundances and neutron amplitudes.

Lengths use the native GSAS 10^-12 cm unit (sqrt(barn)). The common convention
is b=b'-i*b'', with b''>=0 for absorption. Mixtures average complex amplitudes,
not intensities; their masses use atom fractions, not phase mass fractions.
Native isotope records are installed only inside a single-process calculation
context and are restored afterwards. No installed GSAS source/data is edited.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import numpy as np

MODEL = 'powgen-neutron-isotope-amplitudes/1'
COMPONENTS = 'RADAR-isotope-components'
PREFIX = 'RADARmix-'
_LOCK = threading.Lock()
# CIAAW Isotopic Compositions 2024, https://ciaaw.org/isotopic-abundances.htm:
# only entries explicitly having representative abundance 1 are aliases for
# the natural-element neutron table. Never infer this from a dominant isotope.
MONOISOTOPIC_NATURAL = {'Be':'9','F':'19','Na':'23','Al':'27','P':'31',
    'Sc':'45','Mn':'55','Co':'59','As':'75','Y':'89','Nb':'93','Rh':'103',
    'I':'127','Cs':'133','Pr':'141','Tb':'159','Ho':'165','Tm':'169',
    'Au':'197','Bi':'209','Pa':'231'}


def selections(value):
    """Validate exact native labels or explicit atom-fraction dictionaries."""
    if not isinstance(value, dict) or len(value) > 128:
        raise ValueError('Isotopes must map exact atom types to isotope labels or atom fractions')
    result = {}
    for element, selection in value.items():
        if not isinstance(element, str) or not element or len(element) > 32:
            raise ValueError('An exact atom type is required for each isotope selection')
        if isinstance(selection, str):
            if not selection or len(selection) > 64 or selection.startswith(PREFIX):
                raise ValueError('Select an available native isotope label')
            result[element] = selection
            continue
        if not isinstance(selection, dict) or not 1 <= len(selection) <= 16:
            raise ValueError('An isotope mixture needs one to sixteen explicit atom fractions')
        fractions = {}
        for label, number in selection.items():
            if (not isinstance(label, str) or not label or len(label) > 64 or label.startswith(PREFIX)
                    or type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 1):
                raise ValueError('Isotope atom fractions must be finite numbers between zero and one')
            fractions[label] = float(number)
        if not math.isclose(math.fsum(fractions.values()), 1., rel_tol=0, abs_tol=1e-12):
            raise ValueError('Isotope atom fractions must sum to one; they are not renormalized')
        result[element] = fractions
    return result


def resonant(record):
    if COMPONENTS in record:
        return any(c['atom_fraction'] > 0 and resonant(c['neutron']) for c in record[COMPONENTS])
    return 'BW-LS' in record


def amplitude(record, wavelength):
    """Return complex coherent b(lambda), with one absorptive sign convention."""
    wave = np.asarray(wavelength, float)
    if not np.isfinite(wave).all() or np.any(wave <= 0):
        raise ValueError('Finite positive wavelengths are required')
    if COMPONENTS in record:
        return sum(c['atom_fraction'] * amplitude(c['neutron'], wave)
                   for c in record[COMPONENTS] if c['atom_fraction'] > 0)
    sl = np.asarray(record['SL'], float)
    if sl.shape != (2,) or not np.isfinite(sl).all():
        raise ValueError('Two finite native scattering-length components are required')
    if 'BW-LS' not in record:
        return np.full(wave.shape, complex(sl[0], -abs(sl[1])), dtype=complex)
    bw = np.asarray(record['BW-LS'], float)
    if bw.shape != (9,) or not np.isfinite(bw).all() or bw[4] <= 0 or sl[0] != 0:
        raise ValueError('Unsupported native Breit-Wigner scattering-length convention')
    b0, real_coefficient, imag_coefficient, e0, width, a, e1, b, e2 = bw
    energy = 81.80703 / wave**2
    real = np.full(wave.shape, b0)
    absorptive = np.zeros(wave.shape)
    for strength, centre in ((1., e0), (a, e1), (b, e2)):
        delta = energy - centre
        denominator = delta**2 + width**2 / 4
        real += real_coefficient * strength * delta / denominator
        absorptive += imag_coefficient * strength / denominator
    return real - 1j * np.abs(absorptive)


def isotope_key(element, label):
    """Map native labels, including whitespace and D/T aliases, to NIST keys."""
    match = re.match(r'^[A-Z][a-z]?', element)
    if not match:
        raise ValueError('An exact chemical atom type is required')
    symbol = match.group()
    label = label.strip()
    if symbol in ('D', 'T'):
        if label == 'Nat. Abund.':
            label = '2' if symbol == 'D' else '3'
        symbol = 'H'
    if MONOISOTOPIC_NATURAL.get(symbol) == label:
        return symbol
    return symbol if label == 'Nat. Abund.' else label + symbol


def bound_cross_sections(element, label, record, wavelength, table):
    """Return absorption, incoherent and total bound scattering for one atom.

Spin/within-stock incoherence uses the tabulated thermal value. The isotope
disorder term and coherent cross section use b(lambda), including resonances.
This is not a thermal scattering-law or spin-resolved resonance calculation.
"""
    wave = np.asarray(wavelength, float)
    mean = amplitude(record, wave)
    components = record.get(COMPONENTS, [{'label': label, 'atom_fraction': 1., 'neutron': record}])
    absorption = np.zeros_like(wave)
    spin = 0.
    variance = np.zeros_like(wave)
    for c in components:
        fraction = c['atom_fraction']
        if fraction == 0:
            continue
        key = isotope_key(element, c['label'])
        if key not in table['elements']:
            raise ValueError('No thermal neutron cross-section data for isotope ' + key)
        xs = table['elements'][key]
        local = amplitude(c['neutron'], wave)
        spin += fraction * xs['incoherent_b']
        variance += fraction * np.abs(local - mean)**2
        if resonant(c['neutron']):
            absorption += fraction * 2e4 * wave * np.abs(local.imag)
        else:
            absorption += fraction * xs['absorption_2200_b'] * wave / table['reference_wavelength_a']
    incoherent = spin + 4 * np.pi * variance
    coherent = 4 * np.pi * np.abs(mean)**2
    return absorption, incoherent, coherent + incoherent


@contextmanager
def neutron_backend(declared):
    """Provide abundance records and canonical amplitudes in an isolated worker."""
    declared = selections(declared)
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Use a dedicated native calculation process')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Neutron isotope contexts cannot overlap or nest')
    temporary = {}
    saved = {}
    try:
        from GSASII import GSASIIElem as E
        native_labels = {}
        for element, choice in declared.items():
            info = E.GetAtomInfo(element)
            available = info['Isotopes']
            if isinstance(choice, str):
                if choice not in available:
                    raise ValueError('Unsupported isotope for atom type ' + element + ': ' + choice)
                native_labels[element] = choice
                continue
            if set(choice) - set(available):
                raise ValueError('Unsupported isotope in atom-fraction mixture for ' + element)
            components = [{'label': label, 'atom_fraction': fraction, 'neutron': copy.deepcopy(available[label])}
                          for label, fraction in sorted(choice.items())]
            record = {'Mass': math.fsum(c['atom_fraction'] * c['neutron']['Mass'] for c in components),
                      'SL': [math.fsum(c['atom_fraction'] * c['neutron']['SL'][i] for c in components) for i in (0, 1)],
                      COMPONENTS: components}
            if all('SA' in c['neutron'] for c in components):
                record['SA'] = math.fsum(c['atom_fraction'] * c['neutron']['SA'] for c in components)
            label = PREFIX + hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:24]
            key = E.getElSym(element) + '_' + label
            if key in E.atmdata.AtmBlens and key not in temporary:
                raise RuntimeError('Temporary isotope key collision')
            temporary[key] = record
            E.atmdata.AtmBlens[key] = record
            native_labels[element] = label

        saved = {name: getattr(E, name) for name in ('BlenResCW', 'BlenResTOF')}
        def cw(elements, tables, wave):
            values = np.asarray([amplitude(tables[e][1], wave) for e in elements])
            baseline = np.asarray([tables[e][1]['SL'][0] for e in elements])
            return values.real - baseline, values.imag
        def tof(elements, tables, wave):
            values = np.asarray([amplitude(tables[e][1], wave) for e in elements])
            baseline = np.asarray([tables[e][1]['SL'][0] for e in elements])[:, None]
            return values.real - baseline, values.imag
        E.BlenResCW, E.BlenResTOF = cw, tof
        yield {'version': MODEL, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'native_element_source_sha256': hashlib.sha256(Path(E.__file__).read_bytes()).hexdigest(),
               'native_atomic_data_sha256': hashlib.sha256(Path(E.atmdata.__file__).read_bytes()).hexdigest(),
               'monoisotopic_natural_alias_source': 'CIAAW Isotopic Compositions 2024; https://ciaaw.org/isotopic-abundances.htm',
               'absorptive_convention': "b=b'-i*b_doubleprime; b_doubleprime>=0",
               'abundance_basis': 'atom fractions of independently randomly distributed, unpolarized isotope stocks',
               'native_selections': native_labels,
               'standalone_gpx_requires_planning_adapter': any(isinstance(v, dict) for v in declared.values()),
               'thermal_spin_incoherence_energy_dependence_modelled': False,
               'abundance_uncertainty_supplied_or_propagated': False,
               'experimental_accuracy_certified': False}
    finally:
        if saved:
            for name, function in saved.items():
                setattr(E, name, function)
        if temporary:
            for key in temporary:
                E.atmdata.AtmBlens.pop(key, None)
        _LOCK.release()
