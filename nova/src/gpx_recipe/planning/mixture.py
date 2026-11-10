"""Mass-conserving crystalline mixtures on one explicit intensity basis.

Each native phase pattern is calculated per crystallographic unit cell. Equal
HAP scales do not represent equal masses when the cell masses differ. Volumes
are additive occupied solid volumes; macroscopic packing is applied separately.
Magnetic scattering belongs to the primary phase and is not an extra mass.
"""
import json
import math


def additional_phases(value):
    """Validate up to three declared phases; never renormalize their fractions."""
    if not isinstance(value, list) or len(value) > 3:
        raise ValueError('Declare at most three additional crystalline phases')
    result = []
    allowed = {'cif', 'mass_fraction', 'crystallite_size_um', 'microstrain'}
    for phase in value:
        if not isinstance(phase, dict) or set(phase) - allowed or not {'cif', 'mass_fraction'} <= set(phase):
            raise ValueError('Each additional phase needs its CIF and total-sample mass fraction')
        if not isinstance(phase['cif'], str) or not phase['cif']:
            raise ValueError('An additional phase CIF path is required')
        fraction = phase['mass_fraction']
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Phase mass fractions must be finite numbers between zero and one')
        row = dict(phase, mass_fraction=float(fraction))
        for name in ('crystallite_size_um', 'microstrain'):
            if name not in row:
                continue
            number = row[name]
            if (type(number) not in (int, float) or not math.isfinite(number) or
                    number < 0 or (name == 'crystallite_size_um' and number == 0)):
                raise ValueError('Additional phase size must be positive and microstrain nonnegative')
            row[name] = float(number)
        result.append(row)
    if math.fsum(row['mass_fraction'] for row in result) > 1:
        raise ValueError('Additional phase mass fractions cannot sum to more than one')
    return result


def mass_fractions(value):
    rows = additional_phases(value)
    return [1. - math.fsum(row['mass_fraction'] for row in rows)] + [row['mass_fraction'] for row in rows]


def combine_metadata(components, fractions, reference_cell_mass_da):
    """Return exact mass weights, composition and additive-volume density basis.

The reference cell mass is a normalization choice, not an additional specimen
component. Its equivalent cell count times the combined native curve gives the
same result as summing the actual illuminated cell counts of all phases.
"""
    if not 1 <= len(components) <= 4 or len(components) != len(fractions):
        raise ValueError('Declare one to four phase components and matching mass fractions')
    if (type(reference_cell_mass_da) not in (int, float) or
            not math.isfinite(reference_cell_mass_da) or reference_cell_mass_da <= 0):
        raise ValueError('A finite positive reference cell mass is required')
    if any(type(w) not in (int, float) or not math.isfinite(w) or w < 0 for w in fractions):
        raise ValueError('Invalid mixture mass fractions')
    if not math.isclose(math.fsum(fractions), 1., rel_tol=0, abs_tol=1e-12):
        raise ValueError('Mixture mass fractions must sum to one')
    composition, neutron_table, isotope_selection, isotope_options = {}, {}, {}, {}
    scales, volume = [], 0.
    def identity(value):
        return json.dumps(value, sort_keys=True, default=lambda v: v.tolist())
    for component, fraction in zip(components, fractions):
        mass, cell_volume = component['cell_mass_da'], component['cell_volume_a3']
        if not all(math.isfinite(v) and v > 0 for v in (mass, cell_volume)):
            raise ValueError('Each mixture component needs positive cell mass and volume')
        weight = fraction * reference_cell_mass_da / mass
        scales.append(weight)
        volume += weight * cell_volume
        for element, number in component['composition'].items():
            if not math.isfinite(number) or number < 0:
                raise ValueError('Invalid component cell composition')
            if fraction:
                composition[element] = composition.get(element, 0.) + weight * number
        for key, target in (('neutron_table', neutron_table), ('isotope_selection', isotope_selection),
                            ('isotope_options', isotope_options)):
            for element, entry in component.get(key, {}).items():
                if element in target and identity(target[element]) != identity(entry):
                    raise ValueError('Inconsistent neutron or isotope data for mixture atom type ' + element)
                target[element] = entry
    return dict(cell_mass_da=float(reference_cell_mass_da), cell_volume_a3=volume,
        composition=composition, neutron_table=neutron_table, isotope_selection=isotope_selection,
        isotope_options=isotope_options, nuclear_phase_scales=scales,
        phase_mass_fractions=list(fractions), volume_rule='additive occupied solid volumes')
