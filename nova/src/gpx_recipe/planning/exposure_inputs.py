"""Explicit cumulative proton charge, or time under a delivered-current assumption."""
import math


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return float(value)


def validate_exposure(value):
    if 'charge_uah' in value and 'seconds' in value:
        raise ValueError('Supply cumulative charge or cumulative time, not both')
    field = 'charge_uah' if 'charge_uah' in value else 'seconds'
    values = value.get(field, [60., 300., 1800.])
    if not isinstance(values, list) or not 1 <= len(values) <= 24:
        raise ValueError('Declare 1–24 cumulative exposures')
    values = [_positive(v, field) for v in values]
    if any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError('Cumulative exposures must increase')
    beam = value.get('beam', {})
    allowed = {'power_mw', 'proton_energy_gev', 'average_current_ua', 'neutron_yield_ratio'}
    if not isinstance(beam, dict) or set(beam) - allowed:
        raise ValueError('Unknown beam settings')
    beam = {k: _positive(v, k) for k, v in beam.items() if v is not None}
    if 'power_mw' in beam and 'average_current_ua' in beam:
        raise ValueError('Choose average delivered current or beam power')
    if 'power_mw' in beam and 'proton_energy_gev' not in beam:
        raise ValueError('Power-to-charge conversion requires proton energy')
    if field == 'seconds' and value.get('response_mode', 'profile_only') != 'profile_only':
        if not {'power_mw', 'average_current_ua'} & beam.keys():
            raise ValueError('Counting time requires average delivered current or beam power and energy')
    return {field: values, 'beam': beam}


def cumulative_exposures(request):
    """Return charges and provenance; no future beam availability is predicted."""
    if 'charge_uah' in request:
        return [dict(cumulative_charge_uah=q) for q in request['charge_uah']], dict(
            basis='supplied cumulative proton charge', exposure_unit='microampere_hour',
            charge_uncertainty_included=False, future_delivered_charge_predicted=False)
    beam = request['beam']
    current = beam.get('average_current_ua')
    if current is None:
        current = 1000 * beam['power_mw'] / beam['proton_energy_gev']
    looks = [dict(cumulative_seconds=s, cumulative_charge_uah=s * current / 3600)
             for s in request['seconds']]
    if not all(math.isfinite(row['cumulative_charge_uah']) and row['cumulative_charge_uah'] > 0 for row in looks):
        raise ValueError('Time-to-charge conversion exceeds its finite numerical domain')
    return looks, dict(basis='time conditional on assumed average delivered current',
        average_current_ua=current, exposure_unit='microampere_hour',
        charge_uncertainty_included=False, future_delivered_charge_predicted=False,
        note='Average current includes beam interruptions. Requested time is exposure time; overheads are additional.')


def exposure_label(look):
    charge = look.get('cumulative_charge_uah', look.get('cumulative_exposure'))
    if look.get('cumulative_seconds') is not None:
        return f"{look['cumulative_seconds']:g} s" + (f' · {charge:g} μAh' if charge is not None else '')
    return f'{charge:g} μAh'
