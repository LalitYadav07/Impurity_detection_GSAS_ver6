"""NOMAD scalar NeXus metadata; event banks and inactive sensors are never read."""
from pathlib import Path
from typing import Any
import re

from .powgen_metadata import (_first_snapshot, _text_value, _identifier_value,
    _numeric_log, _temperature_in_kelvin, _duration_seconds, read_powgen_sample_identity)
from .powgen_samples import PowgenSampleIndex
from .facility_metadata import resolve_nexus_path, qualify_temperature


def nomad_nexus_path(ipts: str, run: int) -> Path:
    proposal = str(ipts).upper()
    if not re.fullmatch(r"IPTS-\d+", proposal) or isinstance(run, bool) or int(run) < 1:
        raise ValueError("NOMAD requires an IPTS identifier and a positive run number")
    return resolve_nexus_path('NOM', proposal, run)


def read_nomad_sample_identity(ipts: str, run: int, *, nexus_path=None) -> dict[str, Any]:
    path = Path(nexus_path) if nexus_path else nomad_nexus_path(ipts, run)
    return read_powgen_sample_identity(ipts, run, nexus_path=path, instrument='NOM')


class NomadSampleIndex(PowgenSampleIndex):
    def __init__(self, ipts: str):
        super().__init__(ipts, nexus_path=nomad_nexus_path, identity_reader=read_nomad_sample_identity, instrument='NOM')


def read_nomad_scan_metadata(ipts: str, run: int, *, nexus_path=None) -> dict[str, Any]:
    path = Path(nexus_path) if nexus_path else nomad_nexus_path(ipts, run)
    result: dict[str, Any] = {'schema_version': 1, 'instrument': 'NOM',
                             'nexus_path': str(path), 'available': path.is_file()}
    if not result['available']:
        return result
    stored_run = _identifier_value(path, ('/entry/run_number',))
    stored_proposal = _text_value(path, ('/entry/experiment_identifier',)).upper()
    if stored_run and stored_run != str(run):
        raise ValueError('NOMAD NeXus run identifier disagrees with the selected run')
    if stored_proposal and stored_proposal != str(ipts).upper():
        raise ValueError('NOMAD NeXus proposal disagrees with the selected IPTS')
    result.update(read_nomad_sample_identity(ipts, run, nexus_path=path))
    for key, field in [('run_title','/entry/title'), ('sample_formula','/entry/sample/chemical_formula'),
                       ('start_time','/entry/start_time'), ('end_time','/entry/end_time')]:
        result[key] = _text_value(path, (field,))
    result['duration_seconds'] = _duration_seconds(result['start_time'], result['end_time'])
    result['temperature_sensor'] = _text_value(path, ('/entry/DASlogs/BL1B:SE:SelectSampleTemp/value_strings',))
    # Only the selected SampleTemp summary is meaningful; never fall back to an
    # inactive levitator/furnace channel or infer units from a title/magnitude.
    prefix = '/entry/DASlogs/BL1B:SE:SampleTemp/'
    temp = _numeric_log(_first_snapshot(path, (prefix+'average_value',)))
    if temp:
        temp['statistic'] = 'mean'
        for key, field in [('minimum','minimum_value'), ('maximum','maximum_value')]:
            value = _numeric_log(_first_snapshot(path, (prefix+field,)))
            if value and value['unit'] == temp['unit']:
                temp[key] = value['value']
        result['temperature_observation'] = _temperature_in_kelvin(temp)
        qualified, notices = qualify_temperature(result['temperature_observation'], result.get('run_title',''))
        result['metadata_notices'] = notices
        if qualified:
            result['temperature'] = qualified
    else:
        result['metadata_notices'] = ['Selected sample temperature is unavailable']
    charge = _numeric_log(_first_snapshot(path, ('/entry/proton_charge',)))
    if charge:
        result['proton_charge'] = charge
    # A chopper request is diagnostic metadata, not a monochromatic wavelength.
    request = _numeric_log(_first_snapshot(path, ('/entry/DASlogs/BL1B:Chop:Skf1:WavelengthUserReq/value',)))
    if request:
        result['chopper_wavelength_request'] = request
    return result
