"""Read-only HB-2A individual CW reductions and date-qualified SPICE metadata.

SPICE experiment/scan numbers are a pair, not SNS global run numbers. Detector
efficiency corrections are reduction inputs, never Rietveld peak profiles.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

ARCHIVE_ROOT = Path('/HFIR/HB2A')
VERSION = 'hb2a-cw/1'
FOLDER = 'shared/autoreduce'
MAX_BYTES = 4 * 1024 * 1024
NORMALIZATIONS = ('time', 'monitor', 'legacy')
_PRODUCT = re.compile(r'HB2A_exp(\d+)_scan(\d+)(_autoreduced)?(_norm_time)?\.(?:dat|xye)', re.I)
_RAW = re.compile(r'HB2A_exp(\d+)_scan(\d+)\.dat', re.I)


def proposal_root(ipts):
    value = str(ipts).upper()
    if not re.fullmatch(r'IPTS-[1-9]\d*', value):
        raise ValueError('HB-2A requires IPTS-<positive number>')
    return ARCHIVE_ROOT / value


def signature(path):
    st = Path(path).stat()
    return st.st_size, st.st_mtime_ns


def bounded_bytes(path):
    path = Path(path)
    before = signature(path)
    with path.open('rb') as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError(f'{path.name}: exceeds the bounded HB-2A text-product limit')
    if signature(path) != before:
        raise ValueError(f'{path.name}: still changing; wait for reduction completion')
    return data, before


@dataclass(frozen=True)
class Product:
    path: str
    experiment: int
    scan: int
    normalization: str

    @property
    def run_id(self):
        return f'HB2A_EXP{self.experiment}_SCAN{self.scan:04d}'


def product(path):
    match = _PRODUCT.fullmatch(Path(path).name)
    if not match or int(match[1]) < 1 or int(match[2]) < 1:
        return None
    normalization = 'time' if match[4] else 'monitor' if match[3] else 'legacy'
    return Product(str(path), int(match[1]), int(match[2]), normalization)


def directory_listing(directory, *, normalization='time'):
    if normalization not in NORMALIZATIONS:
        raise ValueError('Choose time, monitor, or explicitly reviewed legacy normalization')
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f'HB-2A reduced folder is unavailable: {root}')
    result = []
    with os.scandir(root) as entries:
        for entry in entries:
            parsed = product(entry.name)
            if not parsed or parsed.normalization != normalization:
                continue  # Sums, raw scans, GSS duplicates and other variants stay separate.
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                st = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            result.append({'path': str(root / entry.name), 'size': st.st_size,
                           'modified_ns': st.st_mtime_ns})
    return result


def discover_listing(definition, rows, state=None, *, newest_limit=None):
    from .powgen_watch import WatchedRun, _fingerprint, _listing_path, _path_in_watch
    found = {}
    wanted = definition.config_refs.get('hb2a_normalization', 'time')
    for entry in rows:
        raw, metadata = _listing_path(entry)
        path = _path_in_watch(definition, raw)
        parsed = product(path) if path else None
        if not parsed or parsed.normalization != wanted:
            continue
        key = (parsed.experiment, parsed.scan)
        if key in found and found[key].source_path != path:
            raise ValueError(f'Ambiguous duplicate reduction for {parsed.run_id}; choose one export')
        found[key] = WatchedRun(run_number=parsed.scan, experiment_number=parsed.experiment,
            source_path=path, fingerprint=_fingerprint(path, metadata), instrument='HB2A')
    result = [found[key] for key in sorted(found)]
    if newest_limit is not None:
        result = result[-newest_limit:]
    if state is not None:
        result = [run for run in result if not state.contains(run)]
        state.discovered.update({run.run_id: run for run in result})
    return result


def raw_path(ipts, exp, scan):
    root = proposal_root(ipts)
    path = root / f'exp{exp}' / 'Datafiles' / f'HB2A_exp{exp:04d}_scan{scan:04d}.dat'
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('HB-2A raw metadata path escapes this IPTS')
    return path


def _date(value, time):
    stamp = datetime.strptime(' '.join((time + ' ' + value).split()), '%I:%M:%S %p %m/%d/%Y')
    # HFIR/SPICE uses local wall time. Keep the named zone and UTC offset.
    return stamp.replace(tzinfo=ZoneInfo('America/New_York'))


def read_spice(path, *, ipts, experiment, scan, condition_bindings=None,
        identity_only=False, collimator_override='', allow_stopped=False):
    data, sig = bounded_bytes(path)
    text = data.decode('utf-8', errors='strict')
    headers, columns, rows, completed, stopped = {}, None, [], None, False
    for line in text.splitlines():
        if line.startswith('#'):
            match = re.match(r'#\s*([\w]+)\s*=\s*(.*)', line)
            if match:
                key, value = match.groups()
                if key in headers and headers[key] != value:
                    raise ValueError(f'Conflicting SPICE header {key}')
                headers[key] = value.strip()
            elif re.match(r'#\s*Pt\.', line):
                columns = line.lstrip('# ').split()
            elif 'scan completed.' in line or 'scan stopped!!' in line:
                stopped = 'scan stopped!!' in line
                match = re.match(r'#\s*(.*?)\s+scan (?:completed\.|stopped!!)', line)
                if match:
                    stamp = match[1].split()
                    if len(stamp) == 3:
                        completed = _date(stamp[2], ' '.join(stamp[:2]))
            continue
        if not line.strip():
            continue
        if columns is None:
            raise ValueError('SPICE data lacks its declared column header')
        values = line.split()
        if len(values) != len(columns):
            raise ValueError('Incomplete or shifted SPICE row; do not repair column alignment')
        row = [float(v) for v in values]
        if not all(math.isfinite(v) for v in row):
            raise ValueError('Non-finite SPICE acquisition metadata')
        rows.append(row)
    if not rows or columns is None:
        raise ValueError('SPICE acquisition contains no completed points')
    expected = {'scan': str(scan), 'proposal': str(ipts).removeprefix('IPTS-'),
                'experiment_number': str(experiment)}
    if any(headers.get(k) != v for k, v in expected.items()):
        raise ValueError('SPICE proposal/experiment/scan disagrees with selected product')
    setup_errors = []
    if completed is None:
        setup_errors.append('SPICE scan has no completion/stop footer; wait for acquisition to finish')
    if stopped and not allow_stopped:
        setup_errors.append('This acquisition was stopped early; explicitly allow stopped scans after reviewing angular coverage')
    start = _date(headers.get('date', ''), headers.get('time', ''))
    if completed is not None and completed < start:
        raise ValueError('SPICE completion time precedes acquisition start')
    if headers.get('def_x', '').strip().lower() != '2theta':
        setup_errors.append('This is a fixed-angle/environment scan, not a 2theta powder pattern')
    channels = {key: [r[i] for r in rows] for i, key in enumerate(columns)}
    times = channels.get('time')
    if times is None or min(times) < 0 or sum(times) <= 0:
        raise ValueError('SPICE acquisition has no positive counting time')
    if len(set(channels.get('Pt.', []))) != len(rows):
        raise ValueError('Repeated SPICE point numbers need acquisition review')
    m1, coll = channels.get('m1'), channels.get('colltrans')
    if not m1:
        setup_errors.append('Measured m1 is absent; the Ge reflection cannot be deduced')
    if m1 and max(m1) - min(m1) > 0.1 or coll and max(coll) - min(coll) > 0.2:
        setup_errors.append('Monochromator/collimator changed within this scan')
    m = sum(m1) / len(m1) if m1 else float('nan')
    choices = [(113, 9.45, 2.41), (115, 0.0, 1.54)]
    if start.date().isoformat() >= '2021-04-25':
        choices.append((117, -4.375, 1.12))
    settings = [(ge, wavelength) for ge, motor, wavelength in choices if abs(m - motor) <= 0.1]
    if len(settings) != 1:
        setup_errors.append('Measured m1 does not match a documented date-qualified Ge reflection')
    c = sum(coll) / len(coll) if coll else float('nan')
    collimator = ''
    if not coll:
        if collimator_override not in {'IN','OUT'}:
            setup_errors.append('Measured colltrans is absent; explicitly select reviewed collimation for this acquisition')
        else:
            collimator = collimator_override
    elif abs(c) <= 0.1:
        collimator = 'OUT' if start.date().isoformat() >= '2021-02-23' else 'IN'
    elif abs(abs(c) - 80) <= 0.1:
        collimator = 'IN' if start.date().isoformat() >= '2021-02-23' else 'OUT'
    else:
        setup_errors.append('Collimator is between documented IN/OUT positions; select a reviewed setup')
    if coll and collimator_override and collimator_override != collimator:
        setup_errors.append('Selected collimation contradicts measured date-qualified colltrans')
    if setup_errors and not identity_only:
        raise ValueError('; '.join(setup_errors))
    name = headers.get('samplename', '').strip()
    if name.casefold() in {'', 'none', 'unknown', 'sample', 'null'}:
        sample_id, basis = '', 'unidentified'
    else:
        reported = headers.get('sample_id') or headers.get('sampleid')
        sample_id = (f'EXP{experiment}:ID:{reported}' if reported else
                     f'EXP{experiment}:NAME:{sha256(name.encode()).hexdigest()[:12]}')
        basis = 'SPICE sample ID scoped to experiment' if reported else 'SPICE sample name scoped to experiment; no sample ID recorded'
    source = str(Path(path))
    result = {'schema_version': 3, 'instrument': 'HB2A', 'facility': 'HFIR', 'available': True,
        'run_number': scan, 'scan_number': scan, 'experiment_number': experiment,
        'source_run_id': f'HB2A_EXP{experiment}_SCAN{scan:04d}',
        'sample_id': sample_id, 'sample_name': name, 'identity_basis': basis,
        'identity_status': 'resolved' if sample_id else 'unidentified',
        'identity_sources': [{'kind':'SPICE acquisition','value':f'exp{experiment}/scan{scan:04d}','path':source}],
        'run_title': headers.get('scan_title', ''),
        'sample_type': headers.get('sampletype', ''), 'start_time': start.isoformat(),
        'end_time': completed.isoformat() if completed else '',
        'duration_seconds': (completed-start).total_seconds() if completed else None,
        'acquisition_status': 'stopped' if stopped else 'completed' if completed else 'incomplete',
        'counting_seconds': sum(times), 'spice_path': source, 'spice_signature': list(sig),
        'channel_schema_sha256':sha256(json.dumps(columns).encode()).hexdigest(),
        'spice_sha256': sha256(data).hexdigest(), 'available_channels': [c for c in columns if not c.startswith('anode')],
        'ge_reflection': settings[0][0] if len(settings)==1 else None,
        'nominal_wavelength_angstrom': settings[0][1] if len(settings)==1 else None,
        'collimator': collimator, 'collimator_convention': 'post-2021-02-23' if start.date().isoformat() >= '2021-02-23' else 'legacy',
        'm1_range': [min(m1), max(m1)] if m1 else None, 'colltrans_range': [min(coll), max(coll)] if coll else None,
        'setup_errors': setup_errors,
        'metadata_notices': ['SPICE sample names are labels, not globally unique acquisition IDs.',
            *(['Collimation supplied by reviewed user mapping because colltrans is absent.'] if not coll and collimator else []),
            *(['Stopped acquisition: review its measured angular coverage.'] if stopped else [])]}
    conditions = {}
    for quantity, binding in (condition_bindings or {}).items():
        if quantity not in {'temperature', 'magnetic_field', 'pressure', 'voltage'}:
            raise ValueError('Choose a supported sample-condition quantity')
        channel, unit = binding.get('channel'), binding.get('unit')
        if channel in {'Pt.','time','monitor','detector','detector_2','2theta','m1','marc','mtrans','colltrans','coll','pgfilter'} or str(channel).startswith('anode'):
            raise ValueError(f'{quantity}: counts, acquisition timing and diffraction motors are not sample-environment readbacks')
        if str(channel).lower().endswith(('_sp','_setpoint')) or channel in {'field','mag_i','working_field','ult_sp','furnace_setpoint'}:
            raise ValueError(f'{quantity}: selected channel is a request/setpoint; choose the measured readback')
        accepted = {'temperature': {'K', 'C'}, 'magnetic_field': {'T'}, 'pressure': {'GPa', 'bar'}, 'voltage': {'V'}}
        if channel not in channels or unit not in accepted[quantity] or not binding.get('reviewed'):
            raise ValueError(f'{quantity}: choose a measured SPICE channel and explicitly review its units')
        values = channels[channel]
        offset = 273.15 if quantity == 'temperature' and unit == 'C' else 0
        scale = (1e9 if unit=='GPa' else 1e5) if quantity=='pressure' else 1
        canonical_unit = 'K' if quantity == 'temperature' else 'Pa' if quantity=='pressure' else unit
        observation = {'value': scale*sum(t*v for t,v in zip(times,values))/sum(times)+offset,
            'minimum': scale*min(values)+offset, 'maximum': scale*max(values)+offset,
            'unit': canonical_unit, 'recorded_unit':unit, 'source': source,
            'channel': channel, 'statistic': 'counting_time_weighted_point_samples',
            'quantity':quantity,'status':'recorded','role':'measured','binding_verified':True,
            'reviewed_binding': True,'sampled_points':len(values),
            'approximation':'Counting-time weighting of measured point samples; no between-point interpolation'}
        if quantity == 'temperature' and observation['minimum'] <= 0:
            raise ValueError('Selected temperature channel gives nonphysical Kelvin values')
        result[quantity] = observation
        conditions[quantity] = observation
    packet={'$schema':'radar-pd-conditions/v1','registry':'HFIR-HB2A-SPICE-conditions/1',
        'channels':list(conditions.values()),'source_sha256':result['spice_sha256'],
        'acquisition_start':result['start_time'],'acquisition_end':result['end_time']}
    packet['metadata_sha256']=sha256(json.dumps(packet,sort_keys=True,allow_nan=False).encode()).hexdigest()
    result['conditions'] = packet
    if 'temperature' not in result:
        result['metadata_notices'].append('No qualified sample-temperature channel selected; titles and inactive sensors are not substituted.')
    return result


def read_metadata(ipts, exp, scan, *, condition_bindings=None, **kwargs):
    return read_spice(raw_path(ipts, exp, scan), ipts=ipts, experiment=exp, scan=scan,
                      condition_bindings=condition_bindings, **kwargs)


@dataclass(frozen=True)
class Setup:
    experiment: int
    scan: int
    normalization: str
    source_sha256: str
    source_signature: tuple
    records: int
    two_theta_min: float
    two_theta_max: float
    masked_points: int
    metadata: dict

    def as_dict(self):
        return asdict(self)


def inspect_reduced_setup(path, *, ipts, condition_bindings=None, collimator_override='', allow_stopped=False):
    selected = product(path)
    if selected is None:
        raise ValueError('Choose one individual HB-2A reduced XYE .dat, not a sum or raw detector file')
    data, sig = bounded_bytes(path)
    values = []
    for line in data.decode('utf-8').splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError('Reduced HB-2A product must contain exactly 2theta, intensity, uncertainty')
        row = tuple(map(float, fields))
        if not all(map(math.isfinite, row)) or row[2] < 0:
            raise ValueError('Nonfinite or negative-uncertainty reduced data')
        if row[1] < 0:
            raise ValueError('Negative CW intensities require a negative-preserving importer; XYE import would clip them')
        if row[2] == 0 and row[1] != 0:
            raise ValueError('Zero uncertainty with nonzero intensity requires corrected reduction')
        values.append(row)
    if not 32 <= len(values) <= 100000:
        raise ValueError('Incomplete or oversized individual HB-2A pattern')
    if not 0 < values[0][0] < values[-1][0] < 180 or any(b[0] <= a[0] for a,b in zip(values,values[1:])):
        raise ValueError('HB-2A 2theta coordinates must increase within (0, 180) degrees')
    metadata = read_metadata(ipts, selected.experiment, selected.scan, condition_bindings=condition_bindings,
        collimator_override=collimator_override, allow_stopped=allow_stopped)
    return Setup(selected.experiment, selected.scan, selected.normalization, sha256(data).hexdigest(),
        sig, len(values), values[0][0], values[-1][0], sum(v[2] == 0 for v in values), metadata)


def validate_profile(setup, path, *, acknowledged=False):
    if not acknowledged:
        raise ValueError('Confirm that the profile matches this cycle, Ge reflection, collimation and sample environment')
    if Path(path).suffix.lower() != '.instprm':
        raise ValueError('HB-2A Live currently requires one modern GSAS-II .instprm profile')
    data, _ = bounded_bytes(path)
    parameters = {}
    for line in data.decode('utf-8').splitlines():
        if line.lstrip().startswith('#'):
            continue
        for term in line.split(';'):
            if ':' not in term:
                continue
            key, value = (v.strip() for v in term.split(':', 1))
            if key in parameters:
                raise ValueError('Multiple/duplicate profile blocks require an explicitly selected single-bank file')
            parameters[key] = value
    if parameters.get('Type') != 'PNC':
        raise ValueError('HB-2A requires neutron constant-wavelength Type:PNC, never PNT/PXC')
    if 'Bank' in parameters and float(parameters['Bank']) != 1:
        raise ValueError('Provide one combined-bank CW profile with Bank:1; other labels need explicit remapping')
    required = ['Lam', 'Zero', 'U', 'V', 'W', 'X', 'Y']
    try:
        numeric = {key: float(parameters[key]) for key in required}
    except (KeyError, ValueError) as exc:
        raise ValueError('CW profile lacks finite Lam/Zero/U/V/W/X/Y parameters') from exc
    if not all(map(math.isfinite, numeric.values())) or numeric['Lam'] <= 0:
        raise ValueError('Invalid CW wavelength/profile coefficients')
    nominal = setup.metadata['nominal_wavelength_angstrom']
    if abs(numeric['Lam']/nominal - 1) > 0.01:
        raise ValueError('Profile wavelength conflicts with the measured Ge monochromator setting')
    # Sanity over the actual measured range, not a nominal instrument range.
    for i in range(101):
        theta = math.radians((setup.two_theta_min + (setup.two_theta_max-setup.two_theta_min)*i/100)/2)
        t = math.tan(theta)
        if numeric['U']*t*t + numeric['V']*t + numeric['W'] <= 0:
            raise ValueError('CW Gaussian profile variance is nonpositive in the measured range')
    return {'profile_sha256': sha256(data).hexdigest(), 'wavelength_angstrom': numeric['Lam'],
        'instrument_type': 'PNC', 'profile_acceptance': 'user-reviewed mapping; not independently calibrated',
        'parameters': numeric}


def stage_pattern(source, setup, profile_path, destination, *, acknowledged=False, facility_profile_mapping=None):
    receipt = validate_profile(setup, profile_path, acknowledged=acknowledged)
    data, sig = bounded_bytes(source)
    if sha256(data).hexdigest() != setup.source_sha256 or sig != setup.source_signature:
        raise ValueError('Reduced product changed after discovery; check a new watch')
    current = bounded_bytes(setup.metadata['spice_path'])[0]
    if sha256(current).hexdigest() != setup.metadata['spice_sha256']:
        raise ValueError('SPICE metadata changed after discovery; check a new watch')
    warnings = [{'code':'hb2a_profile_requires_review', 'message':(
        'HB-2A published cycle profile matched exactly; suitability of sample/environment corrections requires review.'
        if facility_profile_mapping else 'HB-2A profile supplied by user; independent calibration acceptance is pending.')}]
    warnings.extend({'code':'hb2a_acquisition_review','message':notice} for notice in setup.metadata['metadata_notices']
        if 'Collimation supplied' in notice or 'Stopped acquisition' in notice)
    if setup.normalization == 'legacy':
        warnings.append({'code':'hb2a_legacy_normalization', 'message':'Legacy reduction normalization is unqualified; do not compare absolute intensities across products.'})
    if setup.masked_points:
        # GSAS-II's XYE reader masks zero/nonpositive sigma; preserve the bytes.
        warnings.append({'code':'hb2a_masked_bins', 'message':f'{setup.masked_points} zero-intensity/zero-uncertainty bins require importer mask verification.'})
    metadata = setup.metadata
    input_provenance = {'schema':'radar-pd-input-provenance/v1', 'instrument':'HB2A', 'source_bank':1,
        'reduction_family':'HB2A individual reduced XYE', 'calibration':receipt['profile_sha256'],
        'source_sha256':setup.source_sha256, 'spice_sha256':metadata['spice_sha256'],
        'condition_receipt_sha256':metadata['conditions']['metadata_sha256'],
        'experiment_number':setup.experiment, 'scan_number':setup.scan,
        'profile_sha256':receipt['profile_sha256'], 'reduction_setup':{
            'experiment_number':setup.experiment, 'ge_reflection':metadata['ge_reflection'],
            'collimator':metadata['collimator'], 'normalization':setup.normalization,
            'acquisition_schema':metadata['channel_schema_sha256']}, 'warnings':warnings}
    if facility_profile_mapping:
        if facility_profile_mapping['sha256'] != receipt['profile_sha256']:
            raise ValueError('Published calibration receipt does not match the staged HB-2A profile')
        input_provenance['facility_profile_mapping'] = facility_profile_mapping
    marker = ('# RADAR-PD input provenance: '+json.dumps(input_provenance,separators=(',',':'))+'\n').encode('ascii')
    if len(marker)>8192:
        raise ValueError('HB-2A provenance marker exceeds supported bound')
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    staged=out/f'HB2A_exp{setup.experiment:04d}_scan{setup.scan:04d}.xye'
    profile=out/'profile.instprm'
    condition_header=(f"# temperature = {metadata['temperature']['value']:.15g}\n".encode('ascii')
        if metadata.get('temperature') else b'')
    staged.write_bytes(marker+condition_header+data);profile.write_bytes(Path(profile_path).read_bytes())
    if sha256(profile.read_bytes()).hexdigest()!=receipt['profile_sha256']:
        raise ValueError('Profile changed during staging')
    receipt.update(input_provenance, source_path=str(source), source_bank=1, staged_bank=1,
        staged_data_sha256=sha256(staged.read_bytes()).hexdigest(), numeric_sha256=setup.source_sha256,
        setup=setup.as_dict())
    (out/'pattern-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return staged,profile,receipt
