"""POWGEN preparation adapter with captured, never executed reduction evidence."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from .. import powgen_setup
from ..nomad import profile_blocks
from ..tof_fxye import TofBank, inspect_slog_banks, signature
from .reduction_evidence import literal_effects, merge_effects, normalized

INVALID_COUNTS = 'Nonfinite data or nonpositive uncertainty/TOF in POWGEN bank'
VERSION = 'powgen-battery-slog/v1'


@dataclass(frozen=True)
class PowgenBatterySetup:
    run_number: int
    banks: tuple[TofBank, ...]
    source_signature: tuple[int, int]
    source_sha256: str
    evidence: tuple[dict, ...]
    reduction: powgen_setup.ReducedSetup
    warnings: tuple[str, ...]
    intensity_effects: dict

    def as_dict(self):
        return asdict(self)


def run_number_from_filename(name):
    match = re.fullmatch(r'PG3_(\d+)(?:-\d+)?\.(?:gsa|gss)', str(name), re.I)
    return int(match[1]) if match and int(match[1]) > 0 else None


def inspect_reduced_setup(path, *, selected_bank=None, inventory_only=False):
    path = Path(path)
    if selected_bank is not None and (type(selected_bank) is not int or selected_bank not in range(1, 7)):
        raise ValueError('Choose a supported POWGEN histogram label')
    reduction = powgen_setup.inspect_reduced_setup(path)
    banks, stamp, digest, prefix = inspect_slog_banks(path, reduction.run_number, label='POWGEN',
        selected_bank=selected_bank, inventory_only=inventory_only)
    if stamp != reduction.source_signature:
        raise ValueError('POWGEN reduction changed between structural and numeric inspection')
    if '# with Y multiplied by the bin widths.' not in prefix:
        raise ValueError('POWGEN battery preparation requires recorded bin-width-multiplied intensities; review this reduction convention')
    instruments = [line.split(':', 1)[1].strip().upper() for line in prefix if line.startswith('# Instrument:')]
    if instruments and instruments != ['POWGEN']:
        raise ValueError('Reduced header names a different or ambiguous instrument')
    evidence, effects = [], []
    for suffix, expected, checksum in (
            ('.py', reduction.reduction_script_signature, reduction.reduction_script_sha256),
            ('.xml', reduction.reduction_xml_signature, reduction.reduction_xml_sha256)):
        sidecar = path.with_suffix(suffix)
        if (signature(sidecar) if sidecar.is_file() else None) != expected:
            raise ValueError('POWGEN reduction evidence changed during inspection')
        if expected:
            raw = sidecar.read_bytes()
            if sha256(raw).hexdigest() != checksum:
                raise ValueError('POWGEN reduction evidence content changed during inspection')
            evidence.append(dict(path=str(sidecar), signature=expected, sha256=checksum))
            effects.append(literal_effects(raw, suffix, reduction.run_number))
    effects = merge_effects(effects)
    warnings = []
    if not evidence:
        warnings.append('Original reduction receipts are unavailable; supplied profiles require explicit review')
    if 'PushDataPositive' in effects and normalized(effects['PushDataPositive']) != 'none':
        warnings.append('Reduction records PushDataPositive=' + str(effects['PushDataPositive']) + '; intensity shifts require review before quantitative interpretation')
    if 'OffsetData' in effects and normalized(effects['OffsetData']) != 0:
        warnings.append('Reduction records an intensity offset; quantitative interpretation requires review')
    warnings.append('Profile mapping and native preparation do not establish scientific calibration or model adequacy')
    return PowgenBatterySetup(reduction.run_number, banks, stamp, digest, tuple(evidence), reduction, tuple(warnings), effects)


def catalog_profile(setup, bank):
    """Use the existing strict cycle/frame/grouping resolver without fallbacks."""
    return powgen_setup.resolve_bank_profile(setup.reduction, bank)


def verify_evidence(source, setup):
    """Check missing sidecars too: a newly completed receipt changes the review."""
    recorded = {Path(row['path']).suffix: row for row in setup.evidence}
    for suffix in ('.py', '.xml'):
        path = Path(source).with_suffix(suffix)
        row = recorded.get(suffix)
        if (signature(path) if path.is_file() else None) != (row['signature'] if row else None):
            raise ValueError('POWGEN reduction evidence changed after inspection')
        if row and sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('POWGEN reduction evidence content changed after inspection')


def stage_bank(source, setup, bank, profile_path, profile_bank, destination, *, acquisition, allow_invalid_counts=False):
    source, profile_path = Path(source), Path(profile_path)

    def unchanged():
        if signature(source) != setup.source_signature:
            raise ValueError('POWGEN observation changed after inspection')
        verify_evidence(source, setup)

    unchanged()
    selected = next((b for b in setup.banks if b.bank == bank), None)
    if selected is None:
        raise ValueError('Selected physical bank is absent from this run')
    if selected.quality_errors and not (allow_invalid_counts and set(selected.quality_errors) == {INVALID_COUNTS}):
        raise ValueError('Selected POWGEN bank has invalid numerical records')
    blocks = profile_blocks(profile_path)
    if profile_bank not in blocks:
        raise ValueError('Selected profile Bank label is absent')
    text, coefficients = blocks[profile_bank]
    difference = abs(coefficients['difC'] / selected.difc - 1)
    if coefficients['difC'] <= 0 or difference > .05 or abs(coefficients['2-theta'] - selected.angle) > 5:
        raise ValueError('Profile geometry conflicts with the selected POWGEN bank; review its mapping')
    profile_hash = sha256(profile_path.read_bytes()).hexdigest()
    header, payload, active, seen, digest = [], [], False, False, sha256()
    with source.open('rb') as stream:
        for raw in stream:
            digest.update(raw)
            if raw.startswith(b'BANK'):
                seen = True
                active = int(raw.split()[1]) == bank
                if active:
                    payload.append(re.sub(rb'^(BANK\s+)\d+', rb'\g<1>1', raw))
            elif not seen:
                if raw.startswith((b'# Total flight path', b'# Data for spectrum')):
                    continue
                header.append(re.sub(rb'^(#\s*)\d+(\s+Histograms)', rb'\g<1>1\2', raw))
            elif active and raw.strip() and not raw.lstrip().startswith(b'#'):
                payload.append(raw)
    unchanged()
    numeric_hash = sha256(b''.join(payload[1:])).hexdigest()
    if digest.hexdigest() != setup.source_sha256 or len(payload) != selected.records + 1 or numeric_hash != selected.numeric_sha256:
        raise ValueError('POWGEN source or selected numerical records changed')
    invalid, previous = [], 0.
    for index, line in enumerate(payload[1:]):
        tokens = line.split()
        x, y, sigma = map(float, tokens)
        if not math.isfinite(x) or x <= previous:
            raise ValueError('Invalid TOF coordinates cannot be excluded by an intensity/uncertainty policy')
        previous = x
        if not math.isfinite(y) or not math.isfinite(sigma) or sigma <= 0:
            if not allow_invalid_counts:
                raise ValueError('Invalid POWGEN intensity/uncertainty records require an explicit outside-fit policy')
            invalid.append(dict(record_index=index, intensity=tokens[1].decode('ascii'), uncertainty=tokens[2].decode('ascii')))
            payload[index + 1] = tokens[0] + b' 0 1\n'
    text = re.sub(r'(^|;)(\s*Bank\s*:)\s*[^;\n]+', r'\g<1>\g<2>1.0', text, flags=re.M)
    text = re.sub(r'(?mi)^#\s*Bank\s+\d+', '#Bank 1', text)
    reduction = setup.reduction
    provenance = dict(schema='radar-pd-input-provenance/v1', instrument='PG3',
        reduction_family='POWGEN individual GSAS/FXYE', source_bank=bank, profile_bank=profile_bank,
        source_sha256=setup.source_sha256, profile_sha256=profile_hash,
        reduction_setup={key: getattr(reduction, key) for key in
            ('cycle', 'mode', 'wavelength', 'frequency', 'guide', 'calibration_files', 'characterization_files', 'grouping_files')},
        reduction_evidence=[dict(path=Path(row['path']).name, sha256=row['sha256']) for row in setup.evidence],
        intensity_effects=setup.intensity_effects,
        invalid_count_bin_indices=[row['record_index'] for row in invalid], acquisition=acquisition,
        warnings=[dict(code='powgen_preparation_unqualified', message=message) for message in setup.warnings])
    marker = ('# RADAR-PD input provenance: ' + json.dumps(provenance, separators=(',', ':')) + '\n').encode('ascii')
    if len(marker) > 8192:
        raise ValueError('POWGEN observation provenance exceeds the supported header size')
    header.insert(1, marker)
    header.extend([f'# Source POWGEN physical bank {bank}; staged local bank 1\n'.encode(),
                   (selected.geometry_header + '\n').encode('ascii')])
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    data, profile = out / f'PG3_{setup.run_number}_bank{bank}.gsa', out / f'bank{bank}_profile.instprm'
    data.write_bytes(b''.join(header + payload))
    profile.write_text(text)
    if sha256(profile_path.read_bytes()).hexdigest() != profile_hash:
        raise ValueError('Supplied profile changed while staging')
    receipt = dict(resolver=VERSION, verification='preparation_requires_scientific_validation', source_bank=bank,
        profile_bank=profile_bank, profile_sha256=profile_hash, source_sha256=setup.source_sha256,
        data_sha256=sha256(data.read_bytes()).hexdigest(), staged_profile_sha256=sha256(profile.read_bytes()).hexdigest(),
        source_numeric_sha256=numeric_hash, numeric_sha256=sha256(b''.join(payload[1:])).hexdigest(),
        invalid_count_bins=invalid, nominal_difc_relative_difference=difference, setup=setup.as_dict())
    (out / 'bank-provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return data, profile, receipt
