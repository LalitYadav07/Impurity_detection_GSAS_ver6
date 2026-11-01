"""Bounded structural/numeric inspection of explicit neutron SLOG/FXYE banks.

This checks file completeness and export geometry, not calibration validity.
Instrument adapters remain responsible for run naming and reduction evidence.
"""
from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class TofBank:
    bank: int
    channels: int
    records: int
    header: str
    difc: float
    angle: float
    minimum_tof: float
    maximum_tof: float
    numeric_sha256: str
    geometry_header: str
    quality_errors: tuple[str, ...] = ()


def signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def inspect_slog_banks(path, run_number, *, label, selected_bank=None, inventory_only=False, allowed_run_numbers=None):
    path = Path(path)
    before = signature(path)
    if before[0] > 64 * 1024 * 1024:
        raise ValueError(f'{label} reduction exceeds the 64 MiB inspection limit')
    banks, prefix, values, digest = [], [], None, sha256()
    geometry, declared, count, errors = None, None, 0, []
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'

    def finish():
        if values is None:
            return
        if count != values['records']:
            raise ValueError(f'{label} bank is truncated or still being written')
        if errors and not inventory_only and (selected_bank is None or values['bank'] == selected_bank):
            raise ValueError(errors[0])
        banks.append(TofBank(**values, numeric_sha256=bank_digest.hexdigest(),
                             quality_errors=tuple(dict.fromkeys(errors))))

    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            if len(line) > 8192:
                raise ValueError(f'{label} GSAS line exceeds the supported size')
            text = line.decode('utf-8').strip()
            if text.startswith('# Total flight path'):
                found = re.search(r'tth\s+(' + number + r')deg,\s*DIFC\s+(' + number + ')', text)
                if not found:
                    raise ValueError(f'{label} bank export geometry is incomplete')
                geometry = float(found[1]), float(found[2]), text
            if text.startswith('BANK'):
                finish()
                match = re.fullmatch(r'BANK\s+(\d+)\s+(\d+)\s+(\d+)\s+SLOG\s+.*\s+FXYE', text)
                if not match or not geometry:
                    raise ValueError(f'{label} requires explicit SLOG/FXYE banks with export geometry')
                bank, channels, records = map(int, match.groups())
                if bank not in range(1, 7) or records <= 0 or channels != records:
                    raise ValueError(f'Invalid {label} physical bank or record counts')
                if not all(math.isfinite(v) and v > 0 for v in geometry[:2]) or geometry[0] >= 180:
                    raise ValueError(f'Invalid {label} export geometry')
                values = dict(bank=bank, channels=channels, records=records, header=text,
                              angle=geometry[0], difc=geometry[1], geometry_header=geometry[2],
                              minimum_tof=0., maximum_tof=0.)
                geometry, count, bank_digest, errors = None, 0, sha256(), []
                continue
            if values is None:
                prefix.append(text)
                match = re.fullmatch(r'#\s*(\d+)\s+Histograms', text)
                if match:
                    declared = int(match[1])
                if len(prefix) > 100:
                    raise ValueError(f'No {label} bank in the bounded GSAS header')
            elif text and not text.startswith('#'):
                parts = text.split()
                try:
                    if len(parts) != 3:
                        raise ValueError()
                    x, y, sigma = map(float, parts)
                except ValueError:
                    errors.append(f'Malformed {label} FXYE record')
                    count += 1
                    bank_digest.update(line)
                    continue
                if not all(math.isfinite(v) for v in (x, y, sigma)) or sigma <= 0 or x <= 0:
                    errors.append(f'Nonfinite data or nonpositive uncertainty/TOF in {label} bank')
                if count and x <= values['maximum_tof']:
                    errors.append(f'{label} bank TOF must increase strictly')
                if math.isfinite(x) and x > 0:
                    if not values['minimum_tof']:
                        values['minimum_tof'] = x
                    values['maximum_tof'] = x
                count += 1
                bank_digest.update(line)
    finish()
    run = re.search(r'Sample Run:\s*(\d+)', '\n'.join(prefix))
    if not run or int(run[1]) not in (allowed_run_numbers or (run_number,)):
        raise ValueError(f'{label} filename and Sample Run header disagree')
    if not banks or declared != len(banks) or len({b.bank for b in banks}) != len(banks):
        raise ValueError(f'Missing, duplicate or incomplete {label} diffraction banks')
    if signature(path) != before:
        raise ValueError(f'{label} reduction changed during inspection; wait for completion')
    return tuple(banks), before, digest.hexdigest(), prefix
