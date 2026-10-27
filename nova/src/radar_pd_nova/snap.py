"""Individual SNAP Bragg exports with explicit state/group/profile provenance.

SNAPRed's generated GSAS-II profiles are starting estimates, not accepted
peak-shape calibrations. An adjacent file is a candidate, never an automatic
scientific certification. No event reduction or cross-group averaging occurs.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re

from .diffraction_coordinates import TofCalibration

VERSION = "snap-individual-bragg/v1"
DEFAULT_FOLDER = "shared/SNAPRed/export/gsa/all"
LEGACY_FOLDERS = ("shared/autoreduce/gsas", "shared/autoreduce/GSAS",
                  "shared/data/gsas", "shared/gsas", "shared")
MAX_GSA = 64 * 1024 * 1024
MAX_RECEIPT = 1024 * 1024
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_FILE = re.compile(r"(?:(?P<prefix>[A-Za-z][A-Za-z0-9_.-]{0,40})_)?SNAP_?(?P<run>\d+)(?:_(?P<tail>[^/\\]{1,100}))?\.gsa", re.I)
_BANK = re.compile(r"BANK\s+(\d+)\s+(\d+)\s+(\d+)\s+SLOG\s+"+_NUMBER+r"\s+"+_NUMBER+r"\s+"+_NUMBER+r"\s+0\s+FXYE\s*", re.I)


def signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True)
class SnapFilename:
    run_number: int
    grouping: str
    variant: str

    @property
    def family(self):
        return json.dumps([self.grouping, self.variant], separators=(",", ":"))


def parse_filename(path):
    match = _FILE.fullmatch(Path(path).name)
    if not match or int(match["run"]) < 1:
        raise ValueError("Expected one individually numbered SNAP GSAS export")
    tail = match["tail"] or ""
    first, _, rest = tail.partition("_")
    aliases = {"all": "all", "bank": "bank", "banks": "bank", "column": "column", "columns": "column"}
    grouping = aliases.get(first.casefold(), first.casefold() if re.fullmatch(r'\d+(?:p\d+)?deg',first,re.I) else "unclassified")
    variant = rest if grouping != "unclassified" else tail
    two_four = re.fullmatch(r'2_4(?:_(.*))?',tail,re.I)
    if two_four:
        grouping, variant = '2_4', two_four[1] or ''
    if match["prefix"]:
        variant = match["prefix"] + ("_" + variant if variant else "")
    return SnapFilename(int(match["run"]), grouping, variant)


def source_directory(ipts, subfolder=DEFAULT_FOLDER):
    ipts = str(ipts).upper()
    if not re.fullmatch(r"IPTS-[1-9]\d*", ipts):
        raise ValueError("Choose a positive SNAP IPTS identifier")
    modern = re.fullmatch(r"shared/(?:SNAPRed|autoreduce)/export/gsa/[A-Za-z0-9_-]{1,40}", str(subfolder))
    if subfolder not in LEGACY_FOLDERS and not modern:
        raise ValueError("Choose a named SNAP GSAS export folder; reductions are never recursively merged")
    return Path("/SNS/SNAP") / ipts / subfolder


def source_options(ipts):
    options = []
    root = source_directory(ipts, "shared")
    for parent in ("SNAPRed/export/gsa", "autoreduce/export/gsa"):
        folder = root / parent
        if not folder.is_dir():
            continue
        with os.scandir(folder) as entries:
            names = sorted(e.name for e in entries if re.fullmatch(r"[A-Za-z0-9_-]{1,40}", e.name)
                           and e.is_dir(follow_symlinks=False))
        options.extend({"title": f"{parent.split('/')[0]} · {name}", "value": "shared/"+parent+"/"+name} for name in names)
    for folder in LEGACY_FOLDERS:
        path = source_directory(ipts, folder)
        if path.is_dir() and directory_listing(path):
            options.append({"title": "Legacy · "+folder, "value": folder})
    return options


def directory_listing(directory, *, family=None):
    rows = []
    with os.scandir(directory) as entries:
        for entry in entries:
            try:
                parsed = parse_filename(entry.name)
                if family is not None and parsed.family != family:
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except (ValueError, FileNotFoundError):
                continue
            rows.append({"path": str(Path(directory)/entry.name), "size": stat.st_size,
                         "modified_ns": stat.st_mtime_ns, "snap_product_family": parsed.family})
    return rows


def product_families(rows):
    counts = {}
    for row in rows:
        parsed = parse_filename(row["path"])
        counts[parsed.family] = counts.get(parsed.family, 0)+1
    return [{"value": key, "title": f"{group} · {variant or 'standard export'} ({counts[key]} files)"}
            for key in sorted(counts) for group, variant in [json.loads(key)]]


@dataclass(frozen=True)
class SnapBank:
    bank: int
    channels: int
    records: int
    header: str
    difc: float
    angle: float
    geometry_header: str
    minimum_tof: float
    maximum_tof: float
    numeric_sha256: str
    valid_bins: int
    masked_bins: int
    masked_intervals: tuple[tuple[float, float], ...]
    quality_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapSetup:
    run_number: int
    banks: tuple[SnapBank, ...]
    source_signature: tuple[int, int]
    source_sha256: str
    grouping: str
    product_variant: str
    record_path: str
    record_signature: tuple[int, int] | None
    record_sha256: str
    state_id: str
    use_lite_mode: bool | None
    calibration: str
    normalization: str
    corrections: str
    title: str
    warnings: tuple[str, ...]

    def as_dict(self):
        return asdict(self)


def _reduction_record(prefix, path, filename, *, ipts_root=None):
    pointers = re.findall(r'^\s*\{?"redRecord"\s*:\s*"([^"\n]+)"\s*,?\s*$', prefix, re.M)
    if len(pointers) > 1:
        raise ValueError("Several SNAP reduction records are named by this product")
    empty = ("", None, "", "", None, "", "", [])
    if not pointers:
        return (*empty[:-1], ["Original SNAP reduction record is unavailable; state and corrections require review"])
    pointer = Path(pointers[0])
    matched = re.fullmatch(r"/SNS/SNAP/(IPTS-[1-9]\d*)/shared/SNAPRed/([a-f0-9]{16})/(lite|native)/(\d+)/([^/]+)/ReductionRecord\.json", str(pointer))
    if not matched or int(matched[4]) != filename.run_number:
        raise ValueError("SNAP reduction-record path does not identify this individual run/state")
    source_ipts = next((parent.name for parent in path.parents if re.fullmatch(r"IPTS-[1-9]\d*", parent.name)), "")
    if source_ipts and matched[1] != source_ipts:
        raise ValueError("SNAP export names another IPTS reduction record; review the source explicitly")
    actual = Path(ipts_root)/pointer.relative_to(Path('/SNS/SNAP')/matched[1]) if ipts_root else pointer
    state, lite = matched[2], matched[3] == "lite"
    if not actual.is_file():
        return str(actual), None, "", state, lite, "", "", ["The exact SNAP reduction record is missing; no nearest/latest record was substituted"]
    before = signature(actual)
    if before[0] > MAX_RECEIPT:
        raise ValueError("SNAP reduction record exceeds the metadata bound")
    raw = actual.read_bytes()
    record = json.loads(raw)
    if (not isinstance(record, dict) or not str(record.get('runNumber','')).isdigit()
            or int(record['runNumber']) != filename.run_number):
        raise ValueError("SNAP reduction record and exported run disagree")
    if record.get('useLiteMode') is not lite:
        raise ValueError("SNAP native/lite reduction provenance disagrees")
    calibration = record.get('calibration') or {}
    parameters = calibration.get('calculationParameters') or {}
    instrument_state = parameters.get('instrumentState') or {}
    identity = instrument_state.get('id') or {}
    if identity.get('hex') and identity['hex'] != state:
        raise ValueError("SNAP state identifier disagrees with the exact reduction record")
    if signature(actual) != before:
        raise ValueError("SNAP reduction record changed while being read")
    calibration_key = json.dumps({key: calibration.get(key) for key in ('version','runNumber','useLiteMode')}, sort_keys=True)
    normalization = record.get('normalization') or {}
    for kind, receipt in [('calibration',calibration),('normalization',normalization)]:
        if 'useLiteMode' in receipt and receipt['useLiteMode'] is not lite:
            raise ValueError(f'SNAP {kind} uses a different native/lite mapping')
        receipt_state = ((receipt.get('calculationParameters') or {}).get('instrumentState') or {}).get('id') or {}
        if receipt_state.get('hex') and receipt_state['hex'] != state:
            raise ValueError(f'SNAP {kind} state conflicts with this reduction')
    normalization_key = json.dumps({key: normalization.get(key) for key in ('version','runNumber','useLiteMode')}, sort_keys=True)
    notices = []
    if not calibration or not calibration.get('version'):
        notices.append('SNAP reduction calibration version is unrecorded')
    if not normalization:
        notices.append('SNAP normalization provenance is unavailable; intensities need review')
    return str(actual), before, sha256(raw).hexdigest(), state, lite, calibration_key, normalization_key, notices


def _correction_annotations(prefix):
    """Parse only the two literal fields of SNAPWrap's three-line header.

    The exporter omits separators between header keys; this is not treated as
    arbitrary repaired JSON, executable code or calibration authority.
    """
    values = {}
    for key in ('attenuationMethod','backgroundMethod'):
        matches = re.findall(r'^\s*"'+key+r'"\s*:\s*(.*?)\s*$',prefix,re.M)
        if len(matches) > 1:
            raise ValueError('Ambiguous repeated SNAP correction annotations')
        if matches:
            literal = matches[0]
            if key == 'backgroundMethod' and literal.endswith('}'):
                literal = literal[:-1].rstrip()
            try:
                values[key] = json.loads(literal)
            except ValueError as exc:
                raise ValueError('Unsupported SNAP correction annotation; review the exported product') from exc
    return json.dumps(values,sort_keys=True,separators=(',',':'))


def inspect_reduced_setup(path, *, selected_bank=None, inventory_only=False, ipts_root=None):
    path = Path(path)
    filename = parse_filename(path)
    before = signature(path)
    if not 0 < before[0] <= MAX_GSA:
        raise ValueError("SNAP reduced product is empty or exceeds the supported bound")
    banks, prefix, current, geometry = [], [], None, None
    digest = sha256()
    def finish():
        if current is None:
            return
        if current['count'] != current['records']:
            raise ValueError("SNAP histogram is truncated or still being written")
        if current['mask_start'] is not None:
            current['intervals'].append((current['mask_start'], current['maximum_tof']))
        errors = list(dict.fromkeys(current['errors']))
        if current['valid_bins'] < 16:
            errors.append('SNAP histogram has fewer than 16 valid weighted bins')
        if errors and not inventory_only and (selected_bank is None or current['bank'] == selected_bank):
            raise ValueError(errors[0])
        banks.append(SnapBank(**{key: current[key] for key in (
            'bank','channels','records','header','difc','angle','geometry_header','minimum_tof','maximum_tof','valid_bins','masked_bins')},
            numeric_sha256=current['digest'].hexdigest(), masked_intervals=tuple(current['intervals']), quality_errors=tuple(errors)))
    total_bytes = 0
    with path.open('rb') as stream:
        for raw in stream:
            total_bytes += len(raw)
            if total_bytes > MAX_GSA:
                raise ValueError('SNAP reduction grew beyond its supported size while being read')
            digest.update(raw)
            if len(raw) > 8192:
                raise ValueError("SNAP export line exceeds its bound")
            text = raw.decode('utf-8').strip()
            if text.startswith('# Total flight path'):
                found = re.search(r'tth\s+('+_NUMBER+r')deg,\s*DIFC\s+('+_NUMBER+')',text)
                if not found:
                    raise ValueError("SNAP histogram export geometry is incomplete")
                geometry = float(found[1]), float(found[2]), text
            if text.startswith('BANK'):
                finish()
                match = _BANK.fullmatch(text)
                if not match or geometry is None:
                    raise ValueError('SNAP needs explicit SLOG/FXYE histograms with export geometry')
                bank, channels, records = map(int, match.groups())
                if not 0 < bank <= 10000 or not 0 < records <= 1_000_000 or records != channels:
                    raise ValueError('Invalid SNAP histogram label/count')
                angle, difc, geom = geometry
                if not 0 < angle < 180 or not math.isfinite(difc) or difc <= 0:
                    raise ValueError('Invalid SNAP export geometry')
                current = dict(bank=bank,channels=channels,records=records,header=text,difc=difc,angle=angle,
                    geometry_header=geom,minimum_tof=0.,maximum_tof=0.,count=0,valid_bins=0,masked_bins=0,
                    digest=sha256(),intervals=[],mask_start=None,errors=[])
                geometry = None
                continue
            if current is None:
                prefix.append(text)
                if len(prefix) > 100:
                    raise ValueError('No bank in bounded SNAP export header')
            elif text and not text.startswith('#'):
                current['count'] += 1
                current['digest'].update(raw)
                try:
                    if len(text.split()) != 3:
                        raise ValueError()
                    x, y, error = map(float, text.split())
                except ValueError:
                    current['errors'].append('Malformed SNAP FXYE numeric record')
                    continue
                if not math.isfinite(x) or x <= 0 or (current['count'] > 1 and x <= current['maximum_tof']):
                    current['errors'].append('SNAP TOF must be finite, positive and strictly increasing')
                if not current['minimum_tof']:
                    current['minimum_tof'] = x
                current['maximum_tof'] = x
                if error == 0:
                    current['masked_bins'] += 1
                    if current['mask_start'] is None:
                        current['mask_start'] = x
                elif (not math.isfinite(error) or error < 0 or not math.isfinite(y)
                      or error*error == 0 or not math.isfinite(error*error)):
                    current['errors'].append('Nonfinite positive-weight data or invalid SNAP uncertainty')
                else:
                    current['valid_bins'] += 1
                    if current['mask_start'] is not None:
                        current['intervals'].append((current['mask_start'], previous_x))
                        current['mask_start'] = None
                previous_x = x
    finish()
    if not banks or len({b.bank for b in banks}) != len(banks):
        raise ValueError('Missing or duplicate SNAP histogram labels')
    header = '\n'.join(prefix)
    sample_run = re.search(r'Sample Run:\s*(\d+)', header)
    if sample_run and int(sample_run[1]) != filename.run_number:
        raise ValueError('SNAP filename and Sample Run header disagree')
    record = _reduction_record(header,path,filename,ipts_root=ipts_root)
    record_path, record_sig, record_hash, state, lite, calibration, normalization, notices = record
    corrections = _correction_annotations(header)
    if not sample_run and not record_path:
        raise ValueError('SNAP run identity needs an explicit Sample Run or reduction-record pointer')
    grouping_folder = path.parent.name.casefold()
    if '/export/gsa/' in str(path) and filename.grouping != grouping_folder:
        raise ValueError('SNAP filename grouping disagrees with its selected export folder')
    if filename.grouping in {'all','bank','column'}:
        expected = {'all':1,'bank':2,'column':6}[filename.grouping]
        if len(banks) != expected:
            notices.append(f'{filename.grouping} export contains {len(banks)} histograms; review masks/custom grouping')
    if signature(path) != before:
        raise ValueError('SNAP product changed during inspection; wait for completion')
    title = next((line[2:] for line in prefix if line.startswith('# ') and not line.startswith('# Total')), '')
    return SnapSetup(filename.run_number,tuple(banks),before,digest.hexdigest(),filename.grouping,filename.variant,
                     record_path,record_sig,record_hash,state,lite,calibration,normalization,corrections,title,tuple(notices))


def profile_blocks(path):
    path = Path(path)
    if path.suffix.lower() != '.instprm' or not 0 < path.stat().st_size <= MAX_RECEIPT:
        raise ValueError('SNAP requires a bounded GSAS-II .instprm with an explicit histogram mapping')
    text = path.read_text(encoding='utf-8-sig')
    blocks = {}
    for chunk in re.split(r'(?mi)^(?=#\s*Bank\s+\d+)',text):
        fields, table = {}, None
        if re.search(r'(?mi)^pdabc\s*:', chunk):
            match = re.search(r'(?ms)^pdabc\s*:\s*"""(.*?)"""',chunk)
            if not match:
                raise ValueError('SNAP profile lookup table is incomplete')
            try:
                table = [[float(value) for value in line.split(',')] for line in match[1].splitlines() if line.strip()]
            except ValueError as exc:
                raise ValueError('Invalid SNAP profile lookup table') from exc
            if not table or any(len(row) != 5 or not all(math.isfinite(v) for v in row) for row in table):
                raise ValueError('SNAP profile lookup table contains nonfinite/unsupported values; obtain a corrected profile')
            without_table = chunk[:match.start()] + chunk[match.end():]
        else:
            without_table = chunk
        for line in without_table.splitlines():
            if line.lstrip().startswith('#'):
                continue
            for pair in line.split(';'):
                if ':' not in pair:
                    continue
                key, value = (v.strip() for v in pair.split(':',1))
                if key in fields:
                    raise ValueError('Duplicate SNAP instrument parameters')
                fields[key] = value
        if not fields:
            continue
        required = {'Type','Bank','difC','2-theta','alpha','beta-0','beta-1','sig-0','sig-1','sig-2'}
        if not required.issubset(fields) or fields['Type'] != 'PNT':
            raise ValueError('SNAP profile needs complete neutron TOF peak-shape parameters')
        numbers = {}
        for key,value in fields.items():
            if key in {'Type','Diff-type'}:
                continue
            try:
                numbers[key] = float(value)
            except ValueError as exc:
                raise ValueError(f'Unsupported SNAP instrument parameter {key}') from exc
        if not all(math.isfinite(v) for v in numbers.values()):
            raise ValueError('Nonfinite SNAP instrument profile')
        bank = numbers['Bank']
        if bank < 1 or int(bank) != bank or int(bank) in blocks:
            raise ValueError('Duplicate or invalid SNAP profile Bank labels')
        blocks[int(bank)] = (chunk,numbers)
    if not blocks:
        raise ValueError('No usable SNAP profile histogram')
    return blocks


def validate_profile(setup, bank, path, profile_bank, *, acknowledged=False):
    if not acknowledged:
        raise ValueError('Confirm the SNAP state, grouping and profile Bank mapping; adjacent auto-generated profiles are approximate')
    selected = next((b for b in setup.banks if b.bank == bank),None)
    if selected is None:
        raise ValueError('Selected SNAP source histogram is absent')
    if selected.quality_errors:
        raise ValueError(selected.quality_errors[0])
    before_profile = signature(path)
    blocks = profile_blocks(path)
    if profile_bank not in blocks:
        raise ValueError('Selected SNAP profile Bank label is absent; labels are not block positions')
    text, numbers = blocks[profile_bank]
    if numbers['difC'] <= 0 or abs(numbers['difC']/selected.difc-1) > .05 or abs(numbers['2-theta']-selected.angle) > 5:
        raise ValueError('SNAP profile geometry is incompatible with this histogram; verify state and focusing group')
    digest = sha256(Path(path).read_bytes()).hexdigest()
    if signature(path) != before_profile:
        raise ValueError('SNAP profile changed while its mapping was validated')
    calibration = TofCalibration.from_profile(path,source_bank=bank,profile_bank=profile_bank,expected_sha256=digest)
    calibration.spacing([selected.minimum_tof,selected.maximum_tof])
    return text, {'resolver':VERSION,'verification':'user_supplied_not_automatically_validated',
        'source_bank':bank,'profile_bank':profile_bank,'profile_sha256':digest,
        'profile_filename':Path(path).name,'state_id':setup.state_id,'grouping':setup.grouping,
        'use_lite_mode':setup.use_lite_mode,'calibration':setup.calibration,'normalization':setup.normalization,
        'corrections':setup.corrections}


def stage_bank(source, setup, bank, profile_path, profile_bank, destination, *, acknowledged=False,
        acquisition=None,condition_bindings=None):
    source, profile_path = Path(source), Path(profile_path)
    def unchanged():
        if signature(source) != setup.source_signature:
            raise ValueError('SNAP source changed after discovery')
        record = Path(setup.record_path) if setup.record_path else None
        current = signature(record) if record and record.is_file() else None
        if current != setup.record_signature:
            raise ValueError('SNAP reduction record changed after discovery')
        if record and current and sha256(record.read_bytes()).hexdigest() != setup.record_sha256:
            raise ValueError('SNAP reduction-record content changed after inspection')
    unchanged()
    text, receipt = validate_profile(setup,bank,profile_path,profile_bank,acknowledged=acknowledged)
    selected = next(b for b in setup.banks if b.bank == bank)
    payload, active = [], False
    raw = source.read_bytes()
    if sha256(raw).hexdigest() != setup.source_sha256:
        raise ValueError('SNAP source content changed after inspection')
    for line in raw.splitlines(keepends=True):
        if line.startswith(b'BANK'):
            active = int(line.split()[1]) == bank
            if active:
                payload.append(re.sub(rb'^(BANK\s+)\d+',rb'\g<1>1',line))
        elif active and line.strip() and not line.lstrip().startswith(b'#'):
            payload.append(line)
    if len(payload) != selected.records+1 or sha256(b''.join(payload[1:])).hexdigest() != selected.numeric_sha256:
        raise ValueError('SNAP staged histogram differs from its inspected numeric records')
    text = re.sub(r'(^|;)(\s*Bank\s*:)\s*[^;\n]+',r'\g<1>\g<2>1.0',text,flags=re.M)
    text = re.sub(r'(?mi)^#\s*Bank\s+\d+', '#Bank 1', text)
    notices = [*setup.warnings,'SNAP profile mapping is user-reviewed; calibration acceptance and quantitative phase fractions require scientific review']
    provenance = {'schema':'radar-pd-input-provenance/v1','instrument':'SNAP',
        'reduction_family':'SNAP individual GSAS/SLOG','source_bank':bank,'profile_bank':profile_bank,
        'source_sha256':setup.source_sha256,'profile_sha256':receipt['profile_sha256'],
        'reduction_input_sha256':setup.record_sha256,'calibration':setup.calibration,
        'reduction_setup':{'state_id':setup.state_id,'grouping':setup.grouping,'use_lite_mode':setup.use_lite_mode,
                           'calibration':setup.calibration,'normalization':setup.normalization,'corrections':setup.corrections,
                           'product_variant':setup.product_variant},
        'masked_bin_policy':'source_error_zero; retain coordinate grid and zero fit weights',
        'masked_bins':selected.masked_bins,
        'warnings':[{'code':'snap_input_review','message':message} for message in notices]}
    if acquisition is not None:
        provenance['acquisition'] = acquisition
    if condition_bindings:
        provenance['condition_bindings'] = condition_bindings
    marker = '# RADAR-PD input provenance: '+json.dumps(provenance,separators=(',',':'),ensure_ascii=True)+'\n'
    if len(marker.encode()) > 8192:
        raise ValueError('SNAP provenance exceeds its bounded portable header')
    title = ' '.join(str(setup.title).split())[:1200]
    header = (f'Sample Run: {setup.run_number}\n'+marker+'# '+title+'\n# 1 Histograms\n'
              '# Instrument: SNAP\n# with Y multiplied by the bin widths.\n'+selected.geometry_header+'\n')
    out = Path(destination);out.mkdir(parents=True,exist_ok=True)
    data = out/f'SNAP_{setup.run_number}_bank{bank}.gsa'
    profile = out/f'bank{bank}_profile.instprm'
    data.write_bytes(header.encode('utf-8')+b''.join(payload));profile.write_text(text,encoding='utf-8')
    unchanged()
    if sha256(profile_path.read_bytes()).hexdigest() != receipt['profile_sha256']:
        raise ValueError('Supplied SNAP profile changed during staging')
    receipt.update(source_path=str(source),source_sha256=setup.source_sha256,staged_bank=1,
        numeric_sha256=selected.numeric_sha256,data_sha256=sha256(data.read_bytes()).hexdigest(),
        staged_profile_sha256=sha256(profile.read_bytes()).hexdigest(),masked_bins=selected.masked_bins,
        masked_intervals_tof=list(selected.masked_intervals),reduction_input_sha256=setup.record_sha256,
        warnings=notices,setup=setup.as_dict())
    receipt['profile'] = {'path':str(profile),'profile_bank':1,'source_bank':bank,
                          'sha256':receipt['staged_profile_sha256']}
    if acquisition is not None:
        receipt['acquisition'] = acquisition
    receipt['condition_bindings'] = condition_bindings or {}
    (out/'bank-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return data,profile,receipt
