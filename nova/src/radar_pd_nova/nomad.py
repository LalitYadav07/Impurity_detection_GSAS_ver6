"""Strict individual-run NOMAD Bragg products and explicit profile mappings.

No raw reduction, summed products, PDF analysis or guessed calibration rules.
The prototype requires a user-selected bank and profile, and carries the
unverified-profile/reduction-quality notices through to scientific results.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re

VERSION = 'nomad-individual-bragg/v1'
MODERN_FOLDER = 'shared/autoreduce/multi_banks/GSAS'
LEGACY_FOLDER = 'shared/autoreduce'
_FILE = re.compile(r'NOM_(\d+)\.gsa', re.I)
_BANK = re.compile(r'^BANK\s+(\d+)\s+(\d+)\s+(\d+)\s+SLOG\s+.*\s+FXYE\s*$')
_NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
_MAX_GSA = 64 * 1024 * 1024


def signature(path):
    stat = Path(path).stat()
    return (stat.st_size, stat.st_mtime_ns)


def source_directory(ipts, subfolder=MODERN_FOLDER):
    if not re.fullmatch(r'IPTS-\d+', str(ipts).upper()) or subfolder not in {MODERN_FOLDER, LEGACY_FOLDER}:
        raise ValueError('Choose an IPTS and an individual NOMAD Bragg product folder')
    return Path('/SNS/NOM') / str(ipts).upper() / subfolder


def reduction_input_path(path):
    path = Path(path)
    return path.parent.parent / 'Input' / (path.stem+'.json') if path.parent.name == 'GSAS' else path.with_suffix('.json')


def directory_listing(directory):
    import os
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f'NOMAD Bragg product directory is unavailable: {root}')
    rows = []
    with os.scandir(root) as entries:
        for e in entries:
            if not _FILE.fullmatch(e.name) or not e.is_file(follow_symlinks=False):
                continue
            try:
                stat = e.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            rows.append({'path':str(root/e.name), 'size':stat.st_size, 'modified_ns':stat.st_mtime_ns})
    return rows


def source_options(ipts):
    return [{'title': title, 'value': folder} for title, folder in
            [('Individual Bragg reductions',MODERN_FOLDER), ('Legacy individual reductions',LEGACY_FOLDER)]
            if source_directory(ipts,folder).is_dir()]


@dataclass(frozen=True)
class NomadBank:
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


@dataclass(frozen=True)
class NomadSetup:
    run_number: int
    banks: tuple[NomadBank, ...]
    source_signature: tuple[int,int]
    source_sha256: str
    sidecar_path: str
    sidecar_signature: tuple[int,int] | None
    sidecar_sha256: str
    calibration: str
    environment: str
    title: str
    warnings: tuple[str,...]

    def as_dict(self):
        return asdict(self)


def inspect_reduced_setup(path, *, selected_bank=None, inventory_only=False):
    path = Path(path)
    if selected_bank is not None and (isinstance(selected_bank,bool) or selected_bank not in range(1,7)):
        raise ValueError('Choose a supported NOMAD histogram label')
    name = _FILE.fullmatch(path.name)
    before = signature(path)
    if not name or before[0] > _MAX_GSA:
        raise ValueError('Expected a bounded individual NOM_<run>.gsa Bragg reduction')
    banks, prefix, values, digest = [], [], None, sha256()
    geometry = None
    declared = None
    count = 0
    quality_errors = []
    def finish():
        if values is None:
            return
        if count != values['records']:
            raise ValueError('NOMAD bank is truncated or still being written')
        if quality_errors and not inventory_only and (selected_bank is None or values['bank']==selected_bank):
            raise ValueError(quality_errors[0])
        banks.append(NomadBank(**values, numeric_sha256=bank_digest.hexdigest(), quality_errors=tuple(dict.fromkeys(quality_errors))))
    with path.open('rb') as stream:
        for line in stream:
            digest.update(line)
            if len(line)>8192:
                raise ValueError('NOMAD GSAS line exceeds the supported size')
            text=line.decode('utf-8').strip()
            if text.startswith('# Total flight path'):
                found=re.search(r'tth\s+('+_NUMBER+r')deg,\s*DIFC\s+('+_NUMBER+')',text)
                if not found:
                    raise ValueError('NOMAD bank export geometry is incomplete')
                geometry=(float(found[1]),float(found[2]),text)
            if text.startswith('BANK'):
                finish();m=_BANK.fullmatch(text)
                if not m or not geometry:
                    raise ValueError('NOMAD requires explicit SLOG/FXYE banks with export geometry')
                bank,channels,records=map(int,m.groups())
                if bank not in range(1,7) or records<=0 or channels!=records:
                    raise ValueError('Invalid NOMAD physical bank or record counts')
                values={'bank':bank,'channels':channels,'records':records,'header':text,
                        'angle':geometry[0],'difc':geometry[1],'geometry_header':geometry[2],
                        'minimum_tof':0.,'maximum_tof':0.}
                if not all(math.isfinite(v) and v>0 for v in geometry[:2]) or geometry[0]>=180:
                    raise ValueError('Invalid NOMAD export geometry')
                geometry=None;count=0;bank_digest=sha256();quality_errors=[]
                continue
            if values is None:
                prefix.append(text)
                m=re.fullmatch(r'#\s*(\d+)\s+Histograms',text)
                if m:declared=int(m[1])
                if len(prefix)>100:
                    raise ValueError('No NOMAD bank in the bounded GSAS header')
            elif text and not text.startswith('#'):
                parts=text.split()
                if len(parts)!=3:
                    quality_errors.append('Malformed NOMAD FXYE record')
                    count+=1;bank_digest.update(line)
                    continue
                try:x,y,sigma=map(float,parts)
                except ValueError:
                    quality_errors.append('Malformed NOMAD FXYE record')
                    count+=1;bank_digest.update(line)
                    continue
                if not all(math.isfinite(v) for v in (x,y,sigma)) or sigma<=0 or x<=0:
                    quality_errors.append('Nonfinite data or nonpositive uncertainty/TOF in NOMAD bank')
                if count and x<=values['maximum_tof']:
                    quality_errors.append('NOMAD bank TOF must increase strictly')
                if math.isfinite(x) and x>0:
                    if not values['minimum_tof']:values['minimum_tof']=x
                    values['maximum_tof']=x
                count+=1;bank_digest.update(line)
    finish()
    header='\n'.join(prefix);run=re.search(r'Sample Run:\s*(\d+)',header)
    if not run or int(run[1])!=int(name[1]):
        raise ValueError('NOMAD filename and Sample Run header disagree')
    if not banks or declared!=len(banks) or len({b.bank for b in banks})!=len(banks):
        raise ValueError('Missing, duplicate or incomplete NOMAD diffraction banks')
    if signature(path)!=before:
        raise ValueError('NOMAD reduction changed during inspection; wait for completion')
    sidecar=reduction_input_path(path);side_sig=None;side_hash='';calibration='';environment='';notices=[]
    if sidecar.is_file():
        side_sig=signature(sidecar)
        if side_sig[0]>1048576:raise ValueError('NOMAD reduction input exceeds the metadata limit')
        raw=sidecar.read_bytes();side_hash=sha256(raw).hexdigest();config=json.loads(raw)
        if not isinstance(config,dict) or not isinstance(config.get('Sample'),dict):
            raise ValueError('Invalid NOMAD reduction input')
        sample=config['Sample']
        if str(sample.get('Runs','')).strip()!=str(run[1]):
            raise ValueError('Reduction input is summed or does not belong to this individual run')
        if str(config.get('Instrument','NOMAD')).upper() not in {'NOM','NOMAD'}:
            raise ValueError('Reduction input names another instrument')
        calibration_config = config.get('Calibration') or {}
        if not isinstance(calibration_config, dict):
            raise ValueError('Invalid NOMAD detector calibration metadata')
        calibration=str(calibration_config.get('Filename') or '')
        environment=json.dumps(config.get('Environment') or {},sort_keys=True)
        if not isinstance(config.get('Environment'),dict) or not config['Environment'].get('Name'):
            notices.append('Sample environment is not recorded; verify profile compatibility before interpretation')
        if sample.get('DummyInfo') is not False:
            notices.append('Reduction uses placeholder or unconfirmed sample corrections; fractions are exploratory')
        if sample.get('PushPositiveLevel'):
            notices.append('Reduction records an intensity positivity shift; quantitative intensity interpretation needs review')
        if signature(sidecar)!=side_sig:raise ValueError('NOMAD reduction input changed during inspection')
    else:
        notices.append('Original reduction input is unavailable; corrections and calibration provenance are unconfirmed')
    if not calibration:notices.append('Applied detector calibration is not recorded')
    title=next((v[2:] for v in prefix if v.startswith('# ')),'')
    return NomadSetup(int(run[1]),tuple(banks),before,digest.hexdigest(),str(sidecar),side_sig,side_hash,
                      calibration,environment,title,tuple(notices))


def profile_blocks(path):
    path=Path(path)
    if path.suffix.lower()!='.instprm' or path.stat().st_size>1048576:
        raise ValueError('NOMAD currently requires an explicit GSAS-II .instprm file')
    text=path.read_text(encoding='utf-8-sig')
    chunks=re.split(r'(?mi)^(?=#\s*Bank\s+\d+)',text)
    blocks={}
    for chunk in chunks:
        fields={}
        for line in chunk.splitlines():
            if line.lstrip().startswith('#'):continue
            for pair in line.split(';'):
                if ':' not in pair:continue
                key,value=pair.split(':',1);key=key.strip()
                if key in fields:raise ValueError('Duplicate parameter or ambiguous GSAS-II profile block')
                fields[key]=value.strip()
        if not fields:continue
        required={'Type','Bank','difC','2-theta','alpha','beta-0','beta-1','sig-0','sig-1','sig-2'}
        if not required.issubset(fields) or fields.get('Type')!='PNT':
            raise ValueError('NOMAD profile must contain complete neutron TOF peak-shape parameters')
        numeric={k:float(v) for k,v in fields.items() if k!='Type'}
        if not all(math.isfinite(v) for v in numeric.values()):raise ValueError('Nonfinite NOMAD profile parameter')
        bank=int(numeric['Bank'])
        if bank<1 or bank!=numeric['Bank'] or bank in blocks:
            raise ValueError('Duplicate or invalid instrument Bank labels; ambiguous profiles cannot be selected')
        blocks[bank]=(chunk,numeric)
    if not blocks:raise ValueError('No usable GSAS-II profile bank')
    return blocks


def validate_profile(setup, bank, path, profile_bank, *, acknowledged=False):
    if not acknowledged:
        raise ValueError('Confirm the supplied profile and its physical-bank mapping; automatic NOMAD profiles are not yet validated')
    selected=next((b for b in setup.banks if b.bank==bank),None)
    if not selected:raise ValueError('Selected physical bank is absent from this run')
    if selected.quality_errors:raise ValueError('Selected NOMAD bank has invalid numerical records; choose a usable bank or obtain a corrected reduction')
    blocks=profile_blocks(path)
    if profile_bank not in blocks:raise ValueError('Selected profile Bank label is absent; ordinal block numbers are not bank labels')
    text,coeff=blocks[profile_bank]
    # Export DIFC describes nominal geometry, not the calibrated TOF polynomial.
    # Published 2019B bank 2/3 calibrations differ by ~0.84%/2.44%; a 0.5%
    # cutoff incorrectly rejects them. This remains a wrong-bank sanity bound,
    # with explicit acknowledgement and review, never calibration acceptance.
    difference=abs(coeff['difC']/selected.difc-1)
    if coeff['difC']<=0 or difference>.05:
        raise ValueError('Profile DIFC is incompatible with this bank export; do not substitute another bank or nominal geometry')
    if abs(coeff['2-theta']-selected.angle)>5:
        raise ValueError('Profile scattering angle conflicts with the selected physical bank')
    return text,{'resolver':VERSION,'verification':'user_supplied_not_automatically_validated',
        'source_bank':bank,'profile_bank':profile_bank,'profile_filename':Path(path).name,
        'profile_sha256':sha256(Path(path).read_bytes()).hexdigest(),'calibration':setup.calibration,
        'environment':setup.environment,'export_difc':selected.difc,'profile_difc':coeff['difC'],
        'nominal_difc_relative_difference':difference,
        'nominal_geometry_notice':('Calibrated profile DIFC differs from nominal export geometry by more than 0.5%; verify the calibration and physical-bank mapping' if difference>.005 else '')}


def stage_bank(source, setup, bank, profile_path, profile_bank, destination, *, acknowledged=False):
    source=Path(source);sidecar=Path(setup.sidecar_path)
    def unchanged():
        if signature(source)!=setup.source_signature or (signature(sidecar) if sidecar.is_file() else None)!=setup.sidecar_signature:
            raise ValueError('NOMAD source or reduction input changed after inspection')
        if setup.sidecar_signature is not None and sha256(sidecar.read_bytes()).hexdigest()!=setup.sidecar_sha256:
            raise ValueError('NOMAD reduction input content changed after inspection')
    unchanged()
    text,receipt=validate_profile(setup,bank,profile_path,profile_bank,acknowledged=acknowledged)
    header=[];payload=[];active=False;seen_bank=False;actual=sha256()
    selected=next(b for b in setup.banks if b.bank==bank)
    with source.open('rb') as stream:
        for raw in stream:
            actual.update(raw)
            if raw.startswith(b'BANK'):
                seen_bank=True;active=int(raw.split()[1])==bank
                if active:payload.append(re.sub(rb'^(BANK\s+)\d+',rb'\g<1>1',raw))
            elif not seen_bank:
                if raw.startswith((b'# Total flight path',b'# Data for spectrum')):continue
                header.append(re.sub(rb'^(#\s*)\d+(\s+Histograms)',rb'\g<1>1\2',raw))
            elif active and raw.strip() and not raw.lstrip().startswith(b'#'):
                payload.append(raw)
    unchanged()
    if actual.hexdigest()!=setup.source_sha256 or len(payload)!=selected.records+1:
        raise ValueError('NOMAD source changed or selected bank is incomplete')
    numeric_hash=sha256(b''.join(payload[1:])).hexdigest()
    if numeric_hash!=selected.numeric_sha256:raise ValueError('Selected bank numerical records changed')
    text=re.sub(r'(^|;)(\s*Bank\s*:)\s*[^;\n]+',r'\g<1>\g<2>1.0',text,flags=re.M)
    text=re.sub(r'(?mi)^#\s*Bank\s+\d+', '#Bank 1', text)
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    data=out/f'NOM_{setup.run_number}_bank{bank}.gsa';profile=out/f'bank{bank}_profile.instprm'
    header.append(f'# Source NOMAD physical bank {bank}; staged local bank 1\n'.encode())
    header.append((selected.geometry_header+'\n').encode('ascii'))
    geometry_notices=([{'code':'nomad_profile_nominal_geometry_offset','message':receipt['nominal_geometry_notice']}] if receipt['nominal_geometry_notice'] else [])
    input_provenance={'schema':'radar-pd-input-provenance/v1','instrument':'NOM',
        'reduction_family':'NOMAD individual GSAS/SLOG',
        'reduction_setup':{'calibration':setup.calibration,'environment':setup.environment},
        'source_bank':bank,'profile_bank':profile_bank,'calibration':setup.calibration,
        'source_sha256':setup.source_sha256,'reduction_input_sha256':setup.sidecar_sha256,
        'profile_sha256':receipt['profile_sha256'],
        'warnings':[{'code':'unverified_nomad_profile','message':'NOMAD profile supplied by user; scientific calibration acceptance is pending'},
                    *geometry_notices,
                    *[{'code':'nomad_reduction_quality','message':message} for message in setup.warnings]]}
    marker=('# RADAR-PD input provenance: '+json.dumps(input_provenance,separators=(',',':'))+'\n').encode('ascii')
    if len(marker)>8192:raise ValueError('NOMAD observation provenance exceeds its supported header size')
    header.insert(1,marker)
    data.write_bytes(b''.join(header+payload));profile.write_text(text,encoding='utf-8')
    if sha256(Path(profile_path).read_bytes()).hexdigest()!=receipt['profile_sha256']:
        raise ValueError('Supplied profile changed while staging')
    receipt.update({'source_path':str(source),'source_sha256':actual.hexdigest(),'staged_bank':1,
        'numeric_sha256':numeric_hash,'data_sha256':sha256(data.read_bytes()).hexdigest(),
        'staged_profile_sha256':sha256(profile.read_bytes()).hexdigest(),'reduction_input_sha256':setup.sidecar_sha256,
        'warnings':list(setup.warnings),'setup':setup.as_dict()})
    (out/'bank-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return data,profile,receipt
