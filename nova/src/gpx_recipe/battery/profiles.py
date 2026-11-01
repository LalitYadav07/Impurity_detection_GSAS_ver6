"""Strict native conversion of explicitly selected legacy neutron TOF profiles.

The source file is retained. Conversion never certifies calibration or changes
observations, and rejects legacy incident-spectrum/table corrections that would
alter already-reduced data or cannot be represented by this adapter.
"""
from hashlib import sha256
import math
from numbers import Real
from pathlib import Path
from types import SimpleNamespace

ADAPTER = 'gsas-isotropic-tof-coefficients/v2'


def _numbers(text):
    return [float(value.replace('D', 'E').replace('d', 'e')) for value in text.split()]


def _profile_coefficients(records, prefix, old_style, ptype):
    """Read all declared coefficients, not only those the native importer uses.

    GSAS manual (2004), pp. 144, 148, 223: function 1 has 12 coefficients,
    function 3 has 21, stored four per record. Anisotropy, line displacement,
    stacking faults and type-1 constant alpha cannot be represented here.
    """
    header = records[prefix + ('PRCF  ' if old_style else 'PRCF1 ')]
    try:
        # Fixed-width counts can touch the following scientific-notation cutoff.
        count = int(header[5:10])
        cutoff = float(header[10:].strip().replace('D', 'E').replace('d', 'e'))
    except ValueError:
        try:
            values = _numbers(header)
            count = int(values[1])
            if count != values[1] or len(values) != 3:
                raise ValueError
            cutoff = values[2]
        except (ValueError, OverflowError, IndexError) as exc:
            raise ValueError('Invalid legacy profile coefficient count or cutoff') from exc
    if count != {1: 12, 3: 21}[ptype] or not math.isfinite(cutoff) or not 0 < cutoff < 1:
        raise ValueError('Unsupported legacy profile coefficient count or cutoff')
    coefficients = []
    for index in range(1, (count + 3) // 4 + 1):
        key = prefix + ('PRCF ' if old_style else 'PRCF1') + str(index)
        text = records.get(key)
        if text is None:
            raise ValueError('Missing legacy profile coefficient record: ' + key)
        try:
            values = _numbers(text)
        except ValueError:
            try:
                values = [float(text[i:i+15].replace('D', 'E')) for i in range(0, 60, 15) if text[i:i+15].strip()]
                if text[60:].strip():
                    raise ValueError
            except ValueError as exc:
                raise ValueError('Invalid numeric legacy profile coefficient: ' + key) from exc
        remaining = min(4, count - len(coefficients))
        if len(values) != remaining or not all(math.isfinite(value) for value in values):
            raise ValueError('Incomplete or nonfinite legacy profile coefficient record: ' + key)
        coefficients.extend(values)
    unsupported = ([0, *range(7, 12)] if ptype == 1 else list(range(9, 21)))
    nonzero = [index + 1 for index in unsupported if coefficients[index] != 0.0]
    if nonzero:
        raise ValueError('Unsupported nonzero legacy profile coefficients ' + ', '.join(map(str, nonzero)) +
                         '; anisotropic, line-displacement and other unmapped terms cannot be discarded')
    if ptype == 1:
        mapping = dict(zip(('alpha', 'beta-0', 'beta-1', 'sig-0', 'sig-1', 'sig-2'), coefficients[1:7]))
        mapping.update(X=0.0, Y=0.0, Z=0.0)
    else:
        mapping = dict(zip(('alpha', 'beta-0', 'beta-1', 'sig-0', 'sig-1', 'sig-2', 'Z', 'X', 'Y'), coefficients[:9]))
    mapping.update({'beta-q': 0.0, 'sig-q': 0.0})
    return dict(coefficients=coefficients, coefficient_count=count, legacy_peak_cutoff=cutoff,
                coefficient_mapping=mapping, coefficient_definition='GSAS manual 2004, pp. 144, 148, 223')


def legacy_profile_metadata(path, bank):
    path=Path(path)
    if path.suffix.lower() not in {'.prm','.inst','.ins'} or type(bank) is not int or bank<1:
        raise ValueError('Choose a legacy GSAS profile and an explicit positive Bank label')
    with path.open('rb') as stream:raw=stream.read(1024*1024+1)
    if len(raw)>1024*1024:raise ValueError('Legacy profile exceeds 1 MiB')
    text=raw.decode('utf-8-sig');records={};title=''
    if not text.endswith('\n'):
        raise ValueError('Legacy profile is missing its final line terminator; supply a complete file')
    for line in text.splitlines():
        if line.startswith('ID '):title=line[2:].strip()
        if not line.startswith('INS') or '#' in line:continue
        key=line[:12]
        if key in records:raise ValueError('Duplicate legacy profile record; Bank mapping is ambiguous')
        records[key]=line[12:]
    if records.get('INS   HTYPE ','').strip()[:3]!='PNT':
        raise ValueError('Legacy battery profiles must use neutron TOF geometry')
    prefix=f'INS {bank:2d}'
    def tokens(key,minimum):
        values=records.get(key,'').split()
        if len(values)<minimum:raise ValueError('Missing legacy profile record: '+key)
        try:numbers=_numbers(' '.join(values[:minimum]))
        except ValueError as exc:raise ValueError('Invalid numeric legacy profile record: '+key) from exc
        if not all(math.isfinite(value) for value in numbers):raise ValueError('Nonfinite legacy profile parameter')
        return numbers
    tokens('INS   FPATH1',1);tokens(prefix+' ICONS',3);tokens(prefix+'BNKPAR',2)
    old_style=prefix+'PRCF  ' in records
    if old_style and prefix+'PRCF1 ' in records:
        raise ValueError('Ambiguous legacy profile has both old and numbered default records')
    ptype=tokens(prefix+('PRCF  ' if old_style else 'PRCF1 '),1)[0]
    if ptype not in {1.,3.}:
        raise ValueError('Legacy profile function is unsupported or table based; only isotropic functions 1 and 3 have a checked conversion')
    coefficients = _profile_coefficients(records, prefix, old_style, int(ptype))
    incident=prefix+'I ITYP'
    if incident in records and tokens(incident,1)[0]!=0:
        raise ValueError('Legacy incident-spectrum corrections cannot be applied silently to reduced observations')
    return dict(source_sha256=sha256(raw).hexdigest(),profile_bank=bank,profile_function=int(ptype),
                legacy_title=title,calibration_status='requires_scientific_validation', **coefficients)


def native_profile(path, bank, destination, G):
    """Return a checked GSAS-II profile and a conversion receipt, if needed."""
    from radar_pd_nova.nomad import profile_blocks
    path=Path(path)
    if path.suffix.lower()=='.instprm':
        blocks=profile_blocks(path)
        if bank not in blocks:raise ValueError('Selected profile Bank label is absent')
        return path,None
    metadata=legacy_profile_metadata(path,bank)
    reader=SimpleNamespace(Sample={},powderentry=[None,None,bank],instdict={})
    imported=G.load_iprms(str(path),reader,bank=bank)
    if not isinstance(imported,(tuple,list)) or len(imported)!=2 or imported[1]:
        raise ValueError('Legacy profile has unsupported incident-spectrum or table corrections')
    parameters=imported[0]
    if parameters.get('Type',[''])[0]!='PNT':raise ValueError('Native conversion changed neutron TOF geometry')
    # The installed GSAS-II legacy reader drops sig-0 and every Lorentzian
    # term in functions 1/3. Native roundtrip alone cannot detect that loss.
    # Use the independently documented coefficients and retain the difference.
    corrections = {}
    for key, value in metadata['coefficient_mapping'].items():
        old = parameters.get(key, [None, None])[1]
        if old != value:
            corrections[key] = dict(native_loader_value=old, source_value=value)
        parameters[key] = [value, value, False]
    # The native reader expects the Bank header and GSAS-II marker on one
    # line; two consecutive comment headers are not a valid native file.
    rows=[f'#Bank {bank}: GSAS-II instrument parameter file; calibration is unqualified', 'Type:PNT']
    numbers={}
    for key,values in parameters.items():
        if key=='Type':continue
        if key=='CorrectionCode' or not isinstance(values,(tuple,list)) or len(values)<2:
            raise ValueError('Native legacy profile contains unsupported parameters')
        value=bank if key=='Bank' else values[1]
        if not isinstance(value,Real) or isinstance(value,bool) or not math.isfinite(value):
            raise ValueError('Native legacy conversion produced a nonnumeric parameter')
        numbers[key]=float(value);rows.append(f'{key}:{float(value):.17g}')
    converted=Path(destination);converted.parent.mkdir(parents=True,exist_ok=True)
    if converted.exists():raise ValueError('Do not overwrite an earlier profile conversion')
    converted.write_text('\n'.join(rows)+'\n')
    if profile_blocks(converted)[bank][1]!=numbers:
        raise ValueError('Converted profile did not retain native coefficient values')
    check=G.load_iprms(str(converted),SimpleNamespace(Sample={},powderentry=[None,None,bank],instdict={}),bank=bank)
    if check[1] or any(float(check[0][key][1])!=value for key,value in numbers.items()):
        raise ValueError('Converted profile coefficients changed on native reload')
    if sha256(path.read_bytes()).hexdigest()!=metadata['source_sha256']:
        raise ValueError('Legacy profile changed during conversion')
    metadata.update(converted_sha256=sha256(converted.read_bytes()).hexdigest(),converted_parameters=numbers,
                    native_loader_differences=corrections, adapter=ADAPTER)
    return converted,metadata
