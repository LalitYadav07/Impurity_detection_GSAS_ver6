"""Literal reduction declarations affecting individual-exposure intensities.

Never import a facility script. Missing declarations remain unknown; reduction
defaults are not inferred from the executing application's software version.
"""
import ast
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

FIELDS = {'PushDataPositive', 'ScaleData', 'OffsetData', 'TypeOfCorrection',
          'SampleFormula', 'MeasuredMassDensity', 'BackgroundNumber', 'VanadiumNumber',
          'RemovePromptPulseWidth', 'Binning'}


def literal_effects(raw, suffix, run):
    if len(raw) > 1024 * 1024:
        raise ValueError('Reduction evidence exceeds its metadata limit')
    if suffix == '.py':
        records = []
        for call in ast.walk(ast.parse(raw.decode('utf-8'))):
            if not isinstance(call, ast.Call):
                continue
            name = call.func.id if isinstance(call.func, ast.Name) else call.func.attr if isinstance(call.func, ast.Attribute) else ''
            if name != 'SNSPowderReduction':
                continue
            if any(kw.arg is None for kw in call.keywords) or len({kw.arg for kw in call.keywords}) != len(call.keywords):
                raise ValueError('Reduction keyword declarations are ambiguous or dynamic')
            args = {kw.arg: kw.value for kw in call.keywords}
            if 'SaveAs' in args:
                save = args['SaveAs']
                if not isinstance(save, ast.Constant) or not isinstance(save.value, str):
                    raise ValueError('GSAS-producing reduction is not a literal declaration')
                if 'gsas' not in save.value.lower():
                    continue
            if call.args:
                raise ValueError('Positional reduction arguments cannot establish an individual source run')
            for key in ('Filename', 'RunNumber', 'RunNumbers'):
                if key not in args:
                    continue
                value = args[key]
                if not isinstance(value, ast.Constant) or type(value.value) not in {str, int}:
                    raise ValueError('Individual POWGEN source run is not a literal declaration')
                text = str(value.value).strip()
                match = re.fullmatch(r'PG3_(\d+)(?:_event)?\.nxs(?:\.h5)?', Path(text).name, re.I)
                if any(mark in text for mark in (',', ';', '\n')) or (text != str(run) and (not match or int(match[1]) != run)):
                    raise ValueError('Reduction is summed or names a different source run')
            fields = {}
            for key in FIELDS & args.keys():
                value = args[key]
                # unary minus is a literal too; no expression evaluation.
                try:
                    value = ast.literal_eval(value)
                except (ValueError, TypeError):
                    fields[key] = 'unresolved expression'
                    continue
                if value is not None and type(value) not in {str, int, float, bool}:
                    fields[key] = 'unresolved structured value'
                elif isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('Nonfinite reduction intensity declaration')
                else:
                    fields[key] = value
            records.append(fields)
        if len(records) > 1 and any(record != records[0] for record in records[1:]):
            raise ValueError('Multiple GSAS-producing reductions disagree on intensity treatment')
        return records[0] if records else {}
    if suffix != '.xml' or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise ValueError('Unsupported reduction evidence format')
    fields, tags = {}, {key.lower(): key for key in FIELDS}
    tags.update({'sum': 'sum', 'runnumber': 'runnumber'})
    for element in ET.fromstring(raw).iter():
        key = tags.get(element.tag.rsplit('}', 1)[-1].lower())
        if key is None or len(element):
            continue
        if key in fields:
            raise ValueError('Duplicate intensity declarations in reduction XML')
        fields[key] = (element.text or '').strip()
    if fields.pop('sum', '0').lower() not in {'0', 'false', ''} or fields.pop('runnumber', str(run)) != str(run):
        raise ValueError('Reduction XML is summed or names a different source run')
    return {key: value for key, value in fields.items() if value != ''}


def normalized(value):
    if value is None or str(value).strip().lower() == 'none':
        return 'none'
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (ValueError, TypeError):
        pass
    return str(value).strip().lower()


def merge_effects(records):
    merged = {}
    for fields in records:
        for key, value in fields.items():
            if key in merged and normalized(merged[key]) != normalized(value):
                raise ValueError('Saved reduction receipts disagree on intensity treatment: ' + key)
            merged[key] = value
    return merged
