"""Inspect saved Mantid reduction provenance without executing archive scripts.

Generated Python records explicit algorithm arguments; the separate XML may
include configuration defaults or placeholders. Preserve both and report their
disagreements. Agreement does not prove the identity of cached corrections or
establish that a numeric product was generated from these exact bytes.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def _read(path):
    path = Path(path)
    before = path.stat()
    if not 0 < before.st_size <= 256 * 1024:
        raise ValueError('A nonempty provenance file at most 256 KiB is required')
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Reduction provenance changed while being read')
    return raw, {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                 'bytes': len(raw)}


def _literal(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, (list, tuple)):
        return [_literal(v) for v in value]
    if isinstance(value, dict) and all(type(k) is str for k in value):
        return {k: _literal(v) for k, v in value.items()}
    raise ValueError('Only finite JSON-compatible literal parameters are supported')


def inspect_recipe(path):
    raw, source = _read(path)
    text = raw.decode('utf-8')
    tree = ast.parse(text)
    if sum(1 for _ in ast.walk(tree)) > 20000:
        raise ValueError('Saved reduction script exceeds the syntax budget')
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, 'id', getattr(node.func, 'attr', ''))
        if name != 'SNSPowderReduction':
            continue
        parameters, unresolved = {}, []
        if node.args:
            unresolved.append('positional arguments')
        for kw in node.keywords:
            if kw.arg is None:
                unresolved.append('**kwargs')
                continue
            try:
                if kw.arg in parameters:
                    raise ValueError('Duplicate parameter')
                parameters[kw.arg] = _literal(ast.literal_eval(kw.value))
            except (ValueError, TypeError, SyntaxError, RecursionError):
                unresolved.append(kw.arg)
        calls.append({'line': node.lineno, 'parameters': parameters,
                      'unresolved': unresolved})
    version = re.search(r'^# Version ([0-9][^\r\n]*)$', text, re.MULTILINE)
    commit = re.search(r'^# SHA-1 ([a-fA-F0-9]{40})\s*$', text, re.MULTILINE)
    return {'source': source, 'calls': calls,
            'reported_mantid_version': version.group(1).strip() if version else None,
            'reported_mantid_git_sha': commit.group(1).lower() if commit else None,
            'archive_script_executed': False}


def inspect_xml(path):
    raw, source = _read(path)
    if b'<!doctype' in raw.lower() or b'<!entity' in raw.lower():
        raise ValueError('DTD/entity declarations are not accepted in reduction provenance')
    root = ET.fromstring(raw)
    if root.tag != 'Reduction':
        raise ValueError('Expected a POWGEN Reduction XML record')
    fields = {}

    def visit(node, depth=0):
        if depth > 8:
            raise ValueError('Reduction XML exceeds the nesting budget')
        if not isinstance(node.tag, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', node.tag):
            raise ValueError('Unsupported reduction XML field')
        if len(node):
            for child in node:
                visit(child, depth + 1)
        else:
            name = node.tag.lower()
            if name in fields:
                raise ValueError('Ambiguous duplicate reduction XML field: ' + name)
            fields[name] = node.text.strip() if node.text and node.text.strip() else None
            if len(fields) > 256:
                raise ValueError('Reduction XML exceeds the field budget')

    visit(root)
    return {'source': source, 'fields': fields}


def _equivalent(value, text):
    if isinstance(value, (int, float, bool)):
        try:
            numeric = float(text)
            return math.isfinite(numeric) and math.isclose(float(value), numeric, rel_tol=1e-12, abs_tol=1e-12)
        except (ValueError, TypeError, OverflowError):
            return False
    if value is None:
        return text == 'None'
    if isinstance(value, str):
        return value.strip() == text
    try:
        return value == _literal(ast.literal_eval(text))
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return False


def compare_reduction_records(recipe_path, xml_path):
    recipe, xml = inspect_recipe(recipe_path), inspect_xml(xml_path)
    if len(recipe['calls']) != 1 or recipe['calls'][0]['unresolved']:
        raise ValueError('Select one complete literal SNSPowderReduction call; unresolved scripts cannot be replayed')
    fields, parameters = xml['fields'], recipe['calls'][0]['parameters']
    conflicts, agreements, unreported = [], [], []
    aliases = {'GroupingFileName': 'groupingfile', 'GroupingFile': 'groupingfile'}
    for name, value in parameters.items():
        xml_value = fields.get(aliases.get(name, name.lower()))
        if xml_value is None:
            unreported.append(name)
        elif _equivalent(value, xml_value):
            agreements.append(name)
        else:
            conflicts.append({'field': name, 'generated_recipe': value, 'xml': xml_value})
    version = recipe['reported_mantid_version']
    xml_version = fields.get('mantid_version')
    if version and xml_version and version != xml_version:
        conflicts.append({'field': 'mantid_version', 'generated_recipe': version, 'xml': xml_version})
    match = re.search(r'/PG3_(\d+)\.nxs\.h5$', str(parameters.get('Filename', '')))
    if match and fields.get('runnumber') and match.group(1) != fields['runnumber']:
        conflicts.append({'field': 'runnumber', 'generated_recipe': match.group(1), 'xml': fields['runnumber']})
    dependencies = []
    for key in ('Filename', 'CalibrationFile', 'CharacterizationRunsFile', 'GroupingFileName', 'GroupingFile'):
        value = parameters.get(key)
        if isinstance(value, str):
            dependencies.extend({'role': key, 'source': p.strip()} for p in value.split(',') if p.strip())
    return {'schema': 'powgen-reduction-record-comparison/v1', 'recipe': recipe, 'xml': xml,
            'status': 'conflicting_records' if conflicts else 'consistent_reported_fields',
            'conflicts': conflicts, 'matching_parameters': agreements,
            'recipe_parameters_not_reported_in_xml': unreported,
            'declared_dependencies': dependencies,
            'runtime_binary_identity_verified': False, 'cached_corrections_identity_verified': False,
            'numeric_product_link_verified': False, 'experimental_accuracy_certified': False,
            'scope': 'Static provenance comparison only; no archive code execution or scientific qualification'}
