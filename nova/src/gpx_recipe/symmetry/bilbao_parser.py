"""Bounded Bilbao interchange parser; MIT adaptation, see ../STUDIO-LICENSE.

Standalone RADAR-PD code. No reference-project runtime or unsafe eval is used.
"""
from __future__ import annotations
import hashlib,json,math,re
from fractions import Fraction
from typing import Any

def _parse_bilbao_numeric_literal(value: str) -> list[Any]:
    """Parse nested Bilbao brace arrays, including rational components."""
    source = str(value).strip()
    depth=0
    for char in source:
        depth += int(char=='{')-int(char=='}')
        if not 0<=depth<=5:raise ValueError('Numeric array nesting budget exceeded')
    if len(source)>200000:raise ValueError('Numeric array size budget exceeded')
    token_pattern = re.compile('\\{|\\}|,|[+\\-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:/[+\\-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+))?')
    tokens: list[str] = []
    cursor = 0
    for match in token_pattern.finditer(source):
        if source[cursor:match.start()].strip():
            raise ValueError('invalid token in magnetic subgroup numeric array')
        tokens.append(match.group(0))
        cursor = match.end()
    if source[cursor:].strip() or not tokens:
        raise ValueError('invalid magnetic subgroup numeric array')
    position = 0

    def parse_value() -> Any:
        nonlocal position
        if position >= len(tokens):
            raise ValueError('truncated magnetic subgroup numeric array')
        token = tokens[position]
        if token == '{':
            position += 1
            result = []
            need_value = True
            while position < len(tokens):
                if tokens[position] == '}':
                    if need_value and result:
                        raise ValueError('trailing comma in magnetic subgroup array')
                    position += 1
                    return result
                if not need_value or tokens[position] == ',':
                    raise ValueError('invalid separator in magnetic subgroup array')
                result.append(parse_value())
                need_value = False
                if position < len(tokens) and tokens[position] == ',':
                    position += 1
                    need_value = True
            raise ValueError('unterminated magnetic subgroup array')
        if token in {'}', ','}:
            raise ValueError('invalid magnetic subgroup numeric value')
        position += 1
        try:
            return float(Fraction(token)) if '/' in token else float(token)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError('invalid magnetic subgroup numeric value') from exc
    parsed = parse_value()
    if position != len(tokens) or not isinstance(parsed, list):
        raise ValueError('invalid magnetic subgroup numeric array')
    return parsed

def _bilbao_numeric_array(value: str, *, shape: tuple[int, ...]) -> list[Any]:
    """Decode the brace arrays emitted by Bilbao's GSAS-II endpoint safely."""
    parsed = _parse_bilbao_numeric_literal(value)

    def finite_number(item: Any) -> float:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError('magnetic subgroup arrays must contain only numbers')
        result = float(item)
        if not math.isfinite(result) or abs(result) > 10000:
            raise ValueError('magnetic subgroup array value is out of bounds')
        return result
    if shape == (3,):
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            raise ValueError('magnetic subgroup vector must have three components')
        return [finite_number(item) for item in parsed]
    if shape == (3, 3):
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
            raise ValueError('magnetic subgroup matrix must have three rows')
        matrix: list[list[float]] = []
        for row in parsed:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise ValueError('magnetic subgroup matrix must be 3x3')
            matrix.append([finite_number(item) for item in row])
        determinant = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]) + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        if abs(determinant) < 1e-10:
            raise ValueError('magnetic subgroup transformation matrix is singular')
        return matrix
    raise ValueError(f'unsupported Bilbao array shape {shape!r}')

def _bilbao_auxiliary_array(value: str) -> list[Any]:
    """Decode bounded auxiliary lineage data without assigning matrix meaning."""
    parsed = _parse_bilbao_numeric_literal(value)
    if not isinstance(parsed, (list, tuple)) or not parsed or len(parsed) > 64:
        raise ValueError('auxiliary magnetic subgroup array has invalid size')

    def checked(item: Any, *, depth: int=0) -> Any:
        if depth > 3:
            raise ValueError('auxiliary magnetic subgroup array is too deeply nested')
        if isinstance(item, (list, tuple)):
            if len(item) > 64:
                raise ValueError('auxiliary magnetic subgroup array has invalid size')
            return [checked(child, depth=depth + 1) for child in item]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError('auxiliary magnetic subgroup array must be numeric')
        value = float(item)
        if not math.isfinite(value) or abs(value) > 10000:
            raise ValueError('auxiliary magnetic subgroup value is out of bounds')
        return value
    return checked(parsed)

def _magnetic_candidate_id(candidate: dict[str, Any]) -> str:
    identity = {'gid': int(candidate['gid']), 'magnetic_space_group': str(candidate['magnetic_space_group']), 'bns_lattice': str(candidate['bns_lattice']), 'transformation_matrix': candidate['transformation_matrix'], 'origin_shift': candidate['origin_shift']}
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return 'mag:' + hashlib.sha256(encoded.encode('utf-8')).hexdigest()[:20]

def parse_magnetic_subgroups_gsas_output(output: Any) -> dict[str, Any]:
    """Parse k-SUBGROUPSMAG's documented GSAS-II interchange response.

    The regular endpoint is presentation HTML and is intentionally not accepted
    here. ``kSUBGROUPSMAG_GSAS`` emits delimiter records consumed by GSAS-II's
    own ``GetNonStdSubgroupsmag`` implementation. This parser mirrors that
    grammar while replacing upstream ``eval`` calls with bounded literal parsing.
    """
    if not isinstance(output,str) or len(output)>20*1024*1024 or not output.strip():
        raise ValueError('magnetic subgroup response is empty')
    normalized = output.replace('<font style= "text-decoration: overline;">', '<font>-')
    lines = normalized.replace('&', '\n').splitlines()
    start = next((index + 1 for (index, line) in enumerate(lines) if 'yesz' in line), None)
    if start is None:
        raise ValueError('response is not the machine-readable GSAS-II magnetic catalog')
    raw_candidates: list[dict[str, Any]] = []
    base_gids: list[int] = []
    alternative_groups: dict[int, list[int]] = {}
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        records = [item for item in line.split('GGG') if item.strip()]
        group_gids: list[int] = []
        parsed_group: list[dict[str, Any]] = []
        for record in records:
            items = record.split('z')
            if len(items) < 8:
                raise ValueError('truncated magnetic subgroup record')
            try:
                gid = int(items[0])
            except ValueError as exc:
                raise ValueError('invalid magnetic subgroup gid') from exc
            if gid <= 0 or gid in {row['gid'] for row in raw_candidates}:
                raise ValueError('magnetic subgroup gids must be positive and unique')
            symbol = items[4].replace('<i>', '').replace('</i>', '').replace('<sub>', '').replace('</sub>', '').strip()
            if not symbol or len(symbol) > 120 or (not re.fullmatch("[A-Za-z0-9_/'\\- ]+", symbol)):
                raise ValueError('invalid magnetic space-group symbol')
            bns_lattice = ''
            subscript_start = items[4].find('<sub>')
            if subscript_start == 8 and len(symbol) >= 2:
                bns_lattice = symbol[1]
                symbol = f'{symbol[0]}_{bns_lattice} {symbol[2:]}'
            candidate = {'gid': gid, 'bns_number':items[3] if re.fullmatch(r'[1-9][0-9]{0,2}\.[1-9][0-9]{0,3}',items[3]) else None, 'magnetic_space_group': symbol, 'bns_lattice': bns_lattice, 'transformation_matrix': _bilbao_numeric_array(items[5], shape=(3, 3)), 'origin_shift': _bilbao_numeric_array(items[6], shape=(3,)), 'supergroup_lineage': _bilbao_auxiliary_array(items[7])}
            parsed_group.append(candidate)
            group_gids.append(gid)
        if not parsed_group:
            continue
        base_gids.append(group_gids[0])
        alternative_groups[group_gids[0]] = group_gids
        raw_candidates.extend(parsed_group)
    if not raw_candidates:
        raise ValueError('machine-readable magnetic catalog contained no candidates')
    candidates = []
    base_gid_set = set(base_gids)
    for candidate in raw_candidates:
        gid = int(candidate['gid'])
        alternatives = next((group for group in alternative_groups.values() if gid in group), [gid])
        candidate = {**candidate, 'candidate_id': _magnetic_candidate_id(candidate), 'is_base_setting': gid in base_gid_set, 'alternative_gids': alternatives}
        candidates.append(candidate)
    return {'schema_version': 'bilbao-magnetic-subgroup-catalog-v1', 'status': 'parsed', 'catalog_kind': 'magnetic_subgroups', 'catalog_complete': True, 'candidate_count': len(candidates), 'base_candidate_count': len(base_gids), 'base_gids': base_gids, 'candidates': candidates}
