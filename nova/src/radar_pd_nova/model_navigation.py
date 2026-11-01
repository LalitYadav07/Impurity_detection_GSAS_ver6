"""Readable acquisition labels without changing scientific source identities."""
import math
import re


def scan_label(metadata, parent):
    scan = str(metadata.get('scan') or parent)
    parts = [scan]
    condition = metadata.get('condition')
    if (isinstance(condition, (int, float)) and not isinstance(condition, bool)
            and math.isfinite(condition)):
        parts.append(f"{condition:g} {metadata.get('condition_unit') or ''}".strip())
    bank = metadata.get('bank')
    if bank is not None and str(bank) and not re.search(
            r'\bbank\s*' + re.escape(str(bank)) + r'\b', scan, re.IGNORECASE):
        parts.append('bank ' + str(bank))
    return ' · '.join(parts)


def scan_options(rows):
    options = {}
    for row in rows:
        if row['source'] not in options:
            options[row['source']] = scan_label(row['metadata'], row['parent'])
    return options
