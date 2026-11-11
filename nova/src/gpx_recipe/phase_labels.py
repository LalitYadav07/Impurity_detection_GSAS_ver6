"""Display saved symmetry without changing native phase identities."""
import re


def phase_label(phase):
    name = str(phase.get('name') or phase.get('key') or 'Phase')
    group = phase.get('space_group')
    if not isinstance(group, str) or not group.strip():
        return name
    # Constructed children can retain the parent's old name in the GPX.
    # Remove only a trailing explicit SG annotation, never chemical labels.
    name = re.sub(r'\s*\(SG\s+[^()]+\)\s*$', '', name)
    return name + ' · ' + group.strip()
