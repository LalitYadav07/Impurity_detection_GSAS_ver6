"""Plain-language display of the validated recipe's explicit edits."""


def explicit_edit_lines(stage, context, *, physical_banks=None):
    phases = {p['key']:p['name'] for p in context['phases']}
    banks = physical_banks or {}
    hists = {h['key']:h for h in context['histograms']}

    def histogram(key):
        h = hists.get(key, {})
        name = h.get('name', key)
        return f'Physical bank {banks[name]}' if name in banks else name

    def atom(term):
        return phases.get(term['phase'], term['phase'])+' / '+term['atom']

    def units(unit, key):
        if unit != 'native':
            return {'tof':'µs (TOF)', 'd':'Å (d-spacing)', 'Q':'Å⁻¹ (Q)', 'two_theta':'° (2θ)'}.get(unit,unit)
        kind = (hists.get(key, {}).get('instrument') or {}).get('Type')
        return 'µs (TOF)' if kind == 'PNT' else '° (2θ)' if kind in {'PXC','PNC'} else 'native coordinate'

    lines = []
    for key in stage.get('phase_normalization') or []:
        lines.append('Normalize phase scales to sum to 1 · '+histogram(key))
    for relation in stage.get('constraints') or []:
        if relation['kind'] == 'uiso_equal':
            equation = ' = '.join('Uiso('+atom(t)+')' for t in relation['atoms'])
        else:
            terms = [f"{t['coefficient']:g} × occupancy({atom(t)})" for t in relation['terms']]
            equation = ' + '.join(terms)+f" = {relation['total']:g}"
        lines.append('Constraint: '+equation+'. '+relation['reason'])
    for edit in stage.get('fit_ranges') or []:
        lines.append(f"Fit range · {histogram(edit['histogram'])}: {edit['lower']:g}–{edit['upper']:g} {units(edit['unit'],edit['histogram'])}")
    for edit in stage.get('exclusions') or []:
        if edit['mode'] == 'clear':
            lines.append('Clear excluded regions · '+histogram(edit['histogram']))
        else:
            regions = ', '.join(f'{a:g}–{b:g}' for a,b in edit['regions'])
            lines.append(f"{'Add' if edit['mode']=='add' else 'Replace'} excluded regions · {histogram(edit['histogram'])}: {regions} {units(edit['unit'],edit['histogram'])}")
    for edit in stage.get('texture') or []:
        where = phases.get(edit['phase'],edit['phase'])+' / '+histogram(edit['histogram'])
        if edit['model'] == 'march-dollase':
            direction = ' '.join(map(str,edit['direction']))
            setting = 'March–Dollase preferred orientation along ['+direction+']'
            if 'value' in edit:setting += f"; starting ratio {edit['value']:g}"
        else:
            setting = f"Spherical-harmonic preferred orientation, order {edit['order']}"
        lines.append(where+': '+setting)
    return lines
