"""Presentation metadata read from the prepared native GSAS-II model.

The mappings here describe native constraints; they do not calculate scattering
or replace the magnetic/structural construction services.
"""
from __future__ import annotations

import copy
import importlib
import numpy as np


def magnetic_sites_with_defaults(G, source, sites):
    """Preserve a project g value, otherwise ask GSAS-II for its own default.

    GetMagFormFacCoeff selects ion-specific coefficients, not a Landé g value.
    Explicit values remain supported for existing scientific API callers.
    """
    elem = importlib.import_module((G.__package__ + '.' if G.__package__ else '') + 'GSASIIElem')
    project = G.G2Project(str(source))
    phase = next(p.data for p in project.phases() if p.data['General']['Type'] == 'nuclear')
    general = phase['General']; cx, ct, _, _ = general['AtomPtrs']
    by_label = {a[ct-1]: a for a in phase['Atoms']}
    inherited = dict(zip(general.get('AtomTypes', []), general.get('Lande g', [])))
    # A minimal magnetic copy lets SetupGeneral initialize native defaults.
    probe = copy.deepcopy(phase)
    probe['General']['Type'] = 'magnetic'; probe['General'].pop('Lande g', None)
    elem.SetupGeneral(probe, '')
    defaults = dict(zip(probe['General']['AtomTypes'], probe['General']['Lande g']))
    resolved = []
    for site in sites:
        atom_type = by_label[site['label']][ct]
        if 'lande_g' in site:
            value, origin = site['lande_g'], 'explicit scientific request'
        elif atom_type in inherited:
            value, origin = inherited[atom_type], 'parent GSAS-II project'
        else:
            value, origin = defaults[atom_type], 'GSAS-II SetupGeneral default'
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= 3:
            raise ValueError('The native magnetic g value must be finite and in (0, 3]')
        resolved.append({**site, 'lande_g': float(value), 'lande_g_source': origin})
    return resolved


def magnetic_controls(native, template):
    """Extract the exact affine independent-coordinate → site-moment map."""
    from gpx_recipe.symmetry.adapters.gsas_magnetic_preflight import direct_unit_metric
    native._check()
    names = list(native.vary)
    origins = {s['label']: s for s in template['sites']}
    rows = []
    for phase in native.histo_phases[1].values():
        general = phase['General']
        if general['Type'] != 'magnetic': continue
        cx, ct, cs, _ = general['AtomPtrs']
        for index, atom in enumerate(phase['Atoms']):
            rows.append(dict(label=str(atom[ct-1]), parent_label=origins[atom[ct-1]]['parent_label'],
                ion=str(atom[ct]), position=list(map(float, atom[cx:cx+3])),
                site_symmetry=str(atom[cs]), multiplicity=int(atom[cs+1]),
                metric=direct_unit_metric(general['Cell'][1:7]).tolist(),
                variables=[f"{phase['pId']}::AM{axis}:{index}" for axis in 'xyz']))

    def mapped(values):
        parameters = copy.deepcopy(native.parms)
        parameters.update(zip(names, values))
        native.maps.Dict2Map(parameters)
        return np.asarray([[parameters[v] for v in row['variables']] for row in rows], float)

    zero = mapped(np.zeros(len(names)))
    basis = np.stack([mapped(v)-zero for v in np.eye(len(names))], axis=-1)
    for values in (native.x0, np.random.default_rng(194).normal(size=len(names))):
        if not np.allclose(mapped(values), zero + basis @ values, atol=1e-10, rtol=1e-10):
            raise ValueError('Native magnetic control mapping failed verification')
    parameters = []
    for j, name in enumerate(names):
        affects = [f"{row['label']} · M{axis}" for i, row in enumerate(rows)
                   for k, axis in enumerate('xyz') if abs(basis[i, k, j]) > 1e-12]
        direct = next((f"{row['label']} · M{'xyz'[k]}" for row in rows
                       for k, variable in enumerate(row['variables']) if variable == name), None)
        parameters.append(dict(name=name, symbol=f'q{j+1}',
            label=(direct or f'Linked moment q{j+1}')+' (μB)',
            description='Controls '+', '.join(affects), affects=affects))
    for i, row in enumerate(rows):
        row.update(offset=zero[i].tolist(), basis=basis[i].tolist())
    return dict(parameters=parameters, sites=rows,
        convention='Mx, My and Mz are in μB along the child cell’s unit a, b and c axes. Only independent GSAS-II variables are editable; linked and forbidden components follow the subgroup symmetry.')


def structural_controls(G, path):
    """Use the shared, verified native displacement inventory for the form."""
    from gpx_recipe.structural_live.native import inventory, apply_modes, modules
    project = G.G2Project(str(path))
    phases = [p.data for p in project.phases() if p.data.get('ISODISTORT')]
    if len(phases) != 1: raise ValueError('Expected one prepared structural mode phase')
    phase = phases[0]
    info, arrays = inventory(phase)
    _, math_module, *_ = modules()
    apply_modes(phase, np.zeros(len(info['native_mode_names'])), math_module)
    cx, ct, cs, _ = phase['General']['AtomPtrs']
    sites = [dict(label=str(a[ct-1]), ion=str(a[ct]), site_symmetry=str(a[cs]),
                  multiplicity=int(a[cs+1]), offset=list(map(float, a[cx:cx+3])),
                  basis=arrays['coordinate_basis'][i].tolist())
             for i, a in enumerate(phase['Atoms'])]
    parameters = []
    for j, name in enumerate(info['native_mode_names']):
        affects = [f"{row['label']} · {axis}" for i, row in enumerate(sites)
                   for k, axis in enumerate('xyz') if abs(arrays['coordinate_basis'][i,k,j]) > 1e-12]
        parameters.append(dict(name=name, symbol=f'Q{j+1}', label=f'Displacement Q{j+1} (Å)',
            description='Moves '+', '.join(affects) if affects else 'No atomic displacement; inspect the child lattice.',
            affects=affects, mode_label=info['native_mode_labels'][j],
            normalization=info['normalization'][j]))
    return dict(parameters=parameters, sites=sites, cell=list(map(float,phase['General']['Cell'][1:7])),
                space_group=phase['General']['SGData']['SpGrp'],
                convention='Signed amplitudes use GSAS-II’s native ISODISTORT normalization, in Å. The site table shows the resulting fractional coordinates. Zero amplitudes recover the parent positions in the child cell.',
                physical_rank=info['physical_rank'], gauge_rank=info['gauge_rank'])


def control_view(controls, values, kind):
    """Format live readouts from the native map; no scientific recalculation."""
    parameters = [{**p, 'value': values[p['name']]} for p in controls['parameters']]
    vector = np.asarray([p['value'] for p in parameters], float)
    sites = []
    for row in controls['sites']:
        basis, offset = np.asarray(row['basis']), np.asarray(row['offset'])
        result = offset + basis @ vector
        components = []
        for i, axis in enumerate('xyz'):
            terms = []
            if abs(offset[i]) > 1e-12: terms.append(f'{offset[i]:.5g}')
            for coefficient, parameter in zip(basis[i], parameters):
                if abs(coefficient) > 1e-12:
                    terms.append(f'{coefficient:+.5g} {parameter["symbol"]}')
            equation = ' '.join(terms).lstrip('+') or '0'
            components.append(dict(axis=('M' if kind=='magnetic' else '')+axis,
                value=f'{result[i]:.5g}', equation=equation,
                status='linked' if np.any(abs(basis[i])>1e-12) else 'fixed'))
        magnitude = (float(np.sqrt(max(0., result @ np.asarray(row['metric']) @ result))) if kind=='magnetic' else None)
        sites.append({**row, 'components':components,
            'position_text':', '.join(f'{v:.5g}' for v in row.get('position',offset)),
            'magnitude':f'{magnitude:.5g}' if magnitude is not None else ''})
    return parameters, sites
