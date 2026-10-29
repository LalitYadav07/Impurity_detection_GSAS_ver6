"""Audited, deterministic moment-only kernel; adapted from MIT-licensed Studio.

The reference project is a development knowledge source only. This module has
no reference-project imports, services, paths, or dynamically generated code.
See ../STUDIO-LICENSE and symmetry/provenance.json.
"""

def run(source_gpx,ctx,protocol,candidate_record,seed):
    from pathlib import Path
    import copy,hashlib,math,re
    import numpy as np
    from GSASII import GSASIIscriptable as G2sc,GSASIIElem as G2elem,GSASIIlattice as G2lat,GSASIIobj as G2obj,GSASIIspc as G2spc
    from gsas_magnetic_preflight import transform_native_moment,magnetic_magnitude,moment_basis,direction_in_basis
    from .worker import TrialDisposition
    from .magnetic_identity import authoritative_operations,verify_native_operators,GreyGroup,UnverifiedMagneticSetting
    try:
        authority=authoritative_operations(candidate_record)
        definition=G2lat.makeBilbaoPhase([candidate_record['magnetic_space_group'],candidate_record.get('bns_lattice','')],candidate_record['origin_shift'],candidate_record['transformation_matrix'],True)
        ctx.save_json('magnetic-operator-preflight.json',verify_native_operators(candidate_record,definition['SGData'],authority=authority))
    except GreyGroup as exc:raise TrialDisposition('site_incompatible',str(exc))
    except UnverifiedMagneticSetting as exc:raise TrialDisposition('unsupported',str(exc))
    output_gpx=ctx.run_dir/'magnetic-refinement.gpx'
    gpx = G2sc.G2Project(gpxfile=str(source_gpx), newgpx=str(output_gpx))
    from .native_preparation import normalize_flags
    ctx.save_json('legacy-flag-normalization.json',normalize_flags(gpx))
    candidate = {**candidate_record, 'candidate_id': candidate_record['id']}
    requested_phase = protocol['parent_phase']
    requested_histograms = [protocol['histogram']]
    magnetic_elements = {re.match('^([A-Z][a-z]?)', s['form_factor'])[1] for s in protocol['sites']}
    initial_moment = np.array([seed['magnitude'] if x == seed['axis'] else 0.0 for x in 'xyz'])
    initial_moment_direction = seed['parent_direction']
    if initial_moment_direction is not None:
        initial_moment_direction = seed['parent_direction']
    refine_components = set('xyz') if seed['parent_direction'] else {seed['axis']}
    fixed_moment_direction = seed['parent_direction']
    if fixed_moment_direction is not None:
        fixed_moment_direction = seed['parent_direction']
    moment_site_coupling = 'independent'
    cycles = protocol['policy']['cycles']
    minimum_relative_chi2_change = float(1e-08)
    maximum_moment_magnitude = protocol['policy']['maximum_moment']
    phases = {phase.name: phase for phase in gpx.phases()}
    existing_magnetic_phases = [phase.name for phase in phases.values() if str((phase.data.get('General') or {}).get('Type') or '').lower() == 'magnetic']
    if existing_magnetic_phases:
        raise ValueError(f'source_gpx must be a nuclear-only ordered-pattern checkpoint before magnetic-phase construction; existing magnetic phase(s)={sorted(existing_magnetic_phases)}')
    if requested_phase:
        if requested_phase not in phases:
            raise ValueError(f'Nuclear phase {requested_phase!r} is absent; available={sorted(phases)}')
        nuclear = phases[requested_phase]
    else:
        nuclear_phases = [phase for phase in phases.values() if str((phase.data.get('General') or {}).get('Type') or '') == 'nuclear']
        if len(nuclear_phases) != 1:
            raise ValueError('nuclear_phase_name is required unless exactly one nuclear phase exists')
        nuclear = nuclear_phases[0]
    available_histograms = {hist.name: hist for hist in gpx.histograms()}
    if requested_histograms:
        missing = sorted(set(requested_histograms) - set(available_histograms))
        if missing:
            raise ValueError(f'Requested histogram(s) are absent: {missing}; available={sorted(available_histograms)}')
        selected_histograms = list(requested_histograms)
    else:
        selected_histograms = [name for name in nuclear.data.get('Histograms', {}) if name in available_histograms]
    if not selected_histograms:
        raise ValueError('The nuclear phase is not linked to a selected powder histogram')
    from .chemistry import validate_sites
    _cx,_ct,_,_=nuclear.data['General']['AtomPtrs']
    try:validate_sites([dict(label=a[_ct-1],type=a[_ct],occupancy=a[_cx+3]) for a in nuclear.data['Atoms']],protocol['sites'])
    except ValueError as exc:raise TrialDisposition('unsupported',str(exc))
    for name in selected_histograms:
        if name not in nuclear.data.get('Histograms', {}):
            raise ValueError(f'Nuclear phase {nuclear.name!r} is not linked to {name!r}')
        if not str(available_histograms[name].InstrumentParameters['Type'][0]).startswith('PN'):
            raise TrialDisposition('unsupported','This magnetic executor supports neutron powder histograms only')
    gpx.set_Controls('sequential', [])
    gpx.set_Controls('seqCopy', False)
    gpx.set_Controls('Reverse Seq', False)
    for phase in gpx.phases():
        for name, hap in (phase.data.get('Histograms') or {}).items():
            if isinstance(hap, dict):
                hap['Use'] = name in selected_histograms
    baseline_residuals = {name: dict(available_histograms[name].residuals) for name in selected_histograms}

    def powder_profile_arrays(histogram):
        """Return the six native GSAS-II powder-profile arrays."""
        raw_data = histogram.data.get('data')
        arrays = raw_data[1] if isinstance(raw_data, (list, tuple)) and raw_data and isinstance(raw_data[0], dict) else raw_data
        if isinstance(arrays, (list, tuple)) and arrays and isinstance(arrays[0], dict):
            arrays = arrays[1:]
        if isinstance(arrays, np.ndarray) or np.ma.isMaskedArray(arrays):
            matrix = np.asarray(np.ma.filled(arrays, np.nan), dtype=float)
            if matrix.ndim == 2 and matrix.shape[0] >= 6:
                return tuple((matrix[index] for index in range(6)))
        if not isinstance(arrays, (list, tuple)) or len(arrays) < 6:
            raise ValueError(f'Histogram {histogram.name!r} does not expose a powder profile')
        return tuple((np.asarray(value, dtype=float) for value in arrays[:6]))

    def find_positive_residual_targets(histogram, maximum_targets=12):
        """Locate strong underfit peak windows in the nuclear-only baseline.

        These are evidence targets, not a peak-search replacement. The portfolio
        uses the same deterministic windows for every magnetic subgroup so that a
        small global-Rwp change cannot hide failure at the reflections that
        motivated magnetic modelling.
        """
        x, yobs, weights, ycalc, background, _residual = powder_profile_arrays(histogram)
        finite = np.isfinite(x) & np.isfinite(yobs) & np.isfinite(weights) & np.isfinite(ycalc) & np.isfinite(background)
        configured_exclusions = []
        try:
            configured_exclusions = [[float(region[0]), float(region[1])] for region in histogram.Excluded() if isinstance(region, (list, tuple)) and len(region) >= 2]
        except Exception:
            configured_exclusions = []
        for lower, upper in configured_exclusions:
            if np.isfinite(lower) and np.isfinite(upper):
                lo, hi = sorted((lower, upper))
                finite &= ~((x >= lo) & (x <= hi))
        positive_residual = np.where(finite, np.maximum(yobs - ycalc, 0.0), 0.0)
        significance = positive_residual * np.sqrt(np.maximum(weights, 0.0))
        if significance.size >= 7:
            significance = np.convolve(significance, np.ones(7, dtype=float) / 7.0, mode='same')
        candidates = []
        for index in range(20, max(20, len(x) - 20)):
            if significance[index] < 5.0:
                continue
            if significance[index] < significance[index - 1]:
                continue
            if significance[index] < significance[index + 1]:
                continue
            if yobs[index] <= background[index]:
                continue
            candidates.append((float(significance[index]), int(index)))
        selected = []
        for score, index in sorted(candidates, reverse=True):
            if any((abs(index - kept_index) < 24 for _kept_score, kept_index in selected)):
                continue
            selected.append((score, index))
            if len(selected) >= int(maximum_targets):
                break
        return {'arrays': (x, yobs, weights, ycalc, background), 'excluded_regions': configured_exclusions, 'targets': [{'index': int(index), 'x': float(x[index]), 'baseline_significance': float(score)} for score, index in sorted(selected, key=lambda value: value[1])]}
    from GSASII import GSASIIstrMain as _G2strMain
    gpx.save()
    baseline_refresh = []
    for _name in selected_histograms:
        _hist = available_histograms[_name]
        _old = np.asarray(_hist.data['data'][1][3], dtype=float).copy()
        _fresh = np.asarray(_G2strMain.DoNoFit(str(output_gpx), _name), dtype=float)
        if _fresh.shape != _old.shape or not np.all(np.isfinite(_fresh)):
            raise ValueError('Nuclear control could not be recalculated')
        _hist.data['data'][1][3] = _fresh
        _hist.data['data'][1][5] = np.asarray(_hist.data['data'][1][1]) - _fresh
        baseline_refresh.append(dict(histogram=_name, maximum_stale_curve_difference=float(np.max(np.abs(_old - _fresh)))))
    ctx.save_json('baseline-refresh.json', baseline_refresh)
    baseline_target_profiles = {name: find_positive_residual_targets(available_histograms[name]) for name in selected_histograms}
    from gpx_recipe.execution import freeze
    freeze(gpx)
    for histogram in gpx.histograms():
        background = histogram.data.get('Background') or []
        if background and isinstance(background[0], list) and (len(background[0]) > 1):
            background[0][1] = False
        if len(background) > 1 and isinstance(background[1], dict):
            for peak in background[1].get('peaksList') or []:
                if isinstance(peak, list) and len(peak) >= 8:
                    peak[4:8] = [False, False, False, False]
        instrument = histogram.data.get('Instrument Parameters') or []
        if instrument and isinstance(instrument[0], dict):
            for parameter in instrument[0].values():
                if isinstance(parameter, list) and len(parameter) > 2:
                    parameter[2] = False
        sample = histogram.data.get('Sample Parameters') or {}
        for parameter in sample.values():
            if isinstance(parameter, list) and len(parameter) > 1 and isinstance(parameter[1], bool):
                parameter[1] = False
    for phase in gpx.phases():
        phase.data['General']['Cell'][0] = False
        phase_cx, phase_ct, _phase_cs, _phase_cia = phase.data['General']['AtomPtrs']
        for atom in phase.data.get('Atoms') or []:
            atom[phase_ct + 1] = ' '
        for hap in (phase.data.get('Histograms') or {}).values():
            if isinstance(hap.get('Scale'), list) and len(hap['Scale']) > 1:
                hap['Scale'][1] = False
            if isinstance(hap.get('Pref.Ori.'), list) and len(hap['Pref.Ori.']) > 2:
                hap['Pref.Ori.'][2] = False
            for key in ('Size', 'Mustrain'):
                values = hap.get(key)
                if isinstance(values, list):
                    if len(values) > 2 and isinstance(values[2], list):
                        values[2] = [False for _value in values[2]]
                    if len(values) > 5 and isinstance(values[5], list):
                        values[5] = [False for _value in values[5]]
            hstrain = hap.get('HStrain')
            if isinstance(hstrain, list) and len(hstrain) > 1 and isinstance(hstrain[1], list):
                hstrain[1] = [False for _value in hstrain[1]]
            babinet = hap.get('Babinet') or {}
            for parameter in babinet.values():
                if isinstance(parameter, list) and len(parameter) > 1:
                    parameter[1] = False
            extinction = hap.get('Extinction')
            if isinstance(extinction, list) and len(extinction) > 1 and isinstance(extinction[1], bool):
                extinction[1] = False

    from gsas_magnetic_preflight import preflight_magnetic_sites,NativeRepresentationError
    wanted_direction=np.eye(3)['xyz'.index(seed['axis'])] if seed['parent_direction'] is None else transform_native_moment(seed['parent_direction'],nuclear.data['General']['Cell'][1:7],candidate['transformation_matrix'])
    from .magnetic_identity import require_collinear_line
    _cell=G2lat.TransformCell(nuclear.data['General']['Cell'][1:7],candidate['transformation_matrix'])
    try:require_collinear_line([m for m,_ in G2spc.AllOps(definition['SGData'])[2]],G2lat.cell2AB(_cell[:6])[0],_cell[:3],wanted_direction)
    except UnverifiedMagneticSetting as exc:raise TrialDisposition('unsupported',str(exc))
    try:_,site_report=preflight_magnetic_sites(nuclear.data,candidate,magnetic_sites=[s['label'] for s in protocol['sites']],site_policy=protocol['policy']['site_policy'],expected_parent_setting=protocol['parent_setting_sha256'],maximum_cell_multiplier=protocol['policy']['maximum_cell_multiplier'],fixed_direction_child=wanted_direction)
    except NativeRepresentationError as exc:raise TrialDisposition('unsupported',str(exc))
    ctx.save_json('preflight.json',site_report)
    if not site_report['compatible']:
        raise TrialDisposition('site_incompatible','Selected magnetic sites are incompatible with this subgroup under the declared all/any policy')

    old_phase = nuclear.data
    if str((old_phase.get('General') or {}).get('Type') or '') != 'nuclear':
        raise ValueError('The magnetic parent must be a nuclear GSAS-II phase')
    trans = np.asarray(candidate['transformation_matrix'], dtype=float)
    uvec = np.asarray(candidate['origin_shift'], dtype=float)
    if trans.shape != (3, 3) or uvec.shape != (3,):
        raise ValueError('Registered magnetic candidate has invalid transformation dimensions')
    if not np.all(np.isfinite(trans)) or not np.all(np.isfinite(uvec)):
        raise ValueError('Registered magnetic transformation is not finite')
    det_trans = float(abs(np.linalg.det(trans)))
    if det_trans < 1e-10:
        raise ValueError('Registered magnetic transformation is singular')
    vvec = np.zeros(3, dtype=float)
    magnetic_choice = G2lat.makeBilbaoPhase([candidate['magnetic_space_group'], candidate.get('bns_lattice', '')], uvec, trans, True)
    new_phase = copy.deepcopy(old_phase)
    new_phase.pop('magPhases', None)
    new_phase.pop('SUBGROUPS', None)
    new_phase['ranId'] = 1+int(hashlib.sha256((protocol['source']['sha256']+candidate_record['id']).encode()).hexdigest()[:15],16)
    general = new_phase['General']
    magnetic_name = f"{nuclear.name} magnetic {candidate['candidate_id'][-8:]}"
    if magnetic_name in phases:
        raise ValueError(f'Magnetic phase {magnetic_name!r} already exists')
    general['Name'] = magnetic_name
    general['SGData'] = copy.deepcopy(magnetic_choice['SGData'])
    general['Cell'][1:] = list(G2lat.TransformCell(old_phase['General']['Cell'][1:7], trans))
    cutoff=protocol['policy'].get('magnetic_d_min',1.0)  # legacy sealed protocols
    if cutoff==0:
        minima=[]
        for name in selected_histograms:
            hist=available_histograms[name];x=np.asarray(hist.data['data'][1][0],dtype=float)
            ds=np.asarray(G2lat.Pos2dsp(hist.InstrumentParameters,x));lo,hi=hist.data['Limits'][1]
            selected=(x>=lo)&(x<=hi)&np.isfinite(ds)&(ds>0)
            if selected.any():minima.append(float(ds[selected].min()))
        if not minima:raise TrialDisposition('unsupported','The calibrated fit domain contains no positive physical d spacings')
        cutoff=min(minima)*.95  # include tails just outside the measured interval
    general['MagDmin'] = cutoff
    ctx.save_json('magnetic-reflection-range.json',dict(declared_d_min=protocol['policy'].get('magnetic_d_min','legacy 1.0 A'),actual_d_min=cutoff,full_fitted_range=protocol['policy'].get('magnetic_d_min')==0))
    new_phase['MagXform'] = (trans, uvec, vvec)
    new_phase, atom_codes = G2lat.TransformPhase(old_phase, new_phase, trans, uvec, vvec, True)
    parent_seed_moment_direction = initial_moment_direction if initial_moment_direction is not None else fixed_moment_direction
    candidate_seed_moment_direction = None
    candidate_fixed_moment_direction = None
    candidate_refine_components = set(refine_components)
    if parent_seed_moment_direction is not None:
        old_amat, old_bmat = G2lat.cell2AB(old_phase['General']['Cell'][1:7])
        new_amat, new_bmat = G2lat.cell2AB(new_phase['General']['Cell'][1:7])

        def transform_parent_moment_direction(parent_direction):
            return transform_native_moment(parent_direction, old_phase['General']['Cell'][1:7], trans)
        candidate_seed_moment_direction = transform_parent_moment_direction(parent_seed_moment_direction)
        direction_norm = magnetic_magnitude(candidate_seed_moment_direction, new_phase['General']['Cell'][1:7])
        if not math.isfinite(direction_norm) or direction_norm < 1e-10:
            raise ValueError('parent moment direction has an invalid child transform')
        candidate_seed_moment_direction /= direction_norm
        if fixed_moment_direction is not None:
            candidate_fixed_moment_direction = candidate_seed_moment_direction.copy()
            candidate_refine_components = {axis for axis, value in zip(('x', 'y', 'z'), candidate_seed_moment_direction) if abs(float(value)) >= 1e-08}
        else:
            candidate_refine_components = set()
            for requested_axis in refine_components:
                parent_basis = np.zeros(3, dtype=float)
                parent_basis[{'x': 0, 'y': 1, 'z': 2}[requested_axis]] = 1.0
                candidate_basis = transform_parent_moment_direction(parent_basis)
                candidate_refine_components.update({axis for axis, value in zip(('x', 'y', 'z'), candidate_basis) if abs(float(value)) >= 1e-08})
    cx, ct, cs, cia = new_phase['General']['AtomPtrs']
    kept_atoms = []
    kept_codes = []
    kept_parent_indices = []
    moment_audit = []
    omitted_magnetic_sites = []
    component_index = {'x': 0, 'y': 1, 'z': 2}
    component_var = {'x': 'AMx', 'y': 'AMy', 'z': 'AMz'}
    hold_specs = []
    for atom_index, (atom, code) in enumerate(zip(new_phase['Atoms'], atom_codes)):
        parent_index=int(str(code).split(':',1)[0])
        parent_label=old_phase['Atoms'][parent_index][old_phase['General']['AtomPtrs'][1]-1]
        site_choice=next((s for s in protocol['sites'] if s['label']==parent_label),None)
        if site_choice is None:
            continue
        if not G2elem.GetMFtable([site_choice['form_factor']],[site_choice['lande_g']]):
            raise TrialDisposition('unsupported','The selected ionic magnetic form factor is absent from this native runtime')
        atom[ct]=site_choice['form_factor']
        element_match = re.match('^([A-Z][a-z]?)', str(atom[ct]))
        element = element_match.group(1) if element_match else ''
        if element not in magnetic_elements:
            continue
        atom[ct - 1] = f'{atom[ct - 1]}_m{atom_index + 1}'
        atom[ct + 1] = 'M'
        site_symmetry, _multiplicity, _operation, duplicate_direction = G2spc.SytSym(atom[cx:cx + 3], general['SGData'])
        allowed_ids, allowed_coefficients = G2spc.GetCSpqinel(general['SGData']['SpnFlp'], duplicate_direction)
        if not direction_in_basis(moment_basis(allowed_ids,allowed_coefficients),wanted_direction):
            omitted_magnetic_sites.append(dict(label=atom[ct-1],element=element,reason='Declared collinear direction is incompatible with coupled site components'))
            continue
        allowed = {axis for axis, index in component_index.items() if index < len(allowed_ids) and bool(allowed_ids[index])}
        requested_basis_ids = {int(allowed_ids[component_index[axis]]) for axis in candidate_refine_components if axis in allowed and int(allowed_ids[component_index[axis]]) > 0}
        if candidate_fixed_moment_direction is not None:
            incompatibility = []
            for index, basis_id in enumerate(allowed_ids):
                if int(basis_id) <= 0 and abs(float(candidate_fixed_moment_direction[index])) >= 1e-07:
                    incompatibility.append((index, 'forbidden'))
            for basis_id in sorted({int(value) for value in allowed_ids if int(value) > 0}):
                group = [index for index, value in enumerate(allowed_ids) if int(value) == basis_id]
                seed_index = next((index for index in group if abs(float(allowed_coefficients[index])) >= 1e-12), None)
                if seed_index is None:
                    incompatibility.append((basis_id, 'zero_basis'))
                    continue
                amplitude = float(candidate_fixed_moment_direction[seed_index]) / float(allowed_coefficients[seed_index])
                for index in group:
                    expected = amplitude * float(allowed_coefficients[index])
                    if abs(float(candidate_fixed_moment_direction[index]) - expected) >= 1e-07:
                        incompatibility.append((index, 'coupled_ratio'))
            if incompatibility:
                omitted_magnetic_sites.append({'label': atom[ct - 1], 'element': element, 'reason': 'candidate site symmetry is incompatible with the parent-basis fixed moment direction', 'allowed_components': sorted(allowed), 'direction_in_candidate_basis': list(candidate_fixed_moment_direction), 'incompatibility': incompatibility})
                continue
        refined = {axis for axis, index in component_index.items() if int(allowed_ids[index]) in requested_basis_ids}
        if not refined:
            omitted_magnetic_sites.append({'label': atom[ct - 1], 'element': element, 'reason': 'candidate symmetry forbids the requested moment orientation', 'allowed_components': sorted(allowed)})
            continue
        moment = np.zeros(3, dtype=float)
        for basis_id in sorted(requested_basis_ids):
            group = [index for index, value in enumerate(allowed_ids) if int(value) == basis_id]
            requested_group = [index for index in group if ('x', 'y', 'z')[index] in candidate_refine_components]
            seed_index = requested_group[0] if requested_group else group[0]
            seed_coefficient = float(allowed_coefficients[seed_index])
            if abs(seed_coefficient) < 1e-12:
                raise ValueError('GSAS-II returned a zero magnetic-basis coefficient')
            seed_amplitude = float(initial_moment[seed_index])
            if candidate_seed_moment_direction is not None:
                direction_coefficient = float(candidate_seed_moment_direction[seed_index])
                if abs(direction_coefficient) < 1e-08:
                    direction_coefficient = 0.02
                seed_amplitude = float(np.linalg.norm(initial_moment)) * direction_coefficient
            for index in group:
                moment[index] = seed_amplitude * float(allowed_coefficients[index]) / seed_coefficient
        if not any((abs(moment[component_index[axis]]) >= 1e-08 for axis in refined)):
            raise ValueError('A refined magnetic component would start at zero')
        atom[cx + 4:cx + 7] = list(moment)
        for axis in {'x', 'y', 'z'} - refined:
            hold_specs.append((component_var[axis], atom[-1]))
        kept_atoms.append(atom)
        kept_codes.append(code)
        kept_parent_indices.append(int(code.split('+')[0].split(':')[0]))
        moment_audit.append({'label': atom[ct - 1], 'element': element, 'allowed_components': sorted(allowed), 'refined_components': sorted(refined), 'initial_moment': list(moment)})
    ctx.save_json('site-policy-audit.json', dict(kept=moment_audit, omitted=omitted_magnetic_sites, require_all_sites=protocol['policy']['site_policy']=='all'))
    if omitted_magnetic_sites and protocol['policy']['site_policy']=='all':
        raise TrialDisposition('site_incompatible','Required magnetic sites/orientations incompatible: ' + str(omitted_magnetic_sites))
    if not kept_atoms:
        raise TrialDisposition('site_incompatible',f'No site of the requested magnetic elements {sorted(magnetic_elements)} supports the requested symmetry-compatible moment orientation; omitted_sites={omitted_magnetic_sites}')
    new_phase['Atoms'] = kept_atoms
    atom_codes = kept_codes
    general['Lande g'] = [2.0] * len(general.get('AtomTypes', []))
    general['Map'] = copy.deepcopy(G2elem.mapDefault)
    new_phase['Drawing'] = []
    G2sc.SetupGeneral(new_phase, None)
    general['Lande g']=[next(s['lande_g'] for s in protocol['sites'] if s['form_factor']==t) for t in general['AtomTypes']]

    new_phase['Histograms'] = {name: copy.deepcopy(old_phase['Histograms'][name]) for name in selected_histograms}
    n_shkl = len(G2spc.MustrainNames(general['SGData']))
    n_dij = len(G2spc.HStrainNames(general['SGData']))
    transferred_hstrain = {}
    hstrain_transfer_matrices = {}
    for name, hap in new_phase['Histograms'].items():
        hap['Scale'] = np.asarray(hap['Scale'], dtype=float) / det_trans
        hap['Mustrain'][4:6] = [n_shkl * [0.01], n_shkl * [False]]
        parent_effective_cell = G2lat.CellDijCorr(old_phase['General']['Cell'][1:7], old_phase['General']['SGData'], old_phase['Histograms'], name)
        child_effective_cell = list(G2lat.TransformCell(parent_effective_cell, trans)[:6])
        child_base_cell = list(general['Cell'][1:7])
        target_full_dij = np.asarray(G2lat.cell2A(child_effective_cell), dtype=float) - np.asarray(G2lat.cell2A(child_base_cell), dtype=float)
        hstrain_basis = np.stack([np.asarray(G2spc.HStrainVals([1.0 if row == column else 0.0 for row in range(n_dij)], general['SGData']), dtype=float) for column in range(n_dij)], axis=1)
        child_hstrain, _residuals, _rank, _singular = np.linalg.lstsq(hstrain_basis, target_full_dij, rcond=None)
        parent_hstrain_names = list(G2spc.HStrainNames(old_phase['General']['SGData']))
        parent_base_a = np.asarray(G2lat.cell2A(old_phase['General']['Cell'][1:7]), dtype=float)
        transfer_columns = []
        for parent_column in range(len(parent_hstrain_names)):
            parent_basis_values = [1.0 if row == parent_column else 0.0 for row in range(len(parent_hstrain_names))]
            parent_basis_full_dij = np.asarray(G2spc.HStrainVals(parent_basis_values, old_phase['General']['SGData']), dtype=float)
            parent_basis_cell = G2lat.A2cell(parent_base_a + parent_basis_full_dij)
            child_basis_cell = list(G2lat.TransformCell(parent_basis_cell, trans)[:6])
            child_basis_full_dij = np.asarray(G2lat.cell2A(child_basis_cell), dtype=float) - np.asarray(G2lat.cell2A(child_base_cell), dtype=float)
            child_basis_values = np.linalg.lstsq(hstrain_basis, child_basis_full_dij, rcond=None)[0]
            transfer_columns.append(child_basis_values)
        hstrain_transfer_matrix = np.stack(transfer_columns, axis=1)
        transferred_from_parent = hstrain_transfer_matrix @ np.asarray(old_phase['Histograms'][name]['HStrain'][0], dtype=float)
        transfer_error = float(np.max(np.abs(transferred_from_parent - child_hstrain)))
        represented_full_dij = hstrain_basis @ child_hstrain
        representation_error = float(np.max(np.abs(represented_full_dij - target_full_dij)))
        if not np.all(np.isfinite(child_hstrain)) or not math.isfinite(representation_error) or representation_error > 1e-08 or (not math.isfinite(transfer_error)) or (transfer_error > 1e-08):
            raise ValueError(f"Magnetic subgroup cannot represent the selected nuclear histogram's effective cell; histogram={name!r}, maximum reciprocal-metric error={representation_error:.6g}")
        hap['HStrain'] = [list(child_hstrain), n_dij * [False]]
        transferred_hstrain[name] = {'parent_effective_cell': [float(value) for value in parent_effective_cell], 'child_effective_cell': [float(value) for value in child_effective_cell], 'child_hstrain_names': list(G2spc.HStrainNames(general['SGData'])), 'child_hstrain_values': [float(value) for value in child_hstrain], 'parent_hstrain_names': parent_hstrain_names, 'child_from_parent_hstrain_matrix': [[float(value) for value in row] for row in hstrain_transfer_matrix], 'maximum_reciprocal_metric_error': representation_error, 'maximum_transfer_error': transfer_error}
        hstrain_transfer_matrices[name] = [[float(value) for value in row] for row in hstrain_transfer_matrix]
        hap['Use'] = True
    gpx.add_phase(phasefile=None, phasename=magnetic_name, histograms=[], spacegroup='P 1', cell=list(general['Cell'][1:7]))
    gpx.data['Phases'][magnetic_name] = new_phase
    gpx.update_ids()
    constraints = gpx.data['Constraints']['data']
    constraints['Phase'] += G2lat.GenCellConstraints(trans, old_phase['ranId'], new_phase['ranId'], G2lat.cell2A(old_phase['General']['Cell'][1:7]), old_phase['General']['SGData'], new_phase['General']['SGData'])

    def coupled_atom_constraints(old, new, codes, transformation):
        """Tie generated magnetic coordinates, occupancy and Uiso to the parent.

        This is the headless equivalent of GSAS-II's TransConstraints path. The
        upstream MatchGenAtomConstraints implementation in the bundled release has
        an uninitialized-loop-variable defect for fixed special-position atoms, so
        it cannot safely be called from an unattended refinement.
        """
        generated = []
        old_atoms = old['Atoms']
        new_atoms = new['Atoms']
        new_cx, _new_ct, _new_cs, new_cia = new['General']['AtomPtrs']
        old_ran_id = old['ranId']
        new_ran_id = new['ranId']
        coordinate_names = ['dAx', 'dAy', 'dAz']
        for atom_index, code in enumerate(codes):
            new_atom = new_atoms[atom_index]
            if new_atom[new_cia] != 'I':
                raise ValueError('Anisotropic displacement coupling is not yet supported by the headless magnetic adapter')
            site_symmetry = G2spc.SytSym(new_atom[new_cx:new_cx + 3], new['General']['SGData'])[0]
            coordinate_selection = G2spc.GetCSxinel(site_symmetry)[0]
            source, operation = code.split('+')[0].split(':')
            source_index = int(source)
            operation_number = abs(int(operation)) % 100 - 1
            if '-' in operation:
                operation_number *= -1
            operator = np.asarray(old['General']['SGData']['SGOps'][abs(operation_number)][0], dtype=float)
            if operation_number < 0:
                operator *= -1
            inverse_operator = np.linalg.inv(np.inner(operator, transformation))
            for coordinate_index, active in enumerate(coordinate_selection):
                if not active:
                    continue
                independent = [1.0, G2obj.G2VarObj([new_ran_id, None, coordinate_names[coordinate_index], new_atom[-1]])]
                dependent = []
                for old_index, multiplier in enumerate(inverse_operator[coordinate_index]):
                    if abs(float(multiplier)) <= 1e-06:
                        continue
                    dependent.append([float(multiplier), G2obj.G2VarObj([old_ran_id, None, coordinate_names[old_index], old_atoms[source_index][-1]])])
                if len(dependent) == 1:
                    generated.append([dependent[0], independent, None, None, 'e'])
                elif len(dependent) > 1:
                    independent[0] = -1.0
                    generated.append([independent] + dependent + [0.0, None, 'c'])
            for parameter_name in ('Afrac', 'AUiso'):
                generated.append([[1.0, G2obj.G2VarObj([new_ran_id, None, parameter_name, new_atom[-1]])], [1.0, G2obj.G2VarObj([old_ran_id, None, parameter_name, old_atoms[source_index][-1]])], None, None, 'e'])
        return generated
    atom_constraints = coupled_atom_constraints(old_phase, new_phase, atom_codes, trans)
    atom_constraint_warning = ''
    constraints['Phase'] += atom_constraints
    mode_constraints = []
    mode_anchor_axis = None
    if candidate_fixed_moment_direction is not None:
        nonzero_axes = [index for index, value in enumerate(candidate_fixed_moment_direction) if abs(float(value)) >= 1e-08]
        if not nonzero_axes:
            raise ValueError('candidate fixed moment direction must be nonzero')
        mode_anchor_axis = max(nonzero_axes, key=lambda index: abs(float(candidate_fixed_moment_direction[index])))
        axis_names = ['AMx', 'AMy', 'AMz']
        anchor_coefficient = float(candidate_fixed_moment_direction[mode_anchor_axis])
        initial_amplitude = float(np.linalg.norm(initial_moment))
        for atom in new_phase['Atoms']:
            atom[cx + 4:cx + 7] = list(initial_amplitude * candidate_fixed_moment_direction)
            anchor = G2obj.G2VarObj([new_phase['ranId'], None, axis_names[mode_anchor_axis], atom[-1]])
            for index, coefficient in enumerate(candidate_fixed_moment_direction):
                variable = G2obj.G2VarObj([new_phase['ranId'], None, axis_names[index], atom[-1]])
                if index == mode_anchor_axis:
                    continue
                if abs(float(coefficient)) < 1e-08:
                    mode_constraints.append([[1.0, variable], None, None, 'h'])
                else:
                    mode_constraints.append([[float(coefficient), anchor], [-anchor_coefficient, variable], 0.0, None, 'c'])
    if moment_site_coupling != 'independent':
        if mode_anchor_axis is None:
            requested_axes = [index for index, axis in enumerate(('x', 'y', 'z')) if axis in candidate_refine_components]
            if len(requested_axes) != 1:
                raise ValueError('same-parent moment coupling without a fixed direction requires exactly one refined component')
            mode_anchor_axis = requested_axes[0]
        axis_name = ('AMx', 'AMy', 'AMz')[mode_anchor_axis]
        groups = {}
        for atom_index, parent_index in enumerate(kept_parent_indices):
            groups.setdefault(parent_index, []).append(atom_index)
        coupling_sign = 1.0 if moment_site_coupling == 'same_parent_parallel' else -1.0
        for atom_indices in groups.values():
            if len(atom_indices) < 2:
                continue
            reference_atom = new_phase['Atoms'][atom_indices[0]]
            reference = G2obj.G2VarObj([new_phase['ranId'], None, axis_name, reference_atom[-1]])
            for atom_index in atom_indices[1:]:
                atom = new_phase['Atoms'][atom_index]
                if coupling_sign < 0:
                    atom[cx + 4:cx + 7] = [-float(value) for value in atom[cx + 4:cx + 7]]
                variable = G2obj.G2VarObj([new_phase['ranId'], None, axis_name, atom[-1]])
                mode_constraints.append([[1.0, variable], [-coupling_sign, reference], 0.0, None, 'c'])
    constraints['Phase'] += mode_constraints
    for name in selected_histograms:
        constraints['HAP'] += G2lat.GenHAPConstraints(det_trans, old_phase['ranId'], new_phase['ranId'], available_histograms[name].ranId)
        parent_hstrain_names = list(G2spc.HStrainNames(old_phase['General']['SGData']))
        child_hstrain_names = list(G2spc.HStrainNames(general['SGData']))
        histogram_ran_id = available_histograms[name].ranId
        transfer_matrix = np.asarray(hstrain_transfer_matrices[name], dtype=float)
        for child_index, child_name in enumerate(child_hstrain_names):
            child_variable = G2obj.G2VarObj([new_phase['ranId'], histogram_ran_id, child_name, None])
            relation = [[-1.0, child_variable]]
            for parent_index, parent_name in enumerate(parent_hstrain_names):
                coefficient = float(transfer_matrix[child_index, parent_index])
                if abs(coefficient) < 1e-10:
                    continue
                parent_variable = G2obj.G2VarObj([old_phase['ranId'], histogram_ran_id, parent_name, None])
                relation.append([coefficient, parent_variable])
            constraints['HAP'].append(relation + [0.0, None, 'c'])
    for variable_name, atom_ran_id in hold_specs:
        variable = G2obj.G2VarObj([new_phase['ranId'], None, variable_name, atom_ran_id])
        constraints['Phase'].append([[1.0, variable], None, None, 'h'])
    gpx.set_Controls('cycles', cycles)
    controls_entry = gpx.data.get('Controls')
    controls = controls_entry.get('data') if isinstance(controls_entry, dict) else None
    if not isinstance(controls, dict):
        raise RuntimeError('GSAS-II project does not expose Controls data for magnetic convergence')
    controls['min dM/M'] = minimum_relative_chi2_change
    gpx.save(str(output_gpx))
    gpx.index_ids()
    gpx.do_refinements([{}])
    gpx.save(str(output_gpx))
    magnetic = gpx.phase(magnetic_name)
    covariance = (gpx.data.get('Covariance') or {}).get('data') or {}
    rvals = covariance.get('Rvals') or {}
    vary_list = [str(value) for value in covariance.get('varyList', [])]
    raw_sigma_values = covariance.get('sig')
    raw_sigmas = list(raw_sigma_values if raw_sigma_values is not None else [])
    refined_parameter_sigmas = {}
    for variable_name, raw_sigma in zip(vary_list, raw_sigmas):
        try:
            sigma = float(raw_sigma)
        except (TypeError, ValueError):
            sigma = None
        refined_parameter_sigmas[variable_name] = sigma if sigma is not None and math.isfinite(sigma) and (sigma >= 0.0) else None
    refined_moment_variables = [value for value in vary_list if '::AM' in value]
    refined_magnetic_mode_variables = [value for value in vary_list if mode_constraints and value.startswith('::constr')]
    unexpected_refined_variables = [value for value in vary_list if '::AM' not in value and value not in refined_magnetic_mode_variables]
    if not refined_moment_variables and (not refined_magnetic_mode_variables):
        raise TrialDisposition('optimization_unresolved','Native refinement retained no independently varied magnetic moment; the attempted GPX is retained without a qualified fit')
    final_moments = []
    physicality_violations = []
    for atom in magnetic.data['Atoms']:
        moment = np.asarray(atom[cx + 4:cx + 7], dtype=float)
        magnitude = magnetic_magnitude(moment, magnetic.data['General']['Cell'][1:7])
        violation_reason = ''
        if not np.all(np.isfinite(moment)) or not math.isfinite(magnitude):
            violation_reason = 'nonfinite magnetic moment'
        elif magnitude > maximum_moment_magnitude:
            violation_reason = 'moment magnitude exceeds the declared material ceiling'
        if violation_reason:
            physicality_violations.append({'label': atom[ct - 1], 'components': list(moment), 'magnitude': magnitude, 'maximum_moment_magnitude': maximum_moment_magnitude, 'reason': violation_reason})
        component_variables = {}
        component_esd = {}
        for axis, parameter_name in zip(('mx', 'my', 'mz'), ('AMx', 'AMy', 'AMz')):
            variable_name = str(G2obj.G2VarObj([new_phase['ranId'], None, parameter_name, atom[-1]]))
            component_variables[axis] = variable_name
            component_esd[axis] = refined_parameter_sigmas.get(variable_name)
        nonzero_indices = [index for index, value in enumerate(moment) if abs(float(value)) > 1e-12]
        magnitude_esd = None
        if len(nonzero_indices) == 1:
            axis = ('mx', 'my', 'mz')[nonzero_indices[0]]
            magnitude_esd = component_esd.get(axis)
        final_moments.append({'label': atom[ct - 1], 'element': atom[ct], 'moment': list(moment), 'magnitude': magnitude, 'component_variables': component_variables, 'component_esd': component_esd, 'magnitude_esd': magnitude_esd, 'uncertainty_basis': 'GSAS-II covariance standard uncertainties; null components are fixed, dependent, or unavailable in the independent covariance basis'})
    final_residuals = {name: dict(gpx.histogram(name).residuals) for name in selected_histograms}

    def audit_target_peak_windows(histogram_name, half_width=12):
        baseline = baseline_target_profiles[histogram_name]
        baseline_x, baseline_yobs, baseline_weights, baseline_ycalc, baseline_background = baseline['arrays']
        final_x, final_yobs, final_weights, final_ycalc, final_background, _final_residual = powder_profile_arrays(gpx.histogram(histogram_name))
        native_grid_matches = len(final_x) == len(baseline_x) and np.allclose(final_x, baseline_x, rtol=0.0, atol=1e-08)
        final_native_point_count = int(len(final_x))
        if not native_grid_matches:
            finite_x = np.isfinite(final_x)
            if np.count_nonzero(finite_x) < 3:
                raise ValueError(f'Histogram {histogram_name!r} has too few finite profile points for magnetic target auditing')
            order = np.argsort(final_x[finite_x])
            source_x = final_x[finite_x][order]
            unique_x, unique_indices = np.unique(source_x, return_index=True)
            if len(unique_x) < 3:
                raise ValueError(f'Histogram {histogram_name!r} has too few unique profile points for magnetic target auditing')

            def align_to_baseline(values):
                source_values = np.asarray(values, dtype=float)[finite_x][order][unique_indices]
                valid = np.isfinite(source_values)
                if np.count_nonzero(valid) < 3:
                    return np.full_like(baseline_x, np.nan, dtype=float)
                return np.interp(baseline_x, unique_x[valid], source_values[valid], left=np.nan, right=np.nan)
            final_yobs = align_to_baseline(final_yobs)
            final_weights = align_to_baseline(final_weights)
            final_ycalc = align_to_baseline(final_ycalc)
            final_background = align_to_baseline(final_background)
            final_x = baseline_x.copy()
        rows = []
        for target in baseline['targets']:
            index = int(target['index'])
            left = max(0, index - int(half_width))
            right = min(len(final_x), index + int(half_width) + 1)
            if right - left < 3:
                continue
            valid = np.isfinite(baseline_x[left:right]) & np.isfinite(baseline_yobs[left:right]) & np.isfinite(baseline_weights[left:right]) & np.isfinite(baseline_ycalc[left:right]) & np.isfinite(baseline_background[left:right]) & np.isfinite(final_yobs[left:right]) & np.isfinite(final_weights[left:right]) & np.isfinite(final_ycalc[left:right]) & np.isfinite(final_background[left:right])
            if np.count_nonzero(valid) < 3:
                continue
            window_x = final_x[left:right][valid]
            dx = np.abs(np.gradient(window_x))
            window_final_yobs = final_yobs[left:right][valid]
            window_final_weights = final_weights[left:right][valid]
            window_final_ycalc = final_ycalc[left:right][valid]
            window_final_background = final_background[left:right][valid]
            window_baseline_yobs = baseline_yobs[left:right][valid]
            window_baseline_weights = baseline_weights[left:right][valid]
            window_baseline_ycalc = baseline_ycalc[left:right][valid]
            window_baseline_background = baseline_background[left:right][valid]
            observed_signal = float(np.sum(np.maximum(window_final_yobs - window_final_background, 0.0) * dx))
            baseline_calculated_signal = float(np.sum(np.maximum(window_baseline_ycalc - window_baseline_background, 0.0) * dx))
            final_calculated_signal = float(np.sum(np.maximum(window_final_ycalc - window_final_background, 0.0) * dx))
            baseline_error = float(np.sum(np.abs(window_baseline_yobs - window_baseline_ycalc) * np.sqrt(np.maximum(window_baseline_weights, 0.0))))
            final_error = float(np.sum(np.abs(window_final_yobs - window_final_ycalc) * np.sqrt(np.maximum(window_final_weights, 0.0))))
            denominator = max(observed_signal, 1e-12)
            final_coverage = final_calculated_signal / denominator
            rows.append({'x': float(target['x']), 'index': index, 'left_x': float(final_x[left]), 'right_x': float(final_x[right - 1]), 'baseline_significance': float(target['baseline_significance']), 'observed_signal_above_background': observed_signal, 'baseline_calculated_signal_above_background': baseline_calculated_signal, 'final_calculated_signal_above_background': final_calculated_signal, 'baseline_coverage': baseline_calculated_signal / denominator, 'final_coverage': final_coverage, 'coverage_error': abs(1.0 - final_coverage), 'baseline_weighted_absolute_residual': baseline_error, 'final_weighted_absolute_residual': final_error, 'weighted_residual_improvement': (baseline_error - final_error) / max(baseline_error, 1e-12), 'underfit': bool(final_coverage < 0.5)})
        signal_weight = sum((max(float(row['observed_signal_above_background']), 1e-12) for row in rows))
        residual_weight = sum((max(float(row['baseline_weighted_absolute_residual']), 1e-12) for row in rows))
        return {'grid_alignment': 'native' if native_grid_matches else 'final_interpolated_to_baseline', 'baseline_point_count': int(len(baseline_x)), 'final_native_point_count': final_native_point_count, 'target_count': len(rows), 'underfit_peak_count': sum((bool(row['underfit']) for row in rows)), 'weighted_coverage_error': sum((float(row['coverage_error']) * max(float(row['observed_signal_above_background']), 1e-12) for row in rows)) / signal_weight if rows else None, 'weighted_residual_improvement': sum((float(row['weighted_residual_improvement']) * max(float(row['baseline_weighted_absolute_residual']), 1e-12) for row in rows)) / residual_weight if rows else None, 'peaks': rows}
    target_peak_histograms = {name: audit_target_peak_windows(name) for name in selected_histograms}
    all_target_peaks = [row for histogram_audit in target_peak_histograms.values() for row in histogram_audit['peaks']]
    target_signal_weight = sum((max(float(row['observed_signal_above_background']), 1e-12) for row in all_target_peaks))
    target_residual_weight = sum((max(float(row['baseline_weighted_absolute_residual']), 1e-12) for row in all_target_peaks))
    target_peak_audit = {'selection_basis': 'strong positive residual maxima from the fixed nuclear-only baseline; configured GSAS-II excluded intervals omitted; identical windows for every magnetic subgroup trial', 'target_count': len(all_target_peaks), 'excluded_regions': {name: list(baseline_target_profiles[name].get('excluded_regions') or []) for name in selected_histograms}, 'underfit_peak_count': sum((bool(row['underfit']) for row in all_target_peaks)), 'weighted_coverage_error': sum((float(row['coverage_error']) * max(float(row['observed_signal_above_background']), 1e-12) for row in all_target_peaks)) / target_signal_weight if all_target_peaks else None, 'weighted_residual_improvement': sum((float(row['weighted_residual_improvement']) * max(float(row['baseline_weighted_absolute_residual']), 1e-12) for row in all_target_peaks)) / target_residual_weight if all_target_peaks else None, 'histograms': target_peak_histograms}
    rwp_values = [float(values['wR']) for values in final_residuals.values() if values.get('wR') is not None and math.isfinite(float(values['wR']))]
    if not rwp_values:
        raise RuntimeError('Magnetic refinement produced no finite weighted-profile R factor')
    receipt = {'schema_version': 'structural-transition-magnetic-refinement-receipt-v1', 'candidate': candidate, 'nuclear_phase_name': nuclear.name, 'magnetic_phase_name': magnetic_name, 'histogram_names': selected_histograms, 'magnetic_elements': sorted(magnetic_elements), 'moment_initialization': moment_audit, 'fixed_moment_direction': list(fixed_moment_direction) if fixed_moment_direction is not None else None, 'fixed_moment_direction_basis': 'parent_crystallographic', 'initial_moment_direction': list(initial_moment_direction) if initial_moment_direction is not None else None, 'initial_moment_direction_in_candidate_basis': list(candidate_seed_moment_direction) if candidate_seed_moment_direction is not None else None, 'fixed_moment_direction_in_candidate_basis': list(candidate_fixed_moment_direction) if candidate_fixed_moment_direction is not None else None, 'moment_site_coupling': moment_site_coupling, 'magnetic_mode_constraint_count': len(mode_constraints), 'omitted_magnetic_sites': omitted_magnetic_sites, 'refined_moments': final_moments, 'maximum_moment_magnitude': maximum_moment_magnitude, 'physicality_violations': physicality_violations, 'refined_moment_variables': refined_moment_variables, 'refined_magnetic_mode_variables': refined_magnetic_mode_variables, 'refined_parameter_sigmas': refined_parameter_sigmas, 'unexpected_refined_variables': unexpected_refined_variables, 'transferred_histogram_lattice_state': transferred_hstrain, 'screening_parameter_contract': {'refined_family': 'magnetic_moments_only', 'fixed_families': ['background', 'instrument', 'sample', 'nuclear_cell', 'nuclear_atomic_coordinates', 'nuclear_atomic_displacements', 'phase_scale', 'profile_broadening']}, 'minimum_relative_chi2_change': minimum_relative_chi2_change, 'refinement_rvals': rvals, 'baseline_residuals': baseline_residuals, 'final_residuals': final_residuals, 'target_peak_audit': target_peak_audit, 'phase_constraint_count': len(constraints['Phase']), 'hap_constraint_count': len(constraints['HAP']), 'atom_constraint_warning': atom_constraint_warning, 'magnetic_space_group': str(magnetic.data['General']['SGData'].get('MagSpGrp') or ''), 'candidate_only': True}
    ctx.register_file(source_path=output_gpx, relative_path='magnetic-refinement.gpx', artifact_type='magnetic_refinement_gpx', title=f'Magnetic refinement checkpoint: {magnetic_name}')
    listing = output_gpx.with_suffix('.lst')
    if listing.is_file():
        ctx.register_file(source_path=listing, relative_path='magnetic-refinement.lst', artifact_type='gsasii_listing', title='GSAS-II magnetic refinement listing')
    ctx.save_json('magnetic-refinement-receipt.json', receipt, title='Coupled nuclear/magnetic refinement receipt', artifact_type='magnetic_refinement_receipt')
    ctx.log_metric('rwp', min(rwp_values))
    ctx.log_metric('magnetic_phase_count', 1)
    ctx.log_metric('magnetic_atom_count', len(final_moments))
    finite_magnitudes = [float(item['magnitude']) for item in final_moments if math.isfinite(float(item['magnitude']))]
    if finite_magnitudes:
        ctx.log_metric('maximum_refined_moment_magnitude', max(finite_magnitudes))
    ctx.finish(f'Refined {magnetic_name} with the coupled nuclear phase, but retained {len(physicality_violations)} physicality rejection(s); best Rwp={min(rwp_values):.5g}%.' if physicality_violations else f'Refined {magnetic_name} with the coupled nuclear phase; best Rwp={min(rwp_values):.5g}%.')

    status='physical_rejection' if physicality_violations else 'optimization_unresolved' if not rvals.get('converged') or rvals.get('Aborted') or rvals.get('SVD0',0) else 'valid'
    if unexpected_refined_variables:
        raise TrialDisposition('technical_failure','Unexpected parameters varied: '+str(unexpected_refined_variables))
    return dict(status=status,gpx=output_gpx.name,rwp=min(rwp_values),nvars=int(rvals.get('Nvars',len(vary_list))),moments=final_moments,native_receipt='magnetic-refinement-receipt.json')
