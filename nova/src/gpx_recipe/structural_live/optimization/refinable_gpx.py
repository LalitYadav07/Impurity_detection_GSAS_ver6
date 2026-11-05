"""Export a native mode-refinement continuation beside the frozen point fit.

Execute in the export worker, never in a live sampler: GSAS-II constraint
maps are process-global. This preserves the point prediction; it does not
claim that GUI least squares inherits the search likelihood or chemistry.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.linalg import qr

from gpx_recipe.execution import freeze, write_json
from gpx_recipe.fit_domain import native_powder_fit_mask
from gpx_recipe.snapshot import json_safe
from gpx_recipe.structural_live.native import curve, inventory, modules, sha


def audit_inactive_native_limits(controls, active_variables, native_object):
    """Retain source limits that cannot freeze this continuation's variables.

    A recipe can legitimately leave ADP/occupancy limits in a GPX although
    those variables are fixed here. Native exact/wildcard lookup remains the
    authority. The recipe's nonnegative histogram-scale lower bound is
    already enforced by this service's positive log-scale parameterization.
    Other limits on newly active variables still require explicit transfer.
    """
    counts = {}; redundant = []
    for family in ('parmMinDict', 'parmMaxDict'):
        values = controls.get(family, {})
        if not isinstance(values, dict):
            raise ValueError('Native freeze-on-limit controls must be dictionaries')
        counts[family] = len(values)
        if not values:
            continue
        for variable in sorted(active_variables):
            matched, value = native_object.prmLookup(variable, values)
            if matched is not None:
                fields = variable.split(':')
                if (family == 'parmMinDict' and len(fields) == 3 and
                        fields[0] == '' and fields[1] and fields[2] == 'Scale' and value == 0.):
                    redundant.append(dict(variable=variable, rule=str(matched), value=float(value),
                                          enforced_by='positive histogram log-scale parameterization'))
                    continue
                raise ValueError(f'Native freeze-on-limit rule {matched} applies to active variable {variable}; explicit review required')
    return dict(preserved_inactive_rules=counts, active_variables_checked=len(active_variables),
                redundant_active_rules=redundant,
                matching='GSAS-II exact and wildcard parameter lookup', rules_removed=False)


def origin_reference_modes(gauge):
    """Hold a pivot set of native modes, removing only uniform translations.

    The remaining named modes span the entire physical quotient. A held
    amplitude chooses an origin reference; it does not forbid a physical
    displacement. This avoids inventing unnamed GUI refinement variables.
    """
    gauge = np.asarray(gauge, float)
    if gauge.ndim != 2 or not np.isfinite(gauge).all():
        raise ValueError('Invalid origin-gauge basis')
    n, rank = gauge.shape
    if rank > n or not np.allclose(gauge.T @ gauge, np.eye(rank), atol=1e-9):
        raise ValueError('Origin-gauge basis must have orthonormal columns')
    if not rank:
        return []
    _, _, pivots = qr(gauge.T, pivoting=True, mode='economic')
    held = sorted(map(int, pivots[:rank]))
    if np.linalg.matrix_rank(gauge[held], tol=1e-9) != rank:
        raise ValueError('Origin reference does not remove all translation gauges')
    return held


def mode_amplitudes(phase):
    if not phase.get('ISODISTORT', {}).get('G2ModeList'):
        return np.empty(0)
    _, math, *_ = modules()
    return np.asarray(math.CalcIsoDisp(phase)[2], float)


def assess_chemistry(project, phase_name=None):
    """Audit the actual saved structure, including periodic contacts."""
    from types import SimpleNamespace
    from .empirical_catalog import load_declared_catalog, EmpiricalNativeConstraint
    phase = project.phase(phase_name) if phase_name else project.phases()[0]
    _, _, lattice, *_ = modules()
    a0 = np.asarray(lattice.cell2A(phase.data['General']['Cell'][1:7]))
    basis = dict(coordinate_basis=np.zeros((len(phase.data['Atoms']), 3, 0)),
                 physical_basis=np.empty((0, 0)), cell_basis=np.empty((6, 0)), baseline_a=a0)
    native = SimpleNamespace(phase=phase, basis=basis, a_scale=float(np.mean(a0[:3])))
    catalog = load_declared_catalog(Path(__file__).with_name('empirical_chemistry_policy.json'))
    return EmpiricalNativeConstraint(native, catalog).assess([], [])


def continue_native_refinement(project, expected_variables, *, phase_name=None, histogram_name=None,
                               max_passes=3, profile_tolerance=1e-8):
    """Native refinements with explicit saved-parameter prediction checks.

    Native numerical refinement can leave ycalc at its last derivative probe.
    Recompute at the saved parameters and compare with the optimizer's actual
    residual sum, not that cached array. Restart only if that objective differs.
    Every iteration uses GSAS-II unmodified. Failed audits remain explicit.
    This post-fit audit does not enforce nonlinear chemistry during GUI steps.
    """
    if max_passes < 1:
        raise ValueError('At least one native pass is required')
    if phase_name is None and len(project.phases()) != 1:
        raise ValueError('Select the transforming phase for a multiphase continuation')
    if histogram_name is None and len(project.histograms()) != 1:
        raise ValueError('Select the histogram for a multibank continuation')
    phase=project.phase(phase_name) if phase_name else project.phases()[0]
    hist=project.histogram(histogram_name) if histogram_name else project.histograms()[0]
    passes=[]
    for _ in range(max_passes):
        start=time.monotonic();project.refine()
        cov=copy.deepcopy(project.data['Covariance']['data'])
        if set(cov.get('varyList',[]))!=set(expected_variables):
            raise ValueError('Native refinement did not save the intended active variables')
        if np.shape(cov.get('covMatrix'))!=(len(expected_variables),len(expected_variables)) or not np.isfinite(cov['covMatrix']).all():
            raise ValueError('Native refinement did not save finite covariance')
        if project.data['Controls']['data'].get('parmFrozen',{}).get('FrozenList'):
            raise ValueError('Native refinement silently froze variables')
        saved=np.array(hist.data['data'][1][3],copy=True)
        fresh=curve(project,hist)
        gap=float(np.linalg.norm(fresh-saved)/max(np.linalg.norm(fresh),1e-30))
        arrays=hist.data['data'][1][:3]
        x,y,w=[np.asarray(np.ma.getdata(a),float) for a in arrays]
        mask=(w>0)&np.isfinite(y)&np.isfinite(w)&np.isfinite(fresh)
        mask &= native_powder_fit_mask(x,hist.data['Limits'][1],hist.data['Limits'][2:])
        for a in arrays:mask &= ~np.ma.getmaskarray(a)
        fresh_chi2=float(np.sum(w[mask]*(y[mask]-fresh[mask])**2))
        objective_gap=abs(fresh_chi2-cov['Rvals']['chisq'])/max(1.,fresh_chi2)
        chemistry=assess_chemistry(project,phase.name)
        project.data['Covariance']['data']=cov
        hist.data['data'][1][3]=fresh
        hist.data['data'][1][5]=hist.data['data'][1][1]-fresh
        if phase.data.get('ISODISTORT'):
            phase.data['ISODISTORT']['modeDispl']=mode_amplitudes(phase.data).tolist()
        project.save()
        passes.append(dict(seconds=time.monotonic()-start,profile_relative_error=gap,
            objective_relative_error=objective_gap,
            native_rwp_percent=float(cov['Rvals']['Rwp']),chemistry_feasible=chemistry['feasible']))
        if not chemistry['feasible'] or objective_gap<=profile_tolerance:
            break
    return dict(passes=passes,objective_verified=passes[-1]['objective_relative_error']<=profile_tolerance,
        chemistry_feasible=chemistry['feasible'],chemistry=chemistry,
        native_curve=fresh,covariance=cov)


def configure(project, *, refine_cell=True, refine_scale=True, refine_background=True,
              phase_name=None, histogram_name=None):
    """Activate one transforming phase; retain other phases and banks fixed."""
    if phase_name is None and len(project.phases()) != 1:
        raise ValueError('Select the transforming phase for a multiphase export')
    if histogram_name is None and len(project.histograms()) != 1:
        raise ValueError('Select the independently analyzed histogram for export')
    # G2VarObj string resolution uses process-global random-ID indices. A
    # preceding project may have identical display IDs but different ranIds.
    project.index_ids()
    phase = project.phase(phase_name) if phase_name else project.phases()[0]
    hist = project.histogram(histogram_name) if histogram_name else project.histograms()[0]
    if phase.data['General']['Type'] != 'nuclear':
        raise ValueError('Only nuclear displacement-mode continuation is qualified')
    iso = phase.data.get('ISODISTORT', {})
    names = [v.name for v in iso.get('G2ModeList', [])]
    constraints = project.data['Constraints']['data']
    rows = constraints.get('Phase', [])
    if any(constraints.get(k) for k in ('HAP', 'Global')):
        raise ValueError('Additional source constraints need explicit transfer')
    for row in constraints.get('Hist', []):
        if (len(row)!=4 or row[-1]!='h' or row[0][1].name not in
                {f'Back;{i}' for i in range(1,int(hist.data['Background'][0][2]))}
                or str(row[0][1])!=f':{hist.id}:{row[0][1].name}'):
            raise ValueError('Additional histogram constraints need explicit transfer')
    constraints['Hist']=[]  # regenerated exactly once below
    if len(rows) != len(names) or any(c[-1] != 'f' for c in rows):
        raise ValueError('Only the native imported displacement constraints are accepted')
    if [r[-3].name for r in rows] != names:
        raise ValueError('Native mode-constraint identity/order mismatch')
    info, basis = inventory(phase.data)
    cx, ct, _, _ = phase.data['General']['AtomPtrs']
    atoms = {a[-1]: i for i, a in enumerate(phase.data['Atoms'])}
    lift = np.zeros((len(names), len(names)))
    for i, row in enumerate(rows):
        for coefficient, var in row[:-3]:
            if var.name not in ('dAx', 'dAy', 'dAz') or var.atom not in atoms:
                raise ValueError('Mode constraint escaped the displacement basis')
            lift[i] += float(coefficient) * basis['coordinate_basis'][atoms[var.atom], 'xyz'.index(var.name[-1])]
    if not np.allclose(lift, np.eye(len(names)), atol=1e-9, rtol=1e-9):
        raise ValueError('Native constraint and mode-amplitude normalizations disagree')
    held = origin_reference_modes(basis['gauge_basis'])
    freeze(project)
    for i, row in enumerate(rows):
        row[-2] = i not in held
    active_atoms = {v.atom for v in iso.get('G2VarList', [])}
    for atom in phase.data['Atoms']:
        if atom[-1] in active_atoms:
            atom[ct+1] = 'X'
    phase.set_refinements({'Cell': bool(refine_cell)})
    if refine_scale:
        hist.set_refinements({'Sample Parameters': ['Scale']})
    if refine_background:
        hist.data['Background'][0][1] = True
    # GSAS-II exposes one polynomial refinement checkbox. Explicit holds keep
    # higher coefficients at the fitted values, matching our search contract.
    project.update_ids()
    if refine_background:
        for i in range(1, int(hist.data['Background'][0][2])):
            project.add_HoldConstr([f':{hist.id}:Back;{i}'], reloadIdx=False)
    controls = project.data['Controls']['data']
    if controls.get('Seq Data'):
        raise ValueError('Sequential native refinement is not this continuation contract')
    # Native numerical derivatives avoid the known unqualified analytic cell
    # derivative path. GSAS-II remains responsible for every profile evaluation.
    controls['deriv type'] = 'numeric'
    # Prepared forward-only GPXs deliberately have zero cycles. GSAS-II's
    # numerical branch can optimize yet discard the result when this is zero.
    controls['max cyc'] = 30
    controls['min dM/M'] = 1e-8
    controls['shift factor'] = 1.
    frozen = controls.get('parmFrozen', {})
    if not isinstance(frozen, dict) or any(frozen.values()):
        raise ValueError('Previously frozen native variables require explicit review')
    controls['parmFrozen'] = {}
    project.data['Covariance']['data'] = {}
    if names:
        iso['modeDispl'] = mode_amplitudes(phase.data).tolist()
    comments=project.data.get('Comments',{}).get('data')
    if isinstance(comments,list):
        comments.extend(['RADAR-PD structural continuation: instrument parameters fixed.',
            'Named ISODISTORT modes are enabled; any held mode only fixes an origin reference.',
            'Native weighted least squares; empirical chemistry limits and search boxes require the RADAR-PD wrapper.',
            'The adjacent refinement-contract.json records the exact active variables and fitted amplitudes.'])
    project.save()
    from gpx_recipe.live.native_profile import NativeReducedProfile
    native = NativeReducedProfile(project.filename, refresh_cell_reflections=True)
    mode_variables = [f'{phase.id}::nv-{name}' for i, name in enumerate(names) if i not in held]
    allowed = set(mode_variables)
    cell_variables = [v for v in native.vary if v.startswith(f'{phase.id}::A') and v[-1:].isdigit()]
    if len(cell_variables) != (info['cell_rank'] if refine_cell else 0):
        raise ValueError('Native lattice variable count differs from the candidate symmetry')
    allowed.update(cell_variables)
    if refine_scale:
        allowed.add(f':{hist.id}:Scale')
    if refine_background:
        allowed.add(f':{hist.id}:Back;0')
    if set(native.vary) != allowed or len(native.vary) != len(allowed):
        raise ValueError(f'Native active variables differ from export contract: {native.vary}')
    g = modules()[0]
    native_object = importlib.import_module((g.__package__ + '.' if g.__package__ else '') + 'GSASIIobj')
    native_limits = audit_inactive_native_limits(controls, allowed, native_object)
    return dict(native_mode_count=len(names), physical_mode_count=info['physical_rank'],
        origin_gauge_count=len(held), modes=[dict(name=name, label=iso['IsoModeList'][i],
            amplitude=float(mode_amplitudes(phase.data)[i]), refine=i not in held,
            held_reason='origin reference, not a rejected distortion' if i in held else None)
            for i, name in enumerate(names)], native_active_variables=list(native.vary),
        cell_variables=cell_variables, instrument_parameters_fixed=True, native_limit_audit=native_limits,
        higher_background_coefficients_fixed=True, derivative_method='native GSAS-II numeric',
        objective='GSAS-II weighted least squares using the saved observations and weights',
        chemistry_enforced_during_gui_refinement=False, search_parameter_bounds_enforced=False,
        scope='Local native continuation; chemistry and original search domain require wrapper validation')


def export_refinable(source_gpx, output, *, expected_sha256=None, **options):
    begun = time.monotonic()
    source = Path(source_gpx).resolve()
    before = sha(source)
    if expected_sha256 is not None and before != expected_sha256:
        raise ValueError('Source fitted GPX checksum changed')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    g, *_ = modules()
    project = g.G2Project(gpxfile=str(source), newgpx=str(output/'refine.gpx'))
    hist = project.histogram(options['histogram_name']) if options.get('histogram_name') else project.histograms()[0]
    observations = [np.ma.array(a, copy=True) for a in hist.data['data'][1][:3]]
    instrument = copy.deepcopy(hist.data['Instrument Parameters'])
    original = np.asarray(hist.data['data'][1][3]).copy()
    # Fresh evaluation certifies that source ycalc was not stale.
    frozen_curve = curve(project, hist)
    if not np.allclose(original, frozen_curve, rtol=1e-8, atol=1e-10):
        raise ValueError('Source saved curve is stale; regenerate the fitted point first')
    contract = configure(project, **options)
    chemistry = assess_chemistry(project, options.get('phase_name'))
    if not chemistry['feasible']:
        raise ValueError('Fitted structure violates the declared empirical chemistry policy')
    fresh = curve(project, hist)
    if not np.allclose(fresh, frozen_curve, rtol=1e-8, atol=1e-10):
        raise ValueError('Activating refinement changed the fitted prediction')
    # freeze() changes instrument flags, but no instrumental numerical value.
    after_instrument = copy.deepcopy(hist.data['Instrument Parameters'])
    for original_params, after_params in zip(instrument, after_instrument):
        for key, value in original_params.items():
            other = after_params[key]
            if isinstance(value, list) and len(value) > 2 and isinstance(value[2], bool):
                if value[:2] != other[:2] or other[2]:
                    raise ValueError('Instrument parameters changed or became active')
            elif json_safe(value) != json_safe(other):
                raise ValueError('Instrument metadata changed')
    for original_array, after_array in zip(observations, hist.data['data'][1][:3]):
        if not np.array_equal(np.ma.getdata(original_array), np.ma.getdata(after_array), equal_nan=True) or not np.array_equal(np.ma.getmaskarray(original_array), np.ma.getmaskarray(after_array)):
            raise ValueError('Continuation export changed observations or weights')
    hist.data['data'][1][3] = fresh
    hist.data['data'][1][5] = observations[1]-fresh
    project.save()
    if before != sha(source):
        raise ValueError('Read-only source GPX changed during export')
    receipt = dict(schema='structural-refinable-gpx/v1', source=str(source), source_sha256=before,
        path=str(project.filename), sha256=sha(project.filename), contract=contract,
        curve_relative_error=float(np.linalg.norm(fresh-frozen_curve)/max(np.linalg.norm(frozen_curve),1e-30)),
        native_active_set_verified=True, refinement_roundtrip_tested=False,
        chemistry=chemistry,
        implementation_sha256=sha(__file__), seconds=time.monotonic()-begun)
    write_json(output/'refinement-contract.json', receipt)
    return receipt
