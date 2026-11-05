"""Candidate-specific native preparation and nonlinear forward evaluation.

Uses native modes and native cell symmetry. There is no shared-subgroup
scattering matrix or custom approximation to structure factors.
"""
from __future__ import annotations
import copy
import hashlib
import importlib
import inspect
from pathlib import Path

import numpy as np

from ..execution import freeze, write_json
from ..snapshot import gsas
from ..symmetry.displacements import transform_uij, metric
from ..symmetry.native_preparation import normalize_flags
from ..symmetry.compiler import package_signature, require_adapters
from .mode_basis import origin_gauges


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def modules():
    g = gsas()
    prefix = g.__package__ + '.' if g.__package__ else ''
    return g, *(importlib.import_module(prefix + name) for name in
                ['GSASIImath', 'GSASIIlattice', 'GSASIIspc', 'GSASIIstrMain', 'GSASIIstrIO'])


def rotations(sg):
    values = [np.asarray(r, float) for r, _ in sg['SGOps']]
    if sg.get('SGInv'): values += [-r for r in values]
    return values


def require_nuclear_runtime():
    """Bind the actual scattering implementation, not only scriptable wrappers."""
    g = gsas(); prefix = g.__package__ + '.' if g.__package__ else ''
    native = importlib.import_module(prefix+'GSASIIstrMath')
    expected = {
        'StructureFactor2':'adac1ed638a4a062ff606676b2aeb4e43522b7ffa4b44bb388333f49c3672309',
        'StructureFactorDerv2':'ad383d99b7e2ab6adc160e9f9d1daf6e91f1eb74a86ebc3b2cadacc33068cd6a',
    }
    actual = {name:hashlib.sha256(inspect.getsource(getattr(native,name)).encode()).hexdigest() for name in expected}
    if actual != expected:
        raise ValueError('Structural live work requires the reviewed nuclear powder intensity/derivative runtime; build a separate qualified runtime')
    return dict(module_sha256=sha(native.__file__),functions=actual)


def apply_modes(phase, amplitudes, math_module):
    iso = phase.get('ISODISTORT', {})
    n = len(iso.get('G2ModeList', []))
    q = np.asarray(amplitudes, float)
    if q.shape != (n,) or not np.isfinite(q).all():
        raise ValueError('Native displacement vector has wrong dimension or nonfinite values')
    if not n: return
    iso['modeDispl'] = q.tolist()
    phase['Drawing'] = dict(atomPtrs=[2, 1, 6, 17],
                            Atoms=[math_module.MakeDrawAtom(phase, a) for a in phase['Atoms']])
    if math_module.ApplyModeDisp(phase):
        raise ValueError('Native displacement application failed')


def inventory(phase):
    _, m, l, s, _, io = modules()
    iso = phase.get('ISODISTORT', {})
    names = [v.name for v in iso.get('G2ModeList', [])]
    n = len(names); cx = phase['General']['AtomPtrs'][0]
    original = copy.deepcopy(phase)
    try:
        apply_modes(phase, np.zeros(n), m)
        zero = np.asarray([a[cx:cx+3] for a in phase['Atoms']], float)
        mapping = np.zeros((len(zero), 3, n))
        for i in range(n):
            q = np.zeros(n); q[i] = 1.
            apply_modes(phase, q, m)
            mapping[:, :, i] = np.asarray([a[cx:cx+3] for a in phase['Atoms']]) - zero
        q = np.random.default_rng(9217).normal(size=n) * .08
        apply_modes(phase, q, m)
        actual = np.asarray([a[cx:cx+3] for a in phase['Atoms']])
        if not np.allclose(actual, zero + mapping @ q, atol=1e-10, rtol=1e-10):
            raise ValueError('Native displacement coordinates failed affine verification')
        gauges = origin_gauges(mapping, rotations(phase['General']['SGData']))
        # Probe native cellFill to obtain its exact linear reciprocal-metric
        # subspace. This includes zero-displacement/strain-only candidates.
        columns = []
        for i in range(6):
            values = {f'A{j}': float(j == i) for j in range(6)}
            columns.append(io.cellFill('', phase['General']['SGData'], values, {})[0])
        u, singular, _ = np.linalg.svd(np.asarray(columns).T)
        cell_basis = u[:, singular > 1e-10]
        a0 = np.asarray(l.cell2A(phase['General']['Cell'][1:7]))
        if np.max(np.abs(cell_basis @ (cell_basis.T @ a0) - a0)) > 1e-9:
            raise ValueError('Imported cell is incompatible with native cell symmetry')
        return dict(native_mode_names=names, native_mode_labels=list(iso.get('IsoModeList', [])),
                    normalization=list(map(float, iso.get('NormList', []))),
                    cell_rank=cell_basis.shape[1],
                    **{k:v for k,v in gauges.items() if not isinstance(v, np.ndarray)}), dict(
                        coordinate_basis=mapping, physical_basis=gauges['physical_basis'],
                        gauge_basis=gauges['gauge_basis'], cell_basis=cell_basis, baseline_a=a0)
    finally:
        phase.clear(); phase.update(original)


def curve(project, histogram):
    g, _, _, _, native, io = modules()
    prefix=g.__package__+'.' if g.__package__ else ''
    # DoNoFit reads constraints but does not clear an earlier refinement's
    # process-global variable maps. Saved parameters are already physical;
    # stale maps must never be applied to this independent evaluation.
    importlib.import_module(prefix+'GSASIImapvars').InitVars()
    project.data['Covariance']['data'] = {}
    project.save()
    # Native GPX offsets can be stale after same-size rewrites. Explicitly
    # invalidate the native cache, without replacing a native function.
    io.gpxSize = -1
    y = np.asarray(native.DoNoFit(str(project.filename), histogram.name), float)
    if y.shape != np.asarray(histogram.data['data'][1][0]).shape or not np.isfinite(y).all():
        raise ValueError('Invalid native structural profile')
    return y


def refresh_profile_metadata(project, histogram, prediction):
    """Retain GSAS-II's current reflections, background and fit statistics.

    DoNoFit returns its curve but discards its local histogram dictionaries.
    A child otherwise has missing Bragg ticks and its parent's residuals.
    Use the existing native forward adapter, check the same prediction, and
    copy only calculated metadata. Observations, weights and limits stay put.
    """
    from ..live.native_profile import NativeReducedProfile
    from ..fit_domain import native_powder_fit_mask
    from ..symmetry.native_audit import profile_residual_metrics
    forward = NativeReducedProfile(project.filename)
    forward.profile(forward.x0)
    data = forward.histo_phases[0][histogram.name]
    calculated = np.asarray(data['Data'][3], float)
    expected = np.asarray(prediction, float)
    x = np.asarray(np.ma.getdata(histogram.data['data'][1][0]), float)
    mask = native_powder_fit_mask(x, histogram.data['Limits'][1], histogram.data['Limits'][2:])
    mask &= ~np.ma.getmaskarray(histogram.data['data'][1][0])
    if not mask.any() or not np.allclose(calculated[mask], expected[mask], rtol=1e-8, atol=1e-10):
        raise ValueError('Native reflection metadata disagrees with the saved prediction')
    # The profile adapter intentionally uses unit weights. Restore the saved
    # measurement model in its private dictionaries before asking GSAS-II for
    # statistics; never copy its unweighted or inherited parent residuals.
    for name, _ in forward.parts:
        saved = project.histogram(name).data
        native = forward.histo_phases[0][name]
        arrays = saved['data'][1]
        observed, weights = [np.asarray(np.ma.getdata(a), float) for a in arrays[1:3]]
        invalid = (~np.isfinite(observed) | ~np.isfinite(weights) | (weights <= 0)
                   | np.ma.getmaskarray(arrays[1]) | np.ma.getmaskarray(arrays[2]))
        native['Data'][0] = np.ma.array(native['Data'][0],
            mask=np.ma.getmaskarray(native['Data'][0]) | np.ma.getmaskarray(arrays[0]))
        native['Data'][1] = np.ma.array(np.where(invalid, 0., observed), mask=invalid)
        native['Data'][2] = np.ma.array(np.where(invalid, 0., weights), mask=invalid)
        # All-zero observations trigger native random dummy-data generation.
        # There is no meaningful Rwp in that case; do not publish invented data.
        if not np.any(observed[~invalid]):
            raise ValueError('Native residual statistics require nonzero measured intensity')
        factor = float(saved['data'][0].get('wtFactor', 1.))
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError('Native residual statistics require a finite positive weight factor')
        forward.controls[f":{native['hId']}:wtFactor"] = factor
        native['Residuals'] = {}
    forward.math.errRefine(forward.x0, forward.histo_phases, forward.parms,
        forward.histdict, forward.vary, forward.controls, forward.pawley, None)
    # Use the native powder structure-factor extraction for RF/RF2 and Nref,
    # rather than retaining parent-phase counts or approximating their formulas.
    forward.math.GetFobsSq(*forward.histo_phases[:2], forward.parms, forward.controls)
    data = forward.histo_phases[0][histogram.name]
    if not np.allclose(np.asarray(data['Data'][3])[mask], expected[mask], rtol=1e-8, atol=1e-10):
        raise ValueError('Native statistics refresh changed the saved prediction')
    independent = profile_residual_metrics(histogram.data['data'][1][:3],
        expected, histogram.data['Limits'])
    if not np.isclose(data['Residuals']['wR'], min(100., independent['rwp_native']), rtol=1e-8, atol=1e-8):
        raise ValueError('Native exported Rwp disagrees with the measured weighted objective')
    reflections = data['Reflection Lists']
    histogram.data['Reflection Lists'] = copy.deepcopy(reflections)
    # Scriptable GPX stores these in the histogram's first data dictionary;
    # a top-level 'Residuals' entry is discarded by its canonical writer.
    histogram.data['data'][0].update(copy.deepcopy(data['Residuals']))
    histogram.data['data'][1][4] = np.asarray(data['Data'][4]).copy()


def parent_sites(parent, child, transform, origin, native_spc):
    pcx, pct, _, pcia = parent['General']['AtomPtrs']
    cx, ct, _, cia = child['General']['AtomPtrs']
    sg = parent['General']['SGData']
    coords = child.get('ISODISTORT', {}).get('G2parentCoords', [a[cx:cx+3] for a in child['Atoms']])
    rows = []
    for atom, xyz in zip(child['Atoms'], coords):
        target = transform.T @ xyz + origin
        matches = []
        for index, old in enumerate(parent['Atoms']):
            if str(old[pct]).casefold() != str(atom[ct]).casefold(): continue
            operations = []
            for sign in ([1, -1] if sg.get('SGInv') else [1]):
                for r, t in sg['SGOps']:
                    for center in sg['SGCen']:
                        position = sign * (np.asarray(r) @ old[pcx:pcx+3] + t) + center
                        if np.max(np.abs((position - target + .5) % 1 - .5)) < 1e-6:
                            operations.append(sign * np.asarray(r))
            if operations: matches.append((index, old, operations))
        if len(matches) != 1:
            raise ValueError('Child site has ambiguous or missing exact parent orbit')
        index, old, ops = matches[0]
        atom[cx:cx+3] = list(xyz); atom[cx+3] = old[pcx+3]; atom[cia] = old[pcia]
        if old[pcia] == 'I':
            if not np.isfinite(old[pcia+1]) or old[pcia+1] < 0:
                raise ValueError('Invalid parent Uiso')
            atom[cia+1] = old[pcia+1]; atom[cia+2:cia+8] = [0.]*6
        elif old[pcia] == 'A':
            tensors = [transform_uij(old[pcia+2:pcia+8], parent['General']['Cell'][1:7],
                       child['General']['Cell'][1:7], transform, r) for r in ops]
            if not np.allclose(tensors, tensors[0], atol=1e-8, rtol=1e-7):
                raise ValueError('Parent ADP violates its site symmetry')
            atom[cia+2:cia+8] = tensors[0]
        else: raise ValueError('Unknown parent displacement model')
        rows.append(dict(child=atom[ct-1], parent=old[pct-1], parent_index=index, adp_type=old[pcia]))
    determinant = abs(np.linalg.det(transform))
    for index, old in enumerate(parent['Atoms']):
        actual = sum(native_spc.SytSym(a[cx:cx+3], child['General']['SGData'])[1] * a[cx+3]
                     for a,row in zip(child['Atoms'], rows) if row['parent_index'] == index)
        expected = native_spc.SytSym(old[pcx:pcx+3], sg)[1] * old[pcx+3] * determinant
        if not np.isclose(actual, expected, rtol=0, atol=1e-6):
            raise ValueError('Child did not preserve each parent orbit number density')
    return rows


def transfer_isotopes(parent, child, native_scriptable):
    """Preserve explicit neutron isotope selections and rebuild native tables."""
    source = parent['General'].get('Isotope', {})
    target = child['General'].get('Isotopes', {})
    if set(source) != set(target):
        raise ValueError('Parent/child isotope species require an exact mapping')
    if any(value not in target[species] for species, value in source.items()):
        raise ValueError('Parent isotope selection is unavailable in the child native tables')
    child['General']['Isotope'] = copy.deepcopy(source)
    native_scriptable.SetupGeneral(child, '')
    if child['General']['Isotope'] != source:
        raise ValueError('Native setup changed the requested isotope selection')
    return dict(source)


def validate_sample_broadening(hap):
    """Require the physical and saved-refinement domains of native HAP terms."""
    for family in ['Size','Mustrain']:
        item=hap[family]
        if str(item[0]).lower()!='isotropic':
            raise ValueError('Anisotropic sample broadening needs an exact child transfer')
        value,mixing=float(item[1][0]),float(item[1][2])
        if not np.isfinite(value) or not np.isfinite(mixing) or not 0<=mixing<=1:
            raise ValueError(f'{family} requires a finite physical value and mixing fraction in [0,1]')
        if family=='Size' and not .001<=value<=10.:
            raise ValueError('Crystallite size is outside the native saved-refinement domain [0.001,10] micrometres')
        if family=='Mustrain' and value<0:
            raise ValueError('Microstrain must be nonnegative')


def prepare(source_gpx, child_cif, transform, origin, output, *, phase_name=None, histogram_name=None, reference_gpx=None):
    """Produce a private immutable construction template and native audit."""
    require_adapters()
    scattering_identity = require_nuclear_runtime()
    from gsas_isodistort_import import add_isodistort_phase
    g, m, l, s, _, _ = modules()
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    source_gpx = Path(source_gpx).resolve(); child_cif = Path(child_cif).resolve()
    source_hash, child_hash = sha(source_gpx), sha(child_cif)
    reference=None;reference_hash=None
    if reference_gpx is not None:
        reference_gpx=Path(reference_gpx).resolve();reference_hash=sha(reference_gpx)
        reference_project=g.G2Project(gpxfile=str(reference_gpx))
        reference_phase=reference_project.phase(phase_name) if phase_name else reference_project.phases()[0]
        reference=copy.deepcopy(reference_phase.data)
    p = g.G2Project(gpxfile=str(source_gpx), newgpx=str(output/'prepared.gpx'))
    h = p.histogram(histogram_name) if histogram_name else p.histograms()[0]
    histogram_scale = float(h.data['Sample Parameters']['Scale'][0])
    if not np.isfinite(histogram_scale) or histogram_scale <= 0:
        raise ValueError('Structural log-scale fitting requires a finite positive source histogram scale')
    parent = p.phase(phase_name) if phase_name else p.phases()[0]
    if reference is not None:
        from gsas_magnetic_preflight import setting_fingerprint
        if (reference['General']['Type']!='nuclear' or reference['General'].get('Modulated') or
                setting_fingerprint(reference['General']['SGData'])!=setting_fingerprint(parent.data['General']['SGData'])):
            raise ValueError('Reference and target require the same nonmodulated nuclear parent setting')
        def sites(phase):
            cx,ct,_,_=phase['General']['AtomPtrs']
            return {a[ct-1]:(a[ct],float(a[cx+3])) for a in phase['Atoms']}
        if sites(reference)!=sites(parent.data) or reference['General'].get('Isotope')!=parent.data['General'].get('Isotope'):
            raise ValueError('Reference and target chemistry, site identities or isotope selection changed')
        # Reference supplies fractional structure and ADPs. Target supplies
        # the fitted parent cell, HAPs, all secondary phases and observations.
        general=copy.deepcopy(reference['General']);general['Cell']=copy.deepcopy(parent.data['General']['Cell'])
        parent.data['General']=general;parent.data['Atoms']=copy.deepcopy(reference['Atoms'])
        p.update_ids()
    if any(ph.data['General']['Type'] != 'nuclear' or ph.data['General'].get('Modulated') for ph in p.phases()):
        raise ValueError('Structural preparation requires nonmodulated nuclear phases')
    if any(p.data['Constraints']['data'].get(k) for k in ['Phase', 'HAP']):
        raise ValueError('Source phase/HAP constraints require exact child-variable substitution')
    if p.data.get('Restraints', {}).get('data', {}).get(parent.name):
        raise ValueError('Parent restraints require explicit child-site transfer')
    normalize_flags(p); freeze(p)
    p.set_Controls('sequential', []); p.set_Controls('seqCopy', False); p.set_Controls('Reverse Seq', False)
    for ph in p.phases():
        for name,hap in ph.data['Histograms'].items(): hap['Use'] = name == h.name and hap.get('Use', True)
    if not parent.data['Histograms'].get(h.name, {}).get('Use'):
        raise ValueError('Transforming phase is not active in the selected histogram')
    baseline = curve(p, h)
    old = copy.deepcopy(parent.data); hap = copy.deepcopy(old['Histograms'][h.name])
    t = np.asarray(transform, float); o = np.asarray(origin, float)
    if t.shape != (3,3) or o.shape != (3,) or not np.isfinite(t).all() or not np.isfinite(o).all() or abs(np.linalg.det(t)) < 1e-10:
        raise ValueError('Invalid candidate basis/origin')
    # Nonzero parent HStrain changes the effective metric, and thus ADP tensor
    # transfer. Keep this explicit until that combined mapping is qualified.
    if any(abs(x) > 1e-12 for x in hap['HStrain'][0]):
        raise ValueError('Nonzero parent HStrain needs a qualified effective-cell transfer')
    validate_sample_broadening(hap)
    if abs(hap['Pref.Ori.'][1] - 1.) > 1e-12 or old['General'].get('SH Texture', {}).get('Order', 0):
        raise ValueError('Nontrivial texture needs an exact child transfer')
    name = parent.name
    p.data['Phases'].pop(name)
    for row in p.names:
        if row and row[0] == 'Phases' and name in row: row.remove(name)
    h.data.get('Reflection Lists', {}).pop(name, None)
    child, receipt = add_isodistort_phase(p, child_cif, allow_strain_only=True,
                                         phasename=name, histograms=[h.name], fmthint='CIF')
    child.data['General']['Cell'][1:] = list(l.TransformCell(old['General']['Cell'][1:7], t))
    from .site_anchor import anchor_child
    anchoring = anchor_child(old, child.data, t, o, child_cif, s)
    site_map = parent_sites(old, child.data, t, o, s)
    isotope_selection = transfer_isotopes(old, child.data, g)
    new_hap = child.data['Histograms'][h.name]
    for field in ['Size', 'Mustrain', 'Extinction', 'Babinet']:
        new_hap[field] = copy.deepcopy(hap[field])
    new_hap['Scale'] = [hap['Scale'][0] / abs(np.linalg.det(t)), False]
    new_hap['HStrain'] = [[0.]*len(s.HStrainNames(child.data['General']['SGData'])),
                          [False]*len(s.HStrainNames(child.data['General']['SGData']))]
    p.update_ids(); freeze(p)
    details, arrays = inventory(child.data)
    apply_modes(child.data, np.zeros(len(details['native_mode_names'])), m)
    zero = curve(p, h)
    x, _, weights = (np.asarray(v) for v in h.data['data'][1][:3])
    from .reduced import fit_mask
    mask = (weights > 0) & fit_mask(x, h.data['Limits'])
    for array in h.data['data'][1][:3]: mask &= ~np.ma.getmaskarray(array)
    if not mask.any(): raise ValueError('No usable bins in native comparison')
    denominator = max(float(np.sum(weights[mask] * baseline[mask]**2)), 1e-30)
    error = float(np.sqrt(np.sum(weights[mask] * (zero[mask]-baseline[mask])**2) / denominator))
    write_json(output/'zero-control.json', dict(relative_profile_error=error, tolerance=1e-5))
    if error > 1e-5: raise ValueError('Zero-distortion native child does not reproduce parent')
    gauge_errors = []
    for column in arrays['gauge_basis'].T:
        apply_modes(child.data, column * .2, m)
        y = curve(p, h)
        gauge_errors.append(float(np.linalg.norm(y-zero) / max(np.linalg.norm(zero), 1e-30)))
    if any(e > 1e-8 for e in gauge_errors): raise ValueError('A proposed origin gauge changes native diffraction')
    apply_modes(child.data, np.zeros(len(details['native_mode_names'])), m)
    p.data['Covariance']['data'] = {}; p.save()
    np.savez_compressed(output/'basis.npz', **arrays)
    np.savez_compressed(output/'controls.npz', x=x, parent=baseline, child=zero, mask=mask)
    if sha(source_gpx) != source_hash or sha(child_cif) != child_hash:
        raise ValueError('Authoritative preparation inputs changed')
    if reference_gpx is not None and sha(reference_gpx)!=reference_hash:
        raise ValueError('Authoritative parent reference changed')
    result = dict(schema='radar-structural-preparation/v1', source_gpx=str(source_gpx),
                  source_gpx_sha256=source_hash, child_cif=str(child_cif), child_cif_sha256=child_hash,
                  reference_gpx_sha256=reference_hash,
                  template_sha256=sha(output/'prepared.gpx'), basis_sha256=sha(output/'basis.npz'),
                  histogram=h.name, phase=child.name, import_receipt=receipt, site_mapping=site_map,
                  exact_parent_anchor=anchoring,
                  isotope_selection=isotope_selection,
                  transformation_matrix=t.tolist(), origin_shift=o.tolist(),
                  zero_profile_error=error, gauge_profile_errors=gauge_errors,
                  inventory=details, statistical_policy_qualified=False,
                  implementation_sha256=package_signature(),
                  scattering_identity=scattering_identity,
                  native_source_sha256={x.__name__:sha(x.__file__) for x in modules()},
                  scope='native construction and geometry; no stopping-policy qualification')
    write_json(output/'prepared.json', result)
    return result


class NativeCandidate:
    """Private worker instance; never share GSAS-II mutable state across threads."""
    def __init__(self, prepared, work):
        import json
        self.root = Path(prepared); self.meta = json.loads((self.root/'prepared.json').read_text())
        if self.meta.get('implementation_sha256') != package_signature():
            raise ValueError('Prepared scientific package changed; reprepare this candidate')
        if require_nuclear_runtime() != self.meta.get('scattering_identity'):
            raise ValueError('Prepared native scattering implementation changed')
        for file,key in [('prepared.gpx','template_sha256'), ('basis.npz','basis_sha256')]:
            if sha(self.root/file) != self.meta[key]: raise ValueError('Prepared candidate checksum changed')
        g,self.m,self.l,self.s,_,_ = modules()
        if {x.__name__:sha(x.__file__) for x in modules()} != self.meta['native_source_sha256']:
            raise ValueError('Prepared native implementation changed')
        work = Path(work); work.mkdir(parents=True, exist_ok=False)
        self.project = g.G2Project(gpxfile=str(self.root/'prepared.gpx'), newgpx=str(work/'evaluation.gpx'))
        self.phase = self.project.phase(self.meta['phase']); self.hist = self.project.histogram(self.meta['histogram'])
        with np.load(self.root/'basis.npz', allow_pickle=False) as data:
            self.basis = {k:data[k].copy() for k in data.files}
        self.initial_cell = list(self.phase.data['General']['Cell'][1:7])
        self.a_scale = float(np.mean(self.basis['baseline_a'][:3]))
        self.histogram_scale = float(self.hist.data['Sample Parameters']['Scale'][0])
        if not np.isfinite(self.histogram_scale) or self.histogram_scale <= 0:
            raise ValueError('Structural log-scale fitting requires a finite positive source histogram scale')
        self.background0 = float(self.hist.data['Background'][0][3])
        self.observation_limits = copy.deepcopy(self.hist.data['Limits'])

    def evaluate(self, amplitudes, cell_offsets=None, *, log_scale=0., background0_delta=0.):
        q = np.asarray(amplitudes, float)
        if q.shape != (self.basis['physical_basis'].shape[1],) or not np.isfinite(q).all():
            raise ValueError('Invalid physical-mode coordinates')
        if cell_offsets is None: cell_offsets = np.zeros(self.basis['cell_basis'].shape[1])
        c = np.asarray(cell_offsets, float)
        if c.shape != (self.basis['cell_basis'].shape[1],) or not np.isfinite(c).all():
            raise ValueError('Invalid cell coordinates')
        a = self.basis['baseline_a'] + self.a_scale * (self.basis['cell_basis'] @ c)
        reciprocal = np.array([[a[0], a[3]/2, a[4]/2],
                               [a[3]/2, a[1], a[5]/2], [a[4]/2, a[5]/2, a[2]]])
        if np.linalg.eigvalsh(reciprocal).min() <= 0:
            raise ValueError('Proposed reciprocal metric is not positive definite')
        cell = self.l.A2cell(a); metric(cell)
        if not np.isfinite(log_scale) or abs(log_scale) > 100 or not np.isfinite(background0_delta):
            raise ValueError('Invalid histogram scale/background perturbation')
        self.hist.data['Sample Parameters']['Scale'][0] = self.histogram_scale * np.exp(log_scale)
        self.hist.data['Background'][0][3] = self.background0 + background0_delta
        self.phase.data['General']['Cell'][1:] = list(cell) + [float(self.l.calc_V(a))]
        apply_modes(self.phase.data, self.basis['physical_basis'] @ q, self.m)
        return curve(self.project, self.hist)

    def bind_observation(self, observation):
        """Install current observations without changing the prepared native grid."""
        data = self.hist.data['data'][1]
        if not np.array_equal(observation.x, data[0]):
            raise ValueError('Observation grid differs from the prepared native grid')
        from .reduced import native_limits
        if not hasattr(self,'bound_observation_limits') or self.hist.data['Limits']!=self.bound_observation_limits:
            self.observation_limits=copy.deepcopy(self.hist.data['Limits'])
        self.hist.data['Limits']=native_limits(observation,self.observation_limits)
        self.bound_observation_limits=copy.deepcopy(self.hist.data['Limits'])
        # Exclusions suppress the likelihood, not the saved measured intensity
        # or its native Rwp normalization. Do not erase excluded strong peaks.
        valid = observation.measurement_mask
        data[1] = np.ma.array(observation.y, mask=~valid)
        data[2] = np.divide(1., observation.variance, out=np.zeros_like(observation.variance), where=valid)
        self.project.data['Covariance']['data'] = {}

    def save_verified_prediction(self, prediction):
        """Export current parameters and observations with a fresh computed curve."""
        y = curve(self.project, self.hist)
        if not np.allclose(y, prediction, rtol=1e-9, atol=1e-10):
            raise ValueError('Current native state does not match the requested plotted prediction')
        data = self.hist.data['data'][1]; data[3] = y; data[5] = np.asarray(data[1])-y
        refresh_profile_metadata(self.project, self.hist, y)
        self.project.data['Covariance']['data'] = {}; self.project.save()
        return dict(path=str(self.project.filename), sha256=sha(self.project.filename),
                    covariance_cleared=True, native_curve_verified=True,
                    interpretation='conditional numerical estimate, not native least-squares covariance')
