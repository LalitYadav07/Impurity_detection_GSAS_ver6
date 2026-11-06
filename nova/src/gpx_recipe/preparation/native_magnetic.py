"""Compile observation-independent physics through the shared native engine."""
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from .magnetic import MagneticResponse, MAX_BYTES
from .identity import _plain, native_identity, BOOKKEEPING_CONTROLS
from ..live.moment_basis import from_native
from ..live.native_kernel_policy import compile_native_magnetic_kernel
from ..live.quadratic_qualification import check_coefficients,compare_coefficients


def _context(native):
    native._check()
    histograms, phases = native.histo_phases[:2]
    names = [name for name in histograms if name.startswith('PWDR')]
    # errRefine adds scalar whole-refinement summaries beside the histograms.
    if len(names) != 1 or len(native.parts) != 1:
        raise ValueError('Preparation currently requires one independently calibrated histogram')
    histogram = histograms[names[0]]
    hfx = f":{histogram['hId']}:"
    if native.controls[hfx+'histType'] not in {'PNC', 'PNT'}:
        raise ValueError('Preparation requires a reviewed PNC/PNT native profile')
    for phase in phases.values():
        phfx=f"{phase['pId']}:{histogram['hId']}:"
        if (native.parms.get(phfx+'Extinction',0.)!=0.
                or native.parms.get(phfx+'LeBail',False)
                or phase['General'].get('doPawley')):
            raise ValueError('Intensity-dependent or extracted-intensity corrections require full native evaluation, not quadratic preparation')
    background = [k for k in native.vary if k.startswith(hfx+'Back;')]
    scale = hfx+'Scale'
    moments = [k for k in native.vary if '::AM' in k or k.startswith('::constr')]
    if (not 1 <= len(moments) <= 24
            or set(native.vary) != set(moments+background+([scale] if scale in native.vary else []))):
        raise ValueError('Only the complete bounded moment basis and declared scale/background may vary')
    return histogram, phases, hfx, moments, background, scale


def _physics_record(native, basis=None):
    histogram, phases, hfx, moments, background, scale = _context(native)
    if basis is None:
        basis = moment_basis(native)
    dynamic = set(moments+background+[scale]) | {k for _, _, keys in basis.atoms for k in keys}
    ignored = BOOKKEEPING_CONTROLS | {'atomIndx'} | {
        hfx+'Exclude', hfx+'wtFactor'}
    # Native random atom IDs and display histogram/phase names are not
    # scattering inputs. Preserve exact site and variable order, while
    # describing their parameter references through deterministic roles.
    names = {name: ('magnetic' if p['General']['Type'] == 'magnetic' else 'nuclear')+'-'+str(i)
             for i, (name, p) in enumerate(phases.items())}
    ids = {str(p['pId']): names[name] for name, p in phases.items()}
    histogram_name = native.parts[0][0]
    hid = str(histogram['hId'])

    def key(value):
        text = str(value)
        if text in names:
            return names[text]
        if text == histogram_name:
            return 'bank'
        match = re.match(r'^(\d*):(\d*):(.*)$', text)
        if match:
            p, h, rest = match.groups()
            return ':'.join((ids.get(p, p), 'bank' if h == hid else h, rest))
        return text

    def canonical(value):
        if isinstance(value, dict):
            result = {key(k): canonical(v) for k, v in value.items()}
            if len(result) != len(value):
                raise ValueError('Ambiguous magnetic physical parameter roles')
            return result
        if isinstance(value, (list, tuple)):
            return [canonical(v) for v in value]
        return value

    content = dict(runtime=native_identity(native),
        parameters=canonical({k: v for k, v in native.parms.items() if k not in dynamic}),
        controls=canonical({k: v for k, v in native.controls.items() if k not in ignored}),
        histogram_dictionary=canonical(native.histdict),
        grid=np.asarray(np.ma.getdata(histogram['Data'][0]), float),
        phases={names[name]: dict(general={k: v for k, v in p['General'].items() if k not in {'Name', 'Mydir'}},
            histograms=canonical(p['Histograms'])) for name, p in phases.items()},
        variables=dict(moments=[key(k) for k in moments], background=[key(k) for k in background]),
        atoms=[(names[name], index, [key(k) for k in keys]) for name, index, keys in basis.atoms],
        components=basis.components, cartesian=basis.cartesian)
    return _plain(content)


def physics_identity(native, basis=None):
    encoded = json.dumps(_physics_record(native, basis), sort_keys=True, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


def moment_basis(native):
    _, _, _, moments, background, scale = _context(native)
    return from_native(native, native.x0, [native.vary.index(k) for k in moments],
        nuisance=background+[scale])


def qualify_response(response, native):
    """Fresh native forward checks, including after importing a theory file."""
    histogram, _, hfx, moments, background, scale = _context(native)
    if (native.controls.get(hfx+'Exclude')
            or any(np.ma.getmaskarray(a).any() for a in histogram['Data'][:3])):
        raise ValueError('Qualify on the unmasked physical grid, then bind measurement exclusions')
    basis = moment_basis(native)
    if physics_identity(native, basis) != response.physics_sha256:
        raise ValueError('Prepared physics changed; adapt or rebuild this response')
    if not np.array_equal(response.moment_maps, basis.cartesian):
        raise ValueError('Prepared physical moment constraints differ from the native basis')
    _, sl = native.parts[0]
    if not np.array_equal(histogram['Data'][0][sl], response.x):
        raise ValueError('Native qualification grid differs from the prepared response')
    indices = [native.vary.index(k) for k in moments]
    backgrounds = [native.vary.index(k) for k in background]
    original = native.x0.copy()
    source_scale = native.parms[scale]
    scale_index = native.vary.index(scale) if scale in native.vary else None
    errors, signal_errors = [], []
    rng = np.random.default_rng(30917)
    try:
        for _ in range(7):
            point = original.copy()
            point[indices] = rng.normal(size=len(indices))*2.
            point[backgrounds] = rng.normal(size=len(backgrounds))
            common_scale = float(rng.uniform(.3, 3.))
            native.parms[scale] = common_scale
            if scale_index is not None:
                point[scale_index] = common_scale
            direct = native.profile(point)
            predicted = response.predict(point[indices], scale=common_scale, background=point[backgrounds])
            difference = float(np.max(abs(direct-predicted)))
            baseline = response.predict(np.zeros(len(indices)), scale=common_scale, background=point[backgrounds])
            errors.append(difference/max(float(np.max(abs(direct))), 1e-12))
            signal_errors.append(difference/max(float(np.max(abs(direct-baseline))), 1e-12))
        if not np.isfinite(errors+signal_errors).all() or max(errors+signal_errors) > 1e-8:
            raise ValueError('Prepared forward response disagrees with independent native profiles')
        native.parms[scale]=1.
        def derivatives(moment):
            point=original.copy();point[indices]=moment;point[backgrounds]=0.
            if scale_index is not None:point[scale_index]=1.
            _,jacobian=native.profile(point,jacobian=True)
            return jacobian[indices]
        # Full powder Jacobians repeat the peak rendering and numerical
        # derivatives of every native moment at every basis direction. Use
        # complete native reflection F² checks plus one native peak response
        # where supported; retain the full-Jacobian oracle as automatic fallback.
        from ..live.native_reflection import compile_profile_reflections,ReflectionCompilerUnavailable
        anchor=original.copy();anchor[indices]=0.;anchor[backgrounds]=0.
        if scale_index is not None:anchor[scale_index]=1.
        try:
            checked=compile_profile_reflections(native,anchor,indices,verify_coefficients=True)
        except ReflectionCompilerUnavailable as error:
            coefficients=check_coefficients(response.quadratic,derivatives)
            coefficients['fallback_reason']=str(error)
        else:
            rebuilt=checked['quadratic'].transpose(2,0,1)
            comparison=compare_coefficients(response.quadratic,rebuilt)
            coefficients=dict(checked['diagnostics']['coefficient_validation'],
                prepared_tensor_comparison=comparison,profile_capture=checked['diagnostics']['profile_capture'])
        response._seal_native()
        return dict(independent_total_errors=errors, independent_magnetic_errors=signal_errors,
            coefficient_validation=coefficients)
    finally:
        native.parms[scale] = source_scale
        native.profile(original)


def compile_response(native):
    """Compile from a real or theory-only native model, never from its residual.

    Mask-free preparation covers the planned grid; measured fit masks and
    uncertainties are supplied later. No dummy observation is constructed or
    stored by this function, and source observations remain unchanged.
    """
    began = time.monotonic()
    histogram, _, hfx, moments, background, scale = _context(native)
    if (native.controls.get(hfx+'Exclude')
            or any(np.ma.getmaskarray(a).any() for a in histogram['Data'][:3])):
        raise ValueError('Prepare the unmasked physical grid; apply measurement exclusions when binding')
    name, sl = native.parts[0]
    x = np.asarray(histogram['Data'][0][sl], float).copy()
    indices = [native.vary.index(k) for k in moments]
    backgrounds = [native.vary.index(k) for k in background]
    if len(x)*(len(indices)**2+len(background)+3)*8 > MAX_BYTES:
        raise ValueError('Preparation exceeds its numeric response budget')
    basis = moment_basis(native)
    identity = physics_identity(native, basis)
    original = native.x0.copy()
    source_scale = native.parms[scale]
    reference = original.copy()
    reference[indices] = 0.
    reference[backgrounds] = 0.
    scale_index = native.vary.index(scale) if scale in native.vary else None

    def profile(point, intensity_scale):
        values = point.copy()
        native.parms[scale] = intensity_scale
        if scale_index is not None:
            values[scale_index] = intensity_scale
        return native.profile(values)

    try:
        if scale_index is not None:
            reference[scale_index] = 1.
        native.parms[scale] = 1.
        compiled = compile_native_magnetic_kernel(native, reference, indices)
        fixed = profile(reference, 0.)
        nuclear = compiled['base']-fixed
        columns = []
        for index in backgrounds:
            point = reference.copy()
            point[index] = 1.
            columns.append(profile(point, 0.)-fixed)
        response = MagneticResponse(x=x, fixed=fixed, nuclear=nuclear,
            quadratic=compiled['quadratic'].transpose(2, 0, 1),
            background=np.column_stack(columns) if columns else np.empty((len(x), 0)),
            moment_maps=basis.cartesian, physics_sha256=identity, proof={})
        validation = qualify_response(response, native)
        final = MagneticResponse(**{k: getattr(response, k) for k in
            ('x', 'fixed', 'nuclear', 'quadratic', 'background', 'moment_maps')},
            physics_sha256=identity, proof=dict(compiler=compiled['diagnostics'],
                **validation,
                observations_included=False, seconds=time.monotonic()-began,
                geometry_scope='Exact prepared geometry, bank and native implementation'))
        final._seal_native()
        return final
    finally:
        native.parms[scale] = source_scale
        native.profile(original)
