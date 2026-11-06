"""Compile observation-independent physics through the shared native engine."""
import hashlib
import copy
import json
from pathlib import Path
import re
import time
from contextlib import contextmanager

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


def qualify_response(response, native, *, factor_cache=None):
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
            checked=compile_profile_reflections(native,anchor,indices,verify_coefficients=True,
                factor_cache=factor_cache)
        except ReflectionCompilerUnavailable as error:
            coefficients=check_coefficients(response.quadratic,derivatives)
            coefficients['fallback_reason']=str(error)
        else:
            rebuilt=checked['quadratic'].transpose(2,0,1)
            comparison=compare_coefficients(response.quadratic,rebuilt)
            coefficients=dict(checked['diagnostics']['coefficient_validation'],
                prepared_tensor_comparison=comparison,profile_capture=checked['diagnostics']['profile_capture'],
                scattering_identity=checked['diagnostics']['scattering_identity'],
                scattering_factors_reused=checked['diagnostics']['scattering_factors_reused'],
                scattering_qualification_reused=checked['diagnostics']['scattering_qualification_reused'],
                scattering_authenticated_origin_reused=checked['diagnostics'].get('scattering_authenticated_origin_reused',False))
        response._seal_native()
        return dict(independent_total_errors=errors, independent_magnetic_errors=signal_errors,
            coefficient_validation=coefficients)
    finally:
        native.parms[scale] = source_scale
        native.profile(original)


def compile_response(native, *, factor_cache=None):
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
        compiled = compile_native_magnetic_kernel(native, reference, indices, factor_cache=factor_cache)
        fixed = profile(reference, 0.)
        nuclear = compiled['base']-fixed
        def original_background():
            columns=[]
            for index in backgrounds:
                point=reference.copy();point[index]=1.
                columns.append(profile(point,0.)-fixed)
            return np.column_stack(columns) if columns else np.empty((len(x),0))
        background_proof=dict(method='native full powder',columns=len(backgrounds))
        if backgrounds:
            from ..live.native_background import compile_columns,BackgroundCompilerUnavailable
            try:
                columns,background_proof=compile_columns(math=native.math,histogram=histogram,
                    parameters=native.parms,controls=native.controls,axis=x,names=background)
            except BackgroundCompilerUnavailable as error:
                columns=original_background();background_proof['fallback_reason']=str(error)
        else:columns=original_background()
        def make_response(columns):
            return MagneticResponse(x=x,fixed=fixed,nuclear=nuclear,
                quadratic=compiled['quadratic'].transpose(2,0,1),background=columns,
                moment_maps=basis.cartesian,physics_sha256=identity,proof={})
        response=make_response(columns)
        try:validation=qualify_response(response,native,factor_cache=factor_cache)
        except ValueError as error:
            if background_proof['method']!='unchanged native background function':raise
            response=make_response(original_background())
            background_proof=dict(method='native full powder',columns=len(backgrounds),
                fallback_reason='Background-only compilation failed native qualification: '+str(error))
            validation=qualify_response(response,native,factor_cache=factor_cache)
        final = MagneticResponse(**{k: getattr(response, k) for k in
            ('x', 'fixed', 'nuclear', 'quadratic', 'background', 'moment_maps')},
            physics_sha256=identity, proof=dict(compiler=compiled['diagnostics'],
                background_compiler=background_proof,
                **validation,
                observations_included=False, seconds=time.monotonic()-began,
                geometry_scope='Exact prepared geometry, bank and native implementation'))
        final._seal_native()
        return final
    finally:
        native.parms[scale] = source_scale
        native.profile(original)


@contextmanager
def _unmasked_physical_context(native):
    """Temporarily separate measurement exclusions from physical rendering.

    Keep all actual counts, weights and input objects for restoration. This
    does not create a measurement or change the source GPX. New objectives
    still apply the actual masks and weights after rendering.
    """
    native._check()
    saved = []
    try:
        for name, _ in native.parts:
            histogram = native.histo_phases[0][name]
            key = f":{histogram['hId']}:Exclude"
            original_data = histogram['Data']
            saved.append((histogram, original_data, key, key in native.controls, native.controls.get(key)))
            # Data can be one stacked MaskedArray. Saving row views and then
            # assigning into the original container would also clear the
            # saved masks through shared storage. Own a separate container.
            physical_data = copy.deepcopy(original_data)
            for i, value in enumerate(original_data[:3]):
                physical_data[i] = np.asarray(np.ma.getdata(value)).copy()
            histogram['Data'] = physical_data
            native.controls.pop(key, None)
        yield
    finally:
        for histogram, original_data, key, present, exclusion in saved:
            histogram['Data'] = original_data
            if present:
                native.controls[key] = exclusion
            else:
                native.controls.pop(key, None)


def adapt_response(prepared, native, *, factor_cache=None):
    """Automatically render and check the current scan's physical context.

    A changed cell, structure, calibration or grid is never represented by
    shifting the old curve. Exact unchanged scattering dependencies can reuse
    qualified factors while native peak rendering is performed freshly.
    Observations are bound separately after this operation.
    """
    began = time.monotonic()
    if not isinstance(prepared, MagneticResponse):
        raise ValueError('A prepared magnetic response is required for adaptation')
    with _unmasked_physical_context(native):
        current = physics_identity(native)
        if current == prepared.physics_sha256:
            validation = qualify_response(prepared, native, factor_cache=factor_cache)
            response = prepared
            action = 'same_context_requalified'
            reuse = validation['coefficient_validation']
        else:
            response = compile_response(native, factor_cache=factor_cache)
            action = 'current_context_rendered'
            # Report reuse when adapting the old preparation, not the later
            # internal recheck of factors just created for this new context.
            reuse = response.proof['compiler'].get('native_reflection_diagnostics') or {}
    return response, dict(schema='radar-prepared-magnetic-adaptation/v1', action=action,
        previous_physics_sha256=prepared.physics_sha256, physics_sha256=response.physics_sha256,
        scattering_factors_reused=reuse.get('scattering_factors_reused', False),
        scattering_qualification_reused=reuse.get('scattering_qualification_reused', False),
        scattering_authenticated_origin_reused=reuse.get('scattering_authenticated_origin_reused',False),
        native_peak_profiles_rendered=True, observations_included=False, scientific_result=False,
        seconds=time.monotonic()-began)


def render_current_response(native,*,factor_cache=None):
    """Compile current physical scattering separately from measurement masks."""
    with _unmasked_physical_context(native):
        return compile_response(native,factor_cache=factor_cache)


def current_native_profiles(native,coefficients):
    """Evaluate a small finite start set in one owned native context.

    A short k survey need not pay quadratic compilation/qualification costs.
    This uses exact native forward calls and shares their expensive setup;
    optimizer/sampling consumers still use the qualified compiled response.
    """
    points=np.asarray(coefficients,float)
    _,_,_,moments,_,_=_context(native)
    if (points.ndim!=2 or points.shape[1]!=len(moments) or not 1<=len(points)<=64
            or not np.isfinite(points).all()):
        raise ValueError('Declare a bounded complete set of native moment starts')
    indices=[native.vary.index(k) for k in moments];original=native.x0.copy()
    histogram=native.histo_phases[0][native.parts[0][0]]
    x=np.asarray(histogram['Data'][0][native.parts[0][1]],float).copy()
    try:
        with _unmasked_physical_context(native):
            zero=original.copy();zero[indices]=0.;baseline=native.profile(zero)
            curves=[]
            for coefficients in points:
                value=original.copy();value[indices]=coefficients
                curves.append(native.profile(value)-baseline)
        return x,np.asarray(curves),dict(method='shared_native_profiles',
            native_forward_evaluations=len(points)+1,observations_included=False,no_refinement=True)
    finally:native.profile(original)
