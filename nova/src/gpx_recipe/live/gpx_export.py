"""Materialize a conditional posterior maximum without claiming native refinement.

The construction GPX is only a starting model. A separately named checkpoint
contains the selected posterior parameters and the exact snapshot observations.
Only a fresh native profile matching the displayed posterior may be published.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import numpy as np

from .contracts import file_hash


def apply_magnetic_parameters(project, native, parameters):
    """Map all independent posterior coordinates into the native magnetic sites."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != native.x0.shape or not np.isfinite(parameters).all():
        raise ValueError('Posterior parameter dimensions differ from the native model')
    if any('::AM' not in key and not key.startswith('::constr') for key in native.vary):
        raise ValueError('Posterior GPX export only accepts independent magnetic variables')
    before = copy.deepcopy(native.parms)
    native.profile(parameters)
    allowed = set(native.vary)
    for phase in native.histo_phases[1].values():
        if phase['General']['Type'] == 'magnetic':
            for index in range(len(phase['Atoms'])):
                allowed.update(f"{phase['pId']}::AM{axis}:{index}" for axis in 'xyz')
    for key, value in before.items():
        if key in allowed:
            continue
        try:
            unchanged = np.allclose(value, native.parms[key], rtol=0, atol=1e-10, equal_nan=True)
        except (TypeError, ValueError):
            unchanged = value == native.parms[key]
        if not unchanged:
            raise ValueError('Posterior changes a fixed native parameter: ' + key)
    mapped = []
    for name, phase in native.histo_phases[1].items():
        if phase['General']['Type'] != 'magnetic':
            continue
        target = project.phase(name)
        cx = target.data['General']['AtomPtrs'][0]
        if len(target.data['Atoms']) != len(phase['Atoms']):
            raise ValueError('Posterior magnetic sites differ from the saved project')
        for index in range(len(phase['Atoms'])):
            moment = [float(native.parms[f"{phase['pId']}::AM{axis}:{index}"]) for axis in 'xyz']
            target.data['Atoms'][index][cx + 4:cx + 7] = moment
            mapped.append(dict(phase=name, atom=index, components=moment))
    if not mapped:
        raise ValueError('No posterior magnetic parameters were materialized')
    # A conditional Monte Carlo estimate must not inherit a seed's covariance.
    project.data.setdefault('Covariance', {})['data'] = {}
    return mapped


def apply_snapshot_observations(project, request):
    """Preserve the original fitted masks while installing this cumulative revision."""
    if not request.get('observation'):
        return
    if file_hash(request['observation']) != request['observation_sha256']:
        raise ValueError('Posterior snapshot observation changed')
    with np.load(request['observation'], allow_pickle=False) as z:
        observation = {key: z[key].copy() for key in ('x', 'y', 'sigma', 'mask')}
    histograms = project.histograms()
    if len(histograms) != 1:
        raise ValueError('Snapshot export requires one independently analyzed histogram')
    arrays = histograms[0].data['data'][1]
    x = np.asarray(np.ma.getdata(arrays[0]), dtype=float)
    if x.shape != observation['x'].shape or not np.allclose(x, observation['x'], rtol=1e-12, atol=1e-8):
        raise ValueError('Snapshot cannot be represented on this GPX grid')
    old_mask = np.ma.getmaskarray(arrays[0]) | (np.ma.filled(arrays[2], 0) <= 0)
    invalid = old_mask | observation['mask']
    if not np.isfinite(observation['y']).all() or np.any(observation['sigma'][~invalid] <= 0):
        raise ValueError('Invalid posterior snapshot intensities or uncertainties')
    weight = np.zeros_like(x)
    np.divide(1., observation['sigma'] ** 2, out=weight, where=~invalid)
    arrays[1] = np.ma.array(observation['y'], mask=np.ma.getmaskarray(arrays[1]))
    arrays[2] = np.ma.array(weight, mask=np.ma.getmaskarray(arrays[2]))
    histograms[0].data['data'][0]['Dummy'] = False


def export_posterior_gpx(directory, request, *, backend_active=False):
    """Create and verify a GPX at the plotted posterior maximum, atomically."""
    from contextlib import nullcontext
    from GSASII import GSASIIscriptable as G
    from .native_profile import NativeReducedProfile
    from .native_validation import refresh_saved_profiles
    from .validated_backend import active_identity, corrected_backend

    root = Path(directory)
    # Withhold any old export receipt until the new posterior is verified.
    (root/'posterior-gpx.json').unlink(missing_ok=True)
    source = root / 'magnetic-refinement.gpx'
    source_sha = file_hash(source)
    with np.load(root / 'posterior.npz', allow_pickle=False) as z:
        samples, loglike = z['samples'], z['loglike']
        best = samples[int(np.argmax(loglike))].copy()
    with np.load(root / 'kernel.npz', allow_pickle=False) as z:
        x, y, sigma = (z[key].copy() for key in ('x', 'y', 'sigma'))
        expected = z['base'] + np.einsum('i,ijb,j->b', best, z['quadratic'], best)
    if backend_active and active_identity() is None:
        raise ValueError('Posterior GPX export requires the calculation backend')
    destination = root / 'posterior-maximum.gpx'
    temporary = root / 'posterior-maximum.pending.gpx'
    with (nullcontext() if backend_active else corrected_backend()):
        project = G.G2Project(str(source), newgpx=str(temporary))
        native = NativeReducedProfile(source)
        mapped = apply_magnetic_parameters(project, native, best)
        apply_snapshot_observations(project, request)
        project.save()
        refresh_saved_profiles(project)
        project.save()
        # A fresh adapter reopens the artifact: no cached arrays may establish
        # equivalence between the saved checkpoint and the plotted posterior.
        check = NativeReducedProfile(temporary)
        calculated = check.profile(check.x0)
        native_x = np.concatenate([check.histo_phases[0][name]['Data'][0][sl] for name, sl in check.parts])
        native_y = np.concatenate([check.histo_phases[0][name]['Data'][1][sl] for name, sl in check.parts])
        weight = np.concatenate([check.original_weights[name][sl] for name, sl in check.parts])
        valid = np.isfinite(native_x) & (weight > 0)
        if x.shape != native_x[valid].shape or not np.allclose(x, native_x[valid], rtol=1e-12, atol=1e-8):
            raise ValueError('Exported posterior GPX has a different fitted grid')
        if not np.allclose(y, native_y[valid], rtol=1e-12, atol=1e-10) or not np.allclose(sigma, 1/np.sqrt(weight[valid]), rtol=1e-12, atol=1e-10):
            raise ValueError('Exported posterior GPX does not contain the plotted observations')
        difference = np.abs(expected - calculated[valid])
        error = float(np.max(difference) / max(float(np.max(np.abs(expected))), 1e-12))
        error_sigma = float(np.max(difference / sigma))
        if error > 1e-7 or error_sigma > 1e-5:
            raise ValueError('Exported posterior GPX differs from the plotted native profile')
    if file_hash(source) != source_sha:
        raise ValueError('Original magnetic construction project changed during export')
    os.replace(temporary, destination)
    result = dict(schema='radar-conditional-posterior-gpx/v1', status='conditional_checkpoint',
                  starting_gpx=source.name, starting_gpx_sha256=source_sha,
                  fitted_gpx=destination.name, fitted_gpx_sha256=file_hash(destination),
                  kernel_sha256=file_hash(root/'kernel.npz'), posterior_sha256=file_hash(root/'posterior.npz'),
                  source_revision=request['source_revision'], reference_revision=request['reference_revision'],
                  native_profile_relative_error=error, native_profile_max_error_sigma=error_sigma,
                  magnetic_sites=mapped, native_convergence_declared=False, covariance_supplied=False,
                  scope='Posterior maximum conditional on the fixed nuclear response; not a native least-squares convergence claim')
    receipt = root/'posterior-gpx.json'
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
