"""Reuse certified unweighted magnetic bin responses in both native consumers.

The identity describes the original native forward calculation, not an
observation or a refinement recipe. Common histogram scale is applied after
loading; background, weights and exclusions are supplied by each consumer.
Changed physical dependencies keep the existing native compilation path.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .identity import BOOKKEEPING_CONTROLS, _plain, native_identity, without_recorded_conditions
from .magnetic import load


def identity(*, math, histogram, phases, histdict, parameters, controls, basis, diagnostics=None):
    """Exact conservative native dependencies of the moment-only response.

    Constraint helper values are omitted only because their complete expanded
    physical atom map is included. All fixed physical parameters remain. No
    cell rounding, profile interpolation or approximate context matching.
    """
    names = {name: ('magnetic' if p['General']['Type'] == 'magnetic' else 'nuclear')+'-'+str(i)
             for i, (name, p) in enumerate(phases.items())}
    ids = {str(p['pId']): names[name] for name, p in phases.items()}
    hid = str(histogram['hId']); hfx = ':'+hid+':'
    moment_keys = {k for _, _, keys in basis.atoms for k in keys}

    def key(value):
        text = str(value)
        if text in names: return names[text]
        parts = text.split(':', 2)
        if len(parts) == 3 and (not parts[0] or parts[0].isdigit()) and (not parts[1] or parts[1].isdigit()):
            return ':'.join((ids.get(parts[0], parts[0]), 'bank' if parts[1] == hid else parts[1], parts[2]))
        return text

    def canonical(value):
        if isinstance(value, dict):
            result = {key(k): canonical(v) for k, v in value.items()}
            if len(result) != len(value): raise ValueError('Ambiguous rendered magnetic parameter roles')
            return result
        if isinstance(value, (list, tuple)): return [canonical(v) for v in value]
        return value

    # Histogram names are display labels in the histogram dictionary too.
    histdict = {('bank' if k not in names and isinstance(k, str) and k.startswith('PWDR') else k): v
                for k, v in histdict.items()}
    omitted = BOOKKEEPING_CONTROLS | {'atomIndx', 'max cyc', hfx+'Exclude', hfx+'wtFactor'}
    runtime=native_identity(SimpleNamespace(math=math))
    from ..live.validated_backend import PINNED_PRODUCTION_STRMATH
    if runtime.get('native_sources',{}).get('GSASIIstrMath.py')==PINNED_PRODUCTION_STRMATH:
        # On this installed build corrected_backend replaces no functions;
        # its active flag is wrapper bookkeeping. Actual source, function and
        # binary identities remain pinned, including getPowderProfile.
        runtime.pop('active_backend',None)
    content = dict(schema='radar-native-rendered-magnetic-identity/v1',
        runtime=runtime,
        parameters=canonical(without_recorded_conditions(
            {k: v for k, v in parameters.items() if k not in moment_keys
            and not k.startswith('::constr') and k != hfx+'Scale' and not k.startswith(hfx+'Back;')},prefix=hfx)),
        controls=canonical({k: v for k, v in controls.items() if k not in omitted}),
        histogram_dictionary=canonical(histdict),
        axis=np.asarray(np.ma.getdata(histogram['Data'][0]), float),
        phases={names[name]: dict(general={k: v for k, v in p['General'].items() if k not in {'Name', 'Mydir'}},
            histograms={('bank' if k.startswith('PWDR') else k): canonical(v)
                        for k, v in p['Histograms'].items()}) for name, p in phases.items()},
        atoms=[(names[name], index, [key(k) for k in keys]) for name, index, keys in basis.atoms],
        components=basis.components, cartesian=basis.cartesian,
        reflections={names[name]: np.asarray(rows['RefList'])[:, :4]
            for name, rows in histogram['Reflection Lists'].items()})
    encoded = json.dumps(_plain(content), sort_keys=True, allow_nan=False, separators=(',', ':'))
    if diagnostics is not None:
        diagnostics.update({k:hashlib.sha256(json.dumps(_plain(v),sort_keys=True,allow_nan=False,
            separators=(',',':')).encode()).hexdigest() for k,v in content.items()})
    return hashlib.sha256(encoded.encode()).hexdigest()


def metadata(detail):
    """Older or fallback-only preparations retain ordinary factor reuse."""
    if not isinstance(detail,dict):return None
    proof=detail
    for field in ('proof','compiler','native_reflection_diagnostics'):
        if not isinstance(proof,dict):return None
        proof=proof.get(field)
    if not isinstance(proof,dict):return None
    value = proof.get('rendered_identity')
    if isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value):
        return dict(identity=value, physics_sha256=detail['physics_sha256'])
    return None


def read_certified(source, *, checksum, physics_sha256, rendered_identity, axis, moment_maps):
    """A native producer receipt authorizes bytes, never a library's claim.

    The current consumer separately supplies its exact context identity and
    retains independent native checks. This function cannot qualify a new
    geometry or turn a portable checksum into native scientific authority.
    """
    from .authority import magnetic_response_origin
    origin = magnetic_response_origin(checksum, physics_sha256)
    if origin is None: raise ValueError('Rendered response has no verified native producer')
    path = Path(source)
    if path.is_symlink():
        raise ValueError('Certified rendered response changed')
    response = load(path, physics_sha256=physics_sha256, expected_file_sha256=checksum)
    if (metadata(dict(physics_sha256=physics_sha256, proof=response.proof)) !=
            dict(identity=rendered_identity, physics_sha256=physics_sha256)
            or not np.array_equal(response.x, axis) or not np.array_equal(response.moment_maps, moment_maps)):
        raise ValueError('Certified rendered response differs from the current native context')
    # load hashes the exact bounded bytes decoded, so even an atomic path
    # replacement cannot authorize another response. No duplicate file reads
    # are needed after these private arrays have been reconstructed.
    return response.quadratic.copy(), origin
