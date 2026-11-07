"""Observation-independent identity of a structural native forward model."""
import hashlib
import json
import re
import numpy as np

from .identity import _plain, native_identity, BOOKKEEPING_CONTROLS, without_recorded_conditions


def _physics_record(source):
    """Describe the anchor physics, independently of current evaluation state."""
    native = source.fast.native
    native._check()
    histograms, phases = native.histo_phases[:2]
    selected = source.fast.slow.phase.name
    histogram = histograms[source.fast.slow.hist.name]
    hid = str(histogram['hId'])
    names = sorted(phases, key=lambda name: (name != selected, name))
    aliases = {str(phases[name]['pId']): ('primary' if name == selected else 'secondary-'+str(i))
               for i, name in enumerate(names)}

    def key(value):
        text = str(value)
        match = re.match(r'^(\d*):(\d*):(.*)$', text)
        if not match:
            return text
        phase, hist, rest = match.groups()
        return ':'.join((aliases.get(phase, phase), 'bank' if hist == hid else hist, rest))

    def canonical(value):
        if isinstance(value, dict):
            result = {key(k): canonical(v) for k, v in value.items()}
            if len(result) != len(value):
                raise ValueError('Ambiguous canonical structural parameter keys')
            return result
        if isinstance(value, (tuple, list)):
            return [canonical(v) for v in value]
        return value

    # Only a copy is mapped to x0. Hashing after an optimizer call must not
    # depend on the last evaluated mode/cell, or disturb that live context.
    parameters = native.parms.copy()
    parameters.update(zip(native.vary, native.x0))
    native.maps.Dict2Map(parameters)
    phase_records = []
    for name in names:
        phase = phases[name]
        general = phase['General']
        cx, ct, _, cia = general['AtomPtrs']
        phase_records.append(dict(role=aliases[str(phase['pId'])],
            type=general['Type'], symmetry=general['SGData'],
            isotope=general.get('Isotope'), lande_g=general.get('Lande g'),
            atoms=[dict(label=a[ct-1], species=a[ct], coordinates=a[cx:cx+4],
                        adp=a[cia:cia+8]) for a in phase['Atoms']],
            histograms={('bank' if n == source.fast.slow.hist.name else n): v
                        for n, v in phase['Histograms'].items()}))
    # atomIndx is the derived random-ID -> (phase, atom index) lookup. The
    # canonical phase/atom records below carry its complete physical order.
    ignored = BOOKKEEPING_CONTROLS | {'atomIndx', ':'+hid+':wtFactor'}
    content = dict(schema='radar-structural-forward-identity/v1',
        runtime=native_identity(native), parameters=canonical(
            without_recorded_conditions(parameters,prefix=':'+hid+':')),
        controls=canonical({k: v for k, v in native.controls.items() if k not in ignored}),
        histogram_dictionary=canonical(native.histdict), phases=phase_records,
        instrument=histogram['Instrument Parameters'], background=histogram['Background'],
        sample=without_recorded_conditions(
            {k: v for k, v in histogram['Sample Parameters'].items() if k != 'ranId'}),
        grid=np.asarray(source.fast.x, float),
        # Until a caller constructs an explicitly unmasked rendering context,
        # native exclusions/masks are also rendering dependencies: its saved
        # calculated arrays can be zero there. Never reuse such masked curves
        # as if they were a full-grid forward response.
        rendering_masks=[np.ma.getmaskarray(a) for a in histogram['Data'][:3]],
        fit_slice=[source.fast.slice.start, source.fast.slice.stop, source.fast.slice.step],
        basis=source.fast.basis, radii=source.radii, lift=source.lift,
        chemistry=source.chemistry.fingerprint)
    # Native metadata records include non-physical refinement flag booleans,
    # but no observations, variances, covariance matrix or random object IDs.
    return _plain(content)


def structural_physics_identity(source):
    """Hash the anchor physics and basis, not mutable evaluation state or IDs."""
    encoded = json.dumps(_physics_record(source), sort_keys=True, allow_nan=False, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()
