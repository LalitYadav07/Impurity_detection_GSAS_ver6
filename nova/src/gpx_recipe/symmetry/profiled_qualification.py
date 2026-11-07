"""Owned-process reuse of complete magnetic response qualification.

Only successful native checks performed by this process enter the ledger.
Nothing is serialized or accepted from an imported preparation. New native
contexts still evaluate an offset and a nonzero probe; retained solutions
keep their independent native, Jacobian and saved-GPX checks.
"""
from collections import OrderedDict
import hashlib
import json
import os
import pickle
import threading

import numpy as np

MAX_ENTRIES = 64
_pid = os.getpid()
_entries = OrderedDict()
_lock = threading.Lock()


def _plain(value):
    from ..preparation.identity import _plain as ordinary
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(np.ma.getdata(value))
        if data.dtype.hasobject:
            raise ValueError('Object arrays cannot qualify a native response')
        return dict(dtype=data.dtype.str, shape=list(data.shape),
            bytes_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
            mask_sha256=hashlib.sha256(np.ma.getmaskarray(value).tobytes()).hexdigest())
    if isinstance(value, dict):
        names = [str(k) for k in value]
        if len(set(names)) != len(names):
            raise ValueError('Ambiguous response qualification keys')
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return ordinary(value)


def identity(context, model):
    """Bind actual decoded arrays and native dependencies, not a cache label.

    Unknown metadata/runtime conventions decline this optional reuse. They
    continue through the original complete native qualification.
    """
    from ..preparation.identity import native_identity, BOOKKEEPING_CONTROLS
    from ..preparation.kernel_runtime import floating_state
    from .compiler import package_signature
    try:
        context.qualification_reuse_note = ''
        context._check()
        floating = floating_state()
        if floating is None:
            context.qualification_reuse_note = 'Native arithmetic controls are unavailable'
            return None
        parms = context.parms.copy()
        parms.update(zip(context.vary, context.initial))
        context.maps.Dict2Map(parms)
        represented = set(context.vary)
        represented.update(key for _, _, keys in context.atoms for key in keys)
        controls = {k: v for k, v in context.args[4].items()
                    if k not in BOOKKEEPING_CONTROLS | {'atomIndx'}}
        # Native random atom IDs only index GetAtomFXU's table. Its physical
        # ordering, fixed parameters and the complete moment map are explicit.
        modules = (context.math, context.math.G2pwd, context.math.G2lat, context.math.G2el)
        functions = [{name: id(value) for name, value in vars(module).items()
                      if callable(value)} for module in modules]
        # Ownership checks also bind container addresses, which necessarily
        # differ between new adapters. Here bind the complete mapping values.
        # These trusted native objects are hashed only, never deserialized.
        mapping_names = ('dependentParmList', 'indParmList', 'arrayList',
            'invarrayList', 'offsetDict', 'holdParmList', 'holdParmType')
        constraints = hashlib.sha256(pickle.dumps(tuple(
            (name, getattr(context.maps, name, None)) for name in mapping_names),
            protocol=5)).hexdigest()
        histogram = context.histogram
        content = dict(schema='owned-magnetic-response-qualification/v1',
            implementation=package_signature(), runtime=native_identity(context),
            floating=floating, active_functions=functions,
            parameters={k: v for k, v in parms.items() if k not in represented},
            controls=controls, histdict=context.args[2],
            phase_symmetry={name: phase['General']['SGData']
                            for name, phase in context.phases.items()},
            reflection_inventory={name: values['RefList'][:, :5]
                                  for name, values in histogram['Reflection Lists'].items()},
            observations=histogram['Data'][:3], limits=histogram['Limits'],
            constraints=constraints, variables=context.vary,
            moment_atoms=context.atoms, moment_maps=context.component_maps,
            cartesian_maps=context.cartesian_maps,
            coefficients={name: getattr(model, name) for name in
                          ('offset', 'nuclear', 'quadratic', 'background')},
            maximum_moment=model.maximum_moment, fixed_scale=model.fixed_scale,
            normalization=model.normalization)
        plain = _plain(content)
        context._qualification_components = {name: hashlib.sha256(json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()
            for name, value in plain.items()}
        raw = json.dumps(plain, sort_keys=True, allow_nan=False,
                         separators=(',', ':')).encode()
        return hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        try: context.qualification_reuse_note = str(error)[:300]
        except AttributeError: pass
        return None


def _owned():
    global _pid, _entries, _lock
    if os.getpid() != _pid:
        # Do not acquire a possibly inherited locked mutex after a fork.
        _pid = os.getpid()
        _entries = OrderedDict()
        _lock = threading.Lock()


def prior(key, checks):
    _owned()
    if key is None:
        return None
    with _lock:
        values = _entries.get(key)
        if values is None or len(values) < checks:
            return None
        _entries.move_to_end(key)
        return list(values)


def remember(key, errors):
    _owned()
    if (key is None or not errors or not np.isfinite(errors).all()
            or max(errors) > 1e-8):
        return
    with _lock:
        if len(_entries.get(key, ())) <= len(errors):
            _entries[key] = tuple(float(e) for e in errors)
        _entries.move_to_end(key)
        while len(_entries) > MAX_ENTRIES:
            _entries.popitem(last=False)


def forget(key):
    _owned()
    with _lock:
        _entries.pop(key, None)
