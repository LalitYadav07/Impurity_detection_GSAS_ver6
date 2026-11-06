"""Reuse native-generated HKL envelopes across changing nuclear cells.

The current native integer bounds, six metric monomials, inclusion test and
sorting are retained. An envelope is usable only when a checked metric lower
bound proves it contains the current sphere. No reflection is pruned by its
intensity. Unknown native code and unsupported contexts keep native generation.
Only owned process memory is retained; this is not a portable trust assertion.
"""
from collections import Counter, OrderedDict
from contextlib import contextmanager
import hashlib
import inspect
import math
import pickle
import threading

import numpy as np


MAXIMUM_BYTES = 64 * 1024 * 1024
_LOCK = threading.Lock()
_ALPHA = .75
_SOURCES = {
    'GenHLaue': 'cfa42736b6c51b09df2f9294489e0bead2c0cbbc837d2ea48671fd276954b783',
    'calc_rDsq': 'f3ef6e439b5fcff28a10a1c49ed8bf14e57a52c2ba078c42ae2a4dc230513c01',
    'sortHKLd': 'b5db67261e251a282cc894a43f5ac991b29e11af60cfaee9bc101310944d40da',
    'getHKLmax': '0d2097d893d7b8e8b0eab0077e3b56456a1bc50f196adabebb06a03bfe570071',
    'MaxIndex': 'd5179d7f1c9f94f2981748be40aff9cdd2279250db52a0aa504b27bac3cd1e1a',
    'A2Gmat': '021eb5ca44d222e77f85d739cbaaee6e59be6e8e07241450749daa9d3a4acdff',
    'CentCheck': '9803b34b68d0f92e87374247af030a21421fb576158ab5bb09ea0f98a5b13833',
    'SwapIndx': 'cfbc382d712653ecf5c0720cbb8467a52b63ebe34e23e34068aa40cd2c0dbd80',
}
_SUPPORTED = {'-1', '2/m', 'mmm', '4/m', '6/m', '4/mmm', '6/mmm',
              '3m1', '31m', '3', 'm3', 'm3m'}


def _hash(function):
    try:
        return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    except (OSError, TypeError):
        return None


class NativeReflectionMemo:
    def __init__(self, maximum_bytes=MAXIMUM_BYTES, *, enabled=True):
        if type(maximum_bytes) is not int or not 0 <= maximum_bytes <= MAXIMUM_BYTES:
            raise ValueError('A bounded native reflection allocation is required')
        if type(enabled) is not bool:
            raise TypeError('Native reflection reuse selection must be boolean')
        self.maximum_bytes = maximum_bytes
        self.enabled = enabled
        self.records = OrderedDict()
        self.bytes = self.builds = self.hits = self.evictions = self.calls = 0
        self.bypasses = Counter()
        self.reviewed = None

    def clear(self):
        self.records.clear()
        self.bytes = 0

    def diagnostics(self):
        return dict(method='native HKL envelope with current metric and integer bounds',
                    enabled=self.enabled, calls=self.calls, builds=self.builds,
                    hits=self.hits, evictions=self.evictions, entries=len(self.records),
                    accounted_cache_bytes=self.bytes, maximum_cache_bytes=self.maximum_bytes,
                    bypasses=dict(self.bypasses), intensity_pruning=False,
                    profile_interpolation=False, persisted_predictions=False)

    @staticmethod
    def _order(hkl, symmetry, lattice):
        laue = symmetry['SGLaue']
        if laue == '-1':
            return hkl[2], hkl[1], hkl[0]
        if laue == '2/m':
            axis = 1 + ['a', 'b', 'c'].index(symmetry['SGUniq'])
            return tuple(lattice.SwapIndx(axis, hkl))
        if laue in {'mmm', '4/m', '6/m', '4/mmm', '6/mmm', '3m1', '31m', '3'}:
            return hkl[2], hkl[0], hkl[1]
        return tuple(hkl)

    def _calculate(self, record, dmin, symmetry, values, bounds, lattice):
        bounds = np.asarray(bounds).copy()
        laue = symmetry['SGLaue']
        if laue in {'3m1', '31m', '3'}:
            bounds[1] = bounds[0]
        elif laue in {'m3', 'm3m'}:
            bounds[:] = bounds[0]
        # These are the pinned native calc_rDsq monomials and NumPy
        # multiply/reduction, batched over independent HKLs. Their scalar
        # equivalence, ties and range-boundary behavior have native controls.
        metric = np.sum(values[:, None] * record['monomials'], axis=0)
        selected = np.flatnonzero(np.all(abs(record['hkls']) <= bounds, axis=1)
                                  & (metric > 0) & (metric <= 1. / (dmin ** 2)))
        result = [[*map(int, record['hkls'][i]), 1. / math.sqrt(metric[i])]
                  for i in selected]
        return lattice.sortHKLd(result, True, True)

    def _generate(self, original, lattice, dmin, symmetry, values):
        self.calls += 1
        a = np.asarray(values)
        if (symmetry.get('SGLaue') not in _SUPPORTED or a.shape != (6,)
                or a.dtype != np.dtype('float64') or not np.isfinite(a).all()
                or not np.isscalar(dmin) or not np.isfinite(dmin) or dmin <= 0):
            self.bypasses['unsupported native metric or setting'] += 1
            return original(dmin, symmetry, values)
        # Hash only trusted in-process symmetry, never unpickle uploaded data.
        encoded = pickle.dumps(symmetry, protocol=5)
        if len(encoded) > 1024 * 1024:
            self.bypasses['symmetry metadata budget'] += 1
            return original(dmin, symmetry, values)
        key = hashlib.sha256(encoded).digest()
        metric, _ = lattice.A2Gmat(a)
        bounds = np.asarray(lattice.getHKLmax(dmin, symmetry, values))
        record = self.records.get(key)
        usable = False
        if record is not None and dmin >= record['cut'] / math.sqrt(_ALPHA):
            try:
                # If G >= alpha*Gref, every current reflection is contained
                # by the reference sphere at sqrt(alpha)*dmin. The extra
                # margin avoids admitting a rounding-level boundary proof.
                np.linalg.cholesky(metric - _ALPHA * record['metric']
                                   - np.eye(3) * np.linalg.norm(metric) * 1e-12)
                usable = bool(np.all(bounds <= record['bounds']))
            except np.linalg.LinAlgError:
                pass
        if usable:
            self.hits += 1
            self.records.move_to_end(key)
            return self._calculate(record, dmin, symmetry, a, bounds, lattice)
        cut = dmin * math.sqrt(_ALPHA) * .99
        envelope_bounds = np.asarray(lattice.getHKLmax(cut, symmetry, values))
        # Bound additional construction work before asking the native
        # generator for a larger list. Full native output/RSS is separate
        # from this optional cache's numeric storage accounting.
        nodes = math.prod(2 * int(x) + 1 for x in envelope_bounds)
        if nodes * 1024 > self.maximum_bytes:
            self.bypasses['envelope construction budget'] += 1
            return original(dmin, symmetry, values)
        expanded = original(cut, symmetry, values)
        if len(expanded) < 128:
            self.bypasses['small reflection inventory'] += 1
            return original(dmin, symmetry, values)
        expanded.sort(key=lambda row: self._order(row[:3], symmetry, lattice))
        hkls = np.asarray([row[:3] for row in expanded], dtype=np.int64)
        h = hkls.T
        monomials = np.asarray([h[0] ** 2, h[1] ** 2, h[2] ** 2,
                               h[0] * h[1], h[0] * h[2], h[1] * h[2]])
        size = hkls.nbytes + monomials.nbytes + 4096
        if size > self.maximum_bytes:
            self.bypasses['cache allocation budget'] += 1
            return original(dmin, symmetry, values)
        record = dict(hkls=hkls, monomials=monomials, metric=metric.copy(),
                      cut=cut, bounds=envelope_bounds.copy(), bytes=size)
        calculated = self._calculate(record, dmin, symmetry, a, bounds, lattice)
        native = original(dmin, symmetry, values)
        if not np.array_equal(np.asarray(calculated), np.asarray(native)):
            self.bypasses['native anchor disagreement'] += 1
            return native
        previous = self.records.pop(key, None)
        if previous is not None:
            self.bytes -= previous['bytes']
        while self.records and (len(self.records) >= 4 or self.bytes + size > self.maximum_bytes):
            _, old = self.records.popitem(last=False)
            self.bytes -= old['bytes']
            self.evictions += 1
        for array in (hkls, monomials, record['metric'], record['bounds']):
            array.flags.writeable = False
        self.records[key] = record
        self.bytes += size
        self.builds += 1
        return calculated

    @contextmanager
    def evaluation(self, lattice, phases):
        functions = tuple(getattr(lattice, name, None) for name in _SOURCES)
        reason = None
        if not self.enabled or not self.maximum_bytes:
            reason = 'disabled native reference'
        elif threading.current_thread() is not threading.main_thread():
            reason = 'requires owned main thread'
        elif any(p['General'].get('Type') != 'nuclear' or p['General'].get('Modulated')
                 or p['General'].get('doPawley') for p in phases.values()):
            reason = 'requires commensurate nuclear Rietveld phases'
        elif self.reviewed != functions:
            self.clear()
            if any(_hash(fn) != _SOURCES[name] for name, fn in zip(_SOURCES, functions)):
                reason = 'unreviewed native reflection functions or active hook'
            else:
                self.reviewed = functions
        if reason is not None:
            self.bypasses[reason] += 1
            yield
            return
        if not _LOCK.acquire(blocking=False):
            self.bypasses['another native reflection scope owns this process'] += 1
            yield
            return
        original = lattice.GenHLaue
        def generated(dmin, SGData, A):
            return self._generate(original, lattice, dmin, SGData, A)
        try:
            lattice.GenHLaue = generated
            yield
        finally:
            lattice.GenHLaue = original
            _LOCK.release()
