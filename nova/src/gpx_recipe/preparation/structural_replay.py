"""Bounded worker-local training rows replayed from native paired examples.

This owns no uploaded program and publishes no qualified model. The shared
cohort assembler still computes every row, merges exact aliases and assigns
whole-trajectory holdouts. Its row hashes bind subsequent reconstructions.
Only redundant expanded feature storage is omitted. Portable library readers
and their existing numerical admission limits remain unchanged.
"""
from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat
import os
import tempfile

import numpy as np

from .structural_augmentation import Controls, ObservationAugmenter
from .structural_cohort import assemble_into
from .structural_features import ThreeCurveProjector
from .structural_paired_blocks import read, filenames
from .structural_training_data import FeatureRows, MAX_BATCH_ROWS, MAX_STORE_BYTES, _PERMIT
from . import structural_replay_limits as limits
from .structural_moments import Moments

MAX_NATIVE_CACHE_BYTES = 64*1024**2


def _file_state(paths):
    result = []
    for path in paths:
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('Replay sources must remain regular owned native files')
        result.append((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns))
    return tuple(result)


class _RowHashes:
    def reserve(self, count, bins):
        if not 1 <= count <= limits.MAX_ROWS or count*limits.INDEX_BYTES_PER_ROW > limits.MAX_INDEX_BYTES:
            raise ValueError('Replay rows exceed the bounded seekable index inventory')
        self.capacity = count; self.bins = bins; self.hashes = []; self.digest = hashlib.sha256()
        self.moments = Moments(count, bins)

    def append(self, feature):
        if (feature.shape != (3, self.bins) or feature.dtype != np.float32
                or not np.isfinite(feature).all() or len(self.hashes) >= self.capacity):
            raise ValueError('Invalid or excess replay feature row')
        raw=feature.tobytes()
        self.hashes.append(hashlib.sha256(raw).digest()); self.digest.update(raw)
        self.moments.append(feature)

    def finish(self, rows):
        if rows != len(self.hashes):
            raise ValueError('Replay feature inventory is incomplete')
        self.moments.seal()


class _NativeArrays:
    """Owned seekable native arrays, deduplicated only on identical bytes.

    Random minibatches must not decompress a whole candidate block per row.
    Capture these data during the assembler's existing native-block passes;
    every original dtype/value is retained, with an 8 GiB disk budget shared
    with the existing working-store limit. The temporary file is unlinked
    and closes with its worker. It is never an uploaded executable/cache.
    """
    def __init__(self, destination=None):
        self.stream = tempfile.TemporaryFile(dir=destination)
        self.index = {}; self.records = []; self.blocks = []; self.size = 0
        self.cache = OrderedDict(); self.cache_bytes = 0; self.reads = 0

    def _put(self, value):
        raw = value.tobytes(); key = value.dtype.str, len(raw), hashlib.sha256(raw).digest()
        if key in self.index:
            index = self.index[key]; offset, size, dtype = self.records[index]
            # Identity is byte equality, not an assertion based on a hash.
            if os.pread(self.stream.fileno(), size, offset) != raw:
                raise ValueError('Native array digest collision cannot merge source values')
            return index
        if self.size+len(raw) > MAX_STORE_BYTES or len(self.records) >= 5*limits.MAX_ROWS:
            raise ValueError('Native replay arrays exceed their bounded working-store budget')
        index = len(self.records); self.index[key] = index
        self.records.append((self.size, len(raw), value.dtype.str))
        self.stream.write(raw); self.stream.flush(); self.size += len(raw)
        return index

    def add(self, index, block):
        if index < len(self.blocks): return
        if index != len(self.blocks): raise ValueError('Capture native replay blocks in declared order')
        a = block.arrays; q = self._put(a['q']); bg = self._put(a['background'])
        rows = np.empty((len(a['theory_intensity']), 5), np.uint32)
        for i, curve in enumerate(a['theory_intensity']):
            rows[i] = q, self._put(curve), bg, self._put(a['parent_intensity'][i]), self._put(a['parent_background'][i])
        self.blocks.append(np.frombuffer(rows.tobytes(), dtype=rows.dtype).reshape(rows.shape))

    def seal(self):
        self.stream.flush(); os.fsync(self.stream.fileno())
        self.state = os.fstat(self.stream.fileno())
        self.index.clear()  # Only construction needs the large digest index.

    def check(self):
        now = os.fstat(self.stream.fileno())
        if (now.st_size, now.st_mtime_ns, now.st_ctime_ns) != (
                self.state.st_size, self.state.st_mtime_ns, self.state.st_ctime_ns):
            raise ValueError('The captured native replay arrays changed during training')

    def get(self, index):
        index = int(index)
        if index not in self.cache:
            offset, size, dtype = self.records[index]
            raw = os.pread(self.stream.fileno(), size, offset)
            if len(raw) != size: raise ValueError('Incomplete captured native array read')
            while self.cache and self.cache_bytes+size > MAX_NATIVE_CACHE_BYTES:
                _, old = self.cache.popitem(last=False); self.cache_bytes -= old.nbytes
            if size > MAX_NATIVE_CACHE_BYTES:
                raise ValueError('One native array exceeds the replay row-cache budget')
            self.cache[index] = np.frombuffer(raw, dtype=np.dtype(dtype))
            self.cache_bytes += size; self.reads += 1
        self.cache.move_to_end(index)
        return self.cache[index]

    def close(self):
        self.cache.clear(); self.cache_bytes = 0; self.stream.close()


class ReplayRows(FeatureRows):
    """Same bounded training interface, without an expanded features.npy.

    Construct through build(), after the existing assembler has checked and
    hashed all rows. A small LRU owns immutable native array rows from a
    bounded seekable data file. Archive decompression happens at assembly,
    never once per random minibatch. Changed files/values fail.
    """
    def __init__(self, directories, names, cohort, hashes, states, augmenter, bank, feature_sha, moments, *, _permit=None):
        if _permit is not _PERMIT:
            raise TypeError('Build replay rows from a checked paired cohort')
        super().__init__(None, len(hashes), storage='native_pair_replay',
                         bins=len(cohort['q_edges'])-1, channels=3, _permit=_PERMIT)
        self._paths = tuple(tuple(p/name for name in filenames(names)) for p in directories)
        self._states = tuple(states)
        self._feature_sha = feature_sha
        self._moments = moments; self._statistics_hits = 0; self._statistics_fallbacks = 0
        self._edges = np.frombuffer(cohort['q_edges'].tobytes(), dtype=np.float64)
        self._hashes = np.frombuffer(b''.join(hashes), dtype=np.uint8).reshape(len(hashes), 32)
        selectors = np.empty((len(hashes), 3), np.int32)
        provenance = np.empty((len(hashes), 32), np.uint8)
        for i, (refs, simulations) in enumerate(zip(cohort['references'], cohort['example_simulations'])):
            simulation = simulations[0]
            if simulation == dict(kind='clean_native'): variant = -1
            elif (augmenter is not None and simulation.get('kind') == 'simulated_observation'
                    and type(simulation.get('variant')) is int
                    and 0 <= simulation['variant'] < augmenter.controls.variants):
                variant = simulation['variant']
            else: raise ValueError('A replay row has no declared observation variant')
            selectors[i] = *refs[0], variant
            provenance[i] = np.frombuffer(hashlib.sha256(json.dumps(simulation, sort_keys=True).encode()).digest(), np.uint8)
        self._selectors = np.frombuffer(selectors.tobytes(), dtype=np.int32).reshape(selectors.shape)
        self._provenance = np.frombuffer(provenance.tobytes(), dtype=np.uint8).reshape(provenance.shape)
        self._augmenter = augmenter
        self._bank = bank; self._projectors = OrderedDict(); self._clean = OrderedDict()
        self._closed = False; self._regenerated = 0

    def _check(self, index):
        if _file_state(self._paths[index]) != self._states[index]:
            raise ValueError('Native replay sources changed during training')

    def take(self, indices):
        indices = np.asarray(indices)
        if self._closed:
            raise ValueError('The native replay training rows are closed')
        if (indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) > MAX_BATCH_ROWS
                or np.any(indices < 0) or np.any(indices >= len(self))):
            raise ValueError('Read a bounded batch of valid structural feature rows')
        blocks = set(map(int, self._selectors[indices, 0]))
        for index in blocks: self._check(index)
        self._bank.check()
        result = np.empty((len(indices), *self.shape[1:]), np.float32)
        for out_index, index in enumerate(indices):
            block_index, native_index, variant = map(int, self._selectors[int(index)])
            locations = self._bank.blocks[block_index][native_index]
            q, current, background, parent, parent_background = (self._bank.get(i) for i in locations)
            grid = int(locations[0])
            if grid not in self._projectors:
                self._projectors[grid] = ThreeCurveProjector(q, self._edges)
                if len(self._projectors) > 8: self._projectors.popitem(last=False)
            self._projectors.move_to_end(grid); projector = self._projectors[grid]
            key = (block_index, native_index)
            if key not in self._clean:
                clean = projector.features(current, parent, background=background,
                    reference_background=parent_background)['features']
                self._clean[key] = clean, hashlib.sha256(clean.tobytes()).hexdigest()
                if len(self._clean) > 64: self._clean.popitem(last=False)
            self._clean.move_to_end(key); clean, identity = self._clean[key]
            if variant == -1:
                value = clean
            else:
                if self._augmenter is None:
                    raise ValueError('A replay observation has no declared generator')
                draw = self._augmenter.observation(q, current, background,
                    clean_identity=identity, variant=variant)
                encoded = json.dumps(dict(kind='simulated_observation', **draw['provenance']), sort_keys=True).encode()
                if hashlib.sha256(encoded).digest() != self._provenance[int(index)].tobytes():
                    raise ValueError('Replay observation provenance differs from cohort assembly')
                value = projector.features(draw['intensity'], parent,
                    background=draw['background'], reference_background=parent_background)['features']
            if hashlib.sha256(value.tobytes()).digest() != self._hashes[int(index)].tobytes():
                raise ValueError('Replayed feature bytes differ from the assembled row')
            result[out_index] = value; self._regenerated += 1
        for index in blocks: self._check(index)
        self._bank.check()
        return result

    def diagnostics(self):
        return dict(storage=self.storage, expanded_feature_disk_bytes=0,
            captured_native_disk_bytes=self._bank.size, native_disk_byte_limit=MAX_STORE_BYTES,
            retained_native_bytes=self._bank.cache_bytes, native_cache_byte_limit=MAX_NATIVE_CACHE_BYTES,
            retained_clean_rows=len(self._clean), retained_grid_plans=len(self._projectors),
            source_blocks_captured=len(self._bank.blocks), native_array_reads=self._bank.reads,
            reconstructed_rows=self._regenerated, owned_moment_bytes=self._moments.nbytes if self._moments is not None else 0,
            certified_statistics_queries=self._statistics_hits,
            original_statistics_fallbacks=self._statistics_fallbacks, screening_qualified=False)

    def certified_statistics(self, indices, *, minimum_scale, center):
        self.feature_sha256()  # The immutable summary cannot bypass changed sources.
        from .structural_moments import MAX_BYTES, BYTES_PER_ROW
        if len(indices)*BYTES_PER_ROW > MAX_BYTES:
            self._statistics_fallbacks += 1
            return None
        result = self._moments.statistics(indices, minimum_scale=minimum_scale, center=center)
        self.feature_sha256()
        if result is None: self._statistics_fallbacks += 1
        else: self._statistics_hits += 1
        return result

    def residual_strengths(self, indices):
        self.feature_sha256()
        result = self._moments.strengths(indices)
        self.feature_sha256()
        return result

    def feature_sha256(self):
        if self._closed:raise ValueError('The native replay training rows are closed')
        self._bank.check()
        for index in range(len(self._paths)):self._check(index)
        return self._feature_sha

    @property
    def row_layout(self):
        return limits.validate(dict(schema=limits.SCHEMA, rows=len(self), bins=self.shape[2], channels=3,
            native_bytes=self._bank.size,
            index_bytes=self._selectors.nbytes+self._provenance.nbytes+self._hashes.nbytes))

    def close(self):
        if not self._closed: self._bank.close()
        self._projectors.clear(); self._clean.clear(); self._moments = None; self._closed = True


def build(directories, *, bins=4096, split_seed=69271, augmenter=None, names='pairs', destination=None):
    """Assemble and replay the same scientific cohort within existing limits.

    Working state remains owned by this worker; it is not an alternate upload
    format. All aliases, simulation provenance and every class holdout are
    retained by the shared assembler. No model prediction is evaluated here.
    """
    directories = tuple(Path(p) for p in directories); native_names = filenames(names)
    if not 1 <= len(directories) <= 5000:
        raise ValueError('A bounded paired native catalog is required')
    paths = [tuple(p/name for name in native_names) for p in directories]
    states = tuple(_file_state(p) for p in paths)
    if augmenter is not None:
        if not isinstance(augmenter, ObservationAugmenter):
            raise TypeError('An explicit native observation generator is required')
        # Own the generator and impurity templates; later caller edits cannot
        # change reconstruction seeds or physical input arrays.
        augmenter = ObservationAugmenter(Controls(**asdict(augmenter.controls)),
            instrument_sha256=augmenter.metadata['instrument_sha256'], impurities=[
                dict(q=q, signal=y, source_sha256=sha,
                     instrument_sha256=augmenter.metadata['instrument_sha256'])
                for q, y, sha in augmenter.impurities])
    sink = _RowHashes(); bank = _NativeArrays(destination); passes = 0
    def blocks():
        nonlocal passes
        capture = passes > 0; passes += 1
        for index, path in enumerate(directories):
            block = read(path, names=names)
            # Admission/reservation happens between the assembler's passes.
            # An oversized cohort must fail before writing its native bank.
            if capture: bank.add(index, block)
            yield block
    try:
        cohort = assemble_into(blocks, sink, bins=bins, split_seed=split_seed, augmenter=augmenter, paired=True)
        if tuple(_file_state(p) for p in paths) != states:
            raise ValueError('Native replay sources changed during cohort assembly')
        bank.seal()
        cohort['features'] = ReplayRows(directories, names, cohort, sink.hashes, states,
                                      augmenter, bank, sink.digest.hexdigest(), sink.moments, _permit=_PERMIT)
    except BaseException:
        bank.close(); raise
    return cohort
