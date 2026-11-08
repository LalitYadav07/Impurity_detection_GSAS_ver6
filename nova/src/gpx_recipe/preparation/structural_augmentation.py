"""Declared observation simulations for structural ranking, never native fits.

Effective Poisson counts are a synthetic noise parameter, not a claim about
the counting/normalization statistics of reduced experimental intensities.
Impurity fractions refer to integrated pattern intensity, not weight fraction.
Instrument broadening requires new native profiles; it is not imitated by an
arbitrary convolution of a Q-space curve.
"""
from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class Controls:
    variants: int = 3
    seed: int = 457139
    peak_counts: tuple = (1000., 100000.)
    background_fraction: tuple = (.02, .5)
    background_error: tuple = (0., .05)
    impurity_fraction: tuple = (0., 0.)

    def __post_init__(self):
        if type(self.variants) is not int or not 1 <= self.variants <= 8:
            raise ValueError('Use one to eight simulated observations per native pattern')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('A reproducible unsigned augmentation seed is required')
        for key, minimum, maximum in [('peak_counts', 10., 1e8),
                                      ('background_fraction', 0., 10.),
                                      ('background_error', 0., .5),
                                      ('impurity_fraction', 0., .5)]:
            values = getattr(self, key)
            if (not isinstance(values, (list, tuple)) or len(values) != 2
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in values)
                    or not minimum <= values[0] <= values[1] <= maximum):
                raise ValueError('Invalid physical augmentation bounds: '+key)
            object.__setattr__(self, key, tuple(map(float, values)))


def _area(q, signal):
    return float(np.dot(np.diff(q), (signal[:-1]+signal[1:])*.5))


def _signal(q, values):
    q = np.asarray(q, float); values = np.asarray(values, float)
    if (q.ndim != 1 or not 3 <= len(q) <= 25001 or values.shape != q.shape
            or not np.isfinite(q).all() or q[0] <= 0 or np.any(np.diff(q) <= 0)
            or not np.isfinite(values).all()):
        raise ValueError('A finite increasing Q grid and matching native signal are required')
    peak = float(values.max())
    # Only negligible floating-point negatives may be removed from a native
    # simulated foreground. Real reduced measurements never enter this API.
    if peak <= 0 or np.any(values < -64*np.finfo(float).eps*peak):
        raise ValueError('A nonnegative native simulated foreground is required')
    return q, np.maximum(values, 0.)


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class ObservationAugmenter:
    def __init__(self, controls, *, instrument_sha256, impurities=()):
        if not isinstance(controls, Controls) or not _checksum(instrument_sha256):
            raise ValueError('Declared structural observation controls and profile identity are required')
        if len(impurities) > 8:
            raise ValueError('Use a bounded set of native impurity templates')
        self.controls = controls; self.impurities = []
        identities = []
        for template in impurities:
            if (set(template) != {'q', 'signal', 'source_sha256', 'instrument_sha256'}
                    or template['instrument_sha256'] != instrument_sha256
                    or not _checksum(template['source_sha256'])):
                raise ValueError('Impurity templates require their exact native source and matching profile')
            q, signal = _signal(template['q'], template['signal'])
            # Own immutable numeric copies so later caller edits cannot change
            # a declared experiment or its augmentation seed.
            q = np.frombuffer(q.tobytes(), dtype=q.dtype)
            signal = np.frombuffer(signal.tobytes(), dtype=signal.dtype)
            array_sha = hashlib.sha256(q.tobytes()+signal.tobytes()).hexdigest()
            self.impurities.append((q, signal, template['source_sha256']))
            identities.append(dict(source_sha256=template['source_sha256'], arrays_sha256=array_sha))
        if controls.impurity_fraction[1] > 0 and not self.impurities:
            raise ValueError('Requested impurity variation needs a native impurity template')
        self.metadata = dict(schema='radar-structural-observation-augmentation/v1',
            controls=asdict(controls), instrument_sha256=instrument_sha256,
            impurities=identities, noise='effective_independent_poisson_counts',
            impurity_fraction_kind='integrated_Q_pattern_intensity',
            experimental_counting_statistics_established=False,
            profile_variation_included=False)
        self.sha256 = hashlib.sha256(json.dumps(self.metadata, sort_keys=True).encode()).hexdigest()

    def _context(self, q, native_total, native_background, clean_identity):
        q = np.asarray(q, float); total = np.asarray(native_total, float)
        background = np.asarray(native_background, float)
        if background.shape != q.shape or total.shape != q.shape or not np.isfinite(background).all():
            raise ValueError('Native total and background must share their Q grid')
        q, major = _signal(q, total-background)
        if not _checksum(clean_identity):
            raise ValueError('Simulated siblings require their clean-pattern identity')
        return q, major

    def observation(self, q, native_total, native_background, *, clean_identity, variant):
        """Replay one declared sibling without drawing or storing its neighbors.

        The same draw implementation serves sequential cohort assembly and
        bounded training reads. Variant order never changes a sibling's seed,
        arrays, or provenance. This is synthetic observation generation, not
        a replacement native diffraction calculation.
        """
        if type(variant) is not int or not 0 <= variant < self.controls.variants:
            raise ValueError('Select an existing declared observation variant')
        q, major = self._context(q, native_total, native_background, clean_identity)
        return self._draw(q, major, clean_identity, variant)

    def observations(self, q, native_total, native_background, *, clean_identity):
        q, major = self._context(q, native_total, native_background, clean_identity)
        for index in range(self.controls.variants):
            yield self._draw(q, major, clean_identity, index)

    def _draw(self, q, major, clean_identity, index):
        seed = int.from_bytes(hashlib.sha256(
            f'{self.sha256}:{clean_identity}:{index}'.encode()).digest()[:8], 'little')
        rng = np.random.default_rng(seed); c = self.controls
        count = float(np.exp(rng.uniform(*np.log(c.peak_counts))))
        fraction = float(rng.uniform(*c.impurity_fraction))
        signal = major.copy(); minor_source = None
        if fraction > 0:
            minor_q, minor, minor_source = self.impurities[int(rng.integers(len(self.impurities)))]
            if not np.array_equal(minor_q, q):
                raise ValueError('Native impurity and main-phase templates need the exact same Q grid')
            signal += minor*(_area(q, major)/_area(q, minor))*fraction/(1-fraction)
        peak = float(signal.max()); u = 2*(q-q[0])/(q[-1]-q[0])-1
        level = float(rng.uniform(*c.background_fraction))
        slope = float(rng.uniform(-.2, .2))
        true_background = peak*level*(1.+slope*u+.25*u*u)
        error = float(rng.uniform(*c.background_error))*float(rng.choice([-1, 1]))
        estimated = true_background+peak*level*error*(1.-.5*u*u)
        scale = count/peak; expectation = (signal+true_background)*scale
        observed = rng.poisson(expectation).astype(np.float64)/scale
        sigma = np.sqrt(expectation)/scale
        return dict(intensity=observed, background=estimated, sigma=sigma,
            provenance=dict(variant=index, seed=seed, effective_peak_counts=count,
                background_fraction=level, background_slope=slope, background_error=error,
                impurity_fraction=fraction, impurity_source_sha256=minor_source))
