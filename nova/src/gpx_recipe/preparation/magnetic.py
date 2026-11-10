"""Unweighted magnetic physics; every observation creates a fresh objective."""
from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path
import re
import uuid

import numpy as np

SCHEMA = 'radar-prepared-magnetic-response/v1'
ARRAYS = ('x', 'fixed', 'nuclear', 'quadratic', 'background', 'moment_maps')
MAX_BYTES = 128*1024*1024


def array_digest(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json.dumps([value.dtype.str, value.shape]).encode())
    h.update(value.tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class MagneticResponse:
    """An exact-context forward response, with no counts, weights or score."""
    x: np.ndarray
    fixed: np.ndarray
    nuclear: np.ndarray
    quadratic: np.ndarray
    background: np.ndarray
    moment_maps: np.ndarray
    physics_sha256: str
    proof: dict
    _native_seal: tuple | None = field(init=False, default=None, repr=False, compare=False)

    def __post_init__(self):
        if (not isinstance(self.physics_sha256, str)
                or not re.fullmatch('[a-f0-9]{64}', self.physics_sha256)):
            raise ValueError('A sealed native physics identity is required')
        total = 0
        for name in ARRAYS:
            value = np.asarray(getattr(self, name))
            if value.dtype.kind != 'f' or value.dtype.itemsize != 8:
                raise ValueError('Prepared responses require float64 physical arrays')
            total += value.nbytes
            if total > MAX_BYTES or not np.isfinite(value).all():
                raise ValueError('Nonfinite or oversized prepared response')
            value = value.astype('<f8', copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        n = len(self.x) if self.x.ndim == 1 else 0
        q = self.quadratic
        if (n < 2 or np.any(np.diff(self.x) <= 0) or self.fixed.shape != (n,)
                or self.nuclear.shape != (n,) or q.ndim != 3 or q.shape[0] != n
                or not 1 <= q.shape[1] <= 24 or q.shape[1] != q.shape[2]
                or self.background.ndim != 2 or self.background.shape[0] != n
                or self.moment_maps.ndim != 3 or self.moment_maps.shape[1:] != (3, q.shape[1])
                or not len(self.moment_maps)
                or not np.allclose(q, q.swapaxes(1, 2), rtol=1e-10, atol=1e-12)):
            raise ValueError('Invalid prepared grid, response or physical moment basis')
        encoded = json.dumps(self.proof, allow_nan=False, sort_keys=True)
        if len(encoded) > 65536:
            raise ValueError('Prepared response proof exceeds its metadata budget')
        object.__setattr__(self, 'proof', json.loads(encoded))

    def predict(self, moments, *, scale, background):
        moments = np.asarray(moments, float)
        background = np.asarray(background, float)
        if (moments.shape != (self.quadratic.shape[1],)
                or background.shape != (self.background.shape[1],)
                or not np.isfinite(moments).all() or not np.isfinite(background).all()
                or not np.isfinite(scale) or scale < 0):
            raise ValueError('Finite complete moments, background and nonnegative common scale required')
        return self.fixed + scale*(self.nuclear + np.einsum('i,nij,j->n', moments,
            self.quadratic, moments)) + self.background@background

    def _seal_native(self):
        """Called only after this process checks the actual native evaluator."""
        object.__setattr__(self, '_native_seal', tuple(array_digest(getattr(self, k)) for k in ARRAYS))

    def bind(self, *, x, y, weights, fit_mask, normalization_mask,
             physics_sha256, maximum_moment, fixed_scale=None, weight_factor=1.):
        """Bind actual data without interpolation or a retained likelihood.

        Masks are True for included bins. Native fit exclusions need not remove
        bins from its observed-norm denominator, so both masks are explicit.
        """
        from ..symmetry.profiled_magnetic import ProfiledMagneticObjective
        if (self._native_seal is None or self._native_seal !=
                tuple(array_digest(getattr(self, k)) for k in ARRAYS)):
            raise ValueError('Qualify this prepared response against the current native model before fitting')
        if physics_sha256 != self.physics_sha256:
            raise ValueError('Prepared physics changed; adapt or rebuild this response')
        x, y, weights = (np.asarray(a, float) for a in (x, y, weights))
        fit_mask, normalization_mask = np.asarray(fit_mask), np.asarray(normalization_mask)
        if (x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all()
                or np.any(np.diff(x) <= 0) or any(a.shape != x.shape for a in
                    (y, weights, fit_mask, normalization_mask))
                or fit_mask.dtype != bool or normalization_mask.dtype != bool
                or not fit_mask.any() or np.any(fit_mask & ~normalization_mask)
                or not np.isfinite(weight_factor) or weight_factor <= 0):
            raise ValueError('Valid ordered observations and explicit fit/normalization masks required')
        if not np.array_equal(self.x, x):
            # The native background basis and edge-reflection membership can
            # depend on this exact calculation range; cropping is not enough.
            raise ValueError('The actual grid or fit range needs native response rendering')
        indices = np.arange(len(x))
        if (not np.isfinite(y[normalization_mask]).all()
                or not np.isfinite(weights[normalization_mask]).all()
                or np.any(weights[normalization_mask] <= 0)):
            raise ValueError('Included observations need finite values and positive weights')
        take = indices[fit_mask]
        whitening = np.sqrt(weights[fit_mask]*weight_factor)
        normalization = float(np.sum(weights[normalization_mask]*y[normalization_mask]**2)*weight_factor)
        return ProfiledMagneticObjective(
            offset=(self.fixed[take]-y[fit_mask])*whitening,
            nuclear=self.nuclear[take]*whitening,
            quadratic=self.quadratic[take]*whitening[:, None, None],
            background=self.background[take]*whitening[:, None], moment_maps=self.moment_maps,
            maximum_moment=maximum_moment, normalization=normalization, fixed_scale=fixed_scale)

    def save(self, path):
        path = Path(path)
        manifest = dict(schema=SCHEMA, physics_sha256=self.physics_sha256, proof=self.proof,
            arrays={name: array_digest(getattr(self, name)) for name in ARRAYS},
            observations_included=False, scientific_result=False)
        temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
        try:
            with temporary.open('xb') as stream:
                np.savez_compressed(stream, manifest=np.array(json.dumps(manifest, allow_nan=False,
                    sort_keys=True)), **{name: getattr(self, name) for name in ARRAYS})
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return manifest


def load(path, *, physics_sha256, expected_file_sha256=None):
    from ..live.array_limits import validate_npz_bytes
    if Path(path).is_symlink() or Path(path).stat().st_size > MAX_BYTES:
        raise ValueError('Prepared response exceeds its file budget')
    with Path(path).open('rb') as stream:raw=stream.read(MAX_BYTES+1)
    if expected_file_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_file_sha256:
        raise ValueError('Prepared response bytes differ from their native producer')
    validate_npz_bytes(raw,maximum_bytes=MAX_BYTES)
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        if set(data.files) != set(ARRAYS)|{'manifest'}:
            raise ValueError('Unexpected prepared response array inventory')
        entry = data['manifest']
        if entry.shape != () or entry.dtype.kind != 'U' or entry.nbytes > 262144:
            raise ValueError('Invalid prepared response manifest')
        manifest = json.loads(str(entry))
        if (set(manifest) != {'schema', 'physics_sha256', 'proof', 'arrays',
                'observations_included', 'scientific_result'} or manifest['schema'] != SCHEMA
                or manifest['physics_sha256'] != physics_sha256
                or manifest['observations_included'] is not False
                or manifest['scientific_result'] is not False
                or set(manifest['arrays']) != set(ARRAYS)):
            raise ValueError('Prepared response identity, scope or schema changed')
        arrays = {name: data[name] for name in ARRAYS}
        if any(array_digest(a) != manifest['arrays'][name] for name, a in arrays.items()):
            raise ValueError('Prepared response array checksum mismatch')
        return MagneticResponse(**arrays, physics_sha256=physics_sha256, proof=manifest['proof'])
