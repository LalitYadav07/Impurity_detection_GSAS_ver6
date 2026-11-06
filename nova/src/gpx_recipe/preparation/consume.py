"""Load portable preparation through the same native adaptation engines."""
import json
from pathlib import Path

from .library import confined
from .magnetic import load
from .operations import portable_entry_artifacts
from ..live.reflection_factors import ScatteringFactorCache
from ..symmetry.compiler import file_sha


def restore_magnetic_entry(directory, *, factor_cache=None):
    """Return an unqualified response and lazy, unqualified factor hints.

    The caller's owned operation subsequently adapts to its actual native
    model and binds fresh observations. An uploaded proof is not attestation.
    Older response-only entries remain usable via native recomputation.
    """
    root = Path(directory).resolve()
    path = confined(root, 'entry.json')
    if path.stat().st_size > 8*1024*1024:
        raise ValueError('Preparation manifest exceeds its budget')
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != 'radar-experiment-preparation-entry/v1'
            or manifest.get('kind') != 'magnetic'
            or manifest.get('observations_included') is not False
            or manifest.get('scientific_result') is not False
            or not isinstance(manifest.get('artifacts'), dict)):
        raise ValueError('A theory-only magnetic preparation entry is required')
    names = portable_entry_artifacts('magnetic', manifest)
    for name in names:
        if file_sha(confined(root, name)) != manifest['artifacts'][name]:
            raise ValueError('A portable magnetic artifact changed')
    detail = manifest.get('detail')
    if not isinstance(detail, dict):
        raise ValueError('Native physical preparation metadata is missing')
    response = load(confined(root, 'response.npz'), physics_sha256=detail.get('physics_sha256'))
    if factor_cache is None:
        factor_cache = ScatteringFactorCache()
    if not isinstance(factor_cache, ScatteringFactorCache):
        raise ValueError('Use the shared bounded native scattering-factor cache')
    if 'scattering.npz' in names:
        scattering = detail['scattering']
        factor_cache.attach_source(confined(root, 'scattering.npz'), identity=scattering['identity'])
    return response, factor_cache, dict(entry_sha256=file_sha(path),
        observations_included=False, scientific_result=False, native_qualification_required=True,
        scattering_available='scattering.npz' in names)
