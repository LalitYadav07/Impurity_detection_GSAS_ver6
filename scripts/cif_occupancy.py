"""Compatibility entry point for the shared, hash-bound CIF loader."""
import importlib.util
from pathlib import Path
import sys
try:
    from gpx_recipe import cif_occupancy as _implementation
except ModuleNotFoundError as exc:
    if exc.name != 'gpx_recipe':
        raise
    _path=Path(__file__).resolve().parents[1]/'nova/src/gpx_recipe/cif_occupancy.py'
    _spec=importlib.util.spec_from_file_location('radar_cif_occupancy_core',_path)
    if _spec is None or _spec.loader is None:
        raise ImportError('The shared CIF loader is unavailable') from exc
    _implementation=importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name]=_implementation
    _spec.loader.exec_module(_implementation)
sys.modules[__name__]=_implementation
