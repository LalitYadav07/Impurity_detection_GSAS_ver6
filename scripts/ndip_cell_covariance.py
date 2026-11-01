"""Compatibility import for the one shared native covariance implementation."""
try:
    from gpx_recipe.cell_covariance import CELL_KEYS, cell_fit_information
except ModuleNotFoundError as exc:
    if exc.name not in {'gpx_recipe', 'gpx_recipe.cell_covariance'}:
        raise
    # Standalone source-tree scripts also work before installing NOVA.
    import importlib.util
    from pathlib import Path
    _source = Path(__file__).resolve().parents[1] / 'nova/src/gpx_recipe/cell_covariance.py'
    _spec = importlib.util.spec_from_file_location('_radar_cell_covariance', _source)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    CELL_KEYS, cell_fit_information = _module.CELL_KEYS, _module.cell_fit_information
