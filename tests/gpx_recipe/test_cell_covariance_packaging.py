"""Cell uncertainty must not depend on a repository-only scripts import."""
from pathlib import Path
import subprocess
import sys


def test_script_and_worker_covariance_calculations_remain_identical():
    root = Path(__file__).resolve().parents[2]
    import importlib.util
    from gpx_recipe.cell_covariance import cell_fit_information
    spec = importlib.util.spec_from_file_location('standalone_covariance', root / 'scripts/ndip_cell_covariance.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    assert module.cell_fit_information is cell_fit_information


def test_worker_exports_resolve_covariance_without_repository_scripts(tmp_path):
    root = Path(__file__).resolve().parents[2]
    script = """import importlib.util
import sys
from gpx_recipe.artifacts import cell_fit_information
assert importlib.util.find_spec('ndip_lattice') is None
assert cell_fit_information.__module__ == 'gpx_recipe.cell_covariance'
assert not any(name.startswith('GSASII') for name in sys.modules)
"""
    result = subprocess.run([sys.executable, '-c', script], cwd=tmp_path,
        env={'PATH': str(Path(sys.executable).parent), 'PYTHONPATH': str(root / 'nova/src')},
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
