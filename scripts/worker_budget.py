"""Share the scientific allocation guard without requiring a NOVA installation.

The command-line pipeline also runs outside Galaxy. Only that local path may
use host capacity as its default; container/runner jobs require an entitlement.
"""
from functools import lru_cache
import importlib.util
import os
from pathlib import Path


@lru_cache(maxsize=1)
def _resources():
    # A normal source checkout has the same pure-Python allocation module.
    # Loading this one file avoids installing the interactive app's dependencies.
    path = Path(__file__).resolve().parents[1] / 'nova/src/gpx_recipe/symmetry/resources.py'
    if not path.is_file():
        from gpx_recipe.symmetry import resources
        return resources
    spec = importlib.util.spec_from_file_location('_radar_scientific_resources', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Scientific worker allocation module is unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def analysis_allocation(requested, *, tasks, environ=None):
    """Return the same CPU/RAM ceiling used by native model jobs.

    Explicit analysis overrides are requests, never grants. The standalone
    local path retains its previous CPU heuristic, bounded by affinity/RAM.
    """
    if type(requested) is not int or not 1 <= requested <= 256:
        raise ValueError('Request one to 256 analysis workers')
    if type(tasks) is not int or tasks < 1:
        raise ValueError('A worker pool needs a positive number of tasks')
    env = os.environ if environ is None else environ
    module = _resources()
    budget = module.allocation(requested, environ=env, maximum_workers=256)
    if (env.get('RADAR_PD_EXECUTION_CONTEXT') != 'galaxy'
            and not budget['allocation_reported']):
        limits = [requested, budget['cpu_affinity_count'] or budget['visible_cpu_count'] or 1]
        if budget['memory_mb']:
            limits.append(max(1, (budget['memory_mb'] - budget['reserve_mb']) // budget['minimum_worker_mb']))
        budget.update(workers=min(limits), allocation_source='standalone_local_capacity')
    budget['workers'] = min(budget['workers'], tasks)
    budget['execution_context'] = env.get('RADAR_PD_EXECUTION_CONTEXT', 'standalone')
    return budget


def analysis_workers(requested, *, tasks):
    return analysis_allocation(requested, tasks=tasks)['workers']
