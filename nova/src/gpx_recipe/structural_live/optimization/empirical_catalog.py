"""Versioned catalog entry point; legacy MP-only policies remain reproducible."""
import json
from pathlib import Path
from .materials_project_chemistry import (
    MaterialsProjectBondCatalog, EmpiricalNativeConstraint, EmpiricalCheckedForward,
    EmpiricalCheckedSlowForward, EmpiricalConfigurationRejected, digest, element_label,
    load_declared_catalog as load_legacy_catalog,
)
from .combined_bond_chemistry import load_combined_catalog


def load_declared_catalog(policy_path):
    path = Path(policy_path)
    if path.is_file():
        policy = json.loads(path.read_text())
        if isinstance(policy, dict) and type(policy.get('schema_version')) is int and policy['schema_version'] == 2:
            return load_combined_catalog(path)
    return load_legacy_catalog(path)


def catalog_source_files(catalog):
    """Closed artifact names for archival and portable worker input."""
    if catalog.manifest().get('catalog') == 'mp_cod_ordered':
        return {'chemistry/'+('radii.json' if name == 'radii' else name+'.csv'):
                (path, catalog.hashes[name]) for name, path in catalog.paths.items()}
    if not all(hasattr(catalog, name) for name in ('statistics_sha256', 'histogram_sha256')):
        raise ValueError('Catalog source identity is incomplete')
    return {'chemistry/pair_statistics.csv': (getattr(catalog, 'statistics', None), catalog.statistics_sha256),
            'chemistry/pair_histogram.csv': (getattr(catalog, 'histogram', None), catalog.histogram_sha256)}
