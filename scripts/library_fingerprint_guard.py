"""Validate saved isotope fingerprints before a custom neutron library is ranked.

This reads existing sources and receipts. It never migrates an archive or
regenerates a diffraction pattern. Pure built-in catalog use is outside this
custom-library boundary and is not claimed to have been migrated.
"""
from __future__ import annotations

import json
import math
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aniso_db_loader import CatalogPaths, DBLoader, _load_original_db, _ORIG_DB_CACHE
from cif_occupancy import raw_sites, species_key, _species
from neutron_fingerprints import request_metadata


MAX_UNCACHED_ORIGINAL_BYTES = 16 * 1024 * 1024
MAX_CIF_BYTES = 20 * 1024 * 1024


class LibraryFingerprintError(ValueError):
    pass


def _fail(phase, reason):
    raise LibraryFingerprintError(
        f"Custom neutron library needs rebuilding before candidate ranking ({phase}: {reason}). "
        "Rebuild it from the original CIFs with the current RADAR-PD CIF Library Builder, "
        "or use an explicit mini library with accessible, validated source CIFs. "
        "The saved archive has not been modified.")


def _eligible_ids(loader, config):
    ids = loader.catalog['id'].astype(str).tolist()
    selected = set()
    for dataset in config.get('datasets') or [{}]:
        allowed = dataset.get('allowed_elements', config.get('allowed_elements')) or []
        if not allowed:
            return ids
        ef = {**(config.get('element_filter') or {}), **(dataset.get('element_filter') or {})}
        selected.update(loader.filter_by_element_mask(allowed, ids,
            ignore_elements=ef.get('ignore_elements'), require_base=bool(ef.get('require_base', True)),
            max_offlist_elements=int(ef.get('max_offlist_elements', 0)),
            disallow_offlist=ef.get('disallow_offlist'), wildcard_relation=str(ef.get('wildcard_relation', 'any')),
            sample_env=ef.get('sample_env'), disallow_pure=ef.get('disallow_pure')))
    return [pid for pid in ids if pid in selected]


def _check_database_binding(root, config):
    # A legacy config must not validate one archive while ranking another.
    for dataset in config.get('datasets') or [{}]:
        database = {**(config.get('db') or {}), **(dataset.get('db') or {})}
        if not database:
            database = config.get('db_neutron') or {}
        for key, relative in (('catalog_csv', 'catalog_deduplicated.csv'), ('profiles_dir', 'profiles64')):
            value = dataset.get(key) or database.get(key)
            if value and Path(value).expanduser().resolve() != root / relative:
                _fail('configuration', 'the configured ranking database differs from the selected custom library')


def _source_bytes(loader, phase, stats):
    mapped = (loader._cif_map or {}).get(phase)
    if mapped:
        path = Path(mapped).resolve()
        roots = [Path(loader.paths.catalog_csv).parent.resolve()]
        if loader.paths.original_json:
            roots.append(Path(loader.paths.original_json).parent.resolve())
        if not any(path.is_relative_to(root) for root in roots):
            _fail(phase, 'the source CIF is outside the selected library or its configured base')
        try:
            with path.open('rb') as stream:
                raw = stream.read(MAX_CIF_BYTES + 1)
        except OSError:
            _fail(phase, 'the source CIF is unavailable')
        if len(raw) > MAX_CIF_BYTES:
            _fail(phase, 'the source CIF exceeds the bounded validation size')
        stats['cif_file_reads'] += 1
        stats['cif_bytes_read'] += len(raw)
        return raw
    original = loader.paths.original_json
    if not original:
        _fail(phase, 'hydrogen-bearing source identity is unavailable')
    try:
        stat = Path(original).stat()
        signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        cached = _ORIG_DB_CACHE.get(original)
        already_cached = cached is not None and cached[0] == signature
        if not already_cached and stat.st_size > MAX_UNCACHED_ORIGINAL_BYTES:
            _fail(phase, 'source metadata is too large for bounded intake; source-access review is required')
        data = _load_original_db(original)
        if not already_cached:
            stats['original_json_reads'] += 1
            stats['original_json_bytes_read'] += stat.st_size
        raw = data.get(phase, {}).get('cif_content')
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        if isinstance(exc, LibraryFingerprintError):
            raise
        _fail(phase, 'source metadata could not be validated')
    if not isinstance(raw, str) or not raw.strip():
        _fail(phase, 'hydrogen-bearing source CIF is missing')
    if len(raw.encode('utf-8')) > MAX_CIF_BYTES:
        _fail(phase, 'the source CIF exceeds the bounded validation size')
    return raw.encode('utf-8')


def _expected_receipt(raw, simulation, phase):
    try:
        wavelength = simulation['wavelength']
        lower, upper, top = simulation['two_theta_min'], simulation['two_theta_max'], simulation['topM']
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (wavelength, lower, upper, top))
                or wavelength <= 0 or not 0 <= lower < upper <= 180 or top < 1 or int(top) != top):
            raise ValueError()
        return request_metadata(raw.decode('utf-8'), 'neutron', wavelength, lower, upper, int(top))
    except (KeyError, ValueError, TypeError, UnicodeError):
        _fail(phase, 'the original fingerprint generation settings are unavailable or invalid')


def validate_custom_neutron_library(root, config, *, original_json=None, io_stats=None):
    """Check eligible H-mask rows; D/T require current phase and profile receipts.

    The existing element-mask contract supplies a conservative candidate set.
    CIF parsing is restricted to its hydrogen-bearing rows; the whole mounted
    base metadata is never read merely to prove that it might contain isotopes.
    """
    root = Path(root).resolve()
    _check_database_binding(root, config)
    stats = io_stats if io_stats is not None else {}
    stats.update(catalog_rows=0, eligible_rows=0, hydrogen_rows=0, cif_parses=0,
        cif_file_reads=0, cif_bytes_read=0, original_json_reads=0,
        original_json_bytes_read=0, phase_npz_reads=0, profile_bundle_reads=0)
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    original_path = Path(original_json).expanduser() if original_json is not None else root / 'highsymm_metadata.json'
    if not original_path.is_absolute():
        original_path = root / original_path
    if original_json is not None or original_path.exists():
        if not original_path.is_file():
            _fail('source metadata', 'the configured original JSON file is unavailable')
        original_path = str(original_path)
    else:
        original_path = None
    loader = DBLoader(CatalogPaths(str(root / 'catalog_deduplicated.csv'),
        str(root / 'cif_map.json') if (root / 'cif_map.json').exists() else None,
        original_path))
    ids = _eligible_ids(loader, config)
    stats['catalog_rows'], stats['eligible_rows'] = len(loader.catalog), len(ids)
    hydrogen = [pid for pid in ids if int(loader._m_lo[loader._row_index[pid]]) & 1]
    stats['hydrogen_rows'] = len(hydrogen)
    isotopes = {}
    parsed = {}
    for phase in hydrogen:
        raw = _source_bytes(loader, phase, stats)
        key = sha256(raw).digest()
        if key not in parsed:
            try:
                sites, _ = raw_sites(raw)
                names = tuple(sorted({species_key(_species(row['resolved_symbol']))
                    for row in sites} & {'D', 'T'}))
            except Exception:
                _fail(phase, 'the source isotope identity could not be validated')
            if len(parsed) >= 256:
                parsed.clear()
            parsed[key] = names
            stats['cif_parses'] += 1
        if parsed[key]:
            isotopes[phase] = (raw, parsed[key])
    if isotopes:
        try:
            with np.load(root / 'profiles64/profiles64.npz', allow_pickle=False) as bundle:
                profiles = bundle['profiles']
                provenance = json.loads(str(bundle['fingerprint_provenance'].item()))
                grid = {key: bundle[key].item() for key in ('q_min', 'q_max', 'n_bins', 'sigma_bins')}
            stats['profile_bundle_reads'] += 1
            import pandas as pd
            index = pd.read_csv(root / 'profiles64/index.csv', dtype={'id': str})
            if index['id'].duplicated().any():
                raise ValueError('duplicate profile IDs')
            mapping = dict(zip(index['id'], index['row']))
            simulation = manifest.get('simulation') or {}
            if (provenance.get('scope') != 'listed phase IDs only'
                    or not isinstance(provenance.get('phases'), dict)
                    or any(grid[key] != simulation[key] for key in grid)):
                raise ValueError('profile grid/provenance mismatch')
            if (any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    for value in grid.values()) or not grid['q_min'] < grid['q_max']
                    or grid['sigma_bins'] < 0 or int(grid['n_bins']) != grid['n_bins'] or grid['n_bins'] < 1
                    or profiles.ndim != 2 or profiles.shape[1] != grid['n_bins']):
                raise ValueError('invalid profile grid')
        except (OSError, ValueError, KeyError, TypeError):
            _fail(', '.join(isotopes), 'the saved profile bundle has no compatible isotope provenance')
        from db_pack_builder import _build_profile_from_phase_npz
        for phase, (raw, species) in isotopes.items():
            expected = _expected_receipt(raw, simulation, phase)
            try:
                path = Path(loader.phase_npz_path(phase))
                with np.load(path, allow_pickle=False) as packet:
                    receipt = json.loads(str(packet['fingerprint_provenance'].item()))
                    request_hash = str(packet['fingerprint_request_sha256'].item())
                stats['phase_npz_reads'] += 1
                if receipt != expected or request_hash != expected['request_sha256']:
                    raise ValueError('phase receipt mismatch')
                if provenance['phases'].get(phase) != expected:
                    raise ValueError('profile receipt mismatch')
                position = mapping[phase]
                index = int(position)
                if isinstance(position, bool) or float(position) != index or not 0 <= index < len(profiles):
                    raise ValueError('invalid profile row identity')
                row = profiles[index]
                # Verify the archived histogram against its archived sticks;
                # never regenerate the structure's diffraction or replace data.
                derived = _build_profile_from_phase_npz(path, SimpleNamespace(**grid))
                if row.shape != derived.shape or not np.array_equal(row, derived.astype(row.dtype)):
                    raise ValueError('profile values disagree with their saved sticks')
            except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError):
                _fail(phase, 'missing, stale or inconsistent isotope fingerprint receipts; rebuilding is required')
    return {'schema': 'radar-custom-isotope-fingerprints/v1', 'checked_isotope_phase_ids': list(isotopes),
        'isotopes': {phase: list(value[1]) for phase, value in isotopes.items()}, 'io': stats,
        'scope': 'Eligible hydrogen-bearing rows in the selected custom neutron library; no archive migration',
        'builtin_only_legacy_scope': 'Not validated or migrated by this custom-library guard'}
