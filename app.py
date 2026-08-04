import streamlit as st
import os
import sys
import yaml
import json
import csv
import time
import datetime
import logging
from pathlib import Path
import re
import shutil
import sysconfig
import tempfile
import queue
import html
import psutil
import gc
import random
import subprocess
import importlib.util
import hashlib
import secrets
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONSTANTS ---
PERIODIC_TABLE = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"
]
PROJECT_ROOT = str(Path(__file__).resolve().parent)
IS_HF_SPACES = "SPACE_ID" in os.environ
LIVE_REFRESH_SECONDS = 3.0
STATUS_REFRESH_SECONDS = 5.0
ARTIFACT_REFRESH_SECONDS = 6.0
LIVE_LOG_DISPLAY_LIMIT = 450
FINISHED_LOG_DISPLAY_LIMIT = 1800


def _bootstrap_local_import_paths() -> None:
    """Make repo-local modules and bundled GSAS-II importable in the UI process."""
    project_root = Path(PROJECT_ROOT)
    scripts_path = str(project_root / "scripts")
    explicit_gsas = os.environ.get("RADAR_PD_GSASII_ROOT")
    possible_gsas_roots: list[Path] = []
    if explicit_gsas:
        possible_gsas_roots.append(Path(explicit_gsas))
    possible_gsas_roots.append(project_root / "GSAS-II")
    for site_key in ("purelib", "platlib"):
        site_path = sysconfig.get_paths().get(site_key)
        if site_path:
            possible_gsas_roots.append(Path(site_path) / "radar_pd_gsasii_runtime" / "gsasii")

    gsas_path = next(
        (
            candidate
            for candidate in possible_gsas_roots
            if (candidate / "GSASII" / "GSASIIpath.py").exists()
        ),
        possible_gsas_roots[0],
    )
    if not explicit_gsas and (gsas_path / "GSASII" / "GSASIIpath.py").exists():
        os.environ["RADAR_PD_GSASII_ROOT"] = str(gsas_path)

    candidates = (gsas_path / "GSASII", gsas_path, scripts_path, project_root)
    for candidate in reversed([str(path) for path in candidates]):
        if candidate in sys.path:
            sys.path.remove(candidate)
        sys.path.insert(0, candidate)


_bootstrap_local_import_paths()

# Ensure consistent logging for both CLI and Streamlit sessions
try:
    from scripts.logging_config import configure_logging
    configure_logging()
except Exception:
    # Best-effort: don't crash the UI if logging helper is unavailable
    logging.basicConfig(level=logging.INFO)

from config_builder import build_pipeline_config
from runner import PipelineRunner, stop_process_tree
from aniso_db_loader import DBLoader, CatalogPaths
from db_pack import build_db_config, get_db_pack_layout
from db_pack_builder import build_augmented_db_pack, build_mini_db_pack
from instprm_presets import (
    DEFAULT_LAB_XRAY_PRESET_KEY,
    get_builtin_instprm_preset,
    normalize_instrument_profile_to_instprm,
    SUPPORTED_INSTRUMENT_UPLOAD_EXTENSIONS,
    write_builtin_instprm_file,
)
from xrdml_converter import (
    SUPPORTED_DIFFRACTION_UPLOAD_EXTENSIONS,
    prepare_powder_data_file,
)
from ml_ranker_support import load_first_json_record

try:
    from plotly_interactive import (
        discover_plot_payload_files,
        load_interactive_payload,
        build_plotly_figure_from_payload,
    )
    INTERACTIVE_PLOTS_AVAILABLE = True
except Exception:
    INTERACTIVE_PLOTS_AVAILABLE = False

LAB_XRAY_PRESET = get_builtin_instprm_preset(DEFAULT_LAB_XRAY_PRESET_KEY)

# --- GSAS-II HEALTH CHECK ---
def check_gsas_installation():
    try:
        import GSASII
        import GSASII.GSASIIpath as G2path

        # Check for the core binary module but don't treat direct import failure as a hard crash
        # GSAS-II often relies on pathing that it manages internally.
        try:
            import pyspg
        except ImportError:
            # If GSASII is present, we consider it "OK" but maybe "Degraded" or just
            # needing internal pathing to be set up.
            pass

        return True

    except ImportError:
        st.error("GSAS-II is unavailable. Verify installation and Python path configuration.")
        return False
    except Exception as e:
        st.error(f"GSAS-II initialization failed: {e}")
        return False

# Trigger Check. Re-test if a previous browser rerun cached a failed health check.
if 'gsas_ready' not in st.session_state or not st.session_state.gsas_ready:
    st.session_state.gsas_ready = check_gsas_installation()
GSAS_READY = bool(st.session_state.gsas_ready)

# Fallback for Pixi detection (unused if running via pip)
def is_pixi_available():
    import shutil
    return shutil.which("pixi") is not None

if 'use_pixi' not in st.session_state:
    st.session_state.use_pixi = is_pixi_available()

# --- DATABASE STATUS CHECK ---
DB_NEUTRON_DIR = Path(PROJECT_ROOT) / "data" / "database_neutron"
DB_XRAY_DIR = Path(PROJECT_ROOT) / "data" / "database_xray"

# Each database must be self-contained with its own metadata JSON.
DB_NEUTRON_METADATA_JSON = DB_NEUTRON_DIR / "highsymm_metadata.json"
DB_XRAY_METADATA_JSON = DB_XRAY_DIR / "highsymm_metadata.json"

# Essential files to check (common to both)
REQUIRED_FILES_COMMON = [
    "catalog_deduplicated.csv",
    "mp_experimental_stable.csv"
]

# Google Drive download URLs (placeholder: set to actual GDrive share links)
DB_NEUTRON_GDRIVE_URL = "https://drive.google.com/uc?id=1BxPXjdbn7oYTXKfDeLct5-2PMkhcLVSH"
DB_XRAY_GDRIVE_URL = "https://drive.google.com/file/d/12H19jI3mGcYBpJrQRtY-5_WaMjFyIMah/view?usp=sharing"  # X-ray DB GDrive URL

_DB_WRAPPER_DIRS = {"database_aug", "database_xray", "database_neutron"}
WORKSPACES_ROOT = Path(PROJECT_ROOT) / "workspaces"
USER_DB_PACKS_DIR = Path(PROJECT_ROOT) / "data" / "user_db_packs"
ACTIVE_RUNS_ROOT = Path(PROJECT_ROOT) / "runs"


def repair_database_layout(target_db_dir: Path) -> bool:
    """Normalize malformed archive layouts in-place.

    Some ZIP archives were created on Windows with backslashes embedded in member
    names (for example `database_xray\\catalog_deduplicated.csv`). On Linux those
    become literal filenames rather than nested paths. This helper repairs that
    layout and also flattens one extra wrapper directory when present.
    """
    target_db_dir = Path(target_db_dir)
    if not target_db_dir.exists():
        return False

    changed = False
    files_to_move = [p for p in target_db_dir.rglob("*") if p.is_file()]

    for src in files_to_move:
        rel = src.relative_to(target_db_dir).as_posix().replace("\\", "/")
        parts = [part for part in rel.split("/") if part not in ("", ".")]
        if parts and parts[0] in _DB_WRAPPER_DIRS:
            parts = parts[1:]
        if not parts:
            continue

        dest = target_db_dir.joinpath(*parts)
        if dest == src:
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        shutil.move(str(src), str(dest))
        changed = True

    for root, dirs, _files in os.walk(target_db_dir, topdown=False):
        for dirname in dirs:
            path = Path(root) / dirname
            try:
                if path.exists() and not any(path.iterdir()):
                    path.rmdir()
                    changed = True
            except OSError:
                pass

    return changed

def check_db_integrity(target_db_dir, is_xray=False):
    """Check if a database directory has the required files.

    Neutron: requires profiles64/ directory with .npz files.
        X-ray: requires profiles64.npz monolithic file (from DGX consolidation)
            and its own highsymm_metadata.json.
        Neutron: requires profiles64/ directory and its own highsymm_metadata.json.
    """
    target_db_dir = Path(target_db_dir)
    if not target_db_dir.exists(): return False
    try:
        repair_database_layout(target_db_dir)
    except Exception:
        pass
    for f in REQUIRED_FILES_COMMON:
        if not (target_db_dir / f).exists(): return False
    if is_xray:
        # X-ray uses a monolithic profiles64.npz file
        npz_file = target_db_dir / "profiles64.npz"
        if not npz_file.exists(): return False
        # X-ray must be self-contained with its own metadata JSON
        if not (target_db_dir / "highsymm_metadata.json").exists(): return False
    else:
        # Neutron: requires metadata JSON and profiles64/ directory
        if not (target_db_dir / "highsymm_metadata.json").exists(): return False
        p_dir = target_db_dir / "profiles64"
        if not p_dir.exists() or not any(p_dir.iterdir()): return False
    return True


def _path_mtime(path_like) -> float | None:
    if not path_like:
        return None
    try:
        return Path(path_like).stat().st_mtime
    except OSError:
        return None


@st.cache_data(show_spinner=False, ttl=60)
def cached_check_db_integrity(target_db_dir: str, is_xray: bool = False) -> bool:
    return check_db_integrity(Path(target_db_dir), is_xray=is_xray)


# Initial check (placeholder, real check happens in UI)
DB_EXISTS = False


def _source_key_from_label(source_label: str) -> str:
    return "xray" if source_label == "X-ray" else "neutron"


def _sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return token or ""


def _selected_pack_state_key(source_label: str) -> str:
    return f"selected_custom_pack_{_source_key_from_label(source_label)}"


def _db_mode_matches_kind(selection_mode: str, kind: str) -> bool:
    if selection_mode == "Original":
        return False
    if selection_mode == "Augmented Pack":
        return kind == "augmented"
    if selection_mode == "Mini Pack":
        return kind == "mini"
    return False


def _db_config_is_usable(db_cfg: dict) -> bool:
    required = ("catalog_csv", "stable_csv", "profiles_dir")
    for key in required:
        path = db_cfg.get(key)
        if not path or not Path(path).exists():
            return False
    cif_sources = [db_cfg.get("original_json"), db_cfg.get("cif_map_json")]
    return any(path and Path(path).exists() for path in cif_sources)


def _format_bytes(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "0s"
    total = max(0, int(round(float(seconds))))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def _limits_axis_label(
    *,
    source_label: str,
    example_selection: str,
    builtin_instprm_key: str | None,
    instrument_mode: str,
) -> str:
    if example_selection == "LK-99 (TOF Demo)":
        return "TOF (\u00b5s)"
    if example_selection == "TbSSL (CW Demo)" or builtin_instprm_key:
        return "2\u03b8 (deg)"
    mode = (instrument_mode or "").strip().upper()
    if mode == "TOF":
        return "TOF (\u00b5s)"
    if mode == "CW":
        return "2\u03b8 (deg)"
    if source_label == "X-ray":
        return "2\u03b8 (deg)"
    return "native x-axis (2\u03b8 deg or TOF \u00b5s)"


def _parse_excluded_region_rows(editor_value) -> tuple[list[list[float]], list[str]]:
    if editor_value is None:
        return [], []

    if hasattr(editor_value, "to_dict"):
        records = editor_value.to_dict("records")
    elif isinstance(editor_value, list):
        records = editor_value
    else:
        records = []

    parsed: list[list[float]] = []
    errors: list[str] = []
    for idx, row in enumerate(records, start=1):
        start = row.get("Start", row.get("start"))
        end = row.get("End", row.get("end"))
        if start in (None, "") and end in (None, ""):
            continue
        if start in (None, "") or end in (None, ""):
            errors.append(f"Ignored region row {idx} must include both Start and End values.")
            continue
        try:
            parsed.append([float(start), float(end)])
        except Exception:
            errors.append(f"Ignored region row {idx} must contain numeric Start/End values.")
    return parsed, errors


def _parse_fit_window_inputs(lower_text: str, upper_text: str) -> tuple[tuple[float, float] | None, list[str]]:
    lower_text = str(lower_text or "").strip()
    upper_text = str(upper_text or "").strip()
    if not lower_text and not upper_text:
        return None, []
    if not lower_text or not upper_text:
        return None, ["Fit window override requires both Lower and Upper values."]
    try:
        lo = float(lower_text)
        hi = float(upper_text)
    except Exception:
        return None, ["Fit window override values must be numeric."]
    if lo == hi:
        return None, ["Fit window override must span a non-zero range."]
    return (min(lo, hi), max(lo, hi)), []


def _coerce_excluded_region_rows(value) -> list[dict]:
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        try:
            return value.to_dict("records")
        except Exception:
            return []
    if isinstance(value, list):
        rows = []
        for row in value:
            if isinstance(row, dict):
                rows.append({
                    "Start": row.get("Start", row.get("start")),
                    "End": row.get("End", row.get("end")),
                })
        return rows
    return []


def _merge_excluded_region_editor_state(base_rows, editor_state) -> list[dict]:
    rows = _coerce_excluded_region_rows(base_rows)
    if not isinstance(editor_state, dict):
        return rows

    edited_rows = editor_state.get("edited_rows") or {}
    for raw_idx, patch in edited_rows.items():
        try:
            idx = int(raw_idx)
        except Exception:
            continue
        while idx >= len(rows):
            rows.append({"Start": None, "End": None})
        if isinstance(patch, dict):
            if "Start" in patch or "start" in patch:
                rows[idx]["Start"] = patch.get("Start", patch.get("start"))
            if "End" in patch or "end" in patch:
                rows[idx]["End"] = patch.get("End", patch.get("end"))

    added_rows = editor_state.get("added_rows") or []
    for row in added_rows:
        if isinstance(row, dict):
            rows.append({
                "Start": row.get("Start", row.get("start")),
                "End": row.get("End", row.get("end")),
            })

    deleted_rows = editor_state.get("deleted_rows") or []
    for raw_idx in sorted((int(i) for i in deleted_rows if str(i).strip().isdigit()), reverse=True):
        if 0 <= raw_idx < len(rows):
            rows.pop(raw_idx)

    return rows


def _format_region(lo: float, hi: float, axis_label: str) -> str:
    return f"[{lo:.6f}, {hi:.6f}] {axis_label}"


def _pack_phase_count(pack: dict) -> int | None:
    manifest = pack.get("manifest") or {}
    if manifest.get("kind") == "augmented":
        added = manifest.get("n_added_phases")
        base = manifest.get("n_base_phases")
        if added is not None and base is not None:
            return int(base) + int(added)
    n_phases = manifest.get("n_phases")
    if n_phases is not None:
        return int(n_phases)
    return None


@st.cache_data(show_spinner=False, ttl=60)
def cached_catalog_phase_count(catalog_csv: str, mtime: float | None) -> int | None:
    try:
        with open(catalog_csv, "rb") as fh:
            n_lines = sum(1 for _ in fh)
        return max(0, n_lines - 1)
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60)
def cached_catalog_elements(catalog_csv: str, mtime: float | None) -> list[str]:
    elements: set[str] = set()
    try:
        with open(catalog_csv, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = str(row.get("elements_list", "") or "")
                for token in raw.split(","):
                    token = token.strip()
                    if token:
                        elements.add(token)
    except Exception:
        return []
    return sorted(elements)


def _active_db_phase_count() -> int | None:
    catalog_csv = ACTIVE_DB_CONFIG.get("catalog_csv")
    if catalog_csv and Path(catalog_csv).exists():
        count = cached_catalog_phase_count(str(catalog_csv), _path_mtime(catalog_csv))
        if count is not None:
            return count

    loader = st.session_state.get("db_loader")
    if loader is None:
        return None
    try:
        return int(len(loader.catalog))
    except Exception:
        return None


def _derive_allowed_elements_from_active_db() -> list[str]:
    catalog_csv = ACTIVE_DB_CONFIG.get("catalog_csv")
    if catalog_csv and Path(catalog_csv).exists():
        return cached_catalog_elements(str(catalog_csv), _path_mtime(catalog_csv))
    return []


@st.cache_data(show_spinner=False, ttl=10)
def discover_user_db_packs(source_label: str) -> list[dict]:
    source_key = _source_key_from_label(source_label)
    packs: list[dict] = []
    if not USER_DB_PACKS_DIR.exists():
        return packs

    for pack_group in sorted(USER_DB_PACKS_DIR.iterdir()):
        if not pack_group.is_dir():
            continue
        pack_root = pack_group / source_key
        if not pack_root.is_dir():
            continue

        layout = get_db_pack_layout(pack_root)
        manifest = {}
        if layout.manifest_json.exists():
            try:
                manifest = json.loads(layout.manifest_json.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}

        kind = str(manifest.get("kind", "custom"))
        original_json = None
        if kind == "augmented":
            original_json = manifest.get("base_original_json")
        elif layout.original_json.exists():
            original_json = layout.original_json

        cif_map = layout.cif_map_json if layout.cif_map_json.exists() else None
        try:
            db_cfg = build_db_config(
                layout.root,
                original_json=original_json,
                cif_map_json=cif_map,
            )
        except Exception:
            continue
        if not _db_config_is_usable(db_cfg):
            continue
        n_phases = _pack_phase_count({"manifest": manifest})

        packs.append({
            "name": pack_group.name,
            "root": layout.root,
            "root_str": str(layout.root),
            "kind": kind,
            "manifest": manifest,
            "db_config": db_cfg,
            "label": (
                f"{pack_group.name} [{kind}, {n_phases} phases]"
                if n_phases is not None
                else f"{pack_group.name} [{kind}]"
            ),
            "n_phases": n_phases,
        })
    return packs


def resolve_active_db_selection(source_label: str) -> dict:
    source_key = _source_key_from_label(source_label)
    builtin_root = DB_XRAY_DIR if source_key == "xray" else DB_NEUTRON_DIR
    builtin_cfg = build_db_config(
        builtin_root,
        original_json=builtin_root / "highsymm_metadata.json",
    )
    builtin = {
        "root": builtin_root,
        "db_config": builtin_cfg,
        "label": f"Built-in {source_label}",
        "kind": "original",
        "selection_key": f"{source_label}:original:{builtin_root}",
    }

    selection_mode = st.session_state.get("db_selection_mode", "Original")
    if selection_mode == "Original":
        return builtin

    packs = discover_user_db_packs(source_label)
    selected_root = st.session_state.get(_selected_pack_state_key(source_label))
    selected = next((p for p in packs if p["root_str"] == selected_root and _db_mode_matches_kind(selection_mode, p["kind"])), None)
    if selected is None:
        return builtin

    return {
        "root": selected["root"],
        "db_config": selected["db_config"],
        "label": selected["label"],
        "kind": selected["kind"],
        "selection_key": f"{source_label}:{selected['kind']}:{selected['root_str']}",
    }


@st.cache_resource(show_spinner=False)
def load_cached_db_loader(
    selection_key: str,
    catalog_csv: str,
    cif_map_json: str | None,
    original_json: str | None,
    stable_csv: str | None,
    catalog_mtime: float | None,
    cif_map_mtime: float | None,
    original_mtime: float | None,
    stable_mtime: float | None,
) -> DBLoader:
    """Load and cache a phase catalog for the active radiation/library selection."""
    paths = CatalogPaths(
        catalog_csv=str(catalog_csv),
        cif_map_json=cif_map_json,
        original_json=original_json if original_json and os.path.exists(original_json) else None,
    )
    loader = DBLoader(paths)
    if stable_csv and Path(stable_csv).exists():
        loader.attach_stable_catalog(str(stable_csv))
    return loader


@st.cache_data(show_spinner=False, ttl=5)
def cached_find_run_manifest_path(run_dir: str) -> str | None:
    rdir = Path(run_dir)
    preferred = [
        rdir / "run_manifest.json",
        rdir / "Technical" / "Logs" / "run_manifest.json",
    ]
    for path in preferred:
        if path.exists():
            return str(path)
    path = next(rdir.rglob("run_manifest.json"), None)
    return str(path) if path else None


@st.cache_data(show_spinner=False, ttl=5)
def cached_read_json_file(path: str, mtime: float | None) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False, ttl=10)
def cached_read_yaml_file(path: str, mtime: float | None) -> dict:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


@st.cache_data(show_spinner=False, ttl=5)
def cached_run_csv_files(run_dir: str) -> list[tuple[str, float]]:
    rdir = Path(run_dir)
    files = []
    for path in rdir.rglob("*.csv"):
        try:
            files.append((str(path), path.stat().st_mtime))
        except OSError:
            continue
    return sorted(files, key=lambda item: item[1], reverse=True)


@st.cache_data(show_spinner=False, ttl=10)
def cached_read_csv_file(path: str, mtime: float | None):
    if pd is None:
        raise ImportError("pandas is not available")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, ttl=10)
def cached_recent_interactive_runs(runs_root: str, limit: int = 30) -> list[dict]:
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []

    runs: list[dict] = []
    try:
        run_dirs = [
            item for item in runs_root.iterdir()
            if item.is_dir()
        ]
        run_dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    except OSError:
        return []

    # Avoid a blank-feeling tab caused by recursively scanning every historical run.
    # Recent runs are what users normally inspect, and the function is cached briefly.
    for run_dir in run_dirs[: max(int(limit) * 3, int(limit))]:
        if not run_dir.is_dir():
            continue
        try:
            payloads = list(run_dir.rglob("*.plotdata.json"))
        except OSError:
            continue
        if not payloads:
            continue

        newest = 0.0
        for payload_path in payloads:
            try:
                newest = max(newest, payload_path.stat().st_mtime)
            except OSError:
                continue
        if newest <= 0:
            continue
        runs.append(
            {
                "name": run_dir.name,
                "path": str(run_dir),
                "plot_count": len(payloads),
                "mtime": newest,
                "label": f"{run_dir.name} ({len(payloads)} plots, {time.strftime('%Y-%m-%d %H:%M', time.localtime(newest))})",
            }
        )

    runs.sort(key=lambda item: item["mtime"], reverse=True)
    return runs[:limit]


@st.cache_data(show_spinner=False, ttl=10)
def cached_reusable_run_configs(runs_root: str, limit: int = 80) -> list[dict]:
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return []
    entries: list[dict] = []
    for cfg_path in runs_root.glob("*/pipeline_config.yaml"):
        try:
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            dataset = (cfg.get("datasets") or [{}])[0] or {}
            data_path = Path(str(dataset.get("data_path") or ""))
            inst_path = Path(str(dataset.get("instprm_path") or ""))
            if not data_path.exists() or not inst_path.exists():
                continue
            main_cif_raw = dataset.get("main_cif")
            main_cif_path = Path(str(main_cif_raw)) if main_cif_raw else None
            if main_cif_path is not None and not main_cif_path.exists():
                main_cif_path = None
            mtime = cfg_path.stat().st_mtime
            mode = "Rapid" if (cfg.get("analysis_mode") == "rapid_hypothesis" or (cfg.get("rapid_hypothesis") or {}).get("enabled")) else "Full"
            stage4 = cfg.get("stage4") or {}
            radiation = str(stage4.get("radiation") or "").strip().title() or "-"
            mtime_text = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
            data_name = data_path.name
            inst_name = inst_path.name
            main_name = main_cif_path.name if main_cif_path else "none"
            entries.append(
                {
                    "label": f"{cfg_path.parent.name} | {mode} | {radiation} | {mtime_text}",
                    "run_name": cfg_path.parent.name,
                    "run_dir": str(cfg_path.parent),
                    "config_path": str(cfg_path),
                    "mtime": mtime,
                    "mtime_text": mtime_text,
                    "mode": mode,
                    "radiation": radiation,
                    "data_path": str(data_path),
                    "data_name": data_name,
                    "instprm_path": str(inst_path),
                    "instprm_name": inst_name,
                    "main_cif": str(main_cif_path) if main_cif_path else "",
                    "main_cif_name": main_name,
                    "search_text": " ".join(
                        [
                            cfg_path.parent.name,
                            mode,
                            radiation,
                            mtime_text,
                            data_name,
                            inst_name,
                            main_name,
                        ]
                    ).lower(),
                }
            )
        except Exception:
            continue
    entries.sort(key=lambda item: item["mtime"], reverse=True)
    return entries[:limit]


@st.cache_data(show_spinner=False, ttl=5)
def cached_workspace_run_entries(runs_root: str, limit: int = 80) -> list[dict]:
    runs_root_path = Path(runs_root)
    if not runs_root_path.exists():
        return []
    entries: list[dict] = []
    for run_dir in runs_root_path.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "pipeline_config.yaml"
        log_path = run_dir / "pipeline.log"
        summary_path = run_dir / "rapid_results" / "summary.json"
        manifest_paths = list(run_dir.rglob("run_manifest.json"))
        try:
            mtime = max(
                [
                    path.stat().st_mtime
                    for path in [cfg_path, log_path, summary_path, *manifest_paths]
                    if path.exists()
                ]
                or [run_dir.stat().st_mtime]
            )
        except OSError:
            mtime = time.time()
        mode = "Unknown"
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                mode = "Rapid" if (cfg.get("analysis_mode") == "rapid_hypothesis" or (cfg.get("rapid_hypothesis") or {}).get("enabled")) else "Full"
            except Exception:
                mode = "Unknown"
        elif (run_dir / "rapid_results").exists():
            mode = "Rapid"
        plot_count = 0
        csv_count = 0
        try:
            plot_count = sum(1 for _ in run_dir.rglob("*.plotdata.json"))
            csv_count = sum(1 for _ in run_dir.rglob("*.csv"))
        except OSError:
            pass
        failure_info = _run_failure_info(run_dir)
        manifest_status = str((_read_run_manifest(run_dir) or {}).get("status") or "").strip().lower()
        status = "Running" if not summary_path.exists() and log_path.exists() and not manifest_paths else "Saved"
        if failure_info:
            status = "Failed"
        elif manifest_status in {"running", "starting", "processing"}:
            status = "Running"
        elif summary_path.exists() or manifest_paths:
            status = "Complete"
        entries.append(
            {
                "name": run_dir.name,
                "path": str(run_dir),
                "mode": mode,
                "status": status,
                "mtime": mtime,
                "plot_count": int(plot_count),
                "csv_count": int(csv_count),
                "label": (
                    f"{run_dir.name} | {mode} | {status} | "
                    f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}"
                ),
            }
        )
    entries.sort(key=lambda item: item["mtime"], reverse=True)
    return entries[:limit]


def _compact_middle_text(text: object, max_chars: int = 28, *, min_tail: int = 6) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    if max_chars <= 8:
        return value[: max(1, max_chars - 3)] + "..."
    marker = "..."
    available = max_chars - len(marker)
    tail_len = min(max(min_tail, available // 2), available - 4)
    head_len = max(4, available - tail_len)
    return f"{value[:head_len]}{marker}{value[-tail_len:]}"


def _compact_run_name(name: str, max_chars: int = 24) -> str:
    text = str(name or "").strip()
    if text.startswith("run_"):
        return _compact_middle_text(text, max_chars=max_chars, min_tail=6)
    return _compact_middle_text(text, max_chars=max_chars, min_tail=6)


def _saved_run_sidebar_label(entry: dict) -> str:
    name = _compact_run_name(str(entry.get("name") or Path(str(entry.get("path") or "")).name), max_chars=24)
    return name


def _reusable_run_sidebar_label(entry: dict) -> str:
    name = _compact_run_name(str(entry.get("run_name") or Path(str(entry.get("run_dir") or "")).name), max_chars=24)
    return name


def _apply_reused_config_to_session(cfg: dict) -> None:
    dataset = (cfg.get("datasets") or [{}])[0] or {}
    allowed = cfg.get("allowed_elements") or []
    env = ((cfg.get("element_filter") or {}).get("sample_env") or {}).get("elements") or []
    st.session_state.allowed_elements_input = ", ".join(str(x) for x in allowed)
    st.session_state.sample_env_elements_input = ", ".join(str(x) for x in env)

    mode = str(dataset.get("mode") or cfg.get("instrument_mode") or "auto").upper()
    st.session_state.reused_instrument_mode = "Auto" if mode == "AUTO" else mode

    limits = dataset.get("limits")
    if isinstance(limits, (list, tuple)) and len(limits) == 2:
        st.session_state.fit_window_override_enabled = True
        st.session_state.fit_window_lower = str(limits[0])
        st.session_state.fit_window_upper = str(limits[1])
    else:
        st.session_state.fit_window_override_enabled = False
        st.session_state.fit_window_lower = ""
        st.session_state.fit_window_upper = ""

    excluded_rows = []
    for pair in dataset.get("exclude_regions") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            excluded_rows.append({"Start": pair[0], "End": pair[1]})
    st.session_state.excluded_regions_rows = excluded_rows
    st.session_state.excluded_regions_editor_buffer = excluded_rows

    background = cfg.get("background") or {}
    bg_mode = str(background.get("mode") or "auto_fixed_points")
    st.session_state.bg_mode_label = "Auto Fixed Points" if bg_mode == "auto_fixed_points" else "Function"
    st.session_state.bg_type = str(background.get("type") or "chebyschev-1")
    st.session_state.bg_terms = int(background.get("terms") or 6)

    ref_masks = cfg.get("reference_phase_exclusions") or {}
    st.session_state.auto_reference_phase_mask_enabled = bool(ref_masks.get("enabled", False))
    st.session_state.reference_mask_presets = list(ref_masks.get("presets") or [])
    if "include_cu_kbeta" in ref_masks:
        st.session_state.reference_phase_mask_include_kbeta = bool(ref_masks.get("include_cu_kbeta"))

    light_cal = cfg.get("light_calibration") or {}
    if "enabled" in light_cal:
        st.session_state.light_pxrd_calibration_enabled = bool(light_cal.get("enabled"))

    magnetic = cfg.get("magnetic_precheck") or {}
    if "enabled" in magnetic:
        st.session_state.magnetic_precheck_enabled = bool(magnetic.get("enabled"))
    if "q_max" in magnetic:
        st.session_state.magnetic_precheck_q_max = float(magnetic.get("q_max") or 6.0)
    if "denominators" in magnetic:
        st.session_state.magnetic_precheck_denominators = ",".join(str(v) for v in (magnetic.get("denominators") or [2, 3]))

    shadow = cfg.get("main_phase_shadow") or {}
    if "enabled" in shadow:
        st.session_state.main_phase_shadow_filter_enabled = bool(shadow.get("enabled"))
    if "nudge_filter_enabled" in shadow:
        st.session_state.main_phase_shadow_filter_enabled = bool(shadow.get("nudge_filter_enabled"))

    cleanup = cfg.get("main_phase_cleanup") or {}
    cleanup_enabled = bool(cleanup.get("enabled", False))
    if "enabled" in cleanup:
        st.session_state.main_phase_cleanup_enabled = cleanup_enabled
    if "refine_u_iso" in cleanup:
        st.session_state.main_phase_cleanup_refine_u_iso = bool(cleanup.get("refine_u_iso")) if cleanup_enabled else False
    if "refine_positions" in cleanup:
        st.session_state.main_phase_cleanup_refine_positions = bool(cleanup.get("refine_positions")) if cleanup_enabled else False

    rapid = cfg.get("rapid_hypothesis") or {}
    st.session_state.analysis_mode = "Rapid Hypothesis Mode" if rapid.get("enabled") or cfg.get("analysis_mode") == "rapid_hypothesis" else "Full RADAR-PD"
    if rapid.get("beam_depth"):
        st.session_state.rapid_hypothesis_phase_count = int(rapid.get("beam_depth"))
    if rapid.get("stage_output_limit"):
        st.session_state.rapid_stage_output_limit = int(rapid.get("stage_output_limit"))
    if rapid.get("gsas_validation_limit"):
        st.session_state.rapid_gsas_validation_limit = int(rapid.get("gsas_validation_limit"))
    if rapid.get("gsas_parallel_workers"):
        st.session_state.rapid_gsas_parallel_workers = int(rapid.get("gsas_parallel_workers"))
    if "final_polish_enabled" in rapid:
        st.session_state.rapid_final_polish_enabled = bool(rapid.get("final_polish_enabled"))
    if "show_family_variants" in rapid:
        st.session_state.rapid_enable_family_variants = bool(rapid.get("show_family_variants"))


def _run_config_summary_items(cfg: dict) -> list[tuple[str, str]]:
    dataset = (cfg.get("datasets") or [{}])[0] or {}
    background = cfg.get("background") or {}
    rapid = cfg.get("rapid_hypothesis") or {}
    magnetic = cfg.get("magnetic_precheck") or {}
    cleanup = cfg.get("main_phase_cleanup") or {}
    shadow = cfg.get("main_phase_shadow") or {}
    xray_doublet = cfg.get("xray_doublet") or {}
    db = cfg.get("db") or {}
    env = ((cfg.get("element_filter") or {}).get("sample_env") or {}).get("elements") or []
    analysis_label = "Rapid Hypothesis Mode" if rapid.get("enabled") or cfg.get("analysis_mode") == "rapid_hypothesis" else "Full RADAR-PD"
    main_info = _rapid_main_phase_info_from_cif_path(dataset.get("main_cif"))
    main_cif_display = main_info.get("label") or (Path(str(dataset.get("main_cif") or "")).name if dataset.get("main_cif") else "none")
    items = [
        ("Run", str(dataset.get("name") or "-")),
        ("Analysis", analysis_label),
        ("Data", Path(str(dataset.get("data_path") or "-")).name),
        ("Instrument", Path(str(dataset.get("instprm_path") or "-")).name),
        ("Main phase", main_cif_display),
        ("Mode", str(dataset.get("mode") or cfg.get("instrument_mode") or "auto")),
        ("Elements", ", ".join(str(x) for x in (cfg.get("allowed_elements") or [])) or "[blank]"),
        ("Can/environment", ", ".join(str(x) for x in env) or "none"),
        ("Library", Path(str(db.get("catalog_csv") or "-")).parent.name),
        ("Background", f"{background.get('mode', '-')}, {background.get('type', '-')}, {background.get('terms', '-')} terms"),
        ("Magnetic precheck", "on" if magnetic.get("enabled") else "off"),
        ("Main-CIF cleanup", "on" if cleanup.get("enabled") else "off"),
        ("Main-phase lookalike filter", "on" if shadow.get("enabled", True) and shadow.get("nudge_filter_enabled", True) else "off"),
        ("PXRD doublet", str(xray_doublet.get("enabled", "auto"))),
        ("Fit window", ", ".join(str(x) for x in dataset.get("limits", [])) if dataset.get("limits") else "full pattern"),
        ("Ignored regions", str(len(dataset.get("exclude_regions") or []))),
    ]
    if analysis_label == "Rapid Hypothesis Mode":
        items.extend([
            ("Rapid phases", str(rapid.get("beam_depth", "-"))),
            ("Stage rows", str(rapid.get("stage_output_limit", "-"))),
            ("Refinement checks", str(rapid.get("gsas_validation_limit", "-"))),
        ])
    else:
        items.extend([
            ("Max passes", str(cfg.get("max_passes", "-"))),
            ("Min phase", f"{cfg.get('min_impurity_percent', '-')} wt%"),
            ("Candidates/pass", str(cfg.get("top_candidates", "-"))),
        ])
    return items


def render_run_config_summary(config_path: Path, *, title: str = "Configuration", expanded: bool = True) -> None:
    if not config_path.exists():
        st.caption("No saved configuration is available yet.")
        return
    cfg = cached_read_yaml_file(str(config_path), _path_mtime(config_path))
    if not cfg:
        st.caption("Saved configuration could not be read.")
        return
    with st.expander(title, expanded=expanded):
        for label, value in _run_config_summary_items(cfg):
            st.caption(f"**{label}:** {value}")


def _plot_payload_group(payload: dict) -> str:
    kind = payload.get("plot_kind", "")
    if kind == "gsas_fit_with_ticks_v1":
        return "Fit / refinement"
    if kind == "ml_hist_grid_v1":
        return "ML screening histograms"
    if kind == "residual_bin_debug_v1":
        return "Residual debug"
    return "Other"


def _plot_pass_number(text_value: str) -> int:
    match = re.search(r"pass(\d+)", text_value or "", flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _plot_display_label(payload_path: Path, payload: dict, run_dir: Path) -> str:
    source = str(payload.get("source_plot") or payload_path.name.replace(".plotdata.json", ""))
    title = str(payload.get("title") or "").strip()
    group = _plot_payload_group(payload)
    lower = source.lower()
    pass_ix = _plot_pass_number(source) or _plot_pass_number(title)

    if group == "Fit / refinement":
        if "main_phase" in lower:
            label = "Main phase fit"
        elif "accepted" in lower:
            label = f"Accepted model - pass {pass_ix}" if pass_ix else "Accepted model"
        elif "trial" in lower or "blend" in lower:
            label = f"Trial blend - pass {pass_ix}" if pass_ix else "Trial blend"
        else:
            label = title or source
    elif group == "ML screening histograms":
        stage = "stage0" if "stage0" in str(payload_path).lower() else (f"pass {pass_ix}" if pass_ix else "screening")
        label = f"ML candidate histogram - {stage}"
    elif group == "Residual debug":
        label = title or "Residual debug"
    else:
        try:
            label = str(payload_path.relative_to(run_dir))
        except Exception:
            label = source

    return label


def _plot_sort_key(info: dict) -> tuple:
    group_rank = {
        "Fit / refinement": 0,
        "ML screening histograms": 1,
        "Residual debug": 2,
        "Other": 3,
    }.get(info["group"], 9)
    name = info["path"].name.lower()
    pass_ix = info.get("pass_ix", 0)
    if "main_phase" in name:
        subtype = 0
    elif "trial" in name or "blend" in name:
        subtype = 1
    elif "accepted" in name:
        subtype = 2
    else:
        subtype = 3
    return (group_rank, pass_ix, subtype, str(info["path"]).lower())


def _read_plot_payload_header(payload_path: Path) -> dict:
    try:
        with open(payload_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        data.pop("arrays", None)
        return data
    except Exception:
        return {"plot_kind": "", "title": payload_path.name}


def _render_plot_meta_cards(
    items: list[tuple[str, object]],
    *,
    numeric_labels: set[str] | None = None,
    compact: bool = False,
) -> None:
    numeric_labels = numeric_labels or set()
    cells = []
    for label, value in items:
        value_text = str(value)
        display_text = value_text
        classes = ["radar-plot-kpi-value"]
        if label in numeric_labels:
            classes.append("numeric")
        elif len(value_text) > 14:
            classes.append("long")
            if len(value_text) > 34:
                display_text = value_text[:31].rstrip() + "..."
        cells.append(
            "<div class=\"radar-plot-kpi\">"
            f"<div class=\"radar-plot-kpi-label\">{html.escape(str(label))}</div>"
            f"<div class=\"{' '.join(classes)}\" title=\"{html.escape(value_text)}\">"
            f"{html.escape(display_text)}</div>"
            "</div>"
        )
    wrapper_class = "radar-plot-kpis radar-plot-kpis-compact" if compact else "radar-plot-kpis"
    st.markdown(f"<div class=\"{wrapper_class}\">{''.join(cells)}</div>", unsafe_allow_html=True)


def _rapid_fmt_number(value, digits: int = 3) -> str:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _rapid_to_int(value) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _rapid_display_rank(value, *, manual_threshold: int = 9000) -> str:
    rank = _rapid_to_int(value)
    if rank is None:
        return "-"
    return "Manual" if rank >= manual_threshold else str(rank)


def _rapid_to_float(value) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return None if np.isnan(out) else out
    except Exception:
        return None


def _rapid_split_phases(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _rapid_needs_formula(value: object, pid: str | None = None) -> bool:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "unknown"}:
        return True
    if pid is not None and text == str(pid):
        return True
    return bool(re.fullmatch(r"(?:mp|mvc|cod)-\d+", text, flags=re.IGNORECASE))


def _rapid_render_formula_amount(amount: float) -> str:
    rounded = round(float(amount))
    if abs(float(amount) - rounded) < 0.03:
        amount = float(max(1, rounded))
    if abs(amount - 1.0) < 1e-8:
        return ""
    return str(int(amount)) if float(amount).is_integer() else f"{amount:.4g}"


def _rapid_formula_from_composition(comp: object) -> str | None:
    if not isinstance(comp, dict) or not comp:
        return None
    vals: list[tuple[str, float]] = []
    for elem, amount in comp.items():
        try:
            val = float(amount)
        except Exception:
            continue
        if val > 0:
            vals.append((str(elem), val))
    if not vals:
        return None
    scale = min(val for _elem, val in vals if val > 0)
    parts = [f"{elem}{_rapid_render_formula_amount(val / scale)}" for elem, val in vals]
    return "".join(parts) or None


def _rapid_formula_from_cif_content(cif_text: object) -> str | None:
    text = str(cif_text or "")
    if not text:
        return None
    for tag in ("_chemical_formula_structural", "_chemical_formula_sum"):
        match = re.search(rf"^{re.escape(tag)}\s+(.+?)\s*$", text, flags=re.MULTILINE)
        if not match:
            continue
        raw = match.group(1).strip().strip("'\"")
        if not raw:
            continue
        if tag == "_chemical_formula_structural" and not _rapid_needs_formula(raw):
            return raw.replace(" ", "")
        parts = re.findall(r"([A-Z][a-z]?)\s*([0-9]*\.?[0-9]*)", raw)
        if parts:
            formula = _rapid_formula_from_composition(
                {elem: float(amount) if amount else 1.0 for elem, amount in parts}
            )
            if formula:
                return formula
    return None


def _rapid_formula_from_metadata_record(record: object) -> str | None:
    if not isinstance(record, dict):
        return None
    for key in ("formula_pretty", "pretty_formula", "formula", "pretty_name"):
        val = str(record.get(key) or "").strip()
        if not _rapid_needs_formula(val):
            return val
    formula = _rapid_formula_from_cif_content(record.get("cif_content"))
    if formula:
        return formula
    return _rapid_formula_from_composition(record.get("composition"))


def _rapid_formula_key(formula: object) -> str:
    text = str(formula or "").strip()
    if _rapid_needs_formula(text):
        return text or "unknown"
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    if not parts:
        return text
    vals: list[tuple[str, float]] = []
    for elem, amount in parts:
        try:
            vals.append((elem, float(amount) if amount else 1.0))
        except Exception:
            return text
    min_val = min((v for _e, v in vals if v > 0), default=1.0)
    return "".join(f"{elem}{_rapid_render_formula_amount(val / min_val)}" for elem, val in vals) or text


RAPID_STOICHIOMETRY_FAMILY_TOLERANCE = 0.05


def _rapid_formula_ratio_signature(formula: object) -> tuple[tuple[str, float], ...]:
    text = str(formula or "").strip()
    parts = re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", text)
    vals: list[tuple[str, float]] = []
    for elem, amount in parts:
        try:
            val = float(amount) if amount else 1.0
        except Exception:
            continue
        if val > 0:
            vals.append((elem, val))
    if not vals:
        return ((text or "unknown", 1.0),)
    min_val = min(val for _elem, val in vals)
    return tuple(sorted((elem, round(val / min_val, 3)) for elem, val in vals))


def _rapid_formula_ratio_map(formula: object) -> dict[str, float]:
    signature = _rapid_formula_ratio_signature(formula)
    return {elem: float(value) for elem, value in signature if re.match(r"^[A-Z][a-z]?$", str(elem))}


def _rapid_pure_element_from_ratios(ratios: dict[str, float]) -> str:
    if len(ratios) != 1:
        return ""
    return next(iter(ratios))


def _rapid_near_elemental_dominant_from_ratios(
    ratios: dict[str, float],
    *,
    trace_fraction_max: float = 0.02,
) -> str:
    if len(ratios) <= 1:
        return ""
    total = float(sum(max(0.0, value) for value in ratios.values()))
    if total <= 0:
        return ""
    elem, amount = max(ratios.items(), key=lambda item: item[1])
    if float(amount) / total >= (1.0 - float(trace_fraction_max)):
        return elem
    return ""


def _rapid_same_stoichiometric_family(
    formula_a: object,
    sg_a: object,
    formula_b: object,
    sg_b: object,
    *,
    tolerance: float = RAPID_STOICHIOMETRY_FAMILY_TOLERANCE,
) -> bool:
    ratios_a = _rapid_formula_ratio_map(formula_a)
    ratios_b = _rapid_formula_ratio_map(formula_b)
    pure_a = _rapid_pure_element_from_ratios(ratios_a)
    pure_b = _rapid_pure_element_from_ratios(ratios_b)
    near_a = _rapid_near_elemental_dominant_from_ratios(ratios_a)
    near_b = _rapid_near_elemental_dominant_from_ratios(ratios_b)
    if (pure_a and near_b and pure_a == near_b) or (pure_b and near_a and pure_b == near_a):
        return True
    sg_a_text = _rapid_sg_value(sg_a)
    sg_b_text = _rapid_sg_value(sg_b)
    if sg_a_text and sg_b_text and sg_a_text != sg_b_text:
        return False
    if not ratios_a or set(ratios_a) != set(ratios_b):
        return False
    for elem in ratios_a:
        a = float(ratios_a[elem])
        b = float(ratios_b[elem])
        denom = max(abs(a), abs(b), 1e-12)
        if abs(a - b) / denom > float(tolerance):
            return False
    return True


def _rapid_formula_niceness_score(formula: object) -> tuple[float, float, str]:
    ratios = _rapid_formula_ratio_map(formula)
    if not ratios:
        return (999.0, 999.0, str(formula or ""))
    values = list(ratios.values())
    integer_error = sum(abs(value - round(value)) for value in values)
    integer_sum = sum(max(1.0, round(value)) for value in values)
    return (float(integer_error), float(integer_sum), str(formula or ""))


def _rapid_sg_value(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "0", "-"}:
        return ""
    try:
        number = int(float(text))
        return str(number) if number > 0 else ""
    except Exception:
        return text


def _rapid_display_formula(formula: object, sg: object = "") -> str:
    label = str(formula or "").strip() or "-"
    sg_text = _rapid_sg_value(sg)
    return f"{label} (SG {sg_text})" if sg_text else label


def _rapid_space_group_from_cif_content(cif_text: object) -> str:
    text = str(cif_text or "")
    if not text:
        return ""
    for tag in ("_space_group_IT_number", "_symmetry_Int_Tables_number", "_space_group.it_number"):
        match = re.search(rf"^{re.escape(tag)}\s+([0-9]+)", text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            return _rapid_sg_value(match.group(1))
    return ""


def _rapid_main_phase_info_from_cif_path(cif_path: object) -> dict[str, str]:
    path = Path(str(cif_path or ""))
    if not cif_path or not path.exists():
        return {}
    formula = ""
    sg = ""
    try:
        from pymatgen.io.cif import CifParser
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        structure = CifParser(str(path), occupancy_tolerance=2.0).parse_structures(primitive=False)[0]
        try:
            formula = str(structure.composition.reduced_formula)
        except Exception:
            formula = ""
        try:
            sg = _rapid_sg_value(SpacegroupAnalyzer(structure, symprec=0.1).get_space_group_number())
        except Exception:
            sg = ""
    except Exception:
        pass
    try:
        cif_text = path.read_text(errors="ignore")
    except Exception:
        cif_text = ""
    if not formula:
        formula = _rapid_formula_from_cif_content(cif_text) or path.stem
    if not sg:
        sg = _rapid_space_group_from_cif_content(cif_text)
    label = _rapid_display_formula(formula, sg)
    return {
        "label": label,
        "formula": str(formula),
        "formula_key": _rapid_formula_key(formula),
        "space_group": sg,
        "path": str(path),
    }


def _rapid_main_phase_info_from_config_path(config_path: Path | None) -> dict[str, str]:
    if config_path is None or not Path(config_path).exists():
        return {}
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    datasets = cfg.get("datasets") or []
    dataset = datasets[0] if isinstance(datasets, list) and datasets else {}
    if not isinstance(dataset, dict):
        return {}
    for key in ("main_cif_cleanup_path", "main_cif_prenudged_path", "main_cif"):
        info = _rapid_main_phase_info_from_cif_path(dataset.get(key))
        if info.get("label"):
            return info
    return {}


def _rapid_report_root_from_output_path(path: Path) -> Path | None:
    for parent in [path.parent, *path.parents]:
        if parent.name == "rapid_results":
            return parent
    return None


def _rapid_main_phase_info_from_output_path(path: Path) -> dict[str, str]:
    report_root = _rapid_report_root_from_output_path(path)
    if report_root is not None:
        summary_path = report_root / "summary.json"
        if summary_path.exists():
            try:
                summary = cached_read_json_file(str(summary_path), _path_mtime(summary_path))
                for value in summary.values():
                    if isinstance(value, dict):
                        info = value.get("main_phase") or value.get("main_phase_display")
                        if isinstance(info, dict) and str(info.get("label") or "").strip():
                            return {str(k): str(v) for k, v in info.items() if v is not None}
            except Exception:
                pass
        info = _rapid_main_phase_info_from_config_path(report_root.parent / "pipeline_config.yaml")
        if info:
            return info
    return _rapid_main_phase_info_from_config_path(_rapid_run_config_path_for_output(path))


def _rapid_main_phase_label_from_output_path(path: Path) -> str:
    info = _rapid_main_phase_info_from_output_path(path)
    label = str(info.get("label") or "").strip()
    return label if label and label != "Main phase" else ""


def _rapid_replace_main_phase_label(value: object, label: str) -> object:
    if not label:
        return value
    text = str(value or "")
    return label if text == "Main phase" else value


def _rapid_choice_formula(choice: object) -> str:
    text = str(choice or "").strip()
    return re.sub(r"\s*\(SG\s+[^)]*\)\s*$", "", text).strip()


def _rapid_choice_sg(choice: object) -> str:
    match = re.search(r"\(SG\s+([^)]*)\)\s*$", str(choice or "").strip())
    return _rapid_sg_value(match.group(1)) if match else ""


def _rapid_phase_signature(formula: object, sg: object = "") -> tuple[tuple[tuple[str, float], ...], str]:
    return (_rapid_formula_ratio_signature(formula), _rapid_sg_value(sg))


def _rapid_space_groups_from_row(row) -> list[str]:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    pids = _rapid_split_phases(getter("phase_ids") or getter("phase_id"))
    expected_count = len(pids)
    if expected_count == 0:
        expected_count = len(
            _rapid_split_phases(
                getter("formulas")
                or getter("formula_keys")
                or getter("formula")
                or getter("formula_key")
            )
        )
    sgs = [_rapid_sg_value(item) for item in _rapid_split_phases(getter("space_groups") or getter("space_group"))]
    if expected_count == 0:
        return sgs
    if len(sgs) >= expected_count:
        return sgs[:expected_count]
    out = list(sgs)
    try:
        db = get_active_db_loader()
    except Exception:
        db = None
    for pid in pids[len(out):expected_count]:
        sg_val = ""
        if db is not None:
            try:
                sg_val = _rapid_sg_value(db.get_space_group_number(str(pid)))
            except Exception:
                sg_val = ""
        out.append(sg_val)
    while len(out) < expected_count:
        out.append("")
    return out


def _rapid_phase_labels_from_row(row) -> list[str]:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    phases = _rapid_split_phases(
        getter("formulas")
        or getter("formula_keys")
        or getter("formula")
        or getter("formula_key")
        or getter("phase_ids")
        or getter("phase_id")
    )
    sgs = _rapid_space_groups_from_row(row)
    return [
        _rapid_display_formula(phase, sgs[idx] if idx < len(sgs) else "")
        for idx, phase in enumerate(phases)
    ]


def _rapid_run_config_path_for_output(path: Path) -> Path | None:
    for parent in [path.parent, *path.parents]:
        cfg = parent / "pipeline_config.yaml"
        if cfg.exists():
            return cfg
    return None


@st.cache_data(show_spinner=False, ttl=300)
def _rapid_metadata_formula_lookup(metadata_json: str, mtime: float | None) -> dict[str, str]:
    try:
        with open(metadata_json, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    lookup: dict[str, str] = {}
    for pid, record in data.items():
        formula = _rapid_formula_from_metadata_record(record)
        if formula:
            lookup[str(pid)] = formula
    return lookup


@st.cache_data(show_spinner=False, ttl=60)
def _rapid_metadata_path_from_config(config_path: str, mtime: float | None) -> str | None:
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    path = ((cfg.get("db") or {}).get("original_json") or "").strip()
    if path and Path(path).exists():
        return path
    return None


def _rapid_formula_lookup_for_output(path: Path) -> dict[str, str]:
    cfg_path = _rapid_run_config_path_for_output(path)
    metadata_path = None
    if cfg_path is not None:
        metadata_path = _rapid_metadata_path_from_config(str(cfg_path), _path_mtime(cfg_path))
    if metadata_path is None:
        active_meta = ACTIVE_DB_CONFIG.get("original_json")
        if active_meta and Path(active_meta).exists():
            metadata_path = str(active_meta)
    if not metadata_path:
        return {}
    return _rapid_metadata_formula_lookup(str(metadata_path), _path_mtime(metadata_path))


def _rapid_repair_phase_dataframe(df, path: Path):
    if df is None or df.empty or pd is None:
        return df
    candidate_cols = {"phase_ids", "formulas", "formula_keys", "phase_id", "formula", "formula_key", "weights_json"}
    if not candidate_cols.intersection(set(df.columns)):
        return df

    main_phase_label = _rapid_main_phase_label_from_output_path(path)
    needs_main_label = bool(main_phase_label) and any(
        "Main phase" in str(value)
        for col in [c for c in ("weights_json", "formulas", "formula", "formula_keys", "formula_key") if c in df.columns]
        for value in df[col].astype(str).head(100).tolist()
    )
    needs_lookup = False
    for col in [c for c in ("formulas", "formula_keys", "formula", "formula_key", "phase_ids", "phase_id", "weights_json") if c in df.columns]:
        values = df[col].astype(str).head(100).tolist()
        if any(_rapid_needs_formula(token) for value in values for token in _rapid_split_phases(value)):
            needs_lookup = True
            break
    needs_sg_labels = "weights_json" in df.columns and any(c in df.columns for c in ("formulas", "formula_keys", "space_groups"))
    if not needs_lookup and not needs_main_label and not needs_sg_labels:
        return df

    lookup = _rapid_formula_lookup_for_output(path) if needs_lookup else {}
    if needs_lookup and not lookup and not needs_main_label and not needs_sg_labels:
        return df

    repaired = df.copy()

    def repair_pipe(row, value_col: str, *, key_mode: bool = False) -> str:
        pids = _rapid_split_phases(row.get("phase_ids") or row.get("phase_id"))
        vals = _rapid_split_phases(row.get(value_col))
        if not vals and pids:
            vals = pids
        out: list[str] = []
        for idx, val in enumerate(vals):
            pid = pids[idx] if idx < len(pids) else val
            formula = lookup.get(str(pid))
            if formula and _rapid_needs_formula(val, pid):
                out.append(_rapid_formula_key(formula) if key_mode else formula)
            else:
                out.append(str(val))
        return "|".join(out)

    if "formulas" in repaired.columns and "phase_ids" in repaired.columns:
        repaired["formulas"] = repaired.apply(lambda row: repair_pipe(row, "formulas"), axis=1)
    if "formula_keys" in repaired.columns and "phase_ids" in repaired.columns:
        repaired["formula_keys"] = repaired.apply(lambda row: repair_pipe(row, "formula_keys", key_mode=True), axis=1)
    if "formula" in repaired.columns and "phase_id" in repaired.columns:
        repaired["formula"] = repaired.apply(
            lambda row: lookup.get(str(row.get("phase_id")), row.get("formula"))
            if _rapid_needs_formula(row.get("formula"), row.get("phase_id"))
            else row.get("formula"),
            axis=1,
        )
    if "formula_key" in repaired.columns and "phase_id" in repaired.columns:
        repaired["formula_key"] = repaired.apply(
            lambda row: _rapid_formula_key(lookup.get(str(row.get("phase_id")), row.get("formula_key")))
            if _rapid_needs_formula(row.get("formula_key"), row.get("phase_id"))
            else row.get("formula_key"),
            axis=1,
        )
    if "weights_json" in repaired.columns:
        def repair_weights(row) -> object:
            raw = row.get("weights_json") if hasattr(row, "get") else None
            try:
                weights = json.loads(str(raw or ""))
            except Exception:
                return raw
            if not isinstance(weights, dict):
                return raw
            row_map = _rapid_weight_name_map(row)
            remapped = {}
            for key, val in weights.items():
                key_text = str(key)
                if key_text == "Main phase" and main_phase_label:
                    remapped[main_phase_label] = val
                else:
                    remapped[row_map.get(key_text) or lookup.get(key_text, key_text)] = val
            return json.dumps(remapped)
        repaired["weights_json"] = repaired.apply(repair_weights, axis=1)
    return repaired


def _rapid_experiment_root() -> Path | None:
    candidates: list[Path] = []
    env_root = os.environ.get("RADAR_PD_EXPERIMENTS_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    project_root = Path(PROJECT_ROOT)
    candidates.extend(
        [
            project_root / "experiments",
            project_root.parent / "experiments",
            project_root.parent.parent / "experiments",
            Path.home() / "radar-pd-experiments",
            Path("/home/cloud/radar-pd-experiments"),
        ]
    )
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.expanduser().resolve())
        except Exception:
            resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if (candidate / "sparse_mixture_lk99").exists():
            return candidate
    return None


def _rapid_report_root() -> Path | None:
    if st.session_state.get("run_active"):
        active_run = st.session_state.get("run_dir")
        if active_run:
            active_path = Path(active_run)
            active_mode = _run_analysis_mode_from_dir(active_path)
            if active_mode != "Rapid Hypothesis Mode":
                return None
            live_root = active_path / "rapid_results"
            if live_root.exists() or active_mode == "Rapid Hypothesis Mode":
                return live_root
        return None
    if _is_staging_reused_run_inputs():
        return None
    selected = st.session_state.get("selected_run_dir")
    if selected:
        selected_path = Path(selected)
        selected_mode = _run_analysis_mode_from_dir(selected_path)
        if selected_mode == "Full RADAR-PD":
            return None
        live_root = selected_path / "rapid_results"
        if live_root.exists() or selected_mode == "Rapid Hypothesis Mode":
            return live_root
    current_run = st.session_state.get("run_dir")
    if current_run and not selected:
        current_path = Path(current_run)
        current_mode = _run_analysis_mode_from_dir(current_path)
        if current_mode == "Full RADAR-PD":
            return None
    for state_key in ("run_dir", "last_finished_run_dir"):
        run_dir = st.session_state.get(state_key)
        if not run_dir:
            continue
        live_root = Path(run_dir) / "rapid_results"
        is_current_rapid = (
            state_key == "run_dir"
            and (
                (st.session_state.get("run_summary") or {}).get("analysis_mode") == "Rapid Hypothesis Mode"
                or st.session_state.get("analysis_mode") == "Rapid Hypothesis Mode"
            )
        )
        is_last_rapid = state_key == "last_finished_run_dir" and st.session_state.get("last_finished_analysis_mode") == "Rapid Hypothesis Mode"
        if (live_root / "summary.json").exists() or is_current_rapid or is_last_rapid:
            return live_root
    if st.session_state.get("suppress_latest_run_autoload"):
        return None
    context_run = _selected_or_active_run_dir()
    if context_run is not None:
        context_mode = _run_analysis_mode_from_dir(context_run)
        if context_mode == "Full RADAR-PD":
            return None
        if context_mode == "Rapid Hypothesis Mode":
            live_root = context_run / "rapid_results"
            if live_root.exists() or context_mode == "Rapid Hypothesis Mode":
                return live_root
    latest_run = _latest_rapid_run_dir()
    if latest_run is not None:
        live_root = latest_run / "rapid_results"
        if live_root.exists() or _run_analysis_mode_from_dir(latest_run) == "Rapid Hypothesis Mode":
            return live_root
    if not st.session_state.get("rapid_demo_fixture_enabled", False):
        return None
    exp_root = _rapid_experiment_root()
    if exp_root is None:
        return None
    root = exp_root / "sparse_mixture_lk99" / "merged_end_to_end_report_v1"
    return root if root.exists() else None


def _rapid_nudge_root() -> Path | None:
    report_root = _rapid_report_root()
    if report_root is not None:
        live_nudge = report_root / "nudge"
        if live_nudge.exists():
            return live_nudge
    for state_key in ("run_dir", "last_finished_run_dir"):
        run_dir = st.session_state.get(state_key)
        if not run_dir:
            continue
        live_root = Path(run_dir) / "rapid_results" / "nudge"
        if live_root.exists():
            return live_root
    if not st.session_state.get("rapid_demo_fixture_enabled", False):
        return None
    exp_root = _rapid_experiment_root()
    if exp_root is None:
        return None
    root = exp_root / "sparse_mixture_lk99" / "radar_nudge_then_512_v1"
    return root if root.exists() else None


def _latest_rapid_run_dir() -> Path | None:
    runs_root = ACTIVE_RUNS_ROOT
    if not runs_root.exists():
        return None
    latest: tuple[float, Path] | None = None
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        rapid_root = run_dir / "rapid_results"
        if not rapid_root.exists() and _run_analysis_mode_from_dir(run_dir) != "Rapid Hypothesis Mode":
            continue
        probe_paths = [
            rapid_root / "summary.json",
            run_dir / "run_manifest.json",
            run_dir / "pipeline.log",
            run_dir / "pipeline_config.yaml",
        ]
        try:
            mtime = max([p.stat().st_mtime for p in probe_paths if p.exists()] or [run_dir.stat().st_mtime])
        except OSError:
            continue
        if latest is None or mtime > latest[0]:
            latest = (mtime, run_dir)
    return latest[1] if latest else None


def _rapid_active_or_selected_run_dir() -> Path | None:
    if st.session_state.get("run_active"):
        run_dir = st.session_state.get("run_dir")
        if run_dir:
            try:
                path = Path(run_dir)
                if path.exists():
                    return path
            except Exception:
                pass
    for state_key in ("selected_run_dir", "run_dir", "last_finished_run_dir"):
        run_dir = st.session_state.get(state_key)
        if not run_dir:
            continue
        try:
            path = Path(run_dir)
        except Exception:
            continue
        if path.exists():
            return path
    if st.session_state.get("suppress_latest_run_autoload"):
        return None
    latest_run = _latest_rapid_run_dir()
    if latest_run is not None:
        return latest_run
    return None


def _is_rapid_context() -> bool:
    if st.session_state.get("run_active"):
        summary = st.session_state.get("run_summary") or {}
        summary_mode = summary.get("analysis_mode")
        return summary_mode == "Rapid Hypothesis Mode" or st.session_state.get("analysis_mode") == "Rapid Hypothesis Mode"
    selected_run = _selected_or_active_run_dir()
    if selected_run is not None and (st.session_state.get("selected_run_dir") or not st.session_state.get("run_active")):
        mode = _run_analysis_mode_from_dir(selected_run)
        if mode:
            return mode == "Rapid Hypothesis Mode"
    summary = st.session_state.get("run_summary") or {}
    summary_mode = summary.get("analysis_mode")
    return st.session_state.get("analysis_mode") == "Rapid Hypothesis Mode"


def _rapid_local_path(path_value: object, report_root: Path | None) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.exists():
        return path
    if report_root is not None:
        marker = "merged_end_to_end_report_v1/"
        normalized = text.replace("\\", "/")
        if marker in normalized:
            tail = normalized.split(marker, 1)[1]
            candidate = report_root / tail
            if candidate.exists():
                return candidate
    return path


def _rapid_read_csv(path: Path | None):
    if path is None or not path.exists() or pd is None:
        return None
    try:
        return _rapid_repair_phase_dataframe(cached_read_csv_file(str(path), _path_mtime(path)), path)
    except Exception:
        return None


def _rapid_read_512_table(nudge_root: Path | None):
    if nudge_root is None:
        return None
    final_path = nudge_root / "reranked_512_after_radar_nudge.csv"
    partial_path = nudge_root / "reranked_512_after_radar_nudge.partial.csv"
    final_df = _rapid_read_csv(final_path)
    if final_df is not None:
        return final_df
    return _rapid_read_csv(partial_path)


def _rapid_read_512_status(nudge_root: Path | None) -> dict:
    if nudge_root is None:
        return {}
    path = nudge_root / "rerank512_status.json"
    if not path.exists():
        return {}
    try:
        return cached_read_json_file(str(path), _path_mtime(path))
    except Exception:
        return {}


def _rapid_payload_path_from_row(row, report_root: Path | None) -> Path | None:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    path = _rapid_local_path(getter("component_payload"), report_root)
    if path is not None and path.exists():
        return path
    return None


def _rapid_hypothesis_label_from_row(row) -> str:
    phases = _rapid_phase_labels_from_row(row)
    if not phases:
        return "-"
    return " + ".join(phases)


def _rapid_canonical_phase_label(token: object) -> str:
    return _rapid_display_formula(_rapid_choice_formula(token), _rapid_choice_sg(token))


def _rapid_phase_labels_same_family(label_a: str, label_b: str) -> bool:
    return _rapid_same_stoichiometric_family(
        _rapid_choice_formula(label_a),
        _rapid_choice_sg(label_a),
        _rapid_choice_formula(label_b),
        _rapid_choice_sg(label_b),
    )


def _rapid_family_representative_label(token: object, representative_map: dict[str, str] | None = None) -> str:
    label = _rapid_canonical_phase_label(token)
    if representative_map and label in representative_map:
        return representative_map[label]
    return label


def _rapid_build_representative_map(*frames) -> dict[str, str]:
    labels: list[str] = []
    seen: set[str] = set()
    for df in frames:
        if df is None or getattr(df, "empty", True):
            continue
        for _, row in df.iterrows():
            for label in _rapid_phase_labels_from_row(row):
                canonical = _rapid_canonical_phase_label(label)
                if canonical and canonical not in seen:
                    labels.append(canonical)
                    seen.add(canonical)

    families: list[list[str]] = []
    for label in labels:
        matched_family = None
        for family in families:
            if any(_rapid_phase_labels_same_family(label, member) for member in family):
                matched_family = family
                break
        if matched_family is None:
            families.append([label])
        else:
            matched_family.append(label)

    representative_map: dict[str, str] = {}
    for family in families:
        representative = min(
            family,
            key=lambda item: (
                _rapid_formula_niceness_score(_rapid_choice_formula(item)),
                _rapid_choice_formula(item),
                _rapid_choice_sg(item),
            ),
        )
        for label in family:
            representative_map[label] = representative
    return representative_map


def _rapid_hypothesis_family_labels(row, representative_map: dict[str, str] | None = None) -> list[str]:
    labels = _rapid_phase_labels_from_row(row)
    family_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        representative = _rapid_family_representative_label(label, representative_map)
        if representative in seen:
            continue
        family_labels.append(representative)
        seen.add(representative)
    return family_labels


def _rapid_hypothesis_family_key(row, representative_map: dict[str, str] | None = None) -> tuple[str, ...]:
    return tuple(sorted(_rapid_hypothesis_family_labels(row, representative_map)))


def _rapid_hypothesis_family_label_from_row(row, representative_map: dict[str, str] | None = None) -> str:
    labels = _rapid_hypothesis_family_labels(row, representative_map)
    return " + ".join(labels) if labels else _rapid_hypothesis_label_from_row(row)


def _rapid_dedupe_hypotheses(df, *, sort_col: str | None = None, ascending: bool = True, representative_map: dict[str, str] | None = None):
    if df is None or getattr(df, "empty", True) or pd is None:
        return df
    view = df.copy()
    if sort_col and sort_col in view.columns:
        view[sort_col] = pd.to_numeric(view[sort_col], errors="coerce")
        view = view.sort_values(sort_col, ascending=ascending)
    kept_rows = []
    seen: set[tuple[str, ...]] = set()
    for _, row in view.iterrows():
        key = _rapid_hypothesis_family_key(row, representative_map)
        if key in seen:
            continue
        seen.add(key)
        kept_rows.append(row)
    if not kept_rows:
        return view.head(0)
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _rapid_weight_name_map(row) -> dict[str, str]:
    labels = _rapid_phase_labels_from_row(row)
    formulas = _rapid_split_phases((row.get("formulas") if hasattr(row, "get") else "") or (row.get("formula_keys") if hasattr(row, "get") else ""))
    mapping: dict[str, str] = {}
    for idx, formula in enumerate(formulas):
        label = labels[idx] if idx < len(labels) else formula
        mapping[str(formula)] = label
        mapping[_rapid_formula_key(formula)] = label
        mapping[_rapid_choice_formula(label)] = label
    return mapping


def _rapid_weights_label(value: object, row=None, *, max_items: int = 4) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "-"
    try:
        weights = json.loads(text)
    except Exception:
        return "-"
    if not isinstance(weights, dict) or not weights:
        return "-"
    name_map = _rapid_weight_name_map(row) if row is not None else {}
    rows: list[tuple[str, float]] = []
    for key, val in weights.items():
        try:
            label = name_map.get(str(key), str(key))
            rows.append((label, float(val)))
        except Exception:
            continue
    if not rows:
        return "-"
    rows.sort(key=lambda item: abs(item[1]), reverse=True)
    rendered = [f"{name}: {value:.1f}%" for name, value in rows[:max_items]]
    if len(rows) > max_items:
        rendered.append(f"+{len(rows) - max_items} more")
    return "; ".join(rendered)


def _rapid_status_label(value: object) -> str:
    status = str(value or "").strip().lower()
    return {
        "ok": "Converged",
        "refine_warning": "Needs review",
        "error": "Failed",
    }.get(status, status.title() if status else "-")


def _rapid_peak_support_label(value: object) -> str:
    """Expand compact support strings such as `2+0/5` for UI tables."""
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return "-"

    def _expand(match: re.Match[str]) -> str:
        label = (match.group("label") or "").strip()
        supported = int(match.group("supported"))
        weak = int(match.group("weak"))
        total = int(match.group("total"))
        missing = max(0, total - supported - weak)
        summary = f"{supported} supported, {weak} weak, {missing} missing/review"
        return f"{label}: {summary}" if label else summary

    pattern = re.compile(
        r"(?:(?P<label>[^:;]+):\s*)?(?P<supported>\d+)\+(?P<weak>\d+)/(?P<total>\d+)"
    )
    return pattern.sub(_expand, text)


def _rapid_cell_summary(row) -> str:
    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    abc: list[str] = []
    angles: list[str] = []
    for key in ["a", "b", "c"]:
        val = _rapid_to_float(getter(key))
        abc.append("-" if val is None else f"{val:.3f}")
    for key in ["alpha", "beta", "gamma"]:
        val = _rapid_to_float(getter(key))
        angles.append("-" if val is None else f"{val:.2f}")
    return f"a,b,c {', '.join(abc)}; angles {', '.join(angles)}"


def _rapid_render_refinement_summary_cards(display) -> None:
    if display is None or getattr(display, "empty", True):
        return

    cards: list[str] = []
    for _, row in display.head(5).iterrows():
        rank = html.escape(str(row.get("Rank") or "-"))
        hypothesis = html.escape(str(row.get("Hypothesis") or "-"))
        fractions = html.escape(str(row.get("Phase fractions") or "-"))
        status = html.escape(str(row.get("Status") or "-"))
        pattern_rank = html.escape(str(row.get("Pattern rank") or "-"))
        quality = _rapid_fmt_number(row.get("Refinement quality"), 3)
        seconds = _rapid_fmt_number(row.get("Time (s)"), 1)
        cards.append(
            "<div class=\"rapid-refinement-card\">"
            "<div class=\"rapid-refinement-card-topline\">"
            f"<span>Final rank {rank}</span>"
            f"<span>Quality {html.escape(quality)}</span>"
            "</div>"
            f"<div class=\"rapid-refinement-card-hypothesis\">{hypothesis}</div>"
            "<div class=\"rapid-refinement-card-fractions\">"
            "<span>Phase fractions</span>"
            f"<strong>{fractions}</strong>"
            "</div>"
            "<div class=\"rapid-refinement-card-meta\">"
            f"<span>Pattern rank {pattern_rank}</span>"
            f"<span>{status}</span>"
            f"<span>{html.escape(seconds)} s</span>"
            "</div>"
            "</div>"
        )

    if not cards:
        return
    st.markdown("**Top final refinements**")
    st.markdown(
        "<div class=\"rapid-refinement-card-grid\">" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _render_responsive_result_table(display) -> None:
    if display is None or getattr(display, "empty", True):
        return

    def _cell_text(value, col: str) -> str:
        try:
            if pd is not None and pd.isna(value):
                return "-"
        except Exception:
            pass
        if value is None:
            return "-"
        if isinstance(value, (float, np.floating)):
            if col == "Time (s)":
                return f"{float(value):.2f}"
            if col == "Cell adjustment":
                return f"{float(value):.4f}"
            return f"{float(value):.3f}"
        return str(value)

    def _cell_class(col: str, value) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", col.lower()).strip("-")
        classes = [f"radar-cell-{slug}"]
        if isinstance(value, (int, float, np.integer, np.floating)) and col not in {
            "Rank",
            "Coarse rank",
            "Pattern rank",
        }:
            classes.append("radar-cell-num")
        return " ".join(classes)

    header = "".join(
        f'<th class="{html.escape(_cell_class(str(col), None))}">{html.escape(str(col))}</th>'
        for col in display.columns
    )
    body_rows: list[str] = []
    for _, row in display.iterrows():
        cells: list[str] = []
        for col in display.columns:
            raw = row.get(col)
            text = _cell_text(raw, str(col))
            css_class = _cell_class(str(col), raw)
            cells.append(
                f'<td class="{html.escape(css_class)}" title="{html.escape(text)}">'
                f"{html.escape(text)}"
                "</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    table_classes = ["radar-table"]
    if "Key peak support" in display.columns:
        table_classes.append("radar-table-wide-support")
    if "Phase fractions" in display.columns:
        table_classes.append("radar-table-wide-fractions")

    st.markdown(
        '<div class="radar-table-wrap">'
        f'<table class="{html.escape(" ".join(table_classes))}">'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>",
        unsafe_allow_html=True,
    )


def _safe_dataframe_for_streamlit(df):
    if df is None or pd is None:
        return df
    safe = df.copy()

    def _safe_object_value(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, (dict, list, tuple, set)):
            try:
                return json.dumps(value, default=str)
            except Exception:
                return str(value)
        return str(value)

    for col in safe.columns:
        try:
            if safe[col].dtype == object:
                safe[col] = safe[col].map(_safe_object_value)
        except Exception:
            safe[col] = safe[col].astype(str)
    return safe


def _rapid_stage_table(
    df,
    stage: str,
    sort_col: str | None = None,
    ascending: bool = True,
    limit: int = 10,
    show_details: bool = False,
    representative_map: dict[str, str] | None = None,
):
    if df is None or df.empty or pd is None:
        st.caption("No rows available for this stage.")
        return
    view = df.copy()
    for col in view.columns:
        if col.lower().startswith(("rank", "rwp", "score", "gain", "sse", "r2", "seconds", "distance")):
            view[col] = pd.to_numeric(view[col], errors="coerce")
    if sort_col and sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending)

    rows: list[dict[str, object]] = []
    for _, row in view.head(limit).iterrows():
        if stage == "64":
            rank = row.get("rank64", row.get("rank", None))
            rows.append(
                {
                    "Rank": _rapid_display_rank(rank),
                    "Hypothesis": _rapid_hypothesis_family_label_from_row(row, representative_map),
                    "Coarse match": _rapid_to_float(row.get("gain64")),
                    "Unexplained signal": _rapid_to_float(row.get("sse64")),
                }
            )
        elif stage == "nudge":
            rows.append(
                {
                    "Phase": _rapid_display_formula(
                        row.get("formula") or row.get("formula_key") or row.get("phase_id") or "-",
                        row.get("space_group"),
                    ),
                    "Space group": row.get("space_group") or "-",
                    "Nudge match": _rapid_to_float(row.get("best_score")),
                    "Cell adjustment": _rapid_to_float(row.get("distance_from_start")),
                    "Best cell": _rapid_cell_summary(row),
                    "Time (s)": _rapid_to_float(row.get("seconds")),
                    "Status": "Needs review" if str(row.get("error") or "").strip() else "Ready",
                }
            )
        elif stage == "512":
            rows.append(
                {
                    "Rank": _rapid_display_rank(row.get("rank512")),
                    "Hypothesis": _rapid_hypothesis_family_label_from_row(row, representative_map),
                    "Key peak support": _rapid_peak_support_label(row.get("peak_support_summary")),
                    "Coarse rank": _rapid_display_rank(row.get("rank64")),
                    "Pattern match": _rapid_to_float(row.get("score512")),
                    "Explained signal": _rapid_to_float(row.get("r2_512")),
                    "Unexplained signal": _rapid_to_float(row.get("sse512")),
                }
            )
        elif stage == "gsas":
            rows.append(
                {
                    "Rank": _rapid_display_rank(row.get("gsas_rwp_rank")),
                    "Hypothesis": _rapid_hypothesis_family_label_from_row(row, representative_map),
                    "Refinement quality": _rapid_to_float(row.get("rwp")),
                    "Phase fractions": _rapid_weights_label(row.get("weights_json"), row),
                    "Pattern rank": _rapid_display_rank(row.get("rank512")),
                    "Status": _rapid_status_label(row.get("status")),
                    "Time (s)": _rapid_to_float(row.get("seconds")),
                }
            )
    if not rows:
        st.caption("No compatible columns found for this stage.")
        return
    display = pd.DataFrame(rows)
    column_config = {
        "Hypothesis": st.column_config.TextColumn("Hypothesis", width="large"),
        "Phase": st.column_config.TextColumn("Phase", width="medium"),
        "Phase fractions": st.column_config.TextColumn("Phase fractions", width="large"),
        "Best cell": st.column_config.TextColumn("Best cell", width="large"),
        "Key peak support": st.column_config.TextColumn("Key peak support", width="medium"),
        "Status": st.column_config.TextColumn("Status", width="medium"),
        "Coarse match": st.column_config.NumberColumn("Coarse match", format="%.3f"),
        "Pattern match": st.column_config.NumberColumn("Pattern match", format="%.3f"),
        "Explained signal": st.column_config.NumberColumn("Explained signal", format="%.3f"),
        "Unexplained signal": st.column_config.NumberColumn("Unexplained signal", format="%.3f"),
        "Nudge match": st.column_config.NumberColumn("Nudge match", format="%.3f"),
        "Cell adjustment": st.column_config.NumberColumn("Cell adjustment", format="%.4f"),
        "Refinement quality": st.column_config.NumberColumn("Refinement quality", format="%.3f"),
        "Time (s)": st.column_config.NumberColumn("Time (s)", format="%.2f"),
    }
    if stage == "gsas":
        _rapid_render_refinement_summary_cards(display)
        st.caption("Sortable refinement table")
    _render_responsive_result_table(display)

    if show_details:
        with st.expander("Technical fields", expanded=False):
            detail = view.head(limit).copy()
            st.dataframe(detail, hide_index=True, width="stretch")


def _rapid_live_report_root() -> Path | None:
    run_dir = _rapid_active_or_selected_run_dir()
    if run_dir is None:
        return None
    return run_dir / "rapid_results"


def _render_rapid_stage_overview() -> None:
    st.markdown("**Rapid mode stages**")
    st.caption(
        "Coarse search -> Lattice nudge -> Pattern scoring -> Final refinement ranking -> Solution inspector"
    )


def _rapid_row_count(path: Path | None) -> int | None:
    df = _rapid_read_csv(path)
    if df is None:
        return None
    return int(len(df))


def _rapid_best_rwp(path: Path | None) -> str:
    df = _rapid_read_csv(path)
    if df is None or df.empty or "rwp" not in df.columns:
        return "-"
    values = pd.to_numeric(df["rwp"], errors="coerce").dropna()
    if values.empty:
        return "-"
    return _rapid_fmt_number(values.min(), 3)


def render_rapid_run_snapshot() -> None:
    report_root = _rapid_live_report_root()
    if report_root is None:
        st.info("Rapid outputs will appear after you start a rapid run.")
        _render_rapid_stage_overview()
        return
    nudge_root = report_root / "nudge" / "live_run"
    gsas_path = report_root / "all_gsas_validation_summary.csv"
    rerank512_df = _rapid_read_512_table(nudge_root)
    cards = [
        ("Coarse hypotheses", _rapid_row_count(nudge_root / "beam64_input_top.csv") or "-"),
        ("Nudged phases", _rapid_row_count(nudge_root / "nudge_results.csv") or "-"),
        ("Scored hypotheses", "-" if rerank512_df is None else int(len(rerank512_df))),
        ("Refinement checks", _rapid_row_count(gsas_path) or "-"),
        ("Best Rwp", _rapid_best_rwp(gsas_path)),
    ]
    _render_plot_meta_cards(cards, numeric_labels={"Coarse hypotheses", "Nudged phases", "Scored hypotheses", "Refinement checks", "Best Rwp"})
    _render_rapid_stage_overview()
    if not report_root.exists():
        st.caption("Rapid result folder has not been created yet.")
    elif st.session_state.get("run_active"):
        st.caption("This snapshot updates as the rapid search writes each stage artifact.")
        status = _rapid_read_512_status(nudge_root)
        if status:
            processed = int(status.get("processed") or 0)
            total = int(status.get("total") or 0)
            if total > 0 and processed < total:
                st.progress(min(1.0, processed / max(total, 1)), text=f"High-resolution pattern scoring: {processed} / {total} hypotheses")


def render_magnetic_precheck_panel(run_dir: Path | None, *, compact: bool = False) -> None:
    if run_dir is None:
        return
    mag_dir = Path(run_dir) / "Magnetic_Precheck"
    summary_path = mag_dir / "magnetic_precheck_summary.json"
    if not summary_path.exists():
        cfg_path = Path(run_dir) / "pipeline_config.yaml"
        enabled = False
        if cfg_path.exists():
            try:
                cfg = cached_read_yaml_file(str(cfg_path), _path_mtime(cfg_path))
                enabled = bool((cfg.get("magnetic_precheck") or {}).get("enabled"))
            except Exception:
                enabled = False
        if enabled:
            st.info("Magnetic precheck is enabled. Results will appear after the main-phase residual is prepared.")
        return
    try:
        summary = cached_read_json_file(str(summary_path), _path_mtime(summary_path))
    except Exception as exc:
        st.warning(f"Could not read magnetic precheck summary: {exc}")
        return
    if not summary.get("enabled", False):
        return

    evidence = str(summary.get("evidence") or summary.get("status") or "unknown")
    title = "Magnetic Ordering Precheck"
    if compact:
        st.markdown(f"**{title}**")
    else:
        st.subheader(title)
    if evidence == "strong":
        st.success(summary.get("reason") or "Residual peaks show strong consistency with one commensurate k-vector.")
    elif evidence == "moderate":
        st.warning(summary.get("reason") or "Residual peaks show moderate consistency with one commensurate k-vector.")
    elif summary.get("status") == "failed":
        st.error(summary.get("reason") or "Magnetic precheck failed.")
    else:
        st.info(summary.get("reason") or "No strong magnetic k-vector indexing evidence was found.")

    cards = [
        ("Evidence", evidence.title()),
        ("Best k", summary.get("best_k") or "-"),
        ("Null percentile", f"{100.0 * float(summary.get('null_percentile') or 0.0):.1f}%"),
        ("Explained residual", f"{100.0 * float(summary.get('explained_fraction') or 0.0):.1f}%"),
        ("Supported peaks", summary.get("supported_top_peaks", "-")),
        ("Seconds", _format_duration(summary.get("seconds"))),
    ]
    _render_plot_meta_cards(cards, numeric_labels={"Supported peaks"})

    plot_path = mag_dir / "magnetic_residual_fit.png"
    if plot_path.exists() and not compact:
        st.image(str(plot_path), caption="Positive residual and best fixed-position magnetic peak-comb fit.", use_container_width=True)

    rankings_path = mag_dir / "magnetic_k_vector_rankings.csv"
    peaks_path = mag_dir / "magnetic_peak_support.csv"
    with st.expander("Magnetic precheck details", expanded=False):
        if rankings_path.exists() and pd is not None:
            try:
                rankings = cached_read_csv_file(str(rankings_path), _path_mtime(rankings_path)).head(12)
                st.markdown("**Top k-vector candidates**")
                st.dataframe(rankings, hide_index=True, width="stretch")
            except Exception as exc:
                st.caption(f"Could not read k-vector rankings: {exc}")
        if peaks_path.exists() and pd is not None:
            try:
                peaks = cached_read_csv_file(str(peaks_path), _path_mtime(peaks_path)).head(12)
                st.markdown("**Residual peak support**")
                st.dataframe(peaks, hide_index=True, width="stretch")
            except Exception as exc:
                st.caption(f"Could not read residual peak support: {exc}")


def render_rapid_artifacts(root: Path) -> None:
    shown = False
    key_files = [
        root / "summary.json",
        root / "all_gsas_validation_summary.csv",
        root / "nudge" / "live_run" / "beam64_input_top.csv",
        root / "nudge" / "live_run" / "nudge_results.csv",
        root / "nudge" / "live_run" / "reranked_512_after_radar_nudge.csv",
        root / "nudge" / "live_run" / "reranked_512_after_radar_nudge.partial.csv",
        root / "nudge" / "live_run" / "rerank512_status.json",
    ]
    st.markdown("**Rapid outputs**")
    for idx, path in enumerate(key_files):
        shown = render_download_file(path, f"rapid_key_{idx}") or shown
    curve_dir = root / "live_run" / "gsas"
    if curve_dir.exists():
        st.markdown("**Refinement curves and projects**")
        shown = render_file_explorer(
            curve_dir,
            "rapid_gsas",
            [".png", ".csv", ".gpx", ".lst"],
            hide_predicate=_hide_curated_artifact,
            show_downloads=False,
        ) or shown
    if not shown:
        st.caption("Rapid artifacts are not available yet.")


def render_rapid_partial_outputs(report_root: Path) -> None:
    nudge_root = report_root / "nudge" / "live_run"
    beam64_df = _rapid_read_csv(nudge_root / "beam64_input_top.csv")
    nudge_df = _rapid_read_csv(nudge_root / "nudge_results.csv")
    shadow_filter_df = _rapid_read_csv(nudge_root / "main_shadow_nudge_filter.csv")
    rerank512_df = _rapid_read_512_table(nudge_root)
    gsas_partial_df = _rapid_read_csv(report_root / "all_gsas_validation_summary.csv")
    if gsas_partial_df is None or gsas_partial_df.empty:
        gsas_partial_df = _rapid_read_csv(report_root / "all_gsas_validation_summary.partial.csv")
    if all(df is None or df.empty for df in (beam64_df, nudge_df, rerank512_df, gsas_partial_df)):
        st.info("Rapid stage outputs are not available yet. Use Run Monitor while the first stage is running.")
        return
    representative_map = _rapid_build_representative_map(beam64_df, nudge_df, shadow_filter_df, rerank512_df, gsas_partial_df)
    beam64_display_df = _rapid_dedupe_hypotheses(
        beam64_df,
        sort_col="rank64",
        ascending=True,
        representative_map=representative_map,
    )
    rerank512_display_df = _rapid_dedupe_hypotheses(
        rerank512_df,
        sort_col="rank512",
        ascending=True,
        representative_map=representative_map,
    )
    gsas_display_df = _rapid_dedupe_hypotheses(
        gsas_partial_df,
        sort_col="gsas_rwp_rank",
        ascending=True,
        representative_map=representative_map,
    )
    _render_plot_meta_cards(
        [
            ("Coarse hypotheses", 0 if beam64_display_df is None else len(beam64_display_df)),
            ("Nudged phases", 0 if nudge_df is None else len(nudge_df)),
            ("Scored hypotheses", 0 if rerank512_display_df is None else len(rerank512_display_df)),
            ("Refinement checks", 0 if gsas_display_df is None else len(gsas_display_df)),
        ],
        numeric_labels={"Coarse hypotheses", "Nudged phases", "Scored hypotheses", "Refinement checks"},
    )
    status = _rapid_read_512_status(nudge_root)
    if status:
        processed = int(status.get("processed") or 0)
        total = int(status.get("total") or 0)
        done = bool(status.get("done"))
        if total > 0:
            st.progress(
                min(1.0, processed / max(total, 1)),
                text=(
                    "High-resolution pattern scoring complete"
                    if done
                    else f"High-resolution pattern scoring: {processed} / {total} hypotheses processed"
                ),
            )
        best = status.get("best_hypothesis") or {}
        if best:
            st.caption(
                "Current best refined-pattern hypothesis: "
                f"{_rapid_hypothesis_family_label_from_row(best, representative_map)}"
            )
    tab_names = ["Coarse search", "Lattice nudge", "Pattern scoring"]
    if gsas_display_df is not None and not gsas_display_df.empty:
        tab_names.append("Final refinement ranking")
    partial_tabs = st.tabs(tab_names)
    with partial_tabs[0]:
        _rapid_stage_table(
            beam64_display_df,
            "64",
            sort_col="rank64",
            ascending=True,
            limit=int(st.session_state.get("rapid_stage_output_limit", 10)),
            show_details=bool(st.session_state.get("show_expert_params", False)),
            representative_map=representative_map,
        )
    with partial_tabs[1]:
        _rapid_stage_table(
            nudge_df,
            "nudge",
            sort_col="best_score",
            ascending=False,
            limit=max(10, int(st.session_state.get("rapid_stage_output_limit", 10))),
            show_details=bool(st.session_state.get("show_expert_params", False)),
        )
    with partial_tabs[2]:
        _rapid_stage_table(
            rerank512_display_df,
            "512",
            sort_col="rank512",
            ascending=True,
            limit=int(st.session_state.get("rapid_stage_output_limit", 10)),
            show_details=bool(st.session_state.get("show_expert_params", False)),
            representative_map=representative_map,
        )
        _rapid_render_512_fit_selector(
            rerank512_display_df,
            nudge_root,
            report_root,
            key="partial_shortlist",
            limit=int(st.session_state.get("rapid_stage_output_limit", 10)),
            representative_map=representative_map,
        )
    if len(partial_tabs) > 3:
        with partial_tabs[3]:
            _rapid_stage_table(
                gsas_display_df,
                "gsas",
                sort_col="gsas_rwp_rank",
                ascending=True,
                limit=int(st.session_state.get("rapid_gsas_validation_limit", 10)),
                show_details=bool(st.session_state.get("show_expert_params", False)),
                representative_map=representative_map,
            )
            _rapid_render_gsas_fit_selector(
                gsas_display_df,
                report_root,
                key="partial_gsas",
                limit=int(st.session_state.get("rapid_gsas_validation_limit", 10)),
                representative_map=representative_map,
            )


RAPID_REMOVE_PHASE_OPTION = "Remove this phase"


def _rapid_frame_iter(rows):
    if rows is None:
        return
    if isinstance(rows, (list, tuple)):
        for item in rows:
            yield from _rapid_frame_iter(item)
        return
    if getattr(rows, "empty", True):
        return
    yield rows


def _rapid_variant_options(rows, token: str) -> list[str]:
    token_formula = _rapid_choice_formula(token)
    token_sg = _rapid_choice_sg(token)
    variants: set[str] = set()
    for frame in _rapid_frame_iter(rows):
        rows = frame
        for _, row in rows.iterrows():
            phases = _rapid_split_phases(
                row.get("formulas")
                or row.get("formula_keys")
                or row.get("formula")
                or row.get("formula_key")
                or row.get("phase_id")
            )
            sgs = _rapid_space_groups_from_row(row)
            if not sgs and str(row.get("space_group", "")).strip():
                sgs = [str(row.get("space_group", "")).strip()]
            for idx, phase in enumerate(phases):
                sg = sgs[idx] if idx < len(sgs) else ""
                if _rapid_same_stoichiometric_family(token_formula, token_sg, phase, sg):
                    variants.add(_rapid_display_formula(phase, sg))
    if token:
        variants.add(token)
    ordered = sorted(
        variants,
        key=lambda item: (
            _rapid_formula_niceness_score(_rapid_choice_formula(item)),
            _rapid_choice_formula(item),
            _rapid_choice_sg(item),
        ),
    )
    return [*ordered, RAPID_REMOVE_PHASE_OPTION]


def _rapid_match_hypothesis(rows, selected_phases: list[str]):
    if rows is None or rows.empty:
        return None
    wanted = sorted(
        (_rapid_choice_formula(phase), _rapid_choice_sg(phase))
        for phase in selected_phases
        if phase != RAPID_REMOVE_PHASE_OPTION
    )
    for _, row in rows.iterrows():
        formulas = _rapid_split_phases(row.get("formulas") or row.get("formula_keys"))
        sgs = _rapid_space_groups_from_row(row)
        phases = [
            (_rapid_choice_formula(phase), sgs[idx] if idx < len(sgs) else "")
            for idx, phase in enumerate(formulas)
        ]
        if sorted(phases) == wanted:
            return row
    return None


def _rapid_phase_records_from_rows(*frames) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for rows in _rapid_frame_iter(list(frames)):
        for _, row in rows.iterrows():
            pids = _rapid_split_phases(row.get("phase_ids") or row.get("phase_id"))
            formulas = _rapid_split_phases(row.get("formulas") or row.get("formula_keys") or row.get("formula"))
            formula_keys = _rapid_split_phases(row.get("formula_keys") or row.get("formulas") or row.get("formula_key"))
            cif_paths = _rapid_split_phases(row.get("cif_paths") or row.get("nudged_cif") or row.get("cif_path"))
            sgs = _rapid_space_groups_from_row(row)
            if not sgs and str(row.get("space_group", "")).strip():
                sgs = [str(row.get("space_group", "")).strip()]
            for idx, pid in enumerate(pids):
                formula = formulas[idx] if idx < len(formulas) else pid
                formula_key = formula_keys[idx] if idx < len(formula_keys) else _rapid_formula_key(formula)
                sg = sgs[idx] if idx < len(sgs) else ""
                display = _rapid_display_formula(formula, sg)
                record = {
                    "phase_id": str(pid),
                    "formula": str(formula),
                    "formula_key": str(formula_key),
                    "space_group": str(sg),
                    "display": display,
                    "cif_path": str(cif_paths[idx]) if idx < len(cif_paths) else "",
                }
                for token in {str(pid), str(formula), str(formula_key), _rapid_formula_key(formula), display}:
                    token = str(token or "").strip()
                    if token and token not in records:
                        records[token] = record
    return records


def _rapid_manual_validation_row(
    selected_phases: list[str],
    *,
    candidate_rows,
    fallback_row,
) -> tuple[object | None, str | None]:
    if not selected_phases:
        return None, "No phases are selected."
    records = _rapid_phase_records_from_rows(candidate_rows)
    chosen: list[dict[str, str]] = []
    missing: list[str] = []
    for phase in selected_phases:
        clean_phase = _rapid_choice_formula(phase)
        record = (
            records.get(str(phase))
            or records.get(clean_phase)
            or records.get(_rapid_formula_key(clean_phase))
        )
        if record is None:
            missing.append(str(phase))
        else:
            chosen.append(record)
    if missing:
        return None, "Could not map selected phase(s) to catalog IDs: " + ", ".join(missing)

    phase_id_key = "|".join(item["phase_id"] for item in chosen)
    manual_rank512 = 9000 + (int(hashlib.md5(phase_id_key.encode("utf-8")).hexdigest()[:6], 16) % 1000)
    base = dict(fallback_row.to_dict() if hasattr(fallback_row, "to_dict") else dict(fallback_row))
    base.update(
        {
            "rank512": manual_rank512,
            "rank64": _rapid_to_int(base.get("rank64")) or 9999,
            "gsas_rwp_rank": None,
            "phase_ids": phase_id_key,
            "formulas": "|".join(item["formula"] for item in chosen),
            "formula_keys": "|".join(item["formula_key"] for item in chosen),
            "space_groups": "|".join(item.get("space_group", "") for item in chosen),
            "cif_paths": "|".join(item.get("cif_path", "") for item in chosen),
            "score512": 0.0,
            "r2_512": 0.0,
            "sse512": 0.0,
            "phase_coefs512": "",
            "status": "not run",
            "rwp": None,
            "weights_json": "",
            "gpx": "",
            "curve_png": "",
            "curve_csv": "",
            "errors": "",
            "seconds": 0.0,
        }
    )
    return pd.Series(base) if pd is not None else base, None


def _rapid_enrich_row_with_nudged_cifs(row_payload: dict[str, object], report_root: Path) -> dict[str, object]:
    """Prefer the lattice-nudged CIF for every targeted-refinement phase."""
    pids = _rapid_split_phases(row_payload.get("phase_ids"))
    if not pids:
        return row_payload
    supplied = _rapid_split_phases(row_payload.get("cif_paths"))
    nudge_df = _rapid_read_csv(report_root / "nudge" / "live_run" / "nudge_results.csv")
    nudged_by_pid: dict[str, str] = {}
    if nudge_df is not None and not getattr(nudge_df, "empty", True):
        for _, nrow in nudge_df.iterrows():
            pid = str(nrow.get("phase_id", "")).strip()
            path = str(nrow.get("nudged_cif", "")).strip()
            if pid and path and Path(path).exists():
                nudged_by_pid[pid] = str(Path(path).resolve())
    resolved: list[str] = []
    replacements = 0
    for idx, pid in enumerate(pids):
        current = supplied[idx] if idx < len(supplied) else ""
        if current and Path(current).exists():
            resolved.append(str(Path(current).resolve()))
            continue
        nudged = nudged_by_pid.get(str(pid))
        if nudged:
            resolved.append(nudged)
            replacements += 1
        else:
            resolved.append(current)
    row_payload["cif_paths"] = "|".join(resolved)
    row_payload["cif_path_policy"] = "prefer_lattice_nudged_cif"
    row_payload["cif_path_replacements"] = replacements
    return row_payload


def _rapid_targeted_base_gpx(report_root: Path, scenario: str) -> Path | None:
    """Return the same optimized base GPX used by rapid GSAS ranking."""
    candidates: list[Path] = []
    summary_path = report_root / "summary.json"
    try:
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for key in (scenario, "live_run"):
                entry = summary.get(key) if isinstance(summary, dict) else None
                signal_gpx = str((entry or {}).get("signal_gpx") or "").strip()
                if signal_gpx:
                    candidates.append(Path(signal_gpx))
    except Exception:
        pass
    for key in (scenario, "live_run"):
        candidates.append(report_root / "nudge" / key / "rapid_base" / "rapid_base.gpx")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _rapid_render_rank_plot(
    df,
    *,
    value_col: str,
    title: str,
    value_label: str,
    limit: int = 12,
    lower_is_better: bool = False,
    representative_map: dict[str, str] | None = None,
) -> None:
    if df is None or df.empty or pd is None:
        st.caption("No plot data available.")
        return
    if value_col not in df.columns:
        st.caption("This plot is not available for the current artifact.")
        return
    plot_df = df.copy()
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[value_col])
    if plot_df.empty:
        st.caption("No numeric values available for this plot.")
        return
    sort_cols = []
    for rank_col in ["gsas_rwp_rank", "rank512", "rank64", "rank"]:
        if rank_col in plot_df.columns:
            plot_df[rank_col] = pd.to_numeric(plot_df[rank_col], errors="coerce")
            sort_cols.append(rank_col)
            break
    if sort_cols:
        plot_df = plot_df.sort_values(sort_cols[0], ascending=True)
    else:
        plot_df = plot_df.sort_values(value_col, ascending=lower_is_better)
    plot_df = plot_df.head(int(limit))
    labels = [
        _rapid_hypothesis_family_label_from_row(row, representative_map)
        for _, row in plot_df.iterrows()
    ]
    values = plot_df[value_col].astype(float).to_list()
    try:
        import plotly.graph_objects as go

        colors = ["#145f46" if not lower_is_better else "#1f6f8b"] * len(values)
        fig = go.Figure(
            data=[
                go.Bar(
                    x=values,
                    y=labels,
                    orientation="h",
                    marker_color=colors,
                    hovertemplate="<b>%{y}</b><br>" + value_label + ": %{x:.4g}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=title,
            xaxis_title=value_label,
            yaxis_title=None,
            yaxis_autorange="reversed",
            height=max(360, min(620, 42 * len(values) + 120)),
            margin=dict(l=10, r=20, t=56, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e4ebe7")
        fig.update_yaxes(tickfont=dict(size=11))
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            key=f"rapid_rank_plot_{value_col}_{title}",
        )
    except Exception:
        st.bar_chart(pd.DataFrame({value_label: values}, index=labels))


@st.cache_data(show_spinner=False, ttl=300)
def _rapid_compute_512_components(
    target_npz: str,
    target_mtime: float | None,
    config_path: str,
    config_mtime: float | None,
    cif_paths: tuple[str, ...],
    labels: tuple[str, ...],
) -> dict[str, object]:
    data = np.load(target_npz)
    q_grid = np.asarray(data["q_grid"], dtype=float)
    y512 = np.asarray(data["y512"], dtype=float)
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) if config_path and Path(config_path).exists() else {}
    from scripts.rapid_hypothesis_pipeline import _background_rows, _fit_rows, _render_cif_profile512

    profiles: list[np.ndarray] = []
    used_labels: list[str] = []
    for idx, cif_path in enumerate(cif_paths):
        path = Path(str(cif_path))
        if not path.exists():
            continue
        profile = _render_cif_profile512(str(path), q_grid, cfg or {})
        if profile is None:
            continue
        profiles.append(np.asarray(profile, dtype=float))
        used_labels.append(labels[idx] if idx < len(labels) else path.stem)
    if not profiles:
        return {"error": "No renderable CIF profiles were available for this hypothesis."}
    background_rows = _background_rows(int(q_grid.size))
    fit_rows = [*background_rows, *profiles]
    coefs, total_fit, sse = _fit_rows(fit_rows, y512)
    row_norms = np.maximum(
        np.linalg.norm(np.vstack(fit_rows).astype(np.float32), axis=1),
        1e-8,
    )
    phase_coefs = np.asarray(coefs[-len(profiles):], dtype=float)
    phase_norms = row_norms[-len(profiles):]
    components = [
        (np.asarray(profile, dtype=float) / float(norm)) * float(coef)
        for profile, norm, coef in zip(profiles, phase_norms, phase_coefs)
    ]
    background_components = [
        (np.asarray(row, dtype=float) / float(norm)) * float(coef)
        for row, norm, coef in zip(background_rows, row_norms[:len(background_rows)], coefs[:len(background_rows)])
    ]
    phase_sum = np.sum(np.vstack(components), axis=0) if components else np.zeros_like(y512)
    background_fit = np.sum(np.vstack(background_components), axis=0) if background_components else np.zeros_like(y512)
    scale_total = float(np.sum(np.maximum(phase_coefs, 0.0)))
    relative_scales = (
        (np.maximum(phase_coefs, 0.0) / scale_total).astype(float)
        if scale_total > 0
        else np.zeros_like(phase_coefs, dtype=float)
    )
    baseline_sse = float(np.sum((y512 - np.mean(y512)) ** 2))
    explained = float((baseline_sse - float(sse)) / max(baseline_sse, 1e-12))
    return {
        "q": q_grid,
        "target": y512,
        "total": np.asarray(total_fit, dtype=float),
        "background": background_fit,
        "components": components,
        "labels": used_labels,
        "coefs": phase_coefs,
        "relative_scales": relative_scales,
        "sse": float(sse),
        "explained": explained,
    }


def _rapid_render_512_payload_plot(payload_path: Path, *, key: str) -> bool:
    try:
        payload = cached_read_json_file(str(payload_path), _path_mtime(payload_path))
    except Exception as exc:
        st.caption(f"Refined pattern payload could not be read: {exc}")
        return False
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        q = np.asarray(payload.get("q", []), dtype=float)
        target = np.asarray(payload.get("target", []), dtype=float)
        total = np.asarray(payload.get("total_fit", []), dtype=float)
        residual = np.asarray(payload.get("residual", []), dtype=float)
        background = np.asarray(payload.get("background", []), dtype=float)
        if q.size == 0 or target.size != q.size or total.size != q.size:
            return False
        phases = list(payload.get("phases") or [])
        palette = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#db2777", "#0891b2", "#65a30d"]
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.055,
            row_heights=[0.58, 0.24, 0.18],
            subplot_titles=("Component fit", "Difference", "Strongest Bragg peaks"),
        )
        fig.add_trace(
            go.Scattergl(
                x=q,
                y=target,
                mode="markers",
                marker=dict(color="#111827", size=4, opacity=0.48),
                name="Measured signal",
                hovertemplate="Q=%{x:.4f}<br>Measured=%{y:.4g}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if background.size == q.size:
            fig.add_trace(
                go.Scatter(
                    x=q,
                    y=background,
                    mode="lines",
                    line=dict(color="#64748b", width=1.2, dash="dash"),
                    name="Baseline term",
                    visible="legendonly",
                    hovertemplate="Q=%{x:.4f}<br>Baseline=%{y:.4g}<extra></extra>",
                ),
                row=1,
                col=1,
            )
        target_span = max(float(np.nanmax(target)) if target.size else 1.0, 1e-6)
        peak_rows: list[dict[str, object]] = []
        for idx, phase in enumerate(phases):
            label = str(phase.get("label") or f"Phase {idx + 1}")
            rel = float(phase.get("relative_scale") or 0.0)
            component = np.asarray(phase.get("component", []), dtype=float)
            if component.size != q.size:
                continue
            color = palette[idx % len(palette)]
            legend_label = f"{label} ({rel * 100:.1f}%)"
            fig.add_trace(
                go.Scatter(
                    x=q,
                    y=component,
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    fill="tozeroy",
                    opacity=0.78,
                    name=legend_label,
                    hovertemplate=f"{label}<br>Q=%{{x:.4f}}<br>Contribution=%{{y:.4g}}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            peaks = list(phase.get("top_peaks") or [])
            if peaks:
                tick_y = np.full(len(peaks), float(idx), dtype=float)
                fig.add_trace(
                    go.Scatter(
                        x=[float(peak.get("grid_q") or peak.get("q")) for peak in peaks],
                        y=tick_y,
                        mode="markers",
                        marker=dict(symbol="line-ns", size=16, color=color, line=dict(color=color, width=3)),
                        name=f"{label} key peaks",
                        showlegend=False,
                        hovertemplate=(
                            f"{label}<br>"
                            "Q=%{x:.4f}<br>"
                            "Support=%{customdata[0]}<br>"
                            "Observed=%{customdata[1]:.4g}<br>"
                            "Modeled=%{customdata[2]:.4g}<extra></extra>"
                        ),
                        customdata=[
                            [
                                str(peak.get("support") or "-"),
                                float(peak.get("observed_peak") or 0.0),
                                float(peak.get("component_peak") or 0.0),
                            ]
                            for peak in peaks
                        ],
                    ),
                    row=3,
                    col=1,
                )
                supported = sum(1 for peak in peaks if str(peak.get("support")) == "supported")
                weak = sum(1 for peak in peaks if str(peak.get("support")) == "weak support")
                missing = max(0, len(peaks) - supported - weak)
                peak_rows.append(
                    {
                        "Phase": label,
                        "Pattern preview contribution (%)": rel * 100.0,
                        "Key peak support": f"{supported} supported, {weak} weak, {missing} missing/reviewed",
                    }
                )
        fig.add_trace(
            go.Scatter(
                x=q,
                y=total,
                mode="lines",
                line=dict(color="#dc2626", width=2.5),
                name="Total hypothesis fit",
                hovertemplate="Q=%{x:.4f}<br>Total=%{y:.4g}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        if residual.size == q.size:
            fig.add_trace(
                go.Scatter(
                    x=q,
                    y=residual,
                    mode="lines",
                    line=dict(color="#0f766e", width=1.0),
                    name="Difference",
                    hovertemplate="Q=%{x:.4f}<br>Difference=%{y:.4g}<extra></extra>",
                ),
                row=2,
                col=1,
            )
            fig.add_hline(y=0, line_width=1, line_color="#94a3b8", row=2, col=1)
        fig.update_layout(
            height=520,
            margin=dict(l=18, r=18, t=48, b=42),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
                bgcolor="rgba(255,255,255,0.94)",
                bordercolor="#cbd5e1",
                borderwidth=1,
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
        )
        fig.update_yaxes(title_text="Scaled signal", gridcolor="#e5e7eb", row=1, col=1)
        fig.update_yaxes(title_text="Difference", gridcolor="#e5e7eb", row=2, col=1)
        fig.update_yaxes(
            title_text="Phases",
            tickmode="array",
            tickvals=[float(i) for i in range(len(phases))],
            ticktext=[str(phase.get("label") or f"Phase {i + 1}")[:34] for i, phase in enumerate(phases)],
            range=[len(phases) - 0.4, -0.6] if phases else None,
            gridcolor="#eef2f7",
            row=3,
            col=1,
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(title_text="Q (1/A)", gridcolor="#e5e7eb", row=3, col=1)
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            key=f"rapid_512_payload_{key}",
        )
        if peak_rows and pd is not None:
            st.caption(
                "Pattern preview contribution is the fitted contribution in this fast pattern-scoring plot, "
                "not the final GSAS-II phase fraction. Key peak support checks the strongest modeled Bragg peaks "
                "for each phase against the measured signal."
            )
            _render_responsive_result_table(pd.DataFrame(peak_rows))
        with st.expander("Strongest modeled peak checks", expanded=False):
            detail_rows: list[dict[str, object]] = []
            for phase in phases:
                for peak in phase.get("top_peaks") or []:
                    detail_rows.append(
                        {
                            "Phase": phase.get("label"),
                            "Q": peak.get("grid_q") or peak.get("q"),
                            "Support": peak.get("support"),
                            "Observed local max": peak.get("observed_peak"),
                            "Modeled contribution": peak.get("component_peak"),
                        }
                    )
            if detail_rows and pd is not None:
                _render_responsive_result_table(pd.DataFrame(detail_rows))
            else:
                st.caption("No peak-support details were stored for this hypothesis.")
        return True
    except Exception as exc:
        st.caption(f"Refined pattern plot could not be rendered: {exc}")
        return False


def _rapid_render_512_component_plot(
    row,
    nudge_scenario_root: Path | None,
    report_root: Path | None,
    *,
    key: str,
    representative_map: dict[str, str] | None = None,
) -> None:
    if row is None or nudge_scenario_root is None or report_root is None:
        st.caption("Refined pattern preview is not available for this run.")
        return
    payload_path = _rapid_payload_path_from_row(row, report_root)
    if payload_path is not None and _rapid_render_512_payload_plot(payload_path, key=key):
        return
    target_npz = nudge_scenario_root / "target_512.npz"
    config_path = report_root.parent / "pipeline_config.yaml"
    if not target_npz.exists():
        st.caption("Refined pattern target is not available yet.")
        return
    cif_paths = tuple(path for path in _rapid_split_phases(row.get("cif_paths") if hasattr(row, "get") else "") if path)
    labels = tuple(_rapid_hypothesis_family_labels(row, representative_map))
    if not cif_paths:
        st.caption("This hypothesis does not include saved CIF paths for refined-pattern rendering.")
        return
    try:
        result = _rapid_compute_512_components(
            str(target_npz),
            _path_mtime(target_npz),
            str(config_path) if config_path.exists() else "",
            _path_mtime(config_path),
            cif_paths,
            labels,
        )
    except Exception as exc:
        st.caption(f"Refined pattern preview could not be rendered: {exc}")
        return
    if result.get("error"):
        st.caption(str(result["error"]))
        return
    try:
        import plotly.graph_objects as go

        q = np.asarray(result["q"], dtype=float)
        target = np.asarray(result["target"], dtype=float)
        total = np.asarray(result["total"], dtype=float)
        background = np.asarray(result["background"], dtype=float)
        relative_scales = np.asarray(result.get("relative_scales", []), dtype=float)
        palette = ["#2563eb", "#d97706", "#16a34a", "#7c3aed", "#db2777", "#0891b2"]
        title = f"Refined pattern component fit | explained {_rapid_fmt_number(result.get('explained'), 3)}"
        st.markdown(f"**{title}**")
        scale_rows = []
        labels_for_table = list(result.get("labels", []))
        coefs_for_table = list(result.get("coefs", []))
        for idx, label in enumerate(labels_for_table):
            rel = relative_scales[idx] if idx < len(relative_scales) else np.nan
            coef = coefs_for_table[idx] if idx < len(coefs_for_table) else np.nan
            scale_rows.append(
                {
                    "Phase": str(label),
                    "Relative scale": None if not np.isfinite(rel) else float(rel * 100.0),
                    "Fitted coefficient": None if not np.isfinite(float(coef)) else float(coef),
                }
            )
        if scale_rows and pd is not None:
            st.dataframe(
                pd.DataFrame(scale_rows),
                hide_index=True,
                width="stretch",
                height=min(150, 38 + 35 * len(scale_rows)),
                column_config={
                    "Relative scale": st.column_config.NumberColumn("Relative scale (%)", format="%.1f"),
                    "Fitted coefficient": st.column_config.NumberColumn("Fitted coefficient", format="%.3g"),
                },
            )
        fig = go.Figure()
        fig.add_trace(
            go.Scattergl(
                x=q,
                y=target,
                mode="markers",
                marker=dict(color="#475569", size=3, opacity=0.42),
                name="Measured refined signal",
                hovertemplate="Q=%{x:.3f}<br>Target=%{y:.4g}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=q,
                y=background,
                mode="lines",
                line=dict(color="#94a3b8", width=1.0, dash="dash"),
                name="Baseline correction",
                visible="legendonly",
                hovertemplate="Q=%{x:.3f}<br>Correction=%{y:.4g}<extra></extra>",
            )
        )
        for idx, component in enumerate(result["components"]):
            label = result["labels"][idx] if idx < len(result["labels"]) else f"Phase {idx + 1}"
            rel = relative_scales[idx] if idx < len(relative_scales) else np.nan
            legend_label = f"{label} ({rel * 100:.1f}%)" if np.isfinite(rel) else str(label)
            color = palette[idx % len(palette)]
            fig.add_trace(
                go.Scatter(
                    x=q,
                    y=np.asarray(component, dtype=float),
                    mode="lines",
                    line=dict(color=color, width=1.8),
                    opacity=0.82,
                    name=str(legend_label),
                    hovertemplate=f"{label}<br>Q=%{{x:.3f}}<br>Contribution=%{{y:.4g}}<extra></extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=q,
                y=total,
                mode="lines",
                line=dict(color="#020617", width=2.4),
                name="Total fit",
                hovertemplate="Q=%{x:.3f}<br>Total=%{y:.4g}<extra></extra>",
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=12, r=12, t=18, b=78),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.18,
                xanchor="left",
                x=0,
                bgcolor="rgba(255,255,255,0.94)",
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(title_text="Q (1/A)", gridcolor="#e5e7eb")
        fig.update_yaxes(title_text="Scaled signal", gridcolor="#e5e7eb")
        st.plotly_chart(
            fig,
            width="stretch",
            config={"displaylogo": False, "scrollZoom": True, "responsive": True},
            key=f"rapid_512_component_{key}",
        )
    except Exception as exc:
        st.caption(f"Refined pattern preview could not be rendered: {exc}")


def _rapid_render_512_fit_selector(
    rerank512_df,
    nudge_scenario_root: Path | None,
    report_root: Path | None,
    *,
    key: str,
    limit: int,
    representative_map: dict[str, str] | None = None,
) -> None:
    if rerank512_df is None or getattr(rerank512_df, "empty", True):
        st.caption("No refined hypotheses are available for a refined-pattern preview.")
        return
    view = rerank512_df.copy()
    if "rank512" in view.columns:
        view["rank512"] = pd.to_numeric(view["rank512"], errors="coerce")
        view = view.sort_values("rank512")
    view = view.head(max(1, int(limit)))
    choice = st.selectbox(
        "Refined pattern preview",
        list(range(len(view))),
        format_func=lambda idx: (
            f"#{_rapid_to_int(view.iloc[idx].get('rank512')) or idx + 1} | "
            f"{_rapid_hypothesis_family_label_from_row(view.iloc[idx], representative_map)}"
        ),
        key=f"rapid_512_fit_selector_{key}",
    )
    _rapid_render_512_component_plot(
        view.iloc[int(choice)],
        nudge_scenario_root,
        report_root,
        key=f"{key}_{choice}",
        representative_map=representative_map,
    )


def _rapid_validation_row_for_payload(payload_path: Path):
    if pd is None:
        return None
    report_root = _rapid_report_root_from_output_path(payload_path)
    if report_root is None:
        return None
    target_plot = str(payload_path)
    if target_plot.endswith(".plotdata.json"):
        target_plot = target_plot[: -len(".plotdata.json")]
    target_path = Path(target_plot)
    csv_candidates = [
        report_root / "all_gsas_validation_summary.csv",
        report_root / "all_gsas_validation_summary.partial.csv",
    ]
    for csv_path in csv_candidates:
        if not csv_path.exists():
            continue
        try:
            df = cached_read_csv_file(str(csv_path), _path_mtime(csv_path))
        except Exception:
            continue
        if df is None or df.empty or "curve_png" not in df.columns:
            continue
        for _, row in df.iterrows():
            raw = str(row.get("curve_png") or "").strip()
            if not raw:
                continue
            raw_path = Path(raw)
            if raw == target_plot:
                return row
            if raw_path.name == target_path.name and raw_path.parent.name == target_path.parent.name:
                return row
    return None


def _rapid_phase_label_map_for_payload(payload_path: Path) -> dict[str, str]:
    row = _rapid_validation_row_for_payload(payload_path)
    if row is None:
        return {}
    labels = _rapid_phase_labels_from_row(row)
    formulas = _rapid_split_phases(row.get("formulas") or row.get("formula_keys") or row.get("phase_ids"))
    pids = _rapid_split_phases(row.get("phase_ids"))
    mapping: dict[str, str] = {}
    for idx, label in enumerate(labels):
        keys = {
            label,
            _rapid_choice_formula(label),
        }
        if idx < len(formulas):
            keys.add(str(formulas[idx]))
            keys.add(_rapid_formula_key(formulas[idx]))
        if idx < len(pids):
            keys.add(str(pids[idx]))
        for key in keys:
            key_text = str(key or "").strip()
            if key_text:
                mapping[key_text] = label
    return mapping


def _rapid_repair_plot_payload_labels(payload: dict, payload_path: Path) -> dict:
    if not isinstance(payload, dict):
        return payload
    phase_order = payload.get("phase_order") or []
    if not phase_order:
        return payload
    existing_labels = dict(payload.get("phase_labels") or {})
    main_phase_label = _rapid_main_phase_label_from_output_path(payload_path)
    repaired_labels = dict(existing_labels)
    changed = False
    if main_phase_label:
        for phase_id in phase_order:
            phase_key = str(phase_id)
            current = str(repaired_labels.get(phase_key) or phase_key)
            if phase_key == "Main phase" or current == "Main phase":
                repaired_labels[phase_key] = main_phase_label
                changed = True
    row_label_map = _rapid_phase_label_map_for_payload(payload_path)
    if row_label_map:
        for phase_id in phase_order:
            phase_key = str(phase_id)
            if phase_key == "Main phase":
                continue
            current = str(repaired_labels.get(phase_key) or phase_key)
            mapped = (
                row_label_map.get(current)
                or row_label_map.get(_rapid_choice_formula(current))
                or row_label_map.get(phase_key)
            )
            if mapped and mapped != current:
                repaired_labels[phase_key] = mapped
                changed = True
    needs_lookup = any(
        _rapid_needs_formula(repaired_labels.get(str(phase_id)) or str(phase_id), str(phase_id))
        for phase_id in phase_order
    )
    if not needs_lookup:
        if not changed:
            return payload
        repaired_payload = dict(payload)
        repaired_payload["phase_labels"] = repaired_labels
        return repaired_payload

    lookup = _rapid_formula_lookup_for_output(payload_path)
    if not lookup:
        if changed:
            repaired_payload = dict(payload)
            repaired_payload["phase_labels"] = repaired_labels
            return repaired_payload
        return payload
    for phase_id in phase_order:
        phase_key = str(phase_id)
        current = repaired_labels.get(phase_key)
        if _rapid_needs_formula(current or phase_key, phase_key):
            formula = lookup.get(phase_key)
            if formula:
                repaired_labels[phase_key] = formula
                changed = True
    if not changed:
        return payload
    repaired_payload = dict(payload)
    repaired_payload["phase_labels"] = repaired_labels
    return repaired_payload


def _peak_support_arrays_from_payload(payload: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    arrays = payload.get("arrays") or {}
    try:
        x = np.asarray(arrays.get("x"), dtype=float)
        yobs = np.asarray(arrays.get("yobs"), dtype=float)
        ycalc = np.asarray(arrays.get("ycalc"), dtype=float)
        resid = np.asarray(arrays.get("resid"), dtype=float)
    except Exception:
        return None
    if min(x.size, yobs.size, ycalc.size, resid.size) < 10:
        return None
    n = min(x.size, yobs.size, ycalc.size, resid.size)
    x, yobs, ycalc, resid = x[:n], yobs[:n], ycalc[:n], resid[:n]
    finite = np.isfinite(x) & np.isfinite(yobs) & np.isfinite(ycalc) & np.isfinite(resid)
    if int(np.sum(finite)) < 10:
        return None
    x, yobs, ycalc, resid = x[finite], yobs[finite], ycalc[finite], resid[finite]
    order = np.argsort(x)
    return x[order], yobs[order], ycalc[order], resid[order]


def _local_peak_support(
    x: np.ndarray,
    yobs: np.ndarray,
    ycalc: np.ndarray,
    resid: np.ndarray,
    tick_x: float,
) -> dict[str, object] | None:
    if not np.isfinite(tick_x) or x.size < 10:
        return None
    idx = int(np.argmin(np.abs(x - float(tick_x))))
    n = int(x.size)
    peak_radius = 4
    shoulder_inner = 7
    shoulder_outer = 18
    lo = max(0, idx - peak_radius)
    hi = min(n, idx + peak_radius + 1)
    shoulder_parts = []
    if idx - shoulder_inner > 0:
        shoulder_parts.append(np.arange(max(0, idx - shoulder_outer), max(0, idx - shoulder_inner)))
    if idx + shoulder_inner < n:
        shoulder_parts.append(np.arange(min(n, idx + shoulder_inner + 1), min(n, idx + shoulder_outer + 1)))
    shoulder_ix = np.concatenate(shoulder_parts) if shoulder_parts else np.arange(lo, hi)
    if shoulder_ix.size == 0:
        shoulder_ix = np.arange(lo, hi)

    obs_base = float(np.nanmedian(yobs[shoulder_ix]))
    calc_base = float(np.nanmedian(ycalc[shoulder_ix]))
    obs_peak = float(np.nanmax(yobs[lo:hi]) - obs_base)
    calc_peak = float(np.nanmax(ycalc[lo:hi]) - calc_base)
    local_resid = np.asarray(resid[max(0, idx - shoulder_outer):min(n, idx + shoulder_outer + 1)], dtype=float)
    resid_med = float(np.nanmedian(local_resid)) if local_resid.size else 0.0
    mad = float(np.nanmedian(np.abs(local_resid - resid_med))) if local_resid.size else 0.0
    noise = max(1.4826 * mad, float(np.nanstd(local_resid)) * 0.5 if local_resid.size else 0.0, 1e-6)
    resid_window = np.asarray(resid[lo:hi], dtype=float)
    neg_resid = float(np.nanmin(resid_window)) if resid_window.size else 0.0
    pos_resid = float(np.nanmax(resid_window)) if resid_window.size else 0.0
    support_ratio = obs_peak / max(calc_peak, 1e-6)

    if calc_peak > max(3.0 * noise, obs_peak * 1.65) and neg_resid < -2.0 * noise:
        verdict = "Possible overfit"
    elif obs_peak > max(3.0 * noise, calc_peak * 1.65) and pos_resid > 2.0 * noise:
        verdict = "Underfit / missing intensity"
    elif calc_peak > 2.0 * noise and obs_peak > 2.0 * noise and 0.45 <= support_ratio <= 1.85:
        verdict = "Supported"
    elif calc_peak <= 2.0 * noise and obs_peak <= 2.0 * noise:
        verdict = "Weak / not visible"
    else:
        verdict = "Ambiguous"

    return {
        "x": float(tick_x),
        "observed_peak": obs_peak,
        "calculated_peak": calc_peak,
        "support_ratio": support_ratio,
        "noise": noise,
        "verdict": verdict,
    }


def _rapid_peak_support_rows(payload: dict, *, top_n: int = 5) -> list[dict[str, object]]:
    arrays = _peak_support_arrays_from_payload(payload)
    if arrays is None:
        return []
    x, yobs, ycalc, resid = arrays
    phase_order = [str(item) for item in (payload.get("phase_order") or [])]
    phase_labels = payload.get("phase_labels") or {}
    phase_weights = payload.get("phase_weights") or {}
    major_details = payload.get("phase_major_tick_details") or {}
    major_ticks = payload.get("phase_major_ticks") or {}
    rows: list[dict[str, object]] = []
    for phase_name in phase_order:
        label = str(phase_labels.get(phase_name) or phase_name)
        details = list(major_details.get(phase_name) or [])
        if not details:
            details = [
                {"x": float(tick), "rank": idx + 1, "hkl": "", "relative_strength": None}
                for idx, tick in enumerate(np.asarray(major_ticks.get(phase_name, []), dtype=float))
                if np.isfinite(tick)
            ]
        details = details[:max(1, int(top_n))]
        peak_summaries = []
        counts = {"Supported": 0, "Possible overfit": 0, "Underfit / missing intensity": 0, "Weak / not visible": 0, "Ambiguous": 0}
        for idx, item in enumerate(details, start=1):
            try:
                tick_x = float(item.get("x"))
            except Exception:
                continue
            support = _local_peak_support(x, yobs, ycalc, resid, tick_x)
            if support is None:
                continue
            verdict = str(support["verdict"])
            counts[verdict] = counts.get(verdict, 0) + 1
            hkl = str(item.get("hkl") or "").strip()
            peak_summaries.append(
                f"#{idx} x={tick_x:.4g}"
                + (f" {hkl}" if hkl else "")
                + f": {verdict.lower()} ({support['support_ratio']:.2f}x)"
            )
        checked = sum(counts.values())
        if checked == 0:
            continue
        if counts.get("Possible overfit", 0) >= max(1, int(np.ceil(checked / 2))):
            overall = "Review: possible overfit"
        elif counts.get("Supported", 0) >= max(1, int(np.ceil(0.6 * checked))) and counts.get("Possible overfit", 0) == 0:
            overall = "Supported"
        elif counts.get("Underfit / missing intensity", 0) >= max(1, int(np.ceil(checked / 2))):
            overall = "Review: underfit"
        else:
            overall = "Mixed / ambiguous"
        try:
            wt_val = float(phase_weights.get(phase_name))
        except Exception:
            wt_val = None
        rows.append(
            {
                "Phase": label,
                "Weight %": wt_val,
                "Key peaks checked": checked,
                "Supported": counts.get("Supported", 0),
                "Overfit flags": counts.get("Possible overfit", 0),
                "Underfit flags": counts.get("Underfit / missing intensity", 0),
                "Weak/ambiguous": counts.get("Weak / not visible", 0) + counts.get("Ambiguous", 0),
                "Verdict": overall,
                "Peak details": "; ".join(peak_summaries),
            }
        )
    return rows


def _rapid_render_peak_support_summary(payload: dict) -> None:
    if pd is None:
        return
    rows = _rapid_peak_support_rows(payload, top_n=5)
    if not rows:
        return
    with st.expander("Key peak support summary", expanded=True):
        st.caption(
            "Checks the strongest Bragg ticks for each phase against local observed and calculated intensity. "
            "This is a screening diagnostic, not a replacement for inspecting the fit."
        )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Weight %": st.column_config.NumberColumn("Weight %", format="%.2f"),
                "Peak details": st.column_config.TextColumn("Peak details", width="large"),
            },
        )


def _rapid_render_curve_plot(
    curve_csv_path: Path | None,
    fallback_png_path: Path | None,
    *,
    title: str,
    key_suffix: str = "default",
) -> None:
    def _plot_key(prefix: str, path: Path | None) -> str:
        key_source = f"{key_suffix}|{path or ''}|{title}"
        digest = hashlib.md5(key_source.encode("utf-8", errors="replace")).hexdigest()[:12]
        return f"{prefix}_{digest}"

    if fallback_png_path and fallback_png_path.exists() and INTERACTIVE_PLOTS_AVAILABLE:
        payload_path = Path(str(fallback_png_path) + ".plotdata.json")
        if payload_path.exists():
            try:
                payload = load_interactive_payload(payload_path)
                payload = _rapid_repair_plot_payload_labels(payload, payload_path)
                fig = build_plotly_figure_from_payload(payload)
                if fig is not None:
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
                        key=_plot_key("rapid_full_curve", payload_path),
                    )
                    _rapid_render_peak_support_summary(payload)
                    with st.expander("Static publication plot", expanded=False):
                        st.image(str(fallback_png_path), width="stretch")
                    return
            except Exception as exc:
                st.warning(f"Publication-style interactive plot could not be rendered: {exc}")

    if curve_csv_path and curve_csv_path.exists() and pd is not None:
        try:
            curve = cached_read_csv_file(str(curve_csv_path), _path_mtime(curve_csv_path))
            required = {"Q", "yobs", "ycalc", "residual"}
            if required.issubset(set(curve.columns)):
                curve = curve.copy()
                for col in ["Q", "yobs", "ycalc", "residual"]:
                    curve[col] = pd.to_numeric(curve[col], errors="coerce")
                curve = curve.dropna(subset=["Q", "yobs", "ycalc", "residual"]).sort_values("Q")
                if not curve.empty:
                    from plotly.subplots import make_subplots
                    import plotly.graph_objects as go

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.07,
                        row_heights=[0.72, 0.28],
                    )
                    fig.add_trace(
                        go.Scattergl(
                            x=curve["Q"],
                            y=curve["yobs"],
                            mode="markers",
                            marker=dict(size=3, color="#243447", opacity=0.55),
                            name="Observed",
                            hovertemplate="Q=%{x:.4f}<br>Observed=%{y:.4g}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=curve["Q"],
                            y=curve["ycalc"],
                            mode="lines",
                            line=dict(width=1.4, color="#c43d3d"),
                            name="Calculated",
                            hovertemplate="Q=%{x:.4f}<br>Calculated=%{y:.4g}<extra></extra>",
                        ),
                        row=1,
                        col=1,
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=curve["Q"],
                            y=curve["residual"],
                            mode="lines",
                            line=dict(width=1.0, color="#087f8c"),
                            name="Difference",
                            hovertemplate="Q=%{x:.4f}<br>Difference=%{y:.4g}<extra></extra>",
                        ),
                        row=2,
                        col=1,
                    )
                    fig.add_hline(y=0, line_width=1, line_color="#6b7280", row=2, col=1)
                    fig.update_layout(
                        title=title,
                        height=620,
                        margin=dict(l=20, r=20, t=58, b=48),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    fig.update_yaxes(title_text="Intensity", row=1, col=1, gridcolor="#e4ebe7")
                    fig.update_yaxes(title_text="Difference", row=2, col=1, gridcolor="#e4ebe7")
                    fig.update_xaxes(title_text="Q (1/A)", row=2, col=1, gridcolor="#e4ebe7")
                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config={"displaylogo": False, "scrollZoom": True, "responsive": True},
                        key=_plot_key("rapid_curve", curve_csv_path),
                    )
                    return
        except Exception as exc:
            st.warning(f"Interactive curve could not be rendered: {exc}")
    if fallback_png_path and fallback_png_path.exists():
        st.image(str(fallback_png_path), width="stretch")
    else:
        st.caption("Curve plot not found for this hypothesis.")


def _rapid_render_gsas_fit_selector(
    gsas_df,
    report_root: Path | None,
    *,
    key: str,
    limit: int,
    representative_map: dict[str, str] | None = None,
) -> None:
    if gsas_df is None or getattr(gsas_df, "empty", True):
        st.caption("No refinement fit plots are available yet.")
        return
    view = gsas_df.copy()
    if "gsas_rwp_rank" in view.columns:
        view["gsas_rwp_rank"] = pd.to_numeric(view["gsas_rwp_rank"], errors="coerce")
        view = view.sort_values("gsas_rwp_rank", na_position="last")
    view = view.head(max(1, int(limit)))
    choice = st.selectbox(
        "Refinement fit preview",
        list(range(len(view))),
        format_func=lambda idx: (
            f"#{_rapid_display_rank(view.iloc[idx].get('gsas_rwp_rank'))} | "
            f"{_rapid_hypothesis_family_label_from_row(view.iloc[idx], representative_map)}"
        ),
        key=f"rapid_gsas_fit_selector_{key}",
    )
    row = view.iloc[int(choice)]
    curve_path = _rapid_local_path(row.get("curve_png"), report_root)
    curve_csv_path = _rapid_local_path(row.get("curve_csv"), report_root)
    _rapid_render_curve_plot(
        curve_csv_path,
        curve_path,
        title=_rapid_hypothesis_family_label_from_row(row, representative_map),
        key_suffix=f"fit_selector_{key}_{choice}",
    )


def _rapid_variant_result_key(scenario: str, row) -> str:
    rank512 = _rapid_to_int(row.get("rank512")) if hasattr(row, "get") else None
    phase_ids = str(row.get("phase_ids") if hasattr(row, "get") else "")
    digest = hashlib.md5(f"{scenario}|{rank512}|{phase_ids}".encode("utf-8")).hexdigest()[:10]
    return f"rapid_variant_validation_{digest}"


def _rapid_can_run_variant_validation() -> tuple[bool, str]:
    report_root = _rapid_report_root()
    if report_root is None:
        return False, "No rapid result set is available for targeted refinement."
    run_dir = report_root.parent if report_root.name == "rapid_results" else None
    if run_dir is None or not (run_dir / "pipeline_config.yaml").exists():
        return False, "Targeted refinement is available for live rapid runs, not the saved benchmark fixture."
    helper = Path(PROJECT_ROOT) / "scripts" / "rapid_hypothesis_pipeline.py"
    if not helper.exists():
        return False, "Rapid refinement helper is missing from this app checkout."
    return True, ""


def _rapid_run_variant_validation(
    scenario: str,
    row,
    *,
    frozen_formula_keys: list[str] | None = None,
    status_box=None,
) -> tuple[dict[str, object] | None, str | None]:
    report_root = _rapid_report_root()
    if report_root is None:
        return None, "No rapid result set is available for targeted refinement."
    run_dir = report_root.parent if report_root.name == "rapid_results" else None
    if run_dir is None:
        return None, "Targeted refinement is available for live rapid runs, not the saved benchmark fixture."
    config_path = run_dir / "pipeline_config.yaml"
    if not config_path.exists():
        return None, f"Could not find run config: {config_path}"
    if pd is None:
        return None, "Pandas is not available in this app environment."
    rank512 = _rapid_to_int(row.get("rank512")) if hasattr(row, "get") else None
    if rank512 is None:
        return None, "The selected hypothesis has no refined-search rank."
    row_payload = dict(row.to_dict() if hasattr(row, "to_dict") else dict(row))
    row_payload = _rapid_enrich_row_with_nudged_cifs(row_payload, report_root)
    if frozen_formula_keys:
        row_payload["frozen_formula_keys"] = "|".join(str(key) for key in frozen_formula_keys if str(key).strip())
        row_payload["targeted_refinement"] = True
    required_cols = {"rank512", "rank64", "phase_ids", "formula_keys", "formulas", "score512", "r2_512", "sse512"}
    missing_cols = sorted(col for col in required_cols if col not in row_payload)
    if missing_cols:
        return None, "The selected hypothesis is missing required rapid-scoring fields: " + ", ".join(missing_cols)
    out_root = report_root / scenario / "interactive_variant_validations"
    out_root.mkdir(parents=True, exist_ok=True)
    base_gpx = _rapid_targeted_base_gpx(report_root, scenario)
    request_path = out_root / f"request_rank512_{rank512}_{int(time.time())}.json"
    request_path.write_text(
        json.dumps(
            {
                "config_path": str(config_path),
                "dataset_name": run_dir.name,
                "row": row_payload,
                "out_root": str(out_root),
                "base_gpx": str(base_gpx) if base_gpx and base_gpx.exists() else "",
            },
            default=str,
        ),
        encoding="utf-8",
    )
    if status_box is not None:
        try:
            status_box.write("Prepared targeted refinement request.")
            status_box.write("Launching GSAS-II refinement for the selected phase set. This usually takes 1-3 minutes.")
        except Exception:
            pass
    code = r"""
import json
import sys
from pathlib import Path

project_root = Path.cwd()
scripts_root = project_root / "scripts"
for candidate in (str(scripts_root), str(project_root)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import pandas as pd
import yaml

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
cfg = yaml.safe_load(Path(request["config_path"]).read_text(encoding="utf-8"))
dataset_name = str(request.get("dataset_name") or "")
dataset = next((d for d in cfg.get("datasets", []) if str(d.get("name")) == dataset_name), None)
if dataset is None:
    dataset = (cfg.get("datasets") or [None])[0]
if dataset is None:
    raise RuntimeError("No dataset was found in pipeline_config.yaml")

from aniso_db_loader import CatalogPaths, DBLoader
from scripts.rapid_hypothesis_pipeline import _validate_gsas

db_cfg = cfg.get("db") or {}
loader = DBLoader(CatalogPaths(
    catalog_csv=str(db_cfg["catalog_csv"]),
    cif_map_json=db_cfg.get("cif_map_json"),
    original_json=db_cfg.get("original_json"),
))
row = dict(request["row"])
df = pd.DataFrame([row])
out = _validate_gsas(
    df,
    loader,
    cfg,
    dataset,
    Path(request["out_root"]),
    top_n=1,
    base_gpx=request.get("base_gpx") or None,
)
if out.empty:
    raise RuntimeError("Targeted refinement returned no rows")
record = out.where(pd.notnull(out), None).iloc[0].to_dict()
print(json.dumps(record, default=str))
"""
    cmd = [
        sys.executable,
        "-c",
        code,
        str(request_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(PROJECT_ROOT))
    except subprocess.TimeoutExpired:
        return None, "Targeted refinement timed out."
    except Exception as exc:
        return None, f"Could not start targeted refinement: {exc}"
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-12:])
        return None, f"Targeted refinement failed.\n{tail}"
    if status_box is not None:
        try:
            status_box.write("GSAS-II completed. Reading refined phase fractions and fit curve.")
        except Exception:
            pass
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            return data, None
        except Exception:
            continue
    return None, "Targeted refinement finished but did not return a result."


def render_rapid_hypothesis_explorer() -> None:
    st.subheader("Rapid Results")
    st.caption(
        "Inspect the fast hypothesis path: coarse matching, lattice nudging, high-resolution pattern scoring, and final refinement ranking."
    )
    if (
        not st.session_state.get("run_active")
        and not st.session_state.get("selected_run_dir")
        and not st.session_state.get("run_dir")
        and not st.session_state.get("last_finished_run_dir")
    ):
        _render_no_run_workspace_empty_state(
            "Rapid Results",
            details=(
                "Rapid results will show coarse hypotheses, lattice nudging, pattern scoring, "
                "final refinement ranking, and the solution inspector after a rapid run is loaded."
            ),
        )
        return

    report_root = _rapid_report_root()
    nudge_root = _rapid_nudge_root()
    live_mode = bool(
        report_root is not None
        and report_root.name == "rapid_results"
        and report_root.parent.name.startswith("run_")
    )
    if not live_mode and not st.session_state.get("rapid_demo_fixture_enabled", False):
        st.info(
            "Live rapid-result artifacts are not present for the current run yet. "
            "Enable the LK-99 benchmark fixture in the sidebar to inspect the current rapid-mode prototype data."
        )
        return
    if report_root is None:
        st.info("No rapid-mode report artifacts were found on this machine.")
        return
    if live_mode:
        st.success(f"Showing rapid results for `{report_root.parent.name}`.")
    else:
        st.info("Showing the saved LK-99 benchmark while live rapid outputs are being connected to uploaded runs.")
    render_magnetic_precheck_panel(report_root.parent if live_mode else VIEW_RUN_DIR, compact=False)

    summary_path = report_root / "summary.json"
    gsas_csv_path = report_root / "all_gsas_validation_summary.csv"
    if not summary_path.exists():
        if live_mode:
            st.info(
                "Rapid results are not ready for this run yet. "
                "Use Run Monitor until coarse search, lattice nudge, high-resolution pattern scoring, and final refinement ranking write their outputs."
            )
            render_rapid_partial_outputs(report_root)
        else:
            st.info("No rapid-mode summary is available yet.")
        return
    try:
        summary = cached_read_json_file(str(summary_path), _path_mtime(summary_path))
    except Exception as exc:
        st.error(f"Could not read rapid-mode summary: {exc}")
        return

    gsas_df = _rapid_read_csv(gsas_csv_path)
    if gsas_df is None or gsas_df.empty:
        st.info("Final refinement ranking is not available yet. Showing the rapid stages completed so far.")
        render_rapid_partial_outputs(report_root)
        return
    if "source_scenario" not in gsas_df.columns:
        gsas_df["source_scenario"] = "live_run"

    scenario_labels = {
        "known_main_real": "Known main phase fixed",
        "unknown_main": "No main phase supplied",
        "live_run": "Live rapid run",
    }
    available_scenarios = [
        key
        for key, value in summary.items()
        if isinstance(value, dict) and not str(key).startswith("_")
    ]
    if not available_scenarios:
        st.info("Rapid results are being prepared. Refresh after the rapid run advances past the first stage.")
        return
    scenario = st.selectbox(
        "Rapid result set",
        available_scenarios,
        format_func=lambda key: (summary.get(key, {}) or {}).get("label") or scenario_labels.get(key, key),
        key="rapid_hypothesis_scenario",
    )

    scenario_gsas = gsas_df[gsas_df["source_scenario"].astype(str) == scenario].copy()
    for col in ["rank64", "rank512", "gsas_rwp_rank", "score512", "rwp", "seconds"]:
        if col in scenario_gsas.columns:
            scenario_gsas[col] = pd.to_numeric(scenario_gsas[col], errors="coerce")

    scenario_summary = summary.get(scenario, {})
    timings = scenario_summary.get("timings", {}) or {}
    rank_cards = [
        ("Coarse rank", scenario_summary.get("target_rank64", "-")),
        ("Robustness rank", scenario_summary.get("target_rank_loo", "-")),
        ("Pattern rank", scenario_summary.get("target_rank512", "-")),
        ("Final rank", scenario_summary.get("target_rank_gsas", "-")),
    ]
    if any(str(value).strip().lower() not in {"", "-", "none", "nan"} for _, value in rank_cards):
        st.markdown("**Ranking snapshot**")
        _render_plot_meta_cards(
            rank_cards,
            numeric_labels={"Coarse rank", "Robustness rank", "Pattern rank", "Final rank"},
        )
    st.markdown("**Stage timing**")
    st.caption("Elapsed time in seconds for each rapid stage.")
    _render_plot_meta_cards(
        [
            ("Input prep", _rapid_fmt_number(timings.get("signal_seconds"), 1)),
            ("Coarse search", _rapid_fmt_number(timings.get("search64_seconds"), 1)),
            ("Lattice nudge", _rapid_fmt_number(timings.get("nudge_seconds"), 1)),
            ("Pattern scoring", _rapid_fmt_number(timings.get("rerank512_seconds"), 1)),
            ("Refinement wall", _rapid_fmt_number(timings.get("gsas_wall_seconds"), 1)),
            ("Refinement total", _rapid_fmt_number(timings.get("gsas_total_seconds"), 1)),
            ("Total", _rapid_fmt_number(timings.get("total_seconds"), 1)),
        ],
        numeric_labels={
            "Input prep",
            "Coarse search",
            "Lattice nudge",
            "Pattern scoring",
            "Refinement wall",
            "Refinement total",
            "Total",
        },
        compact=True,
    )

    nudge_scenario_root = nudge_root / scenario if nudge_root else None
    beam64_df = _rapid_read_csv(nudge_scenario_root / "beam64_input_top.csv" if nudge_scenario_root else None)
    nudge_df = _rapid_read_csv(nudge_scenario_root / "nudge_results.csv" if nudge_scenario_root else None)
    shadow_filter_df = _rapid_read_csv(nudge_scenario_root / "main_shadow_nudge_filter.csv" if nudge_scenario_root else None)
    rerank512_df = _rapid_read_512_table(nudge_scenario_root)
    representative_map = _rapid_build_representative_map(beam64_df, nudge_df, shadow_filter_df, rerank512_df, scenario_gsas)
    beam64_display_df = _rapid_dedupe_hypotheses(
        beam64_df,
        sort_col="rank64",
        ascending=True,
        representative_map=representative_map,
    )
    rerank512_display_df = _rapid_dedupe_hypotheses(
        rerank512_df,
        sort_col="rank512",
        ascending=True,
        representative_map=representative_map,
    )
    scenario_gsas_display = _rapid_dedupe_hypotheses(
        scenario_gsas,
        sort_col="gsas_rwp_rank",
        ascending=True,
        representative_map=representative_map,
    )

    stage_tabs = st.tabs(["Coarse search", "Lattice nudge", "Pattern scoring", "Final refinement ranking", "Solution inspector"])
    with stage_tabs[0]:
        st.markdown("**Best hypotheses after the coarse residual search**")
        st.caption("This stage searches broadly and keeps promising phase combinations for more careful checks.")
        st.caption(f"Stage time: {_rapid_fmt_number(timings.get('search64_seconds'), 1)} s")
        _rapid_stage_table(
            beam64_display_df,
            "64",
            sort_col="rank64",
            ascending=True,
            limit=int(st.session_state.get("rapid_stage_output_limit", 10)),
            show_details=bool(st.session_state.get("show_expert_params", False)),
            representative_map=representative_map,
        )

    with stage_tabs[1]:
        st.markdown("**RADAR-style one-phase lattice nudge results**")
        st.caption("Each unique phase is nudged within a small symmetry-respecting lattice window before high-resolution pattern scoring.")
        st.caption(f"Stage time: {_rapid_fmt_number(timings.get('nudge_seconds'), 1)} s")
        if nudge_df is not None and not nudge_df.empty:
            nudge_view = nudge_df.copy()
            for col in ["best_score", "distance_from_start", "seconds", "a", "b", "c", "alpha", "beta", "gamma"]:
                if col in nudge_view.columns:
                    nudge_view[col] = pd.to_numeric(nudge_view[col], errors="coerce")
            _rapid_stage_table(
                nudge_view,
                "nudge",
                sort_col="best_score",
                ascending=False,
                limit=max(10, int(st.session_state.get("rapid_stage_output_limit", 10))),
                show_details=bool(st.session_state.get("show_expert_params", False)),
            )
        else:
            st.caption("Nudge output is not available in this environment.")

    with stage_tabs[2]:
        st.markdown("**Hypotheses after high-resolution pattern scoring**")
        st.caption("This stage compares the nudged candidate patterns at higher resolution before spending time on refinement.")
        st.caption(f"Stage time: {_rapid_fmt_number(timings.get('rerank512_seconds'), 1)} s")
        _rapid_stage_table(
            rerank512_display_df,
            "512",
            sort_col="rank512",
            ascending=True,
            limit=int(st.session_state.get("rapid_stage_output_limit", 10)),
            show_details=bool(st.session_state.get("show_expert_params", False)),
            representative_map=representative_map,
        )
        _rapid_render_512_fit_selector(
            rerank512_display_df,
            nudge_scenario_root,
            report_root,
            key=f"{scenario}_shortlist",
            limit=int(st.session_state.get("rapid_stage_output_limit", 10)),
            representative_map=representative_map,
        )

    with stage_tabs[3]:
        st.markdown("**Final refinement ranking of candidate hypotheses**")
        st.caption("Lower refinement quality values indicate a better fit after the rapid shortlist is checked by final refinement.")
        st.caption(
            "Wall time: "
            f"{_rapid_fmt_number(timings.get('gsas_wall_seconds'), 1)} s; "
            "summed refinement time: "
            f"{_rapid_fmt_number(timings.get('gsas_total_seconds'), 1)} s"
        )
        _rapid_stage_table(
            scenario_gsas_display,
            "gsas",
            sort_col="gsas_rwp_rank",
            ascending=True,
            limit=int(st.session_state.get("rapid_gsas_validation_limit", 10)),
            show_details=bool(st.session_state.get("show_expert_params", False)),
            representative_map=representative_map,
        )
        st.markdown("**Shortlist component preview**")
        _rapid_render_512_fit_selector(
            rerank512_display_df,
            nudge_scenario_root,
            report_root,
            key=f"{scenario}_final",
            limit=int(st.session_state.get("rapid_stage_output_limit", 10)),
            representative_map=representative_map,
        )
        st.markdown("**Refinement fit preview**")
        _rapid_render_gsas_fit_selector(
            scenario_gsas_display,
            report_root,
            key=f"{scenario}_final",
            limit=int(st.session_state.get("rapid_gsas_validation_limit", 10)),
            representative_map=representative_map,
        )

    with stage_tabs[4]:
        if scenario_gsas_display is None or scenario_gsas_display.empty:
            st.caption("No ranked hypotheses to inspect.")
            return

        st.markdown("**1. Choose a ranked hypothesis**")
        st.caption("Start from the final refinement ranking, then inspect the phase slots and available family variants.")
        scenario_gsas_display = scenario_gsas_display.sort_values("gsas_rwp_rank")
        options = list(range(len(scenario_gsas_display)))
        selected_ix = st.selectbox(
            "Hypothesis",
            options,
            format_func=lambda i: (
                f"Final #{_rapid_display_rank(scenario_gsas_display.iloc[i].get('gsas_rwp_rank'))} | "
                f"Pattern #{_rapid_display_rank(scenario_gsas_display.iloc[i].get('rank512'))} | "
                f"{_rapid_hypothesis_family_label_from_row(scenario_gsas_display.iloc[i], representative_map)}"
            ),
            key=f"rapid_hypothesis_pick_{scenario}",
        )
        selected_row = scenario_gsas_display.iloc[int(selected_ix)]
        selected_phases = _rapid_hypothesis_family_labels(selected_row, representative_map)

        selected_summary_cols = st.columns(3)
        selected_summary_cols[0].metric("Final Rwp", _rapid_fmt_number(selected_row.get("rwp"), 3))
        selected_summary_cols[1].metric("Pattern rank", _rapid_display_rank(selected_row.get("rank512")))
        selected_summary_cols[2].metric("Coarse rank", _rapid_display_rank(selected_row.get("rank64")))

        st.markdown("**2. Phase family choices**")
        edited_phases: list[str] = []
        if selected_phases and st.session_state.get("rapid_enable_family_variants", True):
            phase_cols = st.columns(min(3, max(1, len(selected_phases))))
            variant_source_rows = (scenario_gsas, rerank512_df, nudge_df, shadow_filter_df)
            for idx, phase in enumerate(selected_phases):
                with phase_cols[idx % len(phase_cols)]:
                    options_for_phase = _rapid_variant_options(variant_source_rows, phase)
                    edited_phases.append(
                        st.selectbox(
                            f"Phase {idx + 1}",
                            options_for_phase,
                            index=0,
                            key=f"rapid_variant_{scenario}_{selected_ix}_{idx}",
                        )
                    )
        elif selected_phases:
            edited_phases = selected_phases
            st.caption("Family variant swapping is off in the sidebar.")
        else:
            st.caption("This row has no parsed phase list.")
        edited_phases = [phase for phase in edited_phases if phase != RAPID_REMOVE_PHASE_OPTION]

        if selected_phases and not edited_phases:
            matched_row = None
            rerank_matched_row = None
        else:
            matched_row = _rapid_match_hypothesis(scenario_gsas, edited_phases) if edited_phases else selected_row
            rerank_matched_row = _rapid_match_hypothesis(rerank512_df, edited_phases) if edited_phases else None
        validation_row = rerank_matched_row
        manual_validation_reason = None
        if validation_row is None and edited_phases:
            validation_row, manual_validation_reason = _rapid_manual_validation_row(
                edited_phases,
                candidate_rows=(scenario_gsas, rerank512_df, nudge_df),
                fallback_row=selected_row,
            )
        validation_key = _rapid_variant_result_key(scenario, validation_row) if validation_row is not None else None
        if validation_key and validation_key in st.session_state:
            matched_row = pd.Series(st.session_state[validation_key])
        is_manual_validation_row = validation_row is not None and (_rapid_to_int(validation_row.get("rank512")) or 0) >= 9000
        if validation_row is not None and not is_manual_validation_row:
            st.markdown("**Selected refined-pattern preview**")
            _rapid_render_512_component_plot(
                validation_row,
                nudge_scenario_root,
                report_root,
                key=f"{scenario}_inspector_{validation_key or selected_ix}",
                representative_map=representative_map,
            )
        elif validation_row is not None:
            st.info("This edited combination is not in the refined shortlist. Run targeted refinement to generate its fit preview.")
        selected_formula_keys = _rapid_split_phases(selected_row.get("formula_keys") or selected_row.get("formulas"))
        edited_formula_keys = (
            _rapid_split_phases(validation_row.get("formula_keys") or validation_row.get("formulas"))
            if validation_row is not None
            else []
        )
        frozen_formula_keys: list[str] = []
        if edited_phases and selected_phases and len(selected_formula_keys) == len(edited_formula_keys):
            for old_phase, new_phase, old_key, new_key in zip(
                selected_phases,
                edited_phases,
                selected_formula_keys,
                edited_formula_keys,
            ):
                if (
                    _rapid_choice_formula(old_phase) == _rapid_choice_formula(new_phase)
                    and _rapid_choice_sg(old_phase) == _rapid_choice_sg(new_phase)
                    and str(old_key) == str(new_key)
                ):
                    frozen_formula_keys.append(str(old_key))
        if matched_row is None:
            if validation_row is None:
                st.warning(
                    "This edited combination has not been refined yet, and the app could not map every selected phase to a catalog structure."
                )
            else:
                st.warning(
                    "This edited combination has not been refined yet. Run targeted refinement to evaluate it."
                )
        else:
            if sorted(edited_phases) != sorted(selected_phases):
                st.success("A refinement result for this edited variant combination is available.")

        st.markdown("**3. Targeted refinement result**")
        validate_cols = st.columns([0.62, 0.38])
        can_run_validation, validation_reason = _rapid_can_run_variant_validation()
        button_disabled = validation_row is None or not can_run_validation
        with validate_cols[0]:
            if validation_row is None:
                st.caption(manual_validation_reason or "This phase set cannot be mapped to catalog structures for refinement.")
            elif not can_run_validation:
                st.caption(validation_reason)
            elif rerank_matched_row is None:
                st.caption("This exact phase set was not in the refined shortlist. The button will run a manual targeted refinement for the selected phases.")
            else:
                st.caption("Runs a targeted refinement only for the selected rapid hypothesis.")
        with validate_cols[1]:
            if st.button(
                "Run targeted refinement",
                disabled=button_disabled,
                width="stretch",
                help="Run a targeted final refinement for the selected phase combination.",
            ):
                with st.status("Running targeted refinement", expanded=True) as refinement_status:
                    refinement_status.write("Checking selected phases and available CIFs.")
                    t_validate = time.perf_counter()
                    result, error = _rapid_run_variant_validation(
                        scenario,
                        validation_row,
                        frozen_formula_keys=frozen_formula_keys,
                        status_box=refinement_status,
                    )
                    elapsed = time.perf_counter() - t_validate
                if error:
                    try:
                        refinement_status.update(label="Targeted refinement failed", state="error", expanded=True)
                    except Exception:
                        pass
                    st.error(error)
                elif result and validation_key:
                    st.session_state[validation_key] = result
                    try:
                        cached_read_csv_file.clear()
                        cached_read_json_file.clear()
                    except Exception:
                        pass
                    try:
                        refinement_status.write(f"Completed in {_format_duration(elapsed)}.")
                        refinement_status.update(label="Targeted refinement completed", state="complete", expanded=False)
                    except Exception:
                        pass
                    st.success("Targeted refinement completed.")
                    st.rerun()

        if matched_row is None:
            _render_plot_meta_cards(
                [
                    ("Refinement", "not run"),
                    ("Final rank", "-"),
                    ("Pattern rank", _rapid_display_rank(validation_row.get("rank512")) if validation_row is not None else "-"),
                    ("Coarse rank", _rapid_display_rank(validation_row.get("rank64")) if validation_row is not None else "-"),
                    ("Status", "Pending"),
                    ("Seconds", "-"),
                ],
                numeric_labels={"Final rank", "Pattern rank", "Coarse rank"},
            )
            st.info("No fit plot exists for this edited combination yet. Run targeted refinement to generate the GPX, phase fractions, and interactive fit plot.")
            return
        else:
            _render_plot_meta_cards(
                [
                    ("Refinement", _rapid_fmt_number(matched_row.get("rwp"), 3)),
                    ("Final rank", _rapid_display_rank(matched_row.get("gsas_rwp_rank"))),
                    ("Pattern rank", _rapid_display_rank(matched_row.get("rank512"))),
                    ("Coarse rank", _rapid_display_rank(matched_row.get("rank64"))),
                    ("Status", _rapid_status_label(matched_row.get("status"))),
                    ("Seconds", _rapid_fmt_number(matched_row.get("seconds"), 2)),
                ],
                numeric_labels={"Refinement", "Final rank", "Pattern rank", "Coarse rank", "Seconds"},
            )

        weights_text = matched_row.get("weights_json")
        if weights_text:
            try:
                weights = json.loads(weights_text)
                weight_name_map = _rapid_weight_name_map(matched_row)
                if pd is not None:
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "Phase": weight_name_map.get(str(key), str(key)),
                                    "Phase fraction (%)": value,
                                }
                                for key, value in weights.items()
                            ]
                        ),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "Phase fraction (%)": st.column_config.NumberColumn(
                                "Phase fraction (%)",
                                format="%.2f",
                            )
                        },
                    )
            except Exception:
                st.caption(f"Weights: `{weights_text}`")

        curve_path = _rapid_local_path(matched_row.get("curve_png"), report_root)
        curve_csv_path = _rapid_local_path(matched_row.get("curve_csv"), report_root)
        st.markdown("**Interactive fit inspection**")
        _rapid_render_curve_plot(
            curve_csv_path,
            curve_path,
            title=_rapid_hypothesis_family_label_from_row(matched_row, representative_map),
            key_suffix=f"solution_inspector_{validation_key or selected_ix}",
        )

        with st.expander("Files for reproduction", expanded=False):
            file_cols = st.columns(3)
            for col, label, field in [
                (file_cols[0], "GPX project", "gpx"),
                (file_cols[1], "Curve CSV", "curve_csv"),
                (file_cols[2], "Curve PNG", "curve_png"),
            ]:
                local_path = _rapid_local_path(matched_row.get(field), report_root)
                with col:
                    if local_path and local_path.exists():
                        st.download_button(
                            label,
                            local_path.read_bytes(),
                            file_name=local_path.name,
                            key=f"rapid_download_{scenario}_{field}_{local_path.name}_{selected_ix}",
                        )
                    else:
                        st.caption(f"{label}: not found")

def _render_sidebar_summary_rows(items: list[tuple[str, object]]) -> None:
    rows = []
    for label, value in items:
        rows.append(
            '<div class="radar-sidebar-summary-row">'
            f'<div class="radar-sidebar-summary-label">{html.escape(str(label))}</div>'
            f'<div class="radar-sidebar-summary-value">{html.escape(str(value))}</div>'
            '</div>'
        )
    st.markdown('<div class="radar-sidebar-summary">' + ''.join(rows) + '</div>', unsafe_allow_html=True)


def _render_run_context_banner(
    run_dir: Path | None,
    *,
    mode: str = "",
    status: str = "",
    active: bool = False,
    compact: bool = False,
) -> None:
    if not run_dir:
        return
    title = "Viewing active run" if active else "Viewing saved result"
    if active:
        detail = "Live output is shown here. Setup controls are locked while this run is active."
    elif (
        st.session_state.get("dataset_source_mode") == "Reuse Previous Run"
        and st.session_state.get("reused_run_config_path")
    ):
        detail = (
            "This workspace view is still reading saved outputs. Data Collection has a saved-run template "
            "staged for the next run; setup changes affect only the new run."
        )
    else:
        detail = (
            "This workspace view is reading saved outputs. Prepare an editable copy to start a new run from "
            "these inputs; copied setup controls do not edit this saved result."
        )
    meta_bits = []
    if mode:
        meta_bits.append(html.escape(str(mode)))
    if status:
        meta_bits.append(html.escape(str(status).replace("_", " ").title()))
    meta_html = ""
    if meta_bits:
        meta_html = '<div class="radar-run-context-meta">' + " | ".join(meta_bits) + "</div>"
    class_name = "radar-run-context-banner compact" if compact else "radar-run-context-banner"
    full_run_name = run_dir.name
    visible_run_name = _compact_run_name(full_run_name, max_chars=24 if compact else 44)
    st.markdown(
        (
            f'<div class="{class_name}">'
            '<div class="radar-run-context-title-row">'
            f'<div class="radar-run-context-title">{html.escape(title)}</div>'
            f'<code class="radar-run-name-chip" title="{html.escape(full_run_name, quote=True)}">{html.escape(visible_run_name)}</code>'
            '</div>'
            f'{meta_html}'
            f'<div class="radar-run-context-detail">{html.escape(detail)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )
    if not active and not compact:
        _render_prepare_saved_run_template_action(
            run_dir,
            key=f"prepare_template_banner_{run_dir.name}",
        )


def get_gdrive_id(url):
    import re
    patterns = [
        r'/file/d/([^/]+)',
        r'id=([^&]+)',
        r'/open\?id=([^&]+)'
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None

def download_and_extract_db(url, target_db_dir: Path, is_xray: bool = False):
    """Download and extract a database archive to the given target directory."""
    import requests
    import zipfile
    import gdown
    import shutil

    target_db_dir.mkdir(parents=True, exist_ok=True)
    try:
        with st.status("Downloading database archive...", expanded=True) as status:
            zip_path = Path(PROJECT_ROOT) / "temp_db.zip"

            # --- 1. Download ---
            gdrive_id = get_gdrive_id(url)
            if gdrive_id:
                st.write(f"Detected Google Drive Link (ID: {gdrive_id})")
                gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
            else:
                st.write(f"Fetching from: {url}")
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=81920):
                        if chunk: f.write(chunk)

            # --- 2. Extract ---
            if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
                st.error("Download failed or file is not a valid ZIP archive.")
                zip_path.unlink(missing_ok=True)
                return False

            st.write("Extracting files...")
            temp_extract_dir = Path(PROJECT_ROOT) / "temp_extract"
            if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
            temp_extract_dir.mkdir()

            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(temp_extract_dir)

            repaired_temp = repair_database_layout(temp_extract_dir)
            if repaired_temp:
                st.write("Normalized archive paths from Windows-style separators.")

            # Resolve extracted database root (supports optional wrapper folders).
            extracted_db_path = temp_extract_dir
            if (temp_extract_dir / "database_aug").exists():
                extracted_db_path = temp_extract_dir / "database_aug"
            elif (temp_extract_dir / "database_xray").exists():
                extracted_db_path = temp_extract_dir / "database_xray"
            elif (temp_extract_dir / "database_neutron").exists():
                extracted_db_path = temp_extract_dir / "database_neutron"

            # Copy contents to final destination
            st.write("Organizing folders...")
            for item in extracted_db_path.iterdir():
                dest = target_db_dir / item.name
                if dest.exists():
                    if dest.is_dir(): shutil.rmtree(dest)
                    else: dest.unlink()
                shutil.move(str(item), str(target_db_dir))

            repaired_target = repair_database_layout(target_db_dir)
            if repaired_target:
                st.write("Repaired extracted database layout.")

            # Migration guard: for older X-ray archives that may miss metadata,
            # copy from local neutron DB only if available.
            if is_xray and not (target_db_dir / "highsymm_metadata.json").exists():
                migration_sources = [
                    DB_NEUTRON_METADATA_JSON,
                    Path(PROJECT_ROOT) / "data_old" / "database_neutron" / "highsymm_metadata.json",
                ]
                for src in migration_sources:
                    if src.exists():
                        shutil.copy2(src, target_db_dir / "highsymm_metadata.json")
                        st.write(f"Added missing `highsymm_metadata.json` to X-ray DB from fallback source: {src}")
                        break

            # Cleanup
            zip_path.unlink()
            shutil.rmtree(temp_extract_dir)

            # 3. Validation
            if check_db_integrity(target_db_dir, is_xray=is_xray):
                cached_check_db_integrity.clear()
                load_cached_db_loader.clear()
                status.update(label="Database installed successfully.", state="complete")
                return True
            else:
                if is_xray and not (target_db_dir / "highsymm_metadata.json").exists():
                    st.error("X-ray DB is missing `highsymm_metadata.json`. Rebuild/upload the X-ray ZIP with this file included.")
                status.update(label="Missing files after extraction.", state="error")
                return False

    except Exception as e:
        st.error(f"Database install failed: {e}")
        return False

# --- GSAS-II CHECK & IMPORT ---
try:
    import GSASII.GSASIIscriptable as G2sc
    GSAS_AVAILABLE = True
except ImportError:
    GSAS_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="RADAR-PD",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PREMIUM CSS: ORNL-Inspired Calm Theme ---
st.markdown("""
<style>
    /* ============ Global Theme: Calm Light Mode ============ */
    :root {
        --ornl-green: #154734;
        --ornl-green-light: #1e6b4a;
        --accent-green: #4caf50;
        --bg-primary: #f8f9fa;
        --bg-secondary: #ffffff;
        --bg-tertiary: #e9ecef;
        --sidebar-bg: #f4f7f5;
        --sidebar-panel-bg: #ffffff;
        --sidebar-panel-header: #e7f0eb;
        --sidebar-panel-border: #bdcec4;
        --sidebar-panel-shadow: rgba(21, 71, 52, 0.08);
        --text-primary: #212529;
        --text-secondary: #495057;
        --border-color: #dee2e6;
    }

    /* ============ Stability & Layout ============ */
    /* Force vertical scrollbar to prevent horizontal jitter when content expands */
    html {
        overflow-y: scroll;
    }

    /* Stop the "fading" effect during reruns by minimizing transition noise */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        transition: none !important;
    }

    /* Keep the deployed app in one fixed light experience; hide Streamlit's
       global menu where users can switch Auto/Light/Dark themes. */
    #MainMenu,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    footer {
        display: none !important;
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)
# --- PREMIUM CSS: Part 2 ---
st.markdown("""
<style>
    /* ============ Elegant Buttons ============ */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(135deg, var(--ornl-green) 0%, var(--ornl-green-light) 100%);
        color: #ffffff !important;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(21, 71, 52, 0.2);
    }
    .stButton>button *,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stFormSubmitButton"] button * {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(21, 71, 52, 0.3);
        background: linear-gradient(135deg, var(--ornl-green-light) 0%, var(--accent-green) 100%);
    }
    [data-testid="stFormSubmitButton"] button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(135deg, var(--ornl-green) 0%, var(--ornl-green-light) 100%) !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(21, 71, 52, 0.18);
    }

    /* ============ Metrics Cards ============ */
    div[data-testid="stMetric"] {
        background-color: var(--bg-secondary);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid var(--ornl-green);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary);
    }

    /* ============ Tabs ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: var(--bg-secondary);
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        padding: 0 20px;
        border: 1px solid var(--border-color);
        border-bottom: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--bg-tertiary);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--ornl-green);
        color: white;
        border-color: var(--ornl-green);
    }

    /* ============ Log Viewer ============ */
    .log-viewer {
        height: 350px;
        overflow-y: scroll !important;
        scroll-behavior: smooth;
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 15px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        border-radius: 8px;
        border: 1px solid #334155;
        font-size: 0.85em;
        line-height: 1.6;
        white-space: pre-wrap;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .log-header { color: #a78bfa; font-weight: bold; }
    .log-metric { color: #34d399; }


    /* ============ Interactive Plot Metadata ============ */
    .radar-plot-kpis {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 0.65rem;
        margin: 0.45rem 0 1rem;
    }
    .radar-plot-kpi {
        background: #ffffff;
        border: 1px solid #d5e2db;
        border-left: 4px solid var(--ornl-green);
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(21, 71, 52, 0.06);
        padding: 0.58rem 0.72rem;
        min-width: 0;
        min-height: 3.25rem;
    }
    .radar-plot-kpi-label {
        color: #5f7068;
        font-size: 0.72rem;
        font-weight: 650;
        line-height: 1.15;
        margin-bottom: 0.28rem;
    }
    .radar-plot-kpi-value {
        color: #243a31;
        font-size: 0.98rem;
        font-weight: 650;
        line-height: 1.22;
        overflow-wrap: anywhere;
        word-break: normal;
    }
    .radar-plot-kpi-value.long {
        display: block;
        max-width: 100%;
        font-size: 0.84rem;
        font-weight: 600;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        overflow-wrap: normal;
    }
    .radar-plot-kpi-value.numeric {
        font-size: 1.1rem;
        font-variant-numeric: tabular-nums;
    }
    @media (max-width: 700px) {
        .radar-plot-kpis {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 420px) {
        .radar-plot-kpis {
            grid-template-columns: 1fr;
        }
    }
    .radar-plot-kpis-compact {
        grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
        gap: 0.45rem;
    }
    .radar-plot-kpis-compact .radar-plot-kpi {
        min-height: 2.65rem;
        padding: 0.46rem 0.58rem;
    }
    .radar-plot-kpis-compact .radar-plot-kpi-label {
        font-size: 0.66rem;
        margin-bottom: 0.18rem;
    }
    .radar-plot-kpis-compact .radar-plot-kpi-value.numeric {
        font-size: 0.96rem;
    }
    @media (max-width: 420px) {
        .radar-plot-kpis-compact {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 330px) {
        .radar-plot-kpis-compact {
            grid-template-columns: 1fr;
        }
    }

    /* ============ Sidebar Summary Rows ============ */
    .radar-sidebar-summary {
        display: grid;
        gap: 0.42rem;
        margin: 0.65rem 0 0.45rem;
    }
    .radar-sidebar-summary-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 0.65rem;
        min-height: 2.35rem;
        padding: 0.48rem 0.62rem;
        background: #ffffff;
        border: 1px solid #d5e2db;
        border-left: 4px solid var(--ornl-green);
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(21, 71, 52, 0.06);
    }
    .radar-sidebar-summary-label {
        color: #465d53;
        font-size: 0.76rem;
        font-weight: 650;
        line-height: 1.22;
        overflow-wrap: anywhere;
    }
    .radar-sidebar-summary-value {
        color: #243a31;
        font-size: 1.02rem;
        font-weight: 750;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }
    /* ============ Main Workspace Switcher ============ */
    .radar-workspace-toolbar-label {
        margin: 1.05rem 0 0.35rem;
        color: #3f5a4f;
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.25;
    }
    /* Native segmented workspace control */
    div[role="radiogroup"][aria-label="button group"]:has(button[data-testid^="stBaseButton-segmented_control"]) {
        display: grid !important;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.45rem;
        width: 100%;
        max-width: 720px;
        margin: 0 0 1.25rem;
        padding: 0.38rem;
        background: #eef5f1;
        border: 1px solid #c8d9d0;
        border-radius: 8px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.65);
    }
    div[role="radiogroup"][aria-label="button group"]:has(button[data-testid^="stBaseButton-segmented_control"]) button[data-testid^="stBaseButton-segmented_control"] {
        width: 100%;
        min-height: 2.25rem;
        justify-content: center;
        border-radius: 7px;
        margin: 0 !important;
    }
    div[role="radiogroup"][aria-label="button group"]:has(button[data-testid^="stBaseButton-segmented_control"]) button[data-testid^="stBaseButton-segmented_control"] p {
        font-size: 0.86rem;
        font-weight: 650;
        line-height: 1.2;
        text-align: center;
        white-space: nowrap;
    }
    div[role="radiogroup"][aria-label="button group"]:has(button[data-testid^="stBaseButton-segmented_control"]) button[kind="segmented_controlActive"] {
        background: var(--ornl-green) !important;
        border-color: var(--ornl-green) !important;
        color: #ffffff !important;
    }
    div[role="radiogroup"][aria-label="button group"]:has(button[data-testid^="stBaseButton-segmented_control"]) button[kind="segmented_controlActive"] p {
        color: #ffffff !important;
    }
    /* Visible radio workspace cards */
    div[role="radiogroup"][aria-label="Workspace view"] {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 0.45rem;
        width: 100%;
        max-width: 720px;
        margin: 0 0 1.25rem;
        padding: 0.38rem;
        background: #eef5f1;
        border: 1px solid #c8d9d0;
        border-radius: 8px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.65);
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"] {
        min-width: 0;
        min-height: 2.25rem;
        margin: 0 !important;
        padding: 0.42rem 0.62rem !important;
        border: 1px solid #c8d9d0;
        border-radius: 7px;
        background: #ffffff;
        box-sizing: border-box;
        cursor: pointer;
        transition: background-color 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"]:hover {
        border-color: var(--ornl-green);
        box-shadow: 0 1px 5px rgba(21, 71, 52, 0.12);
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"] p {
        color: #2d4439;
        font-size: 0.86rem;
        font-weight: 650;
        line-height: 1.2;
        white-space: nowrap;
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"]:has(input:checked) {
        background: #f2faf6;
        border-color: var(--ornl-green);
        box-shadow: 0 2px 8px rgba(21, 71, 52, 0.16);
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"]:has(input:checked) p {
        color: var(--ornl-green) !important;
    }
    div[role="radiogroup"][aria-label="Run workspace"] {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(135px, 1fr));
        gap: 0.45rem;
        width: 100%;
        max-width: 720px;
        margin: 0 0 1.25rem;
        padding: 0.38rem;
        background: #eef5f1;
        border: 1px solid #c8d9d0;
        border-radius: 8px;
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.65);
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] {
        display: flex !important;
        position: relative;
        align-items: center;
        justify-content: center;
        min-width: 0;
        min-height: 2.25rem;
        margin: 0 !important;
        padding: 0.42rem 0.72rem !important;
        border: 1px solid #c8d9d0;
        border-radius: 7px;
        background: #ffffff;
        box-sizing: border-box;
        cursor: pointer;
        transition: background-color 0.16s ease, border-color 0.16s ease, box-shadow 0.16s ease;
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:hover {
        border-color: var(--ornl-green);
        box-shadow: 0 1px 5px rgba(21, 71, 52, 0.12);
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] > div:first-child {
        position: absolute !important;
        inset: 0;
        z-index: 2;
        width: 100% !important;
        height: 100% !important;
        margin: 0 !important;
        opacity: 0;
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] > div:last-child {
        position: relative;
        z-index: 1;
        pointer-events: none;
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] p {
        color: #2d4439;
        font-size: 0.86rem;
        font-weight: 650;
        line-height: 1.2;
        white-space: nowrap;
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has(input:checked),
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has([aria-checked="true"]) {
        background: var(--ornl-green);
        border-color: var(--ornl-green);
        box-shadow: 0 2px 8px rgba(21, 71, 52, 0.18);
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has(input:checked) p,
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has([aria-checked="true"]) p {
        color: #ffffff !important;
    }
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] p {
        text-align: center;
    }

    /* ============ File Explorer ============ */
    .file-tree-item {
        padding: 6px 10px;
        border-radius: 6px;
        margin-bottom: 3px;
        transition: background-color 0.2s;
        border-bottom: 1px solid var(--border-color);
        background-color: var(--bg-secondary);
    }
    .file-tree-item:hover {
        background-color: var(--bg-tertiary);
    }
    .file-tree-folder { color: var(--ornl-green); font-weight: bold; }
    .file-tree-file { color: var(--text-secondary); }

    /* ============ Sidebar ============ */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--sidebar-panel-border);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--ornl-green);
    }
    .radar-sidebar-kpis {
        display: grid;
        grid-template-columns: 1fr;
        gap: 0.55rem;
        margin: 0.65rem 0 0.8rem;
    }
    .radar-sidebar-kpi {
        background: #ffffff;
        border: 1px solid #d5e2db;
        border-left: 4px solid var(--ornl-green);
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(21, 71, 52, 0.07);
        padding: 0.65rem 0.75rem;
        min-width: 0;
    }
    .radar-sidebar-kpi-label {
        color: #5f7068;
        font-size: 0.78rem;
        font-weight: 600;
        line-height: 1.2;
        margin-bottom: 0.25rem;
    }
    .radar-sidebar-kpi-value {
        color: #263d33;
        font-size: 0.98rem;
        font-weight: 650;
        line-height: 1.25;
        overflow-wrap: anywhere;
        word-break: normal;
    }
    .radar-sidebar-kpi-value.numeric {
        font-size: 1.05rem;
        font-variant-numeric: tabular-nums;
    }
    .radar-sidebar-run-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.28rem;
        margin: 0.42rem 0 0.55rem;
    }
    .radar-sidebar-run-meta span {
        display: inline-flex;
        align-items: center;
        gap: 0.24rem;
        min-width: 0;
        max-width: 100%;
        padding: 0.24rem 0.4rem;
        background: #ffffff;
        border: 1px solid var(--radar-line);
        border-left: 2px solid var(--radar-brand-700);
        border-radius: 999px;
        color: var(--radar-ink-700);
        font-size: 0.66rem;
        line-height: 1.1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    [data-testid="stSidebar"] .radar-sidebar-run-meta {
        gap: 0.24rem;
    }
    [data-testid="stSidebar"] .radar-sidebar-run-meta span {
        padding: 0.22rem 0.36rem;
        font-size: 0.64rem;
        line-height: 1.08;
    }
    .radar-sidebar-run-meta .status-failed {
        border-left-color: #b42318;
        background: #fff6f4;
        color: #7a271a;
    }
    .radar-sidebar-run-meta .status-running,
    .radar-sidebar-run-meta .status-processing,
    .radar-sidebar-run-meta .status-starting {
        border-left-color: #b7791f;
        background: #fff8e6;
        color: #7a4b00;
    }
    .radar-sidebar-run-meta .status-complete,
    .radar-sidebar-run-meta .status-completed {
        border-left-color: var(--radar-brand-700);
    }
    .radar-sidebar-run-meta strong {
        color: var(--radar-ink-500);
        font-size: 0.56rem;
        font-weight: 750;
        letter-spacing: 0;
        text-transform: uppercase;
        white-space: nowrap;
        flex: 0 0 auto;
    }
    .radar-run-context-banner {
        margin: 0.65rem 0 0.95rem;
        padding: 0.72rem 0.9rem;
        border: 1px solid var(--radar-line-strong);
        border-left: 4px solid var(--radar-brand-700);
        border-radius: 8px;
        background: #f4fbf7;
        box-shadow: 0 2px 8px rgba(15, 52, 40, 0.05);
    }
    .radar-run-context-banner.compact {
        margin: 0.45rem 0 0.65rem;
        padding: 0.58rem 0.64rem;
    }
    .radar-run-context-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.65rem;
        min-width: 0;
    }
    .radar-run-context-title {
        color: var(--radar-brand-900);
        font-size: 0.86rem;
        font-weight: 760;
        line-height: 1.25;
    }
    code.radar-run-name-chip {
        display: inline-block !important;
        max-width: 100%;
        min-width: 0;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        vertical-align: bottom;
    }
    .radar-run-context-title-row code,
    .radar-run-context-title-row code.radar-run-name-chip {
        display: inline-block;
        max-width: 58%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        vertical-align: bottom;
    }
    [data-testid="stSidebar"] .radar-run-context-title-row {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.18rem;
    }
    [data-testid="stSidebar"] .radar-run-context-title-row code,
    [data-testid="stSidebar"] .radar-run-context-title-row code.radar-run-name-chip {
        max-width: 100%;
    }
    .radar-sidebar-path-caption {
        color: var(--radar-ink-500);
        font-size: 0.74rem;
        line-height: 1.25;
        margin: 0.35rem 0 0.45rem;
        min-width: 0;
    }
    .radar-sidebar-path-caption code {
        display: inline-block;
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        vertical-align: bottom;
    }
    .radar-run-context-meta {
        margin-top: 0.2rem;
        color: var(--radar-ink-700);
        font-size: 0.74rem;
        font-weight: 650;
        line-height: 1.25;
    }
    .radar-run-context-detail {
        margin-top: 0.28rem;
        color: var(--radar-ink-500);
        font-size: 0.75rem;
        line-height: 1.35;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        background: var(--sidebar-panel-header);
        border: 1px solid var(--sidebar-panel-border);
        border-left: 4px solid var(--ornl-green);
        border-radius: 8px;
        box-shadow: 0 2px 8px var(--sidebar-panel-shadow);
        padding: 0.6rem 0.75rem;
        margin: 0.9rem 0 0.75rem;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
        color: #30443a;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
        max-width: 100%;
        overflow-wrap: anywhere;
        word-break: normal;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] code {
        white-space: normal;
        overflow-wrap: anywhere;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span {
        white-space: normal !important;
        overflow-wrap: anywhere;
        line-height: 1.25;
    }
    [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--sidebar-panel-bg);
        border: 1px solid var(--sidebar-panel-border);
        border-left: 4px solid var(--ornl-green);
        border-radius: 8px;
        box-shadow: 0 2px 8px var(--sidebar-panel-shadow);
        padding: 0.85rem 0.9rem 0.95rem;
        margin-bottom: 0.85rem;
    }
    [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] h3 {
        margin-top: 0;
        margin-bottom: 0.55rem;
        color: var(--ornl-green);
        font-size: 1rem;
    }

    /* ============ Expanders ============ */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid var(--sidebar-panel-border);
        border-left: 4px solid var(--ornl-green);
        border-radius: 8px;
        background: var(--sidebar-panel-bg);
        box-shadow: 0 2px 8px var(--sidebar-panel-shadow);
        overflow: hidden;
        margin-bottom: 0.85rem;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background: var(--sidebar-panel-header);
        color: var(--ornl-green);
        font-weight: 700;
        min-height: 2.55rem;
        border-bottom: 1px solid #d5e2db;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background: #dceae2;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background: var(--sidebar-panel-bg);
        padding-top: 0.75rem;
    }
    [data-testid="stSidebar"] div[data-baseweb="input"],
    [data-testid="stSidebar"] div[data-baseweb="select"] > div,
    [data-testid="stSidebar"] textarea {
        background: #ffffff !important;
        border: 1px solid #8eaa9a !important;
        border-radius: 7px !important;
        min-height: 2.35rem;
        box-shadow: inset 0 1px 2px rgba(21, 71, 52, 0.04);
    }
    [data-testid="stSidebar"] div[data-baseweb="input"]:focus-within,
    [data-testid="stSidebar"] textarea:focus {
        border-color: var(--ornl-green) !important;
        box-shadow: 0 0 0 2px rgba(21, 71, 52, 0.14);
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        color: #17231e !important;
        opacity: 1 !important;
        caret-color: var(--ornl-green);
    }
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: #6d8176 !important;
        opacity: 1 !important;
    }
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    .streamlit-expanderContent {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }

    /* ============ Progress Bar ============ */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--ornl-green) 0%, var(--accent-green) 100%);
    }

    /* ============ Decision Engine (Knee Analysis) ============ */
    .decision-engine {
        height: 300px;
        overflow-y: auto;
        background-color: #0f172a;
        color: #cbd5e1;
        padding: 15px;
        font-family: 'Inter', sans-serif;
        border-radius: 12px;
        border: 1px solid #1e293b;
        border-left: 6px solid #f59e0b;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-top: 15px;
    }
    .decision-item {
        margin-bottom: 12px;
        padding: 10px;
        background-color: #1e293b;
        border-radius: 6px;
        border-left: 2px solid #f59e0b;
        animation: fadeIn 0.5s ease-out;
    }
    .decision-tag {
        color: #f59e0b;
        font-weight: 800;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        display: block;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ============ Success/Warning/Error Boxes ============ */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px;
    }

    /* ============ Premium Timeline UI ============ */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(21, 71, 52, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(21, 71, 52, 0); }
        100% { box-shadow: 0 0 0 0 rgba(21, 71, 52, 0); }
    }

    .timeline-container {
        padding: 10px 5px;
        font-family: 'Inter', sans-serif;
    }
    .timeline-item {
        position: relative;
        padding-left: 30px;
        padding-bottom: 20px;
        border-left: 2px solid var(--border-color);
        margin-left: 10px;
    }
    .timeline-item.last {
        border-left: none;
    }
    .timeline-item.active {
        border-left-color: var(--ornl-green);
    }
    .timeline-item.complete {
        border-left-color: var(--accent-green);
    }
    .timeline-item.failed {
        border-left-color: #dc2626;
    }

    .timeline-dot {
        position: absolute;
        left: -8px;
        top: 0;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: var(--bg-tertiary);
        border: 2px solid var(--border-color);
        z-index: 1;
    }
    .timeline-item.active .timeline-dot {
        background: var(--ornl-green);
        border-color: var(--ornl-green);
        animation: pulse 2s infinite;
    }
    .timeline-item.complete .timeline-dot {
        background: var(--accent-green);
        border-color: var(--accent-green);
    }
    .timeline-item.failed .timeline-dot {
        background: #dc2626;
        border-color: #dc2626;
        box-shadow: 0 0 0 4px rgba(220, 38, 38, 0.12);
    }

    .timeline-content {
        top: -4px;
        position: relative;
    }
    .timeline-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .timeline-subtitle {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    .timeline-item.active .timeline-title {
        color: var(--ornl-green);
    }
    .timeline-item.complete .timeline-title {
        color: var(--text-secondary);
        text-decoration: line-through;
        opacity: 0.8;
    }
    .timeline-item.failed .timeline-title {
        color: #b42318;
    }
    .timeline-failure {
        margin-top: 0.25rem;
        color: #b42318;
        font-weight: 600;
    }

    /* Sub-steps (Pass stages) */
    .sub-steps {
        margin-top: 10px;
        padding-left: 5px;
        border-left: 1px dashed var(--border-color);
        margin-left: 5px;
    }
    .sub-step {
        padding: 4px 15px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        position: relative;
    }
    .sub-step.active {
        color: var(--ornl-green-light);
        font-weight: 600;
    }
    .sub-step.active::before {
        content: ">";
        position: absolute;
        left: 0;
        animation: bounceX 1s infinite alternate;
    }

    @keyframes bounceX {
        from { transform: translateX(0); }
        to { transform: translateX(3px); }
    }
</style>
""", unsafe_allow_html=True)

# --- PRODUCT CSS: Commercial Research SaaS polish ---
st.markdown("""
<style>
    :root {
        --radar-brand-900: #0d3428;
        --radar-brand-800: #124331;
        --radar-brand-700: #15543c;
        --radar-brand-600: #1f6b4b;
        --radar-accent-blue: #2563eb;
        --radar-accent-cyan: #0891b2;
        --radar-ink-900: #18231f;
        --radar-ink-700: #344740;
        --radar-ink-500: #64746e;
        --radar-surface: #ffffff;
        --radar-surface-muted: #f6faf8;
        --radar-surface-cool: #f2f7fa;
        --radar-line: #d9e5df;
        --radar-line-strong: #b8cec2;
        --radar-shadow-sm: 0 1px 2px rgba(15, 52, 40, 0.06), 0 1px 8px rgba(15, 52, 40, 0.05);
        --radar-shadow-md: 0 8px 24px rgba(15, 52, 40, 0.10);
        --radar-focus: 0 0 0 3px rgba(37, 99, 235, 0.16);
        --radar-radius: 8px;
    }

    html, body, .stApp {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--radar-ink-900);
        letter-spacing: 0;
    }

    .stApp {
        background:
            linear-gradient(180deg, #f5faf8 0, #f7f9fb 360px, #f8faf9 100%);
    }

    [data-testid="stAppViewContainer"] > .main,
    section[data-testid="stSidebar"] {
        background: transparent;
    }

    .block-container {
        max-width: 1220px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    .radar-hero {
        display: grid;
        gap: 0.75rem;
        margin: 0 0 1.25rem;
        padding-bottom: 0.85rem;
        border-bottom: 1px solid var(--radar-line);
    }
    .radar-hero-kicker {
        display: inline-flex;
        width: fit-content;
        align-items: center;
        gap: 0.45rem;
        padding: 0.28rem 0.55rem;
        border: 1px solid #c7dcd2;
        border-radius: 999px;
        background: #edf7f2;
        color: var(--radar-brand-800);
        font-size: 0.72rem;
        font-weight: 760;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .radar-hero-title {
        max-width: 900px;
        color: var(--radar-ink-900);
        font-size: clamp(2.05rem, 3vw, 3.15rem);
        line-height: 1.02;
        font-weight: 780;
        letter-spacing: 0;
        margin: 0;
    }
    .radar-hero-copy {
        max-width: 790px;
        color: var(--radar-ink-700);
        font-size: 1.02rem;
        line-height: 1.58;
        margin: 0;
    }
    .radar-hero-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.1rem;
    }
    .radar-hero-chip {
        display: inline-flex;
        align-items: center;
        min-height: 1.78rem;
        padding: 0.25rem 0.62rem;
        border: 1px solid var(--radar-line);
        border-radius: 999px;
        background: rgba(255,255,255,0.82);
        color: #365147;
        font-size: 0.76rem;
        font-weight: 650;
        box-shadow: 0 1px 4px rgba(15, 52, 40, 0.05);
    }

    .radar-sidebar-brand {
        margin: 0.4rem 0 1rem;
        padding: 0.85rem 0.95rem;
        border: 1px solid var(--radar-line);
        border-left: 4px solid var(--radar-brand-700);
        border-radius: var(--radar-radius);
        background: #ffffff;
        box-shadow: var(--radar-shadow-sm);
    }
    .radar-sidebar-brand-title {
        color: var(--radar-brand-900);
        font-size: 1.08rem;
        font-weight: 780;
        line-height: 1.1;
    }
    .radar-sidebar-brand-subtitle {
        margin-top: 0.28rem;
        color: var(--radar-ink-500);
        font-size: 0.78rem;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        background: #eef5f1 !important;
        border-right: 1px solid #c8dbd1 !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid var(--radar-line-strong);
        border-left: 4px solid var(--radar-brand-700);
        border-radius: var(--radar-radius);
        background: var(--radar-surface);
        box-shadow: var(--radar-shadow-sm);
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        background: #e7f1eb;
        color: var(--radar-brand-900);
        font-size: 0.88rem;
        font-weight: 760;
        min-height: 2.7rem;
    }
    [data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
        background: var(--radar-surface);
    }

    h1, h2, h3, h4 {
        color: var(--radar-ink-900);
        letter-spacing: 0;
    }
    h2, h3 {
        font-weight: 760 !important;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        line-height: 1.55;
    }

    div[data-testid="stAlert"] {
        border-radius: var(--radar-radius);
        border: 1px solid var(--radar-line);
        box-shadow: 0 1px 6px rgba(15, 52, 40, 0.04);
    }

    div[data-testid="stMetric"],
    .radar-sidebar-kpi,
    .radar-sidebar-summary-row,
    .radar-plot-kpi {
        border: 1px solid var(--radar-line);
        border-left: 4px solid var(--radar-brand-700);
        border-radius: var(--radar-radius);
        background: var(--radar-surface);
        box-shadow: var(--radar-shadow-sm);
    }
    .rapid-refinement-card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 0.75rem;
        margin: 0.35rem 0 0.85rem;
    }
    .rapid-refinement-card {
        min-width: 0;
        padding: 0.78rem 0.9rem;
        border: 1px solid var(--radar-line);
        border-left: 4px solid var(--radar-brand-700);
        border-radius: var(--radar-radius);
        background: rgba(255, 255, 255, 0.92);
        box-shadow: var(--radar-shadow-sm);
    }
    .rapid-refinement-card-topline,
    .rapid-refinement-card-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem 0.7rem;
        color: var(--radar-ink-500);
        font-size: 0.74rem;
        font-weight: 720;
    }
    .rapid-refinement-card-hypothesis {
        margin-top: 0.42rem;
        color: var(--radar-ink-900);
        font-size: 0.98rem;
        font-weight: 760;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }
    .rapid-refinement-card-fractions {
        display: grid;
        gap: 0.16rem;
        margin: 0.55rem 0;
        color: var(--radar-ink-500);
        font-size: 0.76rem;
        line-height: 1.35;
    }
    .rapid-refinement-card-fractions strong {
        color: var(--radar-ink-700);
        font-size: 0.86rem;
        font-weight: 650;
        overflow-wrap: anywhere;
    }
    div[data-testid="stMetric"] {
        padding: 0.78rem 0.9rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
    div[data-testid="stMetric"] label {
        color: var(--radar-ink-500) !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: var(--radar-ink-900);
        font-weight: 760;
        font-size: clamp(1.25rem, 1.8vw, 1.75rem) !important;
        line-height: 1.08 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
        max-width: 100%;
    }

    .stButton > button,
    [data-testid="stFormSubmitButton"] button {
        min-height: 2.55rem;
        border-radius: var(--radar-radius) !important;
        border: 1px solid var(--radar-brand-700) !important;
        background: var(--radar-brand-700) !important;
        color: #ffffff !important;
        box-shadow: 0 3px 10px rgba(18, 67, 49, 0.16);
        font-weight: 720 !important;
        transition: border-color 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease, transform 0.15s ease;
    }
    .stButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: var(--radar-brand-800) !important;
        border-color: var(--radar-brand-800) !important;
        box-shadow: var(--radar-shadow-md);
        transform: translateY(-1px);
    }
    .stButton > button:focus,
    [data-testid="stFormSubmitButton"] button:focus {
        box-shadow: var(--radar-focus) !important;
    }

    [data-testid="stDownloadButton"] button {
        min-height: 2.45rem;
        min-width: 5.05rem;
        padding: 0.42rem 0.55rem !important;
        border-radius: var(--radar-radius) !important;
        border: 1px solid #b8cfc4 !important;
        background: #ffffff !important;
        color: var(--radar-brand-800) !important;
        box-shadow: 0 1px 4px rgba(15, 52, 40, 0.08);
        font-weight: 700 !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
        transition: border-color 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stDownloadButton"] button *,
    [data-testid="stDownloadButton"] button p,
    [data-testid="stDownloadButton"] button span {
        width: auto !important;
        max-width: none !important;
        white-space: nowrap !important;
        word-break: keep-all !important;
        overflow-wrap: normal !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: #eef7f2 !important;
        border-color: var(--radar-brand-700) !important;
        box-shadow: var(--radar-shadow-sm);
    }
    [data-testid="stDownloadButton"] button:focus {
        box-shadow: var(--radar-focus) !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] textarea,
    textarea {
        border-radius: var(--radar-radius) !important;
        border-color: #b9cfc4 !important;
        background: #ffffff !important;
        box-shadow: inset 0 1px 2px rgba(15, 52, 40, 0.04);
    }
    div[data-baseweb="input"]:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    textarea:focus {
        border-color: var(--radar-accent-blue) !important;
        box-shadow: var(--radar-focus) !important;
    }
    input, textarea {
        color: var(--radar-ink-900) !important;
    }
    input::placeholder, textarea::placeholder {
        color: #7b8d86 !important;
        opacity: 1 !important;
    }

    [data-testid="stFileUploader"] section {
        border: 1px dashed #9fbaad !important;
        border-radius: var(--radar-radius) !important;
        background: #fbfefd !important;
        padding: 1rem !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: var(--radar-accent-blue) !important;
        background: #f5fbff !important;
    }

    div[role="radiogroup"][aria-label="Workspace view"],
    div[role="radiogroup"][aria-label="Run workspace"],
    div[role="radiogroup"][aria-label="Analysis path"],
    div[role="radiogroup"][aria-label="Diffraction source"],
    div[role="radiogroup"][aria-label="Library mode"] {
        border-radius: var(--radar-radius);
    }

    div[role="radiogroup"][aria-label="Workspace view"],
    div[role="radiogroup"][aria-label="Run workspace"] {
        background: #edf5f1;
        border-color: var(--radar-line-strong);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.7);
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"],
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] {
        border-radius: 7px;
        border-color: #c7d9d0;
        box-shadow: 0 1px 4px rgba(15, 52, 40, 0.04);
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"]:has(input:checked),
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has(input:checked),
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has([aria-checked="true"]) {
        background: var(--radar-brand-700) !important;
        border-color: var(--radar-brand-700) !important;
        box-shadow: 0 6px 14px rgba(18, 67, 49, 0.18);
    }
    div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"]:has(input:checked) p,
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has(input:checked) p,
    div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"]:has([aria-checked="true"]) p {
        color: #ffffff !important;
    }

    .log-viewer {
        height: 360px;
        border-radius: var(--radar-radius);
        border: 1px solid #24364b;
        background: #152235;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.22), 0 8px 20px rgba(15, 35, 55, 0.10);
    }
    .log-viewer,
    .log-viewer * {
        max-width: 100% !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] pre,
    [data-testid="stCodeBlock"] code,
    pre,
    pre code,
    div[data-testid="stCode"] pre,
    div[data-testid="stCode"] code {
        display: block !important;
        width: auto !important;
        min-width: 0 !important;
        max-width: 100% !important;
        overflow-x: auto !important;
        white-space: pre-wrap !important;
        overflow-wrap: anywhere !important;
        word-break: break-word !important;
    }

    .timeline-container {
        padding: 0.65rem 0.35rem 0.2rem;
    }
    .timeline-title {
        font-weight: 720;
    }
    .timeline-subtitle {
        color: var(--radar-ink-500);
    }

    .stDataFrame,
    [data-testid="stDataFrame"] {
        border-radius: var(--radar-radius);
        max-width: 100% !important;
        min-width: 0 !important;
        overflow-x: auto !important;
        overflow-y: hidden;
        box-shadow: var(--radar-shadow-sm);
    }
    [data-testid="stDataFrame"] > div,
    [data-testid="stDataFrame"] .dvn-stack {
        max-width: 100% !important;
        min-width: 0 !important;
    }
    .radar-table-wrap {
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
        border: 1px solid var(--radar-line);
        border-radius: var(--radar-radius);
        background: var(--radar-surface);
        box-shadow: var(--radar-shadow-sm);
    }
    .radar-table {
        width: 100%;
        min-width: 680px;
        border-collapse: collapse;
        table-layout: auto;
        font-size: 0.88rem;
    }
    .radar-table-wide-support {
        min-width: 1040px;
    }
    .radar-table-wide-fractions {
        min-width: 960px;
    }
    .radar-table th,
    .radar-table td {
        padding: 0.58rem 0.68rem;
        border-bottom: 1px solid #e4ece8;
        text-align: left;
        vertical-align: top;
        color: var(--radar-ink-700);
    }
    .radar-table th {
        background: #f7faf9;
        color: var(--radar-ink-500);
        font-size: 0.76rem;
        font-weight: 760;
        white-space: nowrap;
    }
    .radar-table tr:last-child td {
        border-bottom: 0;
    }
    .radar-table td.radar-cell-num {
        text-align: right;
        white-space: nowrap;
        font-variant-numeric: tabular-nums;
    }
    .radar-table td.radar-cell-hypothesis,
    .radar-table td.radar-cell-phase-fractions,
    .radar-table td.radar-cell-best-cell,
    .radar-table td.radar-cell-key-peak-support {
        min-width: 300px;
        max-width: 520px;
        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
        line-height: 1.35;
        color: var(--radar-ink-900);
    }
    .radar-table th.radar-cell-key-peak-support,
    .radar-table td.radar-cell-key-peak-support {
        min-width: 340px;
        width: 340px;
        max-width: 460px;
    }
    .radar-table td.radar-cell-phase {
        min-width: 180px;
        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
    }
    @media (max-width: 520px) {
        .radar-table-wrap::before {
            content: "Swipe table sideways";
            display: block;
            padding: 0.42rem 0.58rem 0.2rem;
            color: var(--radar-ink-500);
            font-size: 0.72rem;
            font-weight: 650;
            border-bottom: 1px solid #e4ece8;
            background: #fbfdfc;
        }
        .radar-table {
            min-width: 620px;
            font-size: 0.82rem;
        }
        .radar-table th,
        .radar-table td {
            padding: 0.48rem 0.52rem;
        }
        .radar-table td.radar-cell-hypothesis,
        .radar-table td.radar-cell-phase-fractions,
        .radar-table td.radar-cell-best-cell,
        .radar-table td.radar-cell-key-peak-support {
            min-width: 280px;
        }
    }

    [data-testid="stExpander"] {
        border-radius: var(--radar-radius);
    }
    [data-testid="stTabs"] [role="tablist"] {
        max-width: 100% !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
    }
    [data-testid="stTabs"] [role="tab"] {
        min-width: fit-content !important;
        white-space: nowrap !important;
    }

    code {
        border-radius: 6px;
        background: #edf6f1 !important;
        color: var(--radar-brand-800) !important;
        padding: 0.08rem 0.28rem;
    }

    hr {
        border-color: var(--radar-line) !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 1.5rem;
        }
        .radar-hero-title {
            font-size: 2rem;
        }
    }

    @media (max-width: 700px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 280px !important;
            min-width: 280px !important;
            max-width: 280px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 280px !important;
            min-width: 280px !important;
            max-width: 280px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 0.45rem !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stColumn"] {
            width: 100% !important;
            min-width: 0 !important;
            flex: 1 1 auto !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stButton"],
        section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stButton"] button {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] p,
        section[data-testid="stSidebar"][aria-expanded="true"] label,
        section[data-testid="stSidebar"][aria-expanded="true"] span {
            word-break: normal !important;
            overflow-wrap: break-word !important;
            hyphens: none !important;
        }
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) section[data-testid="stMain"] {
            left: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
        }
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) section[data-testid="stMain"] .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }
        .radar-hero {
            padding: 1.2rem 0.85rem !important;
        }
        .radar-hero-title {
            font-size: 1.55rem !important;
            line-height: 1.12 !important;
        }
        .radar-hero-copy {
            font-size: 0.86rem !important;
            line-height: 1.45 !important;
        }
        .radar-hero-chips {
            gap: 0.35rem !important;
        }
        .radar-hero-chip {
            font-size: 0.68rem !important;
            padding: 0.32rem 0.5rem !important;
        }
        div[role="radiogroup"][aria-label="Workspace view"],
        div[role="radiogroup"][aria-label="Run workspace"] {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            max-width: 100% !important;
        }
        div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"],
        div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] {
            min-width: 0 !important;
            padding-left: 0.48rem !important;
            padding-right: 0.48rem !important;
        }
        div[role="radiogroup"][aria-label="Workspace view"] label[data-baseweb="radio"] p,
        div[role="radiogroup"][aria-label="Run workspace"] label[data-baseweb="radio"] p {
            font-size: 0.82rem !important;
            line-height: 1.16 !important;
        }
        [data-testid="stTabs"] [role="tablist"] {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.35rem !important;
            border-bottom: 0 !important;
        }
        [data-testid="stTabs"] [role="tab"] {
            flex: 1 1 calc(50% - 0.35rem) !important;
            min-height: 2.5rem !important;
            border: 1px solid var(--radar-line-strong) !important;
            border-radius: var(--radar-radius) !important;
            background: #ffffff !important;
            justify-content: center !important;
        }
        [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            background: var(--radar-brand-700) !important;
            color: #ffffff !important;
        }
        [data-testid="stTabs"] [role="tab"] p {
            font-size: 0.82rem !important;
            line-height: 1.16 !important;
            white-space: normal !important;
            text-align: center !important;
        }
    }

    @media (max-width: 520px) {
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
        }
        body:has(section[data-testid="stSidebar"][aria-expanded="true"]) section[data-testid="stMain"] {
            left: 0 !important;
            width: 100vw !important;
            max-width: 100vw !important;
        }
        .radar-hero-title {
            font-size: 1.35rem !important;
        }
    }

    @media (max-width: 420px) {
        div[role="radiogroup"][aria-label="Workspace view"],
        div[role="radiogroup"][aria-label="Run workspace"] {
            grid-template-columns: 1fr !important;
        }
    }
</style>
""", unsafe_allow_html=True)

DB_EXISTS = cached_check_db_integrity(str(DB_NEUTRON_DIR), is_xray=False)

# --- UTILITIES ---
def make_default_run_name() -> str:
    return f"run_{datetime.datetime.now().strftime('%Y%j_%H%M%S')}"


def mark_run_name_manual() -> None:
    st.session_state.run_name_mode = "manual"


def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024) # MB
    return mem


def get_memory_snapshot() -> dict:
    process = psutil.Process(os.getpid())
    proc_mb = process.memory_info().rss / (1024 * 1024)
    vm = psutil.virtual_memory()
    return {
        "app_mb": proc_mb,
        "system_used_gb": vm.used / (1024 ** 3),
        "system_total_gb": vm.total / (1024 ** 3),
        "system_percent": vm.percent,
    }


def current_run_status_text() -> str:
    status_text = (
        st.session_state.get("current_stage_desc")
        or st.session_state.get("status_msg")
        or "Running"
    )
    normalized = str(status_text).strip().lower()
    if normalized in {"ready", "initializing", "initializing..."}:
        state = st.session_state.get("pipeline_state") or {}
        stage_idx = state.get("global_stage_idx", -1)
        rapid_context = state.get("rapid_mode") or _is_rapid_context()
        stages = RAPID_STAGES if rapid_context else GLOBAL_STAGES
        if isinstance(stage_idx, int) and 0 <= stage_idx < len(stages):
            status_text = stages[stage_idx]
            if not rapid_context and stage_idx == 3 and state.get("current_pass"):
                pass_stage = dict(PASS_STAGES).get(state.get("pass_stage")) or "working"
                status_text = f"Pass {state['current_pass']}: {pass_stage}"
        elif normalized == "ready":
            status_text = "Running"
    elif normalized == "ready":
        status_text = "Running"
    return str(status_text)


def stop_active_pipeline() -> None:
    if st.session_state.get("pipeline_process"):
        stop_process_tree(st.session_state.pipeline_process)
    st.session_state.run_active = False
    st.session_state.run_finished = True
    st.session_state.pipeline_process = None
    st.session_state.log_queue = None
    if st.session_state.get("run_name_mode") == "auto":
        st.session_state.pending_run_name_reset = True


def render_active_run_sidebar_monitor() -> None:
    rapid_context = _is_rapid_context()
    st.success("Rapid hypothesis mode is running" if rapid_context else "RADAR-PD is running")
    run_name = st.session_state.get("run_name") or "current run"
    run_dir = st.session_state.get("run_dir")
    progress = int(max(0, min(100, st.session_state.get("progress", 0) or 0)))

    st.caption(f"Run: `{run_name}`")
    if run_dir:
        st.caption(f"Output folder: `{Path(run_dir).name}`")
    st.progress(progress / 100)
    st.caption(f"{progress}% - {current_run_status_text()}")

    summary = st.session_state.get("run_summary") or {}
    summary_bits = [
        summary.get("measurement"),
        summary.get("runtime_profile"),
        summary.get("dataset"),
    ]
    summary_line = " | ".join(str(bit) for bit in summary_bits if bit)
    if summary_line:
        st.caption(summary_line)
    if summary.get("library"):
        st.caption(f"Library: {summary['library']}")
    if run_dir:
        render_run_config_summary(Path(run_dir) / "pipeline_config.yaml", title="Running configuration", expanded=False)

    if rapid_context:
        st.info(
            "Setup controls are locked while this run is active. "
            "Use Run Monitor for live progress and Rapid Results for hypotheses, variants, and final refinement ranking."
        )
    else:
        st.info(
            "Setup controls are locked while this run is active. "
            "Use the run workspace for live logs, results, plots, and files."
        )
    if st.button("Stop RADAR-PD", width="stretch", key="stop_radar_pd_active"):
        stop_active_pipeline()
        st.warning("Pipeline terminated.")
        st.rerun()


def parse_element_list(text: str) -> list[str]:
    tokens = re.split(r"[,\s]+", str(text or "").strip())
    elements: list[str] = []
    for raw in tokens:
        token = raw.strip()
        if not token:
            continue
        normalized = token[0].upper() + token[1:].lower()
        if normalized not in elements:
            elements.append(normalized)
    return elements


def invalid_element_tokens(elements: list[str]) -> list[str]:
    valid = set(PERIODIC_TABLE)
    return [element for element in elements if element not in valid]

def sync_run_to_hf(run_dir: str, run_name: str):
    """Syncs a completed run directory to a Hugging Face Dataset."""
    token = os.environ.get("HF_TOKEN_WRITE")
    # Default dataset name, can be overridden by env var
    repo_id = os.environ.get("HF_DATASET_ID", "Lalityadav07/phase_detection_records")

    if not token:
        st.warning("HF_TOKEN_WRITE not found. Persistence disabled.")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)

        st.info(f"Syncing results to {repo_id}...")
        api.upload_folder(
            folder_path=run_dir,
            path_in_repo=f"runs/{run_name}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        st.success("Results synced to Hugging Face.")
    except Exception as e:
        st.error(f"Failed to sync to Hugging Face: {e}")

def _sanitize_workspace_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return text[:48]


def _workspace_pin_hash(username: str, pin: str) -> str:
    payload = f"radar-pd-workspace:{username}:{pin}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _workspace_manifest_path(root: Path) -> Path:
    return root / "workspace.json"


def _ensure_workspace_dirs(root: Path) -> None:
    for child in ("runs", "user_db_packs", "uploads"):
        (root / child).mkdir(parents=True, exist_ok=True)


def _workspace_context_from_root(root: Path, *, username: str, mode: str) -> dict:
    _ensure_workspace_dirs(root)
    return {
        "username": username,
        "mode": mode,
        "root": str(root),
        "runs_root": str(root / "runs"),
        "user_db_packs_root": str(root / "user_db_packs"),
        "uploads_root": str(root / "uploads"),
    }


def _sync_workspace_query_hint(context: dict) -> None:
    """Keep a non-secret workspace hint in the URL for refresh/restart recovery."""
    try:
        if str(context.get("mode") or "") == "workspace":
            username = _sanitize_workspace_name(str(context.get("username") or ""))
            if username:
                st.query_params["workspace"] = username
        else:
            for key in ("workspace", "user"):
                if key in st.query_params:
                    del st.query_params[key]
    except Exception:
        pass


def _workspace_storage_counts(root: Path) -> dict:
    root = Path(root)

    def _count_dirs(path: Path) -> int:
        try:
            return sum(1 for child in path.iterdir() if child.is_dir())
        except OSError:
            return 0

    return {
        "runs": _count_dirs(root / "runs"),
        "libraries": _count_dirs(root / "user_db_packs"),
    }


def _clear_staged_upload_state() -> None:
    for key in list(st.session_state.keys()):
        if str(key).startswith("staged_upload_"):
            del st.session_state[key]


def _safe_upload_filename(name: str) -> str:
    stem = Path(str(name or "uploaded_file")).name
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return clean or "uploaded_file"


def _staged_upload_state_key(field: str) -> str:
    return f"staged_upload_{field}"


def _get_staged_upload(field: str) -> dict | None:
    key = _staged_upload_state_key(field)
    entry = st.session_state.get(key)
    if not entry:
        return None
    path = Path(str(entry.get("path", "")))
    if not path.exists() or not path.is_file():
        st.session_state.pop(key, None)
        return None
    return entry


def _stage_uploaded_file(field: str, uploaded_file) -> dict | None:
    if uploaded_file is None:
        return _get_staged_upload(field)

    safe_name = _safe_upload_filename(getattr(uploaded_file, "name", field))
    data = uploaded_file.getbuffer()
    digest = hashlib.sha256(data).hexdigest()[:12]
    stage_dir = WORKSPACE_ROOT / "uploads" / "staged" / field
    stage_dir.mkdir(parents=True, exist_ok=True)
    path = stage_dir / f"{digest}_{safe_name}"
    if not path.exists() or path.stat().st_size != len(data):
        path.write_bytes(data)
    entry = {
        "path": str(path.resolve()),
        "name": safe_name,
        "size": int(len(data)),
        "sha256": digest,
        "staged_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state[_staged_upload_state_key(field)] = entry
    return entry


def _render_staged_upload_status(field: str, label: str) -> None:
    entry = _get_staged_upload(field)
    if not entry:
        return
    st.caption(f"Ready: `{entry['name']}` ({_format_bytes(int(entry.get('size') or 0))})")
    if st.button(f"Clear {label}", key=f"clear_staged_upload_{field}", width="stretch"):
        st.session_state.pop(_staged_upload_state_key(field), None)
        st.rerun()


REUSE_SAVED_FILE = "Use saved file"
REUSE_REPLACEMENT_FILE = "Upload replacement"
REUSE_SAVED_MAIN_CIF = "Use saved main CIF"
REUSE_NO_MAIN_CIF = "No main CIF"


def _staged_upload_is_nonempty(entry: dict | None) -> bool:
    if not entry:
        return False
    try:
        return Path(str(entry.get("path") or "")).stat().st_size > 0
    except OSError:
        return False


def _copy_existing_file_to_dir(source: Path, target_dir: Path) -> Path:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(str(source))
    target = target_dir / _safe_upload_filename(source.name)
    shutil.copy2(source, target)
    return target


def _copy_staged_upload_to_dir(entry: dict | None, target_dir: Path) -> Path:
    if not entry:
        raise FileNotFoundError("No staged upload is available.")
    staged_path = Path(str(entry.get("path") or ""))
    if not staged_path.exists() or not staged_path.is_file():
        raise FileNotFoundError(str(staged_path))
    target = target_dir / _safe_upload_filename(str(entry.get("name") or staged_path.name))
    shutil.copy2(staged_path, target)
    return target


def _create_temporary_workspace_context() -> dict:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = WORKSPACES_ROOT / "_temporary" / f"temp_{stamp}_{secrets.token_hex(3)}"
    manifest = {
        "mode": "temporary",
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "storage_policy": "temporary",
    }
    _ensure_workspace_dirs(root)
    _workspace_manifest_path(root).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return _workspace_context_from_root(root, username="Temporary", mode="temporary")


def _open_or_create_workspace(username: str, pin: str) -> tuple[dict | None, str | None, str | None]:
    clean_name = _sanitize_workspace_name(username)
    clean_pin = str(pin or "").strip()
    if not clean_name:
        return None, "Enter a workspace username.", None
    if not re.fullmatch(r"\d{4}", clean_pin):
        return None, "Enter a 4-digit workspace PIN.", None

    root = WORKSPACES_ROOT / clean_name
    manifest_path = _workspace_manifest_path(root)
    pin_hash = _workspace_pin_hash(clean_name, clean_pin)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    action = "opened" if manifest_path.exists() else "created"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        if manifest.get("pin_hash") != pin_hash:
            return None, "Workspace name exists, but the PIN did not match.", None
        manifest["last_seen_at"] = now
    else:
        root.mkdir(parents=True, exist_ok=True)
        manifest = {
            "mode": "workspace",
            "username": clean_name,
            "pin_hash": pin_hash,
            "created_at": now,
            "last_seen_at": now,
            "storage_policy": "persistent",
        }
    _ensure_workspace_dirs(root)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return _workspace_context_from_root(root, username=clean_name, mode="workspace"), None, action


def _activate_workspace_context(context: dict) -> None:
    st.session_state.workspace_context = context
    _sync_workspace_query_hint(context)
    _clear_staged_upload_state()
    st.session_state.db_selection_mode = "Original"
    st.session_state.selected_custom_pack_xray = None
    st.session_state.selected_custom_pack_neutron = None
    st.session_state.reused_run_config_path = None
    st.session_state.run_dir = None
    st.session_state.last_finished_run_dir = None
    st.session_state.selected_run_dir = None
    st.session_state.selected_run_notice = None
    st.session_state.selected_run_notice_dir = None
    st.session_state.suppress_latest_run_autoload = False
    try:
        discover_user_db_packs.clear()
        cached_reusable_run_configs.clear()
        cached_recent_interactive_runs.clear()
        cached_workspace_run_entries.clear()
        load_cached_db_loader.clear()
    except Exception:
        pass


def _run_analysis_mode_from_dir(run_dir: Path | None) -> str:
    if run_dir is None:
        return ""
    cfg_path = run_dir / "pipeline_config.yaml"
    if cfg_path.exists():
        try:
            cfg = cached_read_yaml_file(str(cfg_path), _path_mtime(cfg_path))
            rapid = cfg.get("rapid_hypothesis") or {}
            if cfg.get("analysis_mode") == "rapid_hypothesis" or rapid.get("enabled"):
                return "Rapid Hypothesis Mode"
            return "Full RADAR-PD"
        except Exception:
            pass
    if (run_dir / "rapid_results").exists():
        return "Rapid Hypothesis Mode"
    return ""


def _latest_running_workspace_run_dir() -> Path | None:
    try:
        entries = cached_workspace_run_entries(str(ACTIVE_RUNS_ROOT), limit=25)
    except Exception:
        return None
    for entry in entries:
        run_dir = Path(str(entry.get("path") or ""))
        if not run_dir.exists():
            continue
        manifest_status = str((_read_run_manifest(run_dir) or {}).get("status") or "").strip().lower()
        if manifest_status in {"running", "starting", "processing"}:
            return run_dir
    return None


def _is_staging_reused_run_inputs() -> bool:
    """True when the left setup panel is building a new run from old inputs."""
    return bool(
        not st.session_state.get("run_active")
        and st.session_state.get("dataset_source_mode") == "Reuse Previous Run"
        and st.session_state.get("reused_run_config_path")
    )


def _selected_or_active_run_dir() -> Path | None:
    if st.session_state.get("run_active"):
        value = st.session_state.get("run_dir")
        if value:
            try:
                path = Path(value)
                if path.exists():
                    return path
            except Exception:
                pass
    running_from_disk = _latest_running_workspace_run_dir()
    if running_from_disk is not None:
        return running_from_disk
    if _is_staging_reused_run_inputs():
        # A saved run can be used as an editable input template without also
        # being the result currently inspected in the main workspace. Keeping
        # these separate avoids stale Full/Rapid result panels during setup.
        return None
    for key in ("selected_run_dir", "run_dir", "last_finished_run_dir"):
        value = st.session_state.get(key)
        if not value:
            continue
        try:
            path = Path(value)
        except Exception:
            continue
        if path.exists():
            return path
    if st.session_state.get("suppress_latest_run_autoload"):
        return None
    try:
        entries = cached_workspace_run_entries(str(ACTIVE_RUNS_ROOT), limit=1)
        if entries:
            path = Path(entries[0]["path"])
            if path.exists():
                return path
    except Exception:
        pass
    return None


def _clear_selected_run_context(*, suppress_latest_autoload: bool = False) -> None:
    st.session_state.selected_run_dir = None
    st.session_state.selected_run_notice = None
    st.session_state.selected_run_notice_dir = None
    st.session_state.show_full_log_history = False
    st.session_state.log_lines = []
    st.session_state.suppress_latest_run_autoload = bool(suppress_latest_autoload)


def _load_selected_run(run_dir: Path) -> None:
    st.session_state.query_run_error = None
    st.session_state.suppress_latest_run_autoload = False
    st.session_state.selected_run_dir = str(run_dir)
    st.session_state.selected_run_notice = f"Loaded run `{run_dir.name}`."
    st.session_state.selected_run_notice_dir = str(run_dir)
    st.session_state.run_name = run_dir.name
    st.session_state.show_full_log_history = False
    st.session_state.log_lines = []
    log_path = run_dir / "pipeline.log"
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)
            st.session_state.log_lines = lines[-FINISHED_LOG_DISPLAY_LIMIT:]
        except Exception:
            st.session_state.log_lines = []
    mode = _run_analysis_mode_from_dir(run_dir)
    if mode:
        st.session_state.last_finished_analysis_mode = mode
        if not st.session_state.get("run_active"):
            st.session_state.analysis_mode = mode
            st.session_state._workspace_analysis_mode = mode
    manifest_status = str((_read_run_manifest(run_dir) or {}).get("status") or "").strip().lower()
    if manifest_status == "failed" or _run_failure_info(run_dir):
        st.session_state.active_run_view = "Run Monitor"
    else:
        st.session_state.active_run_view = "Rapid Results" if mode == "Rapid Hypothesis Mode" else "Results"


def _query_param_value(name: str) -> str:
    try:
        value = st.query_params.get(name)
    except Exception:
        return ""
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _saved_run_from_token(token: str) -> Path | None:
    token = str(token or "").strip()
    if not token:
        return None
    candidate_name = Path(token).name
    direct = ACTIVE_RUNS_ROOT / candidate_name
    if direct.exists() and direct.is_dir():
        return direct
    try:
        for entry in cached_workspace_run_entries(str(ACTIVE_RUNS_ROOT)):
            entry_path = Path(str(entry.get("path") or ""))
            if entry_path.name == candidate_name or str(entry.get("name") or "") == token:
                if entry_path.exists() and entry_path.is_dir():
                    return entry_path
    except Exception:
        return None
    return None


def _apply_saved_run_query_param() -> None:
    token = _query_param_value("run") or _query_param_value("saved_run")
    if not token:
        st.session_state.query_run_error = None
        return
    run_dir = _saved_run_from_token(token)
    if run_dir is None:
        st.session_state.query_run_error = (
            f"No saved run named `{Path(token).name}` exists in this workspace. "
            "Open the workspace that owns the run, then this URL will load it automatically."
        )
        return
    st.session_state.query_run_error = None
    currently_selected = st.session_state.get("selected_run_dir")
    if currently_selected and Path(str(currently_selected)).resolve() == run_dir.resolve():
        return
    _load_selected_run(run_dir)


def _reusable_run_entry_for_dir(run_dir: Path | None) -> dict | None:
    if run_dir is None:
        return None
    try:
        target = Path(run_dir).resolve()
    except Exception:
        return None
    try:
        entries = cached_reusable_run_configs(str(ACTIVE_RUNS_ROOT))
    except Exception:
        return None
    for entry in entries:
        try:
            if Path(str(entry.get("run_dir") or "")).resolve() == target:
                return entry
        except Exception:
            continue
    return None


def _prepare_saved_run_as_template(run_dir: Path) -> tuple[bool, str]:
    entry = _reusable_run_entry_for_dir(run_dir)
    if not entry:
        return (
            False,
            "This saved run does not have reusable diffraction data and instrument files.",
        )
    cfg_path = Path(str(entry.get("config_path") or ""))
    if not cfg_path.exists():
        return False, "This saved run's configuration file is missing."
    try:
        cfg = cached_read_yaml_file(str(cfg_path), _path_mtime(cfg_path))
    except Exception as exc:
        return False, f"Could not read this saved run's configuration: {exc}"

    st.session_state.reused_run_filter = Path(run_dir).name
    st.session_state.reused_run_selector_path = str(cfg_path)
    st.session_state.reused_run_config_path = str(cfg_path)
    for field in ("reuse_diffraction_data", "reuse_instrument_profile", "reuse_main_cif"):
        st.session_state.pop(_staged_upload_state_key(field), None)
    st.session_state.reuse_data_source_mode = REUSE_SAVED_FILE
    st.session_state.reuse_instprm_source_mode = REUSE_SAVED_FILE
    st.session_state.reuse_main_cif_source_mode = (
        REUSE_SAVED_MAIN_CIF if entry.get("main_cif") else REUSE_NO_MAIN_CIF
    )
    _apply_reused_config_to_session(cfg)
    st.session_state.dataset_source_mode = "Reuse Previous Run"
    st.session_state.force_reuse_previous_run_inputs = True
    compact_run_name = _compact_run_name(Path(run_dir).name, max_chars=24)
    st.session_state.prepare_template_notice = (
        f"`{compact_run_name}` is selected for the new run. "
        "Keep or replace each input file before starting."
    )
    return True, "Saved run is ready as an editable new-run template."


def _prepare_saved_run_template_callback(run_dir_str: str) -> None:
    run_dir = Path(run_dir_str)
    ok, message = _prepare_saved_run_as_template(run_dir)
    if ok:
        st.session_state.active_run_view = "Run Monitor"
        st.session_state.active_run_view_picker = "Run Monitor"
    st.session_state.selected_run_notice = message
    st.session_state.selected_run_notice_dir = str(run_dir)
    st.session_state.prepare_template_result_level = "success" if ok else "warning"


def _render_prepare_saved_run_template_action(run_dir: Path | None, *, key: str) -> None:
    if not run_dir or st.session_state.get("run_active"):
        return
    entry = _reusable_run_entry_for_dir(run_dir)
    if not entry:
        st.caption("This saved run cannot be reused as a setup template because required input files are missing.")
        return
    cfg_path = str(entry.get("config_path") or "")
    if (
        st.session_state.get("dataset_source_mode") == "Reuse Previous Run"
        and cfg_path
        and st.session_state.get("reused_run_selector_path") == cfg_path
    ):
        st.success("Editable copy ready in Data Collection.")
        st.caption("Keep the saved files or replace individual inputs before starting the next run.")
        return
    st.button(
        "Prepare Editable Copy",
        width="stretch",
        key=key,
        on_click=_prepare_saved_run_template_callback,
        args=(str(run_dir),),
    )


def render_workspace_access() -> None:
    context = st.session_state.get("workspace_context") or {}
    mode = context.get("mode", "temporary")
    username = context.get("username", "Temporary")
    root = Path(context.get("root", ""))
    workspace_hint = _query_param_value("workspace") or _query_param_value("user")
    pending_saved_run_url = bool(st.session_state.get("query_run_error") and mode == "temporary")
    if mode == "workspace":
        st.success(f"Workspace: {username}")
        st.caption("Persistent runs and custom libraries are stored in this workspace.")
    elif pending_saved_run_url:
        target = workspace_hint or "the requested workspace"
        st.info(f"Open `{target}` to load the saved run.")
        st.caption("Enter the workspace PIN below; the saved run will open automatically.")
    elif workspace_hint:
        st.info(f"Open workspace `{workspace_hint}` to restore saved runs and libraries.")
        st.caption("Enter the workspace PIN below. The PIN is not stored in the URL.")
    else:
        st.warning("Temporary session")
        st.caption("Runs and custom libraries are stored temporarily. Use a workspace to retrieve them later.")
    if not pending_saved_run_url:
        st.caption(f"Storage: `{root.name if root else '-'}`")
        counts = _workspace_storage_counts(root) if str(root) not in ("", ".") else {"runs": 0, "libraries": 0}
        st.caption(f"Saved runs: `{counts['runs']}` | Custom libraries: `{counts['libraries']}`")
    notice = st.session_state.get("workspace_notice")
    if notice:
        level = notice.get("level", "info")
        message = notice.get("message", "")
        detail = notice.get("detail", "")
        if level == "success":
            st.success(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.info(message)
        if detail:
            st.caption(detail)

    disabled = bool(st.session_state.get("run_active"))
    expand_workspace_access = bool(
        mode == "temporary"
        and (
            st.session_state.get("query_run_error")
            or workspace_hint
        )
    )
    with st.expander("Workspace Access", expanded=expand_workspace_access):
        if disabled:
            st.info("Stop or finish the active run before changing workspaces.")
        st.caption("This is lightweight workspace lookup, not secure authentication.")
        if pending_saved_run_url:
            requested_run = Path(_query_param_value("run") or _query_param_value("saved_run")).name
            target = workspace_hint or "the owning workspace"
            st.info(
                f"Open `{target}` in the **Open Saved Run** panel to load `{requested_run}`."
            )
            st.caption("The PIN form is shown once in the main workspace area to avoid editing two copies.")
            return
        with st.form("workspace_login_form"):
            username_input = st.text_input(
                "Workspace username",
                value=(workspace_hint if mode == "temporary" else str(username)),
                key="sidebar_workspace_username",
            )
            pin_input = st.text_input(
                "4-digit PIN",
                type="password",
                max_chars=4,
                key="sidebar_workspace_pin",
            )
            submitted = st.form_submit_button("Open Existing / Create New Workspace", disabled=disabled)
        if submitted:
            context, error, action = _open_or_create_workspace(username_input, pin_input)
            if error:
                st.error(error)
            else:
                counts = _workspace_storage_counts(Path(context["root"]))
                _activate_workspace_context(context)
                if action == "created":
                    st.session_state.workspace_notice = {
                        "level": "success",
                        "message": f"Created new workspace `{context['username']}`.",
                        "detail": "New workspace is empty. Future runs and custom libraries will be saved here.",
                    }
                else:
                    st.session_state.workspace_notice = {
                        "level": "success",
                        "message": f"Opened existing workspace `{context['username']}`.",
                        "detail": f"Loaded {counts['runs']} saved run(s) and {counts['libraries']} custom library pack(s).",
                    }
                _apply_saved_run_query_param()
                st.rerun()
        if st.button("Use New Temporary Session", disabled=disabled, width="stretch"):
            context = _create_temporary_workspace_context()
            _activate_workspace_context(context)
            st.session_state.workspace_notice = {
                "level": "info",
                "message": "Started a new temporary session.",
                "detail": "Runs and custom libraries in this temporary session are not meant for later retrieval.",
            }
            st.rerun()


def render_pending_saved_run_workspace_gate() -> None:
    context = st.session_state.get("workspace_context") or {}
    if context.get("mode", "temporary") != "temporary":
        return
    if not st.session_state.get("query_run_error"):
        return

    requested_run = Path(_query_param_value("run") or _query_param_value("saved_run") or "").name
    workspace_hint = _query_param_value("workspace") or _query_param_value("user") or ""
    target = workspace_hint or "the owning workspace"
    disabled = bool(st.session_state.get("run_active"))

    with st.container(border=True):
        st.markdown("#### Open Saved Run")
        st.caption(
            "This shared/recovery link needs the workspace PIN before the saved run can be loaded. "
            "Use this panel to open the workspace and recover the run."
        )
        if requested_run:
            st.caption(f"Requested run: `{requested_run}`")
        with st.form("main_pending_workspace_login_form"):
            username_input = st.text_input(
                "Workspace username",
                value=workspace_hint,
                placeholder="Workspace username",
                key="main_pending_workspace_username",
                disabled=disabled,
            )
            pin_input = st.text_input(
                "4-digit PIN",
                type="password",
                max_chars=4,
                key="main_pending_workspace_pin",
                disabled=disabled,
            )
            submitted = st.form_submit_button(
                "Open workspace and load saved run",
                disabled=disabled,
                width="stretch",
            )
        if submitted:
            context, error, action = _open_or_create_workspace(username_input, pin_input)
            if error:
                st.error(error)
            else:
                counts = _workspace_storage_counts(Path(context["root"]))
                _activate_workspace_context(context)
                st.session_state.workspace_notice = {
                    "level": "success",
                    "message": f"{'Created' if action == 'created' else 'Opened existing'} workspace `{context['username']}`.",
                    "detail": f"Loaded {counts['runs']} saved run(s) and {counts['libraries']} custom library pack(s).",
                }
                _apply_saved_run_query_param()
                st.rerun()


def render_run_history_loader() -> None:
    entries = cached_workspace_run_entries(str(ACTIVE_RUNS_ROOT))
    selected_path = st.session_state.get("selected_run_dir")
    notice = st.session_state.get("selected_run_notice")
    notice_dir = st.session_state.get("selected_run_notice_dir")
    current_view_run_dir = _selected_or_active_run_dir()
    recovered_running_view = False
    if current_view_run_dir:
        manifest_status = str((_read_run_manifest(current_view_run_dir) or {}).get("status") or "").strip().lower()
        try:
            selected_matches_view = bool(
                selected_path
                and Path(str(selected_path)).resolve() == current_view_run_dir.resolve()
            )
        except Exception:
            selected_matches_view = False
        recovered_running_view = bool(
            manifest_status in {"running", "starting", "processing"}
            and not selected_matches_view
        )
    pending_saved_run_url = bool(
        st.session_state.get("query_run_error")
        and (st.session_state.get("workspace_context") or {}).get("mode", "temporary") == "temporary"
    )
    if selected_path and Path(selected_path).exists() and not recovered_running_view:
        selected_run_path = Path(selected_path)
        _render_run_context_banner(
            selected_run_path,
            mode=_run_analysis_mode_from_dir(selected_run_path),
            status=_run_display_status(selected_run_path),
            compact=True,
        )
    elif st.session_state.get("run_dir"):
        st.caption(f"Viewing active run: `{Path(st.session_state.run_dir).name}`")
    elif current_view_run_dir:
        _render_run_context_banner(
            current_view_run_dir,
            mode=_run_analysis_mode_from_dir(current_view_run_dir),
            status=(
                str((_read_run_manifest(current_view_run_dir) or {}).get("status") or "").strip()
                or _run_display_status(current_view_run_dir, fallback="latest saved result")
            ),
            active=recovered_running_view,
            compact=True,
        )
    if (
        notice
        and not recovered_running_view
        and selected_path
        and notice_dir
        and str(Path(notice_dir)) == str(Path(selected_path))
    ):
        notice_level = st.session_state.get("prepare_template_result_level")
        if notice_level == "success":
            st.success(str(notice))
        elif notice_level == "warning":
            st.warning(str(notice))
        else:
            st.caption(str(notice))

    with st.expander("Open Previous Run", expanded=False):
        if pending_saved_run_url:
            requested_run = Path(_query_param_value("run") or _query_param_value("saved_run") or "").name
            workspace_hint = _query_param_value("workspace") or _query_param_value("user") or "the saved-run workspace"
            st.info(
                f"Open workspace `{workspace_hint}` with its PIN first. "
                f"Then `{requested_run or 'the requested run'}` and other saved runs will appear here."
            )
            return
        if not entries:
            st.caption("No saved runs are available in this workspace yet.")
            return
        with st.form("saved_run_filter_form"):
            st.text_input(
                "Filter saved runs",
                value=st.session_state.get("saved_run_loader_filter", ""),
                placeholder="Run name, Full/Rapid, Failed/Complete...",
                key="saved_run_loader_filter",
                help="Type part of a run name, mode, status, or timestamp to narrow the saved-run list.",
            )
            st.form_submit_button("Apply Filter", width="stretch")
        run_filter = str(st.session_state.get("saved_run_loader_filter", "")).strip().lower()
        previous_filter = str(st.session_state.get("saved_run_loader_last_filter", "")).strip().lower()
        filter_changed = run_filter != previous_filter
        st.session_state.saved_run_loader_last_filter = run_filter
        if run_filter:
            entries = [
                entry
                for entry in entries
                if run_filter
                in " ".join(
                    [
                        str(entry.get("name", "")),
                        str(entry.get("mode", "")),
                        str(entry.get("status", "")),
                        str(entry.get("label", "")),
                    ]
                ).lower()
            ]
            st.caption(f"Showing {len(entries)} matching saved run(s).")
        if not entries:
            st.warning("No saved runs match this filter.")
            return
        entry_options = [str(Path(entry["path"])) for entry in entries]
        entry_by_path = {str(Path(entry["path"])): entry for entry in entries}
        preferred_entry_path = entry_options[0]
        if not run_filter and selected_path and str(Path(selected_path)) in entry_by_path:
            preferred_entry_path = str(Path(selected_path))
        current_choice_path = st.session_state.get("saved_run_loader_choice_path")
        if filter_changed or current_choice_path not in entry_by_path or (run_filter and len(entry_options) == 1):
            st.session_state.saved_run_loader_choice_path = preferred_entry_path
        choice = st.selectbox(
            "Saved run",
            entry_options,
            format_func=lambda path: _saved_run_sidebar_label(entry_by_path[str(path)]),
            key="saved_run_loader_choice_path",
            help="Loads the selected run's logs, results, rapid artifacts, plots, and files from its saved run folder.",
        )
        entry = entry_by_path[str(choice)]
        entry_path = Path(entry["path"])
        loaded_selected_entry = bool(current_view_run_dir and str(current_view_run_dir) == str(entry_path))
        mode_label = html.escape(str(entry.get("mode", "-")))
        status_label = html.escape(str(entry.get("status", "-")))
        status_class = re.sub(r"[^a-z0-9_-]+", "-", str(entry.get("status", "")).strip().lower()).strip("-")
        status_class_attr = f" status-{html.escape(status_class, quote=True)}" if status_class else ""
        plot_count = html.escape(str(entry.get("plot_count", 0)))
        table_count = html.escape(str(entry.get("csv_count", 0)))
        st.markdown(
            (
                '<div class="radar-sidebar-run-meta">'
                f'<span><strong>Mode</strong>{mode_label}</span>'
                f'<span class="{status_class_attr.strip()}"><strong>Status</strong>{status_label}</span>'
                f'<span><strong>Plots</strong>{plot_count}</span>'
                f'<span><strong>Tables</strong>{table_count}</span>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        path_name = entry_path.name
        st.markdown(
            (
                '<div class="radar-sidebar-path-caption">'
                'Path: '
                f'<code title="{html.escape(path_name, quote=True)}">{html.escape(_compact_run_name(path_name, max_chars=28))}</code>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )
        if loaded_selected_entry:
            st.caption("This run is already loaded in the workspace view.")
        else:
            st.info("This run is selected in the menu but not loaded yet.")
        load_label = "Reload This Run" if loaded_selected_entry else "Load This Run"
        if st.button(load_label, width="stretch", key="load_selected_saved_run"):
            _load_selected_run(entry_path)
            st.rerun()
        if selected_path:
            if st.button("Return to Active / Latest Run", width="stretch", key="clear_selected_saved_run"):
                _clear_selected_run_context()
                if st.session_state.get("run_dir"):
                    st.session_state.run_name = Path(st.session_state.run_dir).name
                st.rerun()


# --- STATE INITIALIZATION ---
if 'workspace_context' not in st.session_state:
    st.session_state.workspace_context = _create_temporary_workspace_context()
WORKSPACE_ROOT = Path(st.session_state.workspace_context["root"])
ACTIVE_RUNS_ROOT = Path(st.session_state.workspace_context["runs_root"])
USER_DB_PACKS_DIR = Path(st.session_state.workspace_context["user_db_packs_root"])
_ensure_workspace_dirs(WORKSPACE_ROOT)

if 'run_active' not in st.session_state:
    st.session_state.run_active = False
if 'radiation_source' not in st.session_state:
    st.session_state.radiation_source = "Neutron"
if 'run_finished' not in st.session_state:
    st.session_state.run_finished = False
if 'run_dir' not in st.session_state:
    st.session_state.run_dir = None
if 'last_finished_run_dir' not in st.session_state:
    st.session_state.last_finished_run_dir = None
if 'selected_run_dir' not in st.session_state:
    st.session_state.selected_run_dir = None
if 'selected_run_notice' not in st.session_state:
    st.session_state.selected_run_notice = None
if 'selected_run_notice_dir' not in st.session_state:
    st.session_state.selected_run_notice_dir = None
if 'query_run_error' not in st.session_state:
    st.session_state.query_run_error = None
if 'last_finished_run_name' not in st.session_state:
    st.session_state.last_finished_run_name = None
if 'last_finished_analysis_mode' not in st.session_state:
    st.session_state.last_finished_analysis_mode = None
if 'log_lines' not in st.session_state:
    st.session_state.log_lines = []
if 'show_full_log_history' not in st.session_state:
    st.session_state.show_full_log_history = False
if 'pipeline_process' not in st.session_state:
    st.session_state.pipeline_process = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = None
if 'funnel_data' not in st.session_state:
    st.session_state.funnel_data = {
        "Total Database": 0, "Elements": 0, "Spacegroup": 0, "Stability": 0
    }
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'status_msg' not in st.session_state:
    st.session_state.status_msg = "Ready"
if 'current_stage_desc' not in st.session_state:
    st.session_state.current_stage_desc = "Ready"
if 'custom_run_name' not in st.session_state:
    st.session_state.custom_run_name = make_default_run_name()
if 'run_name_mode' not in st.session_state:
    st.session_state.run_name_mode = "auto"
if st.session_state.pop("pending_run_name_reset", False):
    st.session_state.custom_run_name = make_default_run_name()
    st.session_state.run_name_mode = "auto"
if 'db_selection_mode' not in st.session_state:
    st.session_state.db_selection_mode = "Original"
if 'analysis_mode' not in st.session_state:
    st.session_state.analysis_mode = "Full RADAR-PD"
if 'rapid_stage_output_limit' not in st.session_state:
    st.session_state.rapid_stage_output_limit = 10
if 'rapid_gsas_validation_limit' not in st.session_state:
    st.session_state.rapid_gsas_validation_limit = 10
if 'rapid_gsas_parallel_workers' not in st.session_state:
    st.session_state.rapid_gsas_parallel_workers = 4
if 'rapid_final_polish_enabled' not in st.session_state:
    st.session_state.rapid_final_polish_enabled = False
if 'rapid_hypothesis_phase_count' not in st.session_state:
    st.session_state.rapid_hypothesis_phase_count = 3
if 'rapid_enable_family_variants' not in st.session_state:
    st.session_state.rapid_enable_family_variants = True
if 'rapid_demo_fixture_enabled' not in st.session_state:
    st.session_state.rapid_demo_fixture_enabled = False
if 'magnetic_precheck_enabled' not in st.session_state:
    st.session_state.magnetic_precheck_enabled = False
if 'magnetic_precheck_q_max' not in st.session_state:
    st.session_state.magnetic_precheck_q_max = 6.0
if 'magnetic_precheck_denominators' not in st.session_state:
    st.session_state.magnetic_precheck_denominators = "2,3"
if 'main_phase_shadow_filter_enabled' not in st.session_state:
    st.session_state.main_phase_shadow_filter_enabled = True
if 'main_phase_cleanup_enabled' not in st.session_state:
    st.session_state.main_phase_cleanup_enabled = False
if 'main_phase_cleanup_refine_u_iso' not in st.session_state:
    st.session_state.main_phase_cleanup_refine_u_iso = False
if 'main_phase_cleanup_refine_positions' not in st.session_state:
    st.session_state.main_phase_cleanup_refine_positions = False
if 'reused_run_config_path' not in st.session_state:
    st.session_state.reused_run_config_path = None
if 'reused_instrument_mode' not in st.session_state:
    st.session_state.reused_instrument_mode = "Auto"
if 'selected_custom_pack_xray' not in st.session_state:
    st.session_state.selected_custom_pack_xray = None
if 'selected_custom_pack_neutron' not in st.session_state:
    st.session_state.selected_custom_pack_neutron = None
if 'fit_window_override_enabled' not in st.session_state:
    st.session_state.fit_window_override_enabled = False
if 'fit_window_lower' not in st.session_state:
    st.session_state.fit_window_lower = ""
if 'fit_window_upper' not in st.session_state:
    st.session_state.fit_window_upper = ""
if st.session_state.pop("pending_fit_window_reset", False):
    st.session_state.fit_window_lower = ""
    st.session_state.fit_window_upper = ""
if 'excluded_regions_rows' not in st.session_state:
    st.session_state.excluded_regions_rows = []
if 'excluded_regions_editor_buffer' not in st.session_state:
    st.session_state.excluded_regions_editor_buffer = []
if 'excluded_region_edit_index' not in st.session_state:
    st.session_state.excluded_region_edit_index = None
if 'excluded_region_form_start' not in st.session_state:
    st.session_state.excluded_region_form_start = ""
if 'excluded_region_form_end' not in st.session_state:
    st.session_state.excluded_region_form_end = ""
if st.session_state.pop("pending_excluded_region_form_reset", False):
    st.session_state.excluded_region_form_start = ""
    st.session_state.excluded_region_form_end = ""

if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        "global_stage_idx": -1,
        "global_stage_desc": "Ready",
        "current_pass": 0,
        "pass_stage": None,
        "stage0_status": "pending", # pending, running, complete, skipped
        "stages_complete": set(),
        "rapid_mode": False,
    }
if 'event_file_cursor' not in st.session_state:
    st.session_state.event_file_cursor = 0
if 'event_file_run_dir' not in st.session_state:
    st.session_state.event_file_run_dir = None
if 'last_pipeline_error' not in st.session_state:
    st.session_state.last_pipeline_error = None

# Apply deferred custom-pack activation before any widgets using these keys are created.
pending_db_activation = st.session_state.pop("pending_db_activation", None)
if pending_db_activation:
    try:
        pending_mode = str(pending_db_activation.get("mode", "")).strip()
        pending_source = str(pending_db_activation.get("source", "")).strip().lower()
        pending_pack_root = pending_db_activation.get("pack_root")
        if pending_mode in {"Original", "Augmented Pack", "Mini Pack"}:
            st.session_state.db_selection_mode = pending_mode
        if pending_source in {"xray", "neutron"} and pending_pack_root:
            st.session_state[f"selected_custom_pack_{pending_source}"] = str(pending_pack_root)
    except Exception:
        pass

pending_excluded_region_remove = st.session_state.pop("pending_excluded_region_remove", None)
if pending_excluded_region_remove is not None:
    try:
        idx = int(pending_excluded_region_remove)
        rows = _coerce_excluded_region_rows(st.session_state.get("excluded_regions_editor_buffer"))
        if 0 <= idx < len(rows):
            rows.pop(idx)
            st.session_state.excluded_regions_rows = rows
            st.session_state.excluded_regions_editor_buffer = rows
            edit_idx = st.session_state.get("excluded_region_edit_index")
            if edit_idx == idx:
                st.session_state.excluded_region_edit_index = None
                st.session_state.excluded_region_form_start = ""
                st.session_state.excluded_region_form_end = ""
            elif isinstance(edit_idx, int) and edit_idx > idx:
                st.session_state.excluded_region_edit_index = edit_idx - 1
            st.session_state.pop("excluded_regions_editor", None)
    except Exception:
        pass

editor_state = st.session_state.get("excluded_regions_editor")
if isinstance(editor_state, dict):
    merged_rows = _merge_excluded_region_editor_state(
        st.session_state.get("excluded_regions_editor_buffer"),
        editor_state,
    )
    st.session_state.excluded_regions_editor_buffer = merged_rows
    st.session_state.excluded_regions_rows = merged_rows

# --- DB LOADER INITIALIZATION ---
# Determine active DB based on session state
current_source = st.session_state.get("radiation_source", "Neutron")
IS_XRAY = (current_source == "X-ray")
selection_mode = st.session_state.get("db_selection_mode", "Original")
selected_pack_key = _selected_pack_state_key(current_source)
if selection_mode != "Original":
    available_packs_for_source = discover_user_db_packs(current_source)
    eligible_pack_roots = [
        pack["root_str"]
        for pack in available_packs_for_source
        if _db_mode_matches_kind(selection_mode, pack["kind"])
    ]
    current_selected_root = st.session_state.get(selected_pack_key)
    if eligible_pack_roots and current_selected_root not in eligible_pack_roots:
        st.session_state[selected_pack_key] = eligible_pack_roots[0]

BUILTIN_DB_DIR = DB_NEUTRON_DIR if not IS_XRAY else DB_XRAY_DIR
BUILTIN_DB_EXISTS = cached_check_db_integrity(str(BUILTIN_DB_DIR), is_xray=IS_XRAY)
ACTIVE_DB_SELECTION = resolve_active_db_selection(current_source)
ACTIVE_DB_ROOT = Path(ACTIVE_DB_SELECTION["root"])
ACTIVE_DB_CONFIG = ACTIVE_DB_SELECTION["db_config"]
ACTIVE_DB_LABEL = ACTIVE_DB_SELECTION["label"]
ACTIVE_DB_KIND = ACTIVE_DB_SELECTION["kind"]
ACTIVE_DB_EXISTS = _db_config_is_usable(ACTIVE_DB_CONFIG)
ACTIVE_METADATA_JSON = ACTIVE_DB_CONFIG.get("original_json")

if st.session_state.get("db_loader_source") != ACTIVE_DB_SELECTION["selection_key"]:
    st.session_state.pop("db_loader", None)
    st.session_state.db_loader_source = None


def get_active_db_loader() -> DBLoader | None:
    if not ACTIVE_DB_EXISTS:
        return None
    if (
        "db_loader" in st.session_state
        and st.session_state.db_loader
        and st.session_state.get("db_loader_source") == ACTIVE_DB_SELECTION["selection_key"]
    ):
        return st.session_state.db_loader
    try:
        loader = load_cached_db_loader(
            ACTIVE_DB_SELECTION["selection_key"],
            str(ACTIVE_DB_CONFIG["catalog_csv"]),
            ACTIVE_DB_CONFIG.get("cif_map_json"),
            ACTIVE_METADATA_JSON,
            ACTIVE_DB_CONFIG.get("stable_csv"),
            _path_mtime(ACTIVE_DB_CONFIG["catalog_csv"]),
            _path_mtime(ACTIVE_DB_CONFIG.get("cif_map_json")),
            _path_mtime(ACTIVE_METADATA_JSON),
            _path_mtime(ACTIVE_DB_CONFIG.get("stable_csv")),
        )
        st.session_state.db_loader = loader
        st.session_state.db_loader_source = ACTIVE_DB_SELECTION["selection_key"]
        return loader
    except Exception as e:
        st.error(f"Failed to initialize {ACTIVE_DB_LABEL} database catalog: {e}")
        return None

# --- LOG STATE CLEANUP (Recovery from previous formatting mistakes) ---
if 'log_lines' in st.session_state:
    # If any line contains a span tag, it shouldn't be there. Revert to raw.
    if any("<span" in str(line) for line in st.session_state.log_lines):
        st.session_state.log_lines = [re.sub(r'<[^>]+>', '', line) for line in st.session_state.log_lines]

if 'log_autoscroll' not in st.session_state:
    st.session_state.log_autoscroll = True

# --- HELPER FUNCTIONS ---
def format_log_line(line):
    """Simplified highlighting: Lavender for headers, Cyan for metrics."""
    # Escape HTML
    l = html.escape(line)

    # Check for Headers/Boundaries
    if any(k in l for k in ["STAGE", "PASS", "SUMMARY", "PROCESSING", "====", "----"]):
        return f'<span class="log-header">{l}</span>'

    # Check for Metrics
    if any(k in l for k in ["score", "cos", "alpha", "knee", "explained", "Rwp", "GOF", "Pearson"]):
        return f'<span class="log-metric">{l}</span>'

    return l

# --- STAGE TRACKING ---
GLOBAL_STAGES = [
    "Stage 0: Bootstrap (Find Main Phase)",
    "Stage 1: Main Phase Refinement",
    "Stage 2: Residual Extraction",
    "Sequential Discovery Passes",
    "Final Reporting"
]

PASS_STAGES = [
    ("screening", "ML Screening"),
    ("nudging", "Lattice Nudging"),
    ("pearson", "Pearson Refinement"),
    ("joint", "Joint Refinement"),
    ("polish", "Polishing"),
    ("summary", "Pass Summary")
]


RAPID_STAGES = [
    "Setup and signal import",
    "Coarse hypothesis search",
    "Lattice nudging",
    "Pattern scoring",
    "Final refinement ranking",
    "Rapid report",
]

RAPID_STAGE_NOTES = [
    "Load inputs, candidate library, chemistry filters, and diffraction signal.",
    "Search broad phase combinations with shift-tolerant profiles.",
    "Nudge each unique candidate cell with the RADAR-style Pearson stage.",
    "Rebuild nudged profiles at higher resolution and score hypotheses.",
    "Run targeted, guarded final refinements on the shortlist.",
    "Write summary tables, fit curves, GPX files, and inspector data.",
]


def _rapid_stage_index(stage: str) -> int | None:
    text = str(stage or "").strip().lower()
    if "rapid" not in text:
        return None
    if "complete" in text:
        return len(RAPID_STAGES) - 1
    match = re.search(r"stage\s*(\d+)", text)
    if not match:
        return 0
    stage_num = int(match.group(1))
    return max(0, min(stage_num, len(RAPID_STAGES) - 1))


def parse_pipeline_log_line(line, state):
    """Monotonic log parser using anchored markers."""
    l = line.strip()

    # Global Transitions
    l_up = l.upper()
    if "[RAPID]" in l_up:
        state["rapid_mode"] = True
        if "STARTING" in l_up or "IMPORTING DIFFRACTION SIGNAL" in l_up:
            state["global_stage_idx"] = max(state.get("global_stage_idx", -1), 0)
        elif "LOADING CANDIDATE 64-BIN" in l_up:
            state["global_stage_idx"] = max(state.get("global_stage_idx", -1), 1)
        elif "64-BIN SEARCH COMPLETE" in l_up:
            state["global_stage_idx"] = max(state.get("global_stage_idx", -1), 2)
        elif "NUDGE" in l_up or "NUDGED" in l_up:
            state["global_stage_idx"] = max(state.get("global_stage_idx", -1), 2)
        elif "512-BIN RERANK COMPLETE" in l_up:
            state["global_stage_idx"] = max(state.get("global_stage_idx", -1), 4)
        elif "VALIDATED" in l_up or "GSAS" in l_up:
            state["global_stage_idx"] = max(state.get("global_stage_idx", -1), 4)
        elif "FINISHED" in l_up or "COMPLETE" in l_up:
            state["global_stage_idx"] = len(RAPID_STAGES) - 1
        return state

    if "STAGE 0: BOOTSTRAP" in l_up:
        state["global_stage_idx"] = 0
        state["stage0_status"] = "running"
    elif "STAGE 1: MAIN PHASE REFINEMENT" in l_up:
        if state["global_stage_idx"] < 1:
            if state["stage0_status"] == "pending":
                state["stage0_status"] = "skipped"
            state["global_stage_idx"] = 1
    elif "STAGE 2: RESIDUAL EXTRACTION" in l_up:
        state["global_stage_idx"] = 2
    elif "SEQUENTIAL PHASES" in l_up or ("SEQUENTIAL PASS" in l_up and "discovery" in l):
        state["global_stage_idx"] = 3
        m = re.search(r"PASS (\d+)", l, re.I)
        if m:
            state["current_pass"] = int(m.group(1))
            state["pass_stage"] = "screening"
    elif "FINAL REPORTING" in l_up or "PIPELINE COMPLETED SUCCESSFULLY" in l_up:
        state["global_stage_idx"] = 4

    # Pass-level anchors (within Global Stage 3)
    if state["global_stage_idx"] == 3:
        if "COMPREHENSIVE CANDIDATE SCREENING" in l_up:
            state["pass_stage"] = "screening"
        elif "PROCESSING TOP" in l_up:
            state["pass_stage"] = "nudging"
        elif "[PEARSON]" in l_up:
            state["pass_stage"] = "pearson"
        elif "[CLONE]" in l_up and any(k in l_up for k in ["JOINT", "KEPT", "COMMIT"]):
            state["pass_stage"] = "joint"
        elif "[POLISH] STARTING" in l_up:
            state["pass_stage"] = "polish"
        elif "PASS" in l_up and "SUMMARY" in l_up:
            state["pass_stage"] = "summary"

    return state


def _read_json_safely(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail_text_file(path: Path, lines: int = 80, max_chars: int = 12000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    tail = "\n".join(data.splitlines()[-max(1, int(lines)):])
    return tail[-max_chars:]


def _latest_failure_line(text: str) -> str:
    markers = (
        "Traceback",
        "RuntimeError:",
        "TypeError:",
        "ValueError:",
        "KeyError:",
        "IndexError:",
        "AttributeError:",
        "ImportError:",
        "FileNotFoundError:",
        "ModuleNotFoundError:",
        "[ERROR]",
        "[FATAL]",
        "Pipeline failed",
    )
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if any(marker in line for marker in markers):
            return line
    return ""


def _prefer_specific_failure_reason(*candidates: str) -> str:
    """Choose the most useful short failure message for user-facing status."""
    generic = {"", "Traceback", "Traceback (most recent call last):"}
    cleaned = [str(item or "").strip() for item in candidates]
    for item in cleaned:
        if item and item not in generic and not item.startswith("Traceback"):
            return item
    for item in cleaned:
        if item:
            return item
    return ""


def _failure_log_excerpt(text: str, reason: str = "", *, context: int = 8, max_chars: int = 3200) -> str:
    """Return the most relevant log slice for a failed run."""
    lines = str(text or "").splitlines()
    if not lines:
        return ""
    markers = (
        "Traceback",
        "RuntimeError:",
        "TypeError:",
        "ValueError:",
        "KeyError:",
        "IndexError:",
        "AttributeError:",
        "ImportError:",
        "FileNotFoundError:",
        "ModuleNotFoundError:",
        "[ERROR]",
        "[FATAL]",
        "Pipeline failed",
    )
    hit_idx = None
    reason = str(reason or "").strip()
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if reason and reason in line:
            hit_idx = idx
            break
        if any(marker in line for marker in markers):
            hit_idx = idx
            break
    if hit_idx is None:
        excerpt = "\n".join(lines[-24:])
    else:
        start = max(0, hit_idx - max(1, int(context)))
        for idx in range(hit_idx, -1, -1):
            if "Traceback" in lines[idx]:
                start = max(0, idx)
                break
        end = min(len(lines), hit_idx + max(1, int(context)) + 1)
        excerpt_lines = lines[start:end]
        if start > 0:
            excerpt_lines.insert(0, "[... earlier log omitted ...]")
        if end < len(lines):
            excerpt_lines.append("[... later log omitted ...]")
        excerpt = "\n".join(excerpt_lines)
    return excerpt[-max_chars:]


def _run_manifest_paths(run_dir: Path) -> list[Path]:
    preferred = [
        run_dir / "run_manifest.json",
        run_dir / "Technical" / "Logs" / "run_manifest.json",
    ]
    seen: set[str] = set()
    paths: list[Path] = []
    for path in [*preferred, *sorted(run_dir.rglob("run_manifest.json"))]:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen and path.exists():
            seen.add(key)
            paths.append(path)
    return paths


def _read_run_manifest(run_dir: Path) -> dict:
    manifests: list[dict] = []
    for path in _run_manifest_paths(run_dir):
        manifest = _read_json_safely(path)
        if manifest:
            manifest["_manifest_path"] = str(path)
            manifests.append(manifest)
    if not manifests:
        return {}
    priority = {"failed": 0, "error": 0, "interrupted": 0, "complete": 1, "completed": 1, "running": 2, "starting": 2, "processing": 2}
    manifests.sort(key=lambda item: priority.get(str(item.get("status") or "").strip().lower(), 3))
    return manifests[0]


def _infer_run_stage_from_events(run_dir: Path, *, rapid_mode: bool = False) -> int:
    evt_file = run_dir / "Technical" / "Logs" / "run_events.jsonl"
    if not evt_file.exists():
        return 0
    stage_idx = 0
    try:
        for line in evt_file.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]:
            try:
                evt = json.loads(line)
            except Exception:
                continue
            stage = str(evt.get("stage") or "")
            if rapid_mode or "rapid" in stage.lower():
                rapid_idx = _rapid_stage_index(stage)
                if rapid_idx is not None:
                    stage_idx = max(stage_idx, rapid_idx)
            elif "Stage 1" in stage:
                stage_idx = max(stage_idx, 1)
            elif "Stage 2" in stage:
                stage_idx = max(stage_idx, 2)
            elif "Pass" in stage:
                stage_idx = max(stage_idx, 3)
            elif "Final" in stage or "Complete" in stage:
                stage_idx = max(stage_idx, 4)
    except Exception:
        pass
    return stage_idx


def _write_run_failure_manifest(
    run_dir: Path,
    *,
    returncode: int | None,
    reason: str,
    analysis_mode: str | None = None,
    pipeline_state: dict | None = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = _read_run_manifest(run_dir)
    log_tail = _tail_text_file(run_dir / "pipeline.log", lines=100)
    manifest = dict(existing)
    manifest.update(
        {
            "status": "failed",
            "analysis_mode": (
                "rapid_hypothesis"
                if analysis_mode == "Rapid Hypothesis Mode"
                else existing.get("analysis_mode") or analysis_mode or "unknown"
            ),
            "returncode": returncode,
            "error": reason or _latest_failure_line(log_tail) or "Pipeline failed",
            "failed_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "log_path": str(run_dir / "pipeline.log"),
            "log_tail": log_tail,
            "pipeline_state": pipeline_state or {},
        }
    )
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return manifest


def _run_failure_info(run_dir: Path | None) -> dict | None:
    if not run_dir:
        return None
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return None
    manifest = _read_run_manifest(run_dir)
    status = str(manifest.get("status") or "").strip().lower()
    rapid_mode = (
        manifest.get("analysis_mode") == "rapid_hypothesis"
        or (run_dir / "rapid_results").exists()
    )
    log_tail = str(manifest.get("log_tail") or "") or _tail_text_file(run_dir / "pipeline.log", lines=100)
    if status in {"failed", "error", "interrupted"}:
        state = manifest.get("pipeline_state") if isinstance(manifest.get("pipeline_state"), dict) else {}
        stage_idx = state.get("global_stage_idx")
        if not isinstance(stage_idx, int):
            stage_idx = _infer_run_stage_from_events(run_dir, rapid_mode=rapid_mode)
        latest_reason = _latest_failure_line(log_tail)
        manifest_reason = str(manifest.get("error") or "").strip()
        if not manifest_reason or manifest_reason.startswith("Traceback"):
            manifest_reason = latest_reason
        return {
            "status": status,
            "returncode": manifest.get("returncode"),
            "reason": manifest_reason or "Run failed.",
            "log_tail": log_tail,
            "stage_idx": int(stage_idx),
            "rapid_mode": rapid_mode,
            "manifest_path": manifest.get("_manifest_path") or str(run_dir / "run_manifest.json"),
        }

    if log_tail and "Pipeline finished successfully" not in log_tail and "Rapid Hypothesis Mode finished successfully" not in log_tail:
        latest = _latest_failure_line(log_tail)
        if latest:
            return {
                "status": "failed",
                "returncode": manifest.get("returncode"),
                "reason": latest,
                "log_tail": log_tail,
                "stage_idx": _infer_run_stage_from_events(run_dir, rapid_mode=rapid_mode),
                "rapid_mode": rapid_mode,
                "manifest_path": manifest.get("_manifest_path"),
            }
    return None


def _run_display_status(run_dir: Path | None, *, fallback: str = "saved result") -> str:
    """Return a user-facing saved-run status that preserves failed/complete state."""
    if not run_dir:
        return fallback
    run_dir = Path(run_dir)
    failure_info = _run_failure_info(run_dir)
    if failure_info:
        return "failed"

    manifest = _read_run_manifest(run_dir)
    status = str(manifest.get("status") or manifest.get("run_status") or "").strip().lower()
    if status:
        if status in {"success", "succeeded", "finished"}:
            return "complete"
        return status

    for summary_path in (
        run_dir / "pipeline_summary.json",
        run_dir / "rapid_results" / "summary.json",
        run_dir / "rapid_results" / "rapid_summary.json",
    ):
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(summary.get("status") or summary.get("run_status") or "").strip().lower()
        if status:
            if status in {"success", "succeeded", "finished"}:
                return "complete"
            return status

    log_tail = _tail_text_file(run_dir / "pipeline.log", lines=60)
    if "Pipeline finished successfully" in log_tail or "Rapid Hypothesis Mode finished successfully" in log_tail:
        return "complete"
    return fallback


def update_funnel_metrics(new_lines):
    """Incremental update of funnel metrics from new log lines."""
    data = st.session_state.funnel_data
    for line in new_lines:
        if "catalog size:" in line:
            m = re.search(r"catalog size:\s+(\d+)", line)
            if m: data["Total Database"] = int(m.group(1))
        if "matching elements" in line:
            m = re.search(r"elements:\s+(\d+)", line)
            if m: data["Elements"] = int(m.group(1))
        if "matching spacegroup" in line:
            m = re.search(r"spacegroup:\s+(\d+)", line)
            if m: data["Spacegroup"] = int(m.group(1))
        if "stable phases loaded" in line:
            m = re.search(r"loaded:\s+(\d+)", line)
            if m: data["Stability"] = int(m.group(1))

    # Heuristic fix for missing stability log
    if data["Stability"] == 0 and data["Spacegroup"] > 0:
        data["Stability"] = data["Spacegroup"]

    st.session_state.funnel_data = data


def _hide_curated_artifact(path: Path) -> bool:
    """Hide noisy intermediate artifacts from curated UI views."""
    return "_trial_blend" in path.name.lower()


@st.cache_data(show_spinner=False, ttl=6)
def cached_has_matching_artifact_files(dir_path: str, filter_exts_key: tuple[str, ...], hide_trial_blend: bool) -> bool:
    root = Path(dir_path)
    if not root.is_dir():
        return False
    for child in root.rglob("*"):
        if not child.is_file() or child.name.startswith("."):
            continue
        if hide_trial_blend and "_trial_blend" in child.name.lower():
            continue
        if not filter_exts_key or child.suffix.lower() in filter_exts_key:
            return True
    return False


def render_file_explorer(
    path: Path,
    key_prefix: str,
    filter_exts=None,
    depth=0,
    hide_predicate=None,
    *,
    show_downloads: bool = True,
):
    """Recursive file explorer UI component with improved layout."""
    if not path.is_dir():
        return False

    filter_exts_key = tuple(sorted(str(ext).lower() for ext in (filter_exts or [])))
    hide_trial_blend = hide_predicate is not None
    items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
    has_content = False

    for item in items:
        if item.name.startswith("."): continue

        # Indentation for depth
        margin_left = depth * 20


        # Recursive Folder
        if item.is_dir():
            unique_key = f"{key_prefix}_{item.name}"

            if cached_has_matching_artifact_files(str(item), filter_exts_key, hide_trial_blend):
                has_content = True
                with st.expander(f"{item.name}", expanded=(depth < 1)):
                    render_file_explorer(
                        item,
                        unique_key,
                        filter_exts,
                        depth + 1,
                        hide_predicate=hide_predicate,
                        show_downloads=show_downloads,
                    )

        # File Display
        else:
            if filter_exts and item.suffix.lower() not in filter_exts:
                continue
            if hide_predicate and hide_predicate(item):
                continue

            has_content = True
            file_name = html.escape(item.name)
            file_size = item.stat().st_size / 1024
            file_html = f"""
                <div style="margin-left: {margin_left}px; padding: 4px 0;">
                    <span class="file-tree-file">{file_name}</span>
                    <span style="font-size: 0.8em; color: #718096; margin-left: 10px;">({file_size:.1f} KB)</span>
                </div>
            """
            if show_downloads:
                c1, c2 = st.columns([0.70, 0.30])
                with c1:
                    st.markdown(file_html, unsafe_allow_html=True)
            else:
                st.markdown(file_html, unsafe_allow_html=True)
            if show_downloads:
                with c2:
                    # Key must be unique per file. Avoid building these payloads in live artifact previews.
                    with open(item, "rb") as f:
                        st.download_button(
                            "Download",
                            f,
                            file_name=item.name,
                            key=f"dl_{key_prefix}_{item.name}",
                            help=f"Download {item.name}",
                            width="stretch",
                        )

            # Preview for Artifacts (Lazy Load)
            if item.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                if item.stat().st_size > 0:
                    # Memory optimization: Only auto-expand in Plot folders, otherwise use a toggle
                    # AND: Only auto-expand if the run is NOT active to prevent UI churn
                    is_running = st.session_state.get("run_active", False)
                    should_preview = ("Plots" in str(item.parent) or "Diagnostics" in str(item.parent)) and not is_running

                    if should_preview:
                        try:
                            from PIL import Image
                            st.image(str(item), width="stretch" if hasattr(st, "image") else None)
                        except Exception:
                            st.caption(f"Image {item.name} is still being written.")
                    else:
                        if st.checkbox(f"Preview {item.name}", key=f"pv_{key_prefix}_{item.name}"):
                            from PIL import Image
                            st.image(str(item), width="stretch")

    return has_content


def render_download_file(path: Path, key_prefix: str) -> bool:
    if not path.exists() or not path.is_file():
        return False
    c1, c2 = st.columns([0.70, 0.30])
    with c1:
        file_name = html.escape(path.name)
        file_size = path.stat().st_size / 1024
        st.markdown(
            f"<span class='file-tree-file'>{file_name}</span> "
            f"<span style='font-size: 0.8em; color: #718096;'>({file_size:.1f} KB)</span>",
            unsafe_allow_html=True,
        )
        st.caption(str(path.parent.name))
    with c2:
        with open(path, "rb") as f:
            st.download_button(
                "Download",
                f,
                file_name=path.name,
                key=f"dl_key_{key_prefix}_{path.name}",
                help=f"Download {path.name}",
                width="stretch",
            )
    return True


def render_key_run_files(rdir: Path) -> None:
    shown = False
    key_files = [
        rdir / "pipeline.log",
        rdir / "pipeline_config.yaml",
        rdir / "pipeline_summary.json",
        rdir / "Results" / "Summary_Fractions.csv",
    ]
    st.markdown("**Key files**")
    for idx, path in enumerate(key_files):
        shown = render_download_file(path, f"key_{idx}") or shown

    plot_dir = rdir / "Results" / "Plots"
    if plot_dir.exists():
        plot_files = sorted(
            [p for p in plot_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf"}],
            key=lambda p: p.name.lower(),
        )
        if plot_files:
            st.markdown("**Plots**")
            for idx, path in enumerate(plot_files):
                shown = render_download_file(path, f"plot_{idx}") or shown

    diag_dir = rdir / "Diagnostics"
    if diag_dir.exists():
        diag_files = sorted(
            [
                p for p in diag_dir.iterdir()
                if p.is_file() and p.name.startswith("ml_rank_") and p.suffix.lower() in {".json", ".jsonl"}
            ],
            key=lambda p: p.name.lower(),
        )
        if diag_files:
            st.markdown("**ML diagnostics**")
            for idx, path in enumerate(diag_files):
                shown = render_download_file(path, f"diag_{idx}") or shown

    if not shown:
        st.info("No key output files are available yet.")


def update_ui_state():
    """Polls the runner queue and updates session state."""
    if st.session_state.pipeline_process and st.session_state.log_queue:
        q = st.session_state.log_queue
        process = st.session_state.pipeline_process
        new_lines = []

        try:
            while True:
                line = q.get_nowait()
                new_lines.append(line)
        except queue.Empty:
            pass

        if new_lines:
            st.session_state.log_lines.extend(new_lines)
            update_funnel_metrics(new_lines)
            for line in new_lines:
                plain = str(line).strip()
                if (
                    "[ERROR]" in plain
                    or "Traceback" in plain
                    or "RuntimeError" in plain
                    or "TypeError:" in plain
                    or "ValueError:" in plain
                    or "KeyError:" in plain
                    or "IndexError:" in plain
                    or "AttributeError:" in plain
                    or "ImportError:" in plain
                    or "FileNotFoundError:" in plain
                    or "ModuleNotFoundError:" in plain
                    or "Failed to add" in plain
                    or "Failed to prepare" in plain
                ):
                    st.session_state.last_pipeline_error = _prefer_specific_failure_reason(
                        plain,
                        st.session_state.get("last_pipeline_error"),
                    )

            # MEMORY OPTIMIZATION: Keep only last 2000 lines for live view
            if len(st.session_state.log_lines) > 2000:
                st.session_state.log_lines = st.session_state.log_lines[-2000:]

        # Update Progress State
        state = st.session_state.pipeline_state

        # 1. Fallback / Complementary: Parsing raw logs (Heuristic-based)
        if new_lines:
            for line in new_lines:
                state = parse_pipeline_log_line(line, state)

        # 2. Primary Source: Structured Events (JSONL) - Accurate and high-confidence
        if st.session_state.run_dir:
            run_path = Path(st.session_state.run_dir)
            evt_file = run_path / "Technical" / "Logs" / "run_events.jsonl"

            if evt_file.exists():
                try:
                    run_dir_str = str(run_path)
                    if st.session_state.event_file_run_dir != run_dir_str:
                        st.session_state.event_file_run_dir = run_dir_str
                        st.session_state.event_file_cursor = 0

                    with open(evt_file, "r", encoding="utf-8") as f:
                        try:
                            f.seek(st.session_state.event_file_cursor)
                        except Exception:
                            st.session_state.event_file_cursor = 0
                            f.seek(0)

                        lines = f.readlines()
                        st.session_state.event_file_cursor = f.tell()

                    if lines:
                        for line in lines[-20:]:
                            evt = json.loads(line)
                            if "percent" in evt:
                                st.session_state.progress = int(evt["percent"])

                            stage = evt.get("stage", "")
                            message = str(evt.get("message", "") or "").strip()
                            if message:
                                st.session_state.current_stage_desc = message
                                st.session_state.status_msg = message
                            metrics = evt.get("metrics", {})
                            rapid_idx = _rapid_stage_index(stage)

                            if rapid_idx is not None:
                                state["rapid_mode"] = True
                                state["global_stage_idx"] = rapid_idx
                                state["stage0_status"] = "complete" if rapid_idx > 0 else "running"
                            elif "Stage 0" in stage:
                                state["rapid_mode"] = False
                                state["global_stage_idx"] = 0
                                if "Bootstrap complete" in evt.get("message", ""):
                                    state["stage0_status"] = "complete"
                                else:
                                    state["stage0_status"] = "running"
                            elif "Stage 1" in stage:
                                state["rapid_mode"] = False
                                state["global_stage_idx"] = 1
                            elif "Stage 2" in stage:
                                state["rapid_mode"] = False
                                state["global_stage_idx"] = 2
                            elif "Pass" in stage:
                                state["rapid_mode"] = False
                                state["global_stage_idx"] = 3
                                state["current_pass"] = metrics.get("pass", state["current_pass"])
                                event_type = metrics.get("event")
                                if event_type in ["pass_start", "screening_start"]: state["pass_stage"] = "screening"
                                elif event_type == "nudging_start": state["pass_stage"] = "nudging"
                                elif event_type == "pearson_start": state["pass_stage"] = "pearson"
                                elif event_type in ["joint_compare_start", "joint_refine_start"]: state["pass_stage"] = "joint"
                                elif event_type == "polish_start": state["pass_stage"] = "polish"
                                elif event_type == "pass_end": state["pass_stage"] = "summary"
                            elif "Final" in stage or "Complete" in stage:
                                state["rapid_mode"] = False
                                state["global_stage_idx"] = 4
                except Exception as e:
                    print(f"[ui] Warning: failed to parse recent events from {evt_file}: {e}")

        st.session_state.pipeline_state = state

        # Check Process Status
        if process.poll() is not None and q.empty():
            completed_run_dir = st.session_state.get("run_dir")
            completed_run_name = st.session_state.get("run_name")
            completed_summary = st.session_state.get("run_summary") or {}
            completed_analysis_mode = completed_summary.get("analysis_mode") or st.session_state.get("analysis_mode")
            if completed_run_dir:
                st.session_state.last_finished_run_dir = str(completed_run_dir)
            if completed_run_name:
                st.session_state.last_finished_run_name = str(completed_run_name)
            if completed_analysis_mode:
                st.session_state.last_finished_analysis_mode = str(completed_analysis_mode)
            st.session_state.run_active = False
            st.session_state.run_finished = True
            st.session_state.pipeline_process = None
            st.session_state.log_queue = None
            if completed_analysis_mode == "Rapid Hypothesis Mode":
                st.session_state.analysis_mode = "Rapid Hypothesis Mode"
                st.session_state.active_run_view = "Rapid Results" if process.returncode == 0 else "Run Monitor"
            else:
                st.session_state.active_run_view = "Results"
            if process.returncode == 0:
                if completed_run_dir:
                    try:
                        run_path = Path(completed_run_dir)
                        existing_manifest = _read_run_manifest(run_path)
                        success_manifest = dict(existing_manifest)
                        success_manifest.update(
                            {
                                "status": "complete",
                                "analysis_mode": (
                                    "rapid_hypothesis"
                                    if completed_analysis_mode == "Rapid Hypothesis Mode"
                                    else existing_manifest.get("analysis_mode") or completed_analysis_mode or "full"
                                ),
                                "returncode": 0,
                                "completed_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                                "log_path": str(run_path / "pipeline.log"),
                            }
                        )
                        (run_path / "run_manifest.json").write_text(
                            json.dumps(success_manifest, indent=2, default=str),
                            encoding="utf-8",
                        )
                    except Exception as manifest_exc:
                        print(f"[ui] Warning: failed to update run manifest: {manifest_exc}")
                st.session_state.current_stage_desc = "Finished"
                st.session_state.status_msg = "Finished"
                st.success("Run completed successfully.")
                # Trigger sync to HF
                if IS_HF_SPACES:
                    sync_run_to_hf(st.session_state.run_dir, st.session_state.get('run_name', 'default'))
                st.balloons()
            else:
                recent_failure_reason = _latest_failure_line("\n".join(st.session_state.log_lines[-240:]))
                failure_manifest = {}
                if completed_run_dir:
                    failure_manifest = _write_run_failure_manifest(
                        Path(completed_run_dir),
                        returncode=int(process.returncode) if process.returncode is not None else None,
                        reason=_prefer_specific_failure_reason(
                            recent_failure_reason,
                            st.session_state.get("last_pipeline_error"),
                            f"Pipeline failed with exit code {process.returncode}",
                        ),
                        analysis_mode=completed_analysis_mode,
                        pipeline_state=dict(st.session_state.get("pipeline_state") or {}),
                    )
                failure_reason = (
                    failure_manifest.get("error")
                    if isinstance(failure_manifest, dict)
                    else None
                )
                failure_reason = _prefer_specific_failure_reason(
                    str(failure_reason or ""),
                    recent_failure_reason,
                    st.session_state.get("last_pipeline_error"),
                    f"Pipeline failed with exit code {process.returncode}",
                )
                st.session_state.last_pipeline_error = failure_reason
                st.session_state.current_stage_desc = "Failed"
                st.session_state.status_msg = "Failed"
                st.error(f"Pipeline failed (exit code {process.returncode}). {failure_reason}")

            if st.session_state.get("run_name_mode") == "auto":
                st.session_state.pending_run_name_reset = True

            # Important: Trigger a full rerun to synchronize the "Results" and "Explorer" tabs
            # since they are outside this fragment.
            st.rerun()

# Note: run_monitor_fragment was consolidated into the log/sidebar fragments below
# to ensure data synchronization and avoid queue race conditions.

_apply_saved_run_query_param()

_pending_workspace_mode = (st.session_state.get("workspace_context") or {}).get("mode", "temporary")
_pending_saved_run_gate = bool(st.session_state.get("query_run_error") and _pending_workspace_mode == "temporary")

# --- UI HEADER / PENDING SAVED-RUN GATE ---
if _pending_saved_run_gate:
    _pending_run = Path(_query_param_value("run") or _query_param_value("saved_run")).name
    _pending_workspace = _query_param_value("workspace") or _query_param_value("user") or "the saved-run workspace"
    st.info(
        f"Saved run `{_pending_run}` is waiting for workspace `{_pending_workspace}`. "
        "Enter the workspace PIN below to open it."
    )
    render_pending_saved_run_workspace_gate()
else:
    st.markdown(
        """
        <section class="radar-hero" aria-label="RADAR-PD product header">
            <div class="radar-hero-kicker">RADAR-PD scientific AI workspace</div>
            <h1 class="radar-hero-title">Phase detection for powder diffraction</h1>
            <p class="radar-hero-copy">
                Residual-aware, ML-guided materials analysis with rapid hypothesis search,
                lattice nudging, final refinement ranking, and inspectable scientific outputs.
            </p>
            <div class="radar-hero-chips" aria-label="RADAR-PD capabilities">
                <span class="radar-hero-chip">Neutron and X-ray</span>
                <span class="radar-hero-chip">Custom CIF libraries</span>
                <span class="radar-hero-chip">Rapid hypotheses</span>
                <span class="radar-hero-chip">GSAS-II refinement</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
if IS_HF_SPACES:
    st.info("**Hugging Face Spaces detected**: resource limits (max 2 workers) are active to prevent OOM crashes.")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown(
        """
        <div class="radar-sidebar-brand">
            <div class="radar-sidebar-brand-title">RADAR-PD Setup</div>
            <div class="radar-sidebar-brand-subtitle">Materials analysis configuration</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_workspace_access()
    workspace_mode = (st.session_state.get("workspace_context") or {}).get("mode", "temporary")
    if st.session_state.get("query_run_error") and workspace_mode != "temporary":
        st.warning(st.session_state.query_run_error)
    render_run_history_loader()
    st.divider()
    sidebar_stop_rendered = False
    if st.session_state.run_active:
        @st.fragment(run_every=STATUS_REFRESH_SECONDS)
        def render_active_sidebar_fragment():
            render_active_run_sidebar_monitor()

        render_active_sidebar_fragment()
        sidebar_stop_rendered = True
        st.divider()
    else:
        st.caption("Work from top to bottom. The guide is at the bottom for reference.")

        # 1. Measurement type
        st.markdown("### 1. Measurement Type")
        rad_source = st.radio(
            "Diffraction source",
            ["Neutron", "X-ray"],
            index=0 if st.session_state.radiation_source == "Neutron" else 1,
            key="radiation_selector",
            help="This selects the active neutron or X-ray candidate library and scoring geometry.",
        )
        if rad_source != st.session_state.radiation_source:
            st.session_state.radiation_source = rad_source
            st.session_state.pop("xray_auto_attempted", None)
            st.rerun()

        if not ACTIVE_DB_EXISTS:
            st.error(f"Active candidate library is not usable: {ACTIVE_DB_LABEL}")

        if not BUILTIN_DB_EXISTS:
            if ACTIVE_DB_KIND != "original" and ACTIVE_DB_EXISTS:
                st.warning(
                    f"Built-in {rad_source} database is missing. "
                    f"The current {ACTIVE_DB_KIND} pack is usable for search, but built-in discovery "
                    f"and augmented-pack creation will stay unavailable until the base DB is installed."
                )
            else:
                st.warning(f"Built-in {rad_source} database missing")
                with st.expander("How to fix", expanded=True):
                    default_url = DB_XRAY_GDRIVE_URL if IS_XRAY else DB_NEUTRON_GDRIVE_URL
                    # If X-ray is selected and we have a default GDrive URL, attempt a single automatic download
                    if IS_XRAY and default_url:
                        # Use a session flag to avoid repeated automatic attempts on reruns
                        if not st.session_state.get("xray_auto_attempted", False):
                            st.session_state["xray_auto_attempted"] = True
                            st.info("X-ray database missing - attempting automatic download from configured Google Drive link...")
                            try:
                                success = download_and_extract_db(default_url, target_db_dir=BUILTIN_DB_DIR, is_xray=True)
                                if success:
                                    st.success("X-ray database downloaded and installed. Refreshing UI...")
                                    st.rerun()
                                else:
                                    st.warning("Automatic download failed - please provide a direct link or upload the ZIP manually.")
                            except Exception as _e:
                                st.warning("Automatic download encountered an error - please provide a direct link or upload the ZIP manually.")
                    elif IS_XRAY and not default_url:
                        st.info("X-ray DB: Enter the Google Drive URL for the X-ray database, or leave blank to provide a direct link.")
                    else:
                        st.markdown("""
                            The database was excluded from Git.
                            **Download the ZIP archive** using the pre-filled URL or provide a direct link.
                        """)
                    db_url = st.text_input("Direct Download URL (ZIP)", value=default_url, placeholder="https://drive.google.com/.../database.zip")
                    if st.button("Download & Install Database"):
                        if download_and_extract_db(db_url, target_db_dir=BUILTIN_DB_DIR, is_xray=IS_XRAY):
                            st.success("Database ready! Please refresh.")
                            st.rerun()
                    st.info("Locally, checks for catalog CSV, stable CSV, and profiles.")
        st.caption(f"Active library: {ACTIVE_DB_LABEL}")

        db_notice_peek = st.session_state.get("db_pack_build_notice")
        library_mode_labels = {
            "Original": "Built-in MP/COD catalog",
            "Augmented Pack": "Built-in catalog + my CIFs",
            "Mini Pack": "Only my CIFs",
        }
        with st.expander("2. Candidate Library", expanded=True):
            db_notice = st.session_state.pop("db_pack_build_notice", None)
            if db_notice:
                level = db_notice.get("level", "info")
                message = db_notice.get("message", "")
                detail = db_notice.get("detail")
                if level == "success":
                    st.success(message)
                elif level == "warning":
                    st.warning(message)
                else:
                    st.info(message)
                if detail:
                    st.caption(detail)

            st.markdown("#### Select Phase Library")
            st.caption("Choose whether RADAR-PD searches the built-in MP/COD catalog, a catalog augmented with your CIFs, or only your CIF collection.")

            st.radio(
                "Library mode",
                ["Original", "Augmented Pack", "Mini Pack"],
                key="db_selection_mode",
                format_func=lambda mode: library_mode_labels.get(mode, mode),
                help="Augmented Pack adds uploaded CIFs to the built-in catalog. Mini Pack searches only uploaded CIFs.",
            )

            available_packs = discover_user_db_packs(current_source)
            eligible_packs = [
                pack for pack in available_packs
                if _db_mode_matches_kind(st.session_state.db_selection_mode, pack["kind"])
            ]
            selected_pack_key = _selected_pack_state_key(current_source)
            selected_pack = None

            if st.session_state.db_selection_mode != "Original":
                if eligible_packs:
                    option_roots = [pack["root_str"] for pack in eligible_packs]
                    labels = {pack["root_str"]: pack["label"] for pack in eligible_packs}
                    current_choice = st.session_state.get(selected_pack_key)
                    if current_choice not in option_roots:
                        st.session_state[selected_pack_key] = option_roots[0]
                    st.selectbox(
                        "Saved custom library",
                        option_roots,
                        key=selected_pack_key,
                        format_func=lambda root: labels[root],
                    )
                    selected_pack = next(
                        (pack for pack in eligible_packs if pack["root_str"] == st.session_state.get(selected_pack_key)),
                        None,
                    )
                else:
                    st.warning(f"No {st.session_state.db_selection_mode.lower()} available for {rad_source}.")
                    st.caption(f"Current runs will use fallback DB: {ACTIVE_DB_LABEL}")

            active_phase_count = _active_db_phase_count()
            library_display = library_mode_labels.get(
                st.session_state.db_selection_mode,
                ACTIVE_DB_KIND.title(),
            )
            phase_display = f"{int(active_phase_count):,}" if active_phase_count is not None else "Not available"
            st.markdown(
                """
                <div class="radar-sidebar-kpis">
                  <div class="radar-sidebar-kpi">
                    <div class="radar-sidebar-kpi-label">Library</div>
                    <div class="radar-sidebar-kpi-value">{library}</div>
                  </div>
                  <div class="radar-sidebar-kpi">
                    <div class="radar-sidebar-kpi-label">Active phases</div>
                    <div class="radar-sidebar-kpi-value numeric">{phases}</div>
                  </div>
                </div>
                """.format(library=html.escape(str(library_display)), phases=html.escape(str(phase_display))),
                unsafe_allow_html=True,
            )

            st.info(f"Current run will search: {ACTIVE_DB_LABEL}")
            if st.session_state.db_selection_mode == "Original":
                st.caption("Current runs will use the built-in MP/COD-derived catalog for the selected measurement type.")
            elif ACTIVE_DB_KIND == "original":
                st.warning("A custom-pack mode is selected, but no usable pack is currently selected. Runs will fall back to the built-in database.")
            else:
                detail_bits = [f"Source: {rad_source}", f"Pack type: {ACTIVE_DB_KIND}"]
                if selected_pack and selected_pack.get("n_phases") is not None:
                    detail_bits.append(f"Pack phases: {selected_pack['n_phases']}")
                st.caption(" | ".join(detail_bits))
                st.success("This custom pack is already selected. You do not need to move back up and reselect it.")

            st.divider()
            with st.expander("Create Library from CIFs", expanded=False):
                st.caption("Upload candidate CIF files to create a searchable library in the active workspace. A successful build is auto-selected.")

                build_mode = st.radio(
                    "CIF library type",
                    ["Augmented Pack", "Mini Pack"],
                    horizontal=True,
                    key=f"db_build_mode_{_source_key_from_label(current_source)}",
                    format_func=lambda mode: library_mode_labels.get(mode, mode),
                    help="Augmented adds your CIFs to the built-in catalog. Mini searches only your CIFs.",
                )
                pack_name = st.text_input(
                    "Library name",
                    key=f"db_pack_name_{_source_key_from_label(current_source)}",
                    help="Custom packs are stored under this workspace's user_db_packs/<pack_name>/<source> folder.",
                )
                uploaded_cifs = st.file_uploader(
                    "Candidate CIF files",
                    type=["cif"],
                    accept_multiple_files=True,
                    key=f"db_pack_cifs_{_source_key_from_label(current_source)}",
                    help="No hard file-count limit is enforced here. Practical limits are browser memory, local disk, and pack-build runtime.",
                )

                total_upload_bytes = sum(int(getattr(uploaded, "size", 0)) for uploaded in (uploaded_cifs or []))
                if uploaded_cifs:
                    st.info(
                        f"Selection summary: {len(uploaded_cifs)} CIF files chosen "
                        f"({ _format_bytes(total_upload_bytes) }). This is file selection only, not pack-build progress."
                    )
                    preview_names = [uploaded.name for uploaded in uploaded_cifs[:5]]
                    st.caption("Selected CIFs: " + ", ".join(preview_names) + (" ..." if len(uploaded_cifs) > 5 else ""))
                else:
                    st.caption("No CIF files selected yet.")

                overwrite_pack = st.checkbox(
                    "Overwrite Existing Pack",
                    value=False,
                    key=f"db_pack_overwrite_{_source_key_from_label(current_source)}",
                )

                clean_pack_name = _sanitize_token(pack_name)
                source_key = _source_key_from_label(current_source)
                output_root = USER_DB_PACKS_DIR / clean_pack_name / source_key if clean_pack_name else None
                existing_pack_note = None
                if output_root and output_root.exists():
                    try:
                        existing_layout = get_db_pack_layout(output_root)
                        if existing_layout.manifest_json.exists():
                            existing_manifest = json.loads(existing_layout.manifest_json.read_text(encoding="utf-8"))
                            existing_kind = existing_manifest.get("kind", "custom")
                            existing_count = (
                                existing_manifest.get("n_phases")
                                if existing_manifest.get("n_phases") is not None
                                else existing_manifest.get("n_added_phases")
                            )
                            if existing_count is not None:
                                existing_pack_note = (
                                    f"Existing pack detected at this path: {existing_kind}, {existing_count} phase(s). "
                                    f"{'It will be replaced because overwrite is enabled.' if overwrite_pack else 'Enable overwrite to replace it.'}"
                                )
                            else:
                                existing_pack_note = (
                                    "Existing pack detected at this path. "
                                    f"{'It will be replaced because overwrite is enabled.' if overwrite_pack else 'Enable overwrite to replace it.'}"
                                )
                    except Exception:
                        existing_pack_note = "Existing output folder detected."
                target_lines = [
                    f"Build target: `{library_mode_labels.get(build_mode, build_mode)}` for `{rad_source}`",
                    f"Output folder: `{output_root}`" if output_root else "Output folder: pending pack name",
                    "Base DB merge: built-in database" if build_mode == "Augmented Pack" else "Base DB merge: none (new standalone mini pack)",
                ]
                st.info("\n\n".join(target_lines))
                if existing_pack_note:
                    st.warning(existing_pack_note)

                if st.button("Build Database Pack"):
                    build_errors = []
                    if not clean_pack_name:
                        build_errors.append("Pack name is required.")
                    if not uploaded_cifs:
                        build_errors.append("Please upload at least one CIF file.")
                    if build_mode == "Augmented Pack" and not BUILTIN_DB_EXISTS:
                        build_errors.append(f"The built-in {rad_source} database is required to build an augmented pack.")

                    if build_errors:
                        for msg in build_errors:
                            st.error(msg)
                    else:
                        tmp_parent = Path(PROJECT_ROOT) / ".tmp"
                        tmp_parent.mkdir(parents=True, exist_ok=True)
                        with tempfile.TemporaryDirectory(prefix="db_pack_build_", dir=str(tmp_parent)) as tmpdir:
                            tmpdir_path = Path(tmpdir)
                            cif_paths = []
                            for uploaded in uploaded_cifs:
                                out = tmpdir_path / uploaded.name
                                out.write_bytes(uploaded.getbuffer())
                                cif_paths.append(out)

                            with st.status("Building custom database pack...", expanded=True) as status:
                                status.write(f"Target source: `{rad_source}`")
                                status.write(f"Pack mode: `{build_mode}`")
                                status.write(f"Pack output: `{output_root}`")
                                status.write(f"CIF inputs: `{len(cif_paths)}` file(s), `{_format_bytes(total_upload_bytes)}` total")
                                progress_bar = st.progress(0, text="Preparing custom database pack")
                                progress_detail = st.empty()
                                build_status_started_at = time.perf_counter()

                                def _on_build_progress(event: dict) -> None:
                                    fraction = float(event.get("fraction", 0.0))
                                    message = str(event.get("message", "Building custom database pack"))
                                    progress_bar.progress(max(0, min(100, int(round(fraction * 100)))), text=message)
                                    current = event.get("current")
                                    total = event.get("total")
                                    source_name = event.get("source_name")
                                    elapsed_s = event.get("elapsed_s")
                                    if elapsed_s is None:
                                        elapsed_s = time.perf_counter() - build_status_started_at
                                    stage_elapsed_s = event.get("stage_elapsed_s")
                                    detail_bits = [f"Elapsed: `{_format_duration(elapsed_s)}`"]
                                    step = event.get("step")
                                    if step:
                                        detail_bits.append(f"Stage: `{step}`")
                                    if current is not None and total is not None:
                                        detail_bits.append(f"Progress: `{current}/{total}`")
                                    if stage_elapsed_s is not None:
                                        detail_bits.append(f"Stage time: `{_format_duration(stage_elapsed_s)}`")
                                    counter_labels = [
                                        ("input files", "input_files"),
                                        ("unique CIFs", "unique_cifs"),
                                        ("duplicate uploads", "duplicate_upload_count"),
                                        ("base phases", "base_phase_count"),
                                        ("checked", "checked_count"),
                                        ("queued new", "queued_count"),
                                        ("skipped", "skipped_count"),
                                        ("built", "built_count"),
                                        ("failed", "failed_count"),
                                        ("workers", "workers"),
                                    ]
                                    counters = [
                                        f"{label}: `{event[key]}`"
                                        for label, key in counter_labels
                                        if event.get(key) is not None
                                    ]
                                    detail_lines = [f"**{message}**", " | ".join(detail_bits)]
                                    if counters:
                                        detail_lines.append(" | ".join(counters))
                                    if source_name:
                                        detail_lines.append(f"Current CIF: `{source_name}`")
                                    progress_detail.markdown("  \n".join(detail_lines))

                                try:
                                    if build_mode == "Augmented Pack":
                                        result = build_augmented_db_pack(
                                            cif_paths,
                                            output_root,
                                            source_type=source_key,
                                            base_db_root=BUILTIN_DB_DIR,
                                            overwrite=overwrite_pack,
                                            progress_callback=_on_build_progress,
                                        )
                                    else:
                                        result = build_mini_db_pack(
                                            cif_paths,
                                            output_root,
                                            source_type=source_key,
                                            reference_db_root=BUILTIN_DB_DIR if BUILTIN_DB_EXISTS else None,
                                            overwrite=overwrite_pack,
                                            progress_callback=_on_build_progress,
                                        )
                                    progress_bar.progress(100, text="Custom database pack built")
                                    progress_detail.caption(f"Completed pack build at `{result.pack_root}`")
                                    status.update(label="Custom database pack built", state="complete")
                                    discover_user_db_packs.clear()
                                    load_cached_db_loader.clear()
                                    skipped_count = len(result.failures)
                                    if skipped_count:
                                        st.warning(f"Built with {skipped_count} skipped/failed CIFs. See manifest for details.")
                                    st.session_state["pending_db_activation"] = {
                                        "mode": build_mode,
                                        "source": source_key,
                                        "pack_root": str(result.pack_root),
                                    }
                                    st.session_state["db_pack_build_notice"] = {
                                        "level": "warning" if skipped_count else "success",
                                        "message": (
                                            f"{build_mode} '{clean_pack_name}' built successfully and is now selected for {rad_source.lower()} runs."
                                        ),
                                        "detail": (
                                            f"No further selection is needed. Active pack path: {result.pack_root} | "
                                            f"Usable phases: {len(result.phase_ids)} | Skipped/failed CIFs: {skipped_count}"
                                        ),
                                    }
                                    st.rerun()
                                except Exception as exc:
                                    status.update(label="Pack build failed", state="error")
                                    st.error(f"Failed to build custom pack: {exc}")

        # --- USER GUIDE ---
        if st.session_state.get("_show_inline_workflow_guide", False):
            st.markdown("""
            ### Setup Order
            * **Measurement Type**: choose neutron or X-ray first; this selects the matching candidate-library geometry.
            * **Candidate Library**: search the built-in MP/COD catalog, add your CIFs to it, or search only your CIF collection.
            * **Data Collection**: provide diffraction data, instrument profile, optional known/main CIF, and a fresh run name.
            * **Pattern Regions**: define fit windows or ignored can/artifact regions in the native axis units.
            * **Chemistry Policy**: enter sample elements and sample can/environment elements separately.
            * **Background Correction**: choose the background model before launching the run.
            * **Analysis Mode**: choose full RADAR-PD or rapid hypothesis mode.
            * **Runtime Budget (full mode only)**: start with Balanced; use Quick/Thorough/Custom only when the run-time and sensitivity tradeoff is intentional.

            ### Runtime Monitoring (~5-10 mins)
            * **Initial Verification (< 1 min)**: Check the **Main Phase Fit** in the Artifacts panel to ensure the baseline fitting is accurate.
            * **Sequential Updates**: Look for `seq_pass#N_accepted_model.png` after each pass to see if the added phase explains previously unknown peaks.
            * **Candidate Tracking**: Browse `Diagnostics / Screening_Histograms` or the **Results ML Ranker** tab to see top candidates by formula and space group.

            ### Reviewing Results
            * **Quantification**: Use **Weighted Fraction Pct** from the final Data Sheet.
            * **Interactive Plots**: Open the **Interactive Plots** tab for zoomable, exportable Rietveld fit visualizations.
            """)


        # --- 3. DATA COLLECTION ---
        with st.expander("3. Data Collection", expanded=True):
            staged_data_file = _get_staged_upload("diffraction_data")
            staged_instprm_file = _get_staged_upload("instrument_profile")
            staged_main_cif = _get_staged_upload("main_cif")
            reuse_staged_data_file = _get_staged_upload("reuse_diffraction_data")
            reuse_staged_instprm_file = _get_staged_upload("reuse_instrument_profile")
            reuse_staged_main_cif = _get_staged_upload("reuse_main_cif")
            reuse_data_source_mode = REUSE_SAVED_FILE
            reuse_instprm_source_mode = REUSE_SAVED_FILE
            reuse_main_cif_source_mode = REUSE_NO_MAIN_CIF
            reuse_main_cif_available = False
            example_labels = {
                "None": "Upload my data",
                "Reuse Previous Run": "Reuse saved run inputs",
                "TbSSL (CW Demo)": "Diagnostic example: TbSSL neutron CW",
                "LK-99 (TOF Demo)": "Diagnostic example: LK-99 neutron TOF",
            }
            if st.session_state.pop("force_reuse_previous_run_inputs", False):
                st.session_state.dataset_source_mode = "Reuse Previous Run"
            if st.session_state.get("dataset_source_mode") not in example_labels:
                st.session_state.dataset_source_mode = "None"
            example_selection = st.selectbox(
                "Dataset source",
                ["None", "Reuse Previous Run", "TbSSL (CW Demo)", "LK-99 (TOF Demo)"],
                index=0,
                key="dataset_source_mode",
                format_func=lambda choice: example_labels.get(choice, choice),
                help="Diagnostic examples are for checking the installation and pipeline behavior.",
            )
            using_previous_run_inputs = example_selection == "Reuse Previous Run"
            previous_run_entry = None
            previous_run_cfg = None
            if using_previous_run_inputs:
                template_notice = st.session_state.pop("prepare_template_notice", None)
                template_notice_shown = False
                if template_notice:
                    st.success(template_notice)
                    template_notice_shown = True
                reusable_runs_all = cached_reusable_run_configs(str(ACTIVE_RUNS_ROOT))
                if reusable_runs_all:
                    st.caption(
                        f"Choose any reusable saved run in this workspace. "
                        f"{len(reusable_runs_all)} run(s) have available data and instrument files."
                    )
                    reuse_query = st.text_input(
                        "Filter saved runs",
                        key="reused_run_filter",
                        placeholder="Run name, data file, main CIF, Rapid/Full, Xray/Neutron...",
                    )
                    query_tokens = [
                        token.strip().lower()
                        for token in str(reuse_query or "").replace(",", " ").split()
                        if token.strip()
                    ]
                    reusable_runs = [
                        item for item in reusable_runs_all
                        if all(token in str(item.get("search_text", "")).lower() for token in query_tokens)
                    ]
                    if not reusable_runs:
                        st.warning("No saved runs match the current filter.")
                    else:
                        reusable_options = [str(item.get("config_path")) for item in reusable_runs]
                        reusable_by_config = {str(item.get("config_path")): item for item in reusable_runs}
                        preferred_config_path = reusable_options[0]
                        remembered_cfg = st.session_state.get("reused_run_config_path")
                        if remembered_cfg and str(remembered_cfg) in reusable_by_config:
                            preferred_config_path = str(remembered_cfg)
                        if st.session_state.get("reused_run_selector_path") not in reusable_by_config:
                            st.session_state.reused_run_selector_path = preferred_config_path
                        selected_run_config_path = st.selectbox(
                            "Saved run to reuse",
                            reusable_options,
                            format_func=lambda path: _reusable_run_sidebar_label(reusable_by_config[str(path)]),
                            key="reused_run_selector_path",
                            help="Copies the selected run's input files into the new run and reloads its saved setup values.",
                        )
                        previous_run_entry = reusable_by_config[str(selected_run_config_path)]
                        _render_sidebar_summary_rows(
                            [
                                ("Mode", previous_run_entry.get("mode", "-")),
                                ("Radiation", previous_run_entry.get("radiation", "-")),
                                ("Saved", previous_run_entry.get("mtime_text", "-")),
                            ]
                        )

                        previous_cfg_path = Path(previous_run_entry["config_path"])
                        previous_run_cfg = cached_read_yaml_file(str(previous_cfg_path), _path_mtime(previous_cfg_path))
                        if st.session_state.get("reused_run_config_path") != str(previous_cfg_path):
                            _apply_reused_config_to_session(previous_run_cfg)
                            for field in ("reuse_diffraction_data", "reuse_instrument_profile", "reuse_main_cif"):
                                st.session_state.pop(_staged_upload_state_key(field), None)
                            st.session_state.reuse_data_source_mode = REUSE_SAVED_FILE
                            st.session_state.reuse_instprm_source_mode = REUSE_SAVED_FILE
                            st.session_state.reuse_main_cif_source_mode = (
                                REUSE_SAVED_MAIN_CIF if previous_run_entry.get("main_cif") else REUSE_NO_MAIN_CIF
                            )
                            st.session_state.reused_run_config_path = str(previous_cfg_path)
                            st.rerun()
                        if not template_notice_shown:
                            st.info(
                                "Use the selected run as a template. Keep any saved input file or upload a replacement; "
                                "the new run will copy the chosen files into its own `inputs/` folder."
                            )
                        saved_data_name = previous_run_entry.get("data_name") or Path(previous_run_entry["data_path"]).name
                        saved_inst_name = previous_run_entry.get("instprm_name") or Path(previous_run_entry["instprm_path"]).name
                        previous_main_path = Path(str(previous_run_entry.get("main_cif") or ""))
                        previous_has_main_cif = bool(previous_run_entry.get("main_cif") and previous_main_path.exists())
                        saved_main_name = (
                            previous_run_entry.get("main_cif_name")
                            or (previous_main_path.name if previous_has_main_cif else "none")
                        )

                        if st.session_state.get("reuse_data_source_mode") not in (REUSE_SAVED_FILE, REUSE_REPLACEMENT_FILE):
                            st.session_state.reuse_data_source_mode = REUSE_SAVED_FILE
                        if st.session_state.get("reuse_instprm_source_mode") not in (REUSE_SAVED_FILE, REUSE_REPLACEMENT_FILE):
                            st.session_state.reuse_instprm_source_mode = REUSE_SAVED_FILE
                        main_cif_options = (
                            [REUSE_SAVED_MAIN_CIF, REUSE_REPLACEMENT_FILE, REUSE_NO_MAIN_CIF]
                            if previous_has_main_cif
                            else [REUSE_NO_MAIN_CIF, REUSE_REPLACEMENT_FILE]
                        )
                        if st.session_state.get("reuse_main_cif_source_mode") not in main_cif_options:
                            st.session_state.reuse_main_cif_source_mode = main_cif_options[0]

                        st.markdown("**Input file choices for this new run**")
                        st.caption(
                            "Mix and match saved and new files: keep the old diffraction data, replace only "
                            "the instrument profile, or remove/replace the optional main CIF before launching."
                        )
                        with st.container(border=True):
                            st.caption(f"Saved diffraction data: `{saved_data_name}`")
                            reuse_data_source_mode = st.radio(
                                "Diffraction data source",
                                [REUSE_SAVED_FILE, REUSE_REPLACEMENT_FILE],
                                horizontal=True,
                                key="reuse_data_source_mode",
                                format_func=lambda choice: (
                                    "Keep saved diffraction data"
                                    if choice == REUSE_SAVED_FILE
                                    else "Upload new diffraction data"
                                ),
                                help="Keep the selected run's diffraction data, or replace only this file for the new run.",
                            )
                            if reuse_data_source_mode == REUSE_REPLACEMENT_FILE:
                                replacement_data_file = st.file_uploader(
                                    "Replacement diffraction data",
                                    type=SUPPORTED_DIFFRACTION_UPLOAD_EXTENSIONS,
                                    key="reuse_diffraction_data_upload",
                                    help="This replaces only the diffraction data for the new run.",
                                )
                                reuse_staged_data_file = _stage_uploaded_file("reuse_diffraction_data", replacement_data_file)
                                _render_staged_upload_status("reuse_diffraction_data", "replacement diffraction data")
                            else:
                                st.caption("This run will reuse the saved diffraction data.")

                            st.divider()
                            st.caption(f"Saved instrument profile: `{saved_inst_name}`")
                            reuse_instprm_source_mode = st.radio(
                                "Instrument profile source",
                                [REUSE_SAVED_FILE, REUSE_REPLACEMENT_FILE],
                                horizontal=True,
                                key="reuse_instprm_source_mode",
                                format_func=lambda choice: (
                                    "Keep saved instrument profile"
                                    if choice == REUSE_SAVED_FILE
                                    else "Upload new instrument profile"
                                ),
                                help="Keep the selected run's instrument profile, or replace only this file for the new run.",
                            )
                            if reuse_instprm_source_mode == REUSE_REPLACEMENT_FILE:
                                replacement_instprm_file = st.file_uploader(
                                    "Replacement instrument profile (.instprm, .prm, .inst, or .ins)",
                                    type=SUPPORTED_INSTRUMENT_UPLOAD_EXTENSIONS,
                                    key="reuse_instrument_profile_upload",
                                    help="Legacy instrument files are normalized to `.instprm` when the run starts.",
                                )
                                reuse_staged_instprm_file = _stage_uploaded_file("reuse_instrument_profile", replacement_instprm_file)
                                _render_staged_upload_status("reuse_instrument_profile", "replacement instrument profile")
                            else:
                                st.caption("This run will reuse the saved instrument profile.")

                            st.divider()
                            st.caption(f"Saved known/main phase CIF: `{saved_main_name}`")
                            reuse_main_cif_source_mode = st.radio(
                                "Known/main phase CIF source",
                                main_cif_options,
                                horizontal=True,
                                key="reuse_main_cif_source_mode",
                                format_func=lambda choice: (
                                    "Keep saved main CIF"
                                    if choice == REUSE_SAVED_MAIN_CIF
                                    else "Upload new main CIF"
                                    if choice == REUSE_REPLACEMENT_FILE
                                    else "Run without main CIF"
                                ),
                                help="Keep, replace, or remove the known/main phase CIF for the new run.",
                            )
                            if reuse_main_cif_source_mode == REUSE_REPLACEMENT_FILE:
                                replacement_main_cif = st.file_uploader(
                                    "Replacement known/main phase CIF",
                                    type=["cif"],
                                    key="reuse_main_cif_upload",
                                    help="This replaces only the optional main-phase CIF for the new run.",
                                )
                                reuse_staged_main_cif = _stage_uploaded_file("reuse_main_cif", replacement_main_cif)
                                _render_staged_upload_status("reuse_main_cif", "replacement main CIF")
                            elif reuse_main_cif_source_mode == REUSE_SAVED_MAIN_CIF:
                                st.caption("This run will reuse the saved known/main phase CIF.")
                            else:
                                st.caption("This run will not use a known/main phase CIF.")

                        reuse_main_cif_available = bool(
                            (reuse_main_cif_source_mode == REUSE_SAVED_MAIN_CIF and previous_has_main_cif)
                            or (
                                reuse_main_cif_source_mode == REUSE_REPLACEMENT_FILE
                                and _staged_upload_is_nonempty(reuse_staged_main_cif)
                            )
                        )
                        if st.button("Reload setup values from selected run", width="stretch"):
                            _apply_reused_config_to_session(previous_run_cfg)
                            st.rerun()
                else:
                    st.warning("No saved run with reusable data and instrument files was found in this workspace.")
            rn_col, rn_btn_col = st.columns([0.70, 0.30])
            with rn_col:
                run_name = st.text_input(
                    "Run name",
                    key="custom_run_name",
                    on_change=mark_run_name_manual,
                    help="Used for the output folder under this workspace's runs/. Use a new name for each attempt.",
                )
            with rn_btn_col:
                st.markdown("<div style='height: 1.75rem;'></div>", unsafe_allow_html=True)
                if st.button("New name", width="stretch"):
                    st.session_state.pending_run_name_reset = True
                    st.rerun()
            light_calibration_enabled = False

            if example_selection != "None" and not using_previous_run_inputs:
                if example_selection == "TbSSL (CW Demo)":
                    st.info("Using bundled TbSSL diagnostic dataset. Data, instrument profile, main CIF, and chemistry policy are pre-configured.")
                    st.code("Sample elements: Tb, Be, Ge, O\nSample can/environment: Al\nMain CIF: TbSSL.cif", language="text")
                    allowed_elements_str = "Tb, Be, Ge, O"
                    sample_env_elements_str = "Al"
                    inst_mode = "CW"
                else: # LK-99
                     st.info("Using bundled LK-99 diagnostic dataset. Data, instrument profile, main CIF, and chemistry policy are pre-configured.")
                     st.code("Sample elements: Pb, P, Cu, O, S\nSample can/environment: none\nMain CIF: LK99.cif", language="text")
                     allowed_elements_str = "Pb, P, Cu, O, S"
                     sample_env_elements_str = ""
                     inst_mode = "TOF"

                data_file, instprm_file, main_cif = None, None, None
                builtin_instprm_key = None
            elif using_previous_run_inputs:
                data_file, instprm_file, main_cif = None, None, None
                builtin_instprm_key = None
                if previous_run_cfg:
                    dataset_cfg = (previous_run_cfg.get("datasets") or [{}])[0] or {}
                    mode_value = st.session_state.get("reused_instrument_mode") or str(dataset_cfg.get("mode") or "auto")
                    inst_mode = "Auto" if str(mode_value).lower() == "auto" else str(mode_value).upper()
                else:
                    inst_mode = "Auto"
            else:
                data_file, instprm_file, main_cif = None, None, None
                builtin_instprm_key = None
                data_file = st.file_uploader(
                    "Diffraction data",
                    type=SUPPORTED_DIFFRACTION_UPLOAD_EXTENSIONS,
                    key="diffraction_data_upload",
                    help="Powder diffraction pattern file to fit and search residual peaks against.",
                )
                staged_data_file = _stage_uploaded_file("diffraction_data", data_file)
                _render_staged_upload_status("diffraction_data", "diffraction data")
                if IS_XRAY:
                    profile_choice = st.radio(
                        "Calibration / instrument profile",
                        ["Upload Instrument Params", LAB_XRAY_PRESET["ui_label"]],
                        help="Use a calibrated instrument file when available. The built-in preset is an approximate Cu Kalpha lab PXRD profile for CW screening.",
                    )
                    if profile_choice == LAB_XRAY_PRESET["ui_label"]:
                        builtin_instprm_key = DEFAULT_LAB_XRAY_PRESET_KEY
                        st.info("Using GSAS-II's built-in CuKa lab PXRD profile. This mode is approximate and is best for quick CW lab-XRD screening.")
                        st.caption("A real `.instprm` file will be generated automatically inside the run inputs folder so the pipeline remains fully file-based.")
                    else:
                        instprm_file = st.file_uploader(
                            "Instrument profile (.instprm, .prm, .inst, or .ins)",
                            type=SUPPORTED_INSTRUMENT_UPLOAD_EXTENSIONS,
                            key="instrument_profile_upload",
                            help=(
                                "GSAS-II `.instprm` or legacy GSAS/EXPGUI `.prm` / `.inst` / `.ins` "
                                "instrument files. Legacy formats are normalized to `.instprm` for the run."
                            ),
                        )
                        staged_instprm_file = _stage_uploaded_file("instrument_profile", instprm_file)
                        _render_staged_upload_status("instrument_profile", "instrument profile")
                else:
                    instprm_file = st.file_uploader(
                        "Instrument profile (.instprm, .prm, .inst, or .ins)",
                        type=SUPPORTED_INSTRUMENT_UPLOAD_EXTENSIONS,
                        key="instrument_profile_upload",
                        help=(
                            "GSAS-II `.instprm` or legacy GSAS/EXPGUI `.prm` / `.inst` / `.ins` "
                            "instrument files. Legacy formats are normalized to `.instprm` for the run."
                        ),
                    )
                    staged_instprm_file = _stage_uploaded_file("instrument_profile", instprm_file)
                    _render_staged_upload_status("instrument_profile", "instrument profile")
                main_cif = st.file_uploader("Known/main phase CIF (optional)", type=["cif"], key="main_cif_upload")
                staged_main_cif = _stage_uploaded_file("main_cif", main_cif)
                _render_staged_upload_status("main_cif", "main CIF")

                if builtin_instprm_key:
                    inst_mode = "CW"
                    st.markdown("**Pattern geometry**")
                    st.info("CW (fixed by built-in CuKa lab PXRD preset)")
                else:
                    inst_mode = st.radio(
                        "Pattern geometry",
                        ["Auto", "CW", "TOF"],
                        horizontal=True,
                        help="Use Auto unless RADAR-PD cannot infer whether the pattern is constant-wavelength or time-of-flight.",
                    )

                if IS_XRAY and (main_cif is not None or staged_main_cif is not None):
                    light_calibration_enabled = st.checkbox(
                        "Use Light PXRD Calibration",
                        value=True,
                        key="light_pxrd_calibration_enabled",
                        help=(
                            "Before discovery, run a conservative main-phase-only refinement to "
                            "optimize `Zero + U/V/W` and reuse the generated `.instprm` for the rest of the run."
                        ),
                    )
                    if light_calibration_enabled:
                        st.info(
                            "Main phase provided: RADAR-PD will run a light PXRD calibration "
                            "before discovery to refine `Zero + U/V/W` and reuse the generated `.instprm`."
                        )
                    else:
                        st.caption("Light PXRD calibration is disabled for this run; the original instrument file will be used throughout.")

        # --- 4. CHEMISTRY POLICY ---
        with st.expander("4. Chemistry Policy", expanded=True):
            st.caption("Define which elements belong to the sample and which may come from the sample can or environment.")
            if example_selection != "None" and not using_previous_run_inputs:
                st.text_input("Sample elements", value=allowed_elements_str, disabled=True)
                st.text_input("Sample can / environment", value=sample_env_elements_str or "None", disabled=True)
            else:
                chem_c1, chem_c2 = st.columns(2)
                with chem_c1:
                    allowed_elements_str = st.text_input(
                        "Sample elements",
                        key="allowed_elements_input",
                        placeholder="e.g. Tb, Be, Ge, O",
                        help=(
                            "Comma-separated elements expected in the sample chemistry. "
                            "Leave blank only when using a custom mini/augmented library and you want to allow all elements in that library."
                        ),
                    )
                    if ACTIVE_DB_KIND != "original" and ACTIVE_DB_EXISTS:
                        derived_custom_elements = _derive_allowed_elements_from_active_db()
                        if derived_custom_elements:
                            st.caption(
                                f"Blank field will allow all {len(derived_custom_elements)} elements present in the selected custom library."
                            )
                        else:
                            st.caption("Blank-field fallback is unavailable because the selected custom library could not be inspected.")
                with chem_c2:
                    sample_env_elements_str = st.text_input(
                        "Sample can / environment",
                        key="sample_env_elements_input",
                        placeholder="e.g. Al, V",
                        help="Elements that may appear from holders, cans, windows, or sample-environment hardware.",
                    )

            sample_preview = parse_element_list(allowed_elements_str)
            env_preview = parse_element_list(sample_env_elements_str)
            bad_elements = invalid_element_tokens(sample_preview + env_preview)
            if bad_elements:
                st.warning(f"Unrecognized element symbols: {', '.join(bad_elements)}")
            if env_preview:
                st.caption(
                    "Sample-can policy: pure environment phases are allowed; environment elements mixed "
                    "with sample chemistry are blocked."
                )

        reference_phase_exclusion_config = {"enabled": False, "presets": []}

        # --- 5. PATTERN REGIONS ---
        with st.expander("5. Pattern Regions", expanded=False):

            st.divider()
            axis_label = _limits_axis_label(
                source_label=current_source,
                example_selection=example_selection,
                builtin_instprm_key=builtin_instprm_key,
                instrument_mode=inst_mode,
            )
            st.markdown("**Ignored Regions**")
            st.caption(
                "Optional fit exclusions in the histogram's native x-axis. "
                "These regions are ignored during refinement and downstream residual passes."
            )
            st.caption(f"Units: {axis_label}")

            se_tokens = [element.lower() for element in parse_element_list(sample_env_elements_str)]
            default_ref_presets = []
            if "al" in se_tokens:
                default_ref_presets.append("Al_fcc")
            if "cu" in se_tokens:
                default_ref_presets.append("Cu_fcc")
            if "v" in se_tokens:
                default_ref_presets.append("V_bcc")
            if "reference_mask_presets" not in st.session_state:
                st.session_state.reference_mask_presets = default_ref_presets

            auto_reference_mask_enabled = st.checkbox(
                "Auto-mask sample can/reference peaks",
                value=False,
                key="auto_reference_phase_mask_enabled",
                help=(
                    "Generate ignored regions around known Al, Cu, or V container/reference Bragg peaks. "
                    "This is separate from the Sample can/environment chemistry filter."
                ),
            )
            ref_c1, ref_c2 = st.columns(2)
            with ref_c1:
                reference_mask_presets = st.multiselect(
                    "Can/reference presets",
                    ["Al_fcc", "Cu_fcc", "V_bcc"],
                    key="reference_mask_presets",
                    disabled=not auto_reference_mask_enabled,
                    help="Preset crystal structures used to calculate Bragg peak positions for masking.",
                )
            with ref_c2:
                reference_mask_window_mode_label = st.selectbox(
                    "Mask width",
                    ["Auto from instrument profile", "Fixed half-width"],
                    index=0,
                    disabled=not auto_reference_mask_enabled,
                    help="Auto estimates peak width from the instrument profile; fixed uses one user-supplied half-width.",
                )
            reference_mask_window_mode = "fixed" if reference_mask_window_mode_label == "Fixed half-width" else "auto"
            is_tof_axis = "TOF" in axis_label
            default_min_ref_width = 75.0 if is_tof_axis else 0.35
            default_max_ref_width = 750.0 if is_tof_axis else 2.00
            default_fixed_ref_width = 200.0 if is_tof_axis else 0.75
            default_zero_tolerance = 25.0 if is_tof_axis else 0.05
            width_step = 5.0 if is_tof_axis else 0.05
            width_format = "%.1f" if is_tof_axis else "%.2f"

            width_c1, width_c2, width_c3 = st.columns(3)
            if reference_mask_window_mode == "fixed":
                with width_c1:
                    reference_mask_fixed_half_width = st.number_input(
                        "Fixed half-width",
                        0.01,
                        10000.0,
                        default_fixed_ref_width,
                        step=width_step,
                        format=width_format,
                        disabled=not auto_reference_mask_enabled,
                        help=f"Fixed half-width of each generated ignored region in {axis_label}.",
                    )
                reference_mask_fwhm_factor = 6.0
                reference_mask_fractional_d_tolerance = 0.003
                reference_mask_zero_tolerance = default_zero_tolerance
                reference_mask_min_half_width = default_min_ref_width
                reference_mask_max_half_width = default_max_ref_width
            else:
                with width_c1:
                    reference_mask_fwhm_factor = st.number_input(
                        "Profile FWHM factor",
                        0.5,
                        20.0,
                        6.0,
                        step=0.5,
                        format="%.1f",
                        disabled=not auto_reference_mask_enabled,
                        help="Generated half-width includes profile FWHM, position tolerance, and min/max clamps.",
                    )
                with width_c2:
                    reference_mask_fractional_d_tolerance_pct = st.number_input(
                        "d tolerance (%)",
                        0.0,
                        5.0,
                        0.3,
                        step=0.1,
                        format="%.1f",
                        disabled=not auto_reference_mask_enabled,
                        help="Allows shifted peaks from alloying, temperature, or reference-cell mismatch.",
                    )
                    reference_mask_fractional_d_tolerance = float(reference_mask_fractional_d_tolerance_pct) / 100.0
                with width_c3:
                    reference_mask_zero_tolerance = st.number_input(
                        "Zero tolerance",
                        0.0,
                        10000.0,
                        default_zero_tolerance,
                        step=width_step,
                        format=width_format,
                        disabled=not auto_reference_mask_enabled,
                        help=f"Extra center-position tolerance in {axis_label} for residual zero/calibration mismatch.",
                    )
                minmax_c1, minmax_c2 = st.columns(2)
                with minmax_c1:
                    reference_mask_min_half_width = st.number_input(
                        "Min half-width",
                        0.01,
                        10000.0,
                        default_min_ref_width,
                        step=width_step,
                        format=width_format,
                        disabled=not auto_reference_mask_enabled,
                        help=f"Minimum generated half-width in {axis_label}.",
                    )
                with minmax_c2:
                    reference_mask_max_half_width = st.number_input(
                        "Max half-width",
                        0.01,
                        10000.0,
                        default_max_ref_width,
                        step=width_step,
                        format=width_format,
                        disabled=not auto_reference_mask_enabled,
                        help=f"Maximum generated half-width in {axis_label}.",
                    )
                reference_mask_fixed_half_width = default_fixed_ref_width

            kbeta_c1, kbeta_c2 = st.columns(2)
            with kbeta_c1:
                reference_mask_include_kbeta = st.checkbox(
                    "Also mask Cu K-beta positions",
                    value=bool(IS_XRAY),
                    key="reference_phase_mask_include_kbeta",
                    disabled=not auto_reference_mask_enabled,
                    help="For Cu-anode lab PXRD, also mask K-beta companion peaks from the selected phases.",
                )
            with kbeta_c2:
                st.caption("Cu K-beta uses the same width policy as the selected reference peaks.")
            if auto_reference_mask_enabled:
                reference_phase_exclusion_config = {
                    "enabled": True,
                    "presets": list(reference_mask_presets),
                    "window_mode": reference_mask_window_mode,
                    "include_cu_kbeta": bool(reference_mask_include_kbeta),
                }
                if reference_mask_window_mode == "fixed":
                    if is_tof_axis:
                        reference_phase_exclusion_config["half_width_tof"] = float(reference_mask_fixed_half_width)
                    else:
                        reference_phase_exclusion_config["half_width_deg"] = float(reference_mask_fixed_half_width)
                else:
                    reference_phase_exclusion_config["fwhm_factor"] = float(reference_mask_fwhm_factor)
                    reference_phase_exclusion_config["fractional_d_tolerance"] = float(reference_mask_fractional_d_tolerance)
                    if is_tof_axis:
                        reference_phase_exclusion_config.update({
                            "min_half_width_tof": float(reference_mask_min_half_width),
                            "max_half_width_tof": float(reference_mask_max_half_width),
                            "zero_tolerance_tof": float(reference_mask_zero_tolerance),
                            "zero_tolerance_deg": 0.05,
                            "min_half_width_deg": 0.35,
                            "max_half_width_deg": 2.00,
                        })
                    else:
                        reference_phase_exclusion_config.update({
                            "min_half_width_deg": float(reference_mask_min_half_width),
                            "max_half_width_deg": float(reference_mask_max_half_width),
                            "zero_tolerance_deg": float(reference_mask_zero_tolerance),
                            "zero_tolerance_tof": 25.0,
                            "min_half_width_tof": 75.0,
                            "max_half_width_tof": 750.0,
                        })

            fit_window_enabled = st.checkbox(
                "Use fit window override",
                key="fit_window_override_enabled",
                help=f"Restrict refinement to a single outer range in {axis_label}. "
                     "Ignored regions are applied inside this range.",
            )
            fit_window = None
            fit_window_input_errors: list[str] = []
            if fit_window_enabled:
                fit_c1, fit_c2 = st.columns(2)
                with fit_c1:
                    st.text_input("Fit Window Lower", key="fit_window_lower", help=f"Lower bound in {axis_label}.")
                with fit_c2:
                    st.text_input("Fit Window Upper", key="fit_window_upper", help=f"Upper bound in {axis_label}.")
                fit_window, fit_window_input_errors = _parse_fit_window_inputs(
                    st.session_state.get("fit_window_lower", ""),
                    st.session_state.get("fit_window_upper", ""),
                )
                if fit_window:
                    fcol1, fcol2 = st.columns([0.72, 0.28])
                    with fcol1:
                        st.info(f"Fit window override: {_format_region(fit_window[0], fit_window[1], axis_label)}")
                    with fcol2:
                        if st.button("Clear fit window", key="clear_fit_window", width="stretch"):
                            st.session_state.pending_fit_window_reset = True
                            st.rerun()

            existing_excluded_rows = _coerce_excluded_region_rows(
                st.session_state.get("excluded_regions_editor_buffer")
            )
            deduped_excluded_rows = []
            seen_excluded_regions = set()
            for row in existing_excluded_rows:
                try:
                    start = float(row.get("Start"))
                    end = float(row.get("End"))
                except Exception:
                    deduped_excluded_rows.append(row)
                    continue
                lo, hi = sorted((start, end))
                key = (round(lo, 8), round(hi, 8))
                if key in seen_excluded_regions:
                    continue
                seen_excluded_regions.add(key)
                deduped_excluded_rows.append({"Start": lo, "End": hi})
            if deduped_excluded_rows != existing_excluded_rows:
                existing_excluded_rows = deduped_excluded_rows
                st.session_state.excluded_regions_rows = existing_excluded_rows
                st.session_state.excluded_regions_editor_buffer = existing_excluded_rows
            excluded_region_pairs, excluded_region_input_errors = _parse_excluded_region_rows(existing_excluded_rows)
            if excluded_region_pairs:
                st.markdown("Current ignored regions:")
                for idx, pair in enumerate(excluded_region_pairs):
                    lo, hi = float(min(pair)), float(max(pair))
                    rcol1, rcol2, rcol3 = st.columns([0.62, 0.18, 0.20])
                    with rcol1:
                        st.caption(f"{idx + 1}. {_format_region(lo, hi, axis_label)}")
                    with rcol2:
                        if st.button("Edit", key=f"edit_excluded_region_{idx}", width="stretch"):
                            st.session_state.excluded_region_edit_index = idx
                            st.session_state.excluded_region_form_start = f"{lo:.6f}"
                            st.session_state.excluded_region_form_end = f"{hi:.6f}"
                            st.rerun()
                    with rcol3:
                        if st.button("Remove", key=f"remove_excluded_region_{idx}", width="stretch"):
                            st.session_state["pending_excluded_region_remove"] = idx
                            st.rerun()
                rclear1, rclear2 = st.columns([0.65, 0.35])
                with rclear2:
                    if st.button("Clear all", key="clear_all_excluded_regions", width="stretch"):
                        st.session_state.excluded_regions_rows = []
                        st.session_state.excluded_regions_editor_buffer = []
                        st.session_state.excluded_region_edit_index = None
                        st.session_state.excluded_region_form_start = ""
                        st.session_state.excluded_region_form_end = ""
                        st.rerun()
                st.info(
                    f"{len(excluded_region_pairs)} ignored region(s) configured for this run in {axis_label}."
                )
            else:
                st.caption("No ignored regions configured.")
            edit_idx = st.session_state.get("excluded_region_edit_index")
            editing_region = isinstance(edit_idx, int) and 0 <= edit_idx < len(existing_excluded_rows)
            editor_title = "Edit ignored region" if editing_region else "Add ignored region"
            with st.expander(editor_title, expanded=editing_region):
                with st.form("excluded_region_form", clear_on_submit=False):
                    ec1, ec2 = st.columns(2)
                    with ec1:
                        start_text = st.text_input(
                            "Start",
                            key="excluded_region_form_start",
                            placeholder=f"Start ({axis_label})",
                            help=f"Lower bound of the ignored region in {axis_label}.",
                        )
                    with ec2:
                        end_text = st.text_input(
                            "End",
                            key="excluded_region_form_end",
                            placeholder=f"End ({axis_label})",
                            help=f"Upper bound of the ignored region in {axis_label}.",
                        )
                    ac1, ac2 = st.columns(2)
                    with ac1:
                        submit_region = st.form_submit_button(
                            "Update region" if editing_region else "Add region",
                            width="stretch",
                        )
                    with ac2:
                        cancel_region = st.form_submit_button(
                            "Cancel edit" if editing_region else "Clear entry",
                            width="stretch",
                        )

                if cancel_region:
                    st.session_state.excluded_region_edit_index = None
                    st.session_state.pending_excluded_region_form_reset = True
                    st.rerun()

                if submit_region:
                    start_text = str(start_text or "").strip()
                    end_text = str(end_text or "").strip()
                    if not start_text or not end_text:
                        st.error("Ignored region entry requires both Start and End values.")
                    else:
                        try:
                            start_val = float(start_text.replace(",", ""))
                            end_val = float(end_text.replace(",", ""))
                        except Exception:
                            st.error("Ignored region Start and End must be numeric.")
                        else:
                            lo, hi = sorted((start_val, end_val))
                            next_rows = _coerce_excluded_region_rows(existing_excluded_rows)
                            region_row = {"Start": lo, "End": hi}
                            duplicate = False
                            if editing_region:
                                next_rows[edit_idx] = region_row
                            else:
                                for existing_row in next_rows:
                                    try:
                                        existing_lo, existing_hi = sorted((
                                            float(existing_row.get("Start")),
                                            float(existing_row.get("End")),
                                        ))
                                    except Exception:
                                        continue
                                    if abs(existing_lo - lo) < 1e-8 and abs(existing_hi - hi) < 1e-8:
                                        duplicate = True
                                        break
                                if not duplicate:
                                    next_rows.append(region_row)
                            st.session_state.excluded_regions_rows = next_rows
                            st.session_state.excluded_regions_editor_buffer = next_rows
                            st.session_state.excluded_region_edit_index = None
                            st.session_state.pending_excluded_region_form_reset = True
                            if duplicate and hasattr(st, "toast"):
                                st.toast("That ignored region is already configured.")
                            st.rerun()

            excluded_region_pairs, excluded_region_input_errors = _parse_excluded_region_rows(
                st.session_state.get("excluded_regions_rows")
            )

        # --- 6. BACKGROUND CORRECTION ---
        with st.expander("6. Background Correction", expanded=True):
            bg_c1, bg_c2 = st.columns(2)
            with bg_c1:
                bg_mode_options = ["Auto Fixed Points", "Function"]
                if st.session_state.get("bg_mode_label") not in bg_mode_options:
                    st.session_state.bg_mode_label = "Auto Fixed Points"
                bg_mode_label = st.selectbox(
                    "Correction mode",
                    bg_mode_options,
                    key="bg_mode_label",
                    help=(
                        "Auto Fixed Points estimates low-envelope background points before GSAS-II fits the background. "
                        "Function uses the selected GSAS-II background function directly."
                    ),
                )
                bg_mode = "auto_fixed_points" if bg_mode_label == "Auto Fixed Points" else "function"
                bg_type_options = ["chebyschev-1", "log interpolate", "cosine", "exponential"]
                if st.session_state.get("bg_type") not in bg_type_options:
                    st.session_state.bg_type = "chebyschev-1"
                bg_type = st.selectbox(
                    "Background function",
                    bg_type_options,
                    key="bg_type",
                    help="GSAS-II background function type used during refinement.",
                )
            with bg_c2:
                if "bg_terms" not in st.session_state:
                    st.session_state.bg_terms = 6
                bg_terms = st.number_input(
                    "Background terms",
                    min_value=1,
                    max_value=36,
                    key="bg_terms",
                    help="Number of coefficients used for background refinement.",
                )
                if bg_mode == "auto_fixed_points":
                    st.caption("Uses a scale-aware low-envelope picker to seed GSAS-II fixed background points.")

        # --- 7. MAGNETIC ORDERING PRECHECK ---
        main_phase_available_for_precheck = bool(
            (example_selection in {"TbSSL (CW Demo)", "LK-99 (TOF Demo)"} and not using_previous_run_inputs)
            or (using_previous_run_inputs and reuse_main_cif_available)
            or ((not using_previous_run_inputs) and (staged_main_cif or main_cif))
        )
        with st.expander("7. Magnetic Ordering Precheck", expanded=False):
            st.caption(
                "Optional neutron-only check before impurity search. It tests whether positive residual peaks "
                "from the main-phase refinement can be indexed by a single commensurate magnetic k-vector."
            )
            magnetic_can_run = (not IS_XRAY) and main_phase_available_for_precheck
            if IS_XRAY:
                st.info("Magnetic residual indexing is only offered for neutron diffraction runs.")
            elif not main_phase_available_for_precheck:
                st.warning("Upload or reuse a known/main phase CIF to enable magnetic k-vector indexing.")
            magnetic_precheck_enabled = st.checkbox(
                "Check magnetic ordering possibility before phase detection",
                value=bool(st.session_state.get("magnetic_precheck_enabled", False)),
                key="magnetic_precheck_enabled",
                disabled=not magnetic_can_run,
                help=(
                    "Runs after the nuclear/main-phase fit and before either Full RADAR-PD or Rapid Hypothesis Mode. "
                    "It does not solve a magnetic structure; it ranks whether the residual peak positions are consistent "
                    "with a commensurate propagation vector."
                ),
            )
            if not magnetic_can_run:
                magnetic_precheck_enabled = False
            if magnetic_precheck_enabled:
                m1, m2 = st.columns(2)
                with m1:
                    magnetic_precheck_q_max = st.number_input(
                        "Magnetic check Q max",
                        1.0,
                        12.0,
                        float(st.session_state.get("magnetic_precheck_q_max", 6.0)),
                        step=0.5,
                        key="magnetic_precheck_q_max",
                        help="Magnetic scattering is usually most diagnostic at low Q. Start near 6 1/A.",
                    )
                with m2:
                    magnetic_precheck_denominators = st.text_input(
                        "k-vector denominators",
                        value=str(st.session_state.get("magnetic_precheck_denominators", "2,3")),
                        key="magnetic_precheck_denominators",
                        help="Comma-separated rational denominators for commensurate k grids, e.g. 2,3 or 2,3,4.",
                    )
                st.info(
                    "Output is an evidence report: weak, moderate, or strong indexing consistency. "
                    "RADAR-PD will still continue to impurity search because magnetic peaks and impurity phases can coexist."
                )
            else:
                magnetic_precheck_q_max = float(st.session_state.get("magnetic_precheck_q_max", 6.0))
                magnetic_precheck_denominators = str(st.session_state.get("magnetic_precheck_denominators", "2,3"))

        # --- 8. ANALYSIS MODE ---
        with st.expander("8. Analysis Mode", expanded=True):
            selected_run_mode = ""
            selected_run_value = st.session_state.get("selected_run_dir")
            viewing_saved_result = bool(selected_run_value and not st.session_state.get("run_active"))
            if selected_run_value:
                try:
                    selected_run_mode = _run_analysis_mode_from_dir(Path(selected_run_value))
                except Exception:
                    selected_run_mode = ""
            editing_saved_run_template = st.session_state.get("dataset_source_mode") == "Reuse Previous Run"
            if (
                viewing_saved_result
                and not editing_saved_run_template
                and selected_run_mode in {"Full RADAR-PD", "Rapid Hypothesis Mode"}
            ):
                st.session_state.analysis_mode = selected_run_mode
            analysis_mode = st.radio(
                "Analysis path",
                ["Full RADAR-PD", "Rapid Hypothesis Mode"],
                key="analysis_mode",
                help=(
                    "Full RADAR-PD runs the established ML-guided GSAS-II pipeline. "
                    "Rapid Hypothesis Mode uses coarse matching, lattice nudging, high-resolution pattern scoring, and final refinement ranking."
                ),
            )
            previous_analysis_mode = st.session_state.get("_workspace_analysis_mode")
            mode_changed = previous_analysis_mode is not None and previous_analysis_mode != analysis_mode
            selected_mode_mismatch = bool(selected_run_mode and selected_run_mode != analysis_mode)
            if selected_mode_mismatch:
                if selected_run_value and not viewing_saved_result:
                    _clear_selected_run_context(suppress_latest_autoload=True)
                if not viewing_saved_result:
                    st.session_state.active_run_view = "Run Monitor"
                st.session_state._workspace_analysis_mode = analysis_mode
                if selected_run_value and not viewing_saved_result:
                    st.rerun()
            elif mode_changed and not viewing_saved_result:
                st.session_state.active_run_view = "Run Monitor"
                st.session_state._workspace_analysis_mode = analysis_mode
            st.session_state._workspace_analysis_mode = analysis_mode
            if analysis_mode == "Full RADAR-PD":
                st.caption(
                    "Use this for rigorous sequential impurity discovery and final quantitative refinement."
                )
            else:
                st.caption(
                    "Use this for fast hypothesis generation and inspection. The result view emphasizes ranked phase sets, "
                    "same-family variants, and targeted final refinement."
                )
                rh1, rh2 = st.columns(2)
                with rh1:
                    rapid_hypothesis_phase_count = st.number_input(
                        "Candidate phases per hypothesis",
                        1,
                        5,
                        int(st.session_state.rapid_hypothesis_phase_count),
                        key="rapid_hypothesis_phase_count",
                        help=(
                            "Maximum number of candidate phases in each rapid hypothesis. "
                            "If a known/main CIF is supplied, these are impurity candidates; "
                            "without a main CIF, rapid mode uses this as the total phase count to explain the pattern."
                        ),
                    )
                    rapid_stage_output_limit = st.number_input(
                        "Hypotheses shown per stage",
                        3,
                        50,
                        int(st.session_state.rapid_stage_output_limit),
                        key="rapid_stage_output_limit",
                        help="Number of ranked hypotheses to keep visible in each rapid result stage.",
                    )
                    rapid_gsas_validation_limit = st.number_input(
                        "Final refinement checks",
                        1,
                        20,
                        int(st.session_state.rapid_gsas_validation_limit),
                        key="rapid_gsas_validation_limit",
                        help=(
                            "Number of final hypotheses to send to targeted refinement ranking. "
                            "Use around 10 when weak impurity phases may be hidden below close histogram matches."
                        ),
                    )
                with rh2:
                    rapid_gsas_parallel_workers = st.number_input(
                        "Final refinement workers",
                        1,
                        8,
                        int(st.session_state.rapid_gsas_parallel_workers),
                        key="rapid_gsas_parallel_workers",
                        help=(
                            "Number of GSAS-II refinement processes to run in parallel during the final rapid ranking. "
                            "This VM has 16 vCPUs; 4 is a safe default, 6 can be tried for faster runs if memory remains stable."
                        ),
                    )
                    rapid_enable_family_variants = st.checkbox(
                        "Show same-family variants",
                        value=bool(st.session_state.rapid_enable_family_variants),
                        key="rapid_enable_family_variants",
                        help="Keep near-duplicate formula/space-group/profile variants available for manual swap and targeted refinement.",
                    )
                    rapid_final_polish_enabled = st.checkbox(
                        "Deep final cell polish",
                        value=bool(st.session_state.rapid_final_polish_enabled),
                        key="rapid_final_polish_enabled",
                        help=(
                            "Run the slower transactional cell-polish check after rapid final ranking. "
                            "Leave this off for fast screening; enable it only for a small number of final hypotheses."
                        ),
                    )
                st.info(
                    "Rapid mode still uses the same required setup inputs above: source, library, diffraction data, "
                    "instrument profile, optional main CIF, element policy, masks, and background."
                )
            st.markdown("**Main phase protection**")
            main_phase_shadow_filter_enabled = st.checkbox(
                "Filter candidates that only reproduce main-phase peaks",
                key="main_phase_shadow_filter_enabled",
                help=(
                    "After lattice nudging, remove candidate phases whose strongest peaks mostly coincide with "
                    "the strongest fitted main-phase peaks and have little independent residual support. "
                    "This applies to both Full RADAR-PD and Rapid Hypothesis Mode."
                ),
            )
            if not main_phase_available_for_precheck:
                st.caption("This filter becomes active when a known/main CIF is supplied or reused.")

        analysis_mode = st.session_state.get("analysis_mode", "Full RADAR-PD")
        main_phase_shadow_filter_enabled = bool(st.session_state.get("main_phase_shadow_filter_enabled", True))
        rapid_hypothesis_phase_count = int(st.session_state.get("rapid_hypothesis_phase_count", 3))
        rapid_stage_output_limit = int(st.session_state.get("rapid_stage_output_limit", 10))
        rapid_gsas_validation_limit = int(st.session_state.get("rapid_gsas_validation_limit", 10))
        rapid_gsas_parallel_workers = int(st.session_state.get("rapid_gsas_parallel_workers", 4))
        rapid_enable_family_variants = bool(st.session_state.get("rapid_enable_family_variants", True))
        rapid_final_polish_enabled = bool(st.session_state.get("rapid_final_polish_enabled", False))
        rapid_demo_fixture_enabled = bool(st.session_state.get("rapid_demo_fixture_enabled", False))

        # --- 9. SEARCH STRATEGY ---
        if analysis_mode == "Rapid Hypothesis Mode":
            strategy_mode = "Rapid hypothesis"
            max_passes = max(1, int(rapid_hypothesis_phase_count))
            trace_limit = 0.50
            rwp_eps = 0.06
            max_joint_cycles = 6
            top_n_ml = int(rapid_stage_output_limit)
            wait_for_pass = int(rapid_gsas_validation_limit)
            len_tol = 1.0
            ang_tol = 3.0
            nudge_samples = 5000
            nudge_reps = 50
            dedup_threshold = 0.95
            joint_k = max(1, min(3, int(rapid_gsas_validation_limit)))
            score_q_max = 8.0
            pearson_cell_min_r = 0.50
            lattice_tiebreak_score_tol = 0.0005
            candidate_pruning = True
            knee_max_keep_if_no_knee = 2
            knee_max_keep_at_most = 5
            relative_search_budget = 1.0
            screened_per_run = int(rapid_stage_output_limit)
            nudged_per_run = int(rapid_gsas_validation_limit)
        else:
            with st.expander("9. Runtime Budget", expanded=True):
                strategy_profiles = {
                    "Quick diagnostic": {
                        "description": "Fast file/main-phase check and obvious impurity scan. Lowest sensitivity.",
                        "max_passes": 1, "trace_limit": 1.00, "rwp_eps": 0.12,
                        "top_n_ml": 12, "wait_for_pass": 3, "len_tol": 0.60,
                        "ang_tol": 1.5, "nudge_samples": 800, "nudge_reps": 10,
                        "dedup_threshold": 0.95, "joint_k": 1, "max_joint_cycles": 4,
                        "knee_max_keep_if_no_knee": 1, "knee_max_keep_at_most": 3,
                        "score_q_max": 8.0,
                        "pearson_cell_min_r": 0.50,
                        "lattice_tiebreak_score_tol": 0.0005,
                    },
                    "Balanced first-pass": {
                        "description": "Recommended first real run. Searches fewer candidates than the old Standard preset.",
                        "max_passes": 2, "trace_limit": 0.50, "rwp_eps": 0.06,
                        "top_n_ml": 35, "wait_for_pass": 7, "len_tol": 1.0,
                        "ang_tol": 3.0, "nudge_samples": 5000, "nudge_reps": 50,
                        "dedup_threshold": 0.95, "joint_k": 2, "max_joint_cycles": 6,
                        "knee_max_keep_if_no_knee": 2, "knee_max_keep_at_most": 5,
                        "score_q_max": 8.0,
                        "pearson_cell_min_r": 0.50,
                        "lattice_tiebreak_score_tol": 0.0005,
                    },
                    "Thorough search": {
                        "description": "Use after a balanced run leaves unexplained peaks. Slower and more sensitive.",
                        "max_passes": 3, "trace_limit": 0.25, "rwp_eps": 0.03,
                        "top_n_ml": 75, "wait_for_pass": 12, "len_tol": 2.0,
                        "ang_tol": 5.0, "nudge_samples": 20000, "nudge_reps": 150,
                        "dedup_threshold": 0.95, "joint_k": 3, "max_joint_cycles": 8,
                        "knee_max_keep_if_no_knee": 0, "knee_max_keep_at_most": 10,
                        "score_q_max": 8.0,
                        "pearson_cell_min_r": 0.10,
                        "lattice_tiebreak_score_tol": 0.0005,
                    },
                }
                strategy_mode = st.selectbox(
                    "Runtime profile",
                    ["Balanced first-pass", "Quick diagnostic", "Thorough search", "Custom"],
                    index=0,
                    help="This controls the scientific search budget: rounds, candidates, lattice-nudge breadth, and early stopping.",
                )
                preset = strategy_profiles.get(strategy_mode, strategy_profiles["Balanced first-pass"])
                st.caption(preset["description"])

                s1, s2 = st.columns(2)
                with s1:
                    max_passes = st.number_input(
                        "Impurity discovery rounds",
                        1,
                        10,
                        int(preset["max_passes"]),
                        help="Maximum number of sequential impurity phases to search for. This is usually the largest runtime multiplier.",
                        key=f"max_passes_{strategy_mode}",
                    )
                    trace_limit = st.number_input(
                        "Stop below phase fraction (wt%)",
                        0.0,
                        100.0,
                        float(preset["trace_limit"]),
                        help="If the best newly accepted phase is below this fraction, the run can stop instead of chasing traces.",
                        key=f"trace_limit_{strategy_mode}",
                    )
                with s2:
                    rwp_eps = st.number_input(
                        "Stop if Rwp improves less than",
                        0.0,
                        1.0,
                        float(preset["rwp_eps"]),
                        format="%.3f",
                        help="Early-stop threshold after accepting/polishing an impurity.",
                        key=f"rwp_eps_{strategy_mode}",
                    )
                    max_joint_cycles = st.number_input(
                        "Final compare cycles",
                        1,
                        20,
                        int(preset["max_joint_cycles"]),
                        help="Cycles used when comparing the current model against top impurity candidates.",
                        key=f"max_joint_cycles_{strategy_mode}",
                    )

                if strategy_mode == "Custom":
                    st.markdown("#### Candidate funnel")
                    c1, c2 = st.columns(2)
                    with c1:
                        top_n_ml = st.number_input("Histogram/ML shortlist", 1, 1000, int(preset["top_n_ml"]))
                        wait_for_pass = st.number_input("Lattice-nudge candidates", 1, 100, int(preset["wait_for_pass"]))
                        joint_k = st.number_input("Compare-run candidates", 1, 20, int(preset["joint_k"]))
                        dedup_threshold = st.number_input("Duplicate-candidate threshold", 0.0, 1.0, float(preset["dedup_threshold"]), format="%.2f")
                    with c2:
                        len_tol = st.number_input("Lattice length tolerance (%)", 0.0, 50.0, float(preset["len_tol"]))
                        ang_tol = st.number_input("Lattice angle tolerance (deg)", 0.0, 90.0, float(preset["ang_tol"]))
                        nudge_samples = st.number_input("Nudge Q-target samples", 1, 50000, int(preset["nudge_samples"]))
                        nudge_reps = st.number_input("Nudge representative count", 1, 500, int(preset["nudge_reps"]))
                        score_q_max = st.number_input(
                            "Nudge scoring Q max",
                            0.1,
                            50.0,
                            float(preset["score_q_max"]),
                            format="%.2f",
                            help="Upper Q limit, in inverse angstrom, used for Stage-4 lattice-nudge scoring. Keep near 8 for speed and low-Q robustness.",
                        )
                        pearson_cell_min_r = st.number_input(
                            "Pearson cell-refine cutoff",
                            0.0,
                            1.0,
                            float(preset["pearson_cell_min_r"]),
                            format="%.2f",
                            help=(
                                "Candidate Pearson ranking always runs a scale-only GSAS check first. "
                                "Cell refinement is only added when that correlation is at least this value; "
                                "accepted phases can still receive cell refinement during final polish."
                            ),
                        )
                        lattice_tiebreak_score_tol = st.number_input(
                            "Nudge near-tie score tolerance",
                            0.0,
                            0.01,
                            float(preset["lattice_tiebreak_score_tol"]),
                            format="%.4f",
                            help="When multiple nudged lattices score within this tolerance, prefer the one closest to the starting lattice.",
                        )
                    candidate_pruning = st.toggle(
                        "Use automatic candidate pruning",
                        value=True,
                        help="Knee filters reduce candidates after histogram/nudge/Pearson scoring. Turning this off is slower but more exhaustive.",
                    )
                    knee_max_keep_if_no_knee = st.number_input(
                        "Fallback keep count when no clear knee is found",
                        0,
                        100,
                        int(preset["knee_max_keep_if_no_knee"]),
                        help="0 keeps all candidates when no knee is detected.",
                    )
                    knee_max_keep_at_most = st.number_input(
                        "Maximum candidates kept after pruning",
                        0,
                        100,
                        int(preset["knee_max_keep_at_most"]),
                        help="0 means no cap.",
                    )
                else:
                    top_n_ml = int(preset["top_n_ml"])
                    wait_for_pass = int(preset["wait_for_pass"])
                    len_tol = float(preset["len_tol"])
                    ang_tol = float(preset["ang_tol"])
                    nudge_samples = int(preset["nudge_samples"])
                    nudge_reps = int(preset["nudge_reps"])
                    dedup_threshold = float(preset["dedup_threshold"])
                    joint_k = int(preset["joint_k"])
                    score_q_max = float(preset["score_q_max"])
                    pearson_cell_min_r = float(preset["pearson_cell_min_r"])
                    lattice_tiebreak_score_tol = float(preset["lattice_tiebreak_score_tol"])
                    candidate_pruning = True
                    knee_max_keep_if_no_knee = int(preset["knee_max_keep_if_no_knee"])
                    knee_max_keep_at_most = int(preset["knee_max_keep_at_most"])

                quick_units = (
                    strategy_profiles["Quick diagnostic"]["max_passes"]
                    * strategy_profiles["Quick diagnostic"]["wait_for_pass"]
                    * strategy_profiles["Quick diagnostic"]["nudge_samples"]
                    * strategy_profiles["Quick diagnostic"]["nudge_reps"]
                )
                search_units = max(1, int(max_passes)) * max(1, int(wait_for_pass)) * max(1, int(nudge_samples)) * max(1, int(nudge_reps))
                relative_search_budget = max(0.1, search_units / max(1, quick_units))
                screened_per_run = int(max_passes) * int(top_n_ml)
                nudged_per_run = int(max_passes) * int(wait_for_pass)

                _render_sidebar_summary_rows(
                    [
                        ("Relative search budget", f"{relative_search_budget:.1f}x"),
                        ("Screened candidates", f"{screened_per_run:,}"),
                        ("Nudged/refined candidates", f"{nudged_per_run:,}"),
                    ]
                )
                st.caption(
                    f"Per pass: shortlist {top_n_ml}, lattice-nudge {wait_for_pass}, compare {joint_k}; "
                    f"nudge breadth {nudge_samples} samples x {nudge_reps} representatives."
                )
                if strategy_mode == "Quick diagnostic":
                    st.warning("Quick diagnostic is meant to validate setup and catch obvious phases. Re-run Balanced or Thorough before treating small impurities as absent.")

        # --- 10. EXPERT MODE (Hidden/Searchable) ---
        expert_mode = st.toggle("Show Expert / Debug Controls", value=False, key="show_expert_params")
        if expert_mode:
            with st.expander("Expert Tuning (Internal)", expanded=True):
                st.caption("Algorithm internals for debugging and research.")
                if analysis_mode == "Rapid Hypothesis Mode":
                    rapid_demo_fixture_enabled = st.checkbox(
                        "Show LK-99 benchmark fixture",
                        key="rapid_demo_fixture_enabled",
                        help="Development-only viewer for the saved LK-99 rapid-mode experiment.",
                    )
                st.markdown("**Main phase anchor cleanup**")
                main_phase_cleanup_enabled = st.checkbox(
                    "Refine supplied main-CIF internal parameters after lattice anchoring",
                    key="main_phase_cleanup_enabled",
                    help=(
                        "Runs on a cloned main-phase project after the normal fit/pre-nudge. "
                        "The result is adopted only if Rwp improves and atom/U sanity checks pass."
                    ),
                )
                if not main_phase_cleanup_enabled:
                    st.session_state.main_phase_cleanup_refine_u_iso = False
                    st.session_state.main_phase_cleanup_refine_positions = False
                cc1, cc2 = st.columns(2)
                with cc1:
                    main_phase_cleanup_refine_u_iso = st.checkbox(
                        "Refine Uiso",
                        key="main_phase_cleanup_refine_u_iso",
                        disabled=not main_phase_cleanup_enabled,
                    )
                with cc2:
                    main_phase_cleanup_refine_positions = st.checkbox(
                        "Refine atomic positions",
                        key="main_phase_cleanup_refine_positions",
                        disabled=not main_phase_cleanup_enabled,
                        help="Riskier than Uiso. Keep off unless the supplied main CIF is structurally unoptimized.",
                    )
                st.caption("Cleanup never refines occupancies and does not replace the main phase unless the cloned refinement is accepted.")
                k_min_hist = st.number_input("Knee: Min Points", 1, 100, 5)
                k_span = st.number_input("Knee: Min Span", 0.0, 1.0, 0.03, format="%.3f")
                joint_k = st.number_input(
                    "Joint: Top-K Candidates",
                    1,
                    20,
                    int(joint_k),
                    help="Number of candidates for joint refinement.",
                )
                max_joint_cycles = st.number_input(
                    "Joint: Max Cycles",
                    1,
                    20,
                    int(max_joint_cycles),
                    help="Maximum refinement cycles during candidate compare runs.",
                )
                excluded_sgs = st.text_input("Excluded Space Groups", "1, 2", help="Comma-separated space group numbers to omit from candidate screening.")

                st.divider()
                db_catalog = st.text_input("Catalog CSV", "catalog_deduplicated.csv")
                db_stable = st.text_input("Stable CSV", "mp_experimental_stable.csv")
                db_metadata = st.text_input("Metadata JSON", "highsymm_metadata.json")
        else:
            # Defaults for expert params if not in expert mode
            st.session_state.rapid_demo_fixture_enabled = False
            main_phase_cleanup_enabled = bool(st.session_state.get("main_phase_cleanup_enabled", False))
            main_phase_cleanup_refine_u_iso = bool(st.session_state.get("main_phase_cleanup_refine_u_iso", False))
            main_phase_cleanup_refine_positions = bool(st.session_state.get("main_phase_cleanup_refine_positions", False))
            k_min_hist, k_span = 5, 0.03
            excluded_sgs = "1, 2"
            db_catalog, db_stable, db_metadata = "catalog_deduplicated.csv", "mp_experimental_stable.csv", "highsymm_metadata.json"

        # --- 10. REVIEW ---
        viewing_saved_result = bool(st.session_state.get("selected_run_dir") and not st.session_state.get("run_active"))
        review_title = "10. Review Next Run Plan" if viewing_saved_result else "10. Review Run Plan"
        with st.expander(review_title, expanded=True):
            if viewing_saved_result:
                if st.session_state.get("dataset_source_mode") == "Reuse Previous Run":
                    st.info(
                        "This plan is for the editable copied setup you would start next. "
                        "The saved result currently shown on the right remains unchanged."
                    )
                else:
                    st.info(
                        "This plan is for the next run you would start from the sidebar. "
                        "It is not the configuration summary of the saved result currently shown on the right. "
                        "Use **Prepare Editable Copy** in the saved-result banner to reuse old inputs."
                    )
            sample_preview = parse_element_list(allowed_elements_str)
            env_preview = parse_element_list(sample_env_elements_str)
            active_phase_count = _active_db_phase_count()
            st.markdown(
                "\n".join([
                    f"- Measurement type: **{rad_source}**",
                    f"- Candidate library: **{ACTIVE_DB_LABEL}**"
                    + (f" ({active_phase_count} phases)" if active_phase_count is not None else ""),
                    f"- Dataset source: **{example_labels.get(example_selection, example_selection)}**",
                    f"- Run folder: `{st.session_state.workspace_context.get('username', 'workspace')}/runs/{(run_name or '').strip().replace(' ', '_') or '[missing]'}`",
                    f"- Sample elements: **{', '.join(sample_preview) if sample_preview else '[blank]'}**",
                    f"- Sample can/environment: **{', '.join(env_preview) if env_preview else 'none'}**",
                    f"- Pattern geometry: **{inst_mode}**",
                    f"- Background: **{bg_mode_label} / {bg_type} / {int(bg_terms)} terms**",
                    f"- Magnetic precheck: **{'on' if magnetic_precheck_enabled else 'off'}**",
                    f"- Main-CIF cleanup: **{'on' if main_phase_cleanup_enabled else 'off'}**",
                    f"- Main-phase lookalike filter: **{'on' if main_phase_shadow_filter_enabled else 'off'}**",
                    f"- Analysis path: **{analysis_mode}**",
                    *(
                        [
                            f"- Runtime profile: **{strategy_mode}** (~{relative_search_budget:.1f}x quick diagnostic budget)",
                            f"- Search depth: up to **{int(max_passes)}** impurity rounds; **{int(wait_for_pass)}** nudged candidates/pass",
                            f"- Minimum reported phase: **{float(trace_limit):.3g} wt%**",
                        ]
                        if analysis_mode != "Rapid Hypothesis Mode"
                        else []
                    ),
                    f"- Ignored regions: **{len(excluded_region_pairs)}**",
                    f"- Auto can/reference masks: **{', '.join(reference_phase_exclusion_config.get('presets', [])) if reference_phase_exclusion_config.get('enabled') else 'off'}**",
                    *(
                        [
                            f"- Rapid candidate phases/hypothesis: **{rapid_hypothesis_phase_count}**",
                            f"- Rapid stage rows: **{rapid_stage_output_limit}**",
                            f"- Rapid refinement checks: **{rapid_gsas_validation_limit}**",
                            f"- Rapid refinement workers: **{rapid_gsas_parallel_workers}**",
                            f"- Rapid deep cell polish: **{'on' if rapid_final_polish_enabled else 'off'}**",
                            f"- Rapid family variants: **{'on' if rapid_enable_family_variants else 'off'}**",
                        ]
                        if analysis_mode == "Rapid Hypothesis Mode"
                        else []
                    ),
                ])
            )

            api_config_errors: list[str] = []
            api_allowed_elements = list(sample_preview)
            if not api_allowed_elements:
                if ACTIVE_DB_KIND != "original" and ACTIVE_DB_EXISTS:
                    api_allowed_elements = _derive_allowed_elements_from_active_db()
                else:
                    api_config_errors.append("Enter sample elements before exporting an API config.")
            if fit_window_input_errors:
                api_config_errors.extend(fit_window_input_errors)
            if excluded_region_input_errors:
                api_config_errors.extend(excluded_region_input_errors)
            if reference_phase_exclusion_config.get("enabled") and not reference_phase_exclusion_config.get("presets"):
                api_config_errors.append("Auto-mask sample can/reference peaks requires at least one preset.")

            api_run_name = (run_name or "radar_api_run").strip().replace(" ", "_") or "radar_api_run"
            api_magnetic_denominators: list[int] = []
            for token in str(magnetic_precheck_denominators).replace(";", ",").split(","):
                token = token.strip()
                if token.isdigit() and int(token) > 1:
                    api_magnetic_denominators.append(int(token))
            if not api_magnetic_denominators:
                api_magnetic_denominators = [2, 3]

            api_adv_cfg = {
                "analysis_mode": (
                    "rapid_hypothesis"
                    if analysis_mode == "Rapid Hypothesis Mode"
                    else "full_radar_pd"
                ),
                "rapid_hypothesis": {
                    "enabled": analysis_mode == "Rapid Hypothesis Mode",
                    "beam_depth": int(rapid_hypothesis_phase_count),
                    "beam_width": 40,
                    "branch_top": 160,
                    "nudge_unique_phases": 0,
                    "parallel_nudge": True,
                    "stage_output_limit": int(rapid_stage_output_limit),
                    "gsas_validation_limit": int(rapid_gsas_validation_limit),
                    "gsas_parallel_workers": int(rapid_gsas_parallel_workers),
                    "final_polish_enabled": bool(rapid_final_polish_enabled),
                    "final_polish_strategy": "adaptive" if rapid_final_polish_enabled else "quick",
                    "low_weight_prune_pct": 0.2,
                    "low_weight_skip_min_phases": 2,
                    "show_family_variants": bool(rapid_enable_family_variants),
                    "workflow": "64_bin_to_radar_nudge_to_512_bin_to_final_refinement_ranking",
                },
                "hist_filter": {"topN": int(top_n_ml)},
                "top_candidates": int(wait_for_pass),
                "joint_top_k": int(joint_k),
                "max_joint_cycles": int(max_joint_cycles),
                "adaptive_compare_enabled": True,
                "adaptive_compare_keep": max(1, min(2, int(joint_k))),
                "adaptive_compare_cycles": 1,
                "rwp_improve_eps": float(rwp_eps),
                "polish_defer_main_cell": True,
                "main_phase_prenudge": {
                    "enabled": True,
                    "apply_only_user_main": True,
                    "trigger_rwp": 18.0,
                    "hard_rwp": 35.0,
                    "min_peak_support": 0.50,
                    "min_rwp_for_peak_support_trigger": 8.0,
                    "strongest_barely_supported_fraction": 0.75,
                    "peak_match_tolerance_q": 0.035,
                    "frac_window": 0.01,
                    "angle_window_deg": 1.0,
                },
                "main_phase_guard": {
                    "enabled": True,
                    "apply_only_user_main": True,
                    "min_weight_pct": 20.0,
                },
                "main_phase_shadow": {
                    "enabled": bool(main_phase_shadow_filter_enabled),
                    "top_main_peaks": 8,
                    "top_candidate_peaks": 10,
                    "peak_match_tolerance_q": 0.040,
                    "min_target_prominence_fraction": 0.03,
                    "nudge_filter_enabled": bool(main_phase_shadow_filter_enabled),
                    "filter_top_main_peaks": 5,
                    "filter_top_candidate_peaks": 5,
                    "filter_min_overlap_count": 3,
                    "filter_min_overlap_fraction": 0.60,
                    "filter_min_shadow_intensity_fraction": 0.60,
                    "filter_max_unique_supported_count": 1,
                    "filter_max_unique_supported_fraction": 0.25,
                    "refill_attempts": 2,
                },
                "main_phase_cleanup": {
                    "enabled": bool(main_phase_cleanup_enabled),
                    "apply_only_user_main": True,
                    "refine_u_iso": bool(main_phase_cleanup_enabled) and bool(main_phase_cleanup_refine_u_iso),
                    "refine_positions": bool(main_phase_cleanup_enabled) and bool(main_phase_cleanup_refine_positions),
                    "cycles": 1,
                    "accept_rwp_worsen": 0.15,
                    "min_rwp_improvement": 0.05,
                    "max_position_shift": 0.15,
                    "min_u_iso": 0.0,
                    "max_u_iso": 0.20,
                },
                "stage4": {
                    "samples": int(nudge_samples),
                    "reps": int(nudge_reps),
                    "len_tol_pct": float(len_tol),
                    "ang_tol_deg": float(ang_tol),
                    "score_q_max": float(score_q_max),
                    "pearson_q_max": float(score_q_max),
                    "pearson_defer_export": True,
                    "lattice_tiebreak_score_tol": float(lattice_tiebreak_score_tol),
                    "pearson_cell_refine_min_r": float(pearson_cell_min_r),
                    "radiation": "xray" if IS_XRAY else "neutron",
                },
                "knee_filter": {
                    "enable_hist": True,
                    "enable_nudge": bool(candidate_pruning),
                    "enable_pearson": bool(candidate_pruning),
                    "min_points_hist": int(k_min_hist),
                    "min_points_nudge": 3,
                    "min_points_pearson": 2,
                    "min_rel_span": float(k_span),
                    "guard_frac": 0.05,
                    "max_keep_if_no_knee": int(knee_max_keep_if_no_knee),
                    "min_keep_at_least": 1 if candidate_pruning else 0,
                    "max_keep_at_most": int(knee_max_keep_at_most),
                },
                "corr_threshold": float(dedup_threshold),
                "exclude_sg": [int(s.strip()) for s in excluded_sgs.split(",") if s.strip().isdigit()],
                "background": {
                    "mode": bg_mode,
                    "type": bg_type,
                    "terms": int(bg_terms),
                },
                "magnetic_precheck": {
                    "enabled": bool(magnetic_precheck_enabled and (not IS_XRAY) and main_phase_available_for_precheck),
                    "q_max": float(magnetic_precheck_q_max),
                    "max_hkl": 8,
                    "denominators": api_magnetic_denominators,
                    "width_grid": [0.025, 0.04, 0.065],
                    "pseudo_voigt_eta": 0.35,
                    "max_k_vectors": 80,
                    "max_positions_per_k": 260,
                    "top_residual_peaks": 8,
                    "null_trials": 48,
                    "include_gamma": False,
                },
                "xray_doublet": {
                    "enabled": "auto",
                    "default_intensity_ratio": 0.5,
                    "apply_to_64_ml_input": True,
                    "apply_to_64_similarity": True,
                    "apply_to_512": True,
                    "apply_to_lattice_nudge": True,
                },
                "api_usage": {
                    "upload_fields": {
                        "config": "this YAML file",
                        "data": "diffraction data file",
                        "instrument": "GSAS-II .instprm/.prm/.inst/.ins file",
                        "main_cif": "optional known/main phase CIF",
                    },
                    "submit_endpoint": "POST /api/v1/jobs",
                    "status_endpoint": "GET /api/v1/jobs/{job_id}",
                    "results_endpoint": "GET /api/v1/jobs/{job_id}/results.zip",
                },
            }

            if IS_XRAY and main_phase_available_for_precheck and light_calibration_enabled:
                api_adv_cfg["light_calibration"] = {
                    "enabled": True,
                    "zero_cycles": 1,
                    "profile_cycles": 2,
                    "accept_rwp_worsen": 0.15,
                    "terms": ["Zero", "U", "V", "W"],
                }

            api_db_overrides = {}
            if db_catalog != "catalog_deduplicated.csv":
                api_db_overrides["catalog_csv"] = str(Path(ACTIVE_DB_ROOT) / db_catalog)
            if db_stable != "mp_experimental_stable.csv":
                api_db_overrides["stable_csv"] = str(Path(ACTIVE_DB_ROOT) / db_stable)
            if db_metadata != "highsymm_metadata.json":
                api_db_overrides["original_json"] = str(Path(ACTIVE_DB_ROOT) / db_metadata)
            if api_db_overrides:
                api_adv_cfg["db"] = api_db_overrides

            api_selected_mode = (
                inst_mode.lower()
                if using_previous_run_inputs
                else ("auto" if example_selection != "None" else ("cw" if builtin_instprm_key else inst_mode.lower()))
            )
            api_main_cif_placeholder = "UPLOAD_MAIN_CIF_WITH_FIELD_main_cif.cif" if main_phase_available_for_precheck else None
            if not api_config_errors:
                try:
                    api_cfg_text = build_pipeline_config(
                        run_name=api_run_name,
                        data_file="UPLOAD_DIFFRACTION_DATA_WITH_FIELD_data",
                        instprm_file="UPLOAD_INSTRUMENT_PROFILE_WITH_FIELD_instrument.instprm",
                        allowed_elements=api_allowed_elements,
                        sample_env_elements=env_preview,
                        main_cif=api_main_cif_placeholder,
                        work_root="RADAR_PD_API_WILL_REWRITE_THIS_RUN_DIR",
                        project_root=PROJECT_ROOT,
                        db_root=str(ACTIVE_DB_ROOT),
                        db_config_override=dict(ACTIVE_DB_CONFIG),
                        original_json_override=ACTIVE_DB_CONFIG.get("original_json"),
                        cif_map_json_override=ACTIVE_DB_CONFIG.get("cif_map_json"),
                        min_impurity_percent=trace_limit,
                        max_passes=max_passes,
                        instrument_mode=api_selected_mode,
                        advanced_params=api_adv_cfg,
                        limits=list(fit_window) if fit_window else None,
                        exclude_regions=excluded_region_pairs,
                        reference_phase_exclusions=reference_phase_exclusion_config,
                    )
                    st.download_button(
                        "Download API config",
                        api_cfg_text,
                        file_name=f"{api_run_name}_api_config.yaml",
                        mime="application/x-yaml",
                        width="stretch",
                        help="Use this config with the RADAR-PD API. Upload data/instrument/main_cif files with the request; the API rewrites those placeholder paths.",
                    )
                except Exception as exc:
                    st.warning(f"API config export is not available for this setup yet: {exc}")
            else:
                st.caption("API config export will appear after required setup fields are valid.")
                for message in api_config_errors:
                    st.caption(f"- {message}")

        # START BUTTON
        if not st.session_state.run_active:
            launch_label = "Start Rapid Hypothesis Run" if analysis_mode == "Rapid Hypothesis Mode" else "Start RADAR-PD"
            if st.button(launch_label, width="stretch"):
                    clean_name = (run_name or "").strip().replace(" ", "_")
                    setup_errors = []

                    if not clean_name:
                        setup_errors.append("Please enter a run name before starting the pipeline.")
                    elif (ACTIVE_RUNS_ROOT / clean_name).exists():
                        setup_errors.append(
                            f"Run folder `{clean_name}` already exists in the active workspace. Use New name or enter a different run name."
                        )
                    if not GSAS_READY:
                        setup_errors.append("GSAS-II is not ready. Open System Diagnostics and retry the GSAS-II check.")
                    if not ACTIVE_DB_EXISTS:
                        setup_errors.append(f"The active candidate library is not usable: {ACTIVE_DB_LABEL}.")

                    if using_previous_run_inputs:
                        if not previous_run_entry:
                            setup_errors.append("Select a previous run with reusable input files.")
                        else:
                            reuse_data_mode = st.session_state.get("reuse_data_source_mode", REUSE_SAVED_FILE)
                            reuse_inst_mode = st.session_state.get("reuse_instprm_source_mode", REUSE_SAVED_FILE)
                            reuse_main_mode = st.session_state.get("reuse_main_cif_source_mode", REUSE_NO_MAIN_CIF)

                            if reuse_data_mode == REUSE_REPLACEMENT_FILE:
                                reuse_staged_data_file = _get_staged_upload("reuse_diffraction_data")
                                if not _staged_upload_is_nonempty(reuse_staged_data_file):
                                    setup_errors.append("Upload a replacement diffraction data file, or switch data source back to the saved file.")
                            elif not Path(str(previous_run_entry.get("data_path") or "")).exists():
                                setup_errors.append("The previous run's diffraction data file is no longer available.")

                            if reuse_inst_mode == REUSE_REPLACEMENT_FILE:
                                reuse_staged_instprm_file = _get_staged_upload("reuse_instrument_profile")
                                if not _staged_upload_is_nonempty(reuse_staged_instprm_file):
                                    setup_errors.append("Upload a replacement instrument profile, or switch instrument source back to the saved file.")
                            elif not Path(str(previous_run_entry.get("instprm_path") or "")).exists():
                                setup_errors.append("The previous run's instrument profile file is no longer available.")

                            if reuse_main_mode == REUSE_REPLACEMENT_FILE:
                                reuse_staged_main_cif = _get_staged_upload("reuse_main_cif")
                                if not _staged_upload_is_nonempty(reuse_staged_main_cif):
                                    setup_errors.append("Upload a replacement main CIF, or choose saved/no main CIF.")
                            elif reuse_main_mode == REUSE_SAVED_MAIN_CIF:
                                saved_main_path = Path(str(previous_run_entry.get("main_cif") or ""))
                                if not saved_main_path.exists():
                                    setup_errors.append("The previous run's main CIF is no longer available. Choose a replacement or no main CIF.")
                    elif example_selection == "None":
                        staged_data_file = _get_staged_upload("diffraction_data")
                        staged_instprm_file = _get_staged_upload("instrument_profile")
                        staged_main_cif = _get_staged_upload("main_cif")
                        if not data_file and not staged_data_file:
                            setup_errors.append("Please upload a diffraction data file.")
                        elif data_file is not None and int(getattr(data_file, "size", 0) or 0) <= 0:
                            setup_errors.append("The uploaded diffraction data file appears to be empty.")
                        elif staged_data_file and Path(staged_data_file["path"]).stat().st_size <= 0:
                            setup_errors.append("The staged diffraction data file appears to be empty.")
                        if builtin_instprm_key:
                            if not IS_XRAY:
                                setup_errors.append("The built-in CuKa lab PXRD preset is only available for X-ray runs.")
                        elif not instprm_file and not staged_instprm_file:
                            if IS_XRAY:
                                setup_errors.append("Please upload an instrument parameter file, or select the built-in CuKa lab PXRD preset.")
                            else:
                                setup_errors.append("Please upload an instrument parameter file.")
                        elif instprm_file is not None and int(getattr(instprm_file, "size", 0) or 0) <= 0:
                            setup_errors.append("The uploaded instrument profile appears to be empty.")
                        elif staged_instprm_file and Path(staged_instprm_file["path"]).stat().st_size <= 0:
                            setup_errors.append("The staged instrument profile appears to be empty.")
                        if main_cif is not None and int(getattr(main_cif, "size", 0) or 0) <= 0:
                            setup_errors.append("The uploaded main CIF appears to be empty.")
                        elif staged_main_cif and Path(staged_main_cif["path"]).stat().st_size <= 0:
                            setup_errors.append("The staged main CIF appears to be empty.")

                    if setup_errors:
                        for message in setup_errors:
                            st.error(message)
                    else:
                        rdir = ACTIVE_RUNS_ROOT / clean_name
                        input_dir = rdir / "inputs"
                        input_dir.mkdir(parents=True, exist_ok=True)

                        dpath, ipath, cpath = None, None, None

                        try:
                            if using_previous_run_inputs and previous_run_entry:
                                reuse_data_mode = st.session_state.get("reuse_data_source_mode", REUSE_SAVED_FILE)
                                reuse_inst_mode = st.session_state.get("reuse_instprm_source_mode", REUSE_SAVED_FILE)
                                reuse_main_mode = st.session_state.get("reuse_main_cif_source_mode", REUSE_NO_MAIN_CIF)

                                if reuse_data_mode == REUSE_REPLACEMENT_FILE:
                                    target_data = _copy_staged_upload_to_dir(
                                        _get_staged_upload("reuse_diffraction_data"),
                                        input_dir,
                                    )
                                else:
                                    target_data = _copy_existing_file_to_dir(Path(previous_run_entry["data_path"]), input_dir)
                                dpath = str(prepare_powder_data_file(target_data).resolve())

                                if reuse_inst_mode == REUSE_REPLACEMENT_FILE:
                                    uploaded_instprm_path = _copy_staged_upload_to_dir(
                                        _get_staged_upload("reuse_instrument_profile"),
                                        input_dir,
                                    )
                                    normalized_instprm_path = normalize_instrument_profile_to_instprm(
                                        uploaded_instprm_path,
                                        uploaded_instprm_path.with_suffix(".instprm"),
                                    )
                                    ipath = str(normalized_instprm_path.resolve())
                                else:
                                    target_inst = _copy_existing_file_to_dir(Path(previous_run_entry["instprm_path"]), input_dir)
                                    ipath = str(target_inst.resolve())

                                if reuse_main_mode == REUSE_REPLACEMENT_FILE:
                                    target_cif = _copy_staged_upload_to_dir(
                                        _get_staged_upload("reuse_main_cif"),
                                        input_dir,
                                    )
                                    cpath = str(target_cif.resolve())
                                elif reuse_main_mode == REUSE_SAVED_MAIN_CIF and previous_run_entry.get("main_cif"):
                                    target_cif = _copy_existing_file_to_dir(Path(previous_run_entry["main_cif"]), input_dir)
                                    cpath = str(target_cif.resolve())
                            elif example_selection != "None":
                                if example_selection == "TbSSL (CW Demo)":
                                    orig_d = (Path(PROJECT_ROOT) / "examples" / "tbssl" / "HB2A_TbSSL.dat")
                                    orig_i = (Path(PROJECT_ROOT) / "examples" / "tbssl" / "hb2a_si_ge113.instprm")
                                    orig_c = (Path(PROJECT_ROOT) / "examples" / "tbssl" / "TbSSL.cif")
                                else: # LK-99
                                    orig_d = (Path(PROJECT_ROOT) / "examples" / "lk99" / "PG3_56181-3.dat")
                                    orig_i = (Path(PROJECT_ROOT) / "examples" / "lk99" / "2023A_June_HighRes_60HzB3_CWL2p665.instprm")
                                    orig_c = (Path(PROJECT_ROOT) / "examples" / "lk99" / "LK99.cif")

                                shutil.copy(orig_d, input_dir / orig_d.name)
                                shutil.copy(orig_i, input_dir / orig_i.name)
                                shutil.copy(orig_c, input_dir / orig_c.name)

                                dpath = str((input_dir / orig_d.name).resolve())
                                ipath = str((input_dir / orig_i.name).resolve())
                                cpath = str((input_dir / orig_c.name).resolve())
                            else:
                                staged_data_file = _get_staged_upload("diffraction_data")
                                staged_instprm_file = _get_staged_upload("instrument_profile")
                                staged_main_cif = _get_staged_upload("main_cif")

                                if staged_data_file:
                                    staged_path = Path(staged_data_file["path"])
                                    target_data = input_dir / _safe_upload_filename(staged_data_file.get("name", staged_path.name))
                                    shutil.copy2(staged_path, target_data)
                                else:
                                    target_data = input_dir / _safe_upload_filename(data_file.name)
                                    with open(target_data, "wb") as f:
                                        f.write(data_file.getbuffer())
                                dpath = str(prepare_powder_data_file(target_data).resolve())

                                if builtin_instprm_key:
                                    preset = get_builtin_instprm_preset(builtin_instprm_key)
                                    generated_instprm = input_dir / preset["filename"]
                                    write_builtin_instprm_file(builtin_instprm_key, generated_instprm)
                                    ipath = str(generated_instprm.resolve())
                                else:
                                    if staged_instprm_file:
                                        staged_path = Path(staged_instprm_file["path"])
                                        uploaded_instprm_path = input_dir / _safe_upload_filename(staged_instprm_file.get("name", staged_path.name))
                                        shutil.copy2(staged_path, uploaded_instprm_path)
                                    else:
                                        uploaded_instprm_path = input_dir / instprm_file.name
                                        with open(uploaded_instprm_path, "wb") as f:
                                            f.write(instprm_file.getbuffer())
                                    normalized_instprm_path = normalize_instrument_profile_to_instprm(
                                        uploaded_instprm_path,
                                        uploaded_instprm_path.with_suffix(".instprm"),
                                    )
                                    ipath = str(normalized_instprm_path.resolve())

                                if staged_main_cif:
                                    staged_path = Path(staged_main_cif["path"])
                                    target_cif = input_dir / _safe_upload_filename(staged_main_cif.get("name", staged_path.name))
                                    shutil.copy2(staged_path, target_cif)
                                    cpath = str(target_cif.resolve())
                                elif main_cif:
                                    with open(input_dir / main_cif.name, "wb") as f:
                                        f.write(main_cif.getbuffer())
                                    cpath = str((input_dir / main_cif.name).resolve())
                        except Exception as exc:
                            setup_errors.append(f"Failed to prepare input files: {exc}")

                        if setup_errors:
                            for message in setup_errors:
                                st.error(message)
                        else:
                            els = parse_element_list(allowed_elements_str)
                            env = parse_element_list(sample_env_elements_str)
                            invalid_elements = invalid_element_tokens(els + env)
                            if invalid_elements:
                                setup_errors.append(
                                    "Unrecognized element symbols: " + ", ".join(invalid_elements)
                                )

                            if fit_window_input_errors:
                                setup_errors.extend(fit_window_input_errors)
                            if excluded_region_input_errors:
                                setup_errors.extend(excluded_region_input_errors)
                            if reference_phase_exclusion_config.get("enabled") and not reference_phase_exclusion_config.get("presets"):
                                setup_errors.append(
                                    "Auto-mask sample can/reference peaks requires at least one Al, Cu, or V preset."
                                )
                            if reference_phase_exclusion_config.get("enabled") and reference_phase_exclusion_config.get("window_mode") == "auto":
                                for min_key, max_key in (
                                    ("min_half_width_deg", "max_half_width_deg"),
                                    ("min_half_width_tof", "max_half_width_tof"),
                                ):
                                    if (
                                        min_key in reference_phase_exclusion_config
                                        and max_key in reference_phase_exclusion_config
                                        and float(reference_phase_exclusion_config[max_key])
                                        < float(reference_phase_exclusion_config[min_key])
                                    ):
                                        setup_errors.append(
                                            "Auto-mask sample can/reference peaks requires Max half-width >= Min half-width."
                                        )
                                        break

                            if not els:
                                if ACTIVE_DB_KIND != "original" and ACTIVE_DB_EXISTS:
                                    els = _derive_allowed_elements_from_active_db()
                                    if els:
                                        st.info(
                                            f"No allowed elements were entered. Using all {len(els)} elements "
                                            f"from the selected custom DB: {ACTIVE_DB_LABEL}."
                                        )
                                    else:
                                        setup_errors.append(
                                            "Sample elements is blank, and the selected custom library could not provide element metadata."
                                        )
                                else:
                                    setup_errors.append(
                                        "Sample elements is required when using the built-in candidate library."
                                    )

                        if setup_errors:
                            for message in setup_errors:
                                st.error(message)
                        else:
                            magnetic_denominators: list[int] = []
                            for token in str(magnetic_precheck_denominators).replace(";", ",").split(","):
                                token = token.strip()
                                if token.isdigit() and int(token) > 1:
                                    magnetic_denominators.append(int(token))
                            if not magnetic_denominators:
                                magnetic_denominators = [2, 3]

                            adv_cfg = {
                                "analysis_mode": (
                                    "rapid_hypothesis"
                                    if analysis_mode == "Rapid Hypothesis Mode"
                                    else "full_radar_pd"
                                ),
                                "rapid_hypothesis": {
                                    "enabled": analysis_mode == "Rapid Hypothesis Mode",
                                    "beam_depth": int(rapid_hypothesis_phase_count),
                                    "beam_width": 40,
                                    "branch_top": 160,
                                    "nudge_unique_phases": 0,
                                    "parallel_nudge": True,
                                    "stage_output_limit": int(rapid_stage_output_limit),
                                    "gsas_validation_limit": int(rapid_gsas_validation_limit),
                                    "gsas_parallel_workers": int(rapid_gsas_parallel_workers),
                                    "final_polish_enabled": bool(rapid_final_polish_enabled),
                                    "final_polish_strategy": "adaptive" if rapid_final_polish_enabled else "quick",
                                    "low_weight_prune_pct": 0.2,
                                    "low_weight_skip_min_phases": 2,
                                    "show_family_variants": bool(rapid_enable_family_variants),
                                    "workflow": "64_bin_to_radar_nudge_to_512_bin_to_final_refinement_ranking",
                                },
                                "hist_filter": {"topN": top_n_ml},
                                "top_candidates": wait_for_pass,
                                "joint_top_k": joint_k,
                                "max_joint_cycles": int(max_joint_cycles),
                                "adaptive_compare_enabled": True,
                                "adaptive_compare_keep": max(1, min(2, int(joint_k))),
                                "adaptive_compare_cycles": 1,
                                "rwp_improve_eps": rwp_eps,
                                "polish_defer_main_cell": True,
                                "main_phase_prenudge": {
                                    "enabled": True,
                                    "apply_only_user_main": True,
                                    "trigger_rwp": 18.0,
                                    "hard_rwp": 35.0,
                                    "min_peak_support": 0.50,
                                    "min_rwp_for_peak_support_trigger": 8.0,
                                    "strongest_barely_supported_fraction": 0.75,
                                    "peak_match_tolerance_q": 0.035,
                                    "frac_window": 0.01,
                                    "angle_window_deg": 1.0,
                                },
                                "main_phase_guard": {
                                    "enabled": True,
                                    "apply_only_user_main": True,
                                    "min_weight_pct": 20.0,
                                },
                                "main_phase_shadow": {
                                    "enabled": bool(main_phase_shadow_filter_enabled),
                                    "top_main_peaks": 8,
                                    "top_candidate_peaks": 10,
                                    "peak_match_tolerance_q": 0.040,
                                    "min_target_prominence_fraction": 0.03,
                                    "nudge_filter_enabled": bool(main_phase_shadow_filter_enabled),
                                    "filter_top_main_peaks": 5,
                                    "filter_top_candidate_peaks": 5,
                                    "filter_min_overlap_count": 3,
                                    "filter_min_overlap_fraction": 0.60,
                                    "filter_min_shadow_intensity_fraction": 0.60,
                                    "filter_max_unique_supported_count": 1,
                                    "filter_max_unique_supported_fraction": 0.25,
                                    "refill_attempts": 2,
                                },
                                "main_phase_cleanup": {
                                    "enabled": bool(main_phase_cleanup_enabled),
                                    "apply_only_user_main": True,
                                    "refine_u_iso": bool(main_phase_cleanup_enabled) and bool(main_phase_cleanup_refine_u_iso),
                                    "refine_positions": bool(main_phase_cleanup_enabled) and bool(main_phase_cleanup_refine_positions),
                                    "cycles": 1,
                                    "accept_rwp_worsen": 0.15,
                                    "min_rwp_improvement": 0.05,
                                    "max_position_shift": 0.15,
                                    "min_u_iso": 0.0,
                                    "max_u_iso": 0.20,
                                },
                                "stage4": {
                                    "samples": int(nudge_samples),
                                    "reps": int(nudge_reps),
                                    "len_tol_pct": len_tol,
                                    "ang_tol_deg": ang_tol,
                                    "score_q_max": float(score_q_max),
                                    "pearson_q_max": float(score_q_max),
                                    "pearson_defer_export": True,
                                    "lattice_tiebreak_score_tol": float(lattice_tiebreak_score_tol),
                                    "pearson_cell_refine_min_r": float(pearson_cell_min_r),
                                },
                                "knee_filter": {
                                    "enable_hist": True,
                                    "enable_nudge": bool(candidate_pruning),
                                    "enable_pearson": bool(candidate_pruning),
                                    "min_points_hist": k_min_hist,
                                    "min_points_nudge": 3,
                                    "min_points_pearson": 2,
                                    "min_rel_span": k_span,
                                    "guard_frac": 0.05,
                                    "max_keep_if_no_knee": int(knee_max_keep_if_no_knee),
                                    "min_keep_at_least": 1 if candidate_pruning else 0,
                                    "max_keep_at_most": int(knee_max_keep_at_most),
                                },
                                "corr_threshold": dedup_threshold,
                                "exclude_sg": [int(s.strip()) for s in excluded_sgs.split(",") if s.strip().isdigit()],
                                "background": {
                                    "mode": bg_mode,
                                    "type": bg_type,
                                    "terms": int(bg_terms),
                                },
                                "magnetic_precheck": {
                                    "enabled": bool(magnetic_precheck_enabled and (not IS_XRAY) and bool(cpath)),
                                    "q_max": float(magnetic_precheck_q_max),
                                    "max_hkl": 8,
                                    "denominators": magnetic_denominators,
                                    "width_grid": [0.025, 0.04, 0.065],
                                    "pseudo_voigt_eta": 0.35,
                                    "max_k_vectors": 80,
                                    "max_positions_per_k": 260,
                                    "top_residual_peaks": 8,
                                    "null_trials": 48,
                                    "include_gamma": False,
                                },
                                "xray_doublet": {
                                    "enabled": "auto",
                                    "default_intensity_ratio": 0.5,
                                    "apply_to_64_ml_input": True,
                                    "apply_to_64_similarity": True,
                                    "apply_to_512": True,
                                    "apply_to_lattice_nudge": True,
                                }
                            }

                            db_overrides = {}
                            if db_catalog != "catalog_deduplicated.csv": db_overrides["catalog_csv"] = str(Path(ACTIVE_DB_ROOT) / db_catalog)
                            if db_stable != "mp_experimental_stable.csv": db_overrides["stable_csv"] = str(Path(ACTIVE_DB_ROOT) / db_stable)
                            if db_metadata != "highsymm_metadata.json": db_overrides["original_json"] = str(Path(ACTIVE_DB_ROOT) / db_metadata)
                            if db_overrides: adv_cfg["db"] = db_overrides

                            rad_lower = "xray" if IS_XRAY else "neutron"
                            if "stage4" not in adv_cfg:
                                adv_cfg["stage4"] = {}
                            adv_cfg["stage4"]["radiation"] = rad_lower

                            if using_previous_run_inputs:
                                selected_mode = inst_mode.lower()
                            elif example_selection != "None":
                                selected_mode = "auto"
                            elif builtin_instprm_key:
                                selected_mode = LAB_XRAY_PRESET["instrument_mode"]
                            else:
                                selected_mode = inst_mode.lower()

                            if IS_XRAY and cpath and light_calibration_enabled:
                                adv_cfg["light_calibration"] = {
                                    "enabled": True,
                                    "zero_cycles": 1,
                                    "profile_cycles": 2,
                                    "accept_rwp_worsen": 0.15,
                                    "terms": ["Zero", "U", "V", "W"],
                                }

                            workspace_ctx = st.session_state.get("workspace_context") or {}
                            adv_cfg["workspace"] = {
                                "mode": workspace_ctx.get("mode", "temporary"),
                                "username": workspace_ctx.get("username", "Temporary"),
                                "root": workspace_ctx.get("root", ""),
                            }

                            try:
                                cfg = build_pipeline_config(
                                    run_name=run_name, data_file=dpath, instprm_file=ipath,
                                    allowed_elements=els, sample_env_elements=env, main_cif=cpath,
                                    work_root=str(rdir), project_root=PROJECT_ROOT,
                                    db_root=str(ACTIVE_DB_ROOT),
                                    db_config_override=dict(ACTIVE_DB_CONFIG),
                                    original_json_override=ACTIVE_DB_CONFIG.get("original_json"),
                                    cif_map_json_override=ACTIVE_DB_CONFIG.get("cif_map_json"),
                                    min_impurity_percent=trace_limit, max_passes=max_passes,
                                    instrument_mode=selected_mode,
                                    advanced_params=adv_cfg,
                                    limits=list(fit_window) if fit_window else None,
                                    exclude_regions=excluded_region_pairs,
                                    reference_phase_exclusions=reference_phase_exclusion_config,
                                )

                                with open(rdir / "pipeline_config.yaml", "w") as f:
                                    f.write(cfg)
                                (rdir / "run_manifest.json").write_text(
                                    json.dumps(
                                        {
                                            "status": "running",
                                            "analysis_mode": (
                                                "rapid_hypothesis"
                                                if analysis_mode == "Rapid Hypothesis Mode"
                                                else "full"
                                            ),
                                            "started_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                                            "log_path": str(rdir / "pipeline.log"),
                                        },
                                        indent=2,
                                    ),
                                    encoding="utf-8",
                                )

                                st.session_state.run_dir = str(rdir)
                                st.session_state.selected_run_dir = str(rdir)
                                st.session_state.selected_run_notice = f"Viewing new run `{clean_name}`."
                                st.session_state.selected_run_notice_dir = str(rdir)
                                st.session_state.suppress_latest_run_autoload = False
                                st.session_state.run_name = clean_name
                                st.session_state.log_lines = []
                                st.session_state.show_full_log_history = False
                                st.session_state.event_file_run_dir = str(rdir)
                                st.session_state.event_file_cursor = 0
                                st.session_state.funnel_data = {"Total Database": 0, "Elements": 0, "Spacegroup": 0, "Stability": 0}
                                st.session_state.progress = 0
                                st.session_state.status_msg = "Initializing..."
                                st.session_state.current_stage_desc = "Initializing..."
                                st.session_state.last_pipeline_error = None
                                st.session_state.pipeline_state = {
                                    "global_stage_idx": 0 if analysis_mode == "Rapid Hypothesis Mode" else -1,
                                    "global_stage_desc": "Initializing",
                                    "current_pass": 0,
                                    "pass_stage": None,
                                    "stage0_status": "running" if analysis_mode == "Rapid Hypothesis Mode" else "pending",
                                    "stages_complete": set(),
                                    "rapid_mode": analysis_mode == "Rapid Hypothesis Mode",
                                }
                                st.session_state.run_summary = {
                                    "measurement": rad_source,
                                    "library": ACTIVE_DB_LABEL,
                                    "dataset": example_labels.get(example_selection, example_selection),
                                    "analysis_mode": analysis_mode,
                                    "runtime_profile": strategy_mode,
                                    "max_passes": int(max_passes),
                                }

                                lpath = str(rdir / "pipeline.log")
                                script_name = (
                                    "rapid_hypothesis_pipeline.py"
                                    if analysis_mode == "Rapid Hypothesis Mode"
                                    else "gsas_complete_pipeline_nomain.py"
                                )
                                process, q = PipelineRunner(PROJECT_ROOT, use_pixi=st.session_state.use_pixi).start_non_blocking(
                                    str(rdir / "pipeline_config.yaml"),
                                    clean_name,
                                    log_path=lpath,
                                    script_name=script_name,
                                )
                                st.session_state.pipeline_process = process
                                st.session_state.log_queue = q
                                st.session_state.run_active = True
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed to start the pipeline: {exc}")

    # STOP BUTTON
    if st.session_state.run_active and not sidebar_stop_rendered:
        if st.button("Stop RADAR-PD", width="stretch", key="stop_radar_pd_legacy"):
            stop_active_pipeline()
            st.warning("Pipeline terminated.")
            st.rerun()

    with st.expander("System Diagnostics", expanded=False):
        mem = get_memory_snapshot()
        if GSAS_READY:
            st.success("GSAS-II import check passed")
        else:
            st.error("GSAS-II import check failed")
            if st.button("Retry GSAS-II Check"):
                del st.session_state.gsas_ready
                st.rerun()
        st.metric("App process memory", f"{mem['app_mb']:.0f} MB")
        st.metric(
            "VM memory",
            f"{mem['system_used_gb']:.1f} / {mem['system_total_gb']:.1f} GB",
            f"{mem['system_percent']:.0f}% used",
        )
        st.caption(f"Active catalog key: {st.session_state.get('db_loader_source', 'not loaded')}")
        if IS_HF_SPACES:
            st.info("**Hugging Face Spaces detected**: resource limits (max 2 workers) are active to prevent OOM crashes.")

    with st.expander("Workflow Help", expanded=False):
        st.markdown("""
        **Setup order**

        1. Choose **Neutron** or **X-ray**.
        2. Pick the candidate library: built-in MP/COD, built-in plus your CIFs, or only your CIFs.
        3. Upload diffraction data, instrument profile, and optional known/main CIF.
        4. Enter sample elements and sample can/environment elements.
        5. Add fit windows or ignored artifact regions only when needed.
        6. Choose background correction.
        7. Optionally check if residual peaks can be indexed by a magnetic k-vector.
        8. Choose the analysis path: full RADAR-PD or rapid hypothesis mode.
        9. Pick the runtime budget, review the run plan, then start RADAR-PD.

        **During a run**

        Watch **Live Logs** for file/CIF/GSAS-II errors first, then check generated artifacts and final fractions.
        """)

    if st.session_state.get("last_pipeline_error"):
        st.error(f"Last pipeline error: {st.session_state.last_pipeline_error}")

    st.markdown("---")
    # Simplified sidebar status
    @st.fragment(run_every=STATUS_REFRESH_SECONDS if st.session_state.run_active else None)
    def render_sidebar_status():
        if st.session_state.run_active:
            return
        view_run_dir = _selected_or_active_run_dir()
        failure_info = _run_failure_info(view_run_dir) if view_run_dir else None
        view_manifest = _read_run_manifest(view_run_dir) if view_run_dir else {}
        view_manifest_status = str(view_manifest.get("status") or "").strip().lower()
        if failure_info:
            st.error("Status: Failed")
            st.caption(str(failure_info.get("reason") or "Pipeline exited with an error.")[:180])
        elif view_manifest_status in {"running", "starting", "processing"}:
            st.info("Status: Running")
            if view_run_dir:
                st.caption(f"Monitoring `{view_run_dir.name}` from disk.")
        elif st.session_state.run_finished:
            st.caption(f"Status: {st.session_state.get('current_stage_desc', 'Finished')}")
        else:
            st.caption("Status: Ready")

    render_sidebar_status()

# Keep process state current no matter which main view is selected.
@st.fragment(run_every=LIVE_REFRESH_SECONDS if st.session_state.run_active else None)
def pipeline_heartbeat():
    if st.session_state.run_active:
        update_ui_state()
        if random.random() < 0.03:
            gc.collect()


pipeline_heartbeat()

# --- ACTIVE VIEW ---
VIEW_RUN_DIR = _selected_or_active_run_dir()
try:
    VIEW_RUN_IS_SELECTED = bool(
        st.session_state.get("selected_run_dir")
        and VIEW_RUN_DIR
        and Path(str(st.session_state.get("selected_run_dir"))).resolve() == VIEW_RUN_DIR.resolve()
    )
except Exception:
    VIEW_RUN_IS_SELECTED = False
VIEW_RUN_MANIFEST = _read_run_manifest(VIEW_RUN_DIR) if VIEW_RUN_DIR else {}
VIEW_RUN_MANIFEST_STATUS = str(VIEW_RUN_MANIFEST.get("status") or "").strip().lower()
VIEW_RUN_APPEARS_RUNNING = bool(
    st.session_state.run_active
    or VIEW_RUN_MANIFEST_STATUS in {"running", "starting", "processing"}
)

VIEW_RUN_MODE = _run_analysis_mode_from_dir(VIEW_RUN_DIR) if VIEW_RUN_DIR else ""
if not VIEW_RUN_MODE and VIEW_RUN_APPEARS_RUNNING:
    VIEW_RUN_MODE = (
        "Rapid Hypothesis Mode"
        if st.session_state.get("analysis_mode") == "Rapid Hypothesis Mode"
        else "Full RADAR-PD"
    )

pending_setup_is_rapid = bool(
    not VIEW_RUN_DIR
    and st.session_state.get("analysis_mode") == "Rapid Hypothesis Mode"
)
current_view_is_rapid = VIEW_RUN_MODE == "Rapid Hypothesis Mode" or pending_setup_is_rapid
current_view_has_rapid_outputs = bool(
    VIEW_RUN_DIR
    and (VIEW_RUN_DIR / "rapid_results").exists()
    and VIEW_RUN_MODE != "Full RADAR-PD"
)

if current_view_is_rapid:
    RUN_WORKSPACE_OPTIONS = ["Run Monitor", "Rapid Results", "Run File Browser"]
else:
    RUN_WORKSPACE_OPTIONS = ["Run Monitor", "Results", "Interactive Plots", "Run File Browser"]
    if current_view_has_rapid_outputs:
        RUN_WORKSPACE_OPTIONS.insert(2, "Rapid Results")

_force_workspace_view_sync = False
if st.session_state.get("active_run_view") not in RUN_WORKSPACE_OPTIONS:
    st.session_state.active_run_view = "Run Monitor"
    _force_workspace_view_sync = True
elif VIEW_RUN_APPEARS_RUNNING and not current_view_is_rapid and st.session_state.get("active_run_view") != "Run Monitor":
    st.session_state.active_run_view = "Run Monitor"
    _force_workspace_view_sync = True

if (
    _force_workspace_view_sync
    or st.session_state.get("active_run_view_picker") not in RUN_WORKSPACE_OPTIONS
):
    st.session_state.active_run_view_picker = st.session_state.active_run_view

active_run_view = st.segmented_control(
    "Workspace view",
    RUN_WORKSPACE_OPTIONS,
    selection_mode="single",
    key="active_run_view_picker",
    width="stretch",
)
if active_run_view in RUN_WORKSPACE_OPTIONS:
    st.session_state.active_run_view = active_run_view
else:
    active_run_view = st.session_state.active_run_view

if VIEW_RUN_DIR:
    _render_run_context_banner(
        VIEW_RUN_DIR,
        mode=VIEW_RUN_MODE or "Unknown mode",
        status=VIEW_RUN_MANIFEST_STATUS
        or ("running" if VIEW_RUN_APPEARS_RUNNING else _run_display_status(VIEW_RUN_DIR)),
        active=VIEW_RUN_APPEARS_RUNNING,
    )


def _render_no_run_workspace_empty_state(view_label: str, *, details: str) -> None:
    workspace_mode = str((st.session_state.get("workspace_context") or {}).get("mode") or "temporary")
    saved_run_hint = (
        "Open **Workspace Access** with a username/PIN, then use **Open Previous Run** to inspect saved outputs."
        if workspace_mode == "temporary"
        else "Use **Open Previous Run** in the setup panel to inspect a saved result, or start a new run from the current setup."
    )
    st.info(
        f"No run is selected for **{view_label}** yet. {details} "
        f"{saved_run_hint}"
    )


if active_run_view in {"Run & Progress", "Run Monitor"}:
    c1, c2 = st.columns([2, 1])
    with c1:
        @st.fragment(run_every=LIVE_REFRESH_SECONDS if VIEW_RUN_APPEARS_RUNNING else None)
        def render_logs_and_monitor():
            c1_t, c1_a = st.columns([1, 1])
            if _is_rapid_context():
                c1_t.subheader("Rapid Run Monitor")
                render_rapid_run_snapshot()
                st.markdown("**Live log**")
            else:
                c1_t.subheader("Live Logs")

            failure_info = _run_failure_info(VIEW_RUN_DIR) if VIEW_RUN_DIR and not st.session_state.run_active else None
            failure_log_excerpt_text = ""
            if failure_info:
                st.error(f"Run failed: {failure_info.get('reason') or 'Pipeline exited with an error.'}")
                detail_cols = st.columns([0.62, 0.38])
                with detail_cols[0]:
                    if failure_info.get("returncode") is not None:
                        st.caption(f"Exit code: `{failure_info['returncode']}`")
                    if failure_info.get("manifest_path"):
                        st.caption(f"Failure record: `{Path(str(failure_info['manifest_path'])).name}`")
                with detail_cols[1]:
                    if VIEW_RUN_DIR:
                        st.caption(f"Run folder: `{VIEW_RUN_DIR.name}`")
                if failure_info.get("log_tail"):
                    reason = str(failure_info.get("reason") or "")
                    failure_log_excerpt_text = _failure_log_excerpt(str(failure_info["log_tail"]), reason)
                    with st.expander("Failure details", expanded=False):
                        st.code(failure_log_excerpt_text or str(failure_info["log_tail"])[-3200:], language="text")
                        st.caption("Use Download Log for the complete pipeline log.")

            # Additional controls: Download and Load Full
            c_auto, c_dl, c_full = st.columns([0.3, 0.35, 0.35])
            with c_auto:
                st.session_state.log_autoscroll = st.checkbox("Autoscroll", value=st.session_state.log_autoscroll, key="log_as_toggle")

            with c_dl:
                if VIEW_RUN_DIR:
                    lfile = VIEW_RUN_DIR / "pipeline.log"
                    if lfile.exists():
                        with open(lfile, "rb") as f:
                            st.download_button("Download Log", f, file_name="pipeline.log", key="dl_full_log_btn")

            with c_full:
                if VIEW_RUN_DIR and (st.session_state.run_finished or VIEW_RUN_IS_SELECTED):
                    if st.button("Load Full History", help="Load the entire log file from disk"):
                        lfile = VIEW_RUN_DIR / "pipeline.log"
                        if lfile.exists():
                            with open(lfile, "r", encoding="utf-8") as f:
                                st.session_state.log_lines = f.readlines()
                                st.session_state.show_full_log_history = True

            # Format logs with highlighting
            if VIEW_RUN_DIR and (VIEW_RUN_APPEARS_RUNNING or (not st.session_state.run_active and not st.session_state.log_lines)):
                lfile = VIEW_RUN_DIR / "pipeline.log"
                if lfile.exists():
                    try:
                        tail_limit = LIVE_LOG_DISPLAY_LIMIT if VIEW_RUN_APPEARS_RUNNING else FINISHED_LOG_DISPLAY_LIMIT
                        st.session_state.log_lines = lfile.read_text(
                            encoding="utf-8",
                            errors="replace",
                        ).splitlines(True)[-tail_limit:]
                    except Exception:
                        st.session_state.log_lines = []
            all_log_lines = st.session_state.log_lines
            if VIEW_RUN_APPEARS_RUNNING:
                display_limit = LIVE_LOG_DISPLAY_LIMIT
                display_log_lines = all_log_lines[-display_limit:]
            elif st.session_state.get("show_full_log_history", False):
                display_log_lines = all_log_lines
                display_limit = len(display_log_lines)
            elif failure_info:
                excerpt_text = failure_log_excerpt_text or _failure_log_excerpt(
                    "\n".join(str(line).rstrip("\n") for line in all_log_lines),
                    str(failure_info.get("reason") or ""),
                )
                display_log_lines = [f"{line}\n" for line in excerpt_text.splitlines()]
                display_limit = len(display_log_lines)
                if display_log_lines:
                    st.caption("Showing the failure excerpt. Use Load Full History for the complete log.")
            else:
                display_limit = FINISHED_LOG_DISPLAY_LIMIT
                display_log_lines = all_log_lines[-display_limit:]

            if not failure_info and len(all_log_lines) > len(display_log_lines):
                st.caption(
                    f"Showing latest {len(display_log_lines)} of {len(all_log_lines)} log lines "
                    "to keep the live UI responsive."
                )

            if not VIEW_RUN_DIR and not display_log_lines:
                st.info(
                    "No run is selected yet. Configure inputs in the setup panel on the left "
                    "(use the upper-left chevron if the panel is hidden), then start a run to see "
                    "live logs, progress, artifacts, and results here."
                )
            else:
                formatted_logs = "<br>".join([format_log_line(line.rstrip()) for line in display_log_lines])

                # Unique IDs to prevent collision and "gray out" jank
                run_slug = re.sub(r'[^a-zA-Z0-9]', '_', VIEW_RUN_DIR.name if VIEW_RUN_DIR else st.session_state.get('run_name', 'default'))
                container_id = f"log-container-{run_slug}"

                # Log viewer container
                st.markdown(f'<div id="{container_id}" class="log-viewer">{formatted_logs}</div>', unsafe_allow_html=True)

                # 2. MutationObserver Autoscroll Logic (Most Robust)
                if st.session_state.log_autoscroll:
                    cache_buster = len(display_log_lines)
                    st.markdown(f"""
                        <script data-cache="{cache_buster}">
                        (function() {{
                            const container = window.parent.document.getElementById('{container_id}');
                            if (!container) return;

                            // Immediate scroll to bottom
                            container.scrollTop = container.scrollHeight;

                            // Setup Observer to watch for new lines added by React/Streamlit
                            const observer = new MutationObserver(() => {{
                                container.scrollTop = container.scrollHeight;
                            }});

                            observer.observe(container, {{
                                childList: true,
                                subtree: true,
                                characterData: true
                            }});

                            // Fallback safety scrolls for fragments
                            setTimeout(() => {{ container.scrollTop = container.scrollHeight; }}, 100);
                            setTimeout(() => {{ container.scrollTop = container.scrollHeight; }}, 500);
                        }})();
                        </script>
                        """, unsafe_allow_html=True)

        render_logs_and_monitor()

        # Pipeline Progress Timeline
        st.markdown("---")
        st.markdown("### Rapid Progress" if _is_rapid_context() else "### Pipeline Progress")

        @st.fragment(run_every=LIVE_REFRESH_SECONDS if VIEW_RUN_APPEARS_RUNNING else None)
        def render_pipeline_progress():
            rapid_artifacts_ready = _is_rapid_context() and _rapid_report_root() is not None
            failure_info = _run_failure_info(VIEW_RUN_DIR) if VIEW_RUN_DIR and not st.session_state.run_active else None
            run_complete = bool(
                VIEW_RUN_DIR
                and not VIEW_RUN_APPEARS_RUNNING
                and VIEW_RUN_MANIFEST_STATUS in {"complete", "completed"}
                and not failure_info
            )
            if VIEW_RUN_APPEARS_RUNNING or st.session_state.run_finished or rapid_artifacts_ready or failure_info or run_complete:
                state = dict(st.session_state.pipeline_state)
                if VIEW_RUN_APPEARS_RUNNING and not st.session_state.run_active and VIEW_RUN_DIR:
                    inferred_rapid = _run_analysis_mode_from_dir(VIEW_RUN_DIR) == "Rapid Hypothesis Mode"
                    state["rapid_mode"] = bool(inferred_rapid or state.get("rapid_mode") or _is_rapid_context())
                    inferred_idx = _infer_run_stage_from_events(VIEW_RUN_DIR, rapid_mode=bool(state.get("rapid_mode")))
                    state["global_stage_idx"] = inferred_idx
                    state["stage0_status"] = "complete" if inferred_idx > 0 else "running"
                if rapid_artifacts_ready and not VIEW_RUN_APPEARS_RUNNING and not st.session_state.run_active and not st.session_state.run_finished:
                    state["rapid_mode"] = True
                    state["global_stage_idx"] = len(RAPID_STAGES)
                    state["stage0_status"] = "complete"
                if run_complete:
                    completed_rapid = VIEW_RUN_MODE == "Rapid Hypothesis Mode" or _is_rapid_context()
                    state["rapid_mode"] = completed_rapid
                    state["global_stage_idx"] = len(RAPID_STAGES) if completed_rapid else len(GLOBAL_STAGES)
                    state["stage0_status"] = "complete"
                if failure_info:
                    state["rapid_mode"] = bool(failure_info.get("rapid_mode") or state.get("rapid_mode") or _is_rapid_context())
                    state["global_stage_idx"] = int(failure_info.get("stage_idx", state.get("global_stage_idx", 0)) or 0)
                g_idx = state["global_stage_idx"]
                rapid_timeline = bool(state.get("rapid_mode") or _is_rapid_context())
                stage_names = RAPID_STAGES if rapid_timeline else GLOBAL_STAGES
                stage_notes = RAPID_STAGE_NOTES if rapid_timeline else []
                rapid_icon_labels = ["1.", "2.", "3.", "4.", "5.", "6."]
                standard_icon_labels = ["Step 0", "Step 1", "Step 2", "Passes", "Final"]
                failed_idx = None
                if failure_info:
                    failed_idx = max(0, min(int(failure_info.get("stage_idx", g_idx) or 0), len(stage_names) - 1))

                html_parts = ['<div class="timeline-container">']

                for i, stage_name in enumerate(stage_names):
                    is_last = (i == len(stage_names) - 1)
                    item_class = "timeline-item"
                    if is_last: item_class += " last"

                    # Determine status
                    if i == 0:
                        status = state["stage0_status"]
                        if status == "complete": status_class = "complete"
                        elif status == "running": status_class = "active"
                        elif status == "skipped": status_class = "complete"
                        else: status_class = ""
                    else:
                        if failed_idx is not None and i == failed_idx:
                            status_class = "failed"
                        elif i < g_idx:
                            status_class = "complete"
                        elif i == g_idx:
                            status_class = "active"
                        else:
                            status_class = ""
                    if failed_idx is not None and i == failed_idx:
                        status_class = "failed"

                    html_parts.append(f'<div class="{item_class} {status_class}">')
                    html_parts.append('<div class="timeline-dot"></div>')
                    html_parts.append('<div class="timeline-content">')


                    if rapid_timeline:
                        icon = rapid_icon_labels[i] if i < len(rapid_icon_labels) else "Step"
                    else:
                        icon = standard_icon_labels[i] if i < len(standard_icon_labels) else "Step"
                    title = f"{icon} {stage_name}"
                    html_parts.append(f'<div class="timeline-title">{title}</div>')
                    if rapid_timeline and i < len(stage_notes):
                        html_parts.append(f'<div class="timeline-subtitle">{html.escape(stage_notes[i])}</div>')
                    if failed_idx is not None and i == failed_idx:
                        reason = str(failure_info.get("reason") or "Run failed.")
                        html_parts.append(f'<div class="timeline-subtitle timeline-failure">Failed: {html.escape(reason[:220])}</div>')

                    # Sub-stages for Pass
                    if not rapid_timeline and i == 3 and i == g_idx:
                        curr_pass = state["current_pass"]
                        curr_p_stage = state["pass_stage"]
                        html_parts.append(f'<div class="timeline-subtitle">Pass {curr_pass} in progress</div>')
                        html_parts.append('<div class="sub-steps">')
                        for s_key, s_name in PASS_STAGES:
                            sub_class = "sub-step"
                            if s_key == curr_p_stage: sub_class += " active"
                            html_parts.append(f'<div class="{sub_class}">{s_name}</div>')
                        html_parts.append('</div>')

                    html_parts.append('</div></div>')

                html_parts.append('</div>')
                st.markdown("".join(html_parts), unsafe_allow_html=True)
            else:
                if VIEW_RUN_DIR and VIEW_RUN_IS_SELECTED:
                    st.info("This saved run does not include pipeline progress metadata.")
                else:
                    st.info("Configure inputs in the setup panel, then start a run to see live pipeline progress.")

        render_pipeline_progress()
        render_magnetic_precheck_panel(VIEW_RUN_DIR, compact=True)

    with c2:
        if VIEW_RUN_DIR:
            render_run_config_summary(
                VIEW_RUN_DIR / "pipeline_config.yaml",
                title="Run configuration",
                expanded=st.session_state.run_active,
            )
        st.subheader("Artifacts")
        @st.fragment(run_every=ARTIFACT_REFRESH_SECONDS if VIEW_RUN_APPEARS_RUNNING else None)
        def render_artifacts_fragment():
            if VIEW_RUN_DIR:
                rdir = VIEW_RUN_DIR
                if _is_rapid_context():
                    render_rapid_artifacts(rdir / "rapid_results")
                    return
                p_dir_new = rdir / "Results" / "Plots"
                diag_dir = rdir / "Diagnostics"
                p_dir_old = rdir / "plots"
                sub_plots = rdir / st.session_state.get('run_name', '') / "plots"

                if p_dir_new.exists():
                    st.markdown("**Plots**")
                    has_plots = render_file_explorer(
                        p_dir_new,
                        "art_new",
                        [".png", ".jpg", ".pdf"],
                        hide_predicate=_hide_curated_artifact,
                        show_downloads=False,
                    )
                    if not has_plots:
                        st.caption("No plots generated yet.")

                if diag_dir.exists():
                    st.markdown("**Diagnostics**")
                    has_diag = render_file_explorer(
                        diag_dir,
                        "art_diag",
                        [".png", ".jpg", ".pdf"],
                        hide_predicate=_hide_curated_artifact,
                        show_downloads=False,
                    )
                    if not has_diag:
                        st.caption("No diagnostics generated yet.")

                if not p_dir_new.exists() and not diag_dir.exists():
                    if p_dir_old.exists():
                        render_file_explorer(
                            p_dir_old,
                            "art_root",
                            [".png", ".jpg", ".pdf"],
                            hide_predicate=_hide_curated_artifact,
                            show_downloads=False,
                        )
                    elif sub_plots.exists():
                        render_file_explorer(
                            sub_plots,
                            "art_sub",
                            [".png", ".jpg", ".pdf"],
                            hide_predicate=_hide_curated_artifact,
                            show_downloads=False,
                        )
                    else:
                        st.info("No plots directory found yet.")
            else:
                report_root = _rapid_report_root() if _is_rapid_context() else None
                if report_root is not None:
                    render_rapid_artifacts(report_root)
                else:
                    st.info("Start a run to see artifacts.")

        render_artifacts_fragment()

elif active_run_view == "Results":
    if VIEW_RUN_DIR:
        rdir = VIEW_RUN_DIR
        results_failure_info = _run_failure_info(rdir) if not st.session_state.run_active else None
        if results_failure_info:
            st.error(f"Run failed: {results_failure_info.get('reason') or 'Pipeline exited with an error.'}")
            st.caption(
                "Partial artifacts below may be useful for diagnosis, but they are not a completed RADAR-PD result."
            )
            if results_failure_info.get("returncode") is not None:
                st.caption(f"Exit code: `{results_failure_info['returncode']}`")
        render_magnetic_precheck_panel(rdir)

        # --- ML Ranker Results ---
        @st.fragment(run_every=10.0 if st.session_state.run_active else None)
        def render_ml_results():
            diag_dir = rdir / "Diagnostics"
            if diag_dir.exists():
                result_files = list(diag_dir.glob("ml_rank_result_pass*.jsonl"))
                status_files = list(diag_dir.glob("ml_rank_status_pass*.json"))
                input_files = list(diag_dir.glob("ml_rank_input_pass*.json"))

                def _pass_ix(path: Path) -> int:
                    match = re.search(r"pass(\d+)", path.name)
                    return int(match.group(1)) if match else 0

                result_by_pass = {_pass_ix(path): path for path in result_files}
                status_by_pass = {_pass_ix(path): path for path in status_files}
                input_by_pass = {_pass_ix(path): path for path in input_files}
                all_passes = sorted(set(result_by_pass) | set(status_by_pass) | set(input_by_pass))

                if all_passes:
                    st.subheader("ML Ranker Diagnostics")
                    st.info("**ML Score**: Higher (less negative) is better. Represents relative relevance weight.")
                    enrich_ranker_metadata = st.checkbox(
                        "Show compound names and space groups",
                        value=True,
                        help="Display catalog labels for the ranker table. Turn off only if a live refresh becomes too slow.",
                        key="enrich_ranker_metadata_default_on",
                    )

                    for pass_ix in all_passes:
                        f_path = result_by_pass.get(pass_ix)
                        status_path = status_by_pass.get(pass_ix)
                        input_path = input_by_pass.get(pass_ix)
                        with st.expander(f"Pass: pass{pass_ix}", expanded=(pass_ix == all_passes[-1])):
                            if f_path:
                                try:
                                    data = load_first_json_record(f_path)
                                    if "ranked" in data:
                                        import pandas as pd
                                        df = pd.DataFrame(data["ranked"])
                                        # Optional: this loads the full catalog, so keep it opt-in for responsiveness.
                                        if enrich_ranker_metadata:
                                            db = get_active_db_loader()
                                        else:
                                            db = None
                                        if db:
                                            names, sgs, sg_nums = [], [], []
                                            for pid in df["mp_id"]:
                                                try:
                                                    names.append(db.get_pretty_name(pid))
                                                    sgs.append(db.get_space_group_symbol(pid) or "-")
                                                    sg_nums.append(db.get_space_group_number(pid) or "-")
                                                except Exception as e:
                                                    names.append("unknown")
                                                    sgs.append("-")
                                                    sg_nums.append("-")
                                                    print(f"[ui] Warning: metadata lookup failed for {pid}: {e}")

                                            df.insert(2, "Compound", names)
                                            df.insert(3, "Space Group", sgs)
                                            df.insert(4, "SG #", sg_nums)

                                        st.dataframe(_safe_dataframe_for_streamlit(df), hide_index=True, width='stretch')
                                    else:
                                        st.caption("No ranking data found in output.")
                                except Exception as e:
                                    st.caption(f"Could not read ML ranker result: {e}")

                            if status_path:
                                try:
                                    status_data = json.loads(status_path.read_text(encoding="utf-8"))
                                except Exception as e:
                                    st.caption(f"Could not read ranker status: {e}")
                                    continue

                                status_value = status_data.get("status", "unknown")
                                if status_value == "missing_assets":
                                    st.warning(f"ML ranker unavailable: {status_data.get('error', 'missing checkpoint or script')}")
                                elif status_value in {"failed", "failed_no_output", "failed_readback", "failed_exception"}:
                                    st.error(f"ML ranker failed: {status_data.get('error') or status_data.get('note') or status_data.get('stderr') or 'unknown error'}")
                                elif status_value == "complete":
                                    st.caption(f"Ranker completed with {status_data.get('n_ranked', 0)} ranked candidates.")
                                elif status_value == "complete_empty":
                                    st.info(status_data.get("note", "ML ranker completed but returned no ranked candidates."))
                                elif status_value == "input_ready":
                                    st.caption("Ranker input prepared; waiting for result.")
                            elif input_path:
                                st.warning("Ranker input was generated, but no status/result artifact was found for this run.")
                else:
                    st.caption("No ML ranker diagnostics generated yet.")

        render_ml_results()

        # Metrics Overview (Optional, maybe keep it simple)
        mpath_str = cached_find_run_manifest_path(str(rdir))
        mpath = Path(mpath_str) if mpath_str else None
        if mpath and mpath.exists():
            try:
                m = cached_read_json_file(str(mpath), _path_mtime(mpath))
                mets = m.get("metrics", {})
                stage1_result = ((m.get("stages") or {}).get("Stage 1") or {}).get("result", {}) or {}

                def _fmt_percent(value):
                    try:
                        return f"{float(value):.2f}%"
                    except Exception:
                        return "-"

                c_status, c_final, c_stage1, c_phases = st.columns(4)
                c_status.metric("Run Status", str(m.get("status", "processing")).title())
                final_label = "Partial Final Rwp" if str(m.get("status", "")).lower() == "failed" else "Final Rwp"
                phases_label = "Partial Phases Found" if str(m.get("status", "")).lower() == "failed" else "Phases Found"
                c_final.metric(final_label, _fmt_percent(mets.get("final_rwp")))
                c_stage1.metric("Stage 1 Rwp", _fmt_percent(stage1_result.get("rwp")))
                c_phases.metric(phases_label, str(mets.get("phases_found", "-")))

                calib_status = stage1_result.get("calibration_status")
                calib_note = stage1_result.get("calibration_note")
                calibrated_instprm = stage1_result.get("calibrated_instprm")
                if calib_status == "adopted":
                    st.success(f"PXRD light calibration applied. {calib_note or ''}".strip())
                elif calib_status == "rejected":
                    st.warning(f"PXRD light calibration was not adopted. {calib_note or ''}".strip())
                elif calib_status == "failed":
                    st.error(f"PXRD light calibration failed. {calib_note or ''}".strip())
                elif calib_status == "skipped":
                    st.info(f"PXRD light calibration skipped. {calib_note or ''}".strip())
                if calibrated_instprm:
                    st.caption(f"Calibrated instrument profile: {calibrated_instprm}")
            except Exception as e:
                st.caption(f"Could not read run manifest yet: {e}")

        st.markdown("---")
        st.subheader("Generated Data Sheets" if not results_failure_info else "Partial Data Sheets")

        # Find all CSV files recursively
        csv_files = [(Path(path), mtime) for path, mtime in cached_run_csv_files(str(rdir))]

        if csv_files:
            primary_csvs = [entry for entry in csv_files if entry[0].name == "Summary_Fractions.csv"]
            other_csvs = [entry for entry in csv_files if entry[0].name != "Summary_Fractions.csv"]

            def _render_csv_group(files, *, primary=False):
                for idx, (fcsv, mtime) in enumerate(files):
                    label = f"{fcsv.name}"
                    if primary:
                        label = f"{fcsv.name}"
                    with st.expander(label, expanded=(primary and idx == 0)):
                        try:
                            df = cached_read_csv_file(str(fcsv), mtime)
                            st.dataframe(_safe_dataframe_for_streamlit(df), width="stretch")
                            st.download_button(
                                f"Download {fcsv.name}",
                                fcsv.read_bytes(),
                                file_name=fcsv.name,
                                key=f"dl_res_{fcsv.as_posix()}",
                            )
                        except Exception as e:
                            st.error(f"Could not load {fcsv.name}: {e}")

            def _render_selected_csv(files, *, key_prefix: str) -> None:
                if not files:
                    return
                options = [str(path) for path, _ in files]
                mtime_by_path = {str(path): mtime for path, mtime in files}
                selected_path = st.selectbox(
                    "Data sheet",
                    options,
                    format_func=lambda value: Path(value).name,
                    key=f"{key_prefix}_csv_selector",
                    help="Only the selected table is loaded, which keeps large runs responsive.",
                )
                selected_csv = Path(selected_path)
                selected_mtime = mtime_by_path.get(str(selected_csv))
                try:
                    df = cached_read_csv_file(str(selected_csv), selected_mtime)
                    st.dataframe(_safe_dataframe_for_streamlit(df), width="stretch")
                    st.download_button(
                        f"Download {selected_csv.name}",
                        selected_csv.read_bytes(),
                        file_name=selected_csv.name,
                        key=f"dl_res_selected_{selected_csv.as_posix()}",
                    )
                except Exception as e:
                    st.error(f"Could not load {selected_csv.name}: {e}")

            if primary_csvs:
                st.markdown("**Primary Output**")
                _render_csv_group(primary_csvs, primary=True)

            if other_csvs:
                st.markdown(f"**Other CSV Artifacts** ({len(other_csvs)} files)")
                _render_selected_csv(other_csvs, key_prefix="other_results")
        else:
            st.info("No CSV data files generated yet.")
    else:
        st.subheader("Results")
        _render_no_run_workspace_empty_state(
            "Results",
            details="Results will show summary metrics, phase tables, ML diagnostics, and generated data sheets after a run is loaded.",
        )

elif active_run_view == "Interactive Plots":
    st.subheader("Interactive Plots")
    st.caption("Inspect one generated plot at a time with linked fit, residual, phase, and candidate metadata.")

    if not INTERACTIVE_PLOTS_AVAILABLE:
        st.warning("Interactive plotting helpers are unavailable in this environment.")
    else:
        recent_runs = cached_recent_interactive_runs(str(ACTIVE_RUNS_ROOT))
        run_options: list[dict] = []
        seen_run_paths = set()
        session_run_dir = VIEW_RUN_DIR
        if session_run_dir and Path(session_run_dir).exists():
            current_path = str(Path(session_run_dir))
            seen_run_paths.add(current_path)
            run_options.append(
                {
                    "name": Path(session_run_dir).name,
                    "path": current_path,
                    "plot_count": -1,
                    "mtime": 0.0,
                    "label": f"Current session: {Path(session_run_dir).name}",
                }
            )
        for entry in recent_runs:
            if entry["path"] in seen_run_paths:
                continue
            seen_run_paths.add(entry["path"])
            run_options.append(entry)

        if not run_options:
            if not VIEW_RUN_DIR:
                _render_no_run_workspace_empty_state(
                    "Interactive Plots",
                    details="Interactive plots will show linked fit, residual, phase-tick, and candidate metadata for a selected run.",
                )
            else:
                st.info("This run does not have interactive plot payloads yet.")
        else:
            selected_run = st.selectbox(
                "Run to inspect",
                run_options,
                format_func=lambda item: item["label"],
                key="interactive_plot_run_selector",
                help="You can inspect previous completed runs even after the browser session refreshes.",
            )
            rdir = Path(selected_run["path"])
            plot_failure_info = _run_failure_info(rdir) if not st.session_state.run_active else None
            if plot_failure_info:
                st.warning(
                    "This run failed before completion. Interactive plots below are partial diagnostic artifacts, "
                    "not final validated results."
                )
                st.caption(f"Failure: {plot_failure_info.get('reason') or 'Pipeline exited with an error.'}")

            payload_files = discover_plot_payload_files(rdir)
            plot_infos = []
            for payload_path in payload_files:
                header = _read_plot_payload_header(payload_path)
                group = _plot_payload_group(header)
                source = str(header.get("source_plot") or payload_path.name)
                title = str(header.get("title") or "")
                plot_infos.append(
                    {
                        "path": payload_path,
                        "header": header,
                        "group": group,
                        "label": _plot_display_label(payload_path, header, rdir),
                        "pass_ix": _plot_pass_number(source) or _plot_pass_number(title),
                    }
                )
            plot_infos.sort(key=_plot_sort_key)

            if not plot_infos:
                st.info("This run has no supported interactive plot payloads yet.")
            else:
                group_counts: dict[str, int] = {}
                for info in plot_infos:
                    group_counts[info["group"]] = group_counts.get(info["group"], 0) + 1

                _render_plot_meta_cards(
                    [
                        ("Run", rdir.name),
                        ("Payloads", len(plot_infos)),
                        ("Fit plots", group_counts.get("Fit / refinement", 0)),
                        ("ML histograms", group_counts.get("ML screening histograms", 0)),
                    ],
                    numeric_labels={"Payloads", "Fit plots", "ML histograms"},
                )

                all_groups = [group for group in ["Fit / refinement", "ML screening histograms", "Residual debug", "Other"] if group_counts.get(group)]
                selected_groups = st.multiselect(
                    "Plot types",
                    all_groups,
                    default=all_groups,
                    key="interactive_plot_type_filter",
                )

                filtered_infos = [info for info in plot_infos if info["group"] in selected_groups]
                if not filtered_infos:
                    st.info("No plots match the selected filters.")
                else:
                    selector_key = "interactive_plot_selector_" + hashlib.md5(str(rdir).encode("utf-8")).hexdigest()[:10]
                    selected_plot_index = st.selectbox(
                        "Plot",
                        list(range(len(filtered_infos))),
                        format_func=lambda i: f"{filtered_infos[i]['group']} | {filtered_infos[i]['label']}",
                        key=selector_key,
                    )
                    info = filtered_infos[int(selected_plot_index)]
                    payload_path = info["path"]
                    rel_path = payload_path.relative_to(rdir) if payload_path.is_relative_to(rdir) else payload_path

                    plot_config = {
                        "scrollZoom": True,
                        "displaylogo": False,
                        "responsive": True,
                        "toImageButtonOptions": {"format": "png", "scale": 2},
                        "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
                    }

                    try:
                        payload = load_interactive_payload(payload_path)
                        payload = _rapid_repair_plot_payload_labels(payload, payload_path)
                    except Exception as e:
                        st.error(f"Could not read interactive payload from {payload_path.name}: {e}")
                        payload = None

                    if payload is not None:
                        st.markdown(f"**{info['group']} | {info['label']}**")
                        _render_plot_meta_cards(
                            [
                                ("Plot type", info["group"]),
                                ("Pass", info.get("pass_ix") or "-"),
                                ("Payload", payload_path.name.replace(".plotdata.json", "")),
                            ],
                            numeric_labels={"Pass"} if info.get("pass_ix") else set(),
                        )

                        max_hist_candidates = 6
                        if info["group"] == "ML screening histograms":
                            max_hist_candidates = st.slider(
                                "Candidate panels",
                                min_value=3,
                                max_value=24,
                                value=6,
                                step=3,
                                help="Lower values keep interaction responsive while preserving the full payload on disk.",
                            )

                        if info["group"] == "Fit / refinement":
                            phase_order = payload.get("phase_order", []) or []
                            phase_labels = payload.get("phase_labels", {}) or {}
                            phase_weights = payload.get("phase_weights", {}) or {}
                            phase_ticks = payload.get("phase_ticks", {}) or {}
                            phase_major_ticks = payload.get("phase_major_ticks", {}) or {}
                            rows = []
                            for phase_id in phase_order:
                                label = str(phase_labels.get(phase_id) or phase_id)
                                try:
                                    wt = round(float(phase_weights.get(phase_id)), 4)
                                except Exception:
                                    wt = None
                                rows.append(
                                    {
                                        "Phase": label,
                                        "Weight %": wt,
                                        "Ticks": len(phase_ticks.get(phase_id, []) or []),
                                        "Key peaks": len(phase_major_ticks.get(phase_id, []) or []),
                                    }
                                )
                            if rows and pd is not None:
                                with st.expander("Phase summary", expanded=True):
                                    st.dataframe(_safe_dataframe_for_streamlit(pd.DataFrame(rows)), hide_index=True, width="stretch")
                            _rapid_render_peak_support_summary(payload)

                        if info["group"] == "ML screening histograms":
                            rows = []
                            for rank, cand in enumerate((payload.get("candidates") or [])[:24], start=1):
                                def _round_metric(value):
                                    try:
                                        return round(float(value), 4)
                                    except Exception:
                                        return None
                                rows.append(
                                    {
                                        "Rank": rank,
                                        "Candidate": cand.get("label") or cand.get("phase_id"),
                                        "Score": _round_metric(cand.get("score")),
                                        "Cosine": _round_metric(cand.get("cosine")),
                                        "Present prob": _round_metric(cand.get("present_prob")),
                                    }
                                )
                            if rows and pd is not None:
                                with st.expander("Candidate summary", expanded=True):
                                    st.dataframe(_safe_dataframe_for_streamlit(pd.DataFrame(rows)), hide_index=True, width="stretch")

                        try:
                            fig = build_plotly_figure_from_payload(
                                payload,
                                max_hist_candidates=max_hist_candidates,
                            )
                            if fig is None:
                                st.warning("This payload type is not supported yet.")
                            else:
                                st.plotly_chart(
                                    fig,
                                    width="stretch",
                                    key="int_plot_selected_" + str(payload_path).replace("\\", "_").replace("/", "_"),
                                    config=plot_config,
                                )
                        except Exception as e:
                            st.error(f"Could not render interactive plot from {payload_path.name}: {e}")

                        source_plot = payload.get("source_plot_path")
                        with st.expander("Files", expanded=False):
                            st.caption(f"Payload: `{rel_path}`")
                            if source_plot and Path(source_plot).exists():
                                st.caption(f"Static source: `{source_plot}`")

                        if pd is not None:
                            with st.expander("All plot payloads in this run", expanded=False):
                                st.dataframe(
                                    _safe_dataframe_for_streamlit(
                                        pd.DataFrame(
                                            [
                                                {
                                                    "Type": item["group"],
                                                    "Plot": item["label"],
                                                    "Pass": item.get("pass_ix") or "-",
                                                    "Payload": str(item["path"].relative_to(rdir)) if item["path"].is_relative_to(rdir) else str(item["path"]),
                                                }
                                                for item in plot_infos
                                            ]
                                        )
                                    ),
                                    hide_index=True,
                                    width="stretch",
                                )
elif active_run_view == "Rapid Results":
    @st.fragment(run_every=LIVE_REFRESH_SECONDS if st.session_state.run_active else None)
    def render_rapid_results_fragment():
        render_rapid_hypothesis_explorer()

    render_rapid_results_fragment()

elif active_run_view == "Run File Browser":
    c1, c2 = st.columns([0.8, 0.2])
    with c1:
        st.subheader("Run File Browser")
    with c2:
        if st.button("Refresh", width="stretch"):
            st.rerun()

    if VIEW_RUN_DIR:
        # Show full file tree
        rdir = VIEW_RUN_DIR
        file_failure_info = _run_failure_info(rdir) if not st.session_state.run_active else None
        if file_failure_info:
            st.warning(
                "This run failed before completion. Files below are partial run artifacts preserved for diagnosis."
            )
            st.caption(f"Failure: {file_failure_info.get('reason') or 'Pipeline exited with an error.'}")
        if not rdir.exists():
            st.error(f"Run directory not found: {rdir}")
        else:
            # Check for pipeline.log presence
            lfile = rdir / "pipeline.log"
            if not lfile.exists():
                st.warning("pipeline.log is not available in the run directory yet.")

            show_full_tree = st.toggle(
                "Show full recursive file tree",
                value=False,
                help="The full tree can be slow for large runs. The default view lists key output files only.",
            )
            if show_full_tree:
                render_file_explorer(rdir, "exp_root", None) # No filter, show all
            else:
                render_key_run_files(rdir)
    else:
        _render_no_run_workspace_empty_state(
            "Run File Browser",
            details="The file browser will list logs, configuration files, GPX projects, plots, and tables for a selected run.",
        )
