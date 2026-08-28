from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .aniso_db_loader import DBLoader, CatalogPaths, build_mask
    from .database_catalog_builder import process_one_phase
    from .db_pack import build_db_config, get_db_pack_layout
except ImportError:
    from aniso_db_loader import DBLoader, CatalogPaths, build_mask
    from database_catalog_builder import process_one_phase
    from db_pack import build_db_config, get_db_pack_layout

try:
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    HAVE_PYMATGEN_MATCHER = True
except Exception:
    Structure = None  # type: ignore[assignment]
    StructureMatcher = None  # type: ignore[assignment]
    SpacegroupAnalyzer = None  # type: ignore[assignment]
    HAVE_PYMATGEN_MATCHER = False


XRAY_DEFAULT_WAVELENGTH = 0.4133
NEUTRON_DEFAULT_WAVELENGTH = 1.54184
DEFAULT_TWO_THETA_MIN = 0.0
XRAY_DEFAULT_TWO_THETA_MAX = 180.0
XRAY_DEFAULT_TOP_M = 500
NEUTRON_DEFAULT_TWO_THETA_MAX = 90.0
# Finalized custom neutron builder contract:
# keep the 0..90 profile-generation window, but use the same top-M cut as xray.
NEUTRON_DEFAULT_TOP_M = XRAY_DEFAULT_TOP_M
DEFAULT_Q_MIN = 0.5
DEFAULT_Q_MAX = 6.0
DEFAULT_N_BINS = 64
DEFAULT_SIGMA_BINS = 0.7
XRAY_Q_BUFFER_DEG = 10.0


@dataclass(frozen=True)
class SimulationSettings:
    source_type: str
    radiation: str
    wavelength: Optional[float]
    two_theta_min: float
    two_theta_max: float
    topM: int
    q_min: float
    q_max: float
    n_bins: int
    sigma_bins: float
    source_note: str = "default"

    def to_manifest_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PhaseInput:
    phase_id: str
    source_name: str
    cif_path: Path
    cif_content: str


@dataclass(frozen=True)
class BuildResult:
    pack_root: Path
    layout: Any
    db_config: Dict[str, str]
    manifest_path: Path
    phase_ids: List[str]
    failures: List[Dict[str, str]]


@dataclass
class BaseDuplicateIndex:
    candidate_ids_by_key: Dict[Tuple[int, int, int], List[str]]
    matcher: Any
    structure_cache: Dict[str, Any]


ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit_progress(
    progress_callback: ProgressCallback,
    *,
    step: str,
    message: str,
    fraction: float,
    current: Optional[int] = None,
    total: Optional[int] = None,
    source_name: Optional[str] = None,
    started_at: Optional[float] = None,
    stage_started_at: Optional[float] = None,
    **metrics: Any,
) -> None:
    if progress_callback is None:
        return
    payload: Dict[str, Any] = {
        "step": step,
        "message": message,
        "fraction": max(0.0, min(1.0, float(fraction))),
    }
    if current is not None:
        payload["current"] = int(current)
    if total is not None:
        payload["total"] = int(total)
    if source_name:
        payload["source_name"] = str(source_name)
    now = time.perf_counter()
    if started_at is not None:
        payload["elapsed_s"] = max(0.0, now - float(started_at))
    if stage_started_at is not None:
        payload["stage_elapsed_s"] = max(0.0, now - float(stage_started_at))
    for key, value in metrics.items():
        if value is not None:
            payload[key] = value
    progress_callback(payload)


def _sanitize_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return token or "phase"


_CIF_PHASE_NAME_TAGS = (
    "_pd_phase_name",
    "_chemical_name_common",
    "_chemical_name_mineral",
    "_chemical_name_systematic",
)
_UNHELPFUL_PHASE_NAMES = {
    "#(c)",
    "global",
    "none",
    "unknown",
    "vesta_phase_1",
    "?",
    ".",
}


def _cif_scalar_value(cif_content: str, tags: Sequence[str]) -> str:
    """Read a quoted or unquoted scalar CIF tag without treating IDs as names."""

    for tag in tags:
        match = re.search(
            rf"(?im)^\s*{re.escape(tag)}\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))",
            cif_content,
        )
        if not match:
            continue
        value = next((part for part in match.groups() if part is not None), "").strip()
        if value and value.casefold() not in _UNHELPFUL_PHASE_NAMES:
            return value
    return ""


def infer_phase_display_name(source_name: str, cif_content: str, formula: str) -> str:
    """Choose a stable scientific label while retaining formula separately."""

    reduced_formula = re.sub(r"\s+", "", str(formula or "").strip())
    declared_name = _cif_scalar_value(cif_content, _CIF_PHASE_NAME_TAGS)
    if reduced_formula and declared_name:
        formula_key = re.sub(r"[^a-z0-9]+", "", reduced_formula.casefold())
        name_key = re.sub(r"[^a-z0-9]+", "", declared_name.casefold())
        if formula_key == name_key:
            return reduced_formula
        if formula_key and formula_key in name_key:
            return declared_name
        return f"{reduced_formula} - {declared_name}"
    if reduced_formula:
        return reduced_formula
    if declared_name:
        return declared_name
    filename_label = re.sub(r"[_-]+", " ", Path(source_name).stem).strip()
    return filename_label or "Unknown phase"


def _pick_workers(n_items: int) -> int:
    if n_items <= 1:
        return 1
    try:
        cpus = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except Exception:
        cpus = os.cpu_count() or 1
    if os.environ.get("SPACE_ID"):
        return max(1, min(n_items, cpus, 2))
    return max(1, min(n_items, cpus, 8))


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _phase_id_for_cif(path: Path, cif_content: str, prefix: str = "user") -> str:
    stem = _sanitize_token(path.stem)
    digest = hashlib.sha1(cif_content.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{stem}_{digest}"


def collect_phase_inputs(cif_paths: Sequence[str | Path], prefix: str = "user") -> List[PhaseInput]:
    seen_content_digests: set[str] = set()
    out: List[PhaseInput] = []
    for raw in cif_paths:
        path = Path(raw).expanduser().resolve()
        cif_text = path.read_text(encoding="utf-8")
        content_digest = hashlib.sha1(cif_text.encode("utf-8")).hexdigest()
        if content_digest in seen_content_digests:
            continue
        seen_content_digests.add(content_digest)
        phase_id = _phase_id_for_cif(path, cif_text, prefix=prefix)
        out.append(PhaseInput(
            phase_id=phase_id,
            source_name=path.name,
            cif_path=path,
            cif_content=cif_text,
        ))
    return out


def _load_profiles64_context(profiles_dir: Path) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float]]:
    npz_path = profiles_dir / "profiles64.npz"
    idx_path = profiles_dir / "index.csv"
    if not npz_path.exists():
        raise FileNotFoundError(f"profiles64.npz not found: {npz_path}")
    if not idx_path.exists():
        raise FileNotFoundError(f"index.csv not found: {idx_path}")
    with np.load(npz_path) as z:
        profiles = z["profiles"]
        params = {
            "q_min": float(z["q_min"]),
            "q_max": float(z["q_max"]),
            "n_bins": int(z["n_bins"]),
            "sigma_bins": float(z["sigma_bins"]),
        }
    index_df = pd.read_csv(idx_path)
    return profiles, index_df, params


def _optimized_xray_two_theta_max(q_max: float, wavelength: float, buffer_deg: float = XRAY_Q_BUFFER_DEG) -> float:
    needed_sin_theta = (float(q_max) * float(wavelength)) / (4.0 * np.pi)
    if needed_sin_theta >= 1.0:
        return 180.0
    theta_deg = np.degrees(np.arcsin(needed_sin_theta))
    return min(180.0, float(2.0 * theta_deg + buffer_deg))


def resolve_simulation_settings(
    source_type: str,
    *,
    base_db_root: Optional[str | Path] = None,
    wavelength_override: Optional[float] = None,
) -> SimulationSettings:
    source = str(source_type).strip().lower()
    if source not in {"xray", "neutron"}:
        raise ValueError(f"Unsupported source_type: {source_type}")

    radiation = source
    wavelength = XRAY_DEFAULT_WAVELENGTH if source == "xray" else NEUTRON_DEFAULT_WAVELENGTH
    two_theta_min = DEFAULT_TWO_THETA_MIN
    two_theta_max = XRAY_DEFAULT_TWO_THETA_MAX if source == "xray" else NEUTRON_DEFAULT_TWO_THETA_MAX
    topM = XRAY_DEFAULT_TOP_M if source == "xray" else NEUTRON_DEFAULT_TOP_M
    q_min = DEFAULT_Q_MIN
    q_max = DEFAULT_Q_MAX
    n_bins = DEFAULT_N_BINS
    sigma_bins = DEFAULT_SIGMA_BINS
    source_note = "default"

    if base_db_root is not None:
        layout = get_db_pack_layout(base_db_root)
        if layout.manifest_json.exists():
            manifest = json.loads(layout.manifest_json.read_text(encoding="utf-8"))
            tt = manifest.get("two_theta_range_deg")
            if isinstance(tt, (list, tuple)) and len(tt) == 2:
                two_theta_min = float(tt[0])
                two_theta_max = float(tt[1])
            if manifest.get("topM") is not None:
                topM = int(manifest["topM"])
            if source == "xray" and manifest.get("wavelength") is not None:
                wavelength = float(manifest["wavelength"])
            if manifest.get("radiation"):
                radiation = str(manifest["radiation"]).strip().lower()
            source_note = "base_pack_manifest"
        if layout.profiles_npz.exists():
            with np.load(layout.profiles_npz) as z:
                q_min = float(z["q_min"])
                q_max = float(z["q_max"])
                n_bins = int(z["n_bins"])
                sigma_bins = float(z["sigma_bins"])
            if source_note == "default":
                source_note = "base_pack_profiles64"
            else:
                source_note = f"{source_note}+profiles64"

    if wavelength_override is not None:
        wavelength = float(wavelength_override)
        source_note = f"{source_note}+wavelength_override"

    if source == "xray":
        two_theta_min = DEFAULT_TWO_THETA_MIN
        two_theta_max = _optimized_xray_two_theta_max(q_max, wavelength)
        source_note = f"{source_note}+xray_qmax_window"

    return SimulationSettings(
        source_type=source,
        radiation=radiation,
        wavelength=wavelength,
        two_theta_min=two_theta_min,
        two_theta_max=two_theta_max,
        topM=topM,
        q_min=q_min,
        q_max=q_max,
        n_bins=n_bins,
        sigma_bins=sigma_bins,
        source_note=source_note,
    )


def _build_profile_from_phase_npz(npz_path: Path, settings: SimulationSettings) -> np.ndarray:
    with np.load(npz_path) as z:
        q0 = z["q0"].astype(np.float32)
        I0 = z["I0"].astype(np.float32)

    profile = np.zeros(settings.n_bins, dtype=np.float32)
    if q0.size == 0:
        return profile.astype(np.float16)

    I0 = I0 / max(1e-12, float(I0.max()))
    dq = (settings.q_max - settings.q_min) / float(settings.n_bins)
    for q, w in zip(q0, I0):
        if not (settings.q_min <= float(q) < settings.q_max):
            continue
        center_bin = (float(q) - settings.q_min) / dq - 0.5
        _gaussian_deposit(profile, center_bin, float(w), settings.sigma_bins)
    m = float(profile.max())
    if m > 0:
        profile /= m
    return profile.astype(np.float16)


def _gaussian_deposit(profile: np.ndarray, center_bin: float, weight: float, sigma_bins: float) -> None:
    if sigma_bins <= 0:
        j = int(round(center_bin))
        if 0 <= j < profile.shape[0]:
            profile[j] += weight
        return
    halfw = int(max(1, np.ceil(3.0 * sigma_bins)))
    j0 = int(round(center_bin))
    jL = max(0, j0 - halfw)
    jR = min(profile.shape[0] - 1, j0 + halfw)
    if jR < jL:
        return
    js = np.arange(jL, jR + 1, dtype=np.float32)
    g = np.exp(-0.5 * ((js - center_bin) / sigma_bins) ** 2)
    profile[jL:jR + 1] += weight * g


def _prepare_output_dirs(layout: Any, *, overwrite: bool) -> None:
    if layout.root.exists():
        if any(layout.root.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Output pack already exists and is not empty: {layout.root}")
            shutil.rmtree(layout.root)
    layout.root.mkdir(parents=True, exist_ok=True)
    layout.profiles_dir.mkdir(parents=True, exist_ok=True)
    layout.phases_dir.mkdir(parents=True, exist_ok=True)
    layout.cifs_dir.mkdir(parents=True, exist_ok=True)


def _materialize_phase(
    phase: PhaseInput,
    settings: SimulationSettings,
    layout: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    dest_cif = layout.cifs_dir / f"{phase.phase_id}.cif"
    dest_cif.write_text(phase.cif_content, encoding="utf-8")
    row, fail, _ = process_one_phase(
        phase.phase_id,
        {"cif_content": phase.cif_content},
        str(layout.root),
        settings.two_theta_min,
        settings.two_theta_max,
        settings.topM,
        False,
        False,
        False,
        False,
        settings.radiation,
        settings.wavelength if settings.wavelength is not None else NEUTRON_DEFAULT_WAVELENGTH,
    )
    if row is None:
        msg = "unknown failure"
        if fail:
            msg = str(fail.get("error") or fail)
        return None, None, None, {"id": phase.phase_id, "source_name": phase.source_name, "error": msg}

    row["display_name"] = infer_phase_display_name(
        phase.source_name,
        phase.cif_content,
        str(row.get("pretty_formula") or ""),
    )
    profile = _build_profile_from_phase_npz(layout.root / row["npz"], settings)
    stable_row = {
        "material_id": phase.phase_id,
        "formula_pretty": row["pretty_formula"],
        "spacegroup_number": int(row["space_group"]),
        "spacegroup_symbol": row["SG_symbol"],
        "energy_above_hull_eV_per_atom": -1.0,
    }
    meta_row = {
        "cif_content": phase.cif_content,
        "composition": {},
        "formula_pretty": row["pretty_formula"],
        "display_name": row["display_name"],
        "space_group": int(row["space_group"]),
        "SG_symbol": row["SG_symbol"],
        "source_name": phase.source_name,
    }
    return row, profile, stable_row, None


def _materialize_phase_worker(args: Tuple[PhaseInput, SimulationSettings, str]) -> Tuple[str, str, Optional[Dict[str, Any]], Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Dict[str, str]]]:
    phase, settings, layout_root = args
    layout = get_db_pack_layout(layout_root)
    row, profile, stable_row, failure = _materialize_phase(phase, settings, layout)
    return phase.phase_id, phase.source_name, row, profile, stable_row, failure


def _write_profiles_bundle(layout: Any, profiles: np.ndarray, index_df: pd.DataFrame, settings: SimulationSettings) -> None:
    index_df.to_csv(layout.profiles_index_csv, index=False)
    np.savez_compressed(
        layout.profiles_npz,
        profiles=profiles,
        q_min=settings.q_min,
        q_max=settings.q_max,
        n_bins=settings.n_bins,
        sigma_bins=settings.sigma_bins,
    )


def _write_manifest(layout: Any, payload: Dict[str, Any]) -> None:
    layout.manifest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_cif_map(layout: Any, phase_ids: Iterable[str]) -> None:
    mapping = {pid: f"cifs/{pid}.cif" for pid in phase_ids}
    layout.cif_map_json.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def _write_original_json(layout: Any, phase_inputs: Iterable[PhaseInput], meta_rows: Dict[str, Dict[str, Any]]) -> None:
    payload: Dict[str, Dict[str, Any]] = {}
    for phase in phase_inputs:
        meta = dict(meta_rows.get(phase.phase_id, {}))
        meta["cif_content"] = phase.cif_content
        payload[phase.phase_id] = meta
    layout.original_json.write_text(json.dumps(payload), encoding="utf-8")


def _normalize_catalog_df(catalog_df: pd.DataFrame) -> pd.DataFrame:
    out = catalog_df.copy()
    for col in ("space_group", "n_reflections"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise").astype("Int64")
    for col in ("elements_mask_hi", "elements_mask_lo"):
        if col in out.columns:
            out[col] = out[col].map(
                lambda value: pd.NA
                if value is None or str(value).strip() == ""
                else int(value)
            ).astype("UInt64")
    for col in ("id", "pretty_formula", "SG_symbol", "elements_list", "npz"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "display_name" in out.columns:
        out["display_name"] = out["display_name"].fillna("").astype(str)
    return out


def _normalize_stable_df(stable_df: pd.DataFrame) -> pd.DataFrame:
    out = stable_df.copy()
    if "spacegroup_number" in out.columns:
        out["spacegroup_number"] = pd.to_numeric(out["spacegroup_number"], errors="raise").astype("Int64")
    for col in ("material_id", "formula_pretty", "spacegroup_symbol"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def _build_base_frames(base_db_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame]:
    layout = get_db_pack_layout(base_db_root)
    catalog_df = pd.read_csv(layout.catalog_csv, keep_default_na=False)
    stable_df = pd.read_csv(layout.stable_csv, keep_default_na=False)
    profiles, index_df, _ = _load_profiles64_context(layout.profiles_dir)
    return catalog_df, stable_df, profiles.astype(np.float16, copy=False), index_df


def _build_base_duplicate_index(base_loader: DBLoader) -> Optional[BaseDuplicateIndex]:
    if not HAVE_PYMATGEN_MATCHER:
        return None
    catalog = base_loader.catalog
    required = {"id", "elements_mask_hi", "elements_mask_lo", "space_group"}
    if not required.issubset(set(catalog.columns)):
        return None

    ids = [str(pid) for pid in getattr(base_loader, "_ids", catalog["id"].astype(str).to_numpy())]
    hi_values = getattr(base_loader, "_m_hi", catalog["elements_mask_hi"].astype(object).map(int).to_numpy())
    lo_values = getattr(base_loader, "_m_lo", catalog["elements_mask_lo"].astype(object).map(int).to_numpy())
    sg_values = getattr(base_loader, "_sg_values", pd.to_numeric(catalog["space_group"], errors="coerce").to_numpy())

    candidate_ids_by_key: Dict[Tuple[int, int, int], List[str]] = {}
    for pid, hi, lo, sg in zip(ids, hi_values, lo_values, sg_values):
        if pd.isna(sg):
            continue
        key = (int(hi), int(lo), int(sg))
        candidate_ids_by_key.setdefault(key, []).append(str(pid))

    return BaseDuplicateIndex(
        candidate_ids_by_key=candidate_ids_by_key,
        matcher=StructureMatcher(primitive_cell=True, scale=True, attempt_supercell=False),  # type: ignore[operator]
        structure_cache={},
    )

def _find_matching_base_phase_ids(
    phase: PhaseInput,
    *,
    base_loader: DBLoader,
    duplicate_index: Optional[BaseDuplicateIndex] = None,
) -> List[str]:
    """
    Detect whether an uploaded CIF already exists in the base DB under a different id.

    We narrow by exact element mask and space group first, then use StructureMatcher
    for the final check. This is only used during pack build, so correctness matters
    more than shaving a small amount of runtime.
    """
    if not HAVE_PYMATGEN_MATCHER:
        return []

    try:
        ref_structure = Structure.from_file(str(phase.cif_path))  # type: ignore[union-attr]
        ref_elements = sorted({str(el) for el in ref_structure.composition.as_dict().keys()})
        ref_hi, ref_lo = build_mask(ref_elements)
        ref_sg = int(SpacegroupAnalyzer(ref_structure, symprec=1e-2, angle_tolerance=5.0).get_space_group_number())  # type: ignore[misc]
    except Exception:
        return []

    if duplicate_index is None:
        duplicate_index = _build_base_duplicate_index(base_loader)
    if duplicate_index is None:
        cat = base_loader.catalog.copy()
        cat["id"] = cat["id"].astype(str)
        cat = cat[
            (cat["elements_mask_hi"].astype(object).map(int) == int(ref_hi)) &
            (cat["elements_mask_lo"].astype(object).map(int) == int(ref_lo))
        ]
        if "space_group" in cat.columns:
            cat = cat[pd.to_numeric(cat["space_group"], errors="coerce") == ref_sg]

        candidate_ids = cat["id"].astype(str).tolist()
        if not candidate_ids:
            return []

        matcher = StructureMatcher(primitive_cell=True, scale=True, attempt_supercell=False)  # type: ignore[operator]
        matches: List[str] = []
        for pid in candidate_ids:
            try:
                cand_structure = base_loader.load_structure(pid)
            except Exception:
                continue
            try:
                if matcher.fit(ref_structure, cand_structure):
                    matches.append(pid)
            except Exception:
                continue
        return matches

    candidate_ids = duplicate_index.candidate_ids_by_key.get((int(ref_hi), int(ref_lo), int(ref_sg)), [])
    if not candidate_ids:
        return []

    matches: List[str] = []
    for pid in candidate_ids:
        try:
            if pid not in duplicate_index.structure_cache:
                duplicate_index.structure_cache[pid] = base_loader.load_structure(pid)
            cand_structure = duplicate_index.structure_cache[pid]
        except Exception:
            continue
        try:
            if duplicate_index.matcher.fit(ref_structure, cand_structure):
                matches.append(pid)
        except Exception:
            continue
    return matches


def _build_phase_batch(
    phase_inputs: Sequence[PhaseInput],
    *,
    settings: SimulationSettings,
    layout: Any,
    progress_callback: ProgressCallback,
    progress_start: float,
    progress_span: float,
    build_started_at: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[np.ndarray], Dict[str, Dict[str, Any]], List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, Any]] = []
    stable_rows: List[Dict[str, Any]] = []
    profiles: List[np.ndarray] = []
    meta_rows: Dict[str, Dict[str, Any]] = {}
    failures: List[Dict[str, str]] = []
    phase_ids: List[str] = []

    total_phases = len(phase_inputs)
    if total_phases == 0:
        return rows, stable_rows, profiles, meta_rows, failures, phase_ids

    workers = _pick_workers(total_phases)
    stage_started_at = time.perf_counter()
    _emit_progress(
        progress_callback,
        step="materialize",
        message=f"Building {total_phases} phase profile(s) with {workers} worker(s)",
        fraction=progress_start,
        current=0,
        total=total_phases,
        started_at=build_started_at,
        stage_started_at=stage_started_at,
        workers=workers,
        built_count=0,
        failed_count=0,
    )

    args_iter = [(phase, settings, str(layout.root)) for phase in phase_inputs]
    if workers == 1:
        iterator = map(_materialize_phase_worker, args_iter)
    else:
        chunksize = max(1, total_phases // max(1, workers * 4))
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_worker_init)
        iterator = pool.map(_materialize_phase_worker, args_iter, chunksize=chunksize)

    completed = 0
    try:
        for phase_id, source_name, row, profile, stable_row, failure in iterator:
            completed += 1
            if failure is not None:
                failures.append(failure)
            else:
                rows.append(row)
                stable_rows.append(stable_row)
                profiles.append(profile)
                meta_rows[phase_id] = {
                    "composition": {},
                    "formula_pretty": row["pretty_formula"],
                    "display_name": row["display_name"],
                    "space_group": int(row["space_group"]),
                    "SG_symbol": row["SG_symbol"],
                    "source_name": source_name,
                }
                phase_ids.append(phase_id)

            _emit_progress(
                progress_callback,
                step="materialize",
                message=(
                    f"Built profiles {completed}/{total_phases}; "
                    f"usable {len(rows)}, failed {len(failures)}"
                ),
                fraction=progress_start + progress_span * (completed / max(1, total_phases)),
                current=completed,
                total=total_phases,
                source_name=source_name,
                started_at=build_started_at,
                stage_started_at=stage_started_at,
                workers=workers,
                built_count=len(rows),
                failed_count=len(failures),
            )
    finally:
        if workers != 1:
            pool.shutdown(wait=True)

    return rows, stable_rows, profiles, meta_rows, failures, phase_ids


def build_mini_db_pack(
    cif_paths: Sequence[str | Path],
    output_root: str | Path,
    *,
    source_type: str,
    reference_db_root: Optional[str | Path] = None,
    wavelength_override: Optional[float] = None,
    overwrite: bool = False,
    progress_callback: ProgressCallback = None,
) -> BuildResult:
    build_started_at = time.perf_counter()
    raw_input_count = len(cif_paths)
    phase_inputs = collect_phase_inputs(cif_paths)
    if not phase_inputs:
        raise ValueError("No CIF files provided")
    total_phases = len(phase_inputs)
    duplicate_upload_count = max(0, raw_input_count - total_phases)
    _emit_progress(
        progress_callback,
        step="collect",
        message=(
            f"Collected {total_phases} unique CIF file(s)"
            + (f"; ignored {duplicate_upload_count} duplicate upload(s)" if duplicate_upload_count else "")
        ),
        fraction=0.05,
        current=0,
        total=total_phases,
        started_at=build_started_at,
        input_files=raw_input_count,
        unique_cifs=total_phases,
        duplicate_upload_count=duplicate_upload_count,
    )

    settings = resolve_simulation_settings(
        source_type,
        base_db_root=reference_db_root,
        wavelength_override=wavelength_override,
    )
    layout = get_db_pack_layout(output_root)
    _prepare_output_dirs(layout, overwrite=overwrite)
    _emit_progress(
        progress_callback,
        step="prepare",
        message=f"Prepared output pack at {layout.root}",
        fraction=0.12,
        current=0,
        total=total_phases,
        started_at=build_started_at,
        unique_cifs=total_phases,
    )

    rows, stable_rows, profiles, meta_rows, failures, phase_ids = _build_phase_batch(
        phase_inputs,
        settings=settings,
        layout=layout,
        progress_callback=progress_callback,
        progress_start=0.12,
        progress_span=0.70,
        build_started_at=build_started_at,
    )

    if not rows:
        raise RuntimeError(f"All CIF phases failed to build: {failures}")

    catalog_df = pd.DataFrame(rows, columns=[
        "id", "display_name", "pretty_formula", "space_group", "SG_symbol",
        "elements_list", "elements_mask_hi", "elements_mask_lo",
        "npz", "n_reflections",
    ])
    stable_df = pd.DataFrame(stable_rows)
    index_df = pd.DataFrame({"id": phase_ids, "row": list(range(len(phase_ids)))})
    profiles_arr = np.vstack(profiles).astype(np.float16, copy=False)

    catalog_df = _normalize_catalog_df(catalog_df)
    stable_df = _normalize_stable_df(stable_df)
    _emit_progress(
        progress_callback,
        step="write",
        message=f"Writing mini catalog with {len(phase_ids)} phase profile(s)",
        fraction=0.88,
        current=len(phase_ids),
        total=total_phases,
        started_at=build_started_at,
        built_count=len(phase_ids),
        failed_count=len(failures),
    )
    catalog_df.to_csv(layout.catalog_csv, index=False)
    stable_df.to_csv(layout.stable_csv, index=False)
    _write_profiles_bundle(layout, profiles_arr, index_df, settings)
    _write_cif_map(layout, phase_ids)
    _write_original_json(layout, phase_inputs, meta_rows)

    manifest = {
        "kind": "mini",
        "created_utc": _utc_now(),
        "n_phases": len(phase_ids),
        "source_type": settings.source_type,
        "simulation": settings.to_manifest_dict(),
        "phase_ids": phase_ids,
        "failures": failures,
    }
    _write_manifest(layout, manifest)
    _emit_progress(
        progress_callback,
        step="finalize",
        message=f"Built mini pack with {len(phase_ids)} usable phase(s); skipped/failed {len(failures)}",
        fraction=1.0,
        current=len(phase_ids),
        total=total_phases,
        started_at=build_started_at,
        built_count=len(phase_ids),
        failed_count=len(failures),
        skipped_count=len(failures),
    )

    db_config = build_db_config(
        layout.root,
        original_json=layout.original_json,
        cif_map_json=layout.cif_map_json,
    )
    return BuildResult(
        pack_root=layout.root,
        layout=layout,
        db_config=db_config,
        manifest_path=layout.manifest_json,
        phase_ids=phase_ids,
        failures=failures,
    )


def build_augmented_db_pack(
    cif_paths: Sequence[str | Path],
    output_root: str | Path,
    *,
    source_type: str,
    base_db_root: str | Path,
    wavelength_override: Optional[float] = None,
    overwrite: bool = False,
    progress_callback: ProgressCallback = None,
) -> BuildResult:
    build_started_at = time.perf_counter()
    raw_input_count = len(cif_paths)
    phase_inputs = collect_phase_inputs(cif_paths)
    if not phase_inputs:
        raise ValueError("No CIF files provided")
    total_phases = len(phase_inputs)
    duplicate_upload_count = max(0, raw_input_count - total_phases)
    _emit_progress(
        progress_callback,
        step="collect",
        message=(
            f"Collected {total_phases} unique CIF file(s)"
            + (f"; ignored {duplicate_upload_count} duplicate upload(s)" if duplicate_upload_count else "")
        ),
        fraction=0.05,
        current=0,
        total=total_phases,
        started_at=build_started_at,
        input_files=raw_input_count,
        unique_cifs=total_phases,
        duplicate_upload_count=duplicate_upload_count,
    )

    settings = resolve_simulation_settings(
        source_type,
        base_db_root=base_db_root,
        wavelength_override=wavelength_override,
    )
    layout = get_db_pack_layout(output_root)
    _prepare_output_dirs(layout, overwrite=overwrite)
    _emit_progress(
        progress_callback,
        step="prepare",
        message=f"Prepared output pack at {layout.root}",
        fraction=0.12,
        current=0,
        total=total_phases,
        started_at=build_started_at,
        unique_cifs=total_phases,
    )

    base_load_started_at = time.perf_counter()
    base_catalog_df, base_stable_df, base_profiles, base_index_df = _build_base_frames(Path(base_db_root))
    base_layout = get_db_pack_layout(base_db_root)
    base_loader = DBLoader(CatalogPaths(
        catalog_csv=str(base_layout.catalog_csv),
        cif_map_json=str(base_layout.cif_map_json) if base_layout.cif_map_json.exists() else None,
        original_json=str(base_layout.original_json) if base_layout.original_json.exists() else None,
    ))
    duplicate_index = _build_base_duplicate_index(base_loader)
    _emit_progress(
        progress_callback,
        step="base-load",
        message=f"Loaded base database with {len(base_catalog_df)} phase(s); checking uploaded CIFs",
        fraction=0.20,
        current=0,
        total=total_phases,
        started_at=build_started_at,
        stage_started_at=base_load_started_at,
        base_phase_count=len(base_catalog_df),
        unique_cifs=total_phases,
        duplicate_index_keys=len(duplicate_index.candidate_ids_by_key) if duplicate_index is not None else 0,
    )

    failures: List[Dict[str, str]] = []
    buildable_phases: List[PhaseInput] = []
    skipped_existing_id_count = 0
    skipped_matching_structure_count = 0

    existing_ids = set(base_catalog_df["id"].astype(str))
    precheck_started_at = time.perf_counter()
    for idx, phase in enumerate(phase_inputs, start=1):
        _emit_progress(
            progress_callback,
            step="precheck",
            message=f"Checking CIF {idx}/{total_phases}: {phase.source_name}",
            fraction=0.20 + 0.60 * ((idx - 1) / max(1, total_phases)),
            current=idx - 1,
            total=total_phases,
            source_name=phase.source_name,
            started_at=build_started_at,
            stage_started_at=precheck_started_at,
            checked_count=idx - 1,
            queued_count=len(buildable_phases),
            skipped_count=len(failures),
            skipped_existing_id_count=skipped_existing_id_count,
            skipped_matching_structure_count=skipped_matching_structure_count,
        )
        if phase.phase_id in existing_ids:
            failures.append({
                "id": phase.phase_id,
                "source_name": phase.source_name,
                "error": "phase id already exists in base catalog",
            })
            skipped_existing_id_count += 1
            _emit_progress(
                progress_callback,
                step="precheck",
                message=(
                    f"Checked {idx}/{total_phases} CIFs; "
                    f"queued {len(buildable_phases)} new, skipped {len(failures)}"
                ),
                fraction=0.20 + 0.60 * (idx / max(1, total_phases)),
                current=idx,
                total=total_phases,
                source_name=phase.source_name,
                started_at=build_started_at,
                stage_started_at=precheck_started_at,
                checked_count=idx,
                queued_count=len(buildable_phases),
                skipped_count=len(failures),
                skipped_existing_id_count=skipped_existing_id_count,
                skipped_matching_structure_count=skipped_matching_structure_count,
            )
            continue
        matched_base_ids = _find_matching_base_phase_ids(
            phase,
            base_loader=base_loader,
            duplicate_index=duplicate_index,
        )
        if matched_base_ids:
            failures.append({
                "id": phase.phase_id,
                "source_name": phase.source_name,
                "error": f"phase already exists in base database as {', '.join(matched_base_ids[:5])}",
            })
            skipped_matching_structure_count += 1
            _emit_progress(
                progress_callback,
                step="precheck",
                message=(
                    f"Checked {idx}/{total_phases} CIFs; "
                    f"queued {len(buildable_phases)} new, skipped {len(failures)}"
                ),
                fraction=0.20 + 0.60 * (idx / max(1, total_phases)),
                current=idx,
                total=total_phases,
                source_name=phase.source_name,
                started_at=build_started_at,
                stage_started_at=precheck_started_at,
                checked_count=idx,
                queued_count=len(buildable_phases),
                skipped_count=len(failures),
                skipped_existing_id_count=skipped_existing_id_count,
                skipped_matching_structure_count=skipped_matching_structure_count,
            )
            continue
        buildable_phases.append(phase)
        _emit_progress(
            progress_callback,
            step="precheck",
            message=(
                f"Checked {idx}/{total_phases} CIFs; "
                f"queued {len(buildable_phases)} new, skipped {len(failures)}"
            ),
            fraction=0.20 + 0.60 * (idx / max(1, total_phases)),
            current=idx,
            total=total_phases,
            source_name=phase.source_name,
            started_at=build_started_at,
            stage_started_at=precheck_started_at,
            checked_count=idx,
            queued_count=len(buildable_phases),
            skipped_count=len(failures),
            skipped_existing_id_count=skipped_existing_id_count,
            skipped_matching_structure_count=skipped_matching_structure_count,
        )

    _emit_progress(
        progress_callback,
        step="precheck",
        message=(
            f"Precheck complete: {len(buildable_phases)} new CIF(s), "
            f"{len(failures)} duplicate/skipped CIF(s)"
        ),
        fraction=0.80,
        current=total_phases,
        total=total_phases,
        started_at=build_started_at,
        stage_started_at=precheck_started_at,
        checked_count=total_phases,
        queued_count=len(buildable_phases),
        skipped_count=len(failures),
        skipped_existing_id_count=skipped_existing_id_count,
        skipped_matching_structure_count=skipped_matching_structure_count,
    )

    rows, stable_rows, profiles, _meta_rows, new_failures, phase_ids = _build_phase_batch(
        buildable_phases,
        settings=settings,
        layout=layout,
        progress_callback=progress_callback,
        progress_start=0.80,
        progress_span=0.08,
        build_started_at=build_started_at,
    )
    failures.extend(new_failures)

    if not rows:
        raise RuntimeError(f"No new phases were built for augmentation: {failures}")

    new_catalog_df = pd.DataFrame(rows, columns=[
        "id", "display_name", "pretty_formula", "space_group", "SG_symbol",
        "elements_list", "elements_mask_hi", "elements_mask_lo",
        "npz", "n_reflections",
    ])
    new_stable_df = pd.DataFrame(stable_rows)
    new_profiles_arr = np.vstack(profiles).astype(np.float16, copy=False)

    merged_catalog_df = pd.concat([base_catalog_df, new_catalog_df], ignore_index=True)
    merged_stable_df = pd.concat([base_stable_df, new_stable_df], ignore_index=True)
    merged_stable_df = merged_stable_df.drop_duplicates(subset="material_id", keep="first")

    base_rows = int(base_profiles.shape[0])
    new_index_df = pd.DataFrame({"id": phase_ids, "row": list(range(base_rows, base_rows + len(phase_ids)))})
    merged_index_df = pd.concat([base_index_df, new_index_df], ignore_index=True)
    merged_profiles = np.vstack([base_profiles, new_profiles_arr]).astype(np.float16, copy=False)

    merged_catalog_df = _normalize_catalog_df(merged_catalog_df)
    merged_stable_df = _normalize_stable_df(merged_stable_df)
    _emit_progress(
        progress_callback,
        step="write",
        message=(
            f"Writing merged catalog with {base_rows} base phase(s) "
            f"and {len(phase_ids)} new phase profile(s)"
        ),
        fraction=0.88,
        current=len(phase_ids),
        total=total_phases,
        started_at=build_started_at,
        base_phase_count=base_rows,
        built_count=len(phase_ids),
        skipped_count=len(failures),
        failed_count=len(new_failures),
    )
    merged_catalog_df.to_csv(layout.catalog_csv, index=False)
    merged_stable_df.to_csv(layout.stable_csv, index=False)
    _write_profiles_bundle(layout, merged_profiles, merged_index_df, settings)
    _write_cif_map(layout, phase_ids)

    manifest = {
        "kind": "augmented",
        "created_utc": _utc_now(),
        "base_db_root": str(Path(base_db_root).resolve()),
        "base_original_json": str(base_layout.original_json.resolve()),
        "n_base_phases": int(base_profiles.shape[0]),
        "n_added_phases": len(phase_ids),
        "source_type": settings.source_type,
        "simulation": settings.to_manifest_dict(),
        "phase_ids": phase_ids,
        "failures": failures,
    }
    _write_manifest(layout, manifest)
    _emit_progress(
        progress_callback,
        step="finalize",
        message=f"Built augmented pack with {len(phase_ids)} new phase(s); skipped/failed {len(failures)}",
        fraction=1.0,
        current=len(phase_ids),
        total=total_phases,
        started_at=build_started_at,
        base_phase_count=int(base_profiles.shape[0]),
        built_count=len(phase_ids),
        skipped_count=len(failures),
        failed_count=len(new_failures),
        skipped_existing_id_count=skipped_existing_id_count,
        skipped_matching_structure_count=skipped_matching_structure_count,
    )

    db_config = build_db_config(
        layout.root,
        original_json=str(base_layout.original_json.resolve()),
        cif_map_json=layout.cif_map_json,
    )
    return BuildResult(
        pack_root=layout.root,
        layout=layout,
        db_config=db_config,
        manifest_path=layout.manifest_json,
        phase_ids=phase_ids,
        failures=failures,
    )
