"""
Anisotropic Database Loader & Search

This module handles the loading and querying of the specialized crystallographic database.
Key features:
- Loading large JSON/Pickle databases of crystal structures.
- Fast lookups by element composition and space group.
- Utilities for retrieving CIF content and metadata.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.core.periodic_table import Element as PMGElement


# ------------------------------
# Element bitmask utilities
# ------------------------------

_Z_CACHE: Dict[str, int] = {}
ELEMENT_BITS = 128  # Z = 1..128


def build_mask(elements: Iterable[str]) -> Tuple[np.uint64, np.uint64]:
    """Build 128-bit mask from element symbols. Returns (hi, lo) as np.uint64."""
    lo = np.uint64(0)
    hi = np.uint64(0)
    for el in elements:
        z = _Z_CACHE.get(el)
        if z is None:
            z = PMGElement(el).Z
            _Z_CACHE[el] = z
        if not (1 <= z <= ELEMENT_BITS):
            continue
        idx = z - 1
        if idx < 64:
            lo |= (np.uint64(1) << np.uint64(idx))
        else:
            hi |= (np.uint64(1) << np.uint64(idx - 64))
    return hi, lo


def is_subset_mask(m_hi: np.uint64, m_lo: np.uint64, M_hi: np.uint64, M_lo: np.uint64) -> bool:
    """Check candidate mask m is subset of main mask M using (m & ~M) == 0 on both halves."""
    return (m_hi & ~M_hi) == 0 and (m_lo & ~M_lo) == 0


# ---- tiny bitmask helpers used by CIF-free filtering ----
def _popcount_u128(hi: np.uint64, lo: np.uint64) -> int:
    return int(hi).bit_count() + int(lo).bit_count()


def _mask_or(a_hi: np.uint64, a_lo: np.uint64, b_hi: np.uint64, b_lo: np.uint64) -> Tuple[np.uint64, np.uint64]:
    return (np.uint64(int(a_hi) | int(b_hi)), np.uint64(int(a_lo) | int(b_lo)))


def _mask_and(a_hi: np.uint64, a_lo: np.uint64, b_hi: np.uint64, b_lo: np.uint64) -> Tuple[np.uint64, np.uint64]:
    return (np.uint64(int(a_hi) & int(b_hi)), np.uint64(int(a_lo) & int(b_lo)))


def _mask_and_not(a_hi: np.uint64, a_lo: np.uint64, b_hi: np.uint64, b_lo: np.uint64) -> Tuple[np.uint64, np.uint64]:
    # a & ~b
    return (np.uint64(int(a_hi) & ~int(b_hi)), np.uint64(int(a_lo) & ~int(b_lo)))


def _mask_is_zero(hi: np.uint64, lo: np.uint64) -> bool:
    return int(hi) == 0 and int(lo) == 0


# ------------------------------
# Periodic table helpers (for wildcard relations)
# ------------------------------

def _group_of(el: str) -> Optional[int]:
    try:
        return int(PMGElement(el).group)
    except Exception:
        return None


def _period_of(el: str) -> Optional[int]:
    try:
        return int(PMGElement(el).row)
    except Exception:
        return None


def _family_of(el: str) -> str:
    try:
        e = PMGElement(el)
        if e.is_lanthanoid:
            return "lanthanide"
        if e.is_actinoid:
            return "actinide"
        if e.is_alkali:
            return "alkali"
        if e.is_alkaline:
            return "alkaline_earth"
        if e.is_transition_metal:
            return "transition_metal"
        if e.is_metalloid:
            return "metalloid"
        if e.is_halogen:
            return "halogen"
        if e.is_chalcogen:
            return "chalcogen"
        if e.is_noble_gas:
            return "noble_gas"
        if e.is_post_transition_metal:
            return "post_transition_metal"
        return "other"
    except Exception:
        return "other"


# Cache all chemical symbols (Z=1..118) once for substitution pools
try:
    _ALL_SYMBOLS = tuple(PMGElement.from_Z(z).symbol for z in range(1, 119))
except Exception:
    _ALL_SYMBOLS = tuple()


def _substitution_pool_for_target(target: str, relation: str) -> List[str]:
    """
    For a given target element 'target' and relation mode, return a list of substitute symbols
    (excluding the original target). relation ∈ {"any", "same_group", "same_period", "same_family"}.
    """
    if not _ALL_SYMBOLS:
        return []

    relation = (relation or "any").lower()
    tg = _group_of(target)
    tp = _period_of(target)
    tf = _family_of(target)

    out: List[str] = []
    for s in _ALL_SYMBOLS:
        if s == target:
            continue
        if relation in ("", "any"):
            ok = True
        elif relation == "same_group":
            ok = (_group_of(s) is not None and tg is not None and _group_of(s) == tg)
        elif relation == "same_period":
            ok = (_period_of(s) is not None and tp is not None and _period_of(s) == tp)
        elif relation == "same_family":
            ok = (_family_of(s) == tf)
        else:
            ok = False
        if ok:
            out.append(s)
    return out


# ------------------------------
# Catalog and CIF resolution
# ------------------------------

@dataclass
class CatalogPaths:
    catalog_csv: str
    cif_map_json: Optional[str] = None   # optional mapping {id: "/abs/path/to/id.cif"}
    original_json: Optional[str] = None  # optional JSON: {id: {"cif_content": "..."}}


class DBLoader:
    def _log(self, msg: str) -> None:
        if self.debug:
            print(msg)

    def __init__(self, paths: CatalogPaths):
        self.paths = paths
        self.debug = bool(int(os.environ.get("ANISO_DB_DEBUG", "1")))

        if not os.path.exists(paths.catalog_csv):
            raise FileNotFoundError(f"catalog_csv not found: {paths.catalog_csv}")

        # Load catalog
        self.catalog = pd.read_csv(paths.catalog_csv, keep_default_na=False)
        self._log(f"[DB] catalog loaded: {paths.catalog_csv}")
        self._log(f"[DB] catalog size: {len(self.catalog)} rows")

        # --- Normalize columns (works for different catalog flavors) ---
        cols = set(self.catalog.columns.astype(str))

        # id → 'id'
        if "id" in cols:
            self.catalog["id"] = self.catalog["id"].astype(str)
        elif "material_id" in cols:
            self.catalog["id"] = self.catalog["material_id"].astype(str)
        else:
            raise KeyError("catalog must contain 'id' or 'material_id' column")

        # pretty formula → 'pretty_formula'
        if "pretty_formula" not in cols and "formula_pretty" in cols:
            self.catalog["pretty_formula"] = self.catalog["formula_pretty"]

        # SG number → 'space_group'
        if "space_group" not in cols and "spacegroup_number" in cols:
            self.catalog["space_group"] = pd.to_numeric(self.catalog["spacegroup_number"], errors="coerce")

        # SG symbol (optional)
        if "SG_symbol" not in cols and "spacegroup_symbol" in cols:
            self.catalog["SG_symbol"] = self.catalog["spacegroup_symbol"]

        # --- Required mask columns ---
        if "elements_mask_hi" not in self.catalog.columns or "elements_mask_lo" not in self.catalog.columns:
            raise KeyError("catalog must contain 'elements_mask_hi' and 'elements_mask_lo' columns")

        # --- Required npz column for profile access ---
        if "npz" not in self.catalog.columns:
            raise KeyError("catalog must contain 'npz' column (relative paths to phases/<id>.npz)")

        # Cache ids AFTER normalization
        self._ids = self.catalog["id"].to_numpy()

        # Robust per-row int→uint64 conversion (errors raise)
        def _to_u64(v) -> np.uint64:
            if v is None or (isinstance(v, str) and v.strip() == ""):
                raise ValueError("empty value in elements_mask column")
            return np.uint64(int(v))

        self._m_hi = self.catalog["elements_mask_hi"].astype(object).map(_to_u64).to_numpy()
        self._m_lo = self.catalog["elements_mask_lo"].astype(object).map(_to_u64).to_numpy()

        # Fast row index
        self._row_index: Dict[str, int] = {pid: i for i, pid in enumerate(self._ids)}

        # Optional CIF map
        self._cif_map: Optional[Dict[str, str]] = None
        if paths.cif_map_json:
            if not os.path.exists(paths.cif_map_json):
                raise FileNotFoundError(f"cif_map_json not found: {paths.cif_map_json}")
            with open(paths.cif_map_json) as f:
                self._cif_map = json.load(f)
                if not isinstance(self._cif_map, dict):
                    raise ValueError("cif_map_json must map id -> path")
            self._log(f"[DB] attached CIF map: {paths.cif_map_json}")

        # Optional original JSON
        if paths.original_json and not os.path.exists(paths.original_json):
            raise FileNotFoundError(f"original_json not found: {paths.original_json}")
        if paths.original_json:
            self._log(f"[DB] original_json available: {paths.original_json}")

    # ---------- NPZ path & data ----------
    def phase_npz_path(self, phase_id: str) -> str:
        pid = str(phase_id)
        if pid not in self._row_index:
            raise KeyError(f"phase id not in catalog: {pid}")
        rel = str(self.catalog.iloc[self._row_index[pid]]["npz"])
        base = os.path.dirname(os.path.abspath(self.paths.catalog_csv))
        p = os.path.join(base, rel)
        if not os.path.exists(p):
            raise FileNotFoundError(f"NPZ file not found for {pid}: {p}")
        return p

    def load_npz_fields(self, phase_id: str, *keys: str):
        p = self.phase_npz_path(phase_id)
        with np.load(p) as z:
            out = []
            for k in keys:
                if k not in z.files:
                    raise KeyError(f"NPZ field '{k}' missing in {p}")
                out.append(z[k])
        return out if len(out) > 1 else out[0]

    def load_q0(self, phase_id: str) -> np.ndarray:
        q0 = self.load_npz_fields(phase_id, "q0")
        return np.asarray(q0, dtype=float)

    def load_I0(self, phase_id: str) -> np.ndarray:
        I0 = self.load_npz_fields(phase_id, "I0")
        return np.asarray(I0, dtype=float)

    def load_ratio_hist(self, phase_id: str) -> np.ndarray:
        rh = self.load_npz_fields(phase_id, "ratio_hist")
        return np.asarray(rh, dtype=np.float32)

    # ---------- Pretty name / SG helpers (optional stable attach) ----------
    def attach_stable_catalog(
        self,
        stable_csv_path: str,
        id_col: str = "material_id",
        name_col: str = "formula_pretty",
        sg_col: str = "spacegroup_number",
    ):
        if not os.path.exists(stable_csv_path):
            raise FileNotFoundError(f"stable catalog not found: {stable_csv_path}")
        df = pd.read_csv(stable_csv_path, dtype={id_col: str})
        df[id_col] = df[id_col].astype(str)
        self._pretty_by_id = dict(zip(df[id_col], df[name_col].astype(str)))
        self._sgnum_by_id = {}
        if sg_col in df.columns:
            self._sgnum_by_id = dict(
                zip(df[id_col], pd.to_numeric(df[sg_col], errors="coerce").astype("Int64").astype(object))
            )
        self._log(f"[DB] attached stable catalog: {stable_csv_path} (size={len(df)})")

    def ensure_cif_on_disk(
        self,
        phase_id: str,
        out_dir: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """
        Materialize a CIF file on disk for the given phase_id and return its path.

        Resolution order:
        1) If a mapped CIF path exists in self._cif_map → return it.
        2) If original_json has 'cif_content' for id → write it to <out_dir>/<id>.cif and return path.
        3) Else call load_structure(id) to get a pymatgen Structure and write it to <out_dir>/<id>.cif.

        (This is *not* used by the element filter; CIF access only happens later, e.g., Stage-4.)
        """
        pid = str(phase_id)

        # 0) output dir
        import tempfile
        if out_dir is None:
            out_dir = os.path.join(tempfile.gettempdir(), "cifs_cache_tmp")
        os.makedirs(out_dir, exist_ok=True)

        # 1) mapped path
        if getattr(self, "_cif_map", None) and pid in self._cif_map:
            mapped = self._cif_map[pid]
            if isinstance(mapped, str) and os.path.exists(mapped):
                return mapped

        # 2) embedded CIF content in original JSON
        if self.paths.original_json and os.path.exists(self.paths.original_json):
            recs = _load_original_db(self.paths.original_json)
            rec = recs.get(pid)
            cif_txt = rec.get("cif_content") if rec else None
            if isinstance(cif_txt, str) and cif_txt.strip():
                out_path = os.path.join(out_dir, f"{pid}.cif")
                if overwrite or not os.path.exists(out_path):
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(cif_txt if cif_txt.endswith("\n") else cif_txt + "\n")
                return out_path

        # 3) last resort: build via pymatgen Structure and write
        obj = self.load_structure(pid)  # may raise
        if not isinstance(obj, Structure):
            raise TypeError(f"load_structure({pid}) did not return a pymatgen Structure (got {type(obj)}).")

        out_path = os.path.join(out_dir, f"{pid}.cif")
        if overwrite or not os.path.exists(out_path):
            obj.to(fmt="cif", filename=out_path)
        return out_path

    def get_pretty_name(self, pid: str) -> str:
        pid = str(pid)
        if hasattr(self, "_pretty_by_id") and pid in self._pretty_by_id:
            return self._pretty_by_id[pid]
        row = self.catalog.loc[self.catalog["id"] == pid]
        if row.empty:
            raise KeyError(f"phase id not in catalog: {pid}")
        s = row.iloc[0]
        for k in ("pretty_formula", "formula_pretty", "formula", "pretty_name", "elements_list"):
            val = s.get(k)
            if isinstance(val, str) and val.strip():
                return val
        return "unknown"

    def get_space_group_number(self, pid: str):
        pid = str(pid)
        if hasattr(self, "_sgnum_by_id") and pid in getattr(self, "_sgnum_by_id"):
            v = self._sgnum_by_id[pid]
            return int(v) if v is not None and v == v else None
        row = self.catalog.loc[self.catalog["id"] == pid]
        if row.empty:
            raise KeyError(f"phase id not in catalog: {pid}")
        sg = pd.to_numeric(row.iloc[0].get("space_group"), errors="coerce")
        return int(sg) if pd.notna(sg) else None

    def get_space_group_symbol(self, pid: str) -> Optional[str]:
        pid = str(pid)
        row = self.catalog.loc[self.catalog["id"] == pid]
        if row.empty:
            return None
        # Check SG_symbol first, then SG_symbol if normalized differently
        for k in ("SG_symbol", "spacegroup_symbol", "symbol"):
            s = row.iloc[0].get(k)
            if isinstance(s, str) and s.strip():
                return s.strip()
        return None

    def get_display_name_and_sg(self, pid: str) -> Tuple[str, str]:
        name = self.get_pretty_name(pid)
        num = self.get_space_group_number(pid)
        sym = self.get_space_group_symbol(pid)
        
        if sym and num:
            sg_disp = f"{sym} ({num})"
        elif sym:
            sg_disp = str(sym)
        elif num:
            sg_disp = str(num)
        else:
            sg_disp = "—"
            
        return name, sg_disp

    # ---------- CIF/Structure loading ----------
    def load_structure(self, phase_id: str) -> Structure:
        pid = str(phase_id)
        # 1) explicit map
        if self._cif_map and pid in self._cif_map:
            path = self._cif_map[pid]
            if not os.path.exists(path):
                raise FileNotFoundError(f"CIF path not found for {pid}: {path}")
            return Structure.from_file(path)
        # 2) original JSON with cif_content
        if self.paths.original_json:
            recs = _load_original_db(self.paths.original_json)
            rec = recs.get(pid)
            if not rec or "cif_content" not in rec:
                raise KeyError(f"cif_content missing in original_json for {pid}")
            return Structure.from_str(rec["cif_content"], fmt="cif")
        raise FileNotFoundError(f"No CIF source for {pid}")

    def has_stable_catalog(self) -> bool:
        """Return True if a stable catalog was attached (pretty names/SG available)."""
        return hasattr(self, "_pretty_by_id")

    def filter_to_stable(self, candidate_ids: Iterable[str]) -> List[str]:
        """
        Keep only candidate IDs that exist in the attached stable catalog.
        Requires attach_stable_catalog(...) to have been called.
        """
        if not self.has_stable_catalog():
            raise RuntimeError("Stable catalog not attached; call attach_stable_catalog(...) first.")
        st = set(self._pretty_by_id.keys())
        return [str(pid) for pid in candidate_ids if str(pid) in st]

    def stable_size(self) -> int:
        """How many entries are in the attached stable catalog (0 if none)."""
        return len(getattr(self, "_pretty_by_id", {}))

    # ---------- Vectorized mask subset ----------
    def filter_by_mask_pair_fast(
        self,
        M_hi: np.uint64,
        M_lo: np.uint64,
        candidate_ids: Optional[Iterable[str]] = None,
    ) -> List[str]:
        """Fast subset test using cached uint64 arrays. Excludes rows with zero mask."""
        if candidate_ids is None:
            ok = ((self._m_hi & ~M_hi) == 0) & ((self._m_lo & ~M_lo) == 0)
            nonzero = (self._m_hi | self._m_lo) != np.uint64(0)
            return self._ids[ok & nonzero].tolist()

        # map candidate ids to indices; error if any id not found
        idx = []
        for pid in candidate_ids:
            pid_s = str(pid)
            if pid_s not in self._row_index:
                raise KeyError(f"candidate id not in catalog: {pid_s}")
            idx.append(self._row_index[pid_s])
        idx = np.asarray(idx, dtype=np.int64)

        ok_sub = ((self._m_hi[idx] & ~M_hi) == 0) & ((self._m_lo[idx] & ~M_lo) == 0)
        nonzero = (self._m_hi[idx] | self._m_lo[idx]) != np.uint64(0)
        return self._ids[idx[ok_sub & nonzero]].tolist()

    # ---------- Main elemental filter (variant-list wildcard; CIF-free) ----------
    def filter_by_element_mask(
        self,
        allowed_elements: list[str],
        candidate_ids: list[str] | None = None,
        *,
        ignore_elements: list[str] | None = None,
        require_base: bool = True,
        max_offlist_elements: int = 0,
        disallow_offlist: list[str] | None = None,
        wildcard_relation: str = "any",
        sample_env: dict | None = None,
        disallow_pure: list[str] | None = None,
    ) -> list[str]:
        """
        Variant-list wildcard implementation (one-slot) using *only 128-bit masks*.
        No CIF/Structure I/O in this filter.

        When max_offlist_elements == 0, reduces to a single strict mask (allowed ∪ ignore).
        When >= 1, we generate per-target variants by substituting that target with elements
        from the chosen relation pool and OR the results across variants.
        Sample-env / require_base / disallow_pure gates are applied via masks.
        """
        dprint = self._log

        # ---- defaults (inherit from instance) ----
        _defaults = getattr(self, "element_filter_defaults", None) or {}
        if ignore_elements is None:
            ignore_elements = list(_defaults.get("ignore_elements", []))
        if disallow_offlist is None:
            disallow_offlist = list(_defaults.get("disallow_offlist", []))
        if sample_env is None:
            sample_env = dict(_defaults.get("sample_env", {}) or {})
        if disallow_pure is None:
            disallow_pure = list(_defaults.get("disallow_pure", []))
        if wildcard_relation in (None, "", "any"):
            wildcard_relation = str(_defaults.get("wildcard_relation", "any"))
        if max_offlist_elements == 0:
            max_offlist_elements = int(_defaults.get("max_offlist_elements", 0))
        if require_base is True and "require_base" in _defaults:
            require_base = bool(_defaults["require_base"])

        allowed_set  = set(map(str, allowed_elements or []))
        ignore_set   = set(map(str, ignore_elements or []))
        disallow_set = set(map(str, disallow_offlist or []))
        disallow_pure_set = set(map(str, disallow_pure or []))

        se_elems_set      = set(map(str, sample_env.get("elements", [])))
        se_allow_with_set = set(map(str, sample_env.get("allow_with", [])))
        se_allow_pure     = bool(sample_env.get("allow_pure", True))
        se_ban_cross      = bool(sample_env.get("ban_cross_with_base", True))

        dprint(f"[elem] config → allowed={sorted(allowed_set)}  ignore={sorted(ignore_set)}")
        dprint(f"[elem] gates  → require_base={require_base}  wildcard({max_offlist_elements}, {wildcard_relation})")
        dprint(f"[elem] disallow_offlist={sorted(disallow_set)}  disallow_pure={sorted(disallow_pure_set)}")
        dprint(f"[elem] SE     → elements={sorted(se_elems_set)} allow_with={sorted(se_allow_with_set)} "
               f"allow_pure={se_allow_pure} ban_cross={se_ban_cross}")

        # ---- precompute masks for gates ----
        ALL_hi, ALL_lo   = build_mask(allowed_set)
        IGN_hi, IGN_lo   = build_mask(ignore_set)
        DP_hi,  DP_lo    = build_mask(disallow_pure_set)
        SE_hi,  SE_lo    = build_mask(se_elems_set)
        ALW_hi, ALW_lo   = build_mask(se_allow_with_set)
        SEU_hi, SEU_lo   = _mask_or(SE_hi, SE_lo, ALW_hi, ALW_lo)  # union for SE-only

        dprint(f"[elem] masks  → ALLOWED(hi={int(ALL_hi)}, lo={int(ALL_lo)}) "
               f"SEU(hi={int(SEU_hi)}, lo={int(SEU_lo)})")

        # ---- candidates universe (ids) ----
        if candidate_ids is None:
            cand_all = self._ids.tolist()
        else:
            cand_all = [str(x) for x in candidate_ids]
        dprint(f"[elem] candidate universe: {len(cand_all)} ids")

        # ---- mask-only post-filter (no CIF) ----
        def _post_filter_mask(ids_in: Iterable[str], base_mask: tuple[np.uint64, np.uint64]) -> list[str]:
            out: list[str] = []
            b_hi, b_lo = base_mask
            for pid in ids_in:
                idx = self._row_index.get(str(pid))
                if idx is None:
                    continue
                c_hi = np.uint64(self._m_hi[idx]); c_lo = np.uint64(self._m_lo[idx])
                if _mask_is_zero(c_hi, c_lo):
                    continue

                # remove ignored atoms for gate decisions
                c2_hi, c2_lo = _mask_and_not(c_hi, c_lo, IGN_hi, IGN_lo)

                # SE-only early accept: (cand ⊆ SE ∪ allow_with) and contains at least one SE elem
                has_se = not _mask_is_zero(*_mask_and(c2_hi, c2_lo, SE_hi, SE_lo))
                rest_hi, rest_lo = _mask_and_not(c2_hi, c2_lo, SEU_hi, SEU_lo)
                if se_allow_pure and has_se and _mask_is_zero(rest_hi, rest_lo):
                    out.append(pid)
                    continue

                # disallow pure (e.g., O-only / C-only)
                if not _mask_is_zero(DP_hi, DP_lo) and _popcount_u128(c2_hi, c2_lo) == 1:
                    if not _mask_is_zero(*_mask_and(c2_hi, c2_lo, DP_hi, DP_lo)):
                        continue

                # ban mixing SE with base chemistry (use full allowed-set mask for "base chemistry")
                if se_ban_cross and has_se:
                    has_base_allowed = not _mask_is_zero(*_mask_and(c2_hi, c2_lo, ALL_hi, ALL_lo))
                    if has_base_allowed:
                        continue

                # require at least one element from the variant's base set
                if require_base:
                    if _mask_is_zero(*_mask_and(c2_hi, c2_lo, b_hi, b_lo)):
                        continue

                out.append(pid)
            return out

        admitted: set[str] = set()

        # ---- (NEW) admit SE-only universe first (pure Al or Al + allow_with) ----
        if se_allow_pure and not _mask_is_zero(SE_hi, SE_lo):
            SE_only_mask_elems = se_elems_set | se_allow_with_set | ignore_set
            SEU2_hi, SEU2_lo = build_mask(SE_only_mask_elems)
            ids_se = self.filter_by_mask_pair_fast(SEU2_hi, SEU2_lo, candidate_ids=cand_all)
            kept_se = _post_filter_mask(ids_se, base_mask=(np.uint64(0), np.uint64(0)))
            admitted.update(kept_se)
            dprint(f"[elem][SE-only] mask_elems={sorted(SE_only_mask_elems)} → "
                   f"candidates={len(ids_se)} kept={len(kept_se)}")

        # ---- base mask (allowed ∪ ignore) ----
        base_mask_elems = allowed_set | ignore_set
        M_hi, M_lo = build_mask(base_mask_elems)
        base_ids = self.filter_by_mask_pair_fast(M_hi, M_lo, candidate_ids=cand_all)
        kept_base = _post_filter_mask(base_ids, base_mask=(ALL_hi, ALL_lo))  # base_for_variant = allowed_set
        admitted.update(kept_base)
        dprint(f"[elem][base] mask_elems={sorted(base_mask_elems)} → candidates={len(base_ids)} kept={len(kept_base)}")

        # ---- +1 wildcard via variant enumeration (substitution semantics) ----
        if max_offlist_elements >= 1 and allowed_set:
            total_variant_candidates = 0
            total_variant_kept = 0
            for t in sorted(allowed_set):
                pool = set(_substitution_pool_for_target(t, wildcard_relation))
                # prune pool
                pool -= allowed_set
                pool -= disallow_set
                pool -= ignore_set
                if not pool:
                    dprint(f"[elem][variant] target={t}: pool=0 (after prune)")
                    continue

                base_minus_t = allowed_set - {t}
                Bm_hi, Bm_lo = build_mask(base_minus_t)  # for require_base gate

                variant_union: set[str] = set()
                for s in pool:
                    A = base_minus_t | {s} | ignore_set
                    V_hi, V_lo = build_mask(A)
                    ids_v = self.filter_by_mask_pair_fast(V_hi, V_lo, candidate_ids=cand_all)
                    variant_union.update(ids_v)

                kept_v = _post_filter_mask(variant_union, base_mask=(Bm_hi, Bm_lo))
                admitted.update(kept_v)
                total_variant_candidates += len(variant_union)
                total_variant_kept += len(kept_v)
                dprint(f"[elem][variant] target={t} pool={len(pool)} → cand={len(variant_union)} kept={len(kept_v)}")

            dprint(f"[elem][variant] total_cand={total_variant_candidates} total_kept={total_variant_kept}")

        out_sorted = sorted(admitted)
        dprint(f"[elem] FINAL kept={len(out_sorted)} (unique)")
        if self.debug and out_sorted[:5]:
            head5 = ", ".join(out_sorted[:5])
            dprint(f"[elem] examples: {head5}")
        return out_sorted

    # ---------- Space-group pruning ----------
    def exclude_space_groups(self, candidate_ids: Iterable[str], exclude_sg: Iterable[int] = (1, 2)) -> List[str]:
        """Return candidate IDs with space_group NOT in exclude_sg."""
        ids = [str(x) for x in candidate_ids]
        view = self.catalog[self.catalog["id"].isin(ids)]
        if "space_group" not in view.columns:
            raise KeyError("catalog missing 'space_group' column")
        sg = pd.to_numeric(view["space_group"], errors="coerce").astype("Int64")
        keep = ~sg.isin(list(exclude_sg))
        return view.loc[keep.fillna(False), "id"].astype(str).tolist()

    def drop_low_symmetry_sg(self, candidate_ids: Iterable[str], exclude_sg: Iterable[int] = (1, 2)) -> List[str]:
        """Alias for exclude_space_groups; keeps pipeline stage name self-documenting."""
        return self.exclude_space_groups(candidate_ids, exclude_sg=exclude_sg)


# cache JSON once per process
_ORIG_DB_CACHE: Dict[str, Any] = {}


def _load_original_db(path: str) -> Dict[str, Any]:
    if path in _ORIG_DB_CACHE:
        return _ORIG_DB_CACHE[path]
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("original_json must be a dict mapping id -> record")
    _ORIG_DB_CACHE[path] = data
    return data


# ------------------------------
# Lattice jitter utilities
# ------------------------------

def jitter_lattice(
    lattice: Lattice,
    a_b_c_frac_bounds: Tuple[float, float] = (-0.03, 0.03),
    angle_delta_bounds_deg: Tuple[float, float] = (-1.5, 1.5),
    rng: Optional[np.random.Generator] = None,
) -> Lattice:
    """Return a new Lattice with fractional jitter on (a,b,c) and absolute angle deltas in degrees."""
    rng = rng or np.random.default_rng()
    da, db, dc = rng.uniform(*a_b_c_frac_bounds, size=3)
    d_alpha, d_beta, d_gamma = rng.uniform(*angle_delta_bounds_deg, size=3)

    a = lattice.a * (1.0 + da)
    b = lattice.b * (1.0 + db)
    c = lattice.c * (1.0 + dc)
    alpha = lattice.alpha + d_alpha
    beta = lattice.beta + d_beta
    gamma = lattice.gamma + d_gamma

    return Lattice.from_parameters(a, b, c, alpha, beta, gamma)


# ------------------------------
# Q-grid helper
# ------------------------------

def make_q_grid(qmin: float, qmax: float, n_points: int) -> np.ndarray:
    if not (qmax > qmin > 0):
        raise ValueError("Require 0 < qmin < qmax")
    if n_points < 16:
        raise ValueError("n_points too small; need at least 16")
    return np.linspace(qmin, qmax, n_points, dtype=np.float64)
