from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


CATALOG_FILENAME = "catalog_deduplicated.csv"
STABLE_FILENAME = "mp_experimental_stable.csv"
PROFILES_DIRNAME = "profiles64"
PROFILES_NPZ_FILENAME = "profiles64.npz"
PROFILES_INDEX_FILENAME = "index.csv"
ORIGINAL_JSON_FILENAME = "highsymm_metadata.json"
CIF_MAP_FILENAME = "cif_map.json"
MANIFEST_FILENAME = "manifest.json"
PHASES_DIRNAME = "phases"
CIFS_DIRNAME = "cifs"


@dataclass(frozen=True)
class DBPackLayout:
    root: Path
    catalog_csv: Path
    stable_csv: Path
    profiles_dir: Path
    profiles_npz: Path
    profiles_index_csv: Path
    original_json: Path
    cif_map_json: Path
    manifest_json: Path
    phases_dir: Path
    cifs_dir: Path


def get_db_pack_layout(db_root: str | Path) -> DBPackLayout:
    root = Path(db_root).expanduser().resolve()
    profiles_dir = root / PROFILES_DIRNAME
    return DBPackLayout(
        root=root,
        catalog_csv=root / CATALOG_FILENAME,
        stable_csv=root / STABLE_FILENAME,
        profiles_dir=profiles_dir,
        profiles_npz=profiles_dir / PROFILES_NPZ_FILENAME,
        profiles_index_csv=profiles_dir / PROFILES_INDEX_FILENAME,
        original_json=root / ORIGINAL_JSON_FILENAME,
        cif_map_json=root / CIF_MAP_FILENAME,
        manifest_json=root / MANIFEST_FILENAME,
        phases_dir=root / PHASES_DIRNAME,
        cifs_dir=root / CIFS_DIRNAME,
    )


def build_db_config(
    db_root: str | Path,
    *,
    catalog_csv: Optional[str | Path] = None,
    stable_csv: Optional[str | Path] = None,
    profiles_dir: Optional[str | Path] = None,
    original_json: Optional[str | Path] = None,
    cif_map_json: Optional[str | Path] = None,
    include_existing_optional_sidecars: bool = True,
) -> Dict[str, str]:
    layout = get_db_pack_layout(db_root)
    config: Dict[str, str] = {
        "catalog_csv": str(Path(catalog_csv) if catalog_csv is not None else layout.catalog_csv),
        "stable_csv": str(Path(stable_csv) if stable_csv is not None else layout.stable_csv),
        "profiles_dir": str(Path(profiles_dir) if profiles_dir is not None else layout.profiles_dir),
    }

    if original_json is not None:
        config["original_json"] = str(Path(original_json))
    elif include_existing_optional_sidecars and layout.original_json.exists():
        config["original_json"] = str(layout.original_json)
    else:
        config["original_json"] = str(layout.original_json)

    if cif_map_json is not None:
        config["cif_map_json"] = str(Path(cif_map_json))
    elif include_existing_optional_sidecars and layout.cif_map_json.exists():
        config["cif_map_json"] = str(layout.cif_map_json)

    return config


def validate_db_config(db_cfg: Dict[str, Any], *, require_cif_source: bool = False) -> None:
    missing = [
        key for key in ("catalog_csv", "stable_csv", "profiles_dir")
        if not db_cfg.get(key)
    ]
    if missing:
        raise ValueError(f"db config missing required keys: {', '.join(missing)}")

    if require_cif_source and not (db_cfg.get("original_json") or db_cfg.get("cif_map_json")):
        raise ValueError("db config must define original_json or cif_map_json")
