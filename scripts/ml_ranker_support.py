#!/usr/bin/env python3
"""
Helpers for locating and reading ML ranker artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class MLRankerAssets:
    script_path: Optional[Path]
    model_path: Optional[Path]
    source: str
    error: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return bool(
            self.script_path
            and self.model_path
            and self.script_path.exists()
            and self.model_path.exists()
        )


def discover_ml_ranker_assets(project_root) -> MLRankerAssets:
    """Find the `infer.py` script and `mlp_ranker.pt` checkpoint for the ranker."""
    project_root = Path(project_root).resolve()

    env_script = os.environ.get("RADAR_ML_RANKER_SCRIPT")
    env_model = os.environ.get("RADAR_ML_RANKER_MODEL")
    if env_script or env_model:
        script_path = Path(env_script).expanduser().resolve() if env_script else None
        model_path = Path(env_model).expanduser().resolve() if env_model else None
        if script_path and model_path and script_path.exists() and model_path.exists():
            return MLRankerAssets(script_path=script_path, model_path=model_path, source="env")
        return MLRankerAssets(
            script_path=script_path,
            model_path=model_path,
            source="env",
            error="RADAR_ML_RANKER_SCRIPT/RADAR_ML_RANKER_MODEL were set, but one or both paths do not exist.",
        )

    env_dir = os.environ.get("RADAR_ML_RANKER_DIR")
    candidate_dirs: List[Path] = []
    if env_dir:
        candidate_dirs.append(Path(env_dir).expanduser().resolve())

    ranker_root = project_root / "ML_ranker"
    candidate_dirs.append(ranker_root / "mlp_ranker_for_phase_detection-main")
    if ranker_root.exists():
        for script_path in ranker_root.rglob("infer.py"):
            candidate_dirs.append(script_path.parent)
        for model_path in ranker_root.rglob("mlp_ranker.pt"):
            candidate_dirs.append(model_path.parent)

    seen = set()
    checked: List[str] = []
    for candidate_dir in candidate_dirs:
        if not candidate_dir:
            continue
        candidate_dir = candidate_dir.resolve()
        key = str(candidate_dir)
        if key in seen:
            continue
        seen.add(key)

        script_path = candidate_dir / "infer.py"
        model_path = candidate_dir / "mlp_ranker.pt"
        if script_path.exists() and model_path.exists():
            source = "project_default" if candidate_dir == ranker_root / "mlp_ranker_for_phase_detection-main" else "project_search"
            return MLRankerAssets(script_path=script_path, model_path=model_path, source=source)

        missing = []
        if not script_path.exists():
            missing.append("infer.py")
        if not model_path.exists():
            missing.append("mlp_ranker.pt")
        checked.append(f"{candidate_dir} (missing {', '.join(missing)})")

    error = "No usable ML ranker assets found."
    if checked:
        error += " Checked: " + "; ".join(checked[:8])
    return MLRankerAssets(
        script_path=None,
        model_path=None,
        source="not_found",
        error=error,
    )


def load_first_json_record(path) -> Dict[str, Any]:
    """Read the first non-empty JSON object from a `.json` or `.jsonl` file."""
    path = Path(path)
    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ValueError(f"{path} is empty")

    if path.suffix.lower() == ".json":
        return json.loads(raw_text)

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        return json.loads(line)

    raise ValueError(f"{path} does not contain a JSON record")


def write_ranker_status(path, **payload) -> Path:
    """Write a small JSON status artifact for the ML ranker stage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
