#!/usr/bin/env python3
"""
Helpers for GSAS-II instrument-parameter files.

The app uses these helpers to generate or normalize real `.instprm` files
inside a run's inputs directory so the rest of the pipeline can remain
file-based.
"""

from pathlib import Path
import shutil
import sys
from typing import Dict

DEFAULT_LAB_XRAY_PRESET_KEY = "xray_lab_cuka"
LEGACY_INSTRUMENT_EXTENSIONS = {".prm", ".inst", ".ins"}
SUPPORTED_INSTRUMENT_UPLOAD_EXTENSIONS = ["instprm", "prm", "inst", "ins"]

BUILTIN_INSTPRM_PRESETS: Dict[str, Dict[str, str]] = {
    DEFAULT_LAB_XRAY_PRESET_KEY: {
        "ui_label": "Built-in CuKa Lab PXRD (Approximate)",
        "gsas_label": "CuKa lab data",
        "filename": "generated_CuKa_lab.instprm",
        "radiation_source": "X-ray",
        "instrument_mode": "cw",
        "description": (
            "Uses GSAS-II's built-in Cu Kalpha lab profile for approximate "
            "Bragg-Brentano PXRD screening when no calibrated instrument file is available."
        ),
    }
}


def get_builtin_instprm_preset(key: str) -> Dict[str, str]:
    """Return a copy of the built-in preset metadata for UI or validation."""
    try:
        return dict(BUILTIN_INSTPRM_PRESETS[key])
    except KeyError as exc:
        raise ValueError(f"Unknown instrument preset key: {key}") from exc


def _import_gsas_inst_modules():
    try:
        from GSASII import GSASIIfiles as G2fil
        from GSASII import GSASIIscriptable as G2sc
        return G2fil, G2sc
    except Exception:
        repo_gsas_dir = Path(os.environ.get("RADAR_PD_GSASII_ROOT") or Path(__file__).resolve().parents[1] / "GSAS-II")
        if str(repo_gsas_dir) not in sys.path:
            sys.path.insert(0, str(repo_gsas_dir))
        try:
            from GSASII import GSASIIfiles as G2fil
            from GSASII import GSASIIscriptable as G2sc
            return G2fil, G2sc
        except Exception as nested_exc:
            raise RuntimeError("GSAS-II instrument helpers are unavailable") from nested_exc


def write_builtin_instprm_file(key: str, output_path) -> Path:
    """Write the selected GSAS-II built-in preset to a real `.instprm` file."""
    preset = get_builtin_instprm_preset(key)
    gsas_label = preset["gsas_label"]

    try:
        from GSASII import defaultIparms as dI
    except Exception:
        repo_gsas_dir = Path(os.environ.get("RADAR_PD_GSASII_ROOT") or Path(__file__).resolve().parents[1] / "GSAS-II")
        if str(repo_gsas_dir) not in sys.path:
            sys.path.insert(0, str(repo_gsas_dir))
        try:
            from GSASII import defaultIparms as dI
        except Exception as nested_exc:
            raise RuntimeError("GSAS-II default instrument presets are unavailable") from nested_exc

    try:
        preset_index = dI.defaultIparm_lbl.index(gsas_label)
    except ValueError as exc:
        raise RuntimeError(f"GSAS-II preset not found: {gsas_label}") from exc

    preset_lines = dI.defaultIparms[preset_index]
    if not isinstance(preset_lines, (list, tuple)) or not preset_lines:
        raise RuntimeError(f"GSAS-II preset {gsas_label!r} is empty")

    first_line = str(preset_lines[0])
    if not first_line.startswith("#GSAS-II"):
        raise RuntimeError(
            f"GSAS-II preset {gsas_label!r} is malformed: missing GSAS-II header"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(str(line) for line in preset_lines), encoding="utf-8")

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to create instrument file at {output_path}")

    return output_path


def normalize_instrument_profile_to_instprm(input_path, output_path=None, bank=None) -> Path:
    """
    Normalize a supported GSAS instrument profile into a GSAS-II `.instprm` file.

    `.instprm` inputs are copied through unchanged unless `output_path` points to
    the same file. Legacy GSAS/EXPGUI `.prm` / `.inst` / `.ins` files are read
    through GSAS-II and written back out as new-style `.instprm`.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Instrument profile not found: {input_path}")

    suffix = input_path.suffix.lower()
    output_path = Path(output_path) if output_path else input_path.with_suffix(".instprm")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".instprm":
        if input_path.resolve() != output_path.resolve():
            shutil.copy2(input_path, output_path)
        return output_path

    if suffix not in LEGACY_INSTRUMENT_EXTENSIONS:
        raise ValueError(
            "Unsupported instrument profile extension "
            f"{suffix or '<none>'}; expected .instprm, .prm, .inst, or .ins"
        )

    G2fil, G2sc = _import_gsas_inst_modules()

    class _ReaderStub:
        def __init__(self):
            self.Sample = {}
            self.powderentry = [None, None, 1]
            self.instdict = {}
            self.instbank = None
            self.instfile = None
            self.instmsg = ""

    reader = _ReaderStub()
    try:
        iparm1, iparm2 = G2sc.load_iprms(str(input_path), reader, bank=bank)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read legacy instrument profile {input_path.name}: {exc}"
        ) from exc

    with output_path.open("w", encoding="utf-8") as handle:
        G2fil.WriteInstprm(handle, iparm1, iparm2, reader.Sample)

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to create normalized instrument file at {output_path}")

    return output_path
