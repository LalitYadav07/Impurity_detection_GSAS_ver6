#!/usr/bin/env python3
"""
GSAS-II Core Infrastructure and Coordinate Management

This module provides the foundational classes for interacting with GSAS-II projects
and checking coordinate systems. It includes:
- GSASProjectManager: A context manager for creating and saving .gpx projects.
- CoordinateHandler: Utilities for converting between 2-Theta, Time-of-Flight, and Q-space.
- IntensityNormalizer: Helper for normalizing diffraction patterns.
"""

import os
import sys
import copy
import tempfile
import traceback
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _bootstrap_gsasii_import() -> None:
    """Make the bundled GSAS-II checkout importable outside the CLI driver."""
    repo_root = Path(__file__).resolve().parents[1]
    gsas_dir = str(repo_root / "GSAS-II")
    if gsas_dir not in sys.path:
        sys.path.insert(0, gsas_dir)


_bootstrap_gsasii_import()

_POWDER_HINT_ALIASES = {
    "xye": ("xye", "topas"),
    "qye": ("qye", "topas"),
    "csv": ("comma/tab/semicolon separated", "worksheet", "csv"),
    "gsas": ("gsas powder data", "gsas"),
    "fxye": ("gsas powder data", "gsas", "fxye"),
    "fullprof": ("fullprof .dat", "fullprof"),
    "rigaku": ("rigaku",),
}

try:
    from GSASII import GSASIIscriptable as G2sc
    from GSASII.GSASIIobj import G2Exception
    GSAS_AVAILABLE = True
except ImportError:
    logger.warning("GSAS-II not available. Some functionality will be limited.")
    G2sc = None
    GSAS_AVAILABLE = False
    G2Exception = Exception


class GSASProjectManager:
    """
    Manages GSAS-II project lifecycle, including creation, histogram addition,
    and phase management for impurity detection pipeline.
    """
    
    def __init__(self, work_dir: str, project_name: str = "impurity_detection"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.project_name = project_name
        self.project_path = self.work_dir / f"{project_name}.gpx"
        self.project: Optional[Any] = None
        self.main_histogram: Optional[Any] = None
        self.main_phase: Optional[Any] = None
        self.instrument_type: Optional[str] = None
        
    def create_project(self, overwrite: bool = True, template_gpx: Optional[str] = None) -> bool:
        """Create a new GSAS-II project, optionally from a template."""
        if not GSAS_AVAILABLE:
            raise RuntimeError("GSAS-II not available for project creation")
            
        try:
            if overwrite and self.project_path.exists():
                import time
                for i in range(5):
                    try:
                        self.project_path.unlink()
                        break
                    except PermissionError:
                        if i == 4: raise
                        time.sleep(0.2)
            
            if template_gpx and os.path.exists(template_gpx):
                import shutil
                shutil.copy2(template_gpx, str(self.project_path))
                self.project = G2sc.G2Project(gpxfile=str(self.project_path))
                # GSAS-II sometimes needs to know it's a new path
                self.project.save(str(self.project_path))
            else:
                self.project = G2sc.G2Project(newgpx=str(self.project_path))
                
            logger.info(f"Created GSAS-II project: {self.project_path} (template={template_gpx})")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create GSAS-II project: {e}")
            traceback.print_exc()
            return False
    
    def add_histogram(self, data_file: str, instprm_file: str,
                     fmthint: Optional[str] = None, instrument_type: Optional[str] = None) -> bool:
        """Add powder histogram to the project.

        Uses a stricter, GUI-like reader selection path instead of trusting
        the first scriptable GSAS-II reader that does not throw. This avoids
        accepting malformed imports that only fail later when no reflections
        fall inside the histogram limits.
        """
        if not self.project:
            raise RuntimeError("Project not initialized. Call create_project() first.")

        try:
            self.main_histogram = self._add_histogram_strict(
                data_file=data_file,
                instprm_file=instprm_file,
                fmthint=fmthint,
                instrument_type=instrument_type,
            )
        except Exception as exc:
            logger.warning(f"Failed to add histogram for {data_file}: {exc}")
            traceback.print_exc()
            return False

        # Determine instrument type
        try:
            actual_type = self._get_instrument_type()
            expected_type = (instrument_type or "").upper()
            if expected_type in ["TOF", "CW"] and actual_type not in ("Unknown", expected_type):
                logger.warning(
                    "Histogram import type mismatch: expected %s but loaded %s",
                    expected_type,
                    actual_type,
                )
            self.instrument_type = actual_type if actual_type != "Unknown" else (expected_type or "CW")
            logger.info(f"  → instrument type: {self.instrument_type}")
        except Exception as e:
            logger.warning(f"Could not determine instrument type: {e}")
            self.instrument_type = "CW"  # safe default

        return True

    def _add_histogram_strict(self, data_file: str, instprm_file: str,
                              fmthint: Optional[str], instrument_type: Optional[str]):
        G2sc.LoadG2fil()
        sniff = self._sniff_powder_file(data_file)
        proxy_data_file = self._maybe_prepare_numeric_proxy(data_file, sniff)
        read_data_file = proxy_data_file or data_file
        hint_order = self._build_powder_hint_order(data_file, fmthint, sniff)
        existing_names = [h.name for h in self.project.histograms()]
        errors: List[str] = []

        if proxy_data_file:
            logger.info(
                "Importing %s with powder hint order %s via normalized proxy %s",
                data_file,
                hint_order,
                proxy_data_file,
            )
        else:
            logger.info("Importing %s with powder hint order %s", data_file, hint_order)

        for hint in hint_order:
            for template_reader, ext_flag in self._iter_candidate_readers(read_data_file, hint):
                reader = copy.deepcopy(template_reader)
                fmt = getattr(reader, "formatName", reader.__class__.__name__)
                hint_label = "auto-detect" if hint is None else hint

                reader_ok, reader_err = self._read_with_reader(
                    reader,
                    read_data_file,
                    original_data_file=data_file,
                )
                if not reader_ok:
                    errors.append(f"{fmt} [{hint_label}] read failed: {reader_err}")
                    logger.info("Rejected reader %s for %s: %s", fmt, data_file, reader_err)
                    continue

                try:
                    histname, new_names, pwdrdata = G2sc.load_pwd_from_reader(
                        reader, instprm_file, existingnames=existing_names
                    )
                except Exception as exc:
                    errors.append(f"{fmt} [{hint_label}] load failed: {exc}")
                    logger.info("Reader %s could not build histogram for %s: %s", fmt, data_file, exc)
                    continue

                valid, validation_msg = self._validate_loaded_histogram(
                    pwdrdata,
                    expected_instrument_type=instrument_type,
                    sniff=sniff,
                )
                if not valid:
                    errors.append(f"{fmt} [{hint_label}] invalid histogram: {validation_msg}")
                    logger.info("Rejected histogram from reader %s for %s: %s", fmt, data_file, validation_msg)
                    continue

                histogram = self._attach_loaded_histogram(histname, new_names, pwdrdata)
                logger.info(
                    "Added histogram: %s via reader %s (hint=%s, extmatch=%s)",
                    data_file,
                    fmt,
                    hint_label,
                    ext_flag,
                )
                return histogram

        detail = "\n".join(errors[-10:]) if errors else "No compatible GSAS-II powder readers were available."
        raise RuntimeError(f"Unable to import powder data {data_file}.\n{detail}")

    def _attach_loaded_histogram(self, histname: str, new_names: List[str], pwdrdata: Dict[str, Any]):
        if histname in self.project.data:
            logger.warning("Warning - redefining histogram %s", histname)
        elif self.project.names[-1][0] == 'Phases':
            self.project.names.insert(-1, new_names)
        else:
            self.project.names.append(new_names)
        self.project.data[histname] = pwdrdata
        self.project.update_ids()
        return self.project.histogram(histname)

    def _read_with_reader(self, reader, data_file: str,
                          original_data_file: Optional[str] = None) -> Tuple[bool, str]:
        fmt = getattr(reader, "formatName", reader.__class__.__name__)
        display_file = original_data_file or data_file
        try:
            reader.selections = []
            reader.dnames = []
            reader.ReInitialize()
            reader.errors = ""
            contents_ok = reader.ContentsValidator(data_file)
        except Exception as exc:
            return False, f"validator exception: {exc}"
        if not contents_ok:
            msg = reader.errors or "ContentsValidator rejected file"
            return False, msg

        reader.objname = os.path.basename(display_file)
        try:
            flag = reader.Reader(data_file, buffer={}, blocknum=1)
        except Exception as exc:
            return False, f"reader exception: {exc}"
        if not flag:
            return False, reader.errors or f"{fmt} reader returned False"
        try:
            reader.idstring = os.path.basename(display_file)
        except Exception:
            pass
        try:
            if getattr(reader, "powderentry", None):
                reader.powderentry[0] = display_file
        except Exception:
            pass
        return True, ""

    def _iter_candidate_readers(self, data_file: str, hint: Optional[str]):
        primary = []
        secondary = []
        for reader in G2sc.Readers.get('Pwdr', []):
            if hint is not None and not self._reader_matches_hint(reader, hint):
                continue
            try:
                ext_flag = reader.ExtensionValidator(data_file)
            except Exception:
                continue
            if ext_flag is True:
                primary.append((reader, ext_flag))
            elif ext_flag is None:
                secondary.append((reader, ext_flag))
        return primary + secondary

    def _reader_matches_hint(self, reader, hint: str) -> bool:
        fmt = getattr(reader, "formatName", "").lower()
        aliases = _POWDER_HINT_ALIASES.get((hint or "").lower(), ((hint or "").lower(),))
        return any(alias in fmt for alias in aliases)

    def _build_powder_hint_order(self, data_file: str, fmthint: Optional[str], sniff: Dict[str, Any]) -> List[Optional[str]]:
        ext = Path(data_file).suffix.lower()
        order: List[Optional[str]] = []

        if fmthint and fmthint.lower() != "auto":
            order.append(fmthint.lower())

        kind = sniff.get("kind")
        if kind == "gsas" or ext in {".gsa", ".gss", ".gsas", ".fxye", ".raw", ".gda", ".xra"}:
            order += ["gsas", None]
        elif kind == "fullprof":
            order += ["fullprof", None, "xye", "csv"]
        elif kind == "csv":
            order += ["csv", "xye", None]
        elif kind == "qye":
            order += ["qye", "xye", "csv", None]
        elif kind == "xye":
            order += ["xye", "csv", None]
        else:
            order += [None, "xye", "qye", "gsas", "csv", "fullprof"]

        if ext in {".xye", ".chi"}:
            order = ["xye"] + order
        elif ext in {".qye", ".qchi"}:
            order = ["qye"] + order
        elif ext in {".csv", ".xy"}:
            order = ["csv"] + order
        elif ext == ".dat" and kind == "xye":
            order = ["xye", "csv"] + order

        deduped: List[Optional[str]] = []
        seen = set()
        for item in order:
            key = "__auto__" if item is None else item
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _sniff_powder_file(self, data_file: str) -> Dict[str, Any]:
        info: Dict[str, Any] = {"kind": None, "axis": None, "wrapped_xydata": False}
        ext = Path(data_file).suffix.lower()
        if ext in {".gsa", ".gss", ".gsas", ".fxye", ".raw", ".gda", ".xra"}:
            info["kind"] = "gsas"
            return info
        if ext in {".qye", ".qchi"}:
            info["kind"] = "qye"
            info["axis"] = "q"
            return info
        if ext in {".xye", ".chi"}:
            info["kind"] = "xye"
            return info

        try:
            with open(data_file, "r", encoding="utf-8", errors="replace") as fp:
                raw_lines = [line.rstrip("\n") for _, line in zip(range(80), fp)]
        except OSError:
            return info

        lower_lines = [line.strip().lower() for line in raw_lines if line.strip()]
        if any(line.startswith("bank ") and "fxye" in line for line in lower_lines):
            info["kind"] = "gsas"
            return info
        if any("time-of-flight" in line or line.startswith("'tof") for line in lower_lines):
            info["axis"] = "tof"
        elif any("2-theta" in line or "2theta" in line for line in lower_lines):
            info["axis"] = "2theta"
        elif any(" q " in f" {line} " for line in lower_lines[:5]):
            info["axis"] = "q"

        if any("xydata" in line for line in lower_lines[:5]):
            info["kind"] = "xye"
            info["wrapped_xydata"] = True
            return info

        if any(line.startswith("inter ") for line in lower_lines[:10]):
            info["wrapped_xydata"] = True
            return info

        numeric_counts: List[int] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(("'", "#", "!", "/*", "TITLE")):
                continue
            tokens = stripped.replace(",", " ").replace(";", " ").split()
            try:
                floats = [float(tok) for tok in tokens]
            except ValueError:
                continue
            numeric_counts.append(len(floats))
            if len(numeric_counts) >= 12:
                break

        if numeric_counts:
            if all(count in (2, 3) for count in numeric_counts[:6]):
                info["kind"] = "qye" if info.get("axis") == "q" else "xye"
            elif numeric_counts[0] >= 3 and any(count not in (2, 3) for count in numeric_counts[1:6]):
                info["kind"] = "fullprof"

        if info["kind"] is None and any(("," in line or ";" in line) for line in raw_lines[:10]):
            info["kind"] = "csv"

        return info

    def _maybe_prepare_numeric_proxy(self, data_file: str, sniff: Dict[str, Any]) -> Optional[str]:
        """
        Normalize wrapped lab PXRD text exports into a simple numeric xye proxy.

        Some `.dat` exports contain harmless wrapper records such as `XYDATA`
        and `INTER ...` before an otherwise standard two/three-column table.
        GSAS-II's strict xye validator rejects those headers, so we generate a
        clean numeric proxy file and preserve the original filename for the
        loaded histogram.
        """
        path = Path(data_file)
        if path.suffix.lower() != ".dat":
            return None
        if sniff.get("kind") not in {"xye", "fullprof"} and not sniff.get("wrapped_xydata"):
            return None

        try:
            raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None

        lower_nonempty = [line.strip().lower() for line in raw_lines if line.strip()]
        has_xydata = any(line.startswith("xydata") for line in lower_nonempty[:5])
        has_inter = any(line.startswith("inter ") for line in lower_nonempty[:10])
        if not (has_xydata or has_inter or sniff.get("wrapped_xydata")):
            return None

        numeric_rows: List[List[float]] = []
        for line in raw_lines:
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if stripped.startswith(("'", "#", "!", "/*")):
                continue
            if stripped.startswith("TITLE"):
                continue
            if lowered.startswith("xydata") or lowered.startswith("inter "):
                continue

            tokens = stripped.replace(",", " ").replace(";", " ").split()
            try:
                floats = [float(tok) for tok in tokens]
            except ValueError:
                continue
            if len(floats) not in (2, 3):
                continue
            numeric_rows.append(floats)

        if len(numeric_rows) < 10:
            return None

        x = np.asarray([row[0] for row in numeric_rows], dtype=float)
        if not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
            return None

        proxy_dir = self.work_dir / ".normalized_inputs"
        proxy_dir.mkdir(parents=True, exist_ok=True)
        proxy_path = proxy_dir / f"{path.stem}_normalized.xye"
        lines = ["# Normalized by RADAR-PD from wrapped powder text export\n"]
        for row in numeric_rows:
            if len(row) == 2:
                lines.append(f"{row[0]:.10g} {row[1]:.10g}\n")
            else:
                lines.append(f"{row[0]:.10g} {row[1]:.10g} {row[2]:.10g}\n")
        proxy_path.write_text("".join(lines), encoding="utf-8")
        return str(proxy_path)

    def _validate_loaded_histogram(self, pwdrdata: Dict[str, Any],
                                   expected_instrument_type: Optional[str],
                                   sniff: Dict[str, Any]) -> Tuple[bool, str]:
        try:
            powder_block = pwdrdata["data"][1]
            x = np.asarray(powder_block[0], dtype=float).ravel()
            y = np.asarray(powder_block[1], dtype=float).ravel()
            w = np.asarray(powder_block[2], dtype=float).ravel()
            inst = pwdrdata["Instrument Parameters"][0]
        except Exception as exc:
            return False, f"missing histogram arrays: {exc}"

        npts = min(len(x), len(y), len(w))
        if npts < 10:
            return False, f"too few data points ({npts})"
        x = x[:npts]
        y = y[:npts]
        w = w[:npts]

        if not np.all(np.isfinite(x)):
            return False, "x-axis contains non-finite values"
        if float(np.max(x) - np.min(x)) <= 0.0:
            return False, "x-axis span is zero"
        dx = np.diff(x)
        if dx.size and np.any(~np.isfinite(dx)):
            return False, "x-axis step array contains non-finite values"
        if dx.size and np.any(dx <= 0.0):
            return False, "x-axis is not strictly increasing"

        finite_y = np.isfinite(y)
        if int(np.count_nonzero(finite_y)) < max(10, npts // 2):
            return False, "too few finite intensities"
        positive_y = np.isfinite(y) & (y > 0.0)
        if int(np.count_nonzero(positive_y)) < min(10, npts):
            return False, "all intensities are zero or negative"

        finite_w = np.isfinite(w) & (w >= 0.0)
        if int(np.count_nonzero(finite_w)) < min(10, npts):
            return False, "weights are missing or invalid"

        inst_type_token = str(inst.get("Type", [""])[0])
        loaded_type = "TOF" if "T" in inst_type_token else "CW"
        expected = (expected_instrument_type or "").upper()
        if expected in {"TOF", "CW"} and loaded_type != expected:
            return False, f"instrument type mismatch ({loaded_type} vs expected {expected})"

        sniff_axis = (sniff or {}).get("axis")
        if sniff_axis == "tof" and loaded_type != "TOF":
            return False, "file header looks TOF but instrument parameters are not TOF"

        try:
            qmin, qmax = CoordinateHandler(loaded_type, inst).get_coverage_limits(x)
        except Exception as exc:
            return False, f"failed to derive coordinate coverage: {exc}"
        if not np.isfinite(qmin) or not np.isfinite(qmax) or qmax <= qmin:
            return False, "invalid Q coverage"
        if qmax > 100.0:
            return False, f"unphysical Q coverage ({qmin:.3f}, {qmax:.3f})"

        return True, ""
    
    def add_phase_from_cif(self, cif_file: str, phasename: str = "MainPhase",
                          link_to_histogram: bool = True) -> bool:
        """Add phase from CIF file and optionally link to histogram."""
        if not self.project:
            raise RuntimeError("Project not initialized.")
            
        try:
            histograms = [self.main_histogram] if (link_to_histogram and self.main_histogram) else []
            self.main_phase = self.project.add_phase(
                cif_file, phasename=phasename, histograms=histograms
            )
            logger.info(f"Added phase '{phasename}' from {cif_file}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to add phase: {e}")
            traceback.print_exc()
            return False
    
    def add_phase_from_cif_text(self, cif_text: str, phasename: str = "Phase",
                               link_to_histogram: bool = True) -> bool:
        """Add phase from CIF text string."""
        # Create temporary CIF file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as tmp:
            tmp.write(cif_text)
            tmp.flush()
            temp_path = tmp.name
            
        try:
            result = self.add_phase_from_cif(temp_path, phasename, link_to_histogram)
            return result
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
    
    def save_project(self) -> bool:
        """Save the current project state."""
        if not self.project:
            return False
            
        try:
            self.project.save(str(self.project_path))
            return True
        except Exception as e:
            logger.warning(f"Failed to save project: {e}")
            return False
    
    def get_instrument_type(self) -> Optional[str]:
        """Get instrument type (CW or TOF)."""
        return self.instrument_type
    
    def _get_instrument_type(self) -> str:
        """Determine if instrument is TOF or constant wavelength."""
        if not self.main_histogram:
            return "Unknown"
            
        try:
            inst_params = self.main_histogram.getHistEntryValue(['Instrument Parameters'])[0]
            inst_type = str(inst_params.get('Type', [''])[0])
            return "TOF" if 'T' in inst_type else "CW"
        except Exception:
            return "Unknown"
    
    def cleanup_project(self):
        """Clean up project resources."""
        self.project = None
        self.main_histogram = None
        self.main_phase = None
        self.instrument_type = None


class CoordinateHandler:
    """
    Handles coordinate system conversions between native instrument coordinates
    (2θ for CW, TOF for time-of-flight) and Q-space for impurity detection.
    """
    
    def __init__(self, instrument_type: str, instrument_params: Dict[str, Any]):
        self.instrument_type = instrument_type.upper()
        self.instrument_params = instrument_params
        
        # Extract relevant instrument parameters
        if self.instrument_type == "TOF":
            self.difC = float(instrument_params.get('difC', [0, 0, False])[1])
            self.difA = float(instrument_params.get('difA', [0, 0, False])[1])
            self.difB = float(instrument_params.get('difB', [0, 0, False])[1])
            self.zero_tof = float(instrument_params.get('Zero', [0, 0, False])[1])
        else:  # CW
            self.wavelength = float(instrument_params.get('Lam', [1.54])[0])
            self.zero_2theta = float(instrument_params.get('Zero', [0, 0, False])[1]) if \
                isinstance(instrument_params.get('Zero', [0, 0]), (list, tuple)) else 0.0
    
    @classmethod
    def from_gsas_histogram(cls, histogram):
        """Create CoordinateHandler from GSAS histogram object."""
        if not histogram:
            raise ValueError("Histogram object is None")
            
        try:
            inst_params = histogram.getHistEntryValue(['Instrument Parameters'])[0]
            inst_type = str(inst_params.get('Type', [''])[0])
            instrument_type = "TOF" if 'T' in inst_type else "CW"
            return cls(instrument_type, inst_params)
        except Exception as e:
            raise RuntimeError(f"Failed to extract instrument parameters: {e}")
    
    def q_to_native(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q values to native instrument coordinates."""
        q_values = np.asarray(q_values, dtype=float)
        
        if self.instrument_type == "TOF":
            return self._q_to_tof(q_values)
        else:
            return self._q_to_2theta(q_values)
    
    def native_to_q(self, native_values: np.ndarray) -> np.ndarray:
        """Convert native coordinates to Q values."""
        native_values = np.asarray(native_values, dtype=float)
        
        if self.instrument_type == "TOF":
            return self._tof_to_q(native_values)
        else:
            return self._2theta_to_q(native_values)
    
    def d_to_native(self, d_values: np.ndarray) -> np.ndarray:
        """Convert d-spacing values to native coordinates."""
        d_values = np.asarray(d_values, dtype=float)
        q_values = 2.0 * np.pi / np.maximum(d_values, 1e-10)
        return self.q_to_native(q_values)
    
    def _q_to_2theta(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q to 2θ for constant wavelength."""
        # Q = 4π sin(θ) / λ  =>  sin(θ) = Qλ/(4π)  =>  θ = arcsin(Qλ/(4π))
        sin_theta = (q_values * self.wavelength) / (4.0 * np.pi)
        sin_theta = np.clip(sin_theta, 0.0, 0.999)  # Avoid domain errors
        theta_rad = np.arcsin(sin_theta)
        two_theta_deg = 2.0 * np.degrees(theta_rad) + self.zero_2theta
        return two_theta_deg
    
    def _2theta_to_q(self, two_theta_deg: np.ndarray) -> np.ndarray:
        """Convert 2θ to Q for constant wavelength."""
        two_theta_corrected = two_theta_deg - self.zero_2theta
        theta_rad = np.radians(two_theta_corrected / 2.0)
        q_values = (4.0 * np.pi / self.wavelength) * np.sin(theta_rad)
        return q_values
    
    def _q_to_tof(self, q_values: np.ndarray) -> np.ndarray:
        """Convert Q to TOF using GSAS-II TOF equation."""
        # First convert Q to d-spacing: d = 2π/Q
        d_values = 2.0 * np.pi / np.maximum(q_values, 1e-10)
        
        # TOF equation: TOF = difC*d + difA*d² + difB/d + Zero
        tof_values = (self.difC * d_values + 
                     self.difA * d_values**2 + 
                     self.difB / np.maximum(d_values, 1e-10) + 
                     self.zero_tof)
        return tof_values
    
    def _tof_to_q(self, tof_values: np.ndarray) -> np.ndarray:
        """Convert TOF to Q (requires solving quadratic equation)."""
        # TOF - Zero = difC*d + difA*d² + difB/d
        # This is a cubic in d, but we can solve iteratively or use approximation
        
        tof_corrected = tof_values - self.zero_tof
        
        # Initial guess using linear approximation (difC dominates for most d-values)
        d_guess = tof_corrected / max(self.difC, 1e-10)
        
        # Newton-Raphson iteration to refine d
        for _ in range(5):  # Usually converges quickly
            f_d = (self.difC * d_guess + 
                   self.difA * d_guess**2 + 
                   self.difB / np.maximum(d_guess, 1e-10) - 
                   tof_corrected)
            
            df_d = (self.difC + 
                    2.0 * self.difA * d_guess - 
                    self.difB / np.maximum(d_guess**2, 1e-20))
            
            d_guess = d_guess - f_d / np.maximum(np.abs(df_d), 1e-10)
            d_guess = np.maximum(d_guess, 1e-10)  # Keep positive
        
        # Convert d to Q
        q_values = 2.0 * np.pi / d_guess
        return q_values
    
    def get_coverage_limits(self, data_x: np.ndarray) -> Tuple[float, float]:
        """Get coverage limits in Q-space from native data range."""
        if len(data_x) == 0:
            return 0.0, 10.0
            
        x_min, x_max = float(np.min(data_x)), float(np.max(data_x))
        q_limits = self.native_to_q(np.array([x_min, x_max]))
        return float(np.min(q_limits)), float(np.max(q_limits))
    
    def clip_to_coverage(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                        x_limits: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """Clip data arrays to specified x-range."""
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)
        
        x_min, x_max = x_limits
        mask = (x_vals >= x_min) & (x_vals <= x_max)
        
        return x_vals[mask], y_vals[mask]


class IntensityNormalizer:
    """
    Manages intensity normalization to maintain consistency between
    experimental data, simulated patterns, and GSAS-II scale factors.
    """
    
    def __init__(self):
        self.scale_history: Dict[str, float] = {}
        self.normalization_method: str = "max_unity"
        self.reference_scale: float = 1.0
    
    def normalize_experimental(self, intensity: np.ndarray, 
                             method: str = 'max_unity') -> Tuple[np.ndarray, float]:
        """
        Normalize experimental intensity data.
        
        Args:
            intensity: Raw intensity array
            method: Normalization method ('max_unity', 'range_01', 'zscore')
            
        Returns:
            Tuple of (normalized_intensity, scale_factor_applied)
        """
        intensity = np.asarray(intensity, dtype=float)
        
        if method == 'max_unity':
            max_val = float(np.max(intensity))
            if max_val <= 0:
                return intensity, 1.0
            scale_factor = 1.0 / max_val
            normalized = intensity * scale_factor
            
        elif method == 'range_01':
            min_val = float(np.min(intensity))
            max_val = float(np.max(intensity))
            range_val = max_val - min_val
            if range_val <= 0:
                return intensity, 1.0
            scale_factor = 1.0 / range_val
            normalized = (intensity - min_val) * scale_factor
            
        elif method == 'zscore':
            mean_val = float(np.mean(intensity))
            std_val = float(np.std(intensity))
            if std_val <= 0:
                return intensity - mean_val, 1.0
            scale_factor = 1.0 / std_val
            normalized = (intensity - mean_val) * scale_factor
            
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.scale_history['experimental'] = scale_factor
        self.normalization_method = method
        return normalized, scale_factor
    
    def normalize_simulated(self, intensity: np.ndarray, 
                          ref_scale: float = 1.0) -> Tuple[np.ndarray, float]:
        """Normalize simulated pattern to match experimental scale."""
        intensity = np.asarray(intensity, dtype=float)
        
        if self.normalization_method == 'max_unity':
            max_val = float(np.max(intensity))
            if max_val <= 0:
                return intensity, 1.0
            scale_factor = ref_scale / max_val
            normalized = intensity * scale_factor
            
        else:
            # For other methods, use simple scaling
            scale_factor = ref_scale
            normalized = intensity * scale_factor
        
        return normalized, scale_factor
    
    def denormalize_for_gsas(self, normalized_intensity: np.ndarray, 
                           original_scale: float) -> np.ndarray:
        """Convert normalized intensity back to GSAS-compatible scale."""
        return normalized_intensity / original_scale
    
    def track_scale_factors(self) -> Dict[str, float]:
        """Return dictionary of all tracked scale factors."""
        return self.scale_history.copy()
    
    def get_experimental_scale(self) -> float:
        """Get the experimental data scale factor."""
        return self.scale_history.get('experimental', 1.0)


# Test and validation functions
def test_coordinate_conversion():
    """Test coordinate system conversions."""
    print("Testing coordinate conversions...")
    
    # Mock CW instrument parameters
    cw_params = {
        'Type': ['PXC'],
        'Lam': [1.54056],  # Cu Kα
        'Zero': [0.0, 0.0, False]
    }
    
    # Mock TOF instrument parameters  
    tof_params = {
        'Type': ['PXT'],
        'difC': [0.0, 15000.0, False],
        'difA': [0.0, 0.0, False],
        'difB': [0.0, 0.0, False],
        'Zero': [0.0, 0.0, False]
    }
    
    # Test CW conversions
    cw_handler = CoordinateHandler("CW", cw_params)
    test_q = np.array([2.0, 4.0, 6.0])
    two_theta = cw_handler.q_to_native(test_q)
    q_back = cw_handler.native_to_q(two_theta)
    print(f"CW: Q {test_q} -> 2θ {two_theta} -> Q {q_back}")
    
    # Test TOF conversions
    tof_handler = CoordinateHandler("TOF", tof_params)
    tof_vals = tof_handler.q_to_native(test_q)
    q_back_tof = tof_handler.native_to_q(tof_vals)
    print(f"TOF: Q {test_q} -> TOF {tof_vals} -> Q {q_back_tof}")


if __name__ == "__main__":
    # Run basic tests
    test_coordinate_conversion()
    print("Core infrastructure components ready for integration.")
