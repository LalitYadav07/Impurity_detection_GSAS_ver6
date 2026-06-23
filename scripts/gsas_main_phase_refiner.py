#!/usr/bin/env python3
"""
GSAS-II Main Phase Refinement Engine

This module encapsulates the GSAS-II refinement logic for the impurity detection pipeline.
It provides:
- Automated refinement of lattice parameters, phase fractions, and instrument profile terms.
- Extraction of residuals in both native (2θ/TOF) and Q-space.
- Calculation of Pearson correlations for candidate ranking.
- Generation of difference plots and refinement metrics.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import traceback
import re
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _bootstrap_gsasii_import() -> None:
    """Make the bundled GSAS-II checkout importable outside the CLI driver."""
    repo_root = Path(__file__).resolve().parents[1]
    gsas_dir = os.environ.get("RADAR_PD_GSASII_ROOT") or str(repo_root / "GSAS-II")
    if gsas_dir in sys.path:
        sys.path.remove(gsas_dir)
    sys.path.insert(0, gsas_dir)


_bootstrap_gsasii_import()

try:
    from GSASII import GSASIIscriptable as G2sc
    from GSASII.GSASIIobj import G2Exception
    GSAS_AVAILABLE = True
except ImportError:
    GSAS_AVAILABLE = False
    G2Exception = Exception

# Safe Limits Import
try:
    from gsas_safe_limits import apply_safe_limits
except ImportError:
    # Fallback if not found (during dev)
    def apply_safe_limits(proj): return False

try:
    from auto_background_points import coerce_auto_background_params, estimate_background
except ImportError:
    from .auto_background_points import coerce_auto_background_params, estimate_background


# === XYE writer (used for residual-as-Yobs jobs) ===

def write_xye_from_arrays(out_path: str, x, y, sigma=None, shift_positive: bool = True) -> str:
    """
    Write a Topas-style XYE (x, y, esd) from arrays.
    If shift_positive is True, shift y by a constant so min(y) >= 1.0.
    Pearson correlation is invariant to constant shifts and scaling.
    """
    import numpy as _np
    x = _np.asarray(x, float).ravel()
    y = _np.asarray(y, float).ravel()
    n = int(min(x.size, y.size))
    if n == 0:
        raise ValueError("empty x/y arrays for XYE write")

    yw = y[:n].copy()
    if shift_positive:
        m = _np.nanmin(yw)
        if _np.isfinite(m) and m < 0.0:
            yw = yw - m + 1.0

    if sigma is None:
        sigma = _np.ones(n, float)
    else:
        sigma = _np.asarray(sigma, float).ravel()[:n]
        if sigma.size < n:
            sigma = _np.pad(sigma, (0, n - sigma.size), mode='edge')

    logger.info(f"[XYE] writing {out_path} (n={n}, shift_positive={shift_positive})")
    with open(out_path, "w") as f:
        for i in range(n):
            f.write(f"{x[i]:.6f} {yw[i]:.6f} {sigma[i]:.6f}\n")
    return out_path


# --- CIF cell I/O helpers (text-only; read-only used in flow) ---

_CELL_KEYS = [
    "_cell_length_a", "_cell_length_b", "_cell_length_c",
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
]

HISTOGRAM_HOLD_VARS = ('Zero', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SH/L')

_num_re = re.compile(r"""^[ \t]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\([\d]+\))?""")
LIGHT_CALIBRATION_DEFAULT_TERMS = ("Zero", "U", "V", "W")

def _parse_cif_number(s: str) -> float:
    """Parse a CIF numeric token that may include uncertainty '(...)'."""
    m = _num_re.match(s.strip())
    return float(m.group(1)) if m else float("nan")


def _add_histogram_hold_constraint(proj, hist, varname: str) -> None:
    """Add one histogram hold-constraint, normalizing GSAS var objects to iterable input."""
    varobj = proj.make_var_obj(hist=hist, varname=varname)
    if isinstance(varobj, (list, tuple)):
        hold_vars = [item for item in varobj if item is not None]
    else:
        hold_vars = [varobj] if varobj is not None else []
    if not hold_vars:
        raise ValueError(f"No GSAS variable object created for '{varname}'")
    proj.add_HoldConstr(hold_vars)


def pick_refinable_instrument_terms(inst_params: Dict[str, Any], requested_terms) -> Tuple[str, ...]:
    """Return requested instrument terms that are actually refinable in this histogram."""
    usable: List[str] = []
    for term in requested_terms or ():
        value = inst_params.get(term)
        if isinstance(value, (list, tuple)) and len(value) > 2 and isinstance(value[2], (bool, np.bool_)):
            usable.append(term)
    return tuple(usable)


def histogram_supports_light_instrument_calibration(histogram) -> bool:
    """Only allow the light calibration path for lab PXRD histograms."""
    try:
        inst_params = histogram.getHistEntryValue(['Instrument Parameters'])[0]
        sample_params = histogram.getHistEntryValue(['Sample Parameters'])
    except Exception:
        return False

    inst_type = str(inst_params.get('Type', [''])[0]).upper()
    sample_type = str(sample_params.get('Type', ''))
    return inst_type.startswith("PXC") and sample_type == "Bragg-Brentano"


def normalize_background_config(
    background_config: Optional[Dict[str, Any]] = None,
    *,
    bg_type: Optional[str] = None,
    bg_terms: Optional[int] = None,
    bg_coeffs: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Return a single normalized background config dict."""
    cfg = dict(background_config or {})
    if "mode" not in cfg:
        cfg["mode"] = "function"
    if bg_type is not None:
        cfg["type"] = bg_type
    if bg_terms is not None:
        cfg["terms"] = int(bg_terms)
    if bg_coeffs is not None:
        cfg["coeffs"] = list(bg_coeffs)
    cfg.setdefault("auto_params", {})
    return cfg



# === Robust Pearson ===

def _init_gsas_process():
    """Ensure GSAS-II is initialized correctly in a sub-process (headless, Agg)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import wx
        if not wx.GetApp():
            if os.environ.get("DISPLAY"):
                app = wx.App(False)
            else:
                print("[WARN] DISPLAY not set; skipping wx.App init in worker")
        import GSASII.GSASIIctrlGUI as G2gui
        G2gui.haveGUI = False
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        print(f"[WARN] Worker GSAS init skipped GUI setup: {e}")

def _safe_pearson(a, b) -> float:
    """
    Numerically robust Pearson; returns 0.0 if either vector has <2 valid points or ~zero variance.
    """
    import numpy as _np
    a = _np.asarray(a, float).ravel()
    b = _np.asarray(b, float).ravel()
    m = _np.isfinite(a) & _np.isfinite(b)
    if _np.count_nonzero(m) < 2:
        return 0.0
    a = a[m]; b = b[m]
    sa = float(a.std()); sb = float(b.std())
    if sa == 0.0 or sb == 0.0:
        return 0.0
    ac = (a - a.mean()) / sa
    bc = (b - b.mean()) / sb
    return float((ac * bc).mean())


@dataclass
class RefinementResults:
    """Container for main phase refinement results."""
    success: bool
    rwp: float
    chi2: float
    scale: float
    background_params: Dict[str, Any]
    cell_params: Dict[str, float]
    convergence_cycles: int
    error_message: Optional[str] = None


@dataclass
class LightCalibrationResults:
    """Outcome for a light PXRD instrument calibration pass."""
    success: bool
    skipped: bool
    exported_instprm: Optional[str]
    rwp_before: Optional[float]
    rwp_after: Optional[float]
    refined_terms: Tuple[str, ...]
    error_message: Optional[str] = None


class GSASDataExtractor:
    """
    Extracts various data arrays from GSAS-II histograms using native GSAS methods.
    Uses the built-in conversion functions rather than manual coordinate transforms.
    """

    @staticmethod
    def get_all_arrays(histogram) -> Dict[str, np.ndarray]:
        """Extract all relevant data arrays from GSAS histogram."""
        if not histogram:
            return {}

        data: Dict[str, np.ndarray] = {}
        try:
            raw_arrays = {
                'x_native': histogram.getdata('x'),
                'Q': histogram.getdata('Q'),
                'd': histogram.getdata('d'),
                'yobs': histogram.getdata('yobs'),
                'ycalc': histogram.getdata('ycalc'),
            }

            # GSAS exclusions often mask x/Q/ycalc but not yobs. Use a shared mask so
            # all extracted arrays stay aligned after internal regions are excluded.
            shared_mask = None
            shared_size = None
            for arr in raw_arrays.values():
                arr_np = np.ma.asarray(arr)
                if arr_np.ndim != 1:
                    continue
                if shared_size is None:
                    shared_size = int(arr_np.size)
                elif arr_np.size != shared_size:
                    shared_size = None
                    break
                arr_mask = np.ma.getmaskarray(arr_np)
                if shared_mask is None:
                    shared_mask = np.array(arr_mask, dtype=bool, copy=True)
                else:
                    shared_mask |= arr_mask

            def _aligned_array(arr) -> np.ndarray:
                arr_np = np.ma.asarray(arr)
                values = np.asarray(np.ma.getdata(arr_np))
                if shared_mask is not None and values.ndim == 1 and values.size == shared_mask.size:
                    return values[~shared_mask]
                if hasattr(arr_np, 'compressed'):
                    return arr_np.compressed()
                return values

            for key, arr in raw_arrays.items():
                data[key] = _aligned_array(arr)

            # Background (may not exist via getdata)
            try:
                ybkg = histogram.getdata('background')
            except Exception:
                ybkg = None
            if ybkg is not None:
                data['ybkg'] = _aligned_array(ybkg)

            # Weights (may be 'ywt' in some builds)
            try:
                ywt = histogram.getdata('ywt')
            except Exception:
                try:
                    ywt = histogram.getdata('yweight')
                except Exception:
                    ywt = None
            if ywt is not None:
                data['ywt'] = _aligned_array(ywt)

            # Compute residuals if possible
            if data.get('yobs') is not None and data.get('ycalc') is not None:
                yo = data['yobs']; yc = data['ycalc']
                if yo.size > 0 and yc.size > 0 and yo.size == yc.size:
                    data['residual'] = yo - yc

        except Exception as e:
            logger.warning(f"Warning: Failed to extract some data arrays: {e}")

        return data

    @staticmethod
    def get_residual_both_spaces(histogram) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get residual in both native and Q coordinates.
        Returns: (x_native, residual_native, Q, residual_Q)
        """
        data = GSASDataExtractor.get_all_arrays(histogram)
        x_native = data.get('x_native', np.array([]))
        Q = data.get('Q', np.array([]))
        residual = data.get('residual', np.array([]))
        validate_residual_arrays(
            x_native,
            residual,
            Q,
            residual,
            context="GSAS histogram residual extraction",
        )
        # residual_Q is the same residual sampled at the Q-mapped points
        return x_native, residual, Q, residual


class GSASMainPhaseRefiner:
    """
    Main phase refinement engine using GSAS-II's native refinement capabilities.
    Implements staged refinement: Scale -> Background -> Cell (optional/guarded).
    """

    def __init__(self, project_manager):
        self.project_manager = project_manager
        self.project = project_manager.project
        self.histogram = project_manager.main_histogram
        self.phase = project_manager.main_phase
        self.instrument_type = project_manager.get_instrument_type()

        if not all([self.project, self.histogram, self.phase]):
            raise RuntimeError("Project manager must have project, histogram, and phase initialized")

    def setup_initial_state(self) -> bool:
        """Set up initial refinement state - disable all refinements."""
        try:
            # Set histogram limits to data range
            self._set_limits_to_data()

            # Disable all initial refinements
            self._clear_all_instrument_refinements()
            self._disable_background_refinement()
            self._disable_cell_refinement()
            self._disable_phase_scale_refinement()

            # Set phase to use histogram but don't refine scale initially
            self.phase.set_HAP_refinements({'Use': True, 'Scale': False}, histograms=[self.histogram])

            logger.info("Initial refinement state configured")
            return True

        except Exception as e:
            logger.warning(f"Failed to setup initial state: {e}")
            traceback.print_exc()
            return False

    def refine_stage_scale(self) -> RefinementResults:
        """Stage 1: Refine only sample scale."""
        logger.info("=== Stage 1: Scale Only ===")
        try:
            self._enable_scale_refinement()
            self._disable_background_refinement()
            self._disable_cell_refinement()

            self.project.refine()

            results = self._extract_refinement_results("Scale")
            logger.info(f"Scale refinement: Rwp = {results.rwp:.3f}%, Scale = {results.scale:.6g}")
            return results

        except Exception as e:
            logger.warning(f"Scale refinement failed: {e}")
            traceback.print_exc()
            return RefinementResults(
                success=False, rwp=999.0, chi2=999.0, scale=1.0,
                background_params={}, cell_params={}, convergence_cycles=0,
                error_message=str(e)
            )

    def refine_stage_background(
        self,
        background_config: Optional[Dict[str, Any]] = None,
        bg_type: Optional[str] = None,
        bg_terms: Optional[int] = None,
        bg_coeffs: Optional[List[float]] = None
    ) -> RefinementResults:
        """Stage 2: Refine scale + background."""
        logger.info("=== Stage 2: Scale + Background ===")
        try:
            self._enable_scale_refinement()
            self._configure_background(
                background_config=background_config,
                bg_type=bg_type,
                bg_terms=bg_terms,
                bg_coeffs=bg_coeffs,
            )
            self._enable_background_refinement()
            self._disable_cell_refinement()

            self.project.refine()

            results = self._extract_refinement_results("Scale+Background")
            logger.info(f"Background refinement: Rwp = {results.rwp:.3f}%")
            return results

        except Exception as e:
            logger.warning(f"Background refinement failed: {e}")
            traceback.print_exc()
            return RefinementResults(
                success=False, rwp=999.0, chi2=999.0, scale=1.0,
                background_params={}, cell_params={}, convergence_cycles=0,
                error_message=str(e)
            )

    @staticmethod
    def _get_free_cell_mask(sgdata: dict):
        """Return free-cell flags and a symmetry-enforcing projector."""
        laue = sgdata.get('SGLaue', '-1')
        axis = sgdata.get('SGUniq', 'b')

        if laue == '-1':
            free = [True] * 6
            def _enforce(c): return c
        elif laue == '2/m':
            if axis == 'a':
                free = [True, True, True, True, False, False]
                def _enforce(c): c[4] = 90.0; c[5] = 90.0; return c
            elif axis == 'b':
                free = [True, True, True, False, True, False]
                def _enforce(c): c[3] = 90.0; c[5] = 90.0; return c
            else:
                free = [True, True, True, False, False, True]
                def _enforce(c): c[3] = 90.0; c[4] = 90.0; return c
        elif laue == 'mmm':
            free = [True, True, True, False, False, False]
            def _enforce(c): c[3] = 90.0; c[4] = 90.0; c[5] = 90.0; return c
        elif laue in ('4/m', '4/mmm'):
            free = [True, False, True, False, False, False]
            def _enforce(c): c[1] = c[0]; c[3] = 90.0; c[4] = 90.0; c[5] = 90.0; return c
        elif laue in ('6/m', '6/mmm', '3m1', '31m', '3'):
            free = [True, False, True, False, False, False]
            def _enforce(c): c[1] = c[0]; c[3] = 90.0; c[4] = 90.0; c[5] = 120.0; return c
        elif laue in ('3R', '3mR'):
            free = [True, False, False, True, False, False]
            def _enforce(c): c[1] = c[0]; c[2] = c[0]; c[4] = c[3]; c[5] = c[3]; return c
        elif laue in ('m3', 'm3m'):
            free = [True, False, False, False, False, False]
            def _enforce(c): c[1] = c[0]; c[2] = c[0]; c[3] = 90.0; c[4] = 90.0; c[5] = 90.0; return c
        else:
            free = [True] * 6
            def _enforce(c): return c

        return free, _enforce

    def _read_cell_from_data(self):
        """Return (a, b, c, alpha, beta, gamma) from phase data."""
        try:
            cell_list = self.phase.data['General']['Cell']
            return tuple(float(cell_list[i]) for i in range(1, 7))
        except Exception as e:
            logger.warning(f"_read_cell_from_data failed: {e}")
            return None

    def _write_cell_to_data(self, cell_abcabg, perturb_pct: float = 0.0, seed: int = 0):
        """Write a cell tuple back into GSAS phase data, optionally perturbed."""
        try:
            from GSASII import GSASIIlattice as G2lat
            cell6 = list(cell_abcabg)
            if perturb_pct > 0.0:
                sgdata = self.phase.data['General']['SGData']
                free_mask, enforce_fn = self._get_free_cell_mask(sgdata)
                rng = np.random.default_rng(seed + 1)
                for i, is_free in enumerate(free_mask):
                    if is_free:
                        cell6[i] *= 1.0 + rng.uniform(-perturb_pct, perturb_pct) / 100.0
                cell6 = enforce_fn(cell6)
            a, b, c, alpha, beta, gamma = cell6
            A = G2lat.cell2A(cell6)
            vol = float(G2lat.calc_V(A))
            cell_list = self.phase.data['General']['Cell']
            cell_list[1] = a
            cell_list[2] = b
            cell_list[3] = c
            cell_list[4] = alpha
            cell_list[5] = beta
            cell_list[6] = gamma
            cell_list[7] = vol
            logger.info(
                f"Cell reset -> a={a:.5f} b={b:.5f} c={c:.5f} "
                f"alpha={alpha:.3f} beta={beta:.3f} gamma={gamma:.3f} V={vol:.3f} A^3"
            )
        except Exception as e:
            logger.warning(f"_write_cell_to_data failed: {e}")

    def _set_max_cyc(self, n: int):
        """Set GSAS-II max refinement cycles for the current project."""
        try:
            self.project.data['Controls']['data']['max cyc'] = int(n)
        except Exception as e:
            logger.debug(f"_set_max_cyc({n}) failed: {e}")

    def refine_stage_cell(self) -> RefinementResults:
        """Stage 3: Refine scale + background + cell with metric-error retry."""
        logger.info("=== Stage 3: Scale + Background + Cell ===")
        import contextlib
        import io

        metric_keys = ('invalid', 'metric', 'ouch', 'g2exception', 'cell metric', 'unable to evaluate')
        degenerate_rwp = 99.9
        schedule = [
            (None, 0.00, "standard"),
            (1, 0.00, "1-cycle / no-perturb"),
            (1, 0.05, "1-cycle / +/-0.05% free-params only"),
            (2, 0.10, "2-cycle / +/-0.10% free-params only"),
        ]

        def _is_metric_error(exc: Exception) -> bool:
            return any(k in str(exc).lower() for k in metric_keys)

        def _is_metric_stdout(captured: str) -> bool:
            lo = captured.lower()
            return any(k in lo for k in metric_keys)

        try:
            orig_max_cyc = int(self.project.data['Controls']['data']['max cyc'])
        except Exception:
            orig_max_cyc = None

        cell_before = self._read_cell_from_data()
        last_exc: Optional[Exception] = None

        for attempt, (max_cyc, perturb_pct, label) in enumerate(schedule):
            if attempt > 0:
                logger.warning(f"Cell refinement attempt {attempt + 1}/{len(schedule)}: {label}")
                if cell_before is not None:
                    self._write_cell_to_data(cell_before, perturb_pct=perturb_pct, seed=attempt)
                if max_cyc is not None:
                    self._set_max_cyc(max_cyc)

            try:
                self._enable_scale_refinement()
                self._enable_background_refinement()
                self._enable_cell_refinement()

                stdout_buf = io.StringIO()
                with contextlib.redirect_stdout(stdout_buf):
                    self.project.refine()
                captured = stdout_buf.getvalue()
                if captured:
                    import sys as _sys
                    print(captured, end="", file=_sys.__stdout__)

                results = self._extract_refinement_results("Full")
                silent_metric = _is_metric_stdout(captured) or results.rwp >= degenerate_rwp
                if silent_metric:
                    if attempt < len(schedule) - 1:
                        logger.warning(
                            f"Cell refinement attempt {attempt + 1}/{len(schedule)} detected silent metric failure "
                            f"(Rwp={results.rwp:.3f}%, metric_in_stdout={_is_metric_stdout(captured)}); retrying."
                        )
                        self._disable_cell_refinement()
                        continue
                    last_exc = RuntimeError(
                        f"Cell metric failure persisted through all {len(schedule)} attempts "
                        f"(Rwp={results.rwp:.3f}%)"
                    )
                    self._disable_cell_refinement()
                    break

                logger.info(
                    f"Cell refinement succeeded (attempt {attempt + 1}/{len(schedule)}): "
                    f"Rwp = {results.rwp:.3f}%"
                )
                if max_cyc is not None and orig_max_cyc is not None:
                    self._set_max_cyc(orig_max_cyc)
                return results

            except Exception as e:
                is_metric = _is_metric_error(e)
                last_exc = e
                logger.warning(
                    f"Cell refinement attempt {attempt + 1} failed "
                    f"({'metric error' if is_metric else 'other error'}): {e}"
                )
                self._disable_cell_refinement()
                if is_metric and attempt < len(schedule) - 1:
                    continue
                break

        if orig_max_cyc is not None:
            self._set_max_cyc(orig_max_cyc)

        logger.warning(
            f"Cell refinement failed after {len(schedule)} attempts, "
            f"falling back to Scale+Background only. Last error: {last_exc}"
        )
        self._disable_cell_refinement()
        try:
                self.project.refine()
                results = self._extract_refinement_results("Scale+Background")
                results.error_message = f"Cell refinement failed: {last_exc}"
                return results
        except Exception as e2:
            traceback.print_exc()
            return RefinementResults(
                success=False, rwp=999.0, chi2=999.0, scale=1.0,
                background_params={}, cell_params={}, convergence_cycles=0,
                error_message=f"Cell refinement failed and recovery failed: {e2}"
            )

    def run_staged_refinement(
        self,
        enable_cell: bool = True,
        background_config: Optional[Dict[str, Any]] = None,
        bg_type: Optional[str] = None,
        bg_terms: Optional[int] = None,
        bg_coeffs: Optional[List[float]] = None
    ) -> RefinementResults:
        """Run complete staged refinement workflow."""
        logger.info("=== Running Staged Main Phase Refinement ===")

        if not self.setup_initial_state():
            return RefinementResults(
                success=False, rwp=999.0, chi2=999.0, scale=1.0,
                background_params={}, cell_params={}, convergence_cycles=0,
                error_message="Failed to setup initial state"
            )

        results_scale = self.refine_stage_scale()
        if not results_scale.success:
            return results_scale

        results_bg = self.refine_stage_background(
            background_config=background_config,
            bg_type=bg_type,
            bg_terms=bg_terms,
            bg_coeffs=bg_coeffs,
        )
        if not results_bg.success:
            return results_bg

        if enable_cell:
            results_final = self.refine_stage_cell()
        else:
            results_final = results_bg

        return results_final

    def get_residual_native(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get residual in native instrument coordinates."""
        x_native, residual, _, _ = GSASDataExtractor.get_residual_both_spaces(self.histogram)
        return x_native, residual

    def get_residual_q(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get residual in Q-space coordinates."""
        _, _, Q, residual = GSASDataExtractor.get_residual_both_spaces(self.histogram)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Residual max min: %s %s", np.max(residual), np.min(residual))
            logger.debug("Q max min: %s %s", np.max(Q), np.min(Q))
        return Q, residual

    def get_all_data(self) -> Dict[str, np.ndarray]:
        """Get all data arrays from current state."""
        return GSASDataExtractor.get_all_arrays(self.histogram)

    # Helper methods for refinement control
    def _set_limits_to_data(self):
        """Preserve current refinement limits and exclusions, then apply safe limits."""
        excluded = []
        try:
            excluded = self.histogram.Excluded() or []
        except Exception:
            excluded = []

        lo = hi = None
        try:
            lo = float(self.histogram.Limits('lower'))
            hi = float(self.histogram.Limits('upper'))
        except Exception:
            x_data = np.asarray(self.histogram.getdata('x'))
            if x_data.size > 0:
                lo = float(np.min(x_data))
                hi = float(np.max(x_data))

        if lo is not None and hi is not None and hi > lo:
            self.histogram.set_refinements({
                'Limits': {'low': float(lo), 'high': float(hi)}
            })

        # Enforce safe limits (prevent negative variance at low d/TOF)
        apply_safe_limits(self.project)

        if excluded:
            try:
                cur_lo = float(self.histogram.Limits('lower'))
                cur_hi = float(self.histogram.Limits('upper'))
                set_excluded(self.histogram, normalize_excluded_regions(excluded, cur_lo, cur_hi))
            except Exception:
                set_excluded(self.histogram, excluded)

    def _clear_all_instrument_refinements(self):
        """Disable all instrument parameter refinements."""
        try:
            inst_params = self.histogram.getHistEntryValue(['Instrument Parameters'])[0]
            refinable_params = []
            for key, value in inst_params.items():
                if isinstance(value, (list, tuple)) and len(value) > 2 and isinstance(value[2], (bool, np.bool_)):
                    refinable_params.append(key)
            if refinable_params:
                self.histogram.clear_refinements({'Instrument Parameters': refinable_params})
        except Exception as e:
            logger.warning(f"Warning: Could not clear instrument refinements: {e}")

    def _enable_scale_refinement(self):
        """Enable sample scale refinement."""
        try:
            sample_params = self.histogram.getHistEntryValue(['Sample Parameters'])
            current_scale = float(sample_params['Scale'][0]) if 'Scale' in sample_params else 1.0
            sample_params['Scale'] = [current_scale, True]  # [value, refine_flag]
            self.histogram.setHistEntryValue(['Sample Parameters'], sample_params)
            self.histogram.set_refinements({'Sample Parameters': ['Scale']})
        except Exception as e:
            logger.warning(f"Warning: Could not enable scale refinement: {e}")

    def _disable_phase_scale_refinement(self):
        """Disable phase scale refinement (use histogram scale instead)."""
        try:
            self.phase.set_HAP_refinements({'Scale': False}, histograms=[self.histogram])
        except Exception as e:
            logger.warning(f"Warning: Could not disable phase scale: {e}")

    def _default_bg_by_instrument(self) -> Tuple[str, int]:
        if self.instrument_type == "TOF":
            try:
                x_data = np.asarray(self.histogram.getdata('x'))
                terms = max(2, min(8, len(x_data) // 100)) if x_data.size > 0 else 3
            except Exception:
                terms = 3
            return "log interpolate", terms
        return "chebyschev-1", 12

    def _background_observed_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        x_data = self.histogram.getdata('x')
        yobs_data = self.histogram.getdata('yobs')

        x_mask = np.ma.getmaskarray(x_data) if np.ma.isMaskedArray(x_data) else None
        y_mask = np.ma.getmaskarray(yobs_data) if np.ma.isMaskedArray(yobs_data) else None

        shared_mask = None
        if x_mask is not None:
            shared_mask = np.array(x_mask, dtype=bool)
        if y_mask is not None:
            shared_mask = np.array(y_mask, dtype=bool) if shared_mask is None else (shared_mask | np.array(y_mask, dtype=bool))

        x_vals = np.asarray(np.ma.getdata(x_data), dtype=float)
        y_vals = np.asarray(np.ma.getdata(yobs_data), dtype=float)

        if shared_mask is None:
            finite = np.isfinite(x_vals) & np.isfinite(y_vals)
            return x_vals[finite], y_vals[finite]

        keep = (~shared_mask) & np.isfinite(x_vals) & np.isfinite(y_vals)
        return x_vals[keep], y_vals[keep]

    def _clear_fixed_background_points(self) -> None:
        try:
            self.histogram.clear_refinements({'Background': {'FixedPoints': []}})
        except Exception:
            pass
        try:
            cur = self.histogram.getHistEntryValue(['Background'])
            if isinstance(cur, list) and len(cur) >= 2 and isinstance(cur[1], dict):
                cur[1].pop('FixedPoints', None)
                self.histogram.setHistEntryValue(['Background'], cur)
        except Exception:
            pass

    def _configure_background(
        self,
        background_config: Optional[Dict[str, Any]] = None,
        bg_type: Optional[str] = None,
        bg_terms: Optional[int] = None,
        bg_coeffs: Optional[List[float]] = None,
    ):
        """Configure either a direct GSAS-II background function or auto fixed points."""
        cfg = normalize_background_config(
            background_config,
            bg_type=bg_type,
            bg_terms=bg_terms,
            bg_coeffs=bg_coeffs,
        )

        default_type, default_terms = self._default_bg_by_instrument()
        mode = str(cfg.get("mode", "function") or "function").strip().lower()
        resolved_type = str(cfg.get("type") or default_type)
        resolved_terms = int(cfg.get("terms") if cfg.get("terms") is not None else default_terms)
        resolved_coeffs = cfg.get("coeffs")

        try:
            self.histogram.set_refinements({'Background': {
                'type': resolved_type,
                'no. coeffs': int(resolved_terms),
                'refine': False,
            }})
            logger.info(
                "Background configured: mode=%s type=%s terms=%s",
                mode, resolved_type, resolved_terms,
            )

            if resolved_coeffs is not None:
                coeffs = list(map(float, resolved_coeffs))
                coeffs = (coeffs + [0.0] * int(resolved_terms))[:int(resolved_terms)]
                cur = self.histogram.getHistEntryValue(['Background'])
                if (isinstance(cur, list) and len(cur) >= 2
                        and isinstance(cur[0], list) and isinstance(cur[1], dict)):
                    back = list(cur[0])
                    back[0] = resolved_type
                    back[1] = bool(back[1]) if len(back) > 1 else False
                    back[2] = int(resolved_terms) if len(back) > 2 else int(resolved_terms)
                    back = back[:3] + coeffs
                    self.histogram.setHistEntryValue(['Background'], [back, cur[1]])
                    logger.info("Background coefficients seeded (n=%d)", len(coeffs))

            if mode == "auto_fixed_points":
                x_vals, y_vals = self._background_observed_arrays()
                auto_params = coerce_auto_background_params(cfg.get("auto_params") or {})
                _background_curve, fixed_points, resolved_params = estimate_background(
                    x_vals,
                    y_vals,
                    params=auto_params,
                )
                self._clear_fixed_background_points()
                self.histogram.set_refinements({'Background': {
                    'FixedPoints': fixed_points.tolist(),
                    'fit fixed points': True,
                }})
                logger.info(
                    "Auto fixed-point background prepared (points=%d, snip_iterations=%s)",
                    len(fixed_points),
                    resolved_params.snip_iterations,
                )
            else:
                self._clear_fixed_background_points()

        except Exception as e:
            logger.warning(f"Warning: Could not configure background: {e}")

    def _enable_background_refinement(self):
        """Enable background refinement."""
        try:
            self.histogram.set_refinements({'Background': {'refine': True}})
        except Exception as e:
            logger.warning(f"Warning: Could not enable background refinement: {e}")

    def _disable_background_refinement(self):
        """Disable background refinement."""
        try:
            self.histogram.set_refinements({'Background': {'refine': False}})
        except Exception as e:
            logger.warning(f"Warning: Could not disable background refinement: {e}")

    def _enable_cell_refinement(self):
        """Enable cell parameter refinement."""
        try:
            self.phase.set_refinements({'Cell': True})
        except Exception as e:
            logger.warning(f"Warning: Could not enable cell refinement: {e}")

    def _disable_cell_refinement(self):
        """Disable cell parameter refinement."""
        try:
            self.phase.set_refinements({'Cell': False})
        except Exception as e:
            logger.warning(f"Warning: Could not disable cell refinement: {e}")

    def _enable_instrument_refinement_terms(self, terms) -> Tuple[str, ...]:
        """Enable a selected subset of instrument parameters for refinement."""
        inst_params = self.histogram.getHistEntryValue(['Instrument Parameters'])[0]
        usable_terms = pick_refinable_instrument_terms(inst_params, terms)
        self._clear_all_instrument_refinements()
        if usable_terms:
            self.histogram.set_refinements({'Instrument Parameters': list(usable_terms)})
        return usable_terms

    def load_instrument_profile(self, profile_path: str) -> None:
        """Reload histogram instrument parameters from a `.instprm` file."""
        if not profile_path:
            raise ValueError("profile_path is required to reload instrument parameters")
        self.histogram.LoadProfile(str(profile_path))
        self._clear_all_instrument_refinements()
        logger.info("Reloaded instrument profile from %s", profile_path)

    def run_light_instrument_calibration(
        self,
        *,
        background_config: Optional[Dict[str, Any]] = None,
        bg_type: Optional[str] = None,
        bg_terms: Optional[int] = None,
        bg_coeffs: Optional[List[float]] = None,
        zero_cycles: int = 1,
        profile_cycles: int = 2,
        profile_terms = None,
        export_path: Optional[str] = None,
    ) -> LightCalibrationResults:
        """Run a conservative lab-PXRD instrument-profile calibration and export `.instprm`."""
        logger.info("=== Light PXRD Instrument Calibration ===")

        if not histogram_supports_light_instrument_calibration(self.histogram):
            return LightCalibrationResults(
                success=False,
                skipped=True,
                exported_instprm=None,
                rwp_before=None,
                rwp_after=None,
                refined_terms=(),
                error_message="Histogram is not a Bragg-Brentano PXC powder pattern",
            )

        requested_terms = tuple(profile_terms or LIGHT_CALIBRATION_DEFAULT_TERMS)
        baseline_rwp: Optional[float] = None
        chosen_terms: Tuple[str, ...] = ()

        try:
            orig_max_cyc = int(self.project.data['Controls']['data']['max cyc'])
        except Exception:
            orig_max_cyc = None

        try:
            if not self.setup_initial_state():
                return LightCalibrationResults(
                    success=False,
                    skipped=False,
                    exported_instprm=None,
                    rwp_before=None,
                    rwp_after=None,
                    refined_terms=(),
                    error_message="Failed to setup initial state",
                )

            results_scale = self.refine_stage_scale()
            if not results_scale.success:
                return LightCalibrationResults(
                    success=False,
                    skipped=False,
                    exported_instprm=None,
                    rwp_before=None,
                    rwp_after=None,
                    refined_terms=(),
                    error_message=results_scale.error_message or "Scale refinement failed",
                )

            results_bg = self.refine_stage_background(
                background_config=background_config,
                bg_type=bg_type,
                bg_terms=bg_terms,
                bg_coeffs=bg_coeffs,
            )
            if not results_bg.success:
                return LightCalibrationResults(
                    success=False,
                    skipped=False,
                    exported_instprm=None,
                    rwp_before=None,
                    rwp_after=None,
                    refined_terms=(),
                    error_message=results_bg.error_message or "Background refinement failed",
                )
            baseline_rwp = float(results_bg.rwp)

            self._disable_cell_refinement()
            self._enable_scale_refinement()
            self._enable_background_refinement()

            zero_terms = self._enable_instrument_refinement_terms(("Zero",))
            if zero_terms:
                if orig_max_cyc is not None:
                    self._set_max_cyc(max(1, int(zero_cycles)))
                self.project.refine()

            chosen_terms = self._enable_instrument_refinement_terms(requested_terms)
            if not chosen_terms:
                return LightCalibrationResults(
                    success=False,
                    skipped=True,
                    exported_instprm=None,
                    rwp_before=baseline_rwp,
                    rwp_after=baseline_rwp,
                    refined_terms=(),
                    error_message="No requested instrument profile terms are refinable",
                )

            if orig_max_cyc is not None:
                self._set_max_cyc(max(1, int(profile_cycles)))
            self.project.refine()

            results_final = self._extract_refinement_results("LightCalibration")
            self._clear_all_instrument_refinements()

            exported_instprm = None
            if export_path:
                export_target = Path(export_path)
                export_target.parent.mkdir(parents=True, exist_ok=True)
                self.histogram.SaveProfile(str(export_target))
                exported_instprm = str(export_target)

            logger.info(
                "Light PXRD calibration complete: Rwp %.3f%% -> %.3f%%, terms=%s, export=%s",
                baseline_rwp,
                results_final.rwp,
                ",".join(chosen_terms),
                exported_instprm or "none",
            )
            return LightCalibrationResults(
                success=True,
                skipped=False,
                exported_instprm=exported_instprm,
                rwp_before=baseline_rwp,
                rwp_after=float(results_final.rwp),
                refined_terms=chosen_terms,
            )

        except Exception as e:
            logger.warning(f"Light PXRD calibration failed: {e}")
            traceback.print_exc()
            try:
                self._clear_all_instrument_refinements()
                self._disable_cell_refinement()
            except Exception:
                pass
            return LightCalibrationResults(
                success=False,
                skipped=False,
                exported_instprm=None,
                rwp_before=baseline_rwp,
                rwp_after=None,
                refined_terms=chosen_terms,
                error_message=str(e),
            )
        finally:
            if orig_max_cyc is not None:
                self._set_max_cyc(orig_max_cyc)

    def _extract_refinement_results(self, stage: str) -> RefinementResults:
        """Extract refinement results from current state."""
        try:
            wR = self.histogram.get_wR()
            if wR is None:
                raise ValueError("get_wR() returned None — refinement did not converge (GSAS-II internal error)")
            rwp = float(wR)

            sample_params = self.histogram.getHistEntryValue(['Sample Parameters'])
            scale = float(sample_params['Scale'][0]) if 'Scale' in sample_params else 1.0

            # Background params (best-effort)
            bg_params: Dict[str, Any] = {}
            try:
                bg_data = self.histogram.getHistEntryValue(['Background'])
                if isinstance(bg_data, list) and len(bg_data) > 0:
                    bg_type = bg_data[0][0] if isinstance(bg_data[0], (list, tuple)) else "unknown"
                    coeffs = list(bg_data[0][3:]) if isinstance(bg_data[0], (list, tuple)) and len(bg_data[0]) > 3 else []
                    fixed_points = []
                    if len(bg_data) > 1 and isinstance(bg_data[1], dict):
                        fixed_points = list(bg_data[1].get('FixedPoints', []) or [])
                    bg_params = {
                        'type': bg_type,
                        'coefficients': coeffs,
                        'mode': 'auto_fixed_points' if fixed_points else 'function',
                        'fixed_point_count': len(fixed_points),
                    }
            except Exception:
                pass

            # Cell params
            cell_params: Dict[str, float] = {}
            try:
                cell = self.phase.get_cell()
                if isinstance(cell, dict):
                    cell_params = {k: float(v) for k, v in cell.items() if isinstance(v, (int, float))}
                else:
                    if len(cell) >= 6:
                        cell_params = {
                            'a': float(cell[0]), 'b': float(cell[1]), 'c': float(cell[2]),
                            'alpha': float(cell[3]), 'beta': float(cell[4]), 'gamma': float(cell[5])
                        }
            except Exception as e:
                logger.warning(f"Warning: Could not extract cell parameters: {e}")

            # Rough chi2
            try:
                data = GSASDataExtractor.get_all_arrays(self.histogram)
                if 'residual' in data and 'ywt' in data:
                    residual = data['residual']
                    weights = data['ywt']
                    chi2 = float(np.sum((residual * np.sqrt(np.maximum(weights, 1e-10))) ** 2))
                else:
                    chi2 = rwp ** 2
            except Exception:
                chi2 = rwp ** 2

            return RefinementResults(
                success=True, rwp=rwp, chi2=chi2, scale=scale,
                background_params=bg_params, cell_params=cell_params,
                convergence_cycles=1
            )

        except Exception as e:
            logger.warning(f"Failed to extract refinement results: {e}")
            traceback.print_exc()
            return RefinementResults(
                success=False, rwp=999.0, chi2=999.0, scale=1.0,
                background_params={}, cell_params={}, convergence_cycles=0,
                error_message=str(e)
            )


class GSASPatternAnalyzer:
    """
    Analyzes diffraction patterns and extracts peak information using GSAS-II.
    """

    def __init__(self, histogram, phase):
        self.histogram = histogram
        self.phase = phase

    def get_reflection_positions_q(self) -> np.ndarray:
        """
        Return sorted unique Q=2*pi/d positions for the main-phase reflections.
        """
        try:
            refls = self.histogram.reflections()
            if not refls:
                return np.array([])

            phase_key = None
            if self.phase in refls:
                phase_key = self.phase
            else:
                want_name = getattr(self.phase, 'name', None) or getattr(self.phase, 'phaseName', None) or str(self.phase)
                for k in refls.keys():
                    kname = getattr(k, 'name', None) or getattr(k, 'phaseName', None) or str(k)
                    if kname == want_name:
                        phase_key = k
                        break
            if phase_key is None and len(refls) == 1:
                phase_key = next(iter(refls.keys()))
            if phase_key is None:
                return np.array([])

            refl_data = refls.get(phase_key, {})
            ref_list = np.asarray(refl_data.get('RefList', []))
            if ref_list.size == 0:
                return np.array([])

            is_super = bool(refl_data.get('Super', False))
            d_col = 5 if is_super else 4
            if ref_list.ndim != 2 or ref_list.shape[1] <= d_col:
                return np.array([])

            d_vals = ref_list[:, d_col].astype(float)
            d_vals = d_vals[np.isfinite(d_vals) & (d_vals > 0.0)]
            if d_vals.size == 0:
                return np.array([])

            q_vals = 2.0 * np.pi / d_vals
            q_vals = np.unique(np.round(q_vals, decimals=7))
            return np.sort(q_vals)

        except Exception as e:
            print(f"Warning [get_reflection_positions_q]: {e}")
            return np.array([])


# ================================
# Utilities: limits/exclusions, Pearson metrics, and one-cycle joint refinement
# ================================

# ---- Limits/Excluded helpers ----

def read_abs_limits_or_bounds(hist):
    """Return (abs_lo, abs_hi) if present in Limits[0]; else infer from X data."""
    lim = hist.data.get('Limits')
    if isinstance(lim, (list, tuple)) and len(lim) >= 1 and isinstance(lim[0], (list, tuple)) and len(lim[0]) >= 2:
        return float(lim[0][0]), float(lim[0][1])
    try:
        X, Y, W = hist.getdata()
        return float(np.min(X)), float(np.max(X))
    except Exception:
        return None, None


def normalize_limits(
    limits: Optional[Tuple[float, float] | List[float]],
    abs_lo: Optional[float] = None,
    abs_hi: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Normalize and clip a user-specified fit window."""
    if not limits or len(limits) != 2:
        return None
    lo, hi = float(limits[0]), float(limits[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Limits must be finite numbers")
    lo, hi = min(lo, hi), max(lo, hi)
    if abs_lo is not None:
        lo = max(lo, float(abs_lo))
    if abs_hi is not None:
        hi = min(hi, float(abs_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError(
            f"Invalid limits after clipping to available range [{abs_lo}, {abs_hi}]"
        )
    return float(lo), float(hi)


def normalize_excluded_regions(
    excluded_pairs: Optional[List[Tuple[float, float]]],
    lo: Optional[float] = None,
    hi: Optional[float] = None,
) -> List[List[float]]:
    """Sort, clip, and merge excluded regions in native histogram coordinates."""
    if not excluded_pairs:
        return []

    cleaned: List[List[float]] = []
    for pair in excluded_pairs:
        if pair is None or len(pair) != 2:
            continue
        a, b = float(pair[0]), float(pair[1])
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        a, b = min(a, b), max(a, b)
        if lo is not None:
            a = max(a, float(lo))
        if hi is not None:
            b = min(b, float(hi))
        if b <= a:
            continue
        cleaned.append([float(a), float(b)])

    cleaned.sort(key=lambda pair: (pair[0], pair[1]))
    merged: List[List[float]] = []
    for a, b in cleaned:
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    return merged


def ensure_usable_range(lo: float, hi: float, excluded_pairs: Optional[List[Tuple[float, float]]]) -> None:
    """Raise if exclusions leave no usable points in the active fit window."""
    merged = normalize_excluded_regions(excluded_pairs, lo, hi)
    cursor = float(lo)
    for a, b in merged:
        if a > cursor:
            return
        cursor = max(cursor, b)
    if cursor >= hi:
        raise ValueError("Excluded regions consume the entire active data range")


def validate_residual_arrays(
    x_native: np.ndarray,
    residual_native: np.ndarray,
    q_values: np.ndarray,
    residual_q: np.ndarray,
    *,
    context: str = "GSAS residual extraction",
) -> None:
    """Raise a clear error when GSAS exclusions/limits leave no residual points."""
    x_arr = np.asarray(x_native, float).ravel()
    r_native = np.asarray(residual_native, float).ravel()
    q_arr = np.asarray(q_values, float).ravel()
    r_q = np.asarray(residual_q, float).ravel()

    if x_arr.size == 0 or r_native.size == 0 or q_arr.size == 0 or r_q.size == 0:
        raise RuntimeError(
            f"{context}: no usable residual points remain after applying fit limits "
            "and ignored regions. Reduce or correct the ignored region/fit window so "
            "at least part of the measured pattern remains."
        )

    n_native = min(x_arr.size, r_native.size)
    n_q = min(q_arr.size, r_q.size)
    finite_native = np.isfinite(x_arr[:n_native]) & np.isfinite(r_native[:n_native])
    finite_q = np.isfinite(q_arr[:n_q]) & np.isfinite(r_q[:n_q])
    if not finite_native.any() or not finite_q.any():
        raise RuntimeError(
            f"{context}: residual arrays contain no finite usable points after "
            "applying fit limits and ignored regions."
        )


def set_limits(hist, lo, hi):
    """Set current refinement limits to [lo, hi], respecting old GSAS layouts if needed."""
    try:
        hist.Limits('lower', float(lo))
        hist.Limits('upper', float(hi))
    except Exception:
        lim = hist.data.setdefault('Limits', [[float(lo), float(hi)], [float(lo), float(hi)]])
        if isinstance(lim, list) and len(lim) >= 2 and isinstance(lim[1], list) and len(lim[1]) >= 2:
            lim[1][0] = float(lo)
            lim[1][1] = float(hi)
        else:
            hist.set_refinements({'Limits': [float(lo), float(hi)]})


def set_excluded(hist, excluded_pairs):
    cleaned = normalize_excluded_regions(excluded_pairs)
    try:
        hist.Excluded(cleaned)
    except Exception:
        lim = hist.data.setdefault('Limits', [[0.0, 0.0], [0.0, 0.0]])
        if isinstance(lim, list):
            while len(lim) < 2:
                lim.append([0.0, 0.0])
            lim[2:] = cleaned


# ---- .lst parser ----

def parse_gsas_lst(lst_path: Path, target_hist: str):
    """Return {phase: {'phase_fraction_pct':..., 'weight_fraction_pct':...}}."""
    num = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    phase_hdr_re = re.compile(r'^\s*Phase:\s*(?P<phase>.+?)\s+in\s+histogram:\s*(?P<hist>.+?)\s*$')
    phase_only_re = re.compile(rf'Phase fraction\s*:\s*(?P<pf>{num})\s*(?:Refine\?\s*(?P<ref>True|False))?')
    frac_line_re = re.compile(
        rf'Phase fraction\s*:\s*(?P<pf>{num})\s*,\s*sig\s*(?P<pf_sig>{num})\s*'
        rf'Weight fraction\s*:\s*(?P<wf>{num})\s*,\s*sig\s*(?P<wf_sig>{num})'
    )
    out: Dict[str, Dict[str, float]] = {}
    fallback_only: Dict[str, Dict[str, float]] = {}
    lines = Path(lst_path).read_text(errors="ignore").splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = phase_hdr_re.match(lines[i])
        if m:
            phase = m.group('phase').strip()
            histname = m.group('hist').strip()
            if histname == target_hist:
                fallback_pf = None
                for j in range(i + 1, min(i + 30, n)):
                    m2 = frac_line_re.search(lines[j])
                    if m2:
                        g = {k: float(v) for k, v in m2.groupdict().items() if k in {'pf','pf_sig','wf','wf_sig'}}
                        out[phase] = {
                            "phase_fraction_pct": g["pf"] * 100.0,
                            "weight_fraction_pct": g["wf"] * 100.0,
                        }
                        break
                    if fallback_pf is None:
                        m3 = phase_only_re.search(lines[j])
                        if m3:
                            fallback_pf = float(m3.group('pf'))
                if phase not in out and fallback_pf is not None:
                    fallback_only[phase] = {
                        "phase_fraction_pct": fallback_pf * 100.0,
                        "weight_fraction_pct": fallback_pf * 100.0,
                    }
        i += 1
    if not out and len(fallback_only) == 1:
        return fallback_only
    return out


# ---- Pearson helpers ----

def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = min(len(y_true), len(y_pred))
    if n == 0:
        return float("nan")
    y_true = y_true[:n] - float(np.mean(y_true[:n]))
    y_pred = y_pred[:n] - float(np.mean(y_pred[:n]))
    denom = float(np.sqrt((y_true**2).sum() * (y_pred**2).sum()))
    return float((y_true @ y_pred) / denom) if denom > 0 else float("nan")


def compute_gsas_ycalc_pearson(project_manager) -> float:
    """
    Compute Pearson(yobs, ycalc) for the current project/phase/hist with a 0-cycle calc.
    background=0 fixed; HAP.Use=True, HAP.Scale=1.0 (not refined).
    """
    refiner = GSASMainPhaseRefiner(project_manager)
    proj = project_manager.project
    hist = project_manager.main_histogram
    phase = project_manager.main_phase

    refiner.setup_initial_state()
    try:
        hist.set_refinements({'Background': {'type': 'chebyschev-1', 'refine': False, 'no. coeffs': 12, 'coeffs': [0.0]*12}})
    except Exception:
        pass
    try:
        phase.set_HAP_refinements({'Use': True, 'Scale': False}, histograms=[hist])
        phase.HAPvalue('Scale', 1.0, targethistlist=[hist])
    except Exception:
        pass

    try:
        proj.data['Controls']['data']['max cyc'] = 0
        proj.do_refinements([{'set': {}}])
    except Exception:
        pass

    data = GSASDataExtractor.get_all_arrays(hist)
    yobs = data.get('yobs', np.array([]))
    ycalc = data.get('ycalc', np.array([]))
    if yobs.size == 0 or ycalc.size == 0:
        return float('nan')
    return pearson_corr(yobs, ycalc)

def compute_gsas_pearson_for_cif(
    data_path: str,
    instprm_path: str,
    fmthint: Optional[str],
    cif_path: str,
    work_dir: str,
    limits: Optional[Tuple[float, float]],
    exclude_regions: Optional[List[Tuple[float, float]]],
    tmp_tag: str,
    *,
    refine_cycles: int = 0,                 # kept for API compatibility (unused for staged passes)
    refine_cell: bool = False,              # ignored; we always do staged Scale → Scale+Cell
    refine_hist_scale: bool = False,        # ignored; we always refine hist Scale
    out_refined_cif: Optional[str] = None,  # if None, we will write <stem>_refined.cif (never overwrite source)
    source_cif_for_export: Optional[str] = None,  # ignored for writing; kept for API compatibility
    x_override: Optional[np.ndarray] = None,
    y_override: Optional[np.ndarray] = None,
    fmthint_override: Optional[str] = None,
    shift_positive: bool = True,
    template_gpx: Optional[str] = None,
    background_config: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Build a tiny GSAS-II project, run staged refinement (Pass-1: Scale; Pass-2: Scale+Cell),
    return Pearson(Yobs, Ycalc), and write a refined CIF for the candidate phase using
    GSAS-II's exporter (export_CIF(..., quickmode=True)).

    Notes:
    - This patched version never overwrites the input CIF. The refined file is written to
      <stem>_refined.cif (or 'out_refined_cif' if provided).
    - After writing, the first 'data_' header is sanitized to a short, GSAS-II-friendly label.
    """
    _init_gsas_process()
    import re
    from pathlib import Path as _Path
    try:
        from gsas_core_infrastructure import GSASProjectManager
    except ImportError:
        from .gsas_core_infrastructure import GSASProjectManager

    def _sanitize_cif_data_block(cif_file: str, label_base: str, suffix: str = "refined", maxlen: int = 40) -> None:
        """
        Force a short, parser-friendly CIF 'data_' label on the first data block.
        Example: data_mp_30_refined
        """
        p = _Path(cif_file)
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return
        base = f"{label_base}_{suffix}".replace("-", "_")
        base = re.sub(r"[^A-Za-z0-9_]+", "_", base)[:maxlen]
        safe = f"data_{base}" if not base.startswith("data_") else base
        for i, l in enumerate(lines):
            if l.strip().startswith("data_"):
                if l.strip() != safe:
                    lines[i] = safe
                break
        else:
            lines.insert(0, safe)
        try:
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception:
            pass

    pm = GSASProjectManager(work_dir, f"{_Path(cif_path).stem}_{tmp_tag}")
    if not pm.create_project(template_gpx=template_gpx):
        raise RuntimeError("Failed to create GSAS project for Pearson")

    histogram_loaded_from_template = False
    using_override = (x_override is not None) and (y_override is not None)
    if template_gpx and _Path(template_gpx).exists():
        # Resolve existing histogram from template
        if pm.project.histograms():
            hist = pm.project.histograms()[0]
            pm.main_histogram = hist
            histogram_loaded_from_template = True
        else:
             logger.warning(f"Template GPX loaded from {template_gpx} but has no histograms. Falling back to fresh load.")
             # No raise; fall through to normal loading

    if not histogram_loaded_from_template:
        # Select observed dataset
        local_data_path = data_path
        local_fmthint = fmthint
        if using_override:
            _Path(work_dir).mkdir(parents=True, exist_ok=True)
            base = _Path(data_path).stem if data_path else "obs"
            tmp_xye_path = str(_Path(work_dir) / f"{base}_{tmp_tag}_RESID.xye")
            write_xye_from_arrays(tmp_xye_path, x_override, y_override, sigma=None, shift_positive=shift_positive)
            local_data_path = tmp_xye_path
            local_fmthint  = fmthint_override or "xye"

        if not pm.add_histogram(local_data_path, instprm_path, fmthint=local_fmthint):
            raise RuntimeError("Failed to add histogram for Pearson")
        hist = pm.main_histogram

    # Limits & excludes
    try:
        if using_override:
            xs = np.asarray(x_override, float).ravel()
            if xs.size >= 2 and np.isfinite(xs).any():
                lo = float(np.nanmin(xs)); hi = float(np.nanmax(xs))
                if hi > lo:
                    set_limits(hist, lo, hi)
        else:
            if limits and len(limits) == 2:
                set_limits(hist, float(limits[0]), float(limits[1]))
            if exclude_regions:
                set_excluded(hist, exclude_regions)
    except Exception:
        pass

    # Add phase
    ph_name = _Path(cif_path).stem  # short, file-stem-based phase name
    if not pm.add_phase_from_cif(cif_path, ph_name):
        raise RuntimeError("Failed to add phase in Pearson job")

    # Resolve phase object
    try:
        phase = None
        for p in pm.project.phases():
            pname = getattr(p, 'name', None) or getattr(p, 'phaseName', None) or str(p)
            if pname == ph_name:
                phase = p
                break
        if phase is None:
            phase = pm.project.phases()[0] if pm.project.phases() else getattr(pm, 'main_phase', None)
    except Exception:
        phase = getattr(pm, 'main_phase', None)
    if phase is None:
        raise RuntimeError(f"Could not locate phase '{ph_name}' after add_phase_from_cif")

    # Background model: keep residual-as-Yobs jobs background-free, but allow the
    # configured background path for raw observed-data Pearson re-ranking.
    try:
        if using_override:
            hist.set_refinements({'Background': {'type': 'chebyschev-1', 'no. coeffs': 1, 'coeffs': [0.0], 'refine': False}})
        elif background_config:
            GSASMainPhaseRefiner(pm)._configure_background(background_config=background_config)
        else:
            hist.set_refinements({'Background': {'type': 'chebyschev-1', 'no. coeffs': 1, 'coeffs': [0.0], 'refine': False}})
    except Exception:
        pass
    try:
        phase.set_HAP_refinements({'Use': True, 'Scale': False}, histograms=[hist])
        phase.HAPvalue('Scale', 1.0, targethistlist=[hist])
    except Exception:
        pass

    def _set_flags(hist_scale: bool, cell: bool):
        # Histogram sample Scale
        if hist_scale:
            try:
                hist.set_refinements({'Sample Parameters': ['Scale']})
            except Exception:
                pass
        else:
            try:
                hist.clear_refinements({'Sample Parameters': ['Scale']})
            except Exception:
                pass
        # Phase Cell
        try:
            phase.set_refinements({'Cell': bool(cell)})
        except Exception:
            pass

    def _run_and_r(cycles: int, label: str) -> float:
        try:
            pm.project.data['Controls']['data']['max cyc'] = int(max(0, cycles))
        except Exception:
            pass
        logger.info(f"[PEARSON] {ph_name} {label}: cycles={int(max(0, cycles))}")
        try:
            pm.project.do_refinements([{'set': {'Background': {'refine': False}}}, {'refine': True}])
        except Exception:
            pass
        try:
            Yo = np.asarray(hist.getdata('yobs'), float)
            Yc = np.asarray(hist.getdata('ycalc'), float)
        except Exception:
            _, Yo, _, Yc = hist.getdata()
            Yo = np.asarray(Yo, float); Yc = np.asarray(Yc, float)
        r = _safe_pearson(Yo, Yc)
        logger.info(f"[PEARSON] {ph_name} {label}: r={r:.6f}")
        return r

    # Pass-1: Scale only
    _set_flags(hist_scale=True, cell=False)
    r1 = _run_and_r(1, "pass1-scale")

    # Early exit for clearly poor candidates:
    # If r < 0.1 after scale refinement, it's unlikely to become a top candidate with cell refinement.
    if r1 < 0.1:
        logger.debug(f"[PEARSON] {ph_name} early-exit: r={r1:.4f} too low")
        # Final result is r1
        # Still need to write the refined CIF if requested, but use r1 for return
        r2 = r1
    else:
        # Pass-2: Scale + Cell
        _set_flags(hist_scale=True, cell=True)
        r2 = _run_and_r(1, "pass2-scale+cell")

    # ---- Export refined CIF via GSAS-II (always to a separate file) ----
    stem = _Path(cif_path).stem
    target_write = out_refined_cif or str(_Path(cif_path).with_name(stem + "_refined.cif"))

    try:
        # Export from the refined phase object
        phase.export_CIF(target_write, quickmode=True)

        # Basic sanity
        txt = open(target_write, "r", encoding="utf-8", errors="ignore").read()
        if ("_cell_length_a" not in txt) or ("_atom_site_" not in txt):
            raise RuntimeError("export_CIF wrote an incomplete file (missing cell and/or atom loop)")

        # Sanitize the first data_ header to something short & stable
        _sanitize_cif_data_block(target_write, label_base=ph_name, suffix="refined", maxlen=40)
        logger.info(f"[PEARSON] wrote refined CIF → {target_write}")

    except Exception as ex:
        # Be strict: if export fails, surface the error (keeps behavior obvious)
        raise RuntimeError(f"Failed to export refined CIF to {target_write}: {ex}") from ex

    return r2

# ---- One-cycle joint refinement ----

def clone_gpx(src_gpx: str, dst_gpx: str) -> None:
    src, dst = Path(src_gpx), Path(dst_gpx)
    if not src.exists():
        raise FileNotFoundError(f"Base GPX not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(str(src), str(dst))
    logger.info(f"[clone] {src} -> {dst}")


def get_hist_and_main_phase(proj, main_phase_name: str):
    hists = proj.histograms()
    if not hists:
        raise RuntimeError("No histograms in project.")
    hist = hists[0]
    phases = {p.name: p for p in proj.phases()}
    if main_phase_name not in phases:
        raise RuntimeError(f"Main phase '{main_phase_name}' not found. Have: {list(phases)}")
    logger.info(f"[init] histogram='{hist.name}', main='{main_phase_name}'")
    return hist, phases[main_phase_name]


def set_phase_cell_refine(phase, refine: bool) -> None:
    (phase.set_refinements if refine else phase.clear_refinements)({'Cell': True})
    logger.debug(f"[flags] Phase '{phase.name}' Cell refine={refine}")


def joint_refine_one_cycle(
    base_gpx: str,
    out_gpx: str,
    main_phase_name: str,
    pid_to_cif: Dict[str, str],
    hap_init: float = 0.05,
    max_joint_cycles: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Clone base GPX, add candidates, and run a restricted one-cycle refinement:
      - Background refine ON; HAP Scale refine ON (main + all candidates)
      - Sample Scale (:h:Scale) is held
      - Zero-shift & instrument profile held; all Cell params held
    Returns {phase_name: {"phase_fraction_pct": float, "weight_fraction_pct": float}} in percent.
    """
    from GSASII import GSASIIscriptable as G2sc  # local import to match top-level availability

    clone_gpx(base_gpx, out_gpx)
    proj = G2sc.G2Project(gpxfile=out_gpx)
    apply_safe_limits(proj)
    hist, main_phase = get_hist_and_main_phase(proj, main_phase_name)

    # Init HAP scales to a normalized split
    S0 = 1.0
    N_cand = len(pid_to_cif)
    base_main = 1.0
    base_cand = float(hap_init)
    norm = S0 / (base_main + N_cand * base_cand) if (base_main + N_cand * base_cand) > 0 else 1.0
    main_init = base_main * norm
    cand_init = base_cand * norm
    set_phase_cell_refine(main_phase, refine=False)
    main_phase.set_HAP_refinements({'Scale': True}, histograms=[hist])
    main_phase.HAPvalue('Scale', float(main_init), targethistlist=[hist])

    for pid, cif in pid_to_cif.items():
        p = proj.add_phase(cif, phasename=str(pid), histograms=[hist])
        set_phase_cell_refine(p, refine=False)
        p.set_HAP_refinements({'Scale': True}, histograms=[hist])
        p.HAPvalue('Scale', float(cand_init), targethistlist=[hist])

    # Background ON; Sample Scale held; zero-shift & instrument profile held
    hist.set_refinements({'Background': {'refine': True}})
    try:
        hist.clear_refinements({'Sample Parameters': ['Scale']})
    except Exception:
        pass
    for var in HISTOGRAM_HOLD_VARS:
        try:
            _add_histogram_hold_constraint(proj, hist, var)
        except Exception as e:
            logger.warning(f"[joint+] Could not hold histogram variable {var}: {e}")

    # Ensure no HAP constraints (no sum-to-one coupling)
    cons = proj.data.setdefault('Constraints', {})
    if 'HAP' in cons and cons['HAP']:
        cons['HAP'] = []
        logger.info("[joint] Cleared existing HAP constraints.")

    proj.data['Controls']['data']['max cyc'] = int(max_joint_cycles)
    proj.do_refinements([
        {'set': {'Background': {'refine': True}}},
        {'refine': True},
    ])

    # Parse fractions
    lst_path = Path(out_gpx).with_suffix(".lst")
    parsed = parse_gsas_lst(lst_path, hist.name) if lst_path.exists() else {}
    wanted_names = {main_phase.name, *map(str, pid_to_cif.keys())}
    results: Dict[str, Dict[str, float]] = {}
    for name in wanted_names:
        vals = parsed.get(name)
        results[name] = {
            "phase_fraction_pct": float(vals["phase_fraction_pct"]) if vals else 0.0,
            "weight_fraction_pct": float(vals["weight_fraction_pct"]) if vals else 0.0,
        }
    return results

# === BEGIN ADD: read residual & Rwp from an existing GPX =====================
def extract_residual_from_gpx(gpx_path: str):
    """
    Open a GPX and return:
      (x_native, residual_native, Q, residual_Q, rwp, hist_name, project_obj)

    Notes:
    - Returns the first histogram (single-hist assumption consistent with rest of pipeline).
    - Caller is responsible for .save()/.close() if they modify project; for read-only
      use this as-is and let GC clean up.
    """
    from GSASII import GSASIIscriptable as G2sc
    import numpy as np

    proj = G2sc.G2Project(gpxfile=gpx_path)
    hists = proj.histograms()
    if not hists:
        raise RuntimeError(f"No histograms in {gpx_path}")

    hist = hists[0]
    data = GSASDataExtractor.get_all_arrays(hist)

    # Fallbacks ensure np.array([]) rather than None
    x_native = np.asarray(data.get('x_native', np.array([])), float)
    Q = np.asarray(data.get('Q', np.array([])), float)
    residual = np.asarray(data.get('residual', np.array([])), float)

    # Same residual sampled vs Q points (GSAS stores one residual vector;
    # we expose it twice to match the pipeline API)
    residual_native = residual
    residual_Q = residual
    validate_residual_arrays(
        x_native,
        residual_native,
        Q,
        residual_Q,
        context=f"Residual extraction from {Path(gpx_path).name}",
    )

    try:
        rwp = float(hist.get_wR())
    except Exception:
        rwp = float('nan')

    return x_native, residual_native, Q, residual_Q, rwp, hist.name, proj

def _validate_candidates_individually(
    base_gpx: str,
    out_gpx: str, 
    main_phase_name: str,
    pid_to_cif: Dict[str, str],
    hap_init: float
) -> Dict[str, str]:
    """
    Helper to check each candidate phase in isolation.
    Returns: Dict[pid, cif] of only the SAFE candidates.
    """
    from GSASII import GSASIIscriptable as G2sc
    import os
    import sys
    from io import StringIO
    import shutil

    validated_cifs: Dict[str, str] = {}
    if not pid_to_cif:
        return {}

    logger.info(f"[joint+] Validating {len(pid_to_cif)} candidates individually...")
    val_gpx = str(Path(out_gpx).with_suffix(".validation.gpx"))
    
    for pid, cif in pid_to_cif.items():
        proj_val = None
        try:
            if os.path.exists(val_gpx):
                try:
                    os.remove(val_gpx)
                except Exception as e:
                    logger.warning(f"[joint+] Could not remove stale validation GPX {val_gpx}: {e}")
            
            shutil.copy(base_gpx, val_gpx)
            proj_val = G2sc.G2Project(gpxfile=val_gpx)
            apply_safe_limits(proj_val) # Enforce physical data limits
            hist_val, _ = get_hist_and_main_phase(proj_val, main_phase_name)
            
            # 1. Add phase
            p_val = proj_val.add_phase(cif, phasename=str(pid), histograms=[hist_val])
            
            # 2. Setup Refinement: ONLY Scale and Background
            # Candidate Scale -> refine=True
            p_val.set_HAP_refinements({'Scale': True}, histograms=[hist_val])
            p_val.HAPvalue('Scale', float(hap_init), targethistlist=[hist_val])
            
            # Background -> refine=True
            hist_val.set_refinements({'Background': {'refine': True}})
            
            # EVERYTHING ELSE -> fixed
            set_phase_cell_refine(p_val, refine=False)
            for other_p in proj_val.phases():
                if other_p.name != p_val.name:
                    set_phase_cell_refine(other_p, refine=False)
                    other_p.set_HAP_refinements({'Scale': False}, histograms=[hist_val])
            
            # Sample params, Zero, profile -> ensure fixed
            try:
                hist_val.clear_refinements({'Sample Parameters': ['Scale']})
            except Exception as e:
                logger.warning(f"[joint+] Candidate {pid} could not clear sample Scale refinement: {e}")
            
            proj_val.data['Controls']['data']['max cyc'] = 1
            
            # 3. Test Drive with stdout capture
            capture = StringIO()
            old_stdout = sys.stdout
            try:
                sys.stdout = capture
                proj_val.do_refinements([{'refine': True}])
            finally:
                sys.stdout = old_stdout
            
            output = capture.getvalue()
            
            # Always print full output so user can diagnose
            logger.debug(f"[joint+] Candidate {pid} stdout:\n{output}")
            
            # Only reject on HARD failures, not recoverable warnings like ouch #7
            hard_fail = ("Refinement error" in output 
                         or "ouch #0" in output 
                         or "ouch #3" in output 
                         or "recip-matrix error" in output
                         or "ERROR - Refinement failed" in output)
            if hard_fail:
                logger.warning(f"[joint+] Candidate {pid} REJECTED (hard failure detected)")
                continue
            else:
                logger.info(f"[joint+] Candidate {pid} PASSED individual validation")
                
            validated_cifs[pid] = cif
            
        except Exception as e:
            logger.warning(f"[joint+] Candidate {pid} rejected (exception): {e}")
        finally:
            if proj_val: del proj_val
            if os.path.exists(val_gpx):
                try:
                    os.remove(val_gpx)
                except Exception as e:
                    logger.warning(f"[joint+] Could not cleanup validation GPX {val_gpx}: {e}")
    
    return validated_cifs
# === END ADD =================================================================
# === BEGIN ADD: joint_refine_add_phases ======================================
def joint_refine_add_phases(
    base_gpx: str,
    out_gpx: str,
    main_phase_name: str,
    pid_to_cif_new: Dict[str, str],
    hap_init: float = 0.05,
    max_joint_cycles: int = 1,
    preserve_existing_scales: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Clone base GPX, add *new* candidate phases, and refine HAP Scales for all phases.
    Optimistic approach: try batch first, if crash, validate individually and retry.
    """
    from GSASII import GSASIIscriptable as G2sc
    import sys
    from io import StringIO
    import os

    # 1. Prepare project with ALL candidates
    clone_gpx(base_gpx, out_gpx)
    proj = G2sc.G2Project(gpxfile=out_gpx)
    apply_safe_limits(proj) # Enforce physical data limits
    hist, main_phase = get_hist_and_main_phase(proj, main_phase_name)

    existing = {p.name: p for p in proj.phases()}

    # Existing phases -> setup
    for p in existing.values():
        set_phase_cell_refine(p, refine=False)
        p.set_HAP_refinements({'Scale': True}, histograms=[hist])

    # Add ALL new phases
    for pid, cif in pid_to_cif_new.items():
        if pid in existing:
             continue
        p = proj.add_phase(cif, phasename=str(pid), histograms=[hist])
        set_phase_cell_refine(p, refine=False)
        p.set_HAP_refinements({'Scale': True}, histograms=[hist])
        p.HAPvalue('Scale', float(hap_init), targethistlist=[hist])

    # Background ON; Sample Scale held; zero/profile held
    hist.set_refinements({'Background': {'refine': True}})
    try:
        hist.clear_refinements({'Sample Parameters': ['Scale']})
    except Exception as e:
        logger.warning(f"[joint+] Could not clear sample Scale refinement for batch run: {e}")
    
    # Hold instrument params (U,V,W,X,Y,Z, Zero)
    for var in HISTOGRAM_HOLD_VARS:
        try:
            _add_histogram_hold_constraint(proj, hist, var)
        except Exception as e:
            logger.warning(f"[joint+] Could not hold histogram variable {var}: {e}")

    proj.data['Controls']['data']['max cyc'] = int(max_joint_cycles)
    
    # 2. Try Batch Refinement
    refine_success = False
    batch_error = ""
    try:
        capture = StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = capture
            proj.do_refinements([{'refine': True}])
        finally:
            sys.stdout = old_stdout
            
        output = capture.getvalue()
        
        # Print full batch output for diagnostics
        logger.debug(f"[joint+] Batch refinement stdout:\n{output}")
        
        # Only fail on HARD errors, not recoverable warnings like ouch #7
        hard_fail = ("Refinement error" in output 
                     or "ouch #0" in output 
                     or "ouch #3" in output 
                     or "recip-matrix error" in output
                     or "ERROR - Refinement failed" in output)
        if hard_fail:
             batch_error = f"GSAS-II Refinement failed (stdout): {output[:200]}..."
             raise RuntimeError(batch_error)
             
        refine_success = True
        
    except (Exception, RuntimeError) as e:
        batch_error = str(e)
        logger.warning(f"[joint+] Batch refinement failed ({batch_error}). Falling back to individual validation...")
        
        # 3. Fallback: Validate and Retry
        del proj # Release handle
        
        valid_subset = _validate_candidates_individually(
            base_gpx, out_gpx, main_phase_name, pid_to_cif_new, hap_init
        )
        
        if len(valid_subset) < len(pid_to_cif_new):
             logger.info(f"[joint+] Retrying batch with {len(valid_subset)} valid candidates...")
             # Recursive call for clean state
             if not valid_subset:
                  logger.info("[joint+] No valid candidates found. Returning empty results.")
                  return {}
             
             return joint_refine_add_phases(
                 base_gpx, out_gpx, main_phase_name, valid_subset,
                 hap_init, max_joint_cycles, preserve_existing_scales
             )
        else:
             logger.warning("[joint+] Validation passed all candidates, but batch failed. Hard failure.")
             raise e

    # 4. Success -> Extract Results
    if refine_success:
        # Re-open or use saved results
        lst_path = Path(out_gpx).with_suffix(".lst")
        parsed = parse_gsas_lst(lst_path, hist.name) if lst_path.exists() else {}
        results: Dict[str, Dict[str, float]] = {}
        
        # We need results for ALL phases in the project
        proj = G2sc.G2Project(gpxfile=out_gpx) # reload to be sure
        for p in proj.phases():
            nm = p.name
            vals = parsed.get(nm)
            results[nm] = {
                "phase_fraction_pct": float(vals["phase_fraction_pct"]) if vals else 0.0,
                "weight_fraction_pct": float(vals["weight_fraction_pct"]) if vals else 0.0,
            }
        return results
    
    return {}
# === END ADD =================================================================
# === BEGIN ADD: joint_refine_polish (multi-cycle, optional Cell refine) ======
# === BEGIN: joint_refine_polish (bullet-proof, transactional) ===

import math, re, shutil
from pathlib import Path
from typing import Dict, Tuple, Iterable, Optional, List
import shutil, math, re
from pathlib import Path
from typing import Tuple, Dict, List, Optional




def joint_refine_polish(
    base_gpx: str,
    out_gpx: str,
    main_phase_name: str,
    max_polish_cycles: int = 10,
    refine_cell_for_all: bool = True,
    refine_background: bool = True,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """
    Transactional polish refinement with strict checkpoint integrity.
    Phases are tested cumulatively: each success adds to the enabled set.

    Success/failure is determined SOLELY by the presence of a GOF in the
    .lst (derived from the .gpx name) and GOF <= threshold (default 50).
    """
    import os
    import math, shutil
    from pathlib import Path

    # GOF threshold (configurable via env; default 50.0)
    try:
        GOF_MAX = float(os.environ.get("GSAS_MAX_GOF", "50.0"))
    except Exception:
        GOF_MAX = 50.0

    STAB_CYCLES = max(1, min(3, max_polish_cycles))
    PER_PHASE_MAX = 3

    from GSASII import GSASIIscriptable as G2sc
    try:
        from gsas_main_phase_refiner import parse_gsas_lst as _parse_ext
    except Exception:
        _parse_ext = None  # will fall back to local parse_gsas_lst

    # ---------- utilities ----------
    def _clone(src: str, dst: str) -> None:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def _open(path: str):
        proj = G2sc.G2Project(gpxfile=path)
        apply_safe_limits(proj)
        return proj

    def _save(proj, path: str) -> None:
        proj.save(path)

    def _lst_path(path: str) -> Path:
        return Path(path).with_suffix(".lst")

    def _hist_and_main(proj, main_name: str):
        hists = proj.histograms()
        if not hists:
            raise RuntimeError("No histogram in project")
        hist = hists[0]
        main = None
        for p in proj.phases():
            if p.name == main_name:
                main = p
                break
        return hist, main

    def _set_bg(hist, on: bool) -> None:
        try:
            hist.set_refinements({'Background': {'refine': bool(on)}})
        except Exception:
            pass

    def _set_all_scales_on(proj, hist) -> None:
        for p in proj.phases():
            try:
                p.set_HAP_refinements({'Scale': True}, histograms=[hist])
            except Exception:
                pass

    def _set_all_cell(proj, on: bool) -> None:
        for p in proj.phases():
            try:
                p.set_refinements({'Cell': bool(on)})
            except Exception:
                pass

    def _set_cell_for_list(proj, target_names: List[str]) -> None:
        """Enable cell refinement only for phases in target_names list."""
        for p in proj.phases():
            try:
                p.set_refinements({'Cell': p.name in target_names})
            except Exception:
                pass

    def _set_max_cycles(proj, ncyc: int) -> None:
        try:
            proj.data['Controls']['data']['max cyc'] = int(max(1, ncyc))
        except Exception:
            pass

    def _refine(proj, bg_on: bool) -> None:
        """Run refinement. Note: GSAS-II doesn't reliably return failure status."""
        proj.do_refinements([
            {'set': {'Background': {'refine': bool(bg_on)}}},
            {'refine': True},
        ])

    def _phase_order(proj, hist, main_name: str, lst: Path) -> List[str]:
        weights: Dict[str, float] = {}
        parse_func = _parse_ext or parse_gsas_lst
        if lst.exists():
            try:
                parsed = parse_func(lst, hist.name) or {}
                for p in proj.phases():
                    nm = p.name
                    weights[nm] = float(parsed.get(nm, {}).get('weight_fraction_pct', 0.0))
            except Exception:
                pass

        def key(nm: str):
            return (0 if nm == main_name else 1, -weights.get(nm, 0.0), nm)

        return sorted([p.name for p in proj.phases()], key=key)

    def _phase_cell6(proj, phase_name: str) -> Optional[Tuple[float, float, float, float, float, float]]:
        for p in proj.phases():
            if p.name == phase_name:
                try:
                    cell_dict, _ = p.get_cell_and_esd()
                    a = float(cell_dict['length_a'])
                    b = float(cell_dict['length_b'])
                    c = float(cell_dict['length_c'])
                    al = float(cell_dict['angle_alpha'])
                    be = float(cell_dict['angle_beta'])
                    ga = float(cell_dict['angle_gamma'])
                    return (a, b, c, al, be, ga)
                except Exception:
                    return None
        return None

    def _posdef_G6(cell6: Tuple[float, float, float, float, float, float]) -> bool:
        a, b, c, alp, bet, gam = cell6
        if not (1.5 <= a <= 60.0 and 1.5 <= b <= 60.0 and 1.5 <= c <= 60.0):
            return False
        if not (20.0 <= alp <= 160.0 and 20.0 <= bet <= 160.0 and 20.0 <= gam <= 160.0):
            return False
        ca, cb, cg = math.cos(math.radians(alp)), math.cos(math.radians(bet)), math.cos(math.radians(gam))
        g11, g22, g33 = a*a, b*b, c*c
        g12, g13, g23 = a*b*cg, a*c*cb, b*c*ca
        m1 = g11
        m2 = g11*g22 - g12*g12
        det = (g11*g22*g33 + 2*g12*g13*g23) - (g11*g23*g23 + g22*g13*g13 + g33*g12*g12)
        return (m1 > 0.0) and (m2 > 0.0) and (det > 1e-8)

    # ---- GOF-only success check ----
    _GOF_RE = re.compile(r'GOF\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)', re.IGNORECASE)

    def _read_gof(lst: Path) -> Optional[float]:
        if not lst.exists():
            return None
        try:
            txt = lst.read_text(errors="ignore")
        except Exception:
            return None
        m = _GOF_RE.findall(txt)
        if not m:
            return None
        try:
            return float(m[-1])
        except Exception:
            return None

    def _lst_gof_ok(lst: Path) -> Tuple[bool, Optional[float]]:
        gof = _read_gof(lst)
        if gof is None:
            return (False, None)
        return (gof <= GOF_MAX, gof)

    def _regenerate_lst(gpx_path: str, bg_on: bool) -> bool:
        """
        Regenerate .lst file for a GPX with a short refine.
        Success means a GOF is present and <= GOF_MAX.
        """
        try:
            proj = _open(gpx_path)
            hist = proj.histograms()[0]
            _set_max_cycles(proj, 1)          # 1 cycle to ensure GOF is written
            _set_bg(hist, bg_on)
            _save(proj, gpx_path)
            _refine(proj, bg_on)
            _save(proj, gpx_path)

            ok, gof = _lst_gof_ok(_lst_path(gpx_path))
            if not ok:
                logger.warning(f"[polish] Regenerated .lst failed GOF check (GOF={gof!r}, req ≤ {GOF_MAX}).")
                return False
            logger.info(f"[polish] Regenerated .lst OK (GOF={gof:.3f}) for {Path(gpx_path).name}")
            return True
        except Exception as e:
            logger.warning(f"[polish] Failed to regenerate .lst: {e}")
            return False

    def _scale_fraction_fallback(proj, hist_name: str) -> Dict[str, Dict[str, float]]:
        vals: Dict[str, float] = {}
        for p in proj.phases():
            try:
                hcfg = p.data.get('Histograms', {}).get(hist_name, {})
                if hcfg.get('Use', True) is False:
                    continue
                sc = hcfg.get('Scale', [0.0, False])
                sval = float(sc[0]) if isinstance(sc, (list, tuple)) and len(sc) else float(sc)
                if math.isfinite(sval) and sval > 0:
                    vals[p.name] = sval
            except Exception:
                continue
        tot = sum(vals.values())
        out: Dict[str, Dict[str, float]] = {}
        if tot > 0:
            for k, v in vals.items():
                pct = 100.0 * v / tot
                out[k] = {
                    "phase_fraction_pct": float(pct),
                    "weight_fraction_pct": float(pct),
                }
        return out

    def _final_readout(proj, lst_path: Path, hist_name: str) -> Tuple[Dict[str, Dict[str, float]], float]:
        results: Dict[str, Dict[str, float]] = {}
        parsed = {}
        fallback = _scale_fraction_fallback(proj, hist_name)
        parse_func = _parse_ext or parse_gsas_lst
        try:
            if lst_path.exists():
                parsed = parse_func(lst_path, hist_name) or {}
        except Exception:
            parsed = {}
        for p in proj.phases():
            nm = p.name
            vals = parsed.get(nm)
            if vals is None:
                vals = fallback.get(nm)
            results[nm] = {
                "phase_fraction_pct": float(vals["phase_fraction_pct"]) if vals else 0.0,
                "weight_fraction_pct": float(vals["weight_fraction_pct"]) if vals else 0.0,
            }
        try:
            rwp = float(proj.histograms()[0].get_wR())
        except Exception:
            rwp = float('nan')
        return results, rwp

    # === MAIN FLOW ===

    # Initialize with clean base
    _clone(base_gpx, out_gpx)
    checkpoint = Path(out_gpx).with_suffix(".checkpoint.gpx")
    _clone(out_gpx, str(checkpoint))
    proj0 = _open(out_gpx)
    hist0 = proj0.histograms()[0]
    last_good_results, _ = _final_readout(proj0, _lst_path(out_gpx), hist0.name)

    # === STABILIZATION PHASE ===
    print("[polish] Starting stabilization phase...")
    proj = _open(out_gpx)
    hist, _main = _hist_and_main(proj, main_phase_name)
    _set_all_cell(proj, False)
    _set_bg(hist, refine_background)
    _set_all_scales_on(proj, hist)
    _set_max_cycles(proj, STAB_CYCLES)

    try:
        _save(proj, out_gpx)
        _refine(proj, refine_background)
        _save(proj, out_gpx)
    except Exception as e:
        print(f"[polish] Stabilization exception: {e}. Reverting to base.")
        _clone(str(checkpoint), out_gpx)
        _regenerate_lst(out_gpx, refine_background)
        proj = _open(out_gpx)
        return _final_readout(proj, _lst_path(out_gpx), hist.name)

    # Validate stabilization via GOF
    ok, gof = _lst_gof_ok(_lst_path(out_gpx))
    if not ok:
        print(f"[polish] Stabilization failed GOF check (GOF={gof!r}, req ≤ {GOF_MAX}). Reverting to base.")
        _clone(str(checkpoint), out_gpx)
        _regenerate_lst(out_gpx, refine_background)
        proj = _open(out_gpx)
        return _final_readout(proj, _lst_path(out_gpx), hist.name)

    # Commit stabilization to checkpoint
    print(f"[polish] Stabilization OK (GOF={gof:.3f}). Updating checkpoint.")
    _clone(out_gpx, str(checkpoint))
    proj_ckpt = _open(str(checkpoint))
    last_good_results, _ = _final_readout(proj_ckpt, _lst_path(str(checkpoint)), hist.name)

    # === CUMULATIVE PHASE-BY-PHASE CELL REFINEMENT ===
    remaining = max(0, max_polish_cycles - STAB_CYCLES)
    if remaining == 0:
        print("[polish] No cycles remaining for phase refinement.")
        proj = _open(out_gpx)
        return _final_readout(proj, _lst_path(out_gpx), hist.name)

    proj = _open(out_gpx)
    hist, _main = _hist_and_main(proj, main_phase_name)
    order = _phase_order(proj, hist, main_phase_name, _lst_path(out_gpx))
    per_phase = max(1, min(PER_PHASE_MAX, remaining // max(1, len(order))))

    print(f"[polish] Phase refinement order: {order}")
    print(f"[polish] Cycles per phase: {per_phase}")

    enabled: List[str] = []  # Track successfully enabled phases

    for nm in order:
        if remaining <= 0:
            print(f"[polish] No cycles remaining. Stopping.")
            break

        print(f"[polish] Attempting to add phase: {nm} (enabled: {enabled})")

        # Create candidate list: all previously enabled + current phase
        candidate_enabled = enabled + [nm]

        # Create a temporary working copy from checkpoint
        temp_gpx = str(Path(out_gpx).with_suffix(".temp.gpx"))
        _clone(str(checkpoint), temp_gpx)

        try:
            # Work on the temp copy with cumulative enabling
            proj_temp = _open(temp_gpx)
            hist_temp, _ = _hist_and_main(proj_temp, main_phase_name)

            # Enable cell for all phases in candidate list
            _set_cell_for_list(proj_temp, candidate_enabled)
            _set_bg(hist_temp, refine_background)
            _set_all_scales_on(proj_temp, hist_temp)
            _set_max_cycles(proj_temp, per_phase)

            _save(proj_temp, temp_gpx)
            _refine(proj_temp, refine_background)
            _save(proj_temp, temp_gpx)

            # GOF check FIRST
            ok, gof = _lst_gof_ok(_lst_path(temp_gpx))
            if not ok:
                print(f"[polish] Phase {nm}: GOF check failed (GOF={gof!r}, req ≤ {GOF_MAX}). Skipping.")
                continue

            # Reopen to get fresh cell values - check ALL enabled phases
            proj_temp = _open(temp_gpx)
            all_valid = True
            for phase_name in candidate_enabled:
                c6 = _phase_cell6(proj_temp, phase_name)
                if (c6 is None) or (not _posdef_G6(c6)):
                    print(f"[polish] Phase {phase_name} has invalid cell. Skipping {nm}.")
                    all_valid = False
                    break

            if not all_valid:
                continue

            # Success! Commit temp to both out_gpx and checkpoint
            print(f"[polish] Phase {nm}: success (GOF={gof:.3f}) with {len(candidate_enabled)} phase(s) enabled.")
            _clone(temp_gpx, out_gpx)
            _clone(temp_gpx, str(checkpoint))
            proj_ckpt = _open(str(checkpoint))
            last_good_results, _ = _final_readout(proj_ckpt, _lst_path(str(checkpoint)), hist.name)
            enabled.append(nm)  # Add to enabled list
            remaining -= per_phase

        except Exception as e:
            print(f"[polish] Phase {nm}: Refinement exception: {e}. Skipping.")
            continue
        finally:
            # Clean up temp files
            tp = Path(temp_gpx)
            if tp.exists():
                tp.unlink()
            tl = _lst_path(temp_gpx)
            if tl.exists():
                tl.unlink()

    # === OPTIONAL FINAL POLISH (with remaining cycles) ===
    if enabled and remaining > 0:
        print(f"[polish] Running final polish with {remaining} cycles on {len(enabled)} enabled phase(s)...")

        temp_gpx = str(Path(out_gpx).with_suffix(".temp.gpx"))
        _clone(str(checkpoint), temp_gpx)

        try:
            proj_temp = _open(temp_gpx)
            hist_temp, _ = _hist_and_main(proj_temp, main_phase_name)

            _set_cell_for_list(proj_temp, enabled)
            _set_bg(hist_temp, refine_background)
            _set_all_scales_on(proj_temp, hist_temp)
            _set_max_cycles(proj_temp, remaining)

            _save(proj_temp, temp_gpx)
            _refine(proj_temp, refine_background)
            _save(proj_temp, temp_gpx)

            ok, gof = _lst_gof_ok(_lst_path(temp_gpx))
            if not ok:
                print(f"[polish] Final polish failed GOF check (GOF={gof!r}, req ≤ {GOF_MAX}). Keeping last good state.")
            else:
                # Then check cells
                proj_temp = _open(temp_gpx)
                all_valid = True
                for nm in enabled:
                    c6 = _phase_cell6(proj_temp, nm)
                    if (c6 is None) or (not _posdef_G6(c6)):
                        print(f"[polish] Phase {nm} became invalid in final polish.")
                        all_valid = False
                        break

                if all_valid:
                    print(f"[polish] Final polish successful (GOF={gof:.3f}).")
                    _clone(temp_gpx, out_gpx)
                    _clone(temp_gpx, str(checkpoint))
                    proj_ckpt = _open(str(checkpoint))
                    last_good_results, _ = _final_readout(proj_ckpt, _lst_path(str(checkpoint)), hist.name)
                else:
                    print(f"[polish] Final polish cells invalid. Keeping last good state.")

        except Exception as e:
            print(f"[polish] Final polish exception: {e}. Keeping last good state.")
        finally:
            tp = Path(temp_gpx)
            if tp.exists():
                tp.unlink()
            tl = _lst_path(temp_gpx)
            if tl.exists():
                tl.unlink()

    # CRITICAL: Final restoration and .lst regeneration (ensure .lst matches final GPX)
    logger.info("[polish] Finalizing output...")
    _clone(str(checkpoint), out_gpx)
    regen_ok = _regenerate_lst(out_gpx, refine_background)

    proj = _open(out_gpx)

    if enabled:
        logger.info(f"[polish] Cell refinement enabled for: {', '.join(enabled)}")
    else:
        logger.info("[polish] No phases accepted for cell refinement during polish.")

    final_results, final_rwp = _final_readout(proj, _lst_path(out_gpx), hist.name)
    if (not regen_ok) or (sum(v.get("weight_fraction_pct", 0.0) for v in final_results.values()) <= 0.0):
        logger.warning("[polish] Using last-good fractions fallback after failed/empty final readout.")
        return last_good_results, final_rwp
    return final_results, final_rwp


# === BEGIN REPLACE: plot_gpx_fit_with_ticks (publication-grade, 2 panels) ===
def plot_gpx_fit_with_ticks(
    gpx_path: str,
    out_png: str,
    downsample: int = 1,
    max_ticks_per_phase: int = 1000000,
    phase_labels: Optional[Dict[str, str]] = None,
):
    """
    Publication-grade plot with two panels:
      Top: Observed (points), Calculated (red), and Residual (offset blue) vs native x (2θ or TOF)
      Bottom: Bragg tick rows, one row per phase, labeled "PhaseID — Wt%"

    - Keeps function signature unchanged.
    - Ticks are clipped to the x-range and thinned if needed.
    - Phases are ordered by descending Wt% (main usually first).
    - Designed to avoid text clipping/cropping in saved PNG.

    If plotting fails, it logs a warning and returns.
    """
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        from matplotlib.patches import Rectangle
        import matplotlib.patches as mpatches
    except Exception as e:
        logger.warning(f"[plot] matplotlib not available: {e}")
        return

    try:
        from GSASII import GSASIIscriptable as G2sc
    except Exception as e:
        logger.warning(f"[plot] GSAS-II not available for plotting: {e}")
        return

    try:
        from gsas_core_infrastructure import CoordinateHandler
    except Exception as e:
        logger.warning(f"[plot] CoordinateHandler not available: {e}")
        return

    # Optional parse for Wt% labels
    try:
        from gsas_main_phase_refiner import parse_gsas_lst
    except Exception:
        parse_gsas_lst = None  # graceful fallback

    # ---------------------------
    # Load project / histogram
    # ---------------------------
    proj = G2sc.G2Project(gpxfile=gpx_path)
    hists = proj.histograms()
    if not hists:
        logger.warning(f"[plot] No histogram in {gpx_path}")
        return
    hist = hists[0]

    # ---------------------------
    # Data arrays
    # ---------------------------
    data = GSASDataExtractor.get_all_arrays(hist)
    x = np.asarray(data.get('x_native', np.array([], float)), float)
    yobs = np.asarray(data.get('yobs', np.array([], float)), float)
    ycalc = np.asarray(data.get('ycalc', np.array([], float)), float)
    if x.size == 0 or yobs.size == 0 or ycalc.size == 0:
        logger.info("[plot] Missing x/yobs/ycalc; skipped.")
        return

    resid = yobs - ycalc

    if downsample and downsample > 1:
        x = x[::downsample]
        yobs = yobs[::downsample]
        ycalc = ycalc[::downsample]
        resid = resid[::downsample]

    # ---------------------------
    # Coordinate / axis labeling
    # ---------------------------
    try:
        ch = CoordinateHandler.from_gsas_histogram(hist)
        inst = ch.instrument_type  # "CW" / "TOF"
    except Exception:
        ch, inst = None, "Unknown"
    
    # Enhanced axis labels with proper formatting
    if inst == "CW":
        xlabel = r"$2\theta$ (degrees)"
    elif inst == "TOF":
        xlabel = "Time-of-Flight (μs)"
    else:
        xlabel = "Diffraction Angle"

    # ---------------------------
    # Phase order and Wt% parsing
    # ---------------------------
    phase_names = [p.name for p in proj.phases()]  # IDs in GPX
    wt = {nm: 0.0 for nm in phase_names}
    try:
        from pathlib import Path as _P
        lst_path = _P(gpx_path).with_suffix(".lst")
        parsed = parse_gsas_lst(lst_path, hist.name) if (parse_gsas_lst and lst_path.exists()) else {}
        if isinstance(parsed, dict):
            for nm in phase_names:
                wt[nm] = float(parsed.get(nm, {}).get('weight_fraction_pct', 0.0))
    except Exception:
        pass
    phase_order = sorted(phase_names, key=lambda nm: wt.get(nm, 0.0), reverse=True)

    # ---------------------------
    # Reflection ticks by phase

    # ---------------------------
    ticks_by_phase = {}
    try:
        refls = hist.reflections() or {}
        for p_obj, info in refls.items():
            pname = getattr(p_obj, 'name', None) or getattr(p_obj, 'phaseName', None) or str(p_obj)
            ref_list = np.asarray(info.get('RefList', []))
            if ref_list.size == 0 or ref_list.ndim != 2:
                continue
            is_super = bool(info.get('Super', False))
            d_col = 5 if is_super else 4
            if ref_list.shape[1] <= d_col:
                continue

            d_vals = ref_list[:, d_col].astype(float)
            d_vals = d_vals[np.isfinite(d_vals) & (d_vals > 0.0)]
            if d_vals.size == 0:
                continue

            if ch is not None:
                x_ticks = ch.d_to_native(d_vals)
            else:
                x_ticks = np.array([], float)

            x_ticks = x_ticks[np.isfinite(x_ticks)]
            if x_ticks.size == 0:
                continue

            # BUGFIX: Proper handling of x-range for both CW and TOF
            x_lo, x_hi = float(np.min(x)), float(np.max(x))
            
            # For TOF, data might be in descending order, so handle both cases
            x_min_bound = min(x_lo, x_hi)
            x_max_bound = max(x_lo, x_hi)
            
            # Clip ticks to actual data range with small tolerance for edge cases
            tolerance = (x_max_bound - x_min_bound) * 0.001  # 0.1% tolerance
            m = (x_ticks >= (x_min_bound - tolerance)) & (x_ticks <= (x_max_bound + tolerance))
            x_ticks = x_ticks[m]
            
            if x_ticks.size == 0:
                continue
            
            # Sort based on whether x data is ascending or descending
            if x_lo < x_hi:
                x_ticks = np.sort(x_ticks)  # Ascending (typical for 2theta)
            else:
                x_ticks = np.sort(x_ticks)[::-1]  # Descending (some TOF data)

            if max_ticks_per_phase and x_ticks.size > max_ticks_per_phase:
                step = int(np.ceil(x_ticks.size / max_ticks_per_phase))
                x_ticks = x_ticks[::step]

            ticks_by_phase[pname] = x_ticks
    except Exception as e:
        logger.warning(f"[plot] Could not compute reflection ticks: {e}")
        ticks_by_phase = {}

    # ---------------------------
    # Calculate quality metrics
    # ---------------------------
    try:
        rwp = float(hist.get_wR())
        chi2 = float(hist.get('Durbin-Watson', 0.0))  # Try to get chi-squared
    except Exception:
        rwp = None
        chi2 = None
    
    # Calculate additional metrics
    n_points = len(yobs)
    n_phases = len(phase_order)
    
    # ---------------------------
    # Enhanced aesthetics (rc) & Figure
    # ---------------------------
    rc = {
        # Figure settings
        "figure.constrained_layout.use": True,
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        
        # Font settings - professional publication style
        "font.family": ["sans-serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        
        # Axes settings
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "axes.labelsize": 11,
        "axes.labelweight": "normal",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        
        # Tick settings
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.major.size": 5,
        "ytick.minor.size": 3,
        "xtick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.major.width": 0.8,
        "ytick.minor.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        
        # Grid settings
        "grid.color": "#e0e0e0",
        "grid.alpha": 0.5,
        "grid.linestyle": "-",
        "grid.linewidth": 0.5,
        
        # Legend settings
        "legend.fontsize": 9,
        "legend.frameon": True,
        "legend.fancybox": True,
        "legend.shadow": False,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.borderpad": 0.5,
    }

    with plt.rc_context(rc):
        # Create figure with golden ratio proportions
        fig = plt.figure(figsize=(12, 7.4), constrained_layout=True)
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[4.5, 1.5], hspace=0.02)
        ax = fig.add_subplot(gs[0, 0])
        ax_ticks = fig.add_subplot(gs[1, 0], sharex=ax)

        # ---------------------------
        # Top panel: Obs/Calc/Residual with enhanced styling
        # ---------------------------
        
        # Add subtle background gradient for depth
        ax.axhspan(ax.get_ylim()[0], ax.get_ylim()[1], 
                   facecolor='white', alpha=1.0, zorder=0)
        
        # Observed: sophisticated scatter plot
        if downsample <= 2:
            # High quality plot for low downsampling
            ax.scatter(x, yobs, s=3, c="#1a1a1a", marker="o", 
                      edgecolors="none", label="Observed", 
                      zorder=4, alpha=0.7, rasterized=True)
        else:
            # Line plot with markers for heavy downsampling
            ax.plot(x, yobs, 'o', color="#1a1a1a", markersize=2,
                   label="Observed", zorder=4, alpha=0.6, 
                   markeredgecolor='none', rasterized=True)

        # Calculated: sophisticated red line with subtle shadow
        ax.plot(x, ycalc, color="#d62728", lw=1.5, label="Calculated", 
                zorder=3, alpha=0.95)
        # Add subtle shadow effect
        ax.plot(x, ycalc, color="#d62728", lw=3.0, zorder=2, 
                alpha=0.15)

        # Calculate residual statistics
        y_min = float(np.nanmin([np.nanmin(yobs), np.nanmin(ycalc)]))
        y_max = float(np.nanmax([np.nanmax(yobs), np.nanmax(ycalc)]))
        yr = max(y_max - y_min, 1.0)
        
        # Enhanced residual display with filled area
        resid_amp = float(np.nanmax(np.abs(resid))) if np.isfinite(resid).any() else 1.0
        scale = (0.10 * yr) / max(resid_amp, 1e-12)
        base = y_min - 0.18 * yr
        
        # Fill area for residuals (more visually appealing)
        ax.fill_between(x, base, base + resid * scale, 
                        color="#2ca02c", alpha=0.3, label="Difference",
                        zorder=1)
        ax.plot(x, base + resid * scale, color="#2ca02c", lw=0.8, 
                zorder=2, alpha=0.8)
        
        # Enhanced baseline
        ax.axhline(base, color="#666666", lw=1.0, linestyle="-", 
                   alpha=0.3, zorder=0)

        # Y-axis formatting
        ax.set_ylabel("Intensity (a.u.)", fontweight='medium')
        ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 3))
        
        # Enhanced title with multiple metrics
        title_parts = []
        if rwp is not None:
            title_parts.append(f"R$_{{wp}}$ = {rwp:.2f}%")
        if chi2 and chi2 > 0:
            title_parts.append(f"χ² = {chi2:.2f}")
        title_parts.append(f"N = {n_points:,}")
        
        metrics_str = "  •  ".join(title_parts)
        file_name = out_png.rsplit('/', 1)[-1].replace('_', ' ').replace('.png', '')
        ax.set_title(f"{file_name}\n{metrics_str}", 
                    fontsize=12, fontweight='bold', pad=15)
        
        # Grid styling
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Minor ticks
        ax.minorticks_on()
        ax.tick_params(which="major", length=6, width=1.0)
        ax.tick_params(which="minor", length=3, width=0.6)
        
        # Enhanced legend with custom styling
        leg = ax.legend(loc="upper right", frameon=True, ncol=3, 
                       handlelength=2.0, columnspacing=1.5,
                       borderaxespad=0.8, fancybox=True,
                       shadow=False, framealpha=0.95)
        leg.get_frame().set_linewidth(0.5)
        leg.get_frame().set_edgecolor('#cccccc')

        # ---------------------------
        # Bottom panel: Enhanced Bragg ticks
        # ---------------------------
        nph = len(phase_order)
        if nph == 0:
            ax_ticks.set_ylim(-0.5, 0.5)
            ax_ticks.set_yticks([])
            ax_ticks.text(0.5, 0, "No phases identified", 
                         transform=ax_ticks.transAxes,
                         ha='center', va='center', style='italic',
                         color='#666666')
        else:
            ax_ticks.set_ylim(-0.5, nph - 0.5)
            ax_ticks.set_yticks(range(nph))
            
            # Color scheme - professional palette
            colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 
                     '#ff7f00', '#ffff33', '#a65628', '#f781bf']
            if nph > 8:
                import matplotlib.cm as cm
                cmap = cm.get_cmap('tab20')
                colors = [cmap(i/20) for i in range(nph)]
            
            # Create phase labels with formatting
            ylabels = []
            for i, nm in enumerate(phase_order):
                wt_val = wt.get(nm, 0.0)
                wt_str = f" ({wt_val:.1f}%)" if wt_val > 0 else ""
                
                # Prefer provided custom labels
                if phase_labels and nm in phase_labels:
                    label = f"{phase_labels[nm]}{wt_str}"
                else:
                    label = f"{nm}{wt_str}"
                ylabels.append(label)
            
            ax_ticks.set_yticklabels(ylabels)
            
            # Style y-tick labels
            for i, t in enumerate(ax_ticks.get_yticklabels()):
                t.set_va("center")
                t.set_ha("right")
                t.set_fontsize(9)
                if i < len(colors):
                    t.set_color(colors[i])
                # Bold the main phase (first one)
                if i == 0 and wt.get(phase_order[0], 0) > 30:
                    t.set_weight('bold')
            
            # Draw ticks with enhanced styling
            tick_height = 0.35  # Taller ticks
            for row, nm in enumerate(phase_order):
                xt = ticks_by_phase.get(nm)
                if xt is None or xt.size == 0:
                    # Draw a placeholder line if no ticks
                    ax_ticks.axhline(y=row, color='#cccccc', 
                                   linewidth=0.5, alpha=0.3)
                    continue
                
                color = colors[row % len(colors)]
                
                # Draw ticks with varying intensity based on density
                tick_density = len(xt) / (x.max() - x.min())
                alpha = min(0.9, 0.4 + tick_density * 0.1)
                
                # Main ticks
                ax_ticks.vlines(xt, row - tick_height, row + tick_height, 
                              lw=1.2, colors=[color], alpha=alpha, zorder=2)
                
                # Add subtle background for each phase row
                ax_ticks.axhspan(row - 0.45, row + 0.45, 
                               facecolor=color, alpha=0.05, zorder=0)
        
        # Bottom axis styling
        ax_ticks.set_xlabel(xlabel, fontweight='medium', fontsize=11)
        ax_ticks.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.5)
        ax_ticks.set_axisbelow(True)
        
        # Enhanced tick styling
        ax_ticks.minorticks_on()
        ax_ticks.tick_params(which="major", length=5, width=0.8)
        ax_ticks.tick_params(which="minor", length=2.5, width=0.6)
        
        # Spine styling
        for spine in ["left", "right", "top"]:
            ax_ticks.spines[spine].set_visible(False)
        ax_ticks.spines["bottom"].set_linewidth(1.0)
        ax_ticks.spines["bottom"].set_color("#333333")
        
        # Remove top spines from main plot
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        # Set x-limits with small margin for aesthetics
        x_lo, x_hi = (float(np.min(x)), float(np.max(x)))
        margin = (x_hi - x_lo) * 0.005  # 0.5% margin
        ax.set_xlim(min(x_lo, x_hi) - margin, max(x_lo, x_hi) + margin)
        
        # Add subtle annotation about data quality
        if rwp is not None:
            quality = "Excellent" if rwp < 5 else "Good" if rwp < 10 else "Acceptable" if rwp < 15 else "Poor"
            ax.text(0.02, 0.98, f"Fit Quality: {quality}", 
                   transform=ax.transAxes, fontsize=8,
                   va='top', ha='left', style='italic',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='white', edgecolor='#cccccc',
                            alpha=0.8))
        
        # Add data range info
        x_range_text = f"Range: {x_lo:.1f}–{x_hi:.1f}"
        ax_ticks.text(0.99, 0.95, x_range_text, 
                     transform=ax_ticks.transAxes,
                     ha='right', va='top', fontsize=8,
                     color='#666666', style='italic')
        
        # Ensure proper spacing for long labels
        try:
            max_label_len = max((len(lbl) for lbl in ylabels), default=0)
            if max_label_len > 25:
                fig.subplots_adjust(left=0.20)
            elif max_label_len > 20:
                fig.subplots_adjust(left=0.16)
        except Exception:
            pass

        # Save with high quality settings
        from pathlib import Path as _P
        _P(out_png).parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure with optimal settings
        fig.savefig(out_png, 
                   dpi=300, 
                   bbox_inches="tight", 
                   facecolor="white",
                   edgecolor='none',
                   pad_inches=0.15,
                   format='png',
                   transparent=False)

        try:
            from plot_payload import save_plot_payload

            phase_ticks_json = {
                str(pname): np.asarray(xt, dtype=float).tolist()
                for pname, xt in (ticks_by_phase or {}).items()
            }

            save_plot_payload(
                out_png,
                payload={
                    "plot_kind": "gsas_fit_with_ticks_v1",
                    "title": file_name,
                    "x_label": xlabel,
                    "instrument_type": inst,
                    "rwp": float(rwp) if rwp is not None else None,
                    "phase_order": [str(p) for p in phase_order],
                    "phase_labels": {str(k): str(v) for k, v in (phase_labels or {}).items()},
                    "phase_weights": {str(k): float(v) for k, v in wt.items()},
                    "phase_ticks": phase_ticks_json,
                },
                arrays={
                    "x": x,
                    "yobs": yobs,
                    "ycalc": ycalc,
                    "resid": resid,
                    "resid_scaled": (base + resid * scale),
                },
            )
        except Exception as payload_err:
            logger.warning(f"[plot] Could not save plot payload for {out_png}: {payload_err}")
        
        plt.close(fig)
        logger.info(f"[plot] Publication-ready plot saved: {out_png}")
# === END REPLACE =============================================================

# Test function
def test_gsas_refinement():
    """Test the refinement workflow with mock data."""
    logger.info("GSAS Main Phase Refinement Engine ready for integration.")
    logger.info("Key capabilities: - Staged refinement (Scale -> Background -> Cell); - Native GSAS data extraction; - Robust error handling and recovery; - Reflection position analysis")


if __name__ == "__main__":
    test_gsas_refinement()
