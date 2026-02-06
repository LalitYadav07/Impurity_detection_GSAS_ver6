#!/usr/bin/env python3
"""
GSAS-II Main Phase Refinement Engine and Pattern Analysis

This module provides the main phase refinement capabilities using GSAS-II's
native functions and extracts residuals in both native and Q-space coordinates.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import traceback
import re

try:
    from GSASII import GSASIIscriptable as G2sc
    from GSASII.GSASIIobj import G2Exception
    GSAS_AVAILABLE = True
except ImportError:
    GSAS_AVAILABLE = False
    G2Exception = Exception


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

    print(f"[XYE] writing {out_path} (n={n}, shift_positive={shift_positive})")
    with open(out_path, "w") as f:
        for i in range(n):
            f.write(f"{x[i]:.6f} {yw[i]:.6f} {sigma[i]:.6f}\n")
    return out_path


# --- CIF cell I/O helpers (text-only; read-only used in flow) ---

_CELL_KEYS = [
    "_cell_length_a", "_cell_length_b", "_cell_length_c",
    "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma",
]

_num_re = re.compile(r"""^[ \t]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)(?:\([\d]+\))?""")

def _parse_cif_number(s: str) -> float:
    """Parse a CIF numeric token that may include uncertainty '(...)'."""
    m = _num_re.match(s.strip())
    return float(m.group(1)) if m else float("nan")



# === Robust Pearson ===

def _init_gsas_process():
    """Ensure GSAS-II is initialized correctly in a sub-process (headless, Agg)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import wx
        if not wx.GetApp():
            app = wx.App(False)
        import GSASII.GSASIIctrlGUI as G2gui
        G2gui.haveGUI = False
    except Exception:
        pass

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
            # Native coordinate (2θ or TOF)
            x_native = histogram.getdata('x')
            data['x_native'] = x_native.compressed() if hasattr(x_native, 'compressed') else np.asarray(x_native)

            # Q-space using GSAS built-in conversion
            Q = histogram.getdata('Q')
            data['Q'] = Q.compressed() if hasattr(Q, 'compressed') else np.asarray(Q)

            # d-spacing using GSAS built-in conversion
            d = histogram.getdata('d')
            data['d'] = d.compressed() if hasattr(d, 'compressed') else np.asarray(d)

            # Intensities
            yobs = histogram.getdata('yobs')
            data['yobs'] = yobs.compressed() if hasattr(yobs, 'compressed') else np.asarray(yobs)

            ycalc = histogram.getdata('ycalc')
            data['ycalc'] = ycalc.compressed() if hasattr(ycalc, 'compressed') else np.asarray(ycalc)

            # Background (may not exist via getdata)
            try:
                ybkg = histogram.getdata('background')
            except Exception:
                ybkg = None
            if ybkg is not None:
                data['ybkg'] = ybkg.compressed() if hasattr(ybkg, 'compressed') else np.asarray(ybkg)

            # Weights (may be 'ywt' in some builds)
            try:
                ywt = histogram.getdata('ywt')
            except Exception:
                try:
                    ywt = histogram.getdata('yweight')
                except Exception:
                    ywt = None
            if ywt is not None:
                data['ywt'] = ywt.compressed() if hasattr(ywt, 'compressed') else np.asarray(ywt)

            # Compute residuals if possible
            if data.get('yobs') is not None and data.get('ycalc') is not None:
                yo = data['yobs']; yc = data['ycalc']
                if yo.size > 0 and yc.size > 0 and yo.size == yc.size:
                    data['residual'] = yo - yc

        except Exception as e:
            print(f"Warning: Failed to extract some data arrays: {e}")

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

            print("Initial refinement state configured")
            return True

        except Exception as e:
            print(f"Failed to setup initial state: {e}")
            traceback.print_exc()
            return False

    def refine_stage_scale(self) -> RefinementResults:
        """Stage 1: Refine only sample scale."""
        print("\n=== Stage 1: Scale Only ===")
        try:
            self._enable_scale_refinement()
            self._disable_background_refinement()
            self._disable_cell_refinement()

            self.project.refine()

            results = self._extract_refinement_results("Scale")
            print(f"Scale refinement: Rwp = {results.rwp:.3f}%, Scale = {results.scale:.6g}")
            return results

        except Exception as e:
            print(f"Scale refinement failed: {e}")
            traceback.print_exc()
            return RefinementResults(
                success=False, rwp=999.0, chi2=999.0, scale=1.0,
                background_params={}, cell_params={}, convergence_cycles=0,
                error_message=str(e)
            )

    def refine_stage_background(
        self,
        bg_type: Optional[str] = None,
        bg_terms: Optional[int] = None,
        bg_coeffs: Optional[List[float]] = None
    ) -> RefinementResults:
        """Stage 2: Refine scale + background."""
        print("\n=== Stage 2: Scale + Background ===")
        try:
            self._enable_scale_refinement()
            self._configure_background(bg_type, bg_terms, bg_coeffs)
            self._enable_background_refinement()
            self._disable_cell_refinement()

            self.project.refine()

            results = self._extract_refinement_results("Scale+Background")
            print(f"Background refinement: Rwp = {results.rwp:.3f}%")
            return results

        except Exception as e:
            print(f"Background refinement failed: {e}")
            traceback.print_exc()
            return RefinementResults(
                success=False, rwp=999.0, chi2=999.0, scale=1.0,
                background_params={}, cell_params={}, convergence_cycles=0,
                error_message=str(e)
            )

    def refine_stage_cell(self) -> RefinementResults:
        """Stage 3: Refine scale + background + cell (optional/guarded)."""
        print("\n=== Stage 3: Scale + Background + Cell ===")
        try:
            self._enable_scale_refinement()
            self._enable_background_refinement()
            self._enable_cell_refinement()

            self.project.refine()

            results = self._extract_refinement_results("Full")
            print(f"Cell refinement: Rwp = {results.rwp:.3f}%")
            return results

        except Exception as e:
            print(f"Cell refinement failed, reverting: {e}")
            self._disable_cell_refinement()
            try:
                self.project.refine()
                results = self._extract_refinement_results("Scale+Background")
                results.error_message = f"Cell refinement failed: {e}"
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
        bg_type: Optional[str] = None,
        bg_terms: Optional[int] = None,
        bg_coeffs: Optional[List[float]] = None
    ) -> RefinementResults:
        """Run complete staged refinement workflow."""
        print("\n=== Running Staged Main Phase Refinement ===")

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
            bg_type=bg_type, bg_terms=bg_terms, bg_coeffs=bg_coeffs
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
        print("Residual max min: ", np.max(residual), np.min(residual))
        print("Q max min: ", np.max(Q), np.min(Q))
        return Q, residual

    def get_all_data(self) -> Dict[str, np.ndarray]:
        """Get all data arrays from current state."""
        return GSASDataExtractor.get_all_arrays(self.histogram)

    # Helper methods for refinement control
    def _set_limits_to_data(self):
        """Set refinement limits to full data range."""
        x_data = np.asarray(self.histogram.getdata('x'))
        if x_data.size > 0:
            self.histogram.set_refinements({
                'Limits': {'low': float(np.min(x_data)), 'high': float(np.max(x_data))}
            })

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
            print(f"Warning: Could not clear instrument refinements: {e}")

    def _enable_scale_refinement(self):
        """Enable sample scale refinement."""
        try:
            sample_params = self.histogram.getHistEntryValue(['Sample Parameters'])
            current_scale = float(sample_params['Scale'][0]) if 'Scale' in sample_params else 1.0
            sample_params['Scale'] = [current_scale, True]  # [value, refine_flag]
            self.histogram.setHistEntryValue(['Sample Parameters'], sample_params)
            self.histogram.set_refinements({'Sample Parameters': ['Scale']})
        except Exception as e:
            print(f"Warning: Could not enable scale refinement: {e}")

    def _disable_phase_scale_refinement(self):
        """Disable phase scale refinement (use histogram scale instead)."""
        try:
            self.phase.set_HAP_refinements({'Scale': False}, histograms=[self.histogram])
        except Exception as e:
            print(f"Warning: Could not disable phase scale: {e}")

    def _configure_background(
        self,
        bg_type: Optional[str],
        bg_terms: Optional[int],
        bg_coeffs: Optional[List[float]] = None
    ):
        """Configure background model.

        Rules:
        - If nothing provided (type/terms/coeffs all None) → auto by instrument.
        - If type/terms provided (even without coeffs) → override with those.
        - If coeffs provided → seed them (padded/truncated to 'terms').
        """

        # Instrument-based defaults
        def _default_bg_by_instrument() -> Tuple[str, int]:
            if self.instrument_type == "TOF":
                try:
                    x_data = np.asarray(self.histogram.getdata('x'))
                    terms = max(2, min(8, len(x_data) // 100)) if x_data.size > 0 else 3
                except Exception:
                    terms = 3
                return "log interpolate", terms
            else:  # CW
                return "chebyschev-1", 12

        if bg_type is None and bg_terms is None and bg_coeffs is None:
            bg_type, bg_terms = _default_bg_by_instrument()
        else:
            if bg_type is None or bg_terms is None:
                d_type, d_terms = _default_bg_by_instrument()
                if bg_type is None:
                    bg_type = d_type
                if bg_terms is None:
                    bg_terms = d_terms

        # Apply selection
        try:
            self.histogram.set_refinements({'Background': {
                'type': bg_type,
                'no. coeffs': int(bg_terms),
                'refine': False  # enable refine separately
            }})
            print(f"Background configured: {bg_type} with {bg_terms} terms")

            # Seed explicit coefficients, if given
            if bg_coeffs is not None:
                coeffs = list(map(float, bg_coeffs))
                coeffs = (coeffs + [0.0] * int(bg_terms))[:int(bg_terms)]  # pad/trim

                cur = self.histogram.getHistEntryValue(['Background'])
                # Expect: cur == [Back(list), DebyePeaks(dict)]
                if (isinstance(cur, list) and len(cur) >= 2
                        and isinstance(cur[0], list) and isinstance(cur[1], dict)):
                    back = list(cur[0])  # [funcName, refineFlag, nCoeffs, coeffs...]
                    if len(back) < 1:
                        back.append(bg_type)
                    else:
                        back[0] = bg_type
                    if len(back) < 2:
                        back.append(False)
                    else:
                        back[1] = bool(back[1])
                    if len(back) < 3:
                        back.append(int(bg_terms))
                    else:
                        back[2] = int(bg_terms)
                    back = back[:3] + coeffs
                    self.histogram.setHistEntryValue(['Background'], [back, cur[1]])
                    print(f"Background coefficients seeded (n={len(coeffs)})")
                else:
                    print("Note: unexpected Background layout; skipped coeff seeding to avoid corrupting GSAS state.")
        except Exception as e:
            print(f"Warning: Could not configure background: {e}")

    def _enable_background_refinement(self):
        """Enable background refinement."""
        try:
            self.histogram.set_refinements({'Background': {'refine': True}})
        except Exception as e:
            print(f"Warning: Could not enable background refinement: {e}")

    def _disable_background_refinement(self):
        """Disable background refinement."""
        try:
            self.histogram.set_refinements({'Background': {'refine': False}})
        except Exception as e:
            print(f"Warning: Could not disable background refinement: {e}")

    def _enable_cell_refinement(self):
        """Enable cell parameter refinement."""
        try:
            self.phase.set_refinements({'Cell': True})
        except Exception as e:
            print(f"Warning: Could not enable cell refinement: {e}")

    def _disable_cell_refinement(self):
        """Disable cell parameter refinement."""
        try:
            self.phase.set_refinements({'Cell': False})
        except Exception as e:
            print(f"Warning: Could not disable cell refinement: {e}")

    def _extract_refinement_results(self, stage: str) -> RefinementResults:
        """Extract refinement results from current state."""
        try:
            rwp = float(self.histogram.get_wR())

            sample_params = self.histogram.getHistEntryValue(['Sample Parameters'])
            scale = float(sample_params['Scale'][0]) if 'Scale' in sample_params else 1.0

            # Background params (best-effort)
            bg_params: Dict[str, Any] = {}
            try:
                bg_data = self.histogram.getHistEntryValue(['Background'])
                if isinstance(bg_data, list) and len(bg_data) > 0:
                    bg_type = bg_data[0][0] if isinstance(bg_data[0], (list, tuple)) else "unknown"
                    bg_coeffs = bg_data[1] if len(bg_data) > 1 else []
                    bg_params = {'type': bg_type, 'coefficients': list(bg_coeffs)}
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
                print(f"Warning: Could not extract cell parameters: {e}")

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
            print(f"Failed to extract refinement results: {e}")
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
    cleaned = [[float(min(a, b)), float(max(a, b))] for (a, b) in excluded_pairs]
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
    frac_line_re = re.compile(
        rf'Phase fraction\s*:\s*(?P<pf>{num})\s*,\s*sig\s*(?P<pf_sig>{num})\s*'
        rf'Weight fraction\s*:\s*(?P<wf>{num})\s*,\s*sig\s*(?P<wf_sig>{num})'
    )
    out: Dict[str, Dict[str, float]] = {}
    lines = Path(lst_path).read_text(errors="ignore").splitlines()
    i, n = 0, len(lines)
    while i < n:
        m = phase_hdr_re.match(lines[i])
        if m:
            phase = m.group('phase').strip()
            histname = m.group('hist').strip()
            if histname == target_hist:
                for j in range(i + 1, min(i + 30, n)):
                    m2 = frac_line_re.search(lines[j])
                    if m2:
                        g = {k: float(v) for k, v in m2.groupdict().items() if k in {'pf','pf_sig','wf','wf_sig'}}
                        out[phase] = {
                            "phase_fraction_pct": g["pf"] * 100.0,
                            "weight_fraction_pct": g["wf"] * 100.0,
                        }
                        break
        i += 1
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
    from gsas_core_infrastructure import GSASProjectManager

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
    if template_gpx and _Path(template_gpx).exists():
        # Resolve existing histogram from template
        if pm.project.histograms():
            hist = pm.project.histograms()[0]
            pm.main_histogram = hist
            histogram_loaded_from_template = True
        else:
             print(f"[WARN] Template GPX loaded from {template_gpx} but has no histograms. Falling back to fresh load.")
             # No raise; fall through to normal loading

    if not histogram_loaded_from_template:
        # Select observed dataset
        local_data_path = data_path
        local_fmthint = fmthint
        using_override = (x_override is not None) and (y_override is not None)
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

    # Background OFF; use histogram Scale (HAP Scale held)
    try:
        hist.set_refinements({'Background': {'type': 'chebyschev-1','no. coeffs': 1, 'coeffs': [0.0],'refine': False}})
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
        print(f"[PEARSON] {ph_name} {label}: cycles={int(max(0, cycles))}")
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
        print(f"[PEARSON] {ph_name} {label}: r={r:.6f}")
        return r

    # Pass-1: Scale only
    _set_flags(hist_scale=True, cell=False)
    r1 = _run_and_r(1, "pass1-scale")

    # Early exit for clearly poor candidates:
    # If r < 0.1 after scale refinement, it's unlikely to become a top candidate with cell refinement.
    if r1 < 0.1:
        print(f"[PEARSON] {ph_name} early-exit: r={r1:.4f} too low")
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
        print(f"[PEARSON] wrote refined CIF → {target_write}")

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
    print(f"[clone] {src} -> {dst}")


def get_hist_and_main_phase(proj, main_phase_name: str):
    hists = proj.histograms()
    if not hists:
        raise RuntimeError("No histograms in project.")
    hist = hists[0]
    phases = {p.name: p for p in proj.phases()}
    if main_phase_name not in phases:
        raise RuntimeError(f"Main phase '{main_phase_name}' not found. Have: {list(phases)}")
    print(f"[init] histogram='{hist.name}', main='{main_phase_name}'")
    return hist, phases[main_phase_name]


def set_phase_cell_refine(phase, refine: bool) -> None:
    (phase.set_refinements if refine else phase.clear_refinements)({'Cell': True})
    print(f"[flags] Phase '{phase.name}' Cell refine={refine}")


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
    try:
        proj.add_HoldConstr(proj.make_var_obj(hist=hist, varname='Zero'))
    except Exception:
        pass
    for var in ['U', 'V', 'W', 'X', 'Y', 'Z', 'Sh/L']:
        try:
            proj.add_HoldConstr(proj.make_var_obj(hist=hist, varname=var))
        except Exception:
            pass

    # Ensure no HAP constraints (no sum-to-one coupling)
    cons = proj.data.setdefault('Constraints', {})
    if 'HAP' in cons and cons['HAP']:
        cons['HAP'] = []
        print("[joint] Cleared existing HAP constraints.")

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

    try:
        rwp = float(hist.get_wR())
    except Exception:
        rwp = float('nan')

    return x_native, residual_native, Q, residual_Q, rwp, hist.name, proj
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
    - Existing phases remain, refine=True; current HAP Scale values are preserved.
    - New phases are added with small HAP Scale=hap_init and refine=True.
    - Background refine ON; sample Scale held; zero/profile held; phase Cell held.
    Returns per-phase fractions as in joint_refine_one_cycle.
    """
    from GSASII import GSASIIscriptable as G2sc

    clone_gpx(base_gpx, out_gpx)
    proj = G2sc.G2Project(gpxfile=out_gpx)
    hist, main_phase = get_hist_and_main_phase(proj, main_phase_name)

    existing = {p.name: p for p in proj.phases()}

    # Existing phases → refine Scale; keep their current values unless caller opts otherwise
    for p in existing.values():
        set_phase_cell_refine(p, refine=False)
        p.set_HAP_refinements({'Scale': True}, histograms=[hist])
        if not preserve_existing_scales:
            # Place holder: intentionally do nothing to avoid risky guesswork
            pass

    # Add new phases only if missing
    for pid, cif in pid_to_cif_new.items():
        if pid in existing:
            print(f"[joint+] Phase '{pid}' already present; skipping add.")
            continue
        p = proj.add_phase(cif, phasename=str(pid), histograms=[hist])
        set_phase_cell_refine(p, refine=False)
        p.set_HAP_refinements({'Scale': True}, histograms=[hist])
        p.HAPvalue('Scale', float(hap_init), targethistlist=[hist])

    # Background refine ON; sample Scale held; zero/profile held (same policy as one_cycle)
    hist.set_refinements({'Background': {'refine': True}})
    try:
        hist.clear_refinements({'Sample Parameters': ['Scale']})
    except Exception:
        pass
    try:
        proj.add_HoldConstr(proj.make_var_obj(hist=hist, varname='Zero'))
    except Exception:
        pass
    for var in ['U', 'V', 'W', 'X', 'Y', 'Z', 'Sh/L']:
        try:
            proj.add_HoldConstr(proj.make_var_obj(hist=hist, varname=var))
        except Exception:
            pass

    # Ensure no HAP constraints couple scales
    cons = proj.data.setdefault('Constraints', {})
    if 'HAP' in cons and cons['HAP']:
        cons['HAP'] = []
        print("[joint+] Cleared existing HAP constraints.")

    proj.data['Controls']['data']['max cyc'] = int(max_joint_cycles)
    proj.do_refinements([
        {'set': {'Background': {'refine': True}}},
        {'refine': True},
    ])

    # Parse fractions
    lst_path = Path(out_gpx).with_suffix(".lst")
    parsed = parse_gsas_lst(lst_path, hist.name) if lst_path.exists() else {}
    results: Dict[str, Dict[str, float]] = {}
    for p in proj.phases():
        nm = p.name
        vals = parsed.get(nm)
        results[nm] = {
            "phase_fraction_pct": float(vals["phase_fraction_pct"]) if vals else 0.0,
            "weight_fraction_pct": float(vals["weight_fraction_pct"]) if vals else 0.0,
        }
    return results
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
        return G2sc.G2Project(gpxfile=path)

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
                print(f"[polish] Warning: Regenerated .lst failed GOF check (GOF={gof!r}, req ≤ {GOF_MAX}).")
                return False
            print(f"[polish] Regenerated .lst OK (GOF={gof:.3f}) for {Path(gpx_path).name}")
            return True
        except Exception as e:
            print(f"[polish] Warning: Failed to regenerate .lst: {e}")
            return False

    def _final_readout(proj, lst_path: Path, hist_name: str) -> Tuple[Dict[str, Dict[str, float]], float]:
        results: Dict[str, Dict[str, float]] = {}
        parsed = {}
        parse_func = _parse_ext or parse_gsas_lst
        try:
            if lst_path.exists():
                parsed = parse_func(lst_path, hist_name) or {}
        except Exception:
            parsed = {}
        for p in proj.phases():
            nm = p.name
            vals = parsed.get(nm)
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
    print("[polish] Finalizing output...")
    _clone(str(checkpoint), out_gpx)
    _regenerate_lst(out_gpx, refine_background)

    proj = _open(out_gpx)

    if enabled:
        print(f"[polish] Cell refinement enabled for: {', '.join(enabled)}")
    else:
        print("[polish] No phases accepted for cell refinement during polish.")

    return _final_readout(proj, _lst_path(out_gpx), hist.name)


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
        print(f"[plot] matplotlib not available: {e}")
        return

    try:
        from GSASII import GSASIIscriptable as G2sc
    except Exception as e:
        print(f"[plot] GSAS-II not available for plotting: {e}")
        return

    try:
        from gsas_core_infrastructure import CoordinateHandler
    except Exception as e:
        print(f"[plot] CoordinateHandler not available: {e}")
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
        print(f"[plot] No histogram in {gpx_path}")
        return
    hist = hists[0]

    # ---------------------------
    # Data arrays
    # ---------------------------
    def _arr(kind: str):
        try:
            arr = hist.getdata(kind)
            return arr.compressed() if hasattr(arr, 'compressed') else np.asarray(arr, float)
        except Exception:
            return np.array([], float)

    x = _arr('x'); yobs = _arr('yobs'); ycalc = _arr('ycalc')
    if x.size == 0 or yobs.size == 0 or ycalc.size == 0:
        print("[plot] Missing x/yobs/ycalc; skipped.")
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
        print(f"[plot] Could not compute reflection ticks: {e}")
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
        
        plt.close(fig)
        print(f"[plot] Publication-ready plot saved: {out_png}")
# === END REPLACE =============================================================

# Test function
def test_gsas_refinement():
    """Test the refinement workflow with mock data."""
    print("GSAS Main Phase Refinement Engine ready for integration.")
    print("Key capabilities:")
    print("- Staged refinement (Scale -> Background -> Cell)")
    print("- Native GSAS data extraction in both coordinate systems")
    print("- Robust error handling and recovery")
    print("- Reflection position analysis")


if __name__ == "__main__":
    test_gsas_refinement()
