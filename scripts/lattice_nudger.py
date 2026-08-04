"""
GSAS-II Lattice Refinement Module (Stage-4)

This module implements the "Lattice Nudging" and advanced refinement logic (Stage-4).
It is responsible for:
- Adaptive lattice parameter optimization using a hit-and-run polytope sampling approach.
- Exploring the neighborhood of likely lattice parameters to escape local minima.
- Tikhonov-regularized backsolving for lattice constraints.
- Optimized Q-space metrics and gradients.
"""
from __future__ import annotations

import os
import math
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from functools import lru_cache
import threading
import concurrent.futures
import re
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from pymatgen.core import Structure, Lattice
import logging

logger = logging.getLogger(__name__)


def _load_cif_structure_relaxed(cif_path: str | Path) -> Structure:
    """Load user/CIF-exported structures with the same tolerance used elsewhere.

    GSAS-II can read a number of CIFs that pymatgen's default
    ``Structure.from_file`` rejects, especially refined/exported CIFs with
    slightly over-occupied sites. Main-phase lattice nudging should not fail
    only because this helper used a stricter parser than the rest of RADAR-PD.
    """
    cif_path = Path(cif_path)
    parser_error: Exception | None = None
    try:
        from pymatgen.io.cif import CifParser

        structures = CifParser(str(cif_path), occupancy_tolerance=2.0).parse_structures(primitive=False)
        if structures:
            return structures[0]
        parser_error = RuntimeError("CifParser returned no structures")
    except Exception as exc:
        parser_error = exc

    try:
        return Structure.from_file(str(cif_path))
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse CIF for lattice nudging: {cif_path}. "
            f"Relaxed parser error: {parser_error}; default parser error: {exc}"
        ) from exc
try:
    from xray_doublet import apply_doublet_to_peaks, describe_doublet, is_active_for
except Exception:
    apply_doublet_to_peaks = None
    describe_doublet = None
    is_active_for = None

try:
    import matplotlib.pyplot as plt  # optional; plots are skipped if unavailable
except Exception:
    plt = None

# Numba import for JIT compilation
try:
    from numba import jit, njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorators if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# --- Fast HKL Q via reciprocal metric (Numba JIT, accuracy-identical) ---
try:
    from numba import njit as _njit_fast
except Exception:
    def _njit_fast(*a, **k):
        def d(f): return f
        return d

import numpy as _np

# ======================================================================================
# Config-like defaults
# ======================================================================================
DEFAULT_WAVELENGTH = 1.50                    # Å; covers Q ~0.5–8.2 Å^-1 with 2θ≈5–160°
DEFAULT_TWOTHETA_RANGE: Tuple[float, float] = (5.0, 160.0)
DEFAULT_SCORE_Q_MAX = float(os.environ.get("STAGE4_SCORE_Q_MAX", "8.0"))
DEFAULT_LATTICE_TIEBREAK_SCORE_TOL = float(
    os.environ.get("STAGE4_LATTICE_TIEBREAK_SCORE_TOL", "5e-4")
)
DEFAULT_STAGE4_SEED = int(os.environ.get("STAGE4_SEED", "0"))
DEFAULT_FWHM_Q = float(os.environ.get("STAGE4_FWHM_Q", 0.03))  # Å^-1 peak width in Q

# Debug / Strategy
_DEBUG = bool(int(os.environ.get("STAGE4_DEBUG", "0")))
_QWIN_PCT = float(os.environ.get("STAGE4_QWIN_PCT", "20.0"))     # ±percent window per HKL Q
_REG_L2   = float(os.environ.get("STAGE4_REG", "1e-4"))          # Tikhonov L2 for backsolve
_MAX_Q_BAD = 1e5

# Scoring knobs (kept)
_SCORE_POSRATIO_GLOBAL = float(os.environ.get("STAGE4_SCORE_POSRATIO_GLOBAL", "0.02"))

# ======================================================================================
# Optimized numerical functions using Numba JIT
# ======================================================================================

@njit(cache=True)
def _minmax_numba(x):
    """Numba-optimized min-max normalization"""
    if x.size == 0:
        return x
    lo = np.min(x)
    hi = np.max(x)
    if hi > lo:
        return (x - lo) / (hi - lo)
    else:
        return np.zeros_like(x)

@njit(cache=True)
def _gaussian_kernel_bins_numba(dq, fwhm_Q):
    """Numba-optimized Gaussian kernel computation"""
    sigma = fwhm_Q / 2.354820045
    sigma_bins = max(sigma / max(dq, 1e-12), 1e-3)
    half = int(np.ceil(6.0 * sigma_bins))
    idx = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (idx / sigma_bins) ** 2)
    s = kernel.sum()
    return kernel / (s if s > 0 else 1.0)

@njit(cache=True)
def _add_at_numba(arr, indices, values):
    """Numba-optimized equivalent of np.add.at"""
    for i in range(len(indices)):
        if 0 <= indices[i] < len(arr):
            arr[indices[i]] += values[i]
    return arr

@njit(cache=True)
def _coslog_score_core_numba(Is, Rvec, alpha, eps):
    """Numba-optimized core scoring computation"""
    smax = np.max(Is) if Is.size > 0 else 0.0
    if not np.isfinite(smax) or smax <= 0:
        return 0.0

    Isn = Is / smax
    Svec = np.log1p(alpha * Isn + eps)
    Rv = np.log1p(alpha * Rvec + eps)

    svec_norm = np.linalg.norm(Svec)
    rv_norm = np.linalg.norm(Rv)
    denom = svec_norm * rv_norm

    if not np.isfinite(denom) or denom <= 1e-20:
        return 0.0

    cos = np.dot(Svec, Rv) / denom
    return max(0.0, min(1.0, cos))

@njit(cache=True)
def _uniform_grid_check_numba(q, rtol=1e-3, atol=1e-6):
    """Numba-optimized uniform grid check"""
    if len(q) < 3:
        return True
    dq = q[1:] - q[:-1]
    return np.all(np.abs(dq - dq[0]) <= atol + rtol * np.abs(dq[0]))

# ======================================================================================
# Crystal constraints from space group / lattice (unchanged)
# ======================================================================================

def _crystal_system_from_sgnum(sgnum: int) -> str:
    if 1 <= sgnum <= 2: return "triclinic"
    if 3 <= sgnum <= 15: return "monoclinic"
    if 16 <= sgnum <= 74: return "orthorhombic"
    if 75 <= sgnum <= 142: return "tetragonal"
    if 143 <= sgnum <= 167: return "trigonal"
    if 168 <= sgnum <= 194: return "hexagonal"
    if 195 <= sgnum <= 230: return "cubic"
    return "unknown"

@dataclass
class LatticeConstraints:
    free_a: bool; free_b: bool; free_c: bool
    free_alpha: bool; free_beta: bool; free_gamma: bool
    tie_ab: bool; tie_ac: bool; tie_bc: bool
    alpha0: Optional[float]; beta0: Optional[float]; gamma0: Optional[float]
    tie_alpha_beta: bool = False
    tie_alpha_gamma: bool = False
    tie_beta_gamma: bool = False

def infer_constraints(structure: Structure, sgnum: Optional[int]) -> LatticeConstraints:
    L = structure.lattice
    cons = LatticeConstraints(True, True, True, True, True, True,
                              False, False, False, None, None, None)
    if sgnum is None:
        return _maybe_specialize_trigonal_hex_by_angles(L, cons)

    cs = _crystal_system_from_sgnum(int(sgnum))
    if cs == "cubic":
        return LatticeConstraints(True, False, False, False, False, False,
                                  True, True, True, 90.0, 90.0, 90.0)
    if cs == "tetragonal":
        return LatticeConstraints(True, False, True, False, False, False,
                                  True, False, False, 90.0, 90.0, 90.0)
    if cs == "hexagonal":
        return LatticeConstraints(True, False, True, False, False, False,
                                  True, False, False, 90.0, 90.0, 120.0)
    if cs == "trigonal":
        return _trigonal_constraints_for_lattice(L)
    if cs == "orthorhombic":
        return LatticeConstraints(True, True, True, False, False, False,
                                  False, False, False, 90.0, 90.0, 90.0)
    if cs == "monoclinic":
        return _monoclinic_constraints_for_lattice(L)
    return cons

def _maybe_specialize_trigonal_hex_by_angles(L: Lattice, cons: LatticeConstraints) -> LatticeConstraints:
    a,b,c = L.a, L.b, L.c
    alpha, beta, gamma = L.alpha, L.beta, L.gamma
    if abs(gamma-120.0) < 5 and abs(alpha-90)<5 and abs(beta-90)<5 and abs(a-b)/max(a,b) < 0.01:
        return LatticeConstraints(True, False, True, False, False, False,
                                  True, False, False, 90.0, 90.0, 120.0)
    if abs(alpha-beta) < 1 and abs(beta-gamma) < 1 and abs(a-b)/max(a,b) < 0.01 and abs(b-c)/max(b,c) < 0.01:
        return LatticeConstraints(True, False, False, True, False, False,
                                  True, True, True, None, None, None,
                                  True, True, False)
    return cons

def _trigonal_constraints_for_lattice(L: Lattice) -> LatticeConstraints:
    """Trigonal cells must stay in either hexagonal or rhombohedral setting."""
    a, b, c = L.a, L.b, L.c
    alpha, beta, gamma = L.alpha, L.beta, L.gamma

    hex_score = (
        abs(a - b) / max(a, b, 1e-12) * 100.0
        + abs(alpha - 90.0)
        + abs(beta - 90.0)
        + abs(gamma - 120.0)
    )
    rhom_score = (
        abs(a - b) / max(a, b, 1e-12) * 100.0
        + abs(a - c) / max(a, c, 1e-12) * 100.0
        + abs(alpha - beta)
        + abs(alpha - gamma)
    )

    if hex_score <= rhom_score:
        return LatticeConstraints(True, False, True, False, False, False,
                                  True, False, False, 90.0, 90.0, 120.0)

    return LatticeConstraints(True, False, False, True, False, False,
                              True, True, True, None, None, None,
                              True, True, False)

def _monoclinic_constraints_for_lattice(L: Lattice) -> LatticeConstraints:
    """Keep the observed monoclinic unique-axis setting instead of assuming beta."""
    deviations = {
        "alpha": abs(L.alpha - 90.0),
        "beta": abs(L.beta - 90.0),
        "gamma": abs(L.gamma - 90.0),
    }
    unique = max(deviations, key=deviations.get)
    if deviations[unique] < 0.5:
        unique = "beta"

    if unique == "alpha":
        return LatticeConstraints(True, True, True, True, False, False,
                                  False, False, False, None, 90.0, 90.0)
    if unique == "gamma":
        return LatticeConstraints(True, True, True, False, False, True,
                                  False, False, False, 90.0, 90.0, None)
    return LatticeConstraints(True, True, True, False, True, False,
                              False, False, False, 90.0, None, 90.0)

def _rebuild_with_lattice(base_struct: Structure, new_latt: Lattice) -> Structure:
    # Works for ordered & disordered sites
    per_site_species = [site.species for site in base_struct]  # list of Composition
    return Structure(
        new_latt,
        per_site_species,
        base_struct.frac_coords,
        site_properties=base_struct.site_properties,
        charge=getattr(base_struct, "charge", None),
        coords_are_cartesian=False,
    )

def _lattice_deviation_metric(
    base_lat: Lattice,
    cand_lat: Lattice,
    len_tol_pct: float,
    ang_tol_deg: float,
) -> float:
    """Normalized distance from the starting lattice for near-tie selection."""
    len_scale = max(float(len_tol_pct) / 100.0, 1e-12)
    ang_scale = max(float(ang_tol_deg), 1e-12)
    terms = [
        (cand_lat.a - base_lat.a) / max(abs(base_lat.a) * len_scale, 1e-12),
        (cand_lat.b - base_lat.b) / max(abs(base_lat.b) * len_scale, 1e-12),
        (cand_lat.c - base_lat.c) / max(abs(base_lat.c) * len_scale, 1e-12),
        (cand_lat.alpha - base_lat.alpha) / ang_scale,
        (cand_lat.beta - base_lat.beta) / ang_scale,
        (cand_lat.gamma - base_lat.gamma) / ang_scale,
    ]
    return float(math.sqrt(sum(t * t for t in terms) / len(terms)))

def _choose_best_lattice_index(
    scores: np.ndarray,
    deviations: np.ndarray,
    score_tol: float,
) -> Tuple[int, int]:
    """Choose highest score, breaking near-ties by closest lattice."""
    scores = np.asarray(scores, dtype=float)
    deviations = np.asarray(deviations, dtype=float)
    if scores.size == 0:
        return 0, 0
    best_sc = float(np.max(scores))
    tie_tol = max(0.0, float(score_tol))
    if tie_tol > 0:
        near_best = np.where(scores >= best_sc - tie_tol)[0]
    else:
        near_best = np.where(scores == best_sc)[0]
    if near_best.size:
        return int(near_best[np.argmin(deviations[near_best])]), int(near_best.size)
    return int(np.argmax(scores)), 1

def _lattice_constraint_violations(
    cons: LatticeConstraints,
    lat: Lattice,
    *,
    length_rtol: float = 1e-7,
    angle_atol: float = 1e-5,
) -> List[str]:
    """Return constraint labels violated by a lattice."""
    violations: List[str] = []

    def _rel_bad(x: float, y: float) -> bool:
        return abs(x - y) > length_rtol * max(abs(x), abs(y), 1.0)

    if cons.tie_ab and _rel_bad(lat.a, lat.b):
        violations.append("a=b")
    if cons.tie_ac and _rel_bad(lat.a, lat.c):
        violations.append("a=c")
    if cons.tie_bc and _rel_bad(lat.b, lat.c):
        violations.append("b=c")
    if cons.tie_alpha_beta and abs(lat.alpha - lat.beta) > angle_atol:
        violations.append("alpha=beta")
    if cons.tie_alpha_gamma and abs(lat.alpha - lat.gamma) > angle_atol:
        violations.append("alpha=gamma")
    if cons.tie_beta_gamma and abs(lat.beta - lat.gamma) > angle_atol:
        violations.append("beta=gamma")
    if not cons.free_alpha and cons.alpha0 is not None and abs(lat.alpha - cons.alpha0) > angle_atol:
        violations.append(f"alpha={cons.alpha0:g}")
    if not cons.free_beta and cons.beta0 is not None and abs(lat.beta - cons.beta0) > angle_atol:
        violations.append(f"beta={cons.beta0:g}")
    if not cons.free_gamma and cons.gamma0 is not None and abs(lat.gamma - cons.gamma0) > angle_atol:
        violations.append(f"gamma={cons.gamma0:g}")
    return violations

# ======================================================================================
# HKL signature in Q (cached for performance) + FAST PATH
# ======================================================================================

_HKLS7 = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,1,1)]
_HKLS7_A = _np.array(_HKLS7, dtype=_np.int64)

@_njit_fast(cache=True)
def _reciprocal_metric_fast(a,b,c,al,be,ga):
    ra = _np.deg2rad(al); rb = _np.deg2rad(be); rg = _np.deg2rad(ga)
    cal = _np.cos(ra); cbe = _np.cos(rb); cga = _np.cos(rg)
    G00 = a*a; G11 = b*b; G22 = c*c
    G01 = a*b*cga; G02 = a*c*cbe; G12 = b*c*cal
    G = _np.empty((3,3), dtype=_np.float64)
    G[0,0]=G00; G[1,1]=G11; G[2,2]=G22
    G[0,1]=G01; G[1,0]=G01
    G[0,2]=G02; G[2,0]=G02
    G[1,2]=G12; G[2,1]=G12
    return _np.linalg.inv(G)

@_njit_fast(cache=True)
def _q_from_Gstar(Gs, h, k, l):
    v0=Gs[0,0]; v1=Gs[0,1]; v2=Gs[0,2]
    v3=Gs[1,1]; v4=Gs[1,2]; v5=Gs[2,2]
    dinv2 = h*h*v0 + 2*h*k*v1 + 2*h*l*v2 + k*k*v3 + 2*k*l*v4 + l*l*v5
    if dinv2 <= 0.0:
        return 1e6
    d = 1.0/_np.sqrt(dinv2)
    return 2.0*_np.pi/d

@_njit_fast(cache=True)
def q_signature_fast(a,b,c,al,be,ga):
    Gs = _reciprocal_metric_fast(a,b,c,al,be,ga)
    out = _np.empty(7, dtype=_np.float64)
    for i in range(7):
        h = _HKLS7_A[i,0]; k = _HKLS7_A[i,1]; l = _HKLS7_A[i,2]
        out[i] = _q_from_Gstar(Gs, h,k,l)
    return out

@lru_cache(maxsize=1024)
def _cached_d_hkl(a, b, c, alpha, beta, gamma, hkl):
    """Cache d-spacing calculations for repeated lattice parameters"""
    try:
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        return lattice.d_hkl(hkl)
    except Exception:
        return None

def hkl_Q_signature(lattice: Lattice, hkls: List[Tuple[int,int,int]] = _HKLS7) -> np.ndarray:
    # Fast path for the default 7-reflection signature
    if hkls is _HKLS7:
        return q_signature_fast(lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma)
    # Fallback for non-default hkls (rare path)
    Q = []
    a = round(lattice.a, 6); b = round(lattice.b, 6); c = round(lattice.c, 6)
    alpha = round(lattice.alpha, 3); beta = round(lattice.beta, 3); gamma = round(lattice.gamma, 3)
    for hkl in hkls:
        d = _cached_d_hkl(a, b, c, alpha, beta, gamma, hkl)
        Q.append(2.0*math.pi/max(d, 1e-12) if d is not None else np.nan)
    q = np.array(Q, dtype=float)
    q[~np.isfinite(q)] = 1e6
    return q

# ======================================================================================
# Optimized Utilities
# ======================================================================================

def _minmax(x):
    """Use numba version if available"""
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    if HAS_NUMBA:
        return _minmax_numba(x)
    else:
        lo, hi = float(np.min(x)), float(np.max(x))
        return (x - lo)/(hi-lo) if hi > lo else np.zeros_like(x)

def _is_uniform_grid(q: np.ndarray, rtol=1e-3, atol=1e-6) -> bool:
    if HAS_NUMBA:
        return _uniform_grid_check_numba(q, rtol, atol)
    else:
        if len(q) < 3: return True
        dq = np.diff(q)
        return np.allclose(dq, dq[0], rtol=rtol, atol=atol)

def _gaussian_kernel_bins(dq: float, fwhm_Q: float) -> np.ndarray:
    if HAS_NUMBA:
        return _gaussian_kernel_bins_numba(dq, fwhm_Q)
    else:
        sigma = float(fwhm_Q) / 2.354820045
        sigma_bins = max(sigma / max(dq, 1e-12), 1e-3)
        half = int(np.ceil(6.0 * sigma_bins))
        idx = np.arange(-half, half+1, dtype=float)
        kernel = np.exp(-0.5 * (idx/sigma_bins)**2)
        s = kernel.sum()
        return kernel / (s if s > 0 else 1.0)

# Thread-local storage for FFT plans to avoid recomputation
_thread_local = threading.local()

def _get_or_create_fft_plan(q_grid_hash, q_grid, fwhm_Q):
    """Get cached FFT plan or create new one"""
    if not hasattr(_thread_local, 'fft_plans'):
        _thread_local.fft_plans = {}

    if q_grid_hash not in _thread_local.fft_plans:
        _thread_local.fft_plans[q_grid_hash] = _make_fft_plan(q_grid, fwhm_Q)

    return _thread_local.fft_plans[q_grid_hash]

def gaussian_render_on_grid(Q_peaks: np.ndarray, I_peaks: np.ndarray,
                            Q_grid: np.ndarray, fwhm_Q: float) -> np.ndarray:
    Q_peaks = np.asarray(Q_peaks, float)
    I_peaks = np.asarray(I_peaks, float)
    Q_grid  = np.asarray(Q_grid, float)

    if Q_peaks.size == 0 or Q_grid.size == 0:
        return np.zeros_like(Q_grid, dtype=float)

    if _is_uniform_grid(Q_grid):
        # Use cached FFT plan for better performance
        q_grid_hash = hash((Q_grid[0], Q_grid[-1], len(Q_grid), fwhm_Q))
        plan = _get_or_create_fft_plan(q_grid_hash, Q_grid, fwhm_Q)
        return _render_with_plan(Q_peaks, I_peaks, plan)
    else:
        # Fallback to direct computation for non-uniform grids
        sigma = float(fwhm_Q) / 2.354820045
        sigma = max(sigma, 1e-4)
        out = np.zeros_like(Q_grid, float)
        qmin, qmax = float(np.min(Q_grid)), float(np.max(Q_grid))

        # Vectorized mask computation
        mask = (Q_peaks >= qmin - 6*sigma) & (Q_peaks <= qmax + 6*sigma) & (I_peaks > 0)
        if not np.any(mask):
            return out

        Qp = Q_peaks[mask]
        Ip = I_peaks[mask]

        # Vectorized Gaussian computation
        Q_diff = Q_grid[:, np.newaxis] - Qp[np.newaxis, :]
        gaussian_vals = np.exp(-0.5 * (Q_diff / sigma) ** 2)
        out = np.sum(Ip[np.newaxis, :] * gaussian_vals, axis=1)
        return out

def _prepare_residual_for_interp(Q_res, R_res) -> Tuple[np.ndarray, np.ndarray]:
    """Optimized residual preparation with vectorized operations"""
    Q = np.asarray(Q_res, float).ravel()
    R = np.asarray(R_res, float).ravel()

    # Vectorized finite check
    m = np.isfinite(Q) & np.isfinite(R)
    Q = Q[m]
    R = R[m]

    if Q.size == 0:
        return Q, R

    # Use stable sort for better performance on partially sorted data
    order = np.argsort(Q, kind="stable")
    Q = Q[order]
    R = R[order]

    # Optimized duplicate handling using pandas-like groupby logic
    uq, idx_start, counts = np.unique(Q, return_index=True, return_counts=True)
    if uq.size != Q.size:
        # Use np.add.reduceat for efficient grouped averaging
        R_sums = np.add.reduceat(R, idx_start)
        R_collapsed = R_sums / counts
        Q, R = uq, R_collapsed

    return Q, R

def _q_to_tth(q: float, wl: float) -> float:
    """Q [Å⁻¹] → 2θ [deg] for wavelength wl (clip to valid arcsin domain)."""
    s = max(min(q * wl / (4.0 * math.pi), 1.0), 0.0)
    theta = math.asin(s)
    return math.degrees(2.0 * theta)

def _make_fft_plan(q_grid: np.ndarray, fwhm_Q: float):
    """Precompute FFT kernel/plan for Gaussian render on a *fixed uniform* grid."""
    dq = float(q_grid[1] - q_grid[0])
    kernel = _gaussian_kernel_bins(dq, fwhm_Q)
    n = len(q_grid)
    k = len(kernel)
    nfft = 1
    need = n + k - 1
    while nfft < need:
        nfft <<= 1
    F_kern = np.fft.rfft(kernel, nfft)
    start = (k - 1)//2
    return {"dq": dq, "n": n, "k": k, "nfft": nfft, "F_kern": F_kern, "start": start, "q0": float(q_grid[0])}

def _render_with_plan(Q_peaks: np.ndarray, I_peaks: np.ndarray, plan) -> np.ndarray:
    """Optimized render using prebuilt FFT plan"""
    n = plan["n"]
    dq = plan["dq"]
    nfft = plan["nfft"]
    start = plan["start"]
    F_kern = plan["F_kern"]
    q0 = plan["q0"]

    Qp = np.asarray(Q_peaks, float)
    Ip = np.asarray(I_peaks, float)

    if Qp.size == 0:
        return np.zeros(n, float)

    # Vectorized index computation and masking
    idx = np.round((Qp - q0) / dq).astype(np.int32)
    msk = (idx >= 0) & (idx < n) & np.isfinite(Ip) & (Ip > 0)

    if not np.any(msk):
        return np.zeros(n, float)

    idx = idx[msk]
    vals = Ip[msk]

    # --- Patch 2: use bincount for accumulation on uniform grid ---
    delta = np.bincount(idx, weights=vals, minlength=n).astype(np.float64, copy=False)

    # FFT convolution
    F_delta = np.fft.rfft(delta, nfft)
    conv = np.fft.irfft(F_delta * F_kern, nfft)
    return conv[start:start+n]

def _build_fixed_grid_and_residual(Q_res, R_res, ngrid=2048):
    """Optimized grid building with caching"""
    Qr, Rr = _prepare_residual_for_interp(Q_res, R_res)
    if Qr.size == 0:
        return np.linspace(0.0, 1.0, ngrid), np.zeros(ngrid), "R+"

    qmin, qmax = float(np.min(Qr)), float(np.max(Qr))
    q_grid = np.linspace(qmin, qmax, int(ngrid))
    Rg = np.interp(q_grid, Qr, Rr, left=0.0, right=0.0)

    # Vectorized positive/absolute computation
    Rp = np.maximum(Rg, 0.0)
    Ra = np.abs(Rg)

    abs_mass = float(np.sum(Ra))
    pos_mass = float(np.sum(Rp))

    if abs_mass <= 1e-12:
        return q_grid, np.zeros_like(q_grid), "R+"

    use_abs = (pos_mass / abs_mass) < _SCORE_POSRATIO_GLOBAL
    Ruse = Ra if use_abs else Rp
    return q_grid, _minmax(Ruse), ("|R|" if use_abs else "R+")


def _clip_q_residual_window(
    Q_res,
    R_res,
    q_min: Optional[float] = None,
    q_max: Optional[float] = DEFAULT_SCORE_Q_MAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict residual arrays to the Stage-4 scoring Q window."""
    Q = np.asarray(Q_res, float).ravel()
    R = np.asarray(R_res, float).ravel()
    m = np.isfinite(Q) & np.isfinite(R)
    if q_min is not None and math.isfinite(float(q_min)):
        m &= Q >= float(q_min)
    if q_max is not None and math.isfinite(float(q_max)) and float(q_max) > 0.0:
        m &= Q <= float(q_max)
    if not np.any(m):
        return Q, R
    return Q[m], R[m]

# ======================================================================================
# Optimized Scoring
# ======================================================================================

def _coslog_score_on_fixed_grid(Q_sim, I_sim, plan, Rvec, alpha: float, eps: float) -> float:
    """Optimized scoring using numba if available"""
    Is = _render_with_plan(Q_sim, I_sim, plan)

    if HAS_NUMBA:
        return _coslog_score_core_numba(Is, Rvec, alpha, eps)
    else:
        smax = float(np.max(Is)) if Is.size else 0.0
        if not np.isfinite(smax) or smax <= 0:
            return 0.0
        Isn = Is / smax
        Svec = np.log1p(alpha * Isn + eps)
        Rv   = np.log1p(alpha * Rvec + eps)
        denom = float(np.linalg.norm(Svec) * np.linalg.norm(Rv))
        if not np.isfinite(denom) or denom <= 1e-20:
            return 0.0
        cos = float(np.dot(Svec, Rv) / denom)
        return max(0.0, min(1.0, cos))


# ======================================================================================
# Rest of the code remains the same...
# ======================================================================================

def _sanitize_cif_data_block(cif_path: str, pid: str, suffix: str = "nudged", maxlen: int = 40) -> None:
    """
    Make sure the first line of the CIF is a short, parser-friendly 'data_' label.
    Example: data_mp_20525_nudged
    """
    p = Path(cif_path)
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return

    # short, safe: letters/digits/underscores; hyphens -> underscores; trim length
    base = f"{pid}_{suffix}".replace("-", "_")
    base = re.sub(r"[^A-Za-z0-9_]+", "_", base)[:maxlen]
    safe = f"data_{base}" if not base.startswith("data_") else base

    for i, l in enumerate(lines):
        if l.strip().startswith("data_"):
            if l.strip() != safe:
                lines[i] = safe
            break
    else:
        # no data_ block header found; add one at the top
        lines.insert(0, safe)

    try:
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass

def _free_param_names(cons: LatticeConstraints) -> List[str]:
    names = []
    if cons.free_a: names.append("a")
    if cons.free_b and not cons.tie_ab: names.append("b")
    if cons.free_c and not (cons.tie_ac or cons.tie_bc): names.append("c")
    if cons.free_alpha: names.append("alpha")
    if cons.free_beta and not cons.tie_alpha_beta: names.append("beta")
    if cons.free_gamma and not (cons.tie_alpha_gamma or cons.tie_beta_gamma): names.append("gamma")
    return names

def _apply_params_from_vector(base: Lattice, cons: LatticeConstraints,
                              names: List[str], vec: np.ndarray) -> Lattice:
    a,b,c = base.a, base.b, base.c
    al,be,ga = base.alpha, base.beta, base.gamma
    for name, val in zip(names, vec):
        if name == "a": a = float(val)
        elif name == "b": b = float(val)
        elif name == "c": c = float(val)
        elif name == "alpha": al = float(val)
        elif name == "beta":  be = float(val)
        elif name == "gamma": ga = float(val)

    if cons.tie_ab: b = a
    if cons.tie_ac: c = a
    if cons.tie_bc: c = b

    if cons.tie_alpha_beta: be = al
    if cons.tie_alpha_gamma: ga = al
    if cons.tie_beta_gamma: ga = be

    if not cons.free_alpha and cons.alpha0 is not None: al = float(cons.alpha0)
    if not cons.free_beta  and cons.beta0  is not None: be = float(cons.beta0)
    if not cons.free_gamma and cons.gamma0 is not None: ga = float(cons.gamma0)

    al = float(np.clip(al, 40.0, 140.0))
    be = float(np.clip(be,  40.0, 140.0))
    ga = float(np.clip(ga,  40.0, 140.0))

    return Lattice.from_parameters(a, b, c, al, be, ga)

def _within_lattice_tolerances(base_lat: Lattice,
                               cand_lat: Lattice,
                               len_tol_pct: float,
                               ang_tol_deg: float,
                               cons: LatticeConstraints) -> Tuple[bool, str]:
    """
    Return (ok, reason). Enforce |Δa|/a0,|Δb|/b0,|Δc|/c0 <= len_tol_pct (%)
    and |Δα|,|Δβ|,|Δγ| <= ang_tol_deg (deg). For parameters that are *not free*
    under constraints, the deltas will be 0 anyway, but we skip counting them
    for clarity.
    """
    def _pd(new, base):
        denom = max(abs(base), 1e-12)
        return abs(new - base) / denom * 100.0

    # lengths
    checks_len = []
    if cons.free_a: checks_len.append(("a", _pd(cand_lat.a, base_lat.a)))
    if cons.free_b and not cons.tie_ab: checks_len.append(("b", _pd(cand_lat.b, base_lat.b)))
    if cons.free_c and not (cons.tie_ac or cons.tie_bc): checks_len.append(("c", _pd(cand_lat.c, base_lat.c)))

    # angles (only if free — otherwise they’re fixed by cons or set to canonical)
    checks_ang = []
    if cons.free_alpha: checks_ang.append(("alpha", abs(cand_lat.alpha - base_lat.alpha)))
    if cons.free_beta:  checks_ang.append(("beta",  abs(cand_lat.beta  - base_lat.beta)))
    if cons.free_gamma: checks_ang.append(("gamma", abs(cand_lat.gamma - base_lat.gamma)))

    # evaluate
    for name, pct in checks_len:
        if pct > len_tol_pct:
            return False, f"{name} dev {pct:.2f}% > {len_tol_pct:.2f}%"
    for name, deg in checks_ang:
        if deg > ang_tol_deg:
            return False, f"{name} dev {deg:.2f}° > {ang_tol_deg:.2f}°"

    return True, ""

def _pack_vector_from_lattice(lat: Lattice, cons: LatticeConstraints, names: List[str]) -> np.ndarray:
    vals = []
    for name in names:
        if name == "a": vals.append(lat.a)
        elif name == "b": vals.append(lat.b)
        elif name == "c": vals.append(lat.c)
        elif name == "alpha": vals.append(lat.alpha)
        elif name == "beta":  vals.append(lat.beta)
        elif name == "gamma": vals.append(lat.gamma)
    return np.array(vals, dtype=float)

def _q_valid_mask(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    return np.isfinite(q) & (q > 0) & (q < _MAX_Q_BAD)

def _compute_q_jacobian(base_lat: Lattice, cons: LatticeConstraints,
                        names: List[str],
                        hkls: List[Tuple[int,int,int]] = _HKLS7,
                        rel_len_step: float = 0.005,
                        ang_step_deg: float = 0.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    q0_full = hkl_Q_signature(base_lat, hkls=hkls)
    mask = _q_valid_mask(q0_full)
    q0 = q0_full[mask]

    D = len(names)
    if D == 0 or q0.size == 0:
        return q0_full, np.zeros((q0.size, 0), float), mask

    p0 = _pack_vector_from_lattice(base_lat, cons, names)

    J = np.zeros((q0.size, D), float)
    for j, name in enumerate(names):
        v = p0[j]
        if name in ("a","b","c"):
            dv = max(rel_len_step * max(v, 1e-6), 1e-6)
        else:
            dv = max(ang_step_deg, 1e-3)

        p_plus = p0.copy(); p_plus[j] = v + dv
        p_minus = p0.copy(); p_minus[j] = v - dv

        Lp = _apply_params_from_vector(base_lat, cons, names, p_plus)
        Lm = _apply_params_from_vector(base_lat, cons, names, p_minus)

        qp = hkl_Q_signature(Lp, hkls=hkls)[mask]
        qm = hkl_Q_signature(Lm, hkls=hkls)[mask]

        J[:, j] = (qp - qm) / (2.0 * dv)

    return q0_full, J, mask

def _hit_and_run_polytope(U: np.ndarray, b: np.ndarray, n: int,
                          warmup: int = 64, thin: int = 1,
                          seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Uniformly sample c in { c : -b <= U c <= b } by hit-and-run.
    U: (m x r) with orthonormal columns (from SVD of J); b: (m,)
    Returns a list of c-vectors (length r).
    """
    rng = np.random.default_rng(seed)
    m, r = U.shape
    if r == 0:
        return [np.zeros(0)] * n

    c = np.zeros(r, float)  # start feasible at center

    def _limits(cvec, dvec):
        t_lo, t_hi = -np.inf, np.inf
        Ud = U @ dvec
        Uc = U @ cvec
        for i in range(m):
            ai = Ud[i]
            bi = float(b[i])
            ci = float(Uc[i])
            if abs(ai) < 1e-14:
                if abs(ci) > bi:
                    return None, None
                continue
            lo = (-bi - ci) / ai
            hi = ( bi - ci) / ai
            if lo > hi:
                lo, hi = hi, lo
            if lo > t_lo: t_lo = lo
            if hi < t_hi: t_hi = hi
            if t_lo > t_hi:
                return None, None
        return t_lo, t_hi

    # warmup
    for _ in range(max(0, warmup)):
        d = rng.normal(size=r); d /= max(np.linalg.norm(d), 1e-15)
        lo, hi = _limits(c, d)
        if lo is None:
            c[:] = 0.0
            continue
        t = rng.uniform(lo, hi)
        c += t * d

    # draws
    out: List[np.ndarray] = []
    k = 0
    # simple guard in case of numerical hiccups
    budget = 50 * (n + warmup) if n > 0 else 1000
    while len(out) < n and k < budget:
        d = rng.normal(size=r); d /= max(np.linalg.norm(d), 1e-15)
        lo, hi = _limits(c, d)
        if lo is None:
            c[:] = 0.0
            k += 1
            continue
        t = rng.uniform(lo, hi)
        c += t * d
        if thin <= 1 or (k % thin == 0):
            out.append(c.copy())
        k += 1
    return out

def _generate_q_targets(q0_full: np.ndarray,
                        J: np.ndarray,
                        mask_rows: np.ndarray,
                        max_pct: float,
                        samples: Optional[int] = None,
                        seed: Optional[int] = None) -> List[np.ndarray]:
    """
    Uniform polytope sampling in reachable Q-subspace:
      P = { c : -b <= U c <= b }, b = emax * |q0|, emax = max_pct/100.
    Returns list of full-length q_target vectors (length=7).
    """
    q0 = q0_full[mask_rows]
    m = q0.size
    if m == 0 or J.size == 0 or J.shape[0] == 0:
        if _DEBUG:
            logger.info("[stage4] Sampling skipped: no reachable rows or empty Jacobian.")
        return []

    # SVD to get an orthonormal basis U for range(J)
    try:
        U, svals, _ = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError:
        U, _ = np.linalg.qr(J, mode="reduced")
        svals = np.ones(min(J.shape), float)

    rank = int((svals > 1e-10).sum())
    if rank == 0:
        if _DEBUG:
            logger.info("[stage4] Sampling skipped: J rank=0.")
        return []

    U = U[:, :rank]  # (m x r)

    # auto size if samples not provided or <=0
    requested_raw = samples
    if samples is None or int(samples) <= 0:
        density = int(os.environ.get("STAGE4_DENSITY_PER_DIM", "8"))   # per-dim target
        max_samp = int(os.environ.get("STAGE4_MAX_SAMPLES", "2000"))   # hard cap
        min_samp = int(os.environ.get("STAGE4_MIN_SAMPLES", "200"))    # floor
        samples = max(min_samp, min(max_samp, density ** rank))
        mode = "auto"
    else:
        samples = int(samples)
        mode = "requested"

    emax = max_pct / 100.0
    b = np.maximum(emax * np.abs(q0), 1e-6)

    if _DEBUG:
        svals_str = ", ".join([f"{v:.4g}" for v in svals[:rank]])
        logger.debug(f"Polytope sampling: rank={rank}, reachable_rows(m)={m}, "
                     f"window=±{max_pct:.1f}% of q0; mode={mode}, "
                     f"requested={requested_raw}, planned={samples}, seed={seed}, "
                     f"SVD svals=[{svals_str}]")

    Cs = _hit_and_run_polytope(U, b, n=samples,
                               warmup=64, thin=1,
                               seed=seed)

    # Build targets and dedup in relative signature
    targets: List[np.ndarray] = []
    for c in Cs:
        dq = U @ c
        qt = np.minimum(np.maximum(q0 + dq, q0 - b), q0 + b)  # clamp box
        q_full = q0_full.copy()
        q_full[mask_rows] = qt
        targets.append(q_full)

    dedup: List[np.ndarray] = []
    seen = set()
    q0_eps = np.maximum(q0, 1e-12)
    for qf in targets:
        key = tuple(np.round(qf[mask_rows] / q0_eps, 4))  # 0.01% bins
        if key in seen:
            continue
        seen.add(key)
        dedup.append(qf)

    if _DEBUG:
        dup_rate = 0.0 if len(targets) == 0 else (1.0 - len(dedup) / len(targets))
        logger.debug(f"Polytope samples drawn={len(targets)}; "
                     f"unique_after_dedup={len(dedup)}; duplicate_rate={dup_rate:.1%}")

    return dedup


def _backsolve_params_for_qtarget(base_lat: Lattice, cons: LatticeConstraints,
                                  names: List[str],
                                  q0_full: np.ndarray, J: np.ndarray, mask_rows: np.ndarray,
                                  q_target_full: np.ndarray,
                                  reg_lambda: float = _REG_L2) -> Tuple[Lattice, np.ndarray, float]:
    q0 = q0_full[mask_rows]
    qt = q_target_full[mask_rows]
    dq = qt - q0

    if J.size == 0 or J.shape[0] == 0:
        return base_lat, np.zeros(0), 0.0

    JTJ = J.T @ J
    lam = reg_lambda * max(np.trace(JTJ)/max(JTJ.shape[0],1), 1e-12)
    A = JTJ + lam * np.eye(JTJ.shape[0], dtype=float)
    b = J.T @ dq
    try:
        dp = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        dp = np.linalg.lstsq(A, b, rcond=None)[0]

    pred = J @ dp
    lo = -np.abs(0.01 * _QWIN_PCT * q0)
    hi =  np.abs(0.01 * _QWIN_PCT * q0)
    scale = 1.0
    if pred.size:
        scales = []
        for i in range(pred.size):
            pi = pred[i]
            if pi > hi[i] and pi > 0: scales.append(hi[i]/pi)
            elif pi < lo[i] and pi < 0: scales.append(lo[i]/pi)
        if scales:
            scale = max(min(scales), 0.0)
            dp = dp * scale

    p0 = _pack_vector_from_lattice(base_lat, cons, names)
    cand_lat = _apply_params_from_vector(base_lat, cons, names, p0 + dp)
    return cand_lat, dp, float(scale)


def _farthest_point_reps(sig: np.ndarray, k: int, center_idx: int) -> List[int]:
    N = sig.shape[0]
    chosen = [center_idx]
    dmin = np.full(N, np.inf)
    dmin = np.minimum(dmin, np.linalg.norm(sig - sig[center_idx], axis=1))
    for _ in range(1, min(k, N)):
        dmin[chosen] = -1.0
        idx = int(np.argmax(dmin))
        if dmin[idx] <= 0: break
        chosen.append(idx)
        dmin = np.minimum(dmin, np.linalg.norm(sig - sig[idx], axis=1))
    return chosen


# ======================================================================================
# Neutron/X-ray sticks → Q (windowed simulation) - with caching (thread-safe calculator)
# ======================================================================================

class NDSticksQ:
    def __init__(self, wavelength_ang: float = DEFAULT_WAVELENGTH,
                 two_theta_range: Tuple[float,float] = DEFAULT_TWOTHETA_RANGE,
                 radiation: str = "neutron"):
        self.wl = float(wavelength_ang)
        self.tt = (float(two_theta_range[0]), float(two_theta_range[1]))
        self.radiation = radiation.lower()  # "neutron" or "xray"
        # thread-local calculator instance
        self._local = threading.local()
        # Add caching for repeated structure calculations
        self._pattern_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        if "_local" in state:
            del state["_local"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._local = threading.local()

    def _calc(self, wavelength: Optional[float] = None):
        """Return a thread-local diffraction calculator appropriate for the radiation type."""
        wl = float(wavelength if wavelength is not None else self.wl)
        key = round(wl, 8)
        if not hasattr(self._local, "calcs"):
            self._local.calcs = {}
        if key not in self._local.calcs:
            if self.radiation == "xray":
                from pymatgen.analysis.diffraction.xrd import XRDCalculator
                self._local.calcs[key] = XRDCalculator(wavelength=wl)
            else:
                # Local import to avoid hard dependency at module import time
                from pymatgen.analysis.diffraction.neutron import NDCalculator
                self._local.calcs[key] = NDCalculator(wavelength=wl)
        return self._local.calcs[key]

    @staticmethod
    def tt_to_Q(two_theta_deg: np.ndarray, wl: float) -> np.ndarray:
        theta = np.deg2rad(two_theta_deg)*0.5
        return 4.0*math.pi*np.sin(theta)/float(wl)

    def _effective_wavelength_for_q_window(self, q_window: Optional[Tuple[float, float]]) -> float:
        """Return the configured simulator wavelength.

        Stage-4 ranking intentionally focuses on a low-Q scoring window. High-Q
        coverage is controlled by the caller's score_q_max rather than silently
        shortening the wavelength and expanding the simulated pattern.
        """
        return self.wl

    def _get_structure_hash(self, structure: Structure) -> str:
        """Create a hash for structure caching"""
        lat = structure.lattice
        # Include chemistry and fractional coordinates. Lattice-only keys can
        # incorrectly reuse patterns for distinct phases with similar cells.
        parts = [
            f"{lat.a:.6f}_{lat.b:.6f}_{lat.c:.6f}_{lat.alpha:.3f}_{lat.beta:.3f}_{lat.gamma:.3f}",
            str(len(structure)),
        ]
        for site in structure:
            fc = site.frac_coords
            parts.append(
                f"{site.species_string}@{fc[0]:.5f},{fc[1]:.5f},{fc[2]:.5f}"
            )
        return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16).hexdigest()

    def simulate_QI(self, structure: Structure, q_window: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Create cache key
        wl_eff = self._effective_wavelength_for_q_window(q_window)
        struct_hash = self._get_structure_hash(structure)
        cache_key = (struct_hash, q_window, round(wl_eff, 8))

        # Check cache first
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        # optionally restrict two-theta to cover just a Q-window
        if q_window is None:
            ttr = self.tt
        else:
            qlo, qhi = float(q_window[0]), float(q_window[1])
            tmin = _q_to_tth(max(qlo, 0.0), wl_eff)
            qhi_clip = min(qhi, 4.0 * math.pi * math.sin(math.radians(89.0)) / wl_eff)
            tmax = _q_to_tth(qhi_clip, wl_eff)
            if not (tmax > tmin):
                tmin, tmax = self.tt
            ttr = (tmin, tmax)

        patt = self._calc(wl_eff).get_pattern(structure, two_theta_range=ttr)
        tt = np.asarray(patt.x, float)
        I  = np.asarray(patt.y, float)
        Q  = self.tt_to_Q(tt, wl_eff)
        ok = (Q > 0) & np.isfinite(I) & (I > 0)
        if not np.any(ok):
            result = (np.array([], dtype=float), np.array([], dtype=float))
        else:
            Q, I = Q[ok], I[ok]
            order = np.argsort(Q)
            result = (Q[order], I[order])

        # Cache result (limit cache size to prevent memory issues)
        if len(self._pattern_cache) < 1000:
            self._pattern_cache[cache_key] = result

        return result

    def simulate_hkl_intensity_table(
        self,
        structure: Structure,
        q_window: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build an HKL/intensity table once for fast lattice-only rescoring."""
        wl_eff = self._effective_wavelength_for_q_window(q_window)
        struct_hash = self._get_structure_hash(structure)
        cache_key = ("hkl_table", struct_hash, q_window, round(wl_eff, 8))
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        if q_window is None:
            ttr = self.tt
        else:
            qlo, qhi = float(q_window[0]), float(q_window[1])
            tmin = _q_to_tth(max(qlo, 0.0), wl_eff)
            qhi_clip = min(qhi, 4.0 * math.pi * math.sin(math.radians(89.0)) / wl_eff)
            tmax = _q_to_tth(qhi_clip, wl_eff)
            ttr = (tmin, tmax) if tmax > tmin else self.tt

        patt = self._calc(wl_eff).get_pattern(structure, two_theta_range=ttr)
        hkls: List[Tuple[int, int, int]] = []
        intensities: List[float] = []
        for peak_i, grouped_hkls in enumerate(getattr(patt, "hkls", []) or []):
            try:
                peak_intensity = float(patt.y[peak_i])
            except Exception:
                continue
            if not math.isfinite(peak_intensity) or peak_intensity <= 0.0:
                continue
            entries = list(grouped_hkls or [])
            mult_total = sum(max(1, int(entry.get("multiplicity", 1))) for entry in entries) or 1
            for entry in entries:
                hkl = entry.get("hkl")
                if hkl is None or len(hkl) != 3:
                    continue
                mult = max(1, int(entry.get("multiplicity", 1)))
                hkls.append((int(hkl[0]), int(hkl[1]), int(hkl[2])))
                intensities.append(peak_intensity * mult / mult_total)

        result = (np.asarray(hkls, dtype=np.int64), np.asarray(intensities, dtype=float))
        if len(self._pattern_cache) < 1000:
            self._pattern_cache[cache_key] = result
        return result

    @staticmethod
    def reposition_hkl_intensities(
        hkl_table: np.ndarray,
        intensities: np.ndarray,
        lattice: Lattice,
        q_window: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recompute Q positions for an existing HKL/intensity table."""
        hkls = np.asarray(hkl_table, dtype=np.float64)
        Is = np.asarray(intensities, dtype=float)
        if hkls.size == 0 or Is.size == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        Gs = _reciprocal_metric_fast(
            lattice.a,
            lattice.b,
            lattice.c,
            lattice.alpha,
            lattice.beta,
            lattice.gamma,
        )
        h = hkls[:, 0]
        k = hkls[:, 1]
        l = hkls[:, 2]
        dinv2 = (
            h * h * Gs[0, 0]
            + 2.0 * h * k * Gs[0, 1]
            + 2.0 * h * l * Gs[0, 2]
            + k * k * Gs[1, 1]
            + 2.0 * k * l * Gs[1, 2]
            + l * l * Gs[2, 2]
        )
        ok = np.isfinite(dinv2) & (dinv2 > 0.0) & np.isfinite(Is) & (Is > 0.0)
        Q = 2.0 * math.pi * np.sqrt(np.maximum(dinv2, 0.0))
        if q_window is not None:
            qlo, qhi = float(q_window[0]), float(q_window[1])
            ok &= (Q >= qlo) & (Q <= qhi)
        if not np.any(ok):
            return np.array([], dtype=float), np.array([], dtype=float)
        Q = Q[ok]
        I = Is[ok]
        order = np.argsort(Q)
        return Q[order], I[order]


# ======================================================================================
# Public API - Optimized LatticeNudger
# ======================================================================================

@dataclass
class Stage4Result:
    phase_id: str
    best_params: Dict[str, float]
    best_score: float
    nudged_cif_path: str
    elapsed_s: float = 0.0
    candidate_count: int = 0
    scored_count: int = 0
    target_s: float = 0.0
    prep_s: float = 0.0
    score_s: float = 0.0
    scoring_mode: str = ""
    lattice_deviation: float = 0.0
    score_tie_count: int = 1

class LatticeNudger:
    def __init__(self,
                 db_loader,
                 wavelength_ang: float = DEFAULT_WAVELENGTH,
                 two_theta_range: Tuple[float,float] = DEFAULT_TWOTHETA_RANGE,
                 radiation: str = "neutron",
                 score_q_max: Optional[float] = DEFAULT_SCORE_Q_MAX,
                 fast_peak_reposition: Optional[bool] = None,
                 lattice_tiebreak_score_tol: Optional[float] = DEFAULT_LATTICE_TIEBREAK_SCORE_TOL,
                 xray_doublet_config: Optional[Dict] = None,
                 random_seed: Optional[int] = DEFAULT_STAGE4_SEED):
        self.db = db_loader
        self.radiation = str(radiation).lower()
        self.sim = NDSticksQ(wavelength_ang, two_theta_range, radiation=radiation)
        self._fwhm_Q = DEFAULT_FWHM_Q
        self.score_q_max = score_q_max
        self.xray_doublet_config = xray_doublet_config or {"enabled": False}
        if (
            self.radiation in {"xray", "x-ray", "pxrd"}
            and is_active_for is not None
            and is_active_for(self.xray_doublet_config, "apply_to_lattice_nudge")
        ):
            desc = describe_doublet(self.xray_doublet_config) if describe_doublet else "active"
            logger.info(f"PXRD doublet correction active for lattice nudging: {desc}")
        if fast_peak_reposition is None:
            fast_peak_reposition = self.radiation != "xray"
        self.fast_peak_reposition = bool(fast_peak_reposition)
        self.lattice_tiebreak_score_tol = max(0.0, float(lattice_tiebreak_score_tol or 0.0))
        self.random_seed = random_seed
        # Add structure caching to avoid repeated database loads
        self._structure_cache = {}

    def _structure_for_pid(self, pid: str) -> Structure:
        if pid not in self._structure_cache:
            self._structure_cache[pid] = self.db.load_structure(pid)
        return self._structure_cache[pid]

    def optimize_cif(self,
                     cif_path: str,
                     phase_id: str,
                     Q_res: np.ndarray, R_res: np.ndarray,
                     reps: int = 10, samples: int = 200,
                     frac_window: float = 0.025, angle_window_deg: float = 1.5,
                     out_cif_dir: Optional[str] = None,
                     allow_inner_parallel: bool = True,
                     score_q_max: Optional[float] = None) -> Stage4Result:
        """
        Nudge an arbitrary CIF against a target signal.

        This is used for user-supplied main phases, which are not guaranteed to
        exist in the phase database. The structure is registered in the same
        in-memory cache used by optimize_one, so the normal symmetry-aware
        lattice constraints, Q-signature sampling, scoring, and tie-breaking are
        reused without a separate code path.
        """
        if out_cif_dir is None:
            raise ValueError("out_cif_dir must be specified")
        cif = Path(cif_path)
        if not cif.exists():
            raise FileNotFoundError(f"CIF not found: {cif}")

        safe_pid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(phase_id or cif.stem)).strip("._-")
        if not safe_pid:
            safe_pid = cif.stem or "main_phase"
        safe_pid = safe_pid[:80]

        self._structure_cache[safe_pid] = _load_cif_structure_relaxed(cif)
        return self.optimize_one(
            safe_pid,
            Q_res,
            R_res,
            reps=reps,
            samples=samples,
            frac_window=frac_window,
            angle_window_deg=angle_window_deg,
            out_cif_dir=out_cif_dir,
            allow_inner_parallel=allow_inner_parallel,
            score_q_max=score_q_max,
        )

    def _write_nudged_cif(self, pid: str, structure: Structure, out_dir: str,
                        symprec: float = 1e-3, angle_tolerance: float = 5.0) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{pid}_nudged.cif")
        # Write CIF via pymatgen
        structure.to(fmt="cif", filename=path, symprec=symprec, angle_tolerance=angle_tolerance)
        # Enforce a short, GSAS-II-friendly data_ label
        _sanitize_cif_data_block(path, pid, suffix="nudged", maxlen=5)

        return path

    @staticmethod
    def _resolve_inner_workers(num_reps: int, allow_parallel: bool = True) -> int:
        """
        Decide how many threads to use inside a single candidate optimization.

        When Stage 4 is already distributing candidates across a process pool,
        inner fan-out should stay disabled to avoid multiplying concurrency.
        """
        if num_reps <= 1:
            return 1

        env_workers = int(os.environ.get("STAGE4_REP_WORKERS", "0"))
        if env_workers > 0:
            return max(1, min(env_workers, num_reps))

        is_hf = "SPACE_ID" in os.environ
        if not allow_parallel or is_hf:
            return 1

        cpu = os.cpu_count() or 1
        return max(1, min(num_reps, cpu // 2))

    @staticmethod
    def _resolve_outer_workers(num_candidates: int) -> int:
        """
        Decide how many candidate processes to launch for Stage 4.

        The worker count should never exceed the number of candidates, since
        extra idle workers add large process/memory overhead without speeding up
        the search.
        """
        if num_candidates <= 1:
            return 1

        is_hf = "SPACE_ID" in os.environ
        cpu_count = os.cpu_count() or 1
        max_workers_env = int(os.environ.get("STAGE4_MAX_WORKERS", "0"))
        if max_workers_env > 0:
            workers = max_workers_env
        elif is_hf:
            workers = min(2, cpu_count)
        else:
            workers = max(1, cpu_count // 2)
        return max(1, min(num_candidates, workers))

    # ------------------------ Q-signature constrained strategy (optimized) ------------------------

    def _make_candidates_qsignature(self,
                                base_struct: Structure,
                                cons: LatticeConstraints,
                                reps: int = 10,
                                samples: int = 200,
                                len_tol_pct: Optional[float] = None,
                                ang_tol_deg: Optional[float] = None) -> List[Lattice]:

        base_lat = base_struct.lattice
        names = _free_param_names(cons)

        q0_full, J, mask_rows = _compute_q_jacobian(base_lat, cons, names)
        if _DEBUG:
            q0_dbg = np.round(q0_full, 5)
            valid_cnt = int(mask_rows.sum())
            logger.debug(f"Q-signature(base)={q0_dbg}")
            logger.debug(f"Valid HKL entries={valid_cnt}/7, DOF={len(names)} ({names})")

        if len(names) == 0 or int(mask_rows.sum()) == 0:
            if _DEBUG:
                logger.debug("No DOF or no valid HKL rows; only base lattice will be used.")
            return [base_lat]

        seed_val = getattr(self, "random_seed", DEFAULT_STAGE4_SEED)
        # resolve tolerances (percent for lengths, degrees for angles)
        if len_tol_pct is None:
            len_tol_pct = float(os.environ.get("STAGE4_LEN_TOL_PCT", "3.0"))
        if ang_tol_deg is None:
            ang_tol_deg = float(os.environ.get("STAGE4_ANG_TOL_DEG", "6.0"))
        if _DEBUG:
            logger.info(f"Lattice tolerance: lengths ≤ {len_tol_pct:.2f}%  |  angles ≤ {ang_tol_deg:.2f}°")

        if float(len_tol_pct) <= 0.0 and float(ang_tol_deg) <= 0.0:
            if _DEBUG:
                logger.debug("Zero lattice tolerances requested; skipping target generation and using base lattice only.")
            return [base_lat]

        q_targets = _generate_q_targets(q0_full, J, mask_rows,
                                        max_pct=_QWIN_PCT,
                                        samples=samples,
                                        seed=seed_val)

        cands: List[Lattice] = [base_lat]
        rej_len = 0
        rej_ang = 0

        for idx, qt in enumerate(q_targets):
            Lcand, dp, scl = _backsolve_params_for_qtarget(base_lat, cons, names,
                                                        q0_full, J, mask_rows, qt,
                                                        reg_lambda=_REG_L2)
            ok, reason = _within_lattice_tolerances(base_lat, Lcand, len_tol_pct, ang_tol_deg, cons)
            if _DEBUG and (idx < 8 or idx % 50 == 0):
                dpn = float(np.linalg.norm(dp)) if dp.size else 0.0
                tag = "kept" if ok else f"rejected ({reason})"
                logger.debug(f"Target#{idx:04d}: |Δp|={dpn:.5g}, scale={scl:.3f} → {tag}")

            if ok:
                cands.append(Lcand)
            else:
                # reason-based accounting
                if "dev" in reason and "%" in reason:
                    rej_len += 1
                elif "°" in reason:
                    rej_ang += 1

        # Dedup by Q-signature (0.1% bins)
        uniq: List[Lattice] = []
        seen = set()
        q0_norm = np.maximum(q0_full, 1e-12)
        for L in cands:
            q = hkl_Q_signature(L)
            key = tuple(np.round(q / q0_norm, 4))
            if key not in seen:
                seen.add(key)
                uniq.append(L)

        # FPS to 'reps'
        S = np.vstack([hkl_Q_signature(L) for L in uniq])
        S_proc = np.where(np.isfinite(S), np.clip(S, 0.0, _MAX_Q_BAD), _MAX_Q_BAD)

        k = max(2, int(reps))
        k = min(k, S_proc.shape[0])
        idxs = _farthest_point_reps(S_proc, k=k, center_idx=0)
        reps_lattices = [uniq[i] for i in idxs]

        if _DEBUG:
            # minus 1 because base_lat was pre-seeded in cands
            built_from_targets = len(q_targets)
            kept_targets = len(cands) - 1
            logger.debug(f"lattice tolerance: built={built_from_targets}, kept={kept_targets}, "
                 f"rejected_len={rej_len}, rejected_ang={rej_ang}")
            logger.debug(f"Lattice candidates: built={len(cands)}; "
                 f"unique_by_Qsig={len(uniq)}; FPS_kept={len(reps_lattices)} (reps={reps})")

        # optional: warn if only the base lattice survived
        if len(cands) == 1 and _DEBUG:
            logger.warning("All candidates rejected by lattice tolerances; using base lattice only.")

        return reps_lattices

    def optimize_one(self,
                    phase_id: str,
                    Q_res: np.ndarray, R_res: np.ndarray,
                    reps: int = 10, samples: int = 200,
                    frac_window: float = 0.025, angle_window_deg: float = 1.5,
                    out_cif_dir: Optional[str] = None,
                    allow_inner_parallel: bool = True,
                    score_q_max: Optional[float] = None) -> Stage4Result:
        """
        Nudge one candidate structure against residual R(Q).
        """
        if out_cif_dir is None:
            raise ValueError("out_cif_dir must be specified")
        global _QWIN_PCT  # <-- declare first
        t_total = perf_counter()

        pid = str(phase_id)
        base_struct = self._structure_for_pid(pid)
        base_lat = base_struct.lattice

        sgnum = None
        try:
            sgnum = self.db.get_space_group_number(pid)
        except Exception:
            pass
        if sgnum is None:
            try:
                from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
                sgnum = int(
                    SpacegroupAnalyzer(
                        base_struct,
                        symprec=1e-2,
                        angle_tolerance=5.0,
                    ).get_space_group_number()
                )
            except Exception:
                sgnum = None
        cons = infer_constraints(base_struct, sgnum)

        if _DEBUG:
            logger.info(f"===== Phase {pid} | SG={sgnum if sgnum is not None else 'unknown'} =====")
            logger.info(f"Base lattice: a={base_lat.a:.5f} b={base_lat.b:.5f} c={base_lat.c:.5f} "
                        f"α={base_lat.alpha:.3f} β={base_lat.beta:.3f} γ={base_lat.gamma:.3f}")
            logger.info(f"Constraint window (pre-scale): ±{_QWIN_PCT:.1f}% in 7-D HKL Q-signature")
        # interpret frac_window as fraction (≤1 → fraction; >1 → already percent)
        if frac_window is None:
            len_tol_pct = float(os.environ.get("STAGE4_LEN_TOL_PCT", "3.0"))
        else:
            len_tol_pct = (float(frac_window) * 100.0) if float(frac_window) <= 1.0 else float(frac_window)

        ang_tol_deg = float(angle_window_deg) if angle_window_deg is not None else float(os.environ.get("STAGE4_ANG_TOL_DEG", "5.0"))

        if _DEBUG:
            logger.info(f"Lattice tolerance in use → lengths ≤ {len_tol_pct:.2f}% ; angles ≤ {ang_tol_deg:.2f}°")

        # Determine crystal system and pick a scale
        cs = _crystal_system_from_sgnum(int(sgnum)) if sgnum is not None else "unknown"
        _scale = {
            "cubic": 1.0,
            "tetragonal": 1,
            "hexagonal": 1,
            "orthorhombic": 1,
            "trigonal": 1,
            "monoclinic": 1,
            "triclinic": 1,
        }.get(cs, 1.0)

        _prev_qwin = _QWIN_PCT
        try:
            _QWIN_PCT = _prev_qwin * _scale
            if _DEBUG:
                logger.debug(f"Crystal system={cs}; constraint window: ±{_prev_qwin:.1f}% → ±{_QWIN_PCT:.1f}%")
            t_targets = perf_counter()
            reps_lattices = self._make_candidates_qsignature(base_struct, cons, reps=reps,samples=samples,len_tol_pct=len_tol_pct, ang_tol_deg=ang_tol_deg)
            target_s = perf_counter() - t_targets
        finally:
            _QWIN_PCT = _prev_qwin  # restore


        # ----- Fixed grid + precomputed residual + FFT plan (speed-up) -----
        ngrid_fixed = int(os.environ.get("STAGE4_FIXED_GRID_N", "2048"))
        alpha = float(os.environ.get("STAGE4_COSLOG_ALPHA", "50.0"))
        eps   = float(os.environ.get("STAGE4_COSLOG_EPS", "1e-12"))
        q_limit = self.score_q_max if score_q_max is None else score_q_max
        Q_score, R_score = _clip_q_residual_window(Q_res, R_res, q_max=q_limit)
        if Q_score.size != np.asarray(Q_res).size:
            logger.info(
                f"Stage-4 scoring Q-window: kept {Q_score.size}/{np.asarray(Q_res).size} "
                f"points up to Q={float(q_limit):.3f} A^-1"
            )
        q_grid, Rvec, rlabel = _build_fixed_grid_and_residual(Q_score, R_score, ngrid=ngrid_fixed)
        plan = _make_fft_plan(q_grid, self._fwhm_Q)
        # Q margin to ensure tails of peaks contribute
        sigma = float(self._fwhm_Q) / 2.354820045
        q_margin = float(os.environ.get("STAGE4_Q_MARGIN_SIGMAS", "6.0"))
        q_lo = max(float(q_grid[0])  - q_margin * sigma, 0.0)
        q_hi =      float(q_grid[-1]) + q_margin * sigma
        if _DEBUG:
            logger.debug(f"fixed grid n={len(q_grid)} in [{q_grid[0]:.5f},{q_grid[-1]:.5f}], residual={rlabel}")
            logger.debug(f"sim Q-window with margin: [{q_lo:.5f},{q_hi:.5f}]")

        hkl_table = np.empty((0, 3), dtype=np.int64)
        hkl_intensities = np.empty(0, dtype=float)
        prep_s = 0.0
        use_fast_reposition = bool(self.fast_peak_reposition)
        if use_fast_reposition:
            try:
                t_prep = perf_counter()
                hkl_table, hkl_intensities = self.sim.simulate_hkl_intensity_table(
                    base_struct,
                    q_window=(q_lo, q_hi),
                )
                prep_s = perf_counter() - t_prep
                use_fast_reposition = hkl_table.size > 0 and hkl_intensities.size > 0
            except Exception as exc:
                logger.debug(f"Fast HKL reposition unavailable for {pid}: {exc}")
                use_fast_reposition = False

        # --- Patch 4: parallel representative scoring (adaptive, notebook-safe) ---
        def _score_for_L(L):
            if use_fast_reposition:
                Qs, Is = self.sim.reposition_hkl_intensities(
                    hkl_table,
                    hkl_intensities,
                    L,
                    q_window=(q_lo, q_hi),
                )
            else:
                struct_i = _rebuild_with_lattice(base_struct, L)
                Qs, Is = self.sim.simulate_QI(struct_i, q_window=(q_lo, q_hi))
            if (
                self.radiation in {"xray", "x-ray", "pxrd"}
                and apply_doublet_to_peaks is not None
                and is_active_for is not None
                and is_active_for(self.xray_doublet_config, "apply_to_lattice_nudge")
            ):
                Qs, Is = apply_doublet_to_peaks(
                    Qs,
                    Is,
                    self.xray_doublet_config,
                    apply_key="apply_to_lattice_nudge",
                )
            return _coslog_score_on_fixed_grid(Qs, Is, plan, Rvec, alpha, eps)

        workers = self._resolve_inner_workers(
            len(reps_lattices),
            allow_parallel=allow_inner_parallel,
        )

        t_score = perf_counter()
        if workers > 1 and len(reps_lattices) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as ex:
                scores = np.array(list(ex.map(_score_for_L, reps_lattices)), dtype=float)
        else:
            scores = np.array([_score_for_L(L) for L in reps_lattices], dtype=float)
        score_s = perf_counter() - t_score

        deviations = np.array(
            [
                _lattice_deviation_metric(base_lat, L, len_tol_pct, ang_tol_deg)
                for L in reps_lattices
            ],
            dtype=float,
        )
        best_sc = float(np.max(scores)) if scores.size else -1.0
        best_i, score_tie_count = _choose_best_lattice_index(
            scores,
            deviations,
            self.lattice_tiebreak_score_tol,
        )
        best_dev = float(deviations[best_i]) if deviations.size else 0.0

        if best_sc <= 0:
            L_best = base_lat
            best_sc = 0.0
            best_dev = 0.0
        else:
            L_best = reps_lattices[best_i]

        violations = _lattice_constraint_violations(cons, L_best)
        if violations:
            logger.warning(
                f"{pid}: nudged lattice violated constraints {violations}; "
                "reprojecting through the inferred constraint model."
            )
            names = _free_param_names(cons)
            L_best = _apply_params_from_vector(
                base_lat,
                cons,
                names,
                _pack_vector_from_lattice(L_best, cons, names),
            )

        struct_best = _rebuild_with_lattice(base_struct, L_best)
        out_path = self._write_nudged_cif(pid, struct_best, out_cif_dir)

        if _DEBUG:
            logger.info(f"BEST for {pid}: score={best_sc:.4f}  "
                        f"a={L_best.a:.5f} b={L_best.b:.5f} c={L_best.c:.5f} "
                        f"α={L_best.alpha:.3f} β={L_best.beta:.3f} γ={L_best.gamma:.3f}")
            logger.info(f"→ wrote: {out_path}")

        params = dict(a=L_best.a, b=L_best.b, c=L_best.c,
                      alpha=L_best.alpha, beta=L_best.beta, gamma=L_best.gamma)

        return Stage4Result(phase_id=pid, best_params=params,
                            best_score=float(max(best_sc, 0.0)),
                            nudged_cif_path=out_path,
                            elapsed_s=perf_counter() - t_total,
                            candidate_count=len(reps_lattices),
                            scored_count=int(scores.size),
                            target_s=target_s,
                            prep_s=prep_s,
                            score_s=score_s,
                            scoring_mode="hkl-reposition" if use_fast_reposition else "exact",
                            lattice_deviation=best_dev,
                            score_tie_count=int(score_tie_count))

    def optimize_many(self,
                      candidates: List[str],
                      Q_res: np.ndarray, R_res: np.ndarray,
                      reps: int = 10, samples: int = 200,
                      frac_window: float = 0.025, angle_window_deg: float = 1.5,
                      out_cif_dir: Optional[str] = None,
                      score_q_max: Optional[float] = None) -> List[Stage4Result]:
        """
        Parallel optimization of many candidates.
        """
        if out_cif_dir is None:
            raise ValueError("out_cif_dir must be specified")
        try:
            logger.info(f"residual Q-range: ({float(np.min(Q_res)):.3f}, {float(np.max(Q_res)):.3f})")
        except Exception:
            pass

        workers = self._resolve_outer_workers(len(candidates))
        is_hf = "SPACE_ID" in os.environ
        if is_hf:
            logger.info(f"Hugging Face Space detected. Capping workers to {workers} for RAM safety.")

        logger.info(f"Starting parallel Lattice Nudging for {len(candidates)} candidates with {workers} workers.")
        
        import sys
        sys.stdout.flush()

        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_pid = {
                executor.submit(
                    self.optimize_one,
                    pid,
                    Q_res,
                    R_res,
                    reps,
                    samples,
                    frac_window,
                    angle_window_deg,
                    out_cif_dir,
                    False,
                    score_q_max,
                ): pid for pid in candidates
            }
            
            for future in concurrent.futures.as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    r = future.result()
                    results.append(r)
                    logger.info(
                        f"{pid}: score={r.best_score:.3f}, elapsed={r.elapsed_s:.3f}s "
                        f"(targets={r.target_s:.3f}s, prep={r.prep_s:.3f}s, score={r.score_s:.3f}s, "
                        f"reps={r.scored_count}/{r.candidate_count}, mode={r.scoring_mode}, "
                        f"tie={r.score_tie_count}, dev={r.lattice_deviation:.3f}) "
                        f"-> {os.path.basename(r.nudged_cif_path)}"
                    )
                except Exception as e:
                    logger.warning(f"{pid}: failed: {e}")

        results.sort(
            key=lambda r: (
                -float(r.best_score),
                float(r.lattice_deviation),
                str(r.phase_id),
            )
        )
        for r in sorted(results, key=lambda item: item.elapsed_s, reverse=True)[:3]:
            logger.info(
                f"[time] slow nudge {r.phase_id}: total={r.elapsed_s:.3f}s, "
                f"targets={r.target_s:.3f}s, prep={r.prep_s:.3f}s, score={r.score_s:.3f}s, "
                f"reps={r.scored_count}/{r.candidate_count}, mode={r.scoring_mode}"
            )
        return results
