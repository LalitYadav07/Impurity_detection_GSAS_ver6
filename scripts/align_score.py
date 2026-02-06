"""
Diffraction Pattern Alignment and Scoring Utilities

Provides core logic for:
- Scale-invariant ratio histogram computation for candidate pre-filtering.
- Peak-based alignment scoring for ranking candidates against residuals.
- Chi-square improvement metrics for evaluating candidate fit.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

from ratio_filter import _observed_hist_from_Q
from profiles import render_pattern



def align_score_candidate(
    residual_Q: np.ndarray,
    cand_Q: np.ndarray,
    cand_I: np.ndarray,
    q_tol: float = 0.02,
    coverage_thresh: float = 0.02,
    # --- new (optional) args for χ²-based scoring ---
    Q: np.ndarray | None = None,                 # common grid
    y_resid: np.ndarray | None = None,           # residual on that grid (background already removed)
    broadening: dict | None = None,              # same broadening dict used elsewhere
    w: np.ndarray | None = None,                 # optional weights (e.g., 1/σ), shape = Q
    # weights for combining terms
    lambda_match: float = 1.0,
    lambda_quality: float = 1.0,
    lambda_miss: float = 0.5,
    lambda_chisq: float = 5.0,                   # weight for χ² improvement term
) -> Tuple[float, dict]:
    """
    Updated: score alignment using Δχ² improvement (optional) in addition to positional matching.
    - If (Q, y_resid, broadening) provided, compute the best single-component NNLS fit y_resid ~ α * a
      where a = render_pattern(Q, cand_Q, cand_I, broadening). The χ² (SSE) reduction contributes to the score.
    - Otherwise, fall back to pure positional scoring.

    Returns (score, details). Higher is better.
    """
    # guard rails
    if residual_Q.size == 0 or cand_Q.size == 0:
        return -1e9, {"n_matches": 0, "coverage_misses": 0}

    # ------------- POSITIONAL MATCH TERMS (unchanged) -------------
    idx = np.searchsorted(cand_Q, residual_Q)
    idx0 = np.clip(idx - 1, 0, len(cand_Q) - 1)
    idx1 = np.clip(idx,     0, len(cand_Q) - 1)
    d0 = np.abs(residual_Q - cand_Q[idx0])
    d1 = np.abs(residual_Q - cand_Q[idx1])
    dmin = np.minimum(d0, d1)
    matches = dmin <= q_tol

    n_matches = int(matches.sum())
    if n_matches > 0:
        med_dq = float(np.median(dmin[matches]))
        quality = -med_dq
    else:
        med_dq = None
        quality = -1.0

    sig_mask = cand_I >= (coverage_thresh * np.max(cand_I))
    sig_Q = cand_Q[sig_mask]
    if sig_Q.size:
        j = np.searchsorted(residual_Q, sig_Q)
        j0 = np.clip(j - 1, 0, len(residual_Q) - 1)
        j1 = np.clip(j,     0, len(residual_Q) - 1)
        dm0 = np.abs(sig_Q - residual_Q[j0])
        dm1 = np.abs(sig_Q - residual_Q[j1])
        miss = np.minimum(dm0, dm1) > q_tol
        n_miss = int(miss.sum())
    else:
        n_miss = 0

    # ------------- CHI-SQUARE IMPROVEMENT TERM (new, optional) -------------
    chisq_gain = 0.0
    alpha_opt = 0.0
    sse0 = None
    sse1 = None
    frac_red = None

    use_chisq = (Q is not None) and (y_resid is not None) and (broadening is not None)
    if use_chisq:
        # lazily import or assume render_pattern is in scope
        a = render_pattern(Q, cand_Q, cand_I, broadening)  # candidate's broadened profile on grid

        if w is not None:
            # weighted least squares (use w as per-point weights, e.g., 1/σ)
            a_w = a * w
            y_w = y_resid * w
        else:
            a_w = a
            y_w = y_resid

        # SSE of current residual (baseline)
        sse0 = float(np.dot(y_w, y_w))

        # Best single-component NNLS coefficient: α = max( (a·y)/(a·a), 0 )
        denom = float(np.dot(a_w, a_w)) + 1e-18
        numer = float(np.dot(a_w, y_w))
        alpha_opt = max(numer / denom, 0.0)

        # New SSE after adding this candidate scaled by α
        r1 = y_w - alpha_opt * a_w
        sse1 = float(np.dot(r1, r1))

        delta_sse = sse0 - sse1  # non-negative if alpha_opt > 0 and helpful
        # Normalize by baseline SSE to make comparable across samples
        frac_red = (delta_sse / max(sse0, 1e-18)) if sse0 is not None else 0.0

        # This is the main new term
        chisq_gain = frac_red

    # ------------- FINAL SCORE -------------
    score = (
        lambda_match * (n_matches)
        + lambda_quality * (quality)
        - lambda_miss * (n_miss)
        + lambda_chisq * (chisq_gain)
    )

    details = {
        "n_matches": n_matches,
        "median_dq": med_dq,
        "coverage_misses": n_miss,
        "chisq_frac_reduction": frac_red,
        "chisq_sse0": sse0,
        "chisq_sse1": sse1,
        "alpha_opt": alpha_opt,
        "used_chisq": bool(use_chisq),
        "terms": {
            "match": lambda_match * (n_matches),
            "quality": lambda_quality * (quality),
            "coverage_penalty": -lambda_miss * (n_miss),
            "chisq_gain": lambda_chisq * (chisq_gain),
        },
    }
    return float(score), details
