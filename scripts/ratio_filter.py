#!/usr/bin/env python3
"""
ML-Based Histogram Screening (Ratio Filter)

This module implements the 64-bin histogram screening logic used to rank candidate phases.
Functions include:
- `get_hist_from_resid`: Converts continuous residual signals into binned histograms.
- Candidate scoring and probability estimation using pre-trained ML models.
- "Knee" detection for dynamic thresholding of candidate lists.
"""

# ---------------------------
# Standard library
# ---------------------------
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------
# Third-party
# ---------------------------
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# DB pack loader
# =============================================================================
_PROFILE_CACHE: Dict[str, Any] = {}

def _load_profiles64_metadata(profiles_dir: str) -> Dict[str, Any]:
    """
    Load 64-bin profiles and metadata from profiles64.npz (+ index.csv).
    Caches results in-memory to avoid redundant I/O.
    """
    global _PROFILE_CACHE
    cache_key = str(Path(profiles_dir).resolve())
    if cache_key in _PROFILE_CACHE:
        return _PROFILE_CACHE[cache_key]

    prof_npz = os.path.join(profiles_dir, "profiles64.npz")
    idx_csv = os.path.join(profiles_dir, "index.csv")
    if not os.path.exists(prof_npz):
        raise FileNotFoundError(f"profiles64.npz not found: {prof_npz}")
    if not os.path.exists(idx_csv):
        raise FileNotFoundError(f"index.csv not found: {idx_csv}")

    with np.load(prof_npz) as z:
        profiles = z["profiles"].astype(np.float64)  # (N, 64)
        q_min = float(z["q_min"])
        q_max = float(z["q_max"])
        n_bins = int(z["n_bins"])  # should be 64
        sigma_bins = float(z["sigma_bins"])  # smoothing width in BIN units

    edges = np.linspace(q_min, q_max, n_bins + 1, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])

    index_df = pd.read_csv(idx_csv)
    if "id" not in index_df.columns:
        raise ValueError("index.csv must contain an 'id' column.")

    if "row" in index_df.columns:
        pid_to_row = dict(
            zip(index_df["id"].astype(str), index_df["row"].astype(int))
        )
    else:
        # assume row order is the file order
        pid_to_row = {str(pid): i for i, pid in enumerate(index_df["id"].astype(str))}

    if len(pid_to_row) != profiles.shape[0]:
        raise ValueError(
            f"Index/profiles size mismatch: {len(pid_to_row)} vs {profiles.shape[0]}"
        )
    if profiles.shape[1] != n_bins:
        raise ValueError(f"Profile bins mismatch: {profiles.shape[1]} vs {n_bins}")

    res = {
        "profiles": profiles,
        "pid_to_row": pid_to_row,
        "q_min": q_min,
        "q_max": q_max,
        "n_bins": n_bins,
        "sigma_bins": sigma_bins,
        "edges": edges,
        "centers": centers,
    }
    _PROFILE_CACHE[cache_key] = res
    return res







# =============================================================================
# Residual → 64-bin histogram (continuous, ΔQ-weighted)
# =============================================================================
def _segment_ids_from_q_sequence(Q: np.ndarray, gap_factor: float = 5.0) -> np.ndarray:
    """
    Split a 1D Q sequence into contiguous observed segments.

    Excluded regions remove samples entirely, so the surviving Q values can have
    large jumps. Those jumps should terminate one observed segment and start the
    next; otherwise downstream ΔQ weighting incorrectly bridges across missing
    regions.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 1:
        raise ValueError("Q must be 1D.")
    n = int(Q.size)
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n == 1:
        return np.zeros((1,), dtype=np.int32)

    gaps = np.abs(np.diff(Q))
    pos_gaps = gaps[gaps > 0.0]
    if pos_gaps.size == 0:
        return np.zeros((n,), dtype=np.int32)

    median_gap = float(np.median(pos_gaps))
    if not np.isfinite(median_gap) or median_gap <= 0.0:
        return np.zeros((n,), dtype=np.int32)

    break_mask = gaps > (float(gap_factor) * median_gap)
    seg_ids = np.zeros((n,), dtype=np.int32)
    if np.any(break_mask):
        seg_ids[1:] = np.cumsum(break_mask.astype(np.int32))
    return seg_ids


def _residual_hist_from_continuous_parts(
    Q: np.ndarray,
    R: np.ndarray,
    Q_main_peaks: Optional[np.ndarray],
    edges: np.ndarray,
    sigma_bins: float,
    peak_mask_width: float = 0.015,  # Q half-width to mask around each main-phase peak
    debug_plot: bool = False,        # set True or env RESID_BIN_DEBUG=1 to save a PNG
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 64-bin residual histogram H from continuous R(Q) with:
      - sample-level masking near main-phase peaks,
      - signed *area* accumulation per bin (ΔQ-weighted, order-robust),
      - segment-aware ΔQ weighting so internal excluded gaps are not bridged,
      - optional Gaussian smoothing in *bin units* (sigma_bins from DB),
      - late rectification (clip negatives to 0),
      - a 64-bin observability mask describing which bins truly contain observed data,
      - NO min–max scaling here (active-window normalization happens later).

    IMPORTANT:
      - ΔQ uses absolute neighbor gaps, so ascending/descending Q both work.
      - excluded regions are treated as *unobserved*, not as zeros.
    """
    Q = np.asarray(Q, dtype=float)
    R = np.asarray(R, dtype=float)
    Q_main_peaks = (
        np.asarray(Q_main_peaks, dtype=float)
        if Q_main_peaks is not None
        else np.array([], dtype=float)
    )

    if Q.ndim != 1 or R.ndim != 1 or Q.shape != R.shape:
        raise ValueError("Q and R must be 1D arrays of the same shape.")
    if not np.all(np.isfinite(Q)) or not np.all(np.isfinite(R)):
        raise ValueError("Q and R must be finite.")

    n_bins = int(len(edges) - 1)
    if Q.size < 2 or n_bins <= 0:
        zeros = np.zeros(max(n_bins, 0), dtype=np.float64)
        return zeros, np.zeros_like(zeros, dtype=bool), np.zeros_like(zeros, dtype=np.int64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be strictly increasing.")

    # 1) In-range & sample-level peak masking
    in_range = (Q >= edges[0]) & (Q <= edges[-1])

    if Q_main_peaks.size > 0 and peak_mask_width > 0:
        peak_mask = np.ones(Q.shape[0], dtype=bool)
        for q0 in Q_main_peaks:
            peak_mask &= ~(np.abs(Q - q0) <= peak_mask_width)
        m = in_range & peak_mask
    else:
        m = in_range

    if not np.any(m):
        zeros = np.zeros(n_bins, dtype=np.float64)
        return zeros, np.zeros(n_bins, dtype=bool), np.zeros(n_bins, dtype=np.int64)

    Qm = Q[m]
    Rm = R[m]

    # 2) Segment-aware ΔQ weights and observability support.
    H = np.zeros(n_bins, dtype=np.float64)
    bin_observed_mask = np.zeros(n_bins, dtype=bool)
    C = np.zeros(n_bins, dtype=np.int64)

    seg_ids = _segment_ids_from_q_sequence(Qm)
    for seg_id in np.unique(seg_ids):
        seg_sel = (seg_ids == seg_id)
        Qs = np.asarray(Qm[seg_sel], dtype=np.float64)
        Rs = np.asarray(Rm[seg_sel], dtype=np.float64)
        if Qs.size == 0:
            continue

        order = np.argsort(Qs)
        Qs = Qs[order]
        Rs = Rs[order]

        if Qs.size == 1:
            dq = np.zeros((1,), dtype=np.float64)
            support_lo = Qs.copy()
            support_hi = Qs.copy()
        else:
            gaps = np.abs(np.diff(Qs))
            left_half = np.empty_like(Qs)
            right_half = np.empty_like(Qs)
            left_half[0] = 0.5 * gaps[0]
            right_half[-1] = 0.5 * gaps[-1]
            left_half[1:] = 0.5 * gaps
            right_half[:-1] = 0.5 * gaps
            dq = np.maximum(left_half + right_half, 0.0)
            support_lo = Qs - left_half
            support_hi = Qs + right_half

        idx = np.searchsorted(edges, Qs, side="right") - 1
        idx = np.clip(idx, 0, n_bins - 1)
        np.add.at(H, idx, Rs * dq)
        np.add.at(C, idx, 1)

        for lo, hi in zip(support_lo, support_hi):
            if not np.isfinite(lo) or not np.isfinite(hi):
                continue
            span_lo, span_hi = (float(min(lo, hi)), float(max(lo, hi)))
            if span_hi <= edges[0] or span_lo >= edges[-1]:
                continue
            overlap = (edges[:-1] < span_hi) & (edges[1:] > span_lo)
            bin_observed_mask |= overlap

    # 3) Optional Gaussian smoothing in BIN units, but never across unobserved gaps.
    if sigma_bins is not None and float(sigma_bins) > 0.0:
        sb = float(sigma_bins)
        W = int(np.ceil(3.0 * sb))  # ±3σ support
        if W > 0:
            x = np.arange(-W, W + 1, dtype=np.float64)
            ker = np.exp(-0.5 * (x / sb) ** 2)
            ker /= ker.sum()
            if np.any(bin_observed_mask):
                smoothed = np.zeros_like(H)
                starts = np.flatnonzero(bin_observed_mask & np.r_[True, ~bin_observed_mask[:-1]])
                stops = np.flatnonzero(bin_observed_mask & np.r_[~bin_observed_mask[1:], True])
                for start, stop in zip(starts, stops):
                    segment = H[start : stop + 1]
                    if segment.size == 1:
                        smoothed[start : stop + 1] = segment
                        continue

                    # np.convolve(..., mode="same") returns the larger of the two lengths,
                    # which breaks for short observed segments when the kernel is wider
                    # than the segment. Pad inside the observed segment and trim back to
                    # the original segment length so smoothing never crosses excluded gaps.
                    pad = min(W, max(segment.size - 1, 0))
                    padded = np.pad(segment, (pad, pad), mode="edge")
                    conv = np.convolve(padded, ker, mode="same")
                    smoothed[start : stop + 1] = conv[pad : pad + segment.size]
                H = smoothed

    # 4) Late rectification (non-negative residual mass)
    H = np.maximum(H, 0.0)

    # 5) Debug plot (optional)
    if debug_plot or os.environ.get("RESID_BIN_DEBUG", "") == "1":
        try:
            import matplotlib.pyplot as plt

            centers = 0.5 * (edges[:-1] + edges[1:])
            out = Path(os.environ.get("RESID_BIN_DEBUG_PNG", "./resid_bin_debug.png"))

            fig, axs = plt.subplots(2, 1, figsize=(10.5, 7.0), constrained_layout=True)
            # Panel 1: residual with masked spans
            axs[0].plot(Q, R, lw=1.2, color="#111111", label="Residual R(Q)", zorder=2)
            if Q_main_peaks.size > 0 and peak_mask_width > 0:
                for i, q0 in enumerate(Q_main_peaks):
                    axs[0].axvspan(
                        q0 - peak_mask_width,
                        q0 + peak_mask_width,
                        alpha=0.25,
                        color="#FFB3B3",
                        label="Masked ±Δ" if i == 0 else None,
                        zorder=1,
                    )
            axs[0].set_title("Residual R(Q) with sample-level peak masking")
            axs[0].set_xlabel("Q (Å⁻¹)")
            axs[0].set_ylabel("Intensity (arb.)")
            axs[0].legend(loc="upper right", fontsize=9)
            axs[0].grid(alpha=0.2, linewidth=0.6)

            # Panel 2: 64-bin histogram (+ counts if available)
            axs[1].axhline(0.0, lw=0.8, color="#BBBBBB")
            axs[1].step(centers, H, where="mid", lw=1.8, color="#2CA02C", label="H_final")
            if C is not None:
                axs2 = axs[1].twinx()
                axs2.bar(
                    centers,
                    C,
                    width=(edges[1] - edges[0]) * 0.9,
                    alpha=0.18,
                    align="center",
                    edgecolor="none",
                    color="#7F7F7F",
                    zorder=0,
                )
                axs2.set_ylabel("Samples per bin", fontsize=8, color="#7F7F7F")
                axs2.tick_params(axis="y", labelsize=8, colors="#7F7F7F")
            axs[1].set_title("64-bin residual histogram (rectified)")
            axs[1].set_xlabel("Q (Å⁻¹)")
            axs[1].set_ylabel("Bin mass (arb.)")
            axs[1].legend(loc="upper right", fontsize=9, framealpha=0.9)
            axs[1].grid(alpha=0.2, linewidth=0.6)

            fig.savefig(out, dpi=170)
            plt.close(fig)
            try:
                from plot_payload import save_plot_payload

                save_plot_payload(
                    str(out),
                    payload={
                        "plot_kind": "residual_bin_debug_v1",
                        "title": "Residual R(Q) + 64-bin histogram (debug)",
                        "x_label": "Q (Å⁻¹)",
                        "peak_mask_width": float(peak_mask_width),
                    },
                    arrays={
                        "Q": Q,
                        "R": R,
                        "edges": edges,
                        "centers": centers,
                        "H": H,
                        "C": C,
                        "observed_mask": bin_observed_mask.astype(np.int8),
                        "Q_main_peaks": Q_main_peaks,
                    },
                )
            except Exception as e:
                logger.debug(f"plot payload save failed: {e}")
            logger.debug(f"saved binned-residual plot → {out}")
        except Exception as e:
            logger.debug(f"plot failed: {e}")

    return H, bin_observed_mask, C


def _residual_hist_from_continuous(
    Q: np.ndarray,
    R: np.ndarray,
    Q_main_peaks: Optional[np.ndarray],
    edges: np.ndarray,
    sigma_bins: float,
    peak_mask_width: float = 0.015,
    debug_plot: bool = False,
) -> np.ndarray:
    """Backward-compatible wrapper returning only the final 64-bin residual histogram."""
    H, _, _ = _residual_hist_from_continuous_parts(
        Q,
        R,
        Q_main_peaks,
        edges,
        sigma_bins,
        peak_mask_width=peak_mask_width,
        debug_plot=debug_plot,
    )
    return H


# -----------------------------------------------------------------------------
# Stage-3 (ML): thin wrapper that reuses existing binning + calls ML module
# -----------------------------------------------------------------------------
def shortlist_by_hist_ML(
    Q: np.ndarray,
    R: np.ndarray,
    Q_main_peaks: np.ndarray,
    cand_ids: List[str],
    *,
    profiles_dir: str,
    ctx: Optional[dict] = None,
    topN: Optional[int] = None,
    # minimal guards; (<40 bins is fine — we only guard for no overlap)
    min_active_bins: int = 2,
    min_sum_residual: float = 0.0,
    # ML locations and knobs
    ml_components_dir: Optional[str] = None,
    model_variant: str = "ms64_mhsa2",
    model_ckpt: Optional[str] = None,  # explicit path overrides auto selection
    device: str = "cuda",
    batch_size: int = 512,
    fusion_alpha: float = 1.0,
    fusion_beta:  float = 0.2,
    fusion_cos:   float = 0.6,
    # plotting for parity
    plot: bool = True,
    plot_out_path_png: Optional[str] = None,
    plot_top_k: int = 24,
    plot_label_fn: Optional[Callable[[str], str]] = None,
    plot_title: str = "Stage-3 Histogram (ML)",
) -> Tuple[List[Tuple[str, float]], List[dict], dict]:
    """
    Drop-in ML screener. Returns (scored, details, meta) like the legacy function.
    Auto-selects ML checkpoint:
      - During Stage 0 (runner sets ML_IS_STAGE0=1), uses two_phase_training.pt
      - Otherwise uses residual_training.pt
      - An explicit model_ckpt path always overrides.
    Also prints a short debug line with the chosen checkpoint.
    """
    logger.info("Using ML HIST FILTER")
    # 1) DB pack / metadata
    if ctx is None:
        ctx = _load_profiles64_metadata(profiles_dir)
    profiles = ctx["profiles"].astype(np.float32)
    pid_to_row = ctx["pid_to_row"]
    q_min_db = float(ctx["q_min"]); q_max_db = float(ctx["q_max"])
    edges = ctx["edges"].astype(np.float64); centers = ctx["centers"].astype(np.float64)
    sigma = float(ctx.get("sigma_bins", 1.0))

    # 2) Residual → 64-bin (reuse your existing helper)
    Q = np.asarray(Q, dtype=np.float64); R = np.asarray(R, dtype=np.float64)
    if Q.ndim != 1 or R.ndim != 1 or Q.shape != R.shape:
        raise ValueError("Q and R must be 1D arrays of the same shape.")
    H_res, observed_mask, bin_counts = _residual_hist_from_continuous_parts(
        Q,
        R,
        Q_main_peaks,
        edges,
        sigma,
        debug_plot=False,
    )

    # 3) Active overlap with DB. Internal excluded gaps remain masked out.
    q_min_res = float(np.min(Q)) if Q.size else 0.0
    q_max_res = float(np.max(Q)) if Q.size else 0.0
    q_active_min = max(q_min_res, q_min_db)
    q_active_max = min(q_max_res, q_max_db)

    M_range = observed_mask.astype(bool)
    n_active = int(np.sum(M_range))
    if n_active < int(min_active_bins):
        meta = {
            "mode": "hist_ML",
            "q_min_db": q_min_db, "q_max_db": q_max_db,
            "q_min_res": q_min_res, "q_max_res": q_max_res,
            "active_range": (q_active_min, q_active_max),
            "active_bins": n_active,
            "fragmented_mask": bool(np.any(np.diff(np.flatnonzero(M_range)) > 1)) if n_active > 1 else False,
            "profiles_dir": profiles_dir,
        }
        return [], [], meta

    # 4) Call ML module
    import sys, os
    if not ml_components_dir:
        env_ml_dir = os.environ.get("ML_COMPONENTS_DIR")
        if env_ml_dir:
            ml_components_dir = env_ml_dir
        else:
            ml_components_dir = str(Path(__file__).resolve().parent.parent / "ML_components")

    ml_components_dir = str(Path(ml_components_dir).resolve())
    if ml_components_dir not in sys.path:
        sys.path.insert(0, ml_components_dir)
    from models import shortlist_ml_rank, DEFAULT_CKPT, CKPT_TWO_PHASE, CKPT_RESIDUAL  # from ML_components/models.py

    # Stage detection (no extra args needed):
    #   1) explicit model_ckpt overrides everything
    #   2) otherwise use ctx['stage']==0 if present
    #   3) otherwise use env ML_IS_STAGE0 in {1,true,yes}
    if model_ckpt:
        ckpt = model_ckpt
        ckpt_source = "explicit"
        stage0_flag = None
    else:
        # try ctx
        stage0_flag = None
        try:
            if ctx is not None and ("stage" in ctx):
                stage0_flag = (int(ctx["stage"]) == 0)
        except Exception:
            stage0_flag = None
        # fallback to env
        if stage0_flag is None:
            env_v = (os.environ.get("ML_IS_STAGE0") or "").strip().lower()
            stage0_flag = env_v in {"1", "true", "yes"}
        ckpt = CKPT_TWO_PHASE if stage0_flag else CKPT_RESIDUAL
        ckpt_source = "ctx.stage" if ("stage" in (ctx or {})) else "env.ML_IS_STAGE0"

    # Debug: print which checkpoint we’ll use
    try:
        logger.info(f"[ML-HIST] checkpoint='{os.path.basename(ckpt)}' source={ckpt_source} {('(stage0)' if (stage0_flag is True) else '(residual)' if (stage0_flag is False) else '(override)')}")
    except Exception:
        pass

    scored, details, meta_ml = shortlist_ml_rank(
        H_res=H_res,
        centers=centers,
        profiles=profiles,
        pid_to_row=pid_to_row,
        candidate_ids=cand_ids,
        mask_bool=M_range,
        q_active_min=q_active_min,
        q_active_max=q_active_max,
        topN=(int(topN) if topN is not None else None),
        variant=model_variant,
        ckpt_path=(ckpt if ckpt else DEFAULT_CKPT),
        device=device,
        batch_size=batch_size,
        fusion_alpha=fusion_alpha,
        fusion_beta=fusion_beta,
        fusion_cos=fusion_cos,
        plot=plot,
        plot_out_path_png=plot_out_path_png,
        plot_top_k=plot_top_k,
        plot_label_fn=plot_label_fn,
        plot_title=plot_title,
    )

    # 5) enrich meta with local info for parity
    meta = dict(meta_ml)
    meta.update({
        "q_min_db": q_min_db, "q_max_db": q_max_db,
        "q_min_res": q_min_res, "q_max_res": q_max_res,
        "active_range": (q_active_min, q_active_max),
        "active_bins": n_active,
        "fragmented_mask": bool(np.any(np.diff(np.flatnonzero(M_range)) > 1)) if n_active > 1 else False,
        "profiles_dir": profiles_dir,
        "sigma_bins": sigma,
        "sum_residual": float(np.maximum(H_res[M_range], 0.0).sum()),
        "observed_bin_count_sum": int(np.sum(bin_counts[M_range])),
        "is_stage0": (bool(stage0_flag) if stage0_flag is not None else None),
        "ckpt_source": ckpt_source,
    })
    return scored, details, meta

# =============================================================================
# Smoke-test
# =============================================================================
def test_histogram_screening() -> None:
    logger.info(
        "ratiofilter (continuous) ready: 64-bin DB loading, continuous residual binning, Stage-3 scoring."
    )


if __name__ == "__main__":
    test_histogram_screening()
