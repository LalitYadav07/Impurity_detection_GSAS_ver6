#!/usr/bin/env python3
"""
64-bin histogram screening for impurity detection (continuous residual version).

- Loads DB profiles (64-bin) from profiles64.npz + index.csv
- Builds residual histogram directly from continuous residual R(Q) by summing
  positive residual values per DB bin (ΔQ-weighted area per bin)
- Matches active Q-range with DB, min-max/peak scales within active bins
- Fits per-candidate scale α with p-weighted quantile cap
- Ranks by explained_fraction and purity; returns details & metadata
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


# =============================================================================
# DB pack loader
# =============================================================================
def _load_profiles64_metadata(profiles_dir: str) -> Dict[str, Any]:
    """
    Load 64-bin profiles and metadata from profiles64.npz (+ index.csv).

    Returns a dict with keys:
      profiles (N,64), pid_to_row, q_min, q_max, n_bins, sigma_bins, edges, centers
    """
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

    return {
        "profiles": profiles,
        "pid_to_row": pid_to_row,
        "q_min": q_min,
        "q_max": q_max,
        "n_bins": n_bins,
        "sigma_bins": sigma_bins,
        "edges": edges,
        "centers": centers,
    }







# =============================================================================
# Residual → 64-bin histogram (continuous, ΔQ-weighted)
# =============================================================================
def _residual_hist_from_continuous(
    Q: np.ndarray,
    R: np.ndarray,
    Q_main_peaks: Optional[np.ndarray],
    edges: np.ndarray,
    sigma_bins: float,
    peak_mask_width: float = 0.015,  # Q half-width to mask around each main-phase peak
    debug_plot: bool = False,        # set True or env RESID_BIN_DEBUG=1 to save a PNG
) -> np.ndarray:
    """
    Build a 64-bin residual histogram H from continuous R(Q) with:
      - sample-level masking near main-phase peaks,
      - signed *area* accumulation per bin (ΔQ-weighted, order-robust),
      - optional Gaussian smoothing in *bin units* (sigma_bins from DB),
      - late rectification (clip negatives to 0),
      - NO min–max scaling here (active-window normalization happens later).

    IMPORTANT: ΔQ uses absolute neighbor gaps, so ascending/descending Q both work.
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
        return np.zeros(max(n_bins, 0), dtype=np.float64)
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
        return np.zeros(n_bins, dtype=np.float64)

    Qm = Q[m]
    Rm = R[m]

    # 2) ΔQ weights (absolute gaps)
    dq = np.empty_like(Qm)
    if Qm.size == 1:
        dq[:] = 0.0
    else:
        dq[1:-1] = 0.5 * (np.abs(Qm[2:] - Qm[1:-1]) + np.abs(Qm[1:-1] - Qm[:-2]))
        dq[0] = np.abs(Qm[1] - Qm[0])
        dq[-1] = np.abs(Qm[-1] - Qm[-2])
    dq = np.maximum(dq, 0.0)

    # 3) Bin assignment: [e_i, e_{i+1}) with last bin inclusive
    idx = np.searchsorted(edges, Qm, side="right") - 1
    idx = np.clip(idx, 0, n_bins - 1)

    # 4) Signed AREA accumulation per bin (no per-bin divide)
    H = np.zeros(n_bins, dtype=np.float64)
    np.add.at(H, idx, Rm * dq)

    # Diagnostic counts per bin (used only for debug plot)
    C = None
    if debug_plot or os.environ.get("RESID_BIN_DEBUG", "") == "1":
        C = np.zeros(n_bins, dtype=np.int64)
        np.add.at(C, idx, 1)

    # 5) Optional Gaussian smoothing in BIN units
    if sigma_bins is not None and float(sigma_bins) > 0.0:
        sb = float(sigma_bins)
        W = int(np.ceil(3.0 * sb))  # ±3σ support
        if W > 0:
            x = np.arange(-W, W + 1, dtype=np.float64)
            ker = np.exp(-0.5 * (x / sb) ** 2)
            ker /= ker.sum()
            H = np.convolve(H, ker, mode="same")

    # 6) Late rectification (non-negative residual mass)
    H = np.maximum(H, 0.0)

    # 7) Debug plot (optional)
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
            print(f"[debug] saved binned-residual plot → {out}")
        except Exception as e:
            print(f"[debug] plot failed: {e}")

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
    ml_components_dir: str = "/netdisk/ly6/GSAS_Projects/Impurity_detection_GSAS_ver1 copy/ML_components",
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
    print("---------Using ML HIST FILTER--------")
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
    H_res = _residual_hist_from_continuous(Q, R, Q_main_peaks, edges, sigma, debug_plot=False)

    # 3) Active overlap with DB (allow small windows; just forbid empty)
    q_min_res = float(np.min(Q)) if Q.size else 0.0
    q_max_res = float(np.max(Q)) if Q.size else 0.0
    q_active_min = max(q_min_res, q_min_db)
    q_active_max = min(q_max_res, q_max_db)

    M_range = (centers >= q_active_min) & (centers <= q_active_max)
    n_active = int(np.sum(M_range))
    if n_active < int(min_active_bins):
        meta = {
            "mode": "hist_ML",
            "q_min_db": q_min_db, "q_max_db": q_max_db,
            "q_min_res": q_min_res, "q_max_res": q_max_res,
            "active_range": (q_active_min, q_active_max),
            "active_bins": n_active,
            "profiles_dir": profiles_dir,
        }
        return [], [], meta

    # 4) Call ML module
    import sys, os
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
        print(f"[ML-HIST] checkpoint='{os.path.basename(ckpt)}' "
              f"source={ckpt_source} "
              f"{'(stage0)' if (stage0_flag is True) else '(residual)' if (stage0_flag is False) else '(override)'}")
    except Exception:
        pass

    scored, details, meta_ml = shortlist_ml_rank(
        H_res=H_res,
        centers=centers,
        profiles=profiles,
        pid_to_row=pid_to_row,
        candidate_ids=cand_ids,
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
        "profiles_dir": profiles_dir,
        "sigma_bins": sigma,
        "sum_residual": float(np.maximum(H_res[M_range], 0.0).sum()),
        "is_stage0": (bool(stage0_flag) if stage0_flag is not None else None),
        "ckpt_source": ckpt_source,
    })
    return scored, details, meta

# =============================================================================
# Smoke-test
# =============================================================================
def test_histogram_screening() -> None:
    print(
        "ratiofilter (continuous) ready: 64-bin DB loading, continuous residual binning, Stage-3 scoring."
    )


if __name__ == "__main__":
    test_histogram_screening()
