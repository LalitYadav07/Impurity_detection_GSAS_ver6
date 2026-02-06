# ML_components/models.py
# Residual screener model (+ backward-compatible 3rd head) and inference utils.

from __future__ import annotations
import os, math
from functools import lru_cache
from typing import List, Tuple, Optional, Callable, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------
ML_ROOT = os.path.dirname(__file__)

# Explicit, stage-aware checkpoint names
CKPT_TWO_PHASE = os.path.join(ML_ROOT, "two_phase_training.pt")  # used during Stage 0
CKPT_RESIDUAL  = os.path.join(ML_ROOT, "residual_training.pt")   # used after Stage 0

# Backward-compatible default (later stages)
DEFAULT_CKPT   = CKPT_RESIDUAL

DEFAULT_VARIANT = "ms64_mhsa2"

# Fusion defaults (β weight will be auto-ignored if we disable β)
FUSION_DEFAULT = dict(alpha=1.0, beta=0.2, cos=0.6, explained=0.10)

# Classification gating defaults (used only if checkpoint has a cls head)
CLS_DEFAULT = dict(
    gate_mode="soft",       # "hard" | "soft"
    threshold=0.55,         # e.g., 0.53 if you want your best-F1 epoch threshold
    penalty=1e6,            # hard gate: subtract this if below threshold
    gamma=0.25                 # soft gate: score *= p^gamma
)

# --------------------------------------------------------------------
# Model (backward compatible: 2-head or 3-head)
# Input:  (B, 3, 64) channels=(residual_norm, candidate_norm, mask01)
# Output: alpha_hat, beta_hat in [0,1], and (optionally) cls_prob in [0,1]
# --------------------------------------------------------------------
class BaseHead(nn.Module):
    def __init__(self, embed: int, p_drop: float = 0.10, proj_dim: int = 64, with_cls: bool = True):
        super().__init__()
        self.post = nn.Sequential(nn.Linear(2*embed, embed), nn.GELU(), nn.Dropout(p_drop))
        self.head_alpha = nn.Sequential(nn.Linear(embed, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_beta  = nn.Sequential(nn.Linear(embed, 64), nn.GELU(), nn.Linear(64, 1))
        self.with_cls = bool(with_cls)
        if self.with_cls:
            self.head_cls = nn.Sequential(nn.Linear(embed, 64), nn.GELU(), nn.Linear(64, 1))
        self.projector  = nn.Linear(2*embed, proj_dim)  # unused at inference; kept for compatibility

    def forward_heads(self, h: torch.Tensor):
        a = torch.sigmoid(self.head_alpha(h))
        b = torch.sigmoid(self.head_beta(h))
        if self.with_cls:
            c_prob = torch.sigmoid(self.head_cls(h))   # probability
        else:
            c_prob = None
        return a, b, c_prob

class MultiScaleMHSA(nn.Module):
    def __init__(self, embed: int = 64, ks=(3,7,15), heads: int = 2, p_drop: float = 0.10, se_ratio: int = 8, with_cls: bool = True):
        super().__init__()
        n_br = len(ks); base = embed // n_br
        outs = [base]*n_br; outs[-1] = embed - base*(n_br-1)
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(3, o, k, padding=k//2, bias=False), nn.BatchNorm1d(o), nn.GELU())
            for k, o in zip(ks, outs)
        ])
        self.fuse = nn.Sequential(nn.Conv1d(embed, embed, 1, bias=False), nn.BatchNorm1d(embed), nn.GELU(), nn.Dropout(p_drop))
        se_h = max(8, embed//se_ratio)
        self.se1, self.se2 = nn.Linear(embed, se_h), nn.Linear(se_h, embed)
        self.mhsa = nn.MultiheadAttention(embed_dim=embed, num_heads=heads, batch_first=True)
        self.ln = nn.LayerNorm(embed)
        self.base = BaseHead(embed, p_drop, with_cls=with_cls)

    @staticmethod
    def _masked_mean_pool(z: torch.Tensor, mL: torch.Tensor) -> torch.Tensor:
        w = mL.float()
        return (z * w.unsqueeze(-1)).sum(1) / torch.clamp(w.sum(1, keepdim=True), min=1e-6)

    @staticmethod
    def _masked_cand_weights(cand: torch.Tensor, mL: torch.Tensor) -> torch.Tensor:
        win = cand[:,0,:].masked_fill(~mL, 0.0)
        wmax = torch.amax(win, 1, keepdim=True)
        return (win / torch.clamp(wmax, min=1e-8)).unsqueeze(-1)

    def encode_pair(self, mix: torch.Tensor, cand: torch.Tensor, mask: torch.Tensor):
        x = torch.cat([mix, cand, mask], 1)              # (B,3,L)
        z = torch.cat([br(x) for br in self.branches], 1)
        z = self.fuse(z)
        s = torch.sigmoid(self.se2(F.gelu(self.se1(z.mean(-1)))))  # S/E gating
        z = (z * s.unsqueeze(-1)).transpose(1, 2)        # (B,L,E)
        attn, _ = self.mhsa(z, z, z)
        z = self.ln(z + attn)
        mL = mask.squeeze(1) > 0.5
        mean_pool = self._masked_mean_pool(z, mL)
        cand_pool = (z * self._masked_cand_weights(cand, mL)).sum(1)
        h_cat = torch.cat([mean_pool, cand_pool], 1)
        h = self.base.post(h_cat)
        emb = F.normalize(self.base.projector(h_cat), 1)  # kept for compatibility
        return h, emb

    def forward(self, x: torch.Tensor):
        mix, cand, mask = x[:,0:1,:], x[:,1:2,:], x[:,2:3,:]
        h, _ = self.encode_pair(mix, cand, mask)
        return self.base.forward_heads(h)  # -> (alpha(0..1), beta(0..1), cls_prob or None)

# Expose same variant name
VARIANTS: Dict[str, Tuple[nn.Module, dict]] = {
    "ms64_mhsa2": (MultiScaleMHSA, dict(embed=64, ks=(3,7,15), heads=2, p_drop=0.10)),
}

# --------------------------------------------------------------------
# Inference utilities
# --------------------------------------------------------------------
def window_norm(vec64: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    out = np.zeros_like(vec64, dtype=np.float32)
    if mask_bool.any():
        win = vec64[mask_bool].astype(np.float32, copy=False)
        m = float(win.max())
        if m > 0: win = win / (m + 1e-8)
        out[mask_bool] = win
    return out

def masked_cosine_vec(residual64: np.ndarray, candNx64: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
    """
    Cosine(residual, candidate) computed ONLY on the active-window mask.
    Both residual and candidates are peak-normed (within mask) and L2-normalized.
    """
    m = mask_bool.astype(bool).ravel()
    if not np.any(m):
        return np.zeros((candNx64.shape[0],), dtype=np.float32)

    # residual → peak-norm (within mask) then L2
    a = residual64[m].astype(np.float32)
    a_max = float(a.max())
    if a_max > 0:
        a = a / (a_max + 1e-8)
    an = a / (np.linalg.norm(a) + 1e-8)  # (L,)

    # candidates → peak-norm (within mask) then L2 row-wise
    B = candNx64[:, m].astype(np.float32)         # (N, L)
    if B.shape[1] == 0:
        return np.zeros((candNx64.shape[0],), dtype=np.float32)
    Bmax = B.max(axis=1, keepdims=True)           # (N,1)
    B = B / (np.where(Bmax > 0, Bmax, 1.0) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)  # (N,L)

    return (Bn @ an).astype(np.float32)           # (N,)

@lru_cache(maxsize=2)
def load_ml_model(variant: str = DEFAULT_VARIANT, ckpt_path: str = DEFAULT_CKPT, device: str = "cuda"):
    """
    Auto-detect whether the checkpoint has a classification head.
    Builds the model accordingly and loads with strict=False.
    Returns (model, device_str, meta_dict).
    """

    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'")
    builder, kwargs = VARIANTS[variant]

    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    state = ckpt.get("model", ckpt)

    # Detect classification head in checkpoint keys
    has_cls = any(k.startswith("base.head_cls") or ".head_cls." in k for k in state.keys())

    model = builder(with_cls=has_cls, **kwargs).to(dev)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    meta = dict(has_cls=has_cls, missing_keys=[str(k) for k in missing], unexpected_keys=[str(k) for k in unexpected])
    print(f"[ML] load_ml_model: variant='{variant}', device='{dev}', ckpt='{ckpt_path}'")

    return model, dev, meta

# --------------------------------------------------------------------
# Public entry — used by ratio_filter.shortlist_by_hist_ML
# --------------------------------------------------------------------
def shortlist_ml_rank(
    *,
    H_res: np.ndarray,
    centers: np.ndarray,
    profiles: np.ndarray,
    pid_to_row: Dict[str, int],
    candidate_ids: List[str],
    q_active_min: float,
    q_active_max: float,
    topN: Optional[int] = None,
    variant: str = DEFAULT_VARIANT,
    ckpt_path: str = DEFAULT_CKPT,
    device: str = "cuda",
    batch_size: int = 512,
    fusion_alpha: float = FUSION_DEFAULT["alpha"],
    fusion_beta:  float = FUSION_DEFAULT["beta"],
    fusion_cos:   float = FUSION_DEFAULT["cos"],
    fusion_expl:  float = FUSION_DEFAULT["explained"],
    # classification gating (used only if checkpoint has a cls head)
    cls_gate_mode: str = CLS_DEFAULT["gate_mode"],   # "hard" | "soft"
    cls_threshold: float = CLS_DEFAULT["threshold"],
    cls_penalty: float = CLS_DEFAULT["penalty"],
    cls_gamma: float = CLS_DEFAULT["gamma"],
    # NEW: control β usage in score; default is auto (disable if classifier exists)
    use_beta_in_score: Optional[bool] = None,
    plot: bool = False,
    plot_out_path_png: Optional[str] = None,
    plot_top_k: int = 12,
    plot_label_fn: Optional[Callable[[str], str]] = None,
    plot_title: str = "Stage-3 Histogram (ML)",
) -> Tuple[List[Tuple[str, float]], List[dict], dict]:

    assert H_res.shape[-1] == 64 and centers.shape[-1] == 64
    M = (centers >= float(q_active_min)) & (centers <= float(q_active_max))
    n_active = int(np.sum(M))
    y = np.maximum(H_res[M], 0.0).astype(np.float32)
    if n_active == 0 or y.max() <= 0:
        return [], [], dict(mode="hist_ML", active_bins=int(n_active), n_candidates=len(candidate_ids))

    # Residual: peak-norm inside mask, keep 64 layout
    Rn_full = np.zeros(64, dtype=np.float32)
    Rn_full[M] = (y / (float(y.max()) + 1e-8)).astype(np.float32)
    mask_bool = M.astype(bool)

    # Collect candidates (peak-normed in the same active window)
    rows, valid_ids = [], []
    for pid in candidate_ids:
        r = pid_to_row.get(str(pid))
        if r is None: continue
        Cn = window_norm(profiles[r], mask_bool)     # 64-bin, peak-normed within mask
        if not np.any(Cn[mask_bool] > 0): continue
        rows.append(Cn); valid_ids.append(str(pid))
    if not rows:
        return [], [], dict(mode="hist_ML", active_bins=int(n_active), n_candidates=len(candidate_ids), n_kept=0)

    C = np.stack(rows, 0).astype(np.float32)                                # (N,64)
    X = np.stack([np.broadcast_to(Rn_full,(C.shape[0],64)), C, np.broadcast_to(mask_bool.astype(np.float32),(C.shape[0],64))], 1)  # (N,3,64)

    # Model
    model, dev, meta = load_ml_model(variant=variant, ckpt_path=ckpt_path, device=device)
    has_cls = bool(meta.get("has_cls", False))

    # Decide β usage automatically unless explicitly overridden:
    # - If classifier exists (new checkpoint), default to NOT using β in fusion score.
    # - If no classifier (legacy 2-head), default to using β.
    if use_beta_in_score is None:
        use_beta = not has_cls
    else:
        use_beta = bool(use_beta_in_score)

    # Inference (batched)
    with torch.no_grad():
        xb = torch.from_numpy(X).to(dev, non_blocking=True)
        if xb.shape[0] > batch_size:
            chunks = []
            for s in range(0, xb.shape[0], batch_size):
                a_hat, b_hat, c_prob = model(xb[s:s+batch_size])
                a = a_hat.squeeze(1).detach().cpu().numpy().astype(np.float32)
                b = b_hat.squeeze(1).detach().cpu().numpy().astype(np.float32)
                if has_cls and (c_prob is not None):
                    c = c_prob.squeeze(1).detach().cpu().numpy().astype(np.float32)
                else:
                    c = np.ones_like(a, dtype=np.float32)
                chunks.append((a, b, c))
            alpha = np.concatenate([t[0] for t in chunks], 0)
            beta  = np.concatenate([t[1] for t in chunks], 0)
            pcls  = np.concatenate([t[2] for t in chunks], 0)
        else:
            a_hat, b_hat, c_prob = model(xb)
            alpha = a_hat.squeeze(1).detach().cpu().numpy().astype(np.float32)
            beta  = b_hat.squeeze(1).detach().cpu().numpy().astype(np.float32)
            if has_cls and (c_prob is not None):
                pcls = c_prob.squeeze(1).detach().cpu().numpy().astype(np.float32)
            else:
                pcls = np.ones_like(alpha, dtype=np.float32)

    # Similarity and explained fraction
    cos = masked_cosine_vec(Rn_full, C, mask_bool)
    R_act = Rn_full[mask_bool]; C_act = C[:, mask_bool]
    explained = np.minimum(alpha[:,None]*C_act, R_act[None,:]).sum(1) / (np.maximum(R_act.sum(), 1e-8))

    # Base score (optionally drop β)
    score_base = (FUSION_DEFAULT["alpha"] * alpha) \
               + ((FUSION_DEFAULT["beta"] * beta) if use_beta else 0.0) \
               + ((FUSION_DEFAULT["cos"] * cos) if use_beta else 0.0) \
               + ((FUSION_DEFAULT["explained"] * explained) if use_beta else 0.0)

    # Classification gating (only if cls head exists)
    if has_cls:
        if cls_gate_mode.lower() == "hard":
            bad = (pcls < float(cls_threshold))
            score = score_base.copy()
            score[bad] = score[bad] - float(cls_penalty)
        else:
            score = score_base * (np.power(np.clip(pcls, 1e-6, 1.0), float(cls_gamma)))
    else:
        score = score_base

    # Assemble details
    details = []
    for pid, a, b, c, co, ex, sc in zip(valid_ids, alpha, beta, pcls, cos, explained, score):
        d = {
            "phase_id": pid,
            "ok": True, "pass": True,
            "alpha": float(a),
            "beta": float(b),
            "beta_used": bool(use_beta),
            "present_prob": float(c),
            "cosine": float(co),
            "explained_fraction": float(ex),
            "score_base": float((FUSION_DEFAULT["alpha"]*a)
                                + ((FUSION_DEFAULT["beta"]*b) if use_beta else 0.0)
                                + (FUSION_DEFAULT["cos"]*co)
                                + (FUSION_DEFAULT["explained"]*ex)),
            "score": float(sc),
            "active_bins": int(n_active),
        }
        details.append(d)

    details.sort(key=lambda d: d["score"], reverse=True)
    if topN and topN > 0:
        keep = set([d["phase_id"] for d in details[:topN]])
        details = [d for d in details if d["phase_id"] in keep]
    scored = [(d["phase_id"], float(d["score"])) for d in details]

    meta_out = dict(
        mode="hist_ML",
        active_bins=int(n_active),
        q_active_min=float(q_active_min), q_active_max=float(q_active_max),
        n_candidates=len(candidate_ids), n_kept=len(details),
        fusion=dict(alpha=FUSION_DEFAULT["alpha"], beta=(FUSION_DEFAULT["beta"] if use_beta else 0.0),
                    cos=FUSION_DEFAULT["cos"], explained=FUSION_DEFAULT["explained"]),
        cls=dict(enabled=has_cls, gate_mode=cls_gate_mode, threshold=cls_threshold, gamma=cls_gamma, penalty=cls_penalty),
        beta_used=bool(use_beta),
        variant=variant, ckpt=ckpt_path,
        loader_meta=meta
    )

    # Optional diagnostics plot (title reflects gating & β usage)
    if plot and len(details) > 0:
        try:
            import matplotlib.pyplot as plt
            x = centers[mask_bool]; y_res = R_act
            n_show = min(len(details), max(1, int(plot_top_k)))
            cols = 2 if n_show <= 4 else 3
            rows = int(math.ceil(n_show/cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols*6.0, rows*3.6), squeeze=False)
            axes = axes.ravel()
            def _lbl(pid: str):
                s = pid if plot_label_fn is None else plot_label_fn(pid)
                return s if len(s) <= 48 else (s[:47] + "…")
            ymax = max(1.0, float(y_res.max()))
            for ax, d in zip(axes, details[:n_show]):
                r = pid_to_row.get(d["phase_id"])
                Cn_full = window_norm(profiles[r], mask_bool)
                y_c = d["alpha"] * Cn_full[mask_bool]
                ymax = max(ymax, float(np.max(y_c)))
                ax.step(x, y_res, where="mid", lw=1.8, color="#111"); ax.step(x, y_c, where="mid", lw=1.8, color="#1F77B4")
                ax.fill_between(x, 0, np.minimum(y_c, y_res), step="mid", color="#2CA02C", alpha=0.25)
                ax.set_xlim(x[0], x[-1]); ax.set_ylim(0, ymax*1.05)
                ax.grid(alpha=0.35, linewidth=0.7, color="#CCC")
                title_tail = f" p={d.get('present_prob',1.0):.2f}"
                ax.set_title(f"{_lbl(d['phase_id'])}\nscore={d['score']:.3f} α={d['alpha']:.3f} cos={d['cosine']:.3f}{title_tail}")
                ax.set_xlabel("Q (Å⁻¹)"); ax.set_ylabel("Norm. Intensity")
            for j in range(n_show, len(axes)): axes[j].axis("off")
            out_png = plot_out_path_png or "diag_hist_grid_ml.png"
            beta_tag = "β:ON" if use_beta else "β:OFF"
            gate_tag = f"{cls_gate_mode}{'@'+str(cls_threshold) if has_cls and cls_gate_mode=='hard' else ''}" if has_cls else "no-cls"
            fig.suptitle(f"{plot_title}  |  gate={gate_tag}  |  {beta_tag}", fontsize=12)
            fig.tight_layout(); fig.savefig(out_png, dpi=180); plt.close(fig)
        except Exception as e:
            print(f"[ml-plot] warning: {e}")

    return scored, details, meta_out
