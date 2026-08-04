"""Magnetic residual indexing precheck for RADAR-PD.

This module does not solve a magnetic structure. It asks a narrower question:
after a nuclear/main-phase refinement, are the positive residual peaks
significantly better indexed by one commensurate propagation vector than by
random peak combs?
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import nnls
except Exception:  # pragma: no cover - scipy is available in the app env
    nnls = None

try:
    from pymatgen.core import Lattice, Structure
except Exception:  # pragma: no cover - import error handled at runtime
    Lattice = None  # type: ignore[assignment]
    Structure = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MagneticPrecheckConfig:
    enabled: bool = False
    q_max: float = 6.0
    max_hkl: int = 8
    denominators: Tuple[int, ...] = (2, 3)
    width_grid: Tuple[float, ...] = (0.025, 0.04, 0.065)
    pseudo_voigt_eta: float = 0.35
    max_k_vectors: int = 80
    max_positions_per_k: int = 260
    top_residual_peaks: int = 8
    null_trials: int = 48
    random_seed: int = 13
    include_gamma: bool = False


def _as_config(raw: Optional[Dict[str, Any]]) -> MagneticPrecheckConfig:
    raw = dict(raw or {})

    def _tuple_numbers(key: str, default: Tuple[float, ...], cast=float):
        value = raw.get(key, default)
        if value is None:
            return default
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        try:
            out = tuple(cast(v) for v in value)
        except Exception:
            return default
        return out or default

    return MagneticPrecheckConfig(
        enabled=bool(raw.get("enabled", False)),
        q_max=float(raw.get("q_max", 6.0)),
        max_hkl=int(raw.get("max_hkl", 8)),
        denominators=tuple(int(v) for v in _tuple_numbers("denominators", (2, 3), int)),
        width_grid=tuple(float(v) for v in _tuple_numbers("width_grid", (0.025, 0.04, 0.065), float)),
        pseudo_voigt_eta=float(raw.get("pseudo_voigt_eta", 0.35)),
        max_k_vectors=int(raw.get("max_k_vectors", 80)),
        max_positions_per_k=int(raw.get("max_positions_per_k", 260)),
        top_residual_peaks=int(raw.get("top_residual_peaks", 8)),
        null_trials=int(raw.get("null_trials", 48)),
        random_seed=int(raw.get("random_seed", 13)),
        include_gamma=bool(raw.get("include_gamma", False)),
    )


def _lattice_from_gpx(gpx_path: Optional[str], preferred_phase: Optional[str] = None):
    if not gpx_path:
        return None
    try:
        from GSASII import GSASIIscriptable as G2sc

        project = G2sc.G2Project(gpxfile=str(gpx_path))
        phases = project.phases()
        if not phases:
            return None
        phase = None
        if preferred_phase:
            for candidate in phases:
                if str(getattr(candidate, "name", "")) == str(preferred_phase):
                    phase = candidate
                    break
        phase = phase or phases[0]
        pdata = getattr(phase, "data", {}) or {}
        cell = ((pdata.get("General") or {}).get("Cell") or [])
        # GSAS-II stores [refineFlag, a, b, c, alpha, beta, gamma, volume].
        if len(cell) >= 7 and Lattice is not None:
            return Lattice.from_parameters(
                float(cell[1]),
                float(cell[2]),
                float(cell[3]),
                float(cell[4]),
                float(cell[5]),
                float(cell[6]),
            )
    except Exception:
        return None
    return None


def _lattice_from_cif(cif_path: Optional[str]):
    if not cif_path or Structure is None:
        return None
    try:
        return Structure.from_file(str(cif_path)).lattice
    except Exception:
        return None


def _reciprocal_metric_no_2pi(lattice) -> np.ndarray:
    # Pymatgen's reciprocal_lattice includes 2*pi. Crystallographic reciprocal
    # lattice does not, so |h a* + k b* + l c*| = 1/d and Q = 2*pi/d.
    return np.asarray(lattice.reciprocal_lattice_crystallographic.metric_tensor, dtype=float)


def _canonical_component(value: float) -> float:
    value = float(value) % 1.0
    if value > 0.5:
        value = 1.0 - value
    if abs(value) < 1e-12:
        value = 0.0
    return round(value, 8)


def generate_k_vectors(denominators: Sequence[int], *, include_gamma: bool = False, max_vectors: int = 80) -> List[Tuple[float, float, float]]:
    vectors = set()
    if include_gamma:
        vectors.add((0.0, 0.0, 0.0))
    for denom in sorted({int(d) for d in denominators if int(d) > 1}):
        vals = [_canonical_component(n / float(denom)) for n in range(0, denom)]
        vals = sorted(set(vals))
        for h in vals:
            for k in vals:
                for l in vals:
                    vec = (h, k, l)
                    if vec == (0.0, 0.0, 0.0) and not include_gamma:
                        continue
                    vectors.add(vec)
    ordered = sorted(vectors, key=lambda v: (sum(1 for x in v if x), sum(abs(x) for x in v), v))
    return ordered[: max(1, int(max_vectors))]


def _merge_positions(q_positions: np.ndarray, *, min_sep: float, max_positions: int) -> np.ndarray:
    q_positions = np.asarray(q_positions, dtype=float)
    q_positions = np.sort(q_positions[np.isfinite(q_positions) & (q_positions > 0)])
    if q_positions.size == 0:
        return q_positions
    merged = [float(q_positions[0])]
    for q in q_positions[1:]:
        if float(q) - merged[-1] >= min_sep:
            merged.append(float(q))
        else:
            merged[-1] = 0.5 * (merged[-1] + float(q))
    arr = np.asarray(merged, dtype=float)
    if arr.size <= max_positions:
        return arr
    # Keep a deterministic low-Q-biased subset. Magnetic form factors decay with Q,
    # so low-Q positions are more diagnostic for this precheck.
    return arr[:max_positions]


def satellite_q_positions(
    lattice,
    k_vector: Tuple[float, float, float],
    *,
    q_min: float,
    q_max: float,
    max_hkl: int,
    min_sep: float,
    max_positions: int,
) -> np.ndarray:
    Gs = _reciprocal_metric_no_2pi(lattice)
    vals = np.arange(-int(max_hkl), int(max_hkl) + 1, dtype=float)
    hh, kk, ll = np.meshgrid(vals, vals, vals, indexing="ij")
    base = np.column_stack([hh.ravel(), kk.ravel(), ll.ravel()])
    kv = np.asarray(k_vector, dtype=float).reshape(1, 3)
    vecs = base + kv
    nonzero = np.linalg.norm(vecs, axis=1) > 1e-10
    vecs = vecs[nonzero]
    dinv2 = np.einsum("ni,ij,nj->n", vecs, Gs, vecs)
    ok = np.isfinite(dinv2) & (dinv2 > 0)
    q = 2.0 * math.pi * np.sqrt(np.maximum(dinv2[ok], 0.0))
    q = q[(q >= float(q_min)) & (q <= float(q_max))]
    return _merge_positions(q, min_sep=float(min_sep), max_positions=int(max_positions))


def _moving_average(y: np.ndarray, width: int = 5) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    width = max(1, int(width))
    if width <= 1 or y.size < width:
        return y.copy()
    pad = width // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(width, dtype=float) / float(width)
    return np.convolve(ypad, kernel, mode="valid")[: y.size]


def _detect_residual_peaks(q: np.ndarray, y: np.ndarray, *, top_n: int) -> Tuple[np.ndarray, np.ndarray]:
    ys = _moving_average(y, 5)
    if ys.size < 3:
        return np.array([]), np.array([])
    local = (ys[1:-1] > ys[:-2]) & (ys[1:-1] >= ys[2:])
    idx = np.nonzero(local)[0] + 1
    if idx.size == 0:
        return np.array([]), np.array([])
    med = float(np.median(ys))
    mad = float(np.median(np.abs(ys - med)) + 1e-12)
    idx = idx[(ys[idx] - med) >= 3.0 * mad]
    if idx.size == 0:
        idx = np.argsort(ys)[-min(top_n, ys.size):]
    idx = idx[np.argsort(ys[idx])[::-1]][: max(1, int(top_n))]
    order = np.argsort(q[idx])
    idx = idx[order]
    return q[idx], y[idx]


def _design_matrix(q: np.ndarray, positions: np.ndarray, fwhm: float, eta: float) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1, 1)
    pos = np.asarray(positions, dtype=float).reshape(1, -1)
    if q.size == 0 or pos.size == 0:
        return np.zeros((q.shape[0], 0), dtype=float)
    fwhm = max(float(fwhm), 1e-6)
    eta = min(1.0, max(0.0, float(eta)))
    sigma = fwhm / 2.354820045
    gamma = fwhm / 2.0
    gaussian = np.exp(-0.5 * ((q - pos) / max(sigma, 1e-8)) ** 2)
    lorentzian = 1.0 / (1.0 + ((q - pos) / max(gamma, 1e-8)) ** 2)
    A = (1.0 - eta) * gaussian + eta * lorentzian
    colmax = np.maximum(A.max(axis=0), 1e-12)
    return A / colmax


def _fit_nonnegative(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    if A.size == 0:
        return np.zeros(0, dtype=float)
    if nnls is not None:
        coef, _ = nnls(A, y)
        return np.asarray(coef, dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.maximum(np.asarray(coef, dtype=float), 0.0)


def _score_positions(
    q: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    residual_peak_q: np.ndarray,
    residual_peak_y: np.ndarray,
    *,
    fwhm: float,
    eta: float,
) -> Dict[str, Any]:
    if positions.size == 0 or q.size == 0:
        return {
            "score": 0.0,
            "explained_fraction": 0.0,
            "active_peaks": 0,
            "supported_top_peaks": 0,
            "support_fraction": 0.0,
            "fit": np.zeros_like(y),
            "coefficients": np.zeros(0),
        }
    low_q_weight = 1.0 / (1.0 + (q / 4.0) ** 2)
    A = _design_matrix(q, positions, fwhm, eta)
    Aw = A * np.sqrt(low_q_weight[:, None])
    yw = y * np.sqrt(low_q_weight)
    coef = _fit_nonnegative(Aw, yw)
    fit = A @ coef
    sse = float(np.sum(low_q_weight * (y - fit) ** 2))
    null_sse = float(np.sum(low_q_weight * y**2) + 1e-12)
    explained = max(0.0, min(1.0, 1.0 - sse / null_sse))
    if coef.size:
        active = int(np.sum(coef > max(float(np.max(coef)) * 0.04, 1e-8)))
    else:
        active = 0
    if residual_peak_q.size:
        tol = max(float(fwhm), 0.035)
        dist = np.min(np.abs(residual_peak_q.reshape(-1, 1) - positions.reshape(1, -1)), axis=1)
        matched = dist <= tol
        supported = int(np.sum(matched))
        support_fraction = float(np.sum(residual_peak_y[matched]) / max(np.sum(residual_peak_y), 1e-12))
    else:
        supported = 0
        support_fraction = 0.0
    density_penalty = 0.04 * min(1.0, positions.size / 180.0)
    active_penalty = 0.008 * min(active, 20)
    score = float(0.68 * explained + 0.32 * support_fraction - density_penalty - active_penalty)
    return {
        "score": score,
        "explained_fraction": float(explained),
        "active_peaks": active,
        "supported_top_peaks": supported,
        "support_fraction": float(support_fraction),
        "fit": fit,
        "coefficients": coef,
    }


def _best_width_score(
    q: np.ndarray,
    y: np.ndarray,
    positions: np.ndarray,
    peak_q: np.ndarray,
    peak_y: np.ndarray,
    widths: Sequence[float],
    *,
    eta: float,
) -> Dict[str, Any]:
    best: Optional[Dict[str, Any]] = None
    for fwhm in widths:
        scored = _score_positions(q, y, positions, peak_q, peak_y, fwhm=float(fwhm), eta=eta)
        scored["fwhm_q"] = float(fwhm)
        if best is None or float(scored["score"]) > float(best["score"]):
            best = scored
    return best or {"score": 0.0, "fit": np.zeros_like(y), "coefficients": np.zeros(0), "fwhm_q": float(widths[0])}


def _null_scores(
    q: np.ndarray,
    y: np.ndarray,
    n_positions: int,
    peak_q: np.ndarray,
    peak_y: np.ndarray,
    cfg: MagneticPrecheckConfig,
    *,
    q_min: float,
    q_max: float,
) -> List[float]:
    rng = np.random.default_rng(int(cfg.random_seed))
    scores: List[float] = []
    n_positions = max(8, min(int(n_positions), int(cfg.max_positions_per_k)))
    for _ in range(max(0, int(cfg.null_trials))):
        positions = np.sort(rng.uniform(float(q_min), float(q_max), size=n_positions))
        positions = _merge_positions(positions, min_sep=min(cfg.width_grid) * 0.55, max_positions=n_positions)
        scored = _best_width_score(q, y, positions, peak_q, peak_y, cfg.width_grid, eta=cfg.pseudo_voigt_eta)
        scores.append(float(scored["score"]))
    return scores


def _evidence_label(best: Dict[str, Any], null_scores: Sequence[float]) -> Tuple[str, str]:
    if not best:
        return "not_available", "No candidate k-vector could be scored."
    score = float(best.get("score", 0.0))
    explained = float(best.get("explained_fraction", 0.0))
    supported = int(best.get("supported_top_peaks", 0))
    if null_scores:
        null = np.asarray(null_scores, dtype=float)
        percentile = float(np.mean(null <= score))
        p95 = float(np.quantile(null, 0.95))
    else:
        percentile = 0.0
        p95 = 0.0
    if percentile >= 0.98 and score >= p95 + 0.025 and explained >= 0.18 and supported >= 3:
        return "strong", "Best k-vector is well separated from randomized peak-comb fits."
    if percentile >= 0.90 and explained >= 0.10 and supported >= 2:
        return "moderate", "Best k-vector is better than most randomized peak-comb fits, but should be checked against impurities."
    return "weak", "No k-vector is clearly better than randomized peak-comb fits."


def _write_plot(
    out_path: Path,
    q: np.ndarray,
    y: np.ndarray,
    fit: np.ndarray,
    positions: np.ndarray,
    best: Dict[str, Any],
    label: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(q, y, ".", ms=2.2, color="#263238", alpha=0.62, label="Positive residual")
    ax.plot(q, fit, "-", lw=1.4, color="#c62828", label="Best magnetic comb fit")
    ymax = float(max(np.nanmax(y) if y.size else 1.0, np.nanmax(fit) if fit.size else 1.0, 1e-6))
    for pos in positions[:140]:
        ax.vlines(float(pos), 0.0, 0.08 * ymax, color="#1565c0", lw=0.75, alpha=0.30)
    ax.set_title(
        f"Magnetic residual indexing: {label} | k={best.get('k_label', '-')}"
    )
    ax.set_xlabel("Q (1/A)")
    ax.set_ylabel("Positive residual intensity")
    ax.grid(True, color="#e6ece8", lw=0.6)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_magnetic_precheck(
    *,
    q: Sequence[float],
    residual: Sequence[float],
    main_cif: Optional[str],
    out_dir: str | Path,
    config: Optional[Dict[str, Any]] = None,
    refined_gpx: Optional[str] = None,
    phase_name: Optional[str] = None,
) -> Dict[str, Any]:
    cfg = _as_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if not cfg.enabled:
        summary = {"enabled": False, "status": "skipped", "reason": "Magnetic precheck disabled."}
        (out / "magnetic_precheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    lattice = _lattice_from_gpx(refined_gpx, phase_name) or _lattice_from_cif(main_cif)
    if lattice is None:
        summary = {
            "enabled": True,
            "status": "skipped",
            "evidence": "not_available",
            "reason": "A readable main-phase lattice is required for magnetic k-vector indexing.",
        }
        (out / "magnetic_precheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    q_arr = np.asarray(q, dtype=float).ravel()
    y_arr = np.maximum(np.asarray(residual, dtype=float).ravel(), 0.0)
    n = min(q_arr.size, y_arr.size)
    q_arr, y_arr = q_arr[:n], y_arr[:n]
    mask = np.isfinite(q_arr) & np.isfinite(y_arr) & (q_arr > 0.0) & (q_arr <= float(cfg.q_max))
    q_arr, y_arr = q_arr[mask], y_arr[mask]
    if q_arr.size < 25 or float(np.nanmax(y_arr) if y_arr.size else 0.0) <= 0:
        summary = {
            "enabled": True,
            "status": "complete",
            "evidence": "weak",
            "reason": "Residual has too little positive low-Q signal for magnetic indexing.",
            "points": int(q_arr.size),
            "seconds": float(time.perf_counter() - t0),
        }
        (out / "magnetic_precheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    order = np.argsort(q_arr)
    q_arr, y_arr = q_arr[order], y_arr[order]
    scale = float(np.nanmax(y_arr))
    if scale > 0:
        y_fit = y_arr / scale
    else:
        y_fit = y_arr.copy()
    q_min, q_max = float(np.min(q_arr)), float(np.max(q_arr))
    peak_q, peak_y = _detect_residual_peaks(q_arr, y_fit, top_n=cfg.top_residual_peaks)
    k_vectors = generate_k_vectors(cfg.denominators, include_gamma=cfg.include_gamma, max_vectors=cfg.max_k_vectors)

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_positions = np.array([], dtype=float)
    for kv in k_vectors:
        positions = satellite_q_positions(
            lattice,
            kv,
            q_min=q_min,
            q_max=q_max,
            max_hkl=cfg.max_hkl,
            min_sep=min(cfg.width_grid) * 0.45,
            max_positions=cfg.max_positions_per_k,
        )
        scored = _best_width_score(q_arr, y_fit, positions, peak_q, peak_y, cfg.width_grid, eta=cfg.pseudo_voigt_eta)
        row = {
            "k_h": float(kv[0]),
            "k_k": float(kv[1]),
            "k_l": float(kv[2]),
            "k_label": f"({kv[0]:g}, {kv[1]:g}, {kv[2]:g})",
            "score": float(scored["score"]),
            "explained_fraction": float(scored["explained_fraction"]),
            "support_fraction": float(scored["support_fraction"]),
            "supported_top_peaks": int(scored["supported_top_peaks"]),
            "active_peaks": int(scored["active_peaks"]),
            "candidate_positions": int(positions.size),
            "fwhm_q": float(scored["fwhm_q"]),
        }
        rows.append(row)
        if best is None or row["score"] > float(best.get("score", -1e9)):
            best = {**row, "fit": np.asarray(scored["fit"], dtype=float), "coefficients": np.asarray(scored["coefficients"], dtype=float)}
            best_positions = positions

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i

    null = _null_scores(
        q_arr,
        y_fit,
        int(best.get("candidate_positions", 40)) if best else 40,
        peak_q,
        peak_y,
        cfg,
        q_min=q_min,
        q_max=q_max,
    )
    evidence, reason = _evidence_label(best or {}, null)
    best_score = float(best.get("score", 0.0)) if best else 0.0
    null_percentile = float(np.mean(np.asarray(null) <= best_score)) if null else 0.0
    null_p95 = float(np.quantile(np.asarray(null), 0.95)) if null else 0.0

    rankings_csv = out / "magnetic_k_vector_rankings.csv"
    if rows:
        fieldnames = ["rank", "k_label", "k_h", "k_k", "k_l", "score", "explained_fraction", "support_fraction", "supported_top_peaks", "active_peaks", "candidate_positions", "fwhm_q"]
        with rankings_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    peak_rows = []
    if peak_q.size:
        for q0, yy in zip(peak_q, peak_y):
            if best_positions.size:
                nearest = float(best_positions[np.argmin(np.abs(best_positions - q0))])
                delta = abs(nearest - float(q0))
            else:
                nearest = float("nan")
                delta = float("nan")
            peak_rows.append(
                {
                    "residual_peak_q": float(q0),
                    "relative_height": float(yy),
                    "nearest_best_k_q": nearest,
                    "delta_q": delta,
                    "matched": bool(math.isfinite(delta) and delta <= max(float(best.get("fwhm_q", 0.04)), 0.035)),
                }
            )
    peaks_csv = out / "magnetic_peak_support.csv"
    if peak_rows:
        with peaks_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(peak_rows[0].keys()))
            writer.writeheader()
            writer.writerows(peak_rows)

    plot_path = out / "magnetic_residual_fit.png"
    fit_norm = np.asarray(best.get("fit", np.zeros_like(y_fit)), dtype=float) if best else np.zeros_like(y_fit)
    _write_plot(plot_path, q_arr, y_fit, fit_norm, best_positions, best or {}, evidence)

    payload = {
        "schema_version": "1.0",
        "plot_kind": "magnetic_precheck",
        "title": "Magnetic residual indexing precheck",
        "q": q_arr.round(7).tolist(),
        "positive_residual": y_fit.round(7).tolist(),
        "best_fit": fit_norm.round(7).tolist(),
        "best_k_positions": best_positions.round(7).tolist(),
        "residual_peaks_q": peak_q.round(7).tolist(),
        "best": {k: v for k, v in (best or {}).items() if k not in {"fit", "coefficients"}},
        "evidence": evidence,
        "null_percentile": null_percentile,
    }
    (out / "magnetic_residual_fit.plotdata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary = {
        "enabled": True,
        "status": "complete",
        "evidence": evidence,
        "reason": reason,
        "best_k": None if not best else best.get("k_label"),
        "best_score": best_score,
        "explained_fraction": float(best.get("explained_fraction", 0.0)) if best else 0.0,
        "supported_top_peaks": int(best.get("supported_top_peaks", 0)) if best else 0,
        "active_peaks": int(best.get("active_peaks", 0)) if best else 0,
        "null_percentile": null_percentile,
        "null_p95_score": null_p95,
        "candidate_k_vectors": int(len(k_vectors)),
        "q_max": float(cfg.q_max),
        "points": int(q_arr.size),
        "seconds": float(time.perf_counter() - t0),
        "artifacts": {
            "rankings_csv": str(rankings_csv),
            "peak_support_csv": str(peaks_csv),
            "plot_png": str(plot_path),
            "plot_payload": str(out / "magnetic_residual_fit.plotdata.json"),
        },
    }
    (out / "magnetic_precheck_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "MagneticPrecheckConfig",
    "generate_k_vectors",
    "run_magnetic_precheck",
    "satellite_q_positions",
]
