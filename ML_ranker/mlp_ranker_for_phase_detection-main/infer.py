#!/usr/bin/env python3
"""
Deployment-friendly inference for the MLP phase ranker.

Supports inputs:
- text histogram blocks
- JSON (single sample or dataset list)
- CSV (hist_block column, or long-form candidate rows grouped by a key)

Outputs:
- reranked candidates with predicted relevance scores
- top-k MP IDs
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as e:
    raise SystemExit(
        "This script requires PyTorch. Install with `pip install torch` "
        "(CUDA build if you want GPU inference)."
    ) from e


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class Candidate:
    mp_id: str
    rank: int
    score: Optional[float]
    cos: Optional[float]
    beta: Optional[float]
    alpha: Optional[float]
    p: Optional[float]
    explained: Optional[float]


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


# -----------------------------
# Parsing: histogram text block
# -----------------------------

_ID_PATTERN = re.compile(r"mp-\d+")

_HIST_REGEX = re.compile(
    r"""
    ^\s*-\s*
    (?P<id>mp-\d+)\s*:\s*
    (?P<name>.+?)\s*,\s*SG\s*=\s*(?P<sg>-?\d+)\s*,\s*
    score\s*=\s*(?P<score>-?\d+(?:\.\d+)?)\s*\(
    \s*cos\s*=\s*(?P<cos>-?\d+(?:\.\d+)?)\s*,\s*
    beta\s*=\s*(?P<beta>-?\d+(?:\.\d+)?)\s*,\s*
    (?:α|alpha|a)\s*=\s*(?P<alpha>-?\d+(?:\.\d+)?)\s*,\s*
    p\s*=\s*(?P<p>-?\d+(?:\.\d+)?)\s*,\s*
    explained\s*=\s*(?P<explained>-?\d+(?:\.\d+)?)\s*
    \)\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_hist_block(block_text: str) -> List[Candidate]:
    """
    Parse the histogram block format used in the phase-detection dataset.

    We only care about the numeric metrics needed for feature engineering.
    """
    if not block_text:
        return []

    lines = block_text.splitlines()
    bullet_lines = [ln for ln in lines[1:] if ln.strip().startswith("-")]

    out: List[Candidate] = []
    for idx, ln in enumerate(bullet_lines, start=1):
        m = _HIST_REGEX.match(ln)
        if not m:
            # Fallback: try to recover mp_id, but metrics may be missing
            mp_id = None
            idm = _ID_PATTERN.search(ln)
            if idm:
                mp_id = idm.group(0)
            out.append(
                Candidate(
                    mp_id=mp_id or f"unknown_{idx}",
                    rank=idx,
                    score=None,
                    cos=None,
                    beta=None,
                    alpha=None,
                    p=None,
                    explained=None,
                )
            )
            continue

        out.append(
            Candidate(
                mp_id=m.group("id"),
                rank=idx,
                score=_to_float(m.group("score")),
                cos=_to_float(m.group("cos")),
                beta=_to_float(m.group("beta")),
                alpha=_to_float(m.group("alpha")),
                p=_to_float(m.group("p")),
                explained=_to_float(m.group("explained")),
            )
        )
    return out


# -----------------------------
# Features: exactly 18 dims
# -----------------------------

FEATURE_NAMES: Tuple[str, ...] = (
    # Base (6)
    "score",
    "cos",
    "beta",
    "alpha",
    "p",
    "explained",
    # Rank (4)
    "rank",
    "rank_normalized",
    "score_ratio_to_top",
    "cos_ratio_to_top",
    # Context (8)
    "score_rank_in_list",
    "cos_rank_in_list",
    "p_rank_in_list",
    "explained_rank_in_list",
    "score_percentile",
    "cos_percentile",
    "score_gap_to_prev",
    "score_gap_to_next",
)


def _safe_value(v: Optional[float], default: float = 0.0) -> float:
    return default if v is None else float(v)


def _compute_ranks_desc(values: Sequence[float]) -> List[int]:
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: -x[0])  # descending
    ranks = [0] * len(values)
    for r, (_, orig_idx) in enumerate(indexed, start=1):
        ranks[orig_idx] = r
    return ranks


def _compute_percentiles(values: Sequence[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    indexed = [(v, i) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[0])  # ascending
    pct = [0.0] * n
    for pos, (_, orig_idx) in enumerate(indexed):
        pct[orig_idx] = pos / (n - 1)
    return pct


def extract_features(candidates: Sequence[Candidate]) -> np.ndarray:
    """
    Convert a candidate list into an (N, 18) float32 feature matrix.

    This mirrors the ranker feature engineering from the training repo.
    """
    if not candidates:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

    n = len(candidates)
    top = candidates[0]
    top_score = _safe_value(top.score, 1.0)
    top_cos = _safe_value(top.cos, 1.0)

    all_scores = [_safe_value(c.score) for c in candidates]
    all_cos = [_safe_value(c.cos) for c in candidates]
    all_p = [_safe_value(c.p) for c in candidates]
    all_explained = [_safe_value(c.explained) for c in candidates]

    score_ranks = _compute_ranks_desc(all_scores)
    cos_ranks = _compute_ranks_desc(all_cos)
    p_ranks = _compute_ranks_desc(all_p)
    explained_ranks = _compute_ranks_desc(all_explained)
    score_pct = _compute_percentiles(all_scores)
    cos_pct = _compute_percentiles(all_cos)

    feats: List[np.ndarray] = []
    for c in candidates:
        idx = c.rank - 1
        score = _safe_value(c.score)
        cos = _safe_value(c.cos)
        beta = _safe_value(c.beta)
        alpha = _safe_value(c.alpha)
        p = _safe_value(c.p)
        explained = _safe_value(c.explained)

        rank = float(c.rank)
        rank_norm = rank / n if n > 0 else 0.0
        score_ratio = score / top_score if top_score > 0 else 0.0
        cos_ratio = cos / top_cos if top_cos > 0 else 0.0

        score_gap_prev = all_scores[idx - 1] - all_scores[idx] if idx > 0 else 0.0
        score_gap_next = all_scores[idx] - all_scores[idx + 1] if idx < n - 1 else 0.0

        row = np.array(
            [
                score,
                cos,
                beta,
                alpha,
                p,
                explained,
                rank,
                rank_norm,
                score_ratio,
                cos_ratio,
                float(score_ranks[idx]),
                float(cos_ranks[idx]),
                float(p_ranks[idx]),
                float(explained_ranks[idx]),
                float(score_pct[idx]),
                float(cos_pct[idx]),
                float(score_gap_prev),
                float(score_gap_next),
            ],
            dtype=np.float32,
        )
        feats.append(row)

    return np.stack(feats, axis=0)


# -----------------------------
# Model definition (matches training)
# -----------------------------

class RankingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(max(0, n_layers - 1)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, input_dim)
        return self.mlp(x).squeeze(-1)  # (N,)


def load_model(checkpoint_path: str, device: str) -> Tuple[RankingMLP, torch.device]:
    dev_str = device
    if dev_str == "auto":
        dev_str = "cuda" if torch.cuda.is_available() else "cpu"
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: --device cuda requested but torch.cuda.is_available() is False. Falling back to CPU.",
            file=sys.stderr,
        )
        dev_str = "cpu"

    dev = torch.device(dev_str)
    ckpt = torch.load(checkpoint_path, map_location=dev)

    cfg = ckpt.get("config", {}) or {}
    hidden_dim = int(cfg.get("hidden_dim", 64))
    n_layers = int(cfg.get("n_layers", 2))
    dropout = float(cfg.get("dropout", 0.1))

    model = RankingMLP(
        input_dim=len(FEATURE_NAMES),
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, dev


def rank_candidates(model: RankingMLP, device: torch.device, candidates: Sequence[Candidate]) -> Dict[str, Any]:
    feats = extract_features(candidates)
    if feats.shape[0] == 0:
        return {"top_ids": [], "ranked": []}

    with torch.no_grad():
        x = torch.from_numpy(feats).to(device)
        scores = model(x).detach().cpu().numpy().astype(float)

    order = np.argsort(-scores)
    ranked = []
    for new_rank, idx in enumerate(order, start=1):
        ranked.append({"rank": new_rank, "mp_id": candidates[int(idx)].mp_id, "score": float(scores[int(idx)])})
    return {
        "top_ids": [r["mp_id"] for r in ranked],
        "ranked": ranked,
    }


# -----------------------------
# Input loaders
# -----------------------------

def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_samples_from_json(path: str) -> List[Dict[str, Any]]:
    raw = _read_text(path)
    obj = json.loads(raw)

    # Dataset list format
    if isinstance(obj, list):
        return [{"sample_id": d.get("run_name") or str(i), **d} for i, d in enumerate(obj)]

    # Single sample dict format
    if isinstance(obj, dict):
        return [{"sample_id": obj.get("run_name") or "sample_0", **obj}]

    raise ValueError("Unsupported JSON shape; expected list or dict.")


def _parse_candidate_row(row: Dict[str, str]) -> Candidate:
    # For long-form CSV rows (per-candidate):
    return Candidate(
        mp_id=row.get("mp_id") or row.get("id") or "",
        rank=int(row.get("rank") or 0) if (row.get("rank") or "").strip() else 0,
        score=_to_float(row.get("score")),
        cos=_to_float(row.get("cos")),
        beta=_to_float(row.get("beta")),
        alpha=_to_float(row.get("alpha")),
        p=_to_float(row.get("p")),
        explained=_to_float(row.get("explained")),
    )


def load_samples_from_csv(path: str, group_col: str) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        # Dataset-style CSV: has hist_block column
        if "hist_block" in fieldnames:
            for i, row in enumerate(reader):
                sample_id = row.get("run_name") or row.get(group_col) or str(i)
                samples.append({"sample_id": sample_id, "hist_block": row.get("hist_block") or ""})
            return samples

        # Long-form: candidate rows grouped by group_col
        required = {"mp_id", "score", "cos", "beta", "alpha", "p", "explained"}
        if not required.issubset(set(fieldnames)):
            raise ValueError(
                "CSV input missing required columns. Expected either a `hist_block` column, "
                "or long-form columns: group_id/run_name, mp_id, score, cos, beta, alpha, p, explained "
                "(rank optional)."
            )

        groups: Dict[str, List[Candidate]] = {}
        for row in reader:
            gid = row.get(group_col) or row.get("run_name") or row.get("group_id")
            if not gid:
                continue
            cand = _parse_candidate_row(row)
            if not cand.mp_id:
                continue
            groups.setdefault(gid, []).append(cand)

        for gid, cands in groups.items():
            # Ensure ranks exist and are stable
            if any(c.rank <= 0 for c in cands):
                # If rank missing, order by file order and assign 1..N
                cands = [
                    Candidate(**{**c.__dict__, "rank": i + 1})  # type: ignore[attr-defined]
                    for i, c in enumerate(cands)
                ]
            else:
                cands = sorted(cands, key=lambda c: c.rank)
            samples.append({"sample_id": gid, "candidates": cands})

    return samples


def candidates_from_sample(sample: Dict[str, Any]) -> List[Candidate]:
    if "candidates" in sample and isinstance(sample["candidates"], list):
        # JSON list of structured candidates
        if sample["candidates"] and isinstance(sample["candidates"][0], Candidate):
            return list(sample["candidates"])

        out: List[Candidate] = []
        for i, d in enumerate(sample["candidates"], start=1):
            out.append(
                Candidate(
                    mp_id=str(d.get("mp_id") or d.get("id") or f"unknown_{i}"),
                    rank=int(d.get("rank") or i),
                    score=_to_float(d.get("score")),
                    cos=_to_float(d.get("cos")),
                    beta=_to_float(d.get("beta")),
                    alpha=_to_float(d.get("alpha")),
                    p=_to_float(d.get("p")),
                    explained=_to_float(d.get("explained")),
                )
            )
        return out

    hist_block = sample.get("hist_block")
    if isinstance(hist_block, str) and hist_block.strip():
        return parse_hist_block(hist_block)

    return []


# -----------------------------
# CLI
# -----------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="MLP phase ranker inference (text/json/csv).")
    ap.add_argument("--model", required=True, help="Path to mlp_ranker.pt")
    ap.add_argument("--input", required=True, help="Input path, or '-' for stdin (text/json)")
    ap.add_argument("--format", choices=["auto", "text", "json", "csv"], default="auto")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Limit number of samples (json/csv dataset inputs)")
    ap.add_argument("--group-col", default="group_id", help="CSV long-form grouping column (default: group_id)")
    ap.add_argument("--output", default=None, help="Optional output path (JSONL). Defaults to stdout.")
    args = ap.parse_args(argv)

    fmt = args.format
    if fmt == "auto":
        lower = args.input.lower()
        if lower.endswith(".json"):
            fmt = "json"
        elif lower.endswith(".csv"):
            fmt = "csv"
        else:
            fmt = "text"

    model, dev = load_model(args.model, args.device)

    samples: List[Dict[str, Any]]
    if fmt == "text":
        block = _read_text(args.input)
        samples = [{"sample_id": os.path.basename(args.input) if args.input != "-" else "stdin", "hist_block": block}]
    elif fmt == "json":
        samples = load_samples_from_json(args.input)
    elif fmt == "csv":
        samples = load_samples_from_csv(args.input, group_col=args.group_col)
    else:
        raise AssertionError("unreachable")

    if args.limit is not None:
        samples = samples[: args.limit]

    out_f = open(args.output, "w", encoding="utf-8") if args.output else None
    try:
        for s in samples:
            sample_id = s.get("sample_id") or s.get("run_name") or "sample"
            cands = candidates_from_sample(s)
            res = rank_candidates(model, dev, cands)

            ranked = res["ranked"]
            topk_ranked = ranked[: max(0, int(args.topk))]

            line = {
                "sample_id": sample_id,
                "n_candidates": len(cands),
                "top_ids": [r["mp_id"] for r in topk_ranked],
                "ranked": topk_ranked,
            }
            js = json.dumps(line)
            if out_f:
                out_f.write(js + "\n")
            else:
                print(js)
    finally:
        if out_f:
            out_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
