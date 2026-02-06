# MLP Phase Ranker (Deployment-Friendly Inference)

This folder is a lightweight “inference package” for the **MLP ranker** trained from the `phase-detection` repo.

It is meant to be user-friendly:

- Accepts input as a raw histogram text block, a JSON file, or a CSV file
- Recreates the exact **18-feature** representation used during training
- Loads the saved model weights (`mlp_ranker.pt`)
- Produces reranked candidate phases (top-k IDs + scores)

## What This Model Does

Given a list of candidate impurity phases (from your upstream ML-hist matching pipeline), the model predicts a **relevance score per candidate** and reranks them so the most likely secondary phase rises to the top.

The model does **not** use any chemistry strings, space group, or MP-ID semantics as features. It uses only the numeric metrics available in the histogram block:

- `score`, `cos`, `beta`, `alpha`, `p`, `explained`

From those it derives additional rank/context features (percentiles, ratios, gaps) to reach 18 features per candidate.

## Files

- `mlp_ranker.pt`
  - The trained model checkpoint (weights + small config).
- `infer.py`
  - CLI and library entry point for inference.

## Install Requirements

This deployment code only requires PyTorch and NumPy:

```bash
pip install torch numpy
```

If you want GPU inference, install a CUDA-enabled PyTorch build and ensure a GPU is available.

## Inputs Supported

### 1) Text Block

Provide a file that contains the histogram block text, including lines like:

```text
Top 20 by ML-hist (id, name, SG, score; cos, beta, α , p , explained):
  - mp-1095573: Co,Nb,Si, SG=62, score=0.216 (cos=0.775, beta=0.685, α=0.216, p=0.996, explained=0.376)
  - mp-1220588: Nb3Si, SG=65, score=0.156 (cos=0.769, beta=0.605, α=0.269, p=0.115, explained=0.195)
```

Run:

```bash
python infer.py --model mlp_ranker.pt --input sample.txt --format text --topk 5
```

### 2) JSON

Two JSON shapes are supported:

1. Dataset-style JSON (like `runs_dataset.json`)
   - A list of objects each with `hist_block` (and optionally `run_name`, `primary_id`, etc.)
2. Single-sample JSON
   - Either `{ "hist_block": "..." }`
   - Or `{ "candidates": [ { "mp_id": "...", "score": 0.1, "cos": 0.9, "beta": 0.5, "alpha": 0.2, "p": 0.9, "explained": 0.3 }, ... ] }`

Examples:

```bash
python infer.py --model mlp_ranker.pt --input one.json --format json --topk 5
python infer.py --model mlp_ranker.pt --input runs_dataset.json --format json --limit 100 --topk 5
```

### 3) CSV

Two CSV shapes are supported:

1. Dataset-style CSV with a `hist_block` column (recommended)
2. Long-form candidate rows grouped by `group_id` (or `run_name`)
   - Required columns: `group_id` (or `run_name`), `mp_id`, `score`, `cos`, `beta`, `alpha`, `p`, `explained`

Examples:

```bash
python infer.py --model mlp_ranker.pt --input data.csv --format csv --topk 5
python infer.py --model mlp_ranker.pt --input candidates.csv --format csv --group-col run_name --topk 5
```

## Output

By default, the CLI prints JSON lines, one per sample:

- `top_ids`: the top-k MP-IDs after reranking
- `ranked`: full reranked list `[{rank, mp_id, score}]`

You can also save outputs with `--output`.

## Notes About Candidate Count (Dynamic N)

The model can score a variable number of candidates `N` per sample.

- Feature engineering uses `N` (for percentiles and rank normalization), so it naturally adapts.
- There is no requirement that `N` be 20, 25, or 50 at inference time.

## Quick Sanity Check

If you have the original dataset JSON, you can run:

```bash
python infer.py --model mlp_ranker.pt \
  --input /home/ly6/torch_evs_1/quick_phase_detection/runs_dataset.json \
  --format json --limit 3 --topk 5
```

