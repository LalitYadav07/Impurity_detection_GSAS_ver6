"""Read-only, bounded setup evidence from an existing Live reduction.

These previews do not refine data or qualify a calibration. Recommendations
are explicit user-applied starting values, scoped to the inspected bank.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import nomad, powgen_setup
from .pattern_transition import prepare_pattern
from .powgen_controller import bounded_directory_listing
from .powgen_samples import include_run
from .powgen_watch import WatchDefinition, discover_from_listing
from .transition_input import reduced_observation
from .live_beamlines import beamline_adapter
from .facility_metadata import title_role


class NoPreviewData(ValueError):
    """Readable experiment folder with no eligible complete reduction yet."""


def inspect_inventory(instrument: str, ipts: str, folder: str, sample_index: Any,
                      sample_ids=None, excluded_runs=(), include_background=False,
                      include_ramps=False, **policy) -> dict[str, Any]:
    """Inspect the latest eligible reduction, without creating controller/jobs."""
    adapter = beamline_adapter(instrument)
    if not adapter.preview_runs or not adapter.preview_setup:
        raise ValueError('This beamline has no registered setup preview. Use its ordinary Live controls.')
    definition = WatchDefinition(ipts, folder, "setup-preview", "setup-preview", "setup-preview",
                                 instrument=instrument, facility=adapter.facility)
    runs = adapter.preview_runs(definition, policy)
    matching = []
    for run in reversed(runs):
        if run.run_id in excluded_runs or run.run_number in excluded_runs:
            continue
        identity = sample_index.identity(run)
        role = title_role(identity.get('run_title') or '')
        if instrument in {'PG3','NOM'} and not include_run(run, include_background_runs=include_background,
                                                        include_ramp_runs=include_ramps):
            continue
        if (role == 'background' and not include_background) or (role == 'ramp' and not include_ramps):
            continue
        if sample_ids is None or identity.get("sample_id") in sample_ids:
            matching.append((run, identity))
            break
    if not matching:
        raise NoPreviewData("No completed reduction matches these samples and run choices. You can fill the configuration now and refresh when data arrives.")
    run, identity = matching[0]
    source = Path(run.source_path).resolve()
    root = Path(definition.source_directory).resolve()
    source.relative_to(root)
    setup, banks = adapter.preview_setup(source, definition, policy)
    return {
        "instrument": instrument, "ipts": ipts, "folder": folder,
        "source_path": str(source), "root": str(root), "run_number": run.run_number,
        "run_id": run.run_id, "sample_id": identity.get("sample_id") or "",
        "identity_status": identity.get("identity_status") or "unidentified",
        "sample_name": identity.get("sample_name") or "", "banks": banks,
        "source_signature": list(setup.source_signature), "setup": setup.as_dict(),
        "setup_key": adapter.controller.setup_metadata_key,
        "unit": "degrees 2θ" if adapter.instrument_mode == 'cw' else "µs",
    }


def background_suggestion(pattern) -> tuple[dict[str, Any], Any]:
    """Compare low-order descriptions of an estimated envelope, not raw Rwp.

Blocked envelope errors are a complexity heuristic, not an independent
statistical validation of a physical background or uncertainty estimate.
Missing bins and gaps are kept out of every score.
"""
    prepared = prepare_pattern(pattern)
    valid = pattern.valid & np.isfinite(prepared.baseline)
    if int(valid.sum()) < 200:
        return {"terms": None, "reason": "Too little continuous valid support for a background recommendation."}, prepared
    ids = np.flatnonzero(valid)
    ids = ids[np.linspace(0, len(ids) - 1, min(600, len(ids)), dtype=int)]
    x, baseline = pattern.x[ids], prepared.baseline[ids]
    z = 2 * (x - x[0]) / (x[-1] - x[0]) - 1
    blocks = np.minimum(19, np.arange(len(ids)) * 20 // len(ids))
    scores = []
    for terms in (4, 6, 8, 10, 12):
        errors = []
        for fold in range(5):
            test = blocks % 5 == fold
            train = ~test
            coef = np.polynomial.chebyshev.chebfit(z[train], baseline[train], terms - 1)
            errors.extend((np.polynomial.chebyshev.chebval(z[test], coef) - baseline[test]) ** 2)
        score = float(np.sqrt(np.mean(errors)))
        scores.append({"terms": terms, "envelope_rmse": score})
    best = min(row["envelope_rmse"] for row in scores)
    scale = max(float(np.ptp(baseline)), float(np.median(np.abs(baseline))), 1e-12)
    chosen = next(row["terms"] for row in scores if row["envelope_rmse"] <= best * 1.10 + scale * 0.005)
    if not np.isfinite(best) or best > scale * 0.10:
        chosen = None
    return {
        "terms": chosen,
        "reason": (f"{chosen} Chebyshev-1 terms is the simplest tested description close to the best estimated-envelope score."
                   if chosen else "The estimated envelope is not represented reliably by these smooth models; review the background manually."),
        "method": "low-envelope-blocked-complexity/v1", "scores": scores,
        "limitation": "Heuristic for this bank only. Broad sample/container scattering and crowded peaks can bias the envelope. This does not validate the physical background.",
    }, prepared


def preview_pattern(inventory: dict[str, Any], bank: int, limits=None, exclude_regions=()) -> dict[str, Any]:
    source = Path(inventory["source_path"])
    stat = source.stat()
    if [stat.st_size, stat.st_mtime_ns] != inventory["source_signature"]:
        raise ValueError("The reduction changed after discovery. Refresh the experiment before inspecting its curve.")
    scan = {
        "instrument": inventory["instrument"], "ipts": inventory["ipts"],
        "run_id": inventory["run_id"], "source_path": str(source),
        "transition_source_root": inventory["root"], "bank": bank,
        "metadata": {
            "sample_id": inventory["sample_id"], "identity_status": inventory["identity_status"],
            inventory.get('setup_key', "nomad_setup" if inventory["instrument"] == "NOM" else "powgen_setup"): inventory["setup"],
        },
    }
    observation = reduced_observation(scan)
    # Work on a private mask: the input observations remain unchanged.
    pattern = observation.pattern
    valid = pattern.valid.copy()
    if limits:
        valid &= (pattern.x >= limits[0]) & (pattern.x <= limits[1])
    for lo, hi in exclude_regions:
        valid &= ~((pattern.x >= lo) & (pattern.x <= hi))
    from dataclasses import replace
    pattern = replace(pattern, valid=valid)
    recommendation, prepared = background_suggestion(pattern)
    baseline = prepared.baseline
    ids = np.unique(np.linspace(0, len(pattern.x) - 1, min(2500, len(pattern.x)), dtype=int))
    def values(array):
        return [float(array[i]) if valid[i] and np.isfinite(array[i]) else None for i in ids]
    return {
        "run_id": inventory["run_id"], "bank": bank, "sample_id": inventory["sample_id"],
        "source_sha256": observation.provenance["sha256"],
        "points": len(pattern.x), "valid_points": int(valid.sum()),
        "range": [float(pattern.x[0]), float(pattern.x[-1])], "unit": inventory.get('unit', 'µs'),
        "negative_points": int(np.sum((pattern.intensity < 0) & valid)),
        "x": pattern.x[ids].tolist(), "y": values(pattern.intensity), "background": values(baseline),
        "recommendation": recommendation, "notices": observation.notices,
        "display_note": "Display is sampled; recommendations use native valid points. Masks remain gaps. No fit is submitted.",
    }
