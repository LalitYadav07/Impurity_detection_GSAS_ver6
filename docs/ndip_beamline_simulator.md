# Local beamline simulation and scientific validation

The filesystem simulator in `scripts/beamline_simulator.py` replays small,
hash-sealed subsets of facility data into an isolated directory. It does not
reduce detector events, emulate Galaxy, or certify an instrument calibration.
Private experiment files and proposal abstracts stay outside this repository.

## Corpus contract

Use a JSON manifest with schema `radar-pd-beamline-simulation/v1`. Paths are
relative to the simulated SNS root, preserving instrument, IPTS and reduction
folder names. Each source is sealed by byte count and SHA-256:

```json
{
  "$schema": "radar-pd-beamline-simulation/v1",
  "files": [
    {
      "path": "NOM/IPTS-1/shared/autoreduce/NOM_42.gsa",
      "bytes": 12345,
      "sha256": "replace-with-the-actual-full-sha256"
    }
  ],
  "events": [
    {
      "at_seconds": 0,
      "path": "NOM/IPTS-1/shared/autoreduce/NOM_42.gsa",
      "bytes": 2000
    },
    {
      "at_seconds": 5,
      "path": "NOM/IPTS-1/shared/autoreduce/NOM_42.gsa"
    }
  ]
}
```

The first event exposes a partial file; the second atomically replaces it with
the full original. Events may also remove a file or use another sealed source
via `source`. Metadata can arrive before or after a reduction. Changed-content
experiments need a separately sealed source; do not silently modify the corpus.
For verified archive aliases, use `operation: "link"` with `source` naming an
already published, fully sealed path. Both manifest entries must have matching
hashes. This preserves inode identity without introducing a false raw-source
ambiguity. Confirm selected dataset values and original archive topology before
normalizing metadata-only aliases; retain the original subset hashes.

```bash
python scripts/beamline_simulator.py \
  --manifest /private/test-corpus/manifest.json \
  --corpus /private/test-corpus/SNS \
  --destination /private/test-replay/SNS \
  --realtime
```

Without `--realtime`, events run immediately. Python harnesses can call
`BeamlineSimulator.apply(index)` and inspect the watcher after each event.
The corpus and destination must be disjoint. The destination must be empty or
already marked by the simulator. Writes to `/SNS`, `/HFIR` and `/gpfs` are
refused. Container tests mount the isolated destination at `/SNS` for the
production resolver while applying events through its separate local path.

## Selective facility retrieval

Copy individual reduced GSAS files, their per-run reduction receipts, relevant
instrument files, and scan tables. Check source sizes and permissions before
transfer. Do not execute saved facility reduction scripts; the resolver parses
their supported arguments as metadata.

For large NeXus files, make explicitly marked metadata-only HDF5 subsets. Copy
only the exact scalar/log datasets used by the metadata reader, preserving
shape, dtype, attributes and units. Record original source paths, original byte
counts, copied dataset paths, omitted fields and subset hashes. Do not visit
detector/event datasets. A metadata subset tests the copied reader inputs; it
does not establish coverage of omitted datasets or raw reduction.
Compare the deployed `h5dump` reader against direct original-file reads as
well as native Python tests. Request roundtrip-safe numeric formatting, and
exercise facility nanosecond timestamps on the container's Python version.
Keep original timestamp strings and UTC offsets; parsing may use datetime's
microsecond resolution. Missing or inconsistent timezone information must not
be invented for a scientific time axis.

## Validation layers

1. Run production discovery, sample indexing and preflight against each actual
   folder family. Record ready, explicit-choice, unsupported and ambiguous cases
   separately. An intentional refusal is not a successful refinement.
2. Stage each eligible bank separately. Compare all numeric data rows with the
   source, and retain profile hashes and physical-to-local bank mappings.
3. Replay partial files, completion, late arrivals, missing metadata, retries,
   duplicate arrivals, checkpoint recovery and replaced reductions/receipts.
   Incomplete products must wait without consuming submission attempts. A
   result from replaced input retains its original provenance and review notice.
4. Run the current Analyze entry point on real inputs with declared CIF
   libraries and settings. Keep every failed or scientifically flagged result.
   A process exit code and a low Rwp do not establish model adequacy.
5. Independently read final projects through the GSAS-II public API. Compare
   collected cells, modeled-crystalline fractions and Rwp. Use independent
   direct/reciprocal metric calculations for cell volume and d(hkl).
6. Replay complete series through post-analysis. Check relative/difference
   arithmetic, exact references, fixed-parameter gates, bank/sample/model
   grouping, missing-phase gaps and CSV/JSON/HTML exports. Known mappings must
   distinguish measured conditions from preparation/composition inputs.
7. Exercise the actual NOVA browser UI on real results: discovery, preflight,
   selection controls, fit traces, zoom, derived views and exports. Test desktop
   and narrow screens. A local Galaxy boundary is useful for lifecycle tests,
   but must be identified as such in the validation receipt.

Scientific review status belongs on derived plots and in exported observations.
Fit plots must use the explicit GPX refinement limits and exclusions. Never
infer the fitted interval from calculated zeros. Re-collecting outputs must
replace the canonical JSON/NPZ publication targets so the UI cannot silently
prefer a stale plot from an earlier collection.
Unsupported size/strain and atomic/site-order recipes remain unavailable until
their required refinement and calibration contracts are implemented and tested.

## Critical GUI acceptance

Check actual inspection paths, not only successful rendering or page width.
Keep one selected scientific view in focus by default; comparisons and advanced
definitions should remain available through explicit controls. Verify that the
first plot is reachable without scrolling past a wall of settings, that legends
remain readable, and that blocked views explain the action required.

Use native pointer clicks on actual plot points. Verify scan, phase/model, bank,
reference, units and source details, plus an explicit route to a live scan's fit.
Do not serialize the full Plotly event object; transmit only the required point
identifiers. Exercise searchable observations including missing/excluded values,
categorical preparation states, bank/sample filters, expanded plots and narrow
screens. A display change or zoom must preserve scientific arrays and must not
submit a refinement. Clearly label unavailable recipes in the picker.

## What still requires NDIP

Local tests cannot establish current Galaxy wrapper behavior, facility mount
permissions inside NDIP jobs, dataset collection discovery, proxy/session
injection, hosted desktop startup, cleanup policy, resource limits or rollout.
Build exact release images and complete an NDIP end-to-end smoke test before
claiming deployment acceptance. A cached image with mounted current source is
a runtime compatibility check, not a newly built release image.
