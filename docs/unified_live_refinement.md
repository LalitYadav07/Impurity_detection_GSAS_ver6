# Live, guided refinement and transition evidence

This integration starts at NOVA 0.5.34 (`3a27057`). It ports feature changes
instead of replacing the application with any earlier feature checkout.

## Responsibilities

| Layer | Responsibility | Must not do |
| --- | --- | --- |
| Acquisition | Capture an immutable run/bank/reduction revision and its measured conditions | Treat two cumulative snapshots as independent counts |
| RADAR-PD | Identify/refine candidate nuclear phases and publish checked source GPXs | Rewrite an earlier published source in place |
| Guided refinement | Build a reviewed recipe, apply explicit phase corrections, dispatch finite jobs | Infer lithium content or state of charge without supplied evidence |
| Live counting | Update a conditional magnetic fit against a reviewed nuclear reference | Reuse a response after grid, mask, sample, calibration or model changes |
| Transition tracking | Compare measured curves independently of phase labels | Promote a morphology score to proof of magnetic or structural origin |
| Galaxy transport | Dispatch, recover and publish native artifacts | Run GSAS-II in the interactive NOVA server process |

The original transition indicators, selectable boundary rule and recurring
pattern symbols remain the default. The U-Net is additional, optional evidence.
Its labels are machine-assisted training labels; its scores are not calibrated
transition probabilities. Unknown reduction/normalization is reported rather
than silently treated as comparable exposure.

## Common numerical implementation

Nuclear and magnetic model construction use `gpx_recipe`. Live response
calculation has a distinct objective (fixed nuclear response and conditional
posterior), but it must use the same native phase construction and checked
geometry. Scientific execution lives in workers; UI compatibility imports do
not introduce a second solver.

`gpx_recipe.symmetry.kernel_patch` applies a hash-pinned correction while
building a new immutable worker. It includes direct-axis magnetic geometry,
rotated anisotropic displacement factors and neutron powder Friedel intensity
averaging. Independent Cartesian and complex-amplitude controls run before the
image is accepted. Existing sealed campaigns retain their original protocol
and image identity. A new calculation cannot silently continue an old protocol
with a different implementation.

## Source and output identity

Every operation binds its acquisition, source checksum, model/recipe,
observations, native implementation and output checksum. A live response cache
also binds the exact candidate setting, magnetic sites, propagation vector,
grid, masks and fixed nuclear reference.

The displayed posterior maximum and exported GPX must agree. A constructed
starting project is not a fitted project. Exporting a conditional posterior
must clear unrelated inherited covariance and must not label the project as
a converged least-squares refinement. The original source remains available.

For a cumulative live stream there is one active revision and at most one
pending revision. A newer pending snapshot supersedes the waiting snapshot;
it does not change an active input or erase earlier results. An interrupted
stream requires identity checks before resuming.

## Resource use

`native_job_admission` accounts for active RADAR-PD utilities in the same
Galaxy history before a new utility POST. The default total limit is eight,
configurable by `RADAR_PD_MAX_ACTIVE_JOBS` (1–32). Existing per-workflow limits
still apply. The check and POST are serialized within the viewer. Galaxy owns
cross-viewer admission and actual CPU/RAM allocation; this client-side budget
does not reserve cluster nodes.

A denied or cancelled capacity check occurs before submission and clears the
local pending claim. An ambiguous POST acknowledgement remains recoverable
and is never blindly retried. Native subprocesses retain scheduler allocation
metadata and use one BLAS/OpenMP thread per process. Independent scans/model
starts can run concurrently inside their granted allocation.

## Release gates

Before pinning NDIP to the integrated image:

1. Run the combined application suite, native scientific controls and actual
   packaged worker tests, with the source revisions recorded.
2. Verify batch preparation, phase edits, guided recipes and every output GPX
   through Galaxy; do not remove local-only guards before transport is tested.
3. Test a live sequence containing repeated data, a superseded waiting update,
   a changed mask/grid, a restart and a source change during a running job.
4. Test old campaign review and preserved source selection, optional U-Net
   activation, zoom/cursor interaction and history lengths beyond its cache.
5. Start a fresh NDIP viewer, provision credentials through the existing secure
   mechanism, and test Luna-assisted setup and result recovery. Check both
   deployment jobs and the live catalog before the browser smoke test.

These are release gates, not a claim that every workflow has already passed.
