# Experiment Planner runtime and recovery

The planner uses the existing `neutrons_radar_pd_model_campaign_prototype` tool,
`action=native`, its History admission guard, and its durable native handles.
NOVA captures data and polls jobs; it does not import GSAS-II. The explicit
`RADAR_PD_PLANNING_PYTHON` or `RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND=1` developer
route uses a separate local process. Failure to connect to Galaxy does not
silently select local science.

`radar-model-native/v1` carries the named operations `planning-parent`,
`planning-simulate`, `planning-context`, `planning-export-parent`,
`planning-magnetic-prepare`, `planning-structural-match`,
`planning-structural-modes`, and `planning-structural-prepare`. Every payload has
`source`, `plan`, `children`, and `destination`; transport capture converts them
to relative paths. `radar-planning-job/v1` seals the request and exact input
inventory. A source CIF serves as an opaque byte anchor for forward jobs;
native transition jobs use only GPXs produced by an earlier verified planner
job. Catalog discovery and its credentials remain at the NOVA boundary.

The worker dispatches to the existing `planning.worker.run` and
`planning.magnetic.execute` implementations, which use the shared structural
and magnetic engines. It returns `radar-planning-operation-result/v1` inside the
existing native result envelope. NOVA verifies the original request, source,
files and History/tool/job identity before accepting it. A recorded submission
with an uncertain acknowledgement is recovered by its stable dispatch key,
never resubmitted speculatively. Cancellation addresses that exact job.

Plans are `radar-experiment-plan/v2` ZIPs: `plan.json` plus an explicit inventory
of relative files, byte sizes and SHA-256 hashes. Inputs, settings, result
arrays, receipts and transition context are retained. Duplicate or escaping
paths, symbolic links, undeclared files, oversized archives, object arrays,
nonfinite numeric arrays and oversized NumPy headers are rejected. Uploaded
GPXs remain opaque data. Only recollection from the saved authenticated native
job can restore authority to reuse one as a native input. Fixed-width text
metadata in numeric archives is allowed; object/pickle arrays are not. Recovery
checks numeric artifacts and scientific receipt fields against the original
worker output, then compares the original scenario with the saved settings.
Unverified imported previews are labelled explicitly. Changed inputs or a
changed science signature invalidate predictions while preserving settings.

`save_planning_plan` uploads a bundle to the active Galaxy History. A fresh
session calls `refresh_planning_plans` and restores a selected History dataset.
Each native dispatch first uploads its input checkpoint and handle as a
`radar-planner-operation/v1` History document. `refresh_planning_jobs` lists
these attempts; `recover_planning_job` only polls and collects them. Finished
forward/parent jobs restore the corresponding preview. Other native substep
outputs remain recoverable as checked data; catalog construction can be
repeated explicitly with matching captured requests reusing existing jobs.

UI hooks are `refresh_planning_plans`, `export_planning_plan`,
`import_planning_plan` (`plan_bundle_upload`), `refresh_planning_jobs`, and
`recover_planning_job` (`plan_recover_job_id`, `plan_recoverable_jobs`). Job
identity/state are exposed as `plan_job_id`, `plan_job_state`, and
`plan_job_operation`. Storage and job waits run away from the UI event loop.
Input edits retain generation checks so a late job cannot replace a newer plan.

This transport adds no instrument count calibration. The existing selected
POWGEN 2026B bands retain their conditional count model; unmatched profiles
remain shape previews. Native/Galaxy deployment and fresh-account acceptance
require a matched worker image and live tests; mocked transport tests are not
evidence of a deployed NDIP calculation.
