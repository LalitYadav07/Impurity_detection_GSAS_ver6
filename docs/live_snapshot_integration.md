# Provisional scan integration

The source review used `webmon-deployment-ready` at `d340e26`, including its
uncommitted feature files, copied into an isolated worktree based on NOVA 0.5.34.
The source snapshot and per-file hashes are retained outside this repository in
`audit/2026-11-01/integration-review/webmon-source-snapshot`.

## Boundaries

* `gpx_recipe.live` owns reduced-data parsing, native profile kernels, conditional
  posterior calculations, and scientifically verified artifact export. It
  imports no NOVA UI or Galaxy service. Magnetic candidate construction calls the
  same `gpx_recipe.symmetry.native_magnetic.simulate` implementation used by model
  campaigns.
* `radar_pd_nova.webmon` owns source discovery, observation queue, UI and lifecycle.
  Old scientific module paths are compatibility aliases, not implementations.
* Capturing a scan does not authorize a refinement. A user selects and reviews a
  nuclear checkpoint. A stream follows one acquisition/configuration, retaining
  one active revision and one latest pending revision. Cumulative snapshots
  replace the observation likelihood; they are never added together.
* The main Live monitor attaches the independent snapshot polling loop. Pausing
  monitoring stops new enrollment. Clearing or closing the session also cancels
  the owned poll and calculation tasks. An authenticated Galaxy session can browse the current POWGEN IPTS without a
  prebuilt science manifest; explicit server contexts remain available for local replay.
* Local native commands require `local_replay=true`. NDIP uses a bounded
  Galaxy request packet and optional response-cache HDA. A deployed immutable
  worker and reviewed native preparation are required before scientific submission. No Docker
  socket, GSAS interpreter, or arbitrary browser command is granted to NOVA.

## Artifact identity

`magnetic-refinement.gpx` is a construction checkpoint. It is never presented as
an inferred fit. `posterior-maximum.gpx` is a separate artifact containing the
selected posterior maximum and current cumulative observations, retaining masks,
limits and fixed nuclear parameters. It is reopened through the native adapter
and checked against the plotted native curve and weights before it is exposed.
The receipt includes both project hashes, snapshot/reference revisions, posterior
and kernel hashes, and the mapped site moments. Inherited covariance is cleared:
this file is a conditional estimate, not a native least-squares convergence claim.

The selected posterior project can be explicitly handed to the existing
Refinement workspace with a distinct source ID. Historical completed revisions
remain reviewable after a newer snapshot arrives, but they are visibly stale and
are not accepted as the current source for a new stream. While the next revision
is preparing, the last verified result and its identity remain visible.

## Native runtime

Production accepts the shared build-time-corrected GSAS-II source hash
`2a6ffce516b8edef1760d12599c98414c1eead83cd1006291d49bc5cbfef5593`
without changing native functions at runtime. The earlier Mac/pilot correction
adapter is retained solely behind `RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND=1`, for
explicit local experimental replay. It is not a route to change old immutable
campaigns or their protocol identity.

## Remaining integration gates

A production resolver must supply verified physical bank, reduction configuration,
export geometry, instrument profile, sample identity and acquisition epoch. The
current archived run-summary lookup is insufficient to prove that all of these
remain fixed. A changed producer epoch, grid, bank, sample or reduction must close
an active stream and require a new review. The server must provide an isolated
per-history source manifest and private archive/job roots.

The Galaxy adapter persists each request packet/result manifest to History,
uses `submit_utility` for shared admission, and recovers ambiguous submissions
without duplicate dispatch. Its CLI is `python -m gpx_recipe.live.galaxy_worker
packet.zip work result.zip cache.zip [--cache previous.zip]`. Inputs are
`packet`, optional `response_cache`, and `dispatch_key`; outputs are
`result_bundle` and `response_cache`. The production image/tool definition is
provided by the combined integration release. The local
process adapter is intentionally refused for a production context. The feature
does not yet advertise a production-ready scientific endpoint merely because its
panel is visible.

Counting advice and phase exclusion remain disabled. Supplied reduced-data
variances omit shared nuclear/reference and cross-bin covariance. Passing a
replica agreement screen does not establish correct magnetic structure, global
convergence or valid stopping-time uncertainty.
