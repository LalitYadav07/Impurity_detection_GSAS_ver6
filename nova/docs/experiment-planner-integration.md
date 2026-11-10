# Experiment Planner integration candidate

Candidate version: `0.7.0`. Original planner base: committed NDIP 0.6.64
(`a0495883ecff5be907c9e8671300db1de2fc8dbe`). This candidate is for local
review; no NDIP deployment or registry publication is part of this change.

## Application behavior

The primary tabs are Analysis and Experiment Planner. POWGEN is the initial
planner beamline. Planning needs a nuclear CIF, instrument configuration,
sample assumptions and exposure. It does not require an IPTS or a measured scan. Native calculations use managed Galaxy jobs in NDIP.
The existing Analysis context remains intact while a plan is edited.

Instrument, Sample, Transitions and Exposure form the guided workflow. The
2026B high-resolution 60 Hz bands have packaged measured responses; unmatched
historical/custom profiles offer shape previews. Isotopes, up to three
additional crystalline phases, MCIFs, magnetic subgroup moments, structural
modes, optional child cells and structural-to-magnetic promotion are supported.
There is no user-facing experimental-comparison tab.

Planner 3 reads the independent magnetic variables and their exact site-component
mapping from the prepared native constraint system. Linked/fixed components and
unit-axis metric moment magnitudes are visible as live readouts. The form no
longer asks for Landé g; parent project values are retained when present, otherwise
GSAS-II SetupGeneral provides its native default (2.0). Ionic form factors still
come from the native table. Explicit g remains supported by the scientific API.
Structural preparation now validates the zero-distortion state before exposing
controls, and reuses structural_live.inventory for affected atoms and coordinates.

Plots show capped ±1σ bars and the expected ±1σ per-bin band. Counting-only and
counting-plus-reference errors are separately selectable; an uncertainty plot
shows both standard deviations without peak heights obscuring them. The default
matches the exported predictive uncertainties. Neither choice is a complete
systematic model-uncertainty interval. Methodology is visible before prediction.

The source planning package is imported from the existing research worktree
(C64 `5342b3f61dcfa333a05f329e11e5e26c884ea4b0`). It retains the corrected
native kernel used in the frozen C43 runtime. This is an integration change,
not a new scattering implementation.

## Reused components

| Responsibility | Existing component used |
| --- | --- |
| Main UI, navigation and session lifecycle | `radar_pd_nova.app.RadarPdNovaApp` |
| Uploaded-file ownership and limits | `radar_pd_nova.uploads` |
| POWGEN profile identities | `radar_pd_nova/data/powgen_calibrations/catalog.json` |
| GSAS-II profiles, cross sections, form factors, constraints | Existing pinned native GSAS-II runtime |
| Nuclear/count forward model | `gpx_recipe.planning` |
| Magnetic subgroup service | `gpx_recipe.symmetry.discovery` and Bilbao parser |
| Magnetic model construction | Existing native magnetic/symmetry preparation |
| Structural discovery and modes | Shared ISODISTORT client, matcher and native structural worker |
| Bounded subprocesses and cancellation | `gpx_recipe.process.run_scientific` |
| Scientific environment | `gpx_recipe.jobs.scientific_environment` |
| Atomic saved-plan JSON | `gpx_recipe.execution.write_json` |
| Deployment proxy and process supervision | Existing `run_trame.sh`, `run_nginx.sh`, nginx template and supervisor config |

`native_structural.run` adds an explicit `forward_only` path, preserving the
existing refinement path. Its forward path freezes parameters, transfers the
sample settings and calls the native no-fit calculation. Planner native calculations are submitted through the managed scientific worker.
They do not refine parameters against observations.

## Persistence and boundaries

`PlanningStorageMixin` captures uploaded CIF/profile bytes, their hashes,
settings and owned native-result references. Saved selections are opaque IDs,
not filesystem paths supplied by the browser. Receipts and arrays are verified
before reuse. Engine changes invalidate prepared/result state while retaining
editable inputs. Native GPX pickles cannot be imported through saved-plan
selection. Downloaded result bundles preserve the full forward calculation.

The NDIP planner saves portable, owned bundles and operation checkpoints to Galaxy History.
A fresh session can recover them; uncertain submissions are reconciled without resubmission.
Local temporary files remain isolated per user and History.
Set `RADAR_PD_PLANNING_ROOT` to a persistent **per-user application workspace** for standalone local use.
The local launcher bind-mounts one workspace at `/work`; default planner files
are under `/work/powgen-planning`. Saved plans survive container restarts on
that volume. The standalone launcher can retain local snapshots. Portable archives can be
reopened through the UI as data; native GPX authority requires the original
verified Galaxy receipt or fresh preparation. Do not share one planning
root between independent users.

Asynchronous preparation and simulations own their cancellation event and
input generation. A stale completion cannot publish into a changed plan.
Input edits hide old predictions. Closing a session while a native job runs
requires waiting for it or using the planner's Cancel control. Service calls
have 60-second request timeouts and bounded retries.

## Reviewed input handoffs

The planner remains a separate workspace. Its Use this plan controls create a
reviewed draft for preparation or Live analysis, retaining the owned parent
CIF, profile identity, source-bank mapping and calibrated coordinate range.
Applying a draft does not submit jobs, replace measured data with a simulation,
or change an already-running monitor. A transition requires an explicit choice
to use the original nuclear parent. The full intent archive retains settings
that a destination cannot yet express; such settings block the affected handoff
instead of being silently omitted. Preparation still performs native inspection
before expensive generation, and Live still resolves actual facility inputs.

## Local build and launch

```sh
docker build --platform linux/amd64 -f nova/dockerfiles/Dockerfile.planner-local \
  -t radar-pd-planner-local:0.7.0 nova
python3 nova/tools/planner_local.py start --work-dir ./planner-workspace
```

The launcher exposes only `http://127.0.0.1:8099/index.html`. `status` and
`stop` use the same `--work-dir`. Use `start --replace` after updating the image;
the workspace is retained. Linux hosts should grant the container's `radar`
user write access to the workspace. The image is linux/amd64; Docker Desktop
uses emulation on this Mac.

The standard image command remains the existing NDIP supervisor/nginx launch
on port 8081, using `GALAXY_URL`, `GALAXY_API_KEY`, `HISTORY_ID` and `EP_PATH`.
The local launcher overrides only the process entry point to run Trame directly
on port 8080 without Galaxy credentials. The regular publishing workflow builds the thin production Dockerfile,
including the existing U-Net assets. The native scientific worker carries the
planner engine and matching profile data. Dockerfile.planner-local is an
explicit development option and is not the production image.

Live Bilbao discovery uses deployment-provided `BCS_API_KEY`. ISODISTORT needs
HTTPS access to BYU. On an intercepted organization network, supply a complete
CA bundle with the organization's already-trusted certificate via the local
launcher's `--ca-bundle` option (or standard `REQUESTS_CA_BUNDLE` in deployment).
TLS verification stays enabled. No keys are included in this source or package.
Known CIF/MCIF calculations do not require either external discovery service.

## Verification and scientific limits

The integration audit contains 346 passing planner/application/symmetry/session
regression tests; 12 passing native installation campaigns; and byte-identical
arrays in all 106 numerical files compared with the frozen planner baseline.
Native controls check zero magnetic moments, squared-moment scaling and
zero-mode structural recovery. Real Trame controller checks exercise saved
magnetic, structural and combined states with recorded external responses and
HTTP explicitly forbidden.

The authorized live reference checks also pass: Er₂Be₂GeO₇ at k=0 yields 25
Bilbao subgroup embeddings; YMnO₃ at k=(1/3,1/3,0) yields 32 ISODISTORT
directions, six constructed in the declared highest-symmetry-per-irrep tier.
Selected states pass native preparation, simulation and save/reopen. The
combined child-magnetic service response is tested from a captured catalog,
not newly queried live in this integration check.

These checks establish software behavior and preservation of the existing
calculation. They do **not** establish publication-level absolute intensity or
general experimental agreement. The 2026B count model remains conditional;
reference powder mass/fill-height records lack independent verification.
Exhaustive structural branch construction and automatic retrieval of newly
published cycle calibrations are not implemented. Galaxy History persistence,
managed jobs and reviewed draft handoffs are part of the integration candidate
and require deployed end-to-end acceptance. Independent experimental qualification remains separate work.
