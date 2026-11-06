# Joint structural search

The structural workflow adds a nonlinear backend to **Refinement → Model
exploration → Structural displacement modes**. It uses GSAS-II for diffraction
and instrumental profiles, and shares the existing campaign ledger, Galaxy
admission policy, process cancellation, independent audits and result viewer.
It does not replace RADAR phase discovery, custom nuclear recipes, magnetic
Hessian/grouped optimization or live magnetic posterior analysis.

## User workflow

1. Select the target RADAR or recipe GPX and inspect its parent.
2. Choose **Structural displacement modes**. The application uses one complete
   coordinate workflow; an optimizer choice is not required.
3. Optionally select a well-refined **Parent reference GPX**. Its named parent,
   exact setting, site labels, species, occupancies and isotope selection must
   agree with the target. The reference supplies fractional coordinates and
   ADPs; the target supplies observations, instrument, parent cell, background,
   sample terms and retained secondary phases.
4. Declare a rational k, discover a bounded catalog with ISODISTORT and review
   the selected children. Discovery still uses the existing Method-2 scope;
   it is not an exhaustive enumeration of every supercell or mixed-irrep group.
5. Review displacement/cell bounds, starts, numerical allocations, contact
   assumptions, exact scans and whether to follow newly published matching
   GPXs. Approve the finite plan, then start it.
6. Inspect each fitted curve, residual, reflection ticks, mode amplitudes and
   disposition. Download or open the exact independently audited GPX in hosted
   GSAS-II. These point estimates remain **optimization unresolved** rather than
   being promoted to unique structures or native convergence certificates.

New UI plans retain every independent physical mode in each included child.
The older selected-mode executor is retained internally to read and recover
sealed historical plans; it is not an alternative new-setup option.
Only demonstrated uniform-origin gauges are
removed. Gauge reference holds in exported GPXs preserve that physical space.
One campaign work unit owns a complete child and its internal starting portfolio;
there is no exponential expansion of sign combinations into separate Galaxy
jobs. Independent children and scans use the existing bounded worker pool.

## Scientific scope and safeguards

- One transforming, commensurate, nonmodulated nuclear phase per selected
  histogram. Fixed secondary nuclear phases are retained through calculation
  and export. Each bank is analyzed independently.
- All included independent displacement and symmetry-allowed lattice
  coordinates can vary together. No mode is discarded because it scatters
  weakly. Numerical failure is not subgroup exclusion.
- Occupancy, ADPs and instrument parameters stay fixed. The common histogram
  scale is bounded to exp(±0.3); a bounded additive correction to Back;0 varies.
  Higher background coefficients and additional background terms are retained.
  The actual nuisance bounds are recorded in each fit receipt.
- The contact screen uses versioned, checksum-bound MP/COD aggregate minima
  as explicit assumptions. These are not universal physical laws or model
  probabilities. Unqualified partial-occupancy contacts fail for review.
- Source phase/HAP constraints, parent restraints, nonzero parent HStrain,
  anisotropic sample broadening and nontrivial texture require additional
  transfer implementations. They are refused rather than silently stripped.
- Exact-cell native response reuse is independently checked. Exclusions,
  masked grids, retained secondary phases and nonlinear extinction use a full
  native fallback instead of relaxing the reconstruction tolerance. Models
  above the 256-coordinate response-cache limit also use the complete native
  calculation; their coordinates are not truncated or rejected by that cache.
- For the reviewed single-phase, unmasked PNC/PNT native driver, the service
  records reflection responses during an already-required Rietveld evaluation.
  Original native peak functions, structure factors and constrained mode
  derivatives supply the calculations. No division by parent F² is used, so
  a zero-intensity reflection can acquire intensity under distortion. Responses
  belong to an exact cell and fixed profile context; the owned numeric cache
  has a 64 MiB limit. Unknown native drivers and unsupported contexts retain
  full native evaluation, with a recorded fallback reason. Temporary lattice
  finite-difference probes do not construct unused response matrices. Every
  accepted displacement proposal and independently reopened GPX still receives
  the existing native comparison. This reduces duplicate calculation but does
  not precompute general nonlinear cell/profile changes or certify a global
  minimum. Native context ownership requires separate processes for parallel
  candidates, never concurrent GSAS-II adapters in one process.
- Starting-curve construction reuses the exact fixed-cell native profile for
  displacement-only proposals. It uses the same required-pass capture engine
  as the shared fit service; it does not construct a separate Pawley response.
  Cell changes still use complete native calculations. All starting points,
  contact checks and secant-step choices remain unchanged. The earlier native
  reuse schedule measured 52.45 → 33.38 s preparation in one Linux/amd64 TOF
  control, with all 69 curves agreeing to 1.8e-15 relative error. That historical
  measurement does not measure this newer schedule or complete workflow latency.
- An independent saved-GPX calculation must agree within 1e-8 relative error
  and 0.01 supplied standard deviations. Export verifies active mode variables,
  origin holds, cell freedom and fixed instrumental values, with no stale
  covariance. Ordinary GUI continuation does not enforce the contact screen or
  the original search box; this limitation is recorded in its contract.
- Both saved GPXs are reopened independently to verify measured intensities,
  supplied variances, likelihood bins, Rwp normalization and weighted cost.
  Extra likelihood omissions become exact-bin native exclusions. Valid
  measured weights outside the current fit interval are retained for later
  changes of limits. A different normalization convention is refused.
- Cell-expansion admission runs before either structural executor begins
  native work. The worker audit reconstructs the physical displacement/cell
  bounds and complete active set from the sealed parent and child CIF; it
  independently checks positive bounded scale and constant background.
- Native GSAS-II Rwp and the fitted-bins-only diagnostic are separately named.
  Excluded strong peaks remain in native Rwp normalization. Neither value is a
  transition probability, model posterior or automatic counting-stop rule.

## Shared completed/live API

`gpx_recipe.structural_live.service.prepare_model` prepares geometry and native
starting curves once. `fit_snapshot` fits one complete reduced observation.
`StructuralSession.update` replaces observations and carries only starting
parameters forward, never a previous cumulative likelihood. It rejects stale
or changed same-revision observations, grid/protocol changes and exposure
resets. A failed update leaves the previous publication intact; duplicate
updates do not run another fit.

Restoration, duplicate returns and warm starts verify the sealed receipt and
both GPX digests. Artifact manifests use confined relative paths so a session
can be relocated without rewriting its sealed scientific receipt. A missing
or changed artifact requires repair/re-export; it is not silently refitted or
treated as a reusable publication. This session serializes updates within one
owner instance, not across independent processes.

Optional recovery deadlines retain the independently evaluated incumbent.
The receipt names budget-expired stages. Native identity, objective and physics
errors still propagate and prevent publication; recovery is not a general
exception suppression mechanism.
Recovery uses the same owned response evaluator for displacement derivatives,
anchored by a complete native curve at every derivative request. Its scores,
cell secants and acceptance checks remain original native calculations.
Unsupported contexts retain the full native derivative path. A recovery receipt
records the evaluator counters at admission and afterward, since its response
cache can have been populated by the main fit. The final reported objective and
Rwp follow the retained recovery result, rather than the earlier main fit.

The reduced-profile adapter selects one bounded, process-local peak store for
fixed-cell, fixed-profile workloads; the Model magnetic residual adapter shares
that store across fresh native contexts in the same process. Each context has
its own accounting and independent native constraint ownership. Changing counts,
weights, moments or nuisance coefficients does not copy an earlier likelihood;
only identical peak-function arguments and grid windows can reuse numeric shapes.
The aggregate shared store has one 64 MiB allocation and one entry ceiling.
Native implementation changes clear its qualification and contents. Inherited
adapters detach from a parent process's store before reuse; this does not make
forked GSAS-II state a supported concurrent execution mode. Spawned workers own
independent stores. No file transport or cross-worker cache is introduced.
Variable-cell/profile workloads keep original native rendering: a
complete 185-start control exposed a regression hidden by a shorter rendering
test. This selection does not alter the scientific parameter set.
The memo supports the reviewed native PNC/PNT peak routines.
It reuses only exact scalar arguments and exact grid windows; grid contents
are hashed on every native driver call. Reflections, intensity corrections and
native summation order remain unchanged. No interpolation or uploaded numeric
predictions enter this cache. Exclusions, secondary phases and extinction still
use their full native equations. Unknown drivers, native multiprocessing and
active response-capture hooks bypass the memo, and temporary functions are
restored on exceptions. Each adapter reports its own reuse and bypass counters,
separately from process-store occupancy. The 64 MiB accounted-cache ceiling and
entry bound do not describe total native process memory. Periodic bounded probes compare estimated native work saved
with measured bookkeeping cost. An unprofitable driver call immediately
returns to the original peak functions; later driver calls can probe again.
The bypassed native tail is not included in the observed peak-call counter.
This can reduce repeated rendering, but does not itself solve
portable preparation or nonlinear cell/profile adaptation.

Variable-cell/profile calls can independently avoid masked-array temporary
objects when the reviewed native driver receives an entirely unmasked,
contiguous float64 axis. `NativeUnmaskedAxisView` passes a zero-copy plain
array view to that original driver; it preserves the reflection loop, native
kernel calls, summation order and owned histogram arrays. Real masks,
unusual storage, native multiprocessing and active response-capture hooks
keep their original arguments. Temporary driver wrapping is restored on
exceptions. Explicit zero-byte native reference controls do not enable this
optimization. Receipts report its selection and conversion/bypass counters
separately from peak-cache accounting. This does not make a prepared bank
compatible with a changed cell or profile.

For reviewed single-phase PNC/PNT structural contexts, lattice proposals can
stream the original native peak derivatives together with native reflection
geometry and intensity probes. One complete native anchor restores the
scientific state. The probes' unrendered arrays never enter an objective,
screening score, observation, exported GPX or uncertainty calculation.
Derivative arrays are accumulated immediately under a bounded workspace;
they are not retained for every reflection. A reflection entering or leaving
the native inventory makes that entire lattice direction use original native
differences. Unknown native code, extracted intensities, secondary phases,
masked contexts, nonlinear extinction and unsupported peak conventions also
retain original differences.

These directions are optimization proposals, not qualified exact Jacobians.
Every accepted step uses the complete native objective with freshly profiled
scale/background coefficients. Stalled or apparently stationary proposals,
including the outer small-progress stopping rule, require a fresh native
cell-direction retry. Deadlines between probes restore the genuine native
state before returning. Receipts distinguish scalar probes, full native
anchors, membership fallbacks and reused directions. This is one automatic
execution policy; the force-native argument is an internal regression oracle,
not a user accuracy/speed choice. It does not establish portable structural
acceleration, global optimization, covariance validity or faster whole-catalog
execution by itself.

`ReducedObservation` accepts unknown exposure. Its `mask=True` means usable;
the snapshot adapter explicitly accepts `invalid_mask=True`. Reduced intensity
and supplied variance are never converted into invented counts. Verified raw
counts retain a separate count/exposure contract.

The shared acquisition adapter is
`gpx_recipe.structural_live.acquisition.prepare_acquisition` followed by
`StructuralAcquisition.update`. It runs in an owned scientific process under
the existing WebMON acquisition owner; it has no source polling loop.
Magnetic preparation and structural acquisition share
`gpx_recipe.live.nuclear_transfer.prepare_nuclear_snapshot` for transferring
the reviewed nuclear reference onto the producer's uniform TOF grid. The
original nuclear GPX and instrument remain immutable. Model-specific magnetic
site/catalog review stays separate from structural displacement construction.
Released instrument refinement flags in a recipe output are deliberately held
for transfer; both initial and current numerical X/Y/Z values are preserved.
The final source check uses the originally requested GPX checksum, so an edit
after native loading cannot acquire a new reference identity unnoticed.

The acquisition contract binds the prepared model, native grid/fit exclusions,
reviewed exporter geometry and normalization, physical bank, producer identity,
run/sample scope and spectrum marker. The owner supplies an archived snapshot
receipt and an explicit monotonic revision. Signed reduced values and supplied
uncertainties are preserved; exposure may remain unknown. Newly measured bins
can replace zero-weight placeholders without rebuilding physical preparation.
Every snapshot supplies its own likelihood and native normalization masks.
Changed grids, spectrum markers, banks or producer/sample declarations require
review. A declared mapping is not independent evidence that a live producer
actually retains that configuration.

Snapshot input lineage is sealed before fitting. POSIX writer ownership and a
fresh durable-ledger read prevent recovered owner instances from overlapping
or replacing newer publications with stale updates. Duplicate delivery verifies
the existing receipt and both GPXs, without another nonlinear fit. Restoration
also recomputes the observation identity from its archived raw snapshot. A
failed new fit leaves the previous publication intact. Only previous parameter
guesses carry forward, together with the existing complete starting policy.
Point-fit results carry no structural posterior, model probability, exclusion
permission or counting-stop recommendation.

Acquisition contexts contain measured experiment data. They belong to private
scientific job/History artifacts, not portable theory-only preparation libraries.
The closed `structural-snapshot` native operation now carries a self-contained
or shared-input reviewed Model job, its optional parent-reference GPX, reviewed anchor snapshot,
current snapshot/receipt, producer mapping, scope and monotonic owner revision
through the existing Model Campaign transport. `webmon.structural_runtime`
uses its submit/poll/recover/collect methods; it introduces no acquisition
poller, posterior engine or second job queue. A fresh viewer restores exact
captured inputs before accepting a result. Complete mode inventory, package,
parent setting, snapshot bytes, histogram, publications and source lineage
are checked. The native view contains the current curve and Bragg ticks.

Optional preparation metadata is captured alongside the model inputs. Original
receipt HDAs are routed through the same authenticated origin lookup and fixed
Galaxy-template metadata as Model exploration. A copied manifest does not
attest its own curves. Missing authority or incompatible physical context keeps
the existing native requalification/rebuild policy.

Each operation currently reconstructs an owned structural acquisition context;
it does not yet transport a qualified persistent native context across jobs.
Original prepared banks may help only where their actual-context compatibility
and producer origin checks pass. This is not a claimed fast streaming policy.
The future Galaxy template publishes independently checked point-fit GPXs in
a separate `reviewable_projects` collection, explicitly requiring optimization
and scientific review. The qualified-model collection keeps its prior meaning.
The Live acquisition workspace now offers **Structural distortion models** in
its Live analysis selector. Select a published nuclear GPX, save its reviewed
producer mapping, then choose **Set structural models in Refinement**. Inspect
the parent and optional parent reference in Model exploration, discover the
children, and build and approve the finite plan. Return to Live acquisition and
choose **Use approved structural plan**. Adoption captures all reviewed models
with shared immutable source inputs and starts no scientific jobs.

**Fit current structural snapshot** fits once. **Follow structural updates for
this run** uses the existing acquisition polling and one-active/latest-pending
queue; it introduces no second acquisition service. Pause preserves the latest
pending revision. Stop waiting leaves accepted Galaxy jobs recoverable; recover
them before another snapshot is admitted. Editing the producer mapping or
scale/background-transfer choice requires adopting the current plan again.
The table accounts for every selected model, and native fits become available
as each completes, with calibrated d/Q/TOF axes, Bragg ticks, cell/mode point
values, a continuation-GPX download and **Open continuation in GSAS-II**.
The desktop handoff captures the exact checked GPX selected at click time and
uses the existing hosted desktop transport. Save and close an existing desktop
before opening another project. Original RADAR-PD fits stay in Results.

Local native transport and public artificial UI controls do not establish
deployed acceptance. Cross-job preparation amortization, sustained live queue
latency and complete real-experiment/browser acceptance remain required.
Likewise, Method-3 catalog enumeration, calibrated structural posteriors,
counting advice and named-mode custom recipes remain separate work. Current
custom recipes accept nuclear projects; magnetic posterior checkpoints are
offered as verified GPX downloads for GSAS-II rather than sent to an incompatible
recipe inspector.

## Packaging and qualification

### Cold prepared-bank provenance

The preparation worker can publish a compact, data-only native-origin receipt
alongside its ordinary result ZIP. The fixed Galaxy command announces that
receipt's checksum in job stdout; scientific subprocess output remains in the
console dataset. A later tool invocation obtains creating-job metadata through
the Galaxy template, separately from its uploaded scientific request. A JSON
field claiming native verification cannot establish this provenance.

For a compatible current structural context, an original receipt from the reviewed
Model Campaign wrapper permits reuse of the complete prepared bank after three
fresh native profile probes. The reader still checks every declared start,
chemical feasibility, array shape, grid, checksum and physical context. Every
subsequent fit and exported GPX retains its independent native acceptance
checks. The probes check the current reader/context; the producer evidence
binds the complete bank, including the curves not selected as probes.

For a single nuclear phase with zero extinction and an unmasked PNC/PNT grid,
the bank also retains its native reference background and an identity with
only intensity normalization and existing polynomial background coefficients
removed. A later nuclear fit can change histogram scale, the single-phase HAP
scale or those background values. The reader then uses the exact affine
relation `(prepared_curve - reference_background) * scale_ratio + current_background`.
The scale ratio includes both native intensity-scale factors. The current
background is evaluated by GSAS-II; it is not fitted to the prepared curves.
Native scale floors, a conservative foreground roundoff bound and the complete
remaining dependency identity guard this operation. This does not change any
fit freedom or copy experimental observations into preparation. Recorded
proposal steps remain heuristics from the preparation reference.

The background definition/order, cell, displacement basis, occupancies, ADPs,
instrument/profile values, grid, masks and chemistry remain dependencies.
Secondary phases, nonlinear extinction, exclusions and unsupported background
terms retain complete native reconstruction. This is one automatic policy,
not a user-selectable approximate solver. The qualification receipt records
any affine nuisance binding and its numerical error bound. Origin checks and
fresh native probes apply after rebinding too.

If the receipt is absent, changed, inaccessible, produced by an unreviewed tool,
or incompatible with the current scientific implementation, the normal native
reconstruction remains in force. A changed physical context also rebuilds.
Receipts do not make another cell, profile, grid or unsupported baseline compatible.
Reuse/reconstruction state and native-probe errors are recorded in
`starting-curve-contract.json` and the resulting structural service receipt.

Portable libraries retain only bounded receipt lookup references. NOVA uses
the current user's authorized access to the same Galaxy site to select the
original small dataset; it never follows a site URL supplied by a bundle.
Origin lookup failure does not invalidate checked preparation arrays. Reuse
across accounts, unavailable original Histories, sites and compatible software
upgrades still requires additional qualification; downloadable files alone do
not establish standalone trusted reuse. No API key or signing secret is stored
in a library, receipt, numerical request or child-process environment.

The future Galaxy template introduces these inputs/outputs at Model Campaign
0.2.17. Updating that template does not activate an image or alter the deployed
tool XML. A coordinated immutable release and actual Galaxy acceptance remain
required.

Runtime code never imports benchmark runners or laptop paths. Superseded
interpolation engines and independent prototype schedulers were not installed.
The packaged contact files, policy and source code are bound by the scientific
package identity, including during relocatable sealed-job recovery. The shared
CIF occupancy loader is inside that package; the script entry point is a
compatibility import so the batch and model workers use one implementation.

Release requires coordinated UI and Model Campaign images with identical
scientific package identities. Keep old immutable workers and protocols intact.
Production qualification requires independent model-recovery controls, full
exported-GPX round trips and measured complete workflow latency in the actual
deployment. Four-scan Ba4SbRu3O12 controls recover the expected C2/m candidate,
with all 80 saved-GPX audits passing across the declared local comparisons.
They also expose a peak-width confounder: a warm-scan X/Y/Z adjustment materially
improves the parent fit. The C2/m improvement is much larger in the cold scans,
but missing impurities, conditional ADPs and competing lower-symmetry fits
prevent a unique structure claim. The new image still needs NDIP-test acceptance.
Successful process exits or excellent Rwp on one case do not complete that
qualification.

## Integration validation

The initial integration was checked against the NOVA, campaign, recipe
and CIF contract suites (5,196 passing tests; 19 environment-dependent skips).
The focused repair suite adds 67 passing contract/controller tests with no skips.
Separate offline Linux controls use the deployed scientific image and the
built wheel. They exercise a separate parent reference, a strong excluded
peak, a retained secondary phase, a TOF child with 27 physical displacement
coordinates plus six cell coordinates, and two sequential reduced updates.
The latter verifies prepared-start reuse, restored-session duplicates and
stale-revision rejection. Each successful export is checked by an independent
GSAS-II calculation; these synthetic controls establish contract behavior,
not universal experimental model recovery or deployment latency.
The packaged wheel passed 27 real GSAS-II structural and magnetic controls.
The final nine-case structural acceptance gate runs without skips, including
nine invalid-output injections. A deliberately perturbed two-mode export was continued
with native least squares, recovering its original amplitudes with a finite
seven-variable covariance matrix while preserving the source checkpoint.

The independent review of `088cb51` exposed gaps in observation binding,
admission, independent output audits, optional recovery, saved-artifact reuse,
cache fallback and hidden UI fields. Regression controls now cover these
failures. The Linux release gate generates a synthetic Cmcm ErSi experiment,
executes a sealed compiled worker, audits/export/reopens its GPX, and continues
native refinement after a perturbation. It also tests an extra excluded
outlier, a tiny recovery budget, preserved out-of-range weights, and injected
scale, cell, mode, background and active-set violations. This dedicated gate
requires actual execution with no skipped cases; external fixture variables
are not needed. These controls do not establish experimental model recovery.

The subsequent user-journey review exercised four real POWGEN scans through
discovery, profile resolution, preliminary fitting, five-stage LLM-assisted
nuclear recipes, per-scan fits and temperature trends. It exposed sample-name
parsing, stale warning, transport-status and structural-form presentation bugs.
The local fixes passed 1,027 focused setup/model checks and 517 focused
refinement/UI checks; those suites overlap. The broader host run had 5,146
passes, 19 skips and two local process-control failures. Both failures passed
their focused rerun after replacing a startup-time assumption with a ready-file
handshake and allowing process visibility. The final scientific wheel passed
seven native structural controls without skips, including direct/reused
starting-curve equivalence and a sequential distortion, cell-shift and
return-to-parent series. Experimental structural model recovery and the
coordinated NDIP image release remain outstanding.

Experiment preparation accepts bounded TOF ranges through 2,000,000 µs.
POWGEN reduced patterns can extend beyond the simulation convenience API's
historical 200 ms units guard. For such ranges, preparation retains the native
instrument/HAP initialization and creates the requested empty logarithmic
grid before any scientific calculation. The upper bound is the exclusive
TOF grid edge. Observations are never resampled or synthesized by this path;
actual-grid compatibility and native forward/export checks still apply.
