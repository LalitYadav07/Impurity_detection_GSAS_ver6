# Structural and commensurate magnetic model searches

Status: isolated development, 2026-10-28. This is not enabled in NOVA, the
refinement planner, or NDIP. The deployment checkout is separate.

## What the new capability should answer

Given a credible nuclear parent, an explicit propagation vector in that
parent's reciprocal basis, and selected observed scans, compare a declared
portfolio of symmetry-compatible models. Return supported and unresolved
hypotheses, their fitted patterns, distortion amplitudes or ordered moments,
and the observations that distinguish them. A good fit alone does not identify
a unique space group or prove a transition.

A k-vector does not select an irrep, an order-parameter direction, all star
arms, or a unique child. Two children can have the same space-group number and
different mode content, embeddings, or atom mappings. Gamma-point transitions
can alter existing peaks without adding a superlattice peak. Coupled irreps
can require more than one k-point: the ISODISTORT SrTiO3 example obtains Pnma
from R4+ and M3+, rather than from a single irrep.

## Evidence gathered in this branch

The 2026-10-28 native validation extends the original import-only feasibility
check. Evidence lives in `audit/2026-10-28/symmetry-native-validation/` in the
surrounding workspace; `report.html` and `validation-summary.json` summarize it.
The experimental compiler/runner remains outside the application. No search is
enabled in NOVA or deployed to NDIP.

| Check | Result | Scientific limit |
| --- | --- | --- |
| LaMnO3 public neutron tutorial, 51 k=0 subgroups × three child-axis starts | 153 terminal records: 33 site/orientation rejections, 60 qualified numerical fits, 60 rejected fits; 85 s at four native workers; full corrected Linux rerun 182.5 s including per-seed container startup | One controlled engineering parent derived from a polished public project, not 153 independent experiments or blind discovery; axis starts do not exhaust canting/coupling hypotheses |
| Independent reopening of native magnetic outputs | 120 GPXs, 1,200 checks of actual varying parameters, data/weights, masks, fixed nuclear fields and declared material links | Numerical/parameter correctness does not establish a unique magnetic structure |
| Pnma ↔ Pbnm physical-direction control | Corrected scripts reproduce Rwp 6.2985% and Mn moment 3.654 μB in both settings, on native host and pinned Linux runtime; another physical direction fits worse | No general conversion of arbitrary imported mode tables or all setting conventions |
| Site compatibility: 51 Pnma k=0 and 38 hexagonal Ba/Co k=(0,0,1/2) candidates | 178 checks for explicit `all`/`any` policies on each runtime, plus wrong-setting/missing-site negatives; synthetic anti-inversion control distinguishes partial ordering | Nonzero-k candidates have site preflight, not a completed experimental magnetic search |
| SrTiO3 X-point nuclear modes | 70 OPDs cataloged, 24/24 expanded-tier and 5/5 balanced-tier children native-normalized; balanced tier fitted with 18 signed starts on noiseless and Poisson-noisy synthetic data | Five fitted children, not all 70 OPDs or a general structural validation corpus; public truth was known before the noise-control test |
| Actual structural outputs | 36 native GPXs reopened; seven data/parameter/mode checks each; noisy X1+ solution gives normalized modes 0.1004 and −0.2988 Å for synthetic 0.100 and −0.300 Å | Noisy engineering fixture is not a blind experimental holdout |
| Zero-amplitude structural control | All five children reproduce the parent calculated profile to relative error below 1.2e-15 after exact metric transfer and volume-aware amount linkage | Rounded service-CIF cells initially spoiled the two Cmcm comparisons; do not use rounded cells as the parent control |
| Noiseless optimization control | Two near-exact fits abort at the numerical floor; portfolio reducer reports `optimization_unresolved` instead of selecting a worse stable minimum | A correct-looking low residual is not accepted by weakening convergence checks |
| Magnetic nonorthogonal/supercell forward control | Physical moment norms and exact lattice cosets verified; calculated representation invariance passes at 2.2e-14 relative error in pinned Linux GSAS-II 5843 | Host 5841 fails this same test at 9.9%; runtime identity and forward controls are required |

The magnetic parent is intentionally well conditioned: nuclear parameters come
from a public polished tutorial GPX and are fixed. This isolates magnetic
setup effects. Five distinct magnetic candidates produce numerically identical
best powder curves. Preserve the portfolio, not a unique MSG claim.

A first structural fit used an R-point catalog against an X-point synthetic
pattern. The mismatch was discovered during this investigation, not a planned
blind negative. Its failed/worse fits are retained. Both corrected X-point
protocols use the same finite signed-start policy; no LLM intervenes in any fit.

Focused checks cover exact rational k conversion, coupled allowed moment
components, magnetic metric/handedness, integer supercell cosets, complete
setting fingerprints and conservative incomplete/failed-fit ranking. They are
supplemented by the native GPX audits rather than used as substitutes for them.

## Corrections found by native testing

- The reference parent-to-child moment conversion used the wrong matrix
  transpose for a nontrivial Pbnm transformation. `gsas_magnetic_preflight.py`
  implements the passive axial-vector conversion for GSAS-II unit-axis
  components, verifies physical magnitude and tests handedness explicitly.
- Euclidean component length is wrong for nonorthogonal native moments. Use
  the direct unit-axis metric; `[1,1,0]` in a 120-degree hexagonal cell has
  magnitude 1 μB, not sqrt(2) μB.
- Bounding translations by the largest basis-matrix entry can omit atoms from
  rotated supercells. The new integer-coset utility covers exactly abs(det T)
  lattice classes. Native site preflight rejects lost selected-site density;
  the executor must use a verified complete constructor, not silently proceed.
- A saved GPX can retain a stale calculated curve after its magnetic phase is
  removed. Recalculate the nuclear baseline before deriving residual windows.
- A failed lower-residual start must remain visible. `symmetry_results.py`
  separates numerical uncertainty, hard physical rejection, technical failure
  and deferred coverage; qualified outputs require actual input/effect audits.
- Native covariance `nv-*` values are not a safe substitute for absolute
  normalized ISODISTORT amplitudes. Recompute from final atom coordinates,
  native mode matrices/offsets and normalization. The structural audit does so.
- Service-CIF rounding is distinct from a change of setting. Preserve the raw
  CIF and transfer the exact parent metric through the recorded basis for the
  zero-mode control; do not edit mode tables to conceal a mismatch.
- The pinned Linux 5843 runtime already corrects the magnetic projection found
  wrong in host 5841. No GSAS-II installation was patched. Record module hashes
  and require the forward control in the actual image; a version string alone
  is not sufficient proof for another build.

The existing name-only CIF adapter still refuses origin/basis conversion of
mode tables, magnetic/occupancy/rotational mode imports and unsupported strain
compilation. Original service CIFs are never overwritten. Studio supplies
scientific reference/template knowledge at development time, not a worker or
application runtime dependency.

## Runtime boundary: setup once, execute without an LLM

1. **Prepare deterministically.** Select a source GPX revision, target material,
   representative parent/changed scans, compatible banks and a parent setting.
   Resolve k as exact rational components with an explicit basis. Preserve an
   incommensurate or ambiguous input for review rather than rounding it into a
   large commensurate cell. Query/cache the external candidate catalogs and
   child structures, then inspect the native GSAS-II inventories.
2. **One setup LLM call.** Supply the user request, compact native project
   snapshot, exact candidate/mode IDs, and a small set of supported scientific
   operations and examples. Produce a typed, bounded protocol. No arbitrary
   Python, fabricated mode names, or unconstrained fallback instructions.
3. **Review the full protocol.** Show the candidate tiers, free/fixed parameters
   at each stage, seed set, maximum jobs/cycles, conditional expansion rules,
   physical bounds, rejection criteria and treatment of indistinguishable fits.
   The approval binds hashes of source, catalog, protocol and runtime.
4. **Compile and execute.** A host-owned compiler produces the script/job graph.
   External discovery can be prepared in advance, so native refinement workers
   can run offline. There are zero LLM calls during refinement, ranking,
   retries, candidate expansion or processing later scans. A missing required
   branch or unexpected parameter state stops with evidence.

The protocol is allowed to contain fixed if/then rules, for example: run the
declared second OPD tier if the first tier is inadequate; retry each failed
seed once with a specified cycle limit; or keep every candidate inside an
explicit evidence-equivalence band. These rules must exist before execution.
Exhausting a budget means incomplete coverage, not rejection of the physics.
The user can later approve a new protocol; a running job cannot invent one.

## Structural branch protocol

Each candidate is identified by parent hash/setting, k/star choice, irrep(s),
OPD, transformation, origin and authoritative child-CIF hash. Space-group
number alone is not an identity. Conventional-cell determinant, primitive
lattice index and subgroup index are recorded separately.

The first implementation should support nuclear displacement modes only:

1. Refine or reuse a controlled parent baseline on each selected scan. Fix the
   instrument calibration/profile and use the same masks, weights, other
   phases, background policy and absorption/texture policy for every candidate.
2. Transform/import the child, transfer the parent's atomic and histogram
   state, and explicitly zero all modes. Verify that its calculated pattern
   reproduces the parent control under the transformed cell and scale. Compare
   the calculated profile and chemical inventory, not merely two Rwp values.
   A failed equivalence control is a technical failure.
3. Use native GSAS-II ISODISTORT constraints and exact `New Variable` IDs. Test
   a finite signed, nonzero start set for the permitted primary modes. Raw XYZ
   variables are dependents of these modes, not an extra independent refinement
   freedom. Zero-only starts can hide a mode whose intensity derivative vanishes.
4. Refine the allowed child metric separately from any explicit strain-mode
   parameterization. Never free both redundant descriptions. Add only the
   secondary modes or mode pairs already listed in the approved protocol.
5. Audit the *actual* native varying/dependent parameters, convergence,
   covariance/SVD, bounds, geometry, occupancies, ADPs, target-peak residuals and
   complexity. Keep parent, metric-only and declared nuisance controls.

The comparison must preserve impurity phases and the possibility of a
parent/child coexistence model. That is a separate branch with its own amount
constraints, not evidence obtained by combining two independent child fits.
An initial highest-symmetry-per-irrep tier is triage. Its success is not a
claim that all OPDs, secondary modes or coupled-k alternatives were tested.

## Commensurate magnetic branch protocol

This needs a separate executor. The current nuclear recipe worker deliberately
rejects magnetic/modulated projects; removing that check would not implement
magnetic refinement.

Use the exact nuclear setting/operators to obtain subgroup transformations.
For each candidate, derive the allowed magnetic Wyckoff-site components and
map parent atoms to child orbits. Retain BNS identifiers, basis, origin and star
choices. Filter sites that cannot carry a nonzero moment. Distinguish those
symmetry-incompatible sites from a refined zero-moment outcome.

Before execution, specify magnetic elements/ionic form factors, finite nonzero
moment seeds, permitted collinear/component/canted hypotheses, site couplings,
domain treatment and material-specific moment bounds. Principal-axis seeds
alone are not a complete magnetic search. Transform moments as axial vectors
using native GSAS-II conventions; nonorthogonal crystallographic components
are not arbitrary Cartesian components.

Nuclear and magnetic contributions describe the same material. Their amount,
cell, coordinates and ADPs must be linked consistently. They must not both
enter the ordinary sum-to-one material phase-fraction constraint as independent
materials. Magnetic moment amplitude and magnetic volume fraction are also
degenerate without extra information: do not freely refine both. Stabilize the
nuclear model first, then refine the declared magnetic freedoms and inspect
magnetic-sensitive regions and nuclear controls. k=0 magnetic order can add
intensity to existing reflections; requiring new peaks would miss it.

## Linking to live RADAR-PD

Keep RADAR-PD's source GPX and original results immutable. Store every search
as a named branch linked to the source scan, material, bank, instrument hash,
catalog and approved protocol. The existing transition detector can nominate
scan intervals to inspect, but it cannot prove a space-group assignment.

Initially use selected representative scans. After a model portfolio is
reviewed, apply the same approved portfolio to existing selected scans and new
compatible arrivals. A new bank, source setting, instrument calibration or
material signature requires a new compatibility check. Retain a cold/parent
control alongside any approved warm-start path so history does not force a
previous model onto a new regime.

Place model exploration inside Refinement, with Structural and Magnetic search
types. Show a candidate table with scope/coverage, irrep/MSG, free-parameter
count, Rwp, targeted-region evidence, physical/convergence flags and disposition.
Click a candidate to compare observed/parent/child curves and Bragg ticks in
linked zoom windows. Show mode amplitudes or ordered moments versus the same
eligible metadata axes as other RADAR-PD plots, retaining their normalization
and reference basis. Return an unresolved equivalence class when powder data
cannot distinguish candidates. Do not overwrite the observed-pattern symbols
with a supposedly proven structural phase diagram.

## Adaptation map and next acceptance gates

Diffraction Studio is source/knowledge reference, not a deployment dependency:

| Studio component | Adaptable knowledge | Required NDIP change |
| --- | --- | --- |
| `pdf/external_symmetry.py` | ISODISTORT form/API handling, Bilbao machine catalogs, provenance | Small standalone adapters, cache bundles, typed failure/coverage records |
| `structural_transition_mode_runner.py` | Parent transfer, zero-mode control, native mode starts, fingerprints and identifiability | Native worker without Studio `RunContext` or service runtime |
| `structural_transition_scripts.py` | Parent export, magnetic transforms, constrained material linkage | Separate versioned deterministic compiler and executor |
| `structural_transition_service.py` | Convergence/frozen/bounded checks, finite recovery, evidence-equivalent ranking | Pure receipt validation and fixed selection policy |

Remaining acceptance work includes independent experimental structural and
nonzero-k magnetic refinements, mixed-ion form factors, coupled irreps/k arms,
noncollinear/canted models, impurities/coexistence, multiple banks and weak or
null signals. Freeze acceptance rules before a holdout corpus. Test interruption,
job-acknowledgement loss, catalog corruption, source changes and incremental
arrivals in the eventual integrated dispatcher. These results do not establish
general structure-identification accuracy.

## NDIP execution and resource plan

The inspected continuation path is serial: `RefinementWorkspace._inflight`
allows one `run_next`, and the Galaxy runtime waits for that job to complete.
Submitting many candidates through that loop will not create parallel search.
Keep the deployed continuation path intact and add a campaign dispatcher with
separate prepare, submit, poll, collect and audit operations.

Use Galaxy's configured job destinations or dataset-collection mapping. Do not
run the exploration inside the NOVA interactive process or assume the Galaxy
Kubernetes deployment supplies scientific-job resources. A reasonable unit is
one candidate on one representative scan/bank with a small approved seed bundle;
each native seed runs in an independent process. Respect `GALAXY_SLOTS`, memory
and administrator/user/destination concurrency limits. Keep BLAS at one thread
unless the declared allocation supports more. The actual NDIP destination
capacity and quota still require verification.

Measured on the same 12 small public magnetic trials with warmed caches:

| Native workers | Wall time | Speedup |
| ---: | ---: | ---: |
| 1 | 21.79 s | 1.00× |
| 2 | 12.73 s | 1.71× |
| 4 | 7.08 s | 3.08× |

Peak per-worker RSS was approximately 215 MiB. These are laptop measurements,
not NDIP throughput promises or sizing for large/multibank GPXs. Bundle seeds
to avoid paying container/queue startup for every tiny fit. Cache candidate
catalogs and child CIFs once; refinement workers need no external network.

Persist a dispatch key containing protocol, source revision, bank, candidate
and seed-bundle identities before submission. Recover ambiguous acknowledgments
instead of blindly resubmitting. Keep deferred, failed, timed-out and canceled
branches in coverage. A worker cannot broaden the protocol, change the source,
fix its own path mapping, or invoke an LLM. Stage inputs below a job-local
working directory; absolute Mac audit paths used by the development harness
are not a deployable Galaxy command contract.

Start with two or three informative representative scans and a balanced irrep
or MSG tier. After review, continue a small approved portfolio on chosen existing
runs and future compatible scans. Avoid the cost of every candidate on every
incoming scan. Submitted jobs can survive the interactive page closing; future
scan enrollment still requires an active monitor or an independently configured
Ingress workflow.

## Portfolio presentation

The historical standalone prototype has been superseded by the actual NOVA
implementation in `radar_pd_nova/symmetry_{controller,ui,workspace,plots}.py`.
Model exploration is a subview of the existing Refinement workspace, beside
Recipe and Results. It adds no top-level application tab. Browser acceptance
uses this integrated application; the blocked historical local file was not
reopened or served through a different route.

The current overview shows complete trial coverage, residual versus independent
variable count, and calculated-pattern groups. Every model/start and terminal
reason remains inspectable, including failed and unresolved competitors. Only
profiles from the same acquisition and observed fit domain can share a display
group; matching curves do not establish a unique magnetic space group.

Selecting a group loads one observed/calculated/difference plot with phase Bragg
ticks. Native, d and Q coordinates use the source GPX calibration. Up to two
pinned alternative calculations share the same axes and zoom. Pins survive
refresh; explicit trial selection survives pending-to-terminal regrouping.
Until the user selects a trial, the view follows the best available finite fit.
GPX and trial-receipt downloads recheck the sealed artifact hashes. The receipt
contains parameter-effect audits, runtime identity and the native disposition.

When extending to several scans, use a model-by-condition evidence matrix and
mode/moment trends on verified experimental axes. Clicking a cell opens its
fit. Keep acquisition order, sample, bank and missing/failed coverage visible;
never merge incompatible banks or infer a proven phase diagram from Rwp rank.

Mode trends need a consistent parent, normalization, origin and domain gauge.
Opposite global mode signs or reversal of all magnetic moments can describe the
same powder solution. Compare invariant magnitudes when that ambiguity remains,
and retain explicit branch labels instead of drawing a false sign-change event.
Anisotropic ADPs, strain and other tensor-valued parent state require their own
verified transformation controls; the current Uiso controls do not certify them.

## Local implementation checkpoint — 2026-10-29

This is a development checkpoint on `codex/ndip-symmetry-exploration`, based on
NOVA 0.4.34 after merging the current source branch. It is not an NDIP release or a claim that the acceptance plan below
is complete. The source and Galaxy deployment branches have not been pushed by
this work. Diffraction Studio supplied audited knowledge/MIT source references;
the deployed package and native jobs have no runtime dependency on that project.

| Capability | Current behavior | Required user choice or remaining limit |
| --- | --- | --- |
| Source integration | Captures the exact completed RADAR-PD GPX and its checksum; the UI never unpickles it | Select the parent and histogram/bank; a revised GPX invalidates approval |
| Magnetic discovery | Standalone Bilbao client caches BNS IDs, complete embeddings, origins and exact rational k | Explicit parent-basis k, magnetic sites/ionic form factors, finite moment directions and fit criterion |
| Magnetic execution | Fixed nuclear/nuisance state; verified material/cell/atom linkage; moment-only native refinement | Single-k, collinear branches only; Cartesian starts do not exhaust continuous orientations or relative orbit signs |
| Magnetic representation | Independently compares all rotations, translations and time reversals to the BNS database | Grey groups are incompatible; unverified alternate settings are unsupported, never guessed |
| Reflection coverage | New UI plans explicitly use the full calibrated fitted d range by default; an explicit cutoff is optional | Historical sealed protocols preserve their original 1 Å default and are not silently reinterpreted |
| Structural discovery | Standalone ISODISTORT adapter retains the full OPD discovery record and constructs one highest-symmetry OPD per irrep in the first tier | Other OPDs remain deferred; the current UI does not yet construct arbitrary later tiers |
| Structural execution | Native displacement-mode import; zero-mode forward-equivalence check; selected modes with exhaustive signed starts; optional declared cell/background/global-scale refinement | At most six selected modes per child within the total finite-start budget; no unverified tensor/restraint transfer |
| Durable execution | SQLite ledger; sealed trials; Continue approved queue after reopening; process recovery, cancellation and deadlines | Late verified receipts resolve without resubmission. Changed implementation or ambiguous transport requires review; Galaxy seed bundles/destination adapter remain to be built |
| Streaming | Trial dispatch and partial results work without waiting for every start; acquisition IDs prevent identical GPX bytes from collapsing different scans | This UI starts one representative scan. Bulk/future-scan enrollment and reviewed continuation templates remain incomplete |
| Results | Grouped models, finite-fit overview, exact scan filtering, pins, calibrated axes, Bragg ticks, parameter values and verified downloads | Model-by-condition matrix and consistent-gauge mode/moment trends are still pending |

The native runtime gate includes the full GSAS-II/numerical/compiled-module
fingerprint and the independent BNS database/parser. The tested Linux 5843
fingerprint is `8f99b251923f6086767dcc6f61fed257a1f815b479e1970a5ffc546b9ae69564`.
A matching version label alone is insufficient. The older Mac 5841 magnetic
projection failure remains a regression control, not an allowed executor.

Local acceptance evidence is in the sibling audit directory
`symmetry-integration-pilot/`. It includes compiled relocatable jobs, native
GPXs/audits, the durable ledgers, browser screenshots and `browser-validation.json`.
The merged NOVA/refinement regression suite passed 1,324 tests with two skips;
the subsequent streaming-view change passed its focused UI checks. A browser
workflow captured the actual source, discovered 70 OPDs, constructed five first
tier children, reviewed and ran all 22 signed starts, inspected results and
downloaded byte-identical GPX/receipt artifacts. Its best synthetic-control fit
had Rwp 9.433110% and recovered normalized mode magnitudes 0.100188 and 0.299610 Å
against injected values 0.1 and 0.3 Å. The simultaneous sign reversal is a powder
domain ambiguity. This is a synthetic control, not an independent experiment.

Eight final native controls passed their predeclared dispositions, including
positive magnetic fits, forbidden sites, a grey group, an alternate-setting
refusal, a centered parent with k=(1,0,0), and structural recovery. The GPX
audits check what actually varied and recompute the final calculated profile;
an executable script or decreasing cached Rwp alone is not success.

The same eight controls were prepared as a sealed queue and continued through
the browser after a fresh application process started with no in-memory draft.
All eight completed with exactly eight submission intents. Regression checks
cover late receipt collection and repeated restart after receipt tampering;
the original collected receipt remains the integrity anchor. A subsequent
campaign/interface/refinement integration gate passed 135 tests for these
lifecycle changes. The final result-control UI check passed 22 tests.

The complete current-kernel replay sealed all 732 declared starts across
Cr2WO6 (369), Ba/Co (162) and Pr/Sr manganite (201), without technical failures
or hidden branches. These starts are three experiments, not 732 experimental
validation cases. The later lifecycle-only package revision leaves the native
scientific kernels unchanged and is covered by the repeated eight controls.

Independent experimental acceptance remains incomplete: four experimental
series have been exercised, not twelve. Cr2WO6 and Ba/Co fixed-nuclear screens
remain scientifically inadequate; the reference-compatible Cr2WO6 joint-polish
development control converged but did not meet the reference moment check.
Pr/Sr manganite has adequate fixed-nuclear alternatives, without a unique
structure assignment. The full current-code portfolios are retained separately
from earlier replays. No held-out score or general automatic structure-solution
claim is made. These unresolved cases, the continuation/matrix work and the
Galaxy destination pilot must be completed before release acceptance.

The packaged native mode-matching helper and tables retain their GSAS-II source
license and provenance. This product includes software produced by UChicago
Argonne, LLC under Contract No. DE-AC02-06CH11357 with the Department of Energy.

## Next implementation plan and acceptance gates

Plan recorded 2026-10-28. The user has authorized browser visual inspection.
The current local-file URL restriction is a tool limitation, not missing user
permission. Visual acceptance will target the actual integrated local NOVA web
application over its supported HTTP interface. Serving the blocked standalone
file through a different route is not part of this plan.

### 1. Record the integration baseline and freeze the first supported scope

Inspect the current application/deployment revisions and the existing
Refinement source/worker contract before merging this isolated work. Preserve
the tested runtime identity and all original RADAR-PD source GPX revisions.

The first supported scope is single-k nuclear displacement-mode exploration
and collinear commensurate magnetic exploration with explicit parent, k basis,
magnetic sites, form factors and amount/domain assumptions. Secondary modes,
OPD tiers and seed sets must be declared. Unsupported tensor transformations,
coupled/multi-k or noncollinear branches remain visible as unsupported/deferred.
Do not silently substitute a simpler physical problem.

Gate: a capability matrix says what can run, what needs a user choice, and what
cannot yet run. Nuclear/magnetic amount linkage, normalization, frame and gauge
are part of the protocol rather than incidental script defaults.

### 2. Package the standalone discovery, compiler and native worker

Replace development-time reference-template extraction and Mac-specific paths
with RADAR-PD-owned, versioned adapters and worker templates. Cache ISODISTORT
and Bilbao discoveries with exact settings, candidate identities, mode tables,
transformations, origins and hashes. Package the required runtime helpers and
license notices. Keep credentials external to repositories and images.

Define the job-local input/output schema, source-revision checks, preflight,
parameter stages, finite signed starts, bounds, budget and fixed recovery rules.
Execute in the pinned Linux image without a reference-project runtime or LLM
calls. Audit final native parameter states and normalized absolute modes/moments.

Gate: the known controls replay after moving the job directory, with network
disabled, unchanged source GPXs, complete trial coverage and matching audited
outputs. A changed setting/runtime/input stops with an explicit receipt.

### 3. Build an independent experimental acceptance corpus

Target 12 independent experiment series, split into eight development cases
and four held-out cases before adjusting acceptance rules. Record independent
series separately from scans, candidates and repeated seeds. Include actual
experimental TOF data relevant to POWGEN/NOMAD, alongside public known-solution
controls. Identify each expected outcome from a documented reference solution;
proposal intent alone is insufficient ground truth. Report unavailable or
unresolved cases explicitly instead of filling the experimental count with
synthetic data.

Cover metric/peak-splitting changes, finite-k superlattice distortion, k=0
magnetism on existing reflections, nonzero-k magnetic supercells, several
magnetic sites/partial ordering, alternate settings/origins, and weak/null
signals. Include nuisance controls and challenging out-of-scope models so the
executor's refusal/unresolved behavior is tested as well as successful fits.

Gate: freeze per-case expected outcomes and tolerances before holdout scoring.
The reference-compatible model must survive discovery and remain competitive;
refined values must meet declared physical/reference checks. Equivalent models,
failed lower-residual starts and incomplete coverage must remain visible.
No universal Rwp cutoff or guaranteed unique space-group assignment is assumed.

### 4. Implement and exercise the campaign dispatcher

Introduce durable prepare/submit/poll/collect/audit states and stable dispatch
keys. Bundle a candidate's finite starts for one source scan/bank into a Galaxy
job, using independent native processes. Start with representative scans, then
continue an explicitly reviewed small model portfolio over selected existing
and future compatible runs.

Gate: local end-to-end replay proves restart, duplicate-arrival prevention,
lost-acknowledgment recovery, cancellation, bounded retries and source-revision
handling. One failed branch must not block unrelated work or disappear from
coverage. Completed-prefix behavior must work before the full series exists.

### 5. Integrate the portfolio view and inspect it as a user

Place Model exploration inside Refinement. Connect live job receipts and native
artifacts to grouped models, residual-versus-complexity overview, expandable
ambiguities and diagnostic reasons. Add the model-by-condition matrix and
normalized mode/moment trends on eligible measured axes. Inspected curves use
observed/calculated/difference traces and phase Bragg ticks, with linked zoom,
axis switching and up to two pinned alternatives.

Gate: visually inspect the actual local application in the browser at desktop
and narrow widths, using real completed and partial campaigns. Test 50+ models,
empty/failed/missing results, slow jobs, new arrivals, group selection, pins,
zoom/reset, downloads and navigation back to the original scan. Measure actual
interaction latency; syntax checks and pure grouping timing are not UI tests.

### 6. Measure an NDIP-test pilot and issue the readiness report

Verify the configured Galaxy destination, quotas and available slots/memory.
Measure one, two and four simultaneous small bundles within those limits,
including queue/startup time, memory and unrelated-user impact. Exercise the
same recovery and incoming-scan cases in NDIP-test before expanding enrollment.

Gate: report automatic, user-choice, unsupported and unresolved outcomes for
each independent experimental series, parameter-effect audits, holdout results,
resource measurements, visual findings and remaining failures. Local science
acceptance, local application acceptance and NDIP acceptance are separate
milestones. Passing this pilot supports the declared first-release scope;
additional physics requires additional acceptance cases.

## Primary references

- [ISODISTORT help](https://iso.byu.edu/isodistorthelp.php)
- [ISODISTORT examples](https://iso.byu.edu/isodistortexample.php)
- [Bilbao k-SUBGROUPSMAG help](https://cryst.ehu.eus/cryst/help/k_Subgroupsmag1_help.html)
- [GSAS-II magnetic-structure workflow paper](https://journals.iucr.org/b/issues/2024/05/00/cam5001/cam5001.pdf)
- [Galaxy job resources and concurrency](https://docs.galaxyproject.org/en/latest/admin/jobs.html)
- [Galaxy collection mapping semantics](https://docs.galaxyproject.org/en/release_25.0/dev/collection_semantics.html)
