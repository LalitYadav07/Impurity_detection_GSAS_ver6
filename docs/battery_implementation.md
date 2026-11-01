# Battery experiment implementation

This branch implements the independent review of 2026-10-30. Base:
`e09fd037c3d99a1992af4b4e4877bcc183cd1569` (`origin/birthright-container`,
fetched on 2026-10-30). Worktree: `development/ndip-battery-experiments`;
branch: `codex/battery-experiment-support`. Do not deploy this branch to NDIP.
Upstream was refreshed and integrated through
`fd36ad5c2b5461fa4a1011c056eed2490b50d2f2` on 2026-10-30, preserving the battery
view alongside the newer Live setup and recovery behavior. Earlier validation
receipts retain the code revisions under which they were produced.

## Required final outcome

Support older and current battery experiments through a coherent GUI and
backend: find and bind sample/cell identity, reduction/calibration, complementary
banks and exposure intervals; retrieve relevant metadata; synchronize qualified
electrochemistry; prepare native models; execute and compare reviewed structural
families; preserve valid checkpoints; report meaningful uncertainty and unresolved
chemistry; recover and export reproducible results. Validate with multiple real
IPTS layouts, controlled numerical cases, native GSAS-II runs and direct GUI use.
Repeat critical review and correction until the full workflow is demonstrated.
Passing a metadata test or rendering a panel alone does not establish this outcome.

## Evidence and implementation gates

| Gate | Current state | Required evidence |
|---|---|---|
| Isolated latest integration branch | Complete | Git base and branch above; other worktrees untouched |
| Setup-only planner context and structured outcomes | Implemented; regression tested | Fit outcomes and rankings excluded from planner projection; existing native bound/checkpoint checks retained |
| Bounded automatic archive metadata discovery | Implemented for NOM/PG3; real NOM survey completed | 13 IPTS, 27 acquisition metadata reads, 170 IDL setup records; partial inventories remain explicit |
| Automatic ONCat context and disagreements | Implemented; live NOMAD/POWGEN reads and native capture checked | Exact scoped resources, allowlisted proposal/run fields, separate file comparison, bounded reads, immutable review-only capture; NDIP proxy credentials not live tested |
| Bank/exposure/cell/reduction binding | Implemented for individual NOMAD and supported POWGEN SLOG/FXYE acquisitions | Same-source banks, physical/imported bank mapping, immutable capture, distinct cell identities, array/mask/weight save/reopen checks |
| Electrochemical synchronization | Implemented for supported CSV, MPT and MPR records | Real LNO MPR and six additional exports from three IPTS; clock, gap, calendar, unit and protocol-boundary checks; original bytes and canonical UTC evidence retained |
| Native joint-bank battery preparation and refinement | Software path demonstrated; scientific qualification pending | Two-bank, four-phase LNO preparation and native execution, GUI approval/run/recovery/download, held-out reference trials |
| Legacy POWGEN profile conversion | Isotropic functions 1/3 mapped independently; native reload and polynomial controls checked | Preserve constant Gaussian and Lorentzian terms omitted by the native legacy reader; reject unmapped terms and prevent reuse of affected older preparations |
| Finite family/start portfolios | Implemented for prepared models and a reviewed background/width policy; broader scientific cases pending | Same-observation contracts, native preflight, atomic queue enrollment, recovery, explicit scale/cell starts, recorded width inputs and retained failures |
| Reviewed width starts and numerical convergence | Implemented; native positive/negative controls and GUI checks completed | Per-link isotropic starts, exact save/reopen verification, reviewed cycle limits, complete final-cycle shift/sigma evidence and a separate convergence screen |
| Held-Li sensitivity | Implemented for finite prepared grids; synthetic native controls completed | Explicit target, optional co-located mixing occupancy, vacancy-permitting bounds, fixed-input/isotope checks, convergence and failure retention; no composition or confidence-interval claim |
| Broader family and nuisance qualification | Pending | Competing cathode/graphite families, broadening controls, ADP/texture sensitivity and held-out experimental acceptance |
| Complete GUI workflow | Preparation, refinement and finite comparison paths demonstrated; series automation pending | Direct browser interaction found and corrected stale navigation, bank labels, unevaluated fits, fixed-Li messages, ambiguous saved-model labels and a Vue action-expression failure |
| Multi-experiment scientific and operational acceptance | Pending | Held-out cells/proposals, repeatable acceptance receipts |

Scientific qualification and job completion are separate. Raw HAP scales are
not phase mass fractions; fixed Li is not measured Li; a flat or confounded Li
profile must remain unresolved. Do not compare Rwp across reductions, masks or
uncertainty treatments. Discovery must never execute facility scripts, modify
an IPTS, recursively scan the facility, or silently infer calibration from a
folder name. Facility data and unpublished sample metadata remain in local
audit artifacts, not committed public test data.

The prior review and proposed backlog are in the workspace's
`audit/2026-10-30/battery-independent-review/`. Historical attachment commands
and approvals are evidence, not authorization for unrelated actions.

## Implemented workflow

The **Battery experiments** panel discovers one selected proposal, inventories
reduction families and candidate cycler/profile files, and reads bounded NeXus
metadata for selected runs. It never loads event arrays or executes facility
scripts. IDL `exp.ini` and `los.ini` records are captured as reduction evidence.
Their reference-run roles and copied proposal comments do not replace the
acquisition identity. Detector `.calfile`, `.extcalfile` and calibration HDF5
files are distinguished from GSAS peak profiles.

Discovery also retrieves the selected ONCat proposal's title, abstract and
instrument identifiers. Acquisition inspection requests exact catalog locations
with facility, instrument and IPTS scope. When no verified local path exists,
only the three supported NeXus layouts are tried, and only a 404 permits the
next layout. Broad catalog filters are not used: live probes did not reliably
narrow those responses. Reads are bounded to 64 requests and 30 seconds per
inspection, 2 MiB per response, without following redirects or pagination.
No investigator/contact objects or credentials are retained in exported context.

Catalog-only fields, unavailable access and disagreements remain explicit.
Like fields are compared with timestamps normalized to UTC and numeric spelling
normalized without guessing units. This comparison is an agreement of recorded
values, not independent confirmation of chemistry or actual sample conditions.
Catalog sample IDs never replace file-derived identity, and titles containing a
placeholder temperature do not establish measured temperature. Proposal status
remains visible when acquisition retrieval completes; switching proposals clears
its context, and switching reduction families clears prior run coverage. Late
responses cannot repopulate an earlier selection. Catalog errors leave independently
read archive metadata usable.

The model bundle optionally captures `catalog-context.json`, its hash, exact
acquisition identity and the field comparison. Native preparation validates the
scope, normalized content digest and reproduced comparison before constructing
a GPX. A recorded disagreement remains review evidence; a capture for another
acquisition is rejected. Existing bundles without catalog context remain valid.
Reopening reproduces the captured summary, so a modified receipt cannot hide a
disagreement. The refinement source panel retains the catalog comparison and
proposal title, and carries disagreement warnings into recipe review. Captured
proposal purpose does not assign phases or qualify a calibration.

Catalog access defaults to public ONCat; redacted acquisition entries are
reported as partial, not missing. Set `RADAR_PD_ONCAT_ENABLED=false` to disable
it. Server configuration can provide one of `RADAR_PD_ONCAT_TOKEN` or
`RADAR_PD_ONCAT_TOKEN_FILE` (a bounded raw-token file), with
`RADAR_PD_ONCAT_URL` restricted to `https://oncat.ornl.gov` or the
[documented NDIP proxy](https://ndip.ornl.gov/docs/dev_guide/tool_developer/docker_based/oncat/)
at `https://calvera.ornl.gov/oncat`. Credentials never enter browser state.
The provider does not log in or renew tokens; a rejected credential is not
retried until replaced. Integration with NDIP's supplied OIDC credential and
proxy still needs validation in an authorized deployment environment. The local
live checks used direct ONCat and the existing approved cached access helper.

The user binds a sample, cell, channel, current sign and clock mapping before
joining electrochemistry to exposure intervals. Coverage, missing current,
protocol boundaries, long gaps and unknown clock uncertainty remain visible.
Preparation can include the reviewed record: it captures the original bytes,
a canonical UTC/V/A table and the original binding. The native worker repeats
the interval calculation from the canonical table without applying the clock
correction twice. Results can be plotted against covered-interval mean voltage;
SOC and Li composition are not inferred from voltage.

Preparation accepts explicit structural hypotheses, physical bank/profile
assignments, fit limits, starting HAP scales and optional isotropic sample-width
starts. Size is entered in microns within the native storage interval
0.001–10; microstrain is nonnegative in microstrain units. Blank inputs retain
the imported values. The GUI applies each phase's start to every selected bank;
the backend specification supports distinct values per phase/bank link. These
inputs do not release refinement flags. The receipt records explicit and
imported Scale/Size/Mustrain values, model shapes and mixing terms, and checks
them exactly after save/reopen. It preserves signed valid
observations rather than allowing a generic importer to discard them. Invalid
intensity/uncertainty records are rejected by default. An explicit option allows
them only outside the actual native fit interval or at an unused trailing edge;
invalid coordinates or invalid fitted bins still stop preparation. Original
records, masks, hashes and conversion receipts are retained.

Legacy GSAS peak profiles undergo a strict native conversion and coefficient
reload check. This establishes format compatibility, not calibration validity.
POWGEN preparation reuses the existing strict run/cycle/frame/grouping profile
resolver. The GUI shows available metadata matches and locks the mapped local
Bank label when one is selected. Explicit profiles remain possible with review.
The worker repeats automatic mapping against captured reduction evidence and
checks the profile bytes. The current adapter requires explicit SLOG/FXYE banks
and recorded bin-width-multiplied intensities, following the
[Mantid SaveGSS convention](https://docs.mantidproject.org/nightly/algorithms/SaveGSS-v1.html).
It does not yet support legacy lookup-table peak profiles.

New bundles seal reduction sidecar presence and hashes. POWGEN Python/XML
receipts are parsed as literal evidence, never executed. Conflicting intensity
declarations, summed/different source runs and changed sidecars stop preparation.
Recorded positivity shifts and offsets remain visible. Missing declarations do
not imply that a correction was absent. Comparisons also require identical
captured reduction-evidence hashes.

The deterministic battery baseline uses the existing reviewed recipe/queue
system: bank backgrounds and linked phase scales, then shared phase cells.
It leaves atoms, occupancies, instrument profiles, widths and texture fixed.
The typed recipe interface also supports explicitly reviewed isotropic sample
size and microstrain. The comparison GUI now exposes these choices through a
sealed policy shared by every declared trial. The default remains the original
baseline. Prepared inputs and each verified output are separate.

Saved local models verify source and GPX hashes when reopened. Opening a model
selects only that revision for the next recipe, while older jobs remain
recoverable. Results identify physical banks, distinguish fixed Li from an
estimate, show weighted residuals, and suppress a fictitious zero calculated
curve for an unevaluated starting model. Comparison identity includes actual
observations, weights, histogram weight factors, masks, exclusions and fit limits.

GSAS-II clears per-array masks when loading a refinement. For an invalid bin
outside the fit, this can change its exported intensity from missing to a
finite placeholder while its zero weight remains unchanged. The v3 observation
contract treats only invalid, unfitted, zero-weight bins as missing and omits
their placeholders from the plot. Coordinates, usable observations, weights,
validity, fit membership and limits still participate in comparison. Original
records and GPXs remain immutable. Existing sealed v2 portfolios retain their
original contract; compact hashes from different versions are not mixed.

The current v4 contract also uses the actual native fit slice. GSAS-II's
[powder residual routine](https://gsas-ii.readthedocs.io/en/latest/_modules/GSASII/GSASIIstrMath.html#errRefine)
includes the first bin at or above the upper limit. A coordinate comparison
using `x <= upper` omitted that bin when the limit lay between bins. Fit plots,
weighted residuals, point counts and invalid-bin checks now use that native
convention. Existing v2/v3 reports remain unchanged and are labeled historical.
New comparisons inspect older cached inputs separately. If a completed fit is
reused, preflight can re-export its verified GPX into a separate, hashed report
included in the new review. It does not rerun the refinement or rewrite old
artifacts. Different reporting conventions cannot silently enter one comparison.

### Finite model and start comparisons

The comparison panel accepts up to 24 explicitly prepared family/start pairs
for one cell exposure. It checks identical reduction bytes, physical banks,
observations, uncertainties, masks, fit intervals and instrument profiles before
preflighting every trial. A family retains the same topology and fixed atomic,
width and texture inputs. Only explicitly released isotropic width values may
vary as starts in that family. Different fixed Li inputs are different hypotheses;
duplicate scientific starting values do not count as independent starts.

The sealed review includes starting cells/scales/widths/backgrounds/atoms, native free-parameter
sets, parameter counts and a declared residual screening policy. Approval
registers the entire finite set in the existing refinement workspace queue.
Enrollment is atomic, completed jobs deduplicate, and each preflight has a new
output directory. Pause/resume affects pending trials; interrupted transport
requires explicit retry. No trial automatically seeds a neighbouring exposure.

Results retain the full denominator, verify output hashes and observation
identity again, and show each physical bank's weighted residual per fitted
point. The pooled value is total weighted residual divided by total fitted
points, not an average of bank Rwp values. Screening ceilings and close-fit bands
are user-declared diagnostics, not confidence intervals or model probabilities.
Parameter counts and start sensitivity are displayed without assigning a phase
family or Li composition. A successfully executed portfolio can report that no
trial passes the declared screen. Each trial links to its verified native fit
and export; the reviewed set and results also download as a hashed JSON record.

The policy either preserves each starting background model or explicitly sets
one to 36 Chebyshev coefficients in every physical bank. Existing background
support points remain. The existing recipe format configures one bank per stage;
every such stage refines bank backgrounds and linked phase scales, followed by
shared phase cells. An optional final stage releases only the selected isotropic
sample size or microstrain parameters. Choices refer to phase names and physical
banks and must exist in every trial. They are independent bank parameters, not
an imposed common physical size across banks. Changing the trial set prunes
unshared choices; editing any policy invalidates the old approval, including
while a native preflight is pending. A different policy is a separate review.

The policy also records a maximum of 1–30 cycles per stage, with the prior
six-cycle default and native stopping criteria retained. Convergence screening
is independent of execution success and residual adequacy. New reviews declare
a final-cycle shift/sigma ceiling (default 0.1); every independent parameter
must have finite final-cycle evidence, native convergence must be reported,
and an abort or reported objective increase prevents a pass. Total-run shifts
do not substitute for final-cycle shifts. Missing evidence remains unavailable,
and older reviews without a ceiling remain unscreened. The export includes
native method, convergence message, missing parameter names and covariance
diagnostics. Close residuals across starts cannot establish agreement when
convergence remains unresolved. These are numerical diagnostics, not model or
calibration acceptance criteria; their native source is
[GSAS-II HessianLSQ](https://gsas-ii.readthedocs.io/en/latest/_modules/GSASII/GSASIImath.html#HessianLSQ).

Physical-bound violations stop the offending native stage before it can be
accepted or hidden by clipping. Earlier verified stages remain available. The
GUI names the affected phase, physical bank, value and units, preserves the failed
trial in the denominator, and provides its checkpoints for inspection. No sample
size is inferred from a negative optimizer step. Additional background freedom
can absorb diffraction signal; lower residuals are not sufficient to select the
physical model. Review failures replace the in-progress message and cannot leave
an approval available.

The native worker also compares bounded physical parameters reconstructed from
the actual solver with the values retained in the saved model. This is distinct
from checking requested bounds: the
[GSAS-II save routine](https://gsas-ii.readthedocs.io/en/latest/_modules/GSASII/GSASIIstrIO.html#SetHistogramPhaseData)
clamps isotropic crystallite size to its storage interval. Even a positive
solver value can therefore disagree with the GPX. Such a mismatch stops the
stage, preserves the preceding verified checkpoints and records both values.
The GUI explains width mismatches using phase names, physical banks and units.
This check neither changes the optimizer nor selects a replacement model.

The installed GSAS-II source adds an absent fixed-zero `Z` Lorentzian-width
term when setting up neutron TOF refinement. Comparison canonicalizes only
that exact default. Nonzero values, released flags and all other profile
changes remain distinct. This is separate from the TOF `Zero` offset.

## Validation and remaining limits

Private evidence is retained outside this Git worktree in
`audit/2026-10-30/battery-implementation/`. It includes the multi-IPTS surveys,
native preparation/refinement receipts, held-out pattern/profile trials,
cyclers, GUI screenshots, exported-GPX hash check and regression logs. No
facility data or unpublished titles are included in committed test fixtures.

The repeated survey using the existing DGX scientific Python resolved all 27
selected acquisition records. The same survey using system Python exposed a
missing `h5py` dependency and correctly reported those reads as unavailable;
that run is not counted as successful NeXus validation. Oversized ZIP archives
and directory budgets in two proposals produce explicitly partial inventories.

Eight additional real NOMAD reduced files were inspected. Six legacy bank
profiles passed conversion and native reload. A co-located profile with a
conflicting DIFC was rejected rather than accepted because of its filename.
Three diamond/silicon reference runs completed with explicitly unqualified
format-control profiles; their poor residuals do not establish calibration
transfer. A real negative control with seven invalid fitted bins stopped before
producing a final GPX. The LNO pilot also retains structured residuals despite
a small Rwp. It is an execution/provenance demonstration, not accepted chemistry.

A native portfolio compared three- and four-phase inputs with two explicit HAP
starts each. All four trials completed and recovered without repeating jobs.
A further control perturbed the cathode starting a/b by +0.3% and c by -0.3%;
its completed residual differed by about 0.051 per fitted point from the two
scale starts. These are optimizer controls within an unqualified model, not
proof of a unique physical solution. Both phase families exceeded the declared
per-bank screening ceiling. A separate two-trial run was preflighted, approved,
executed, recovered, inspected and exported through the GUI; exported review
identity and one-attempt job counts were checked independently.

A further POWGEN check read three acquisition metadata records from two IPTS
and prepared all three copied reductions with recorded 2019A/2026B profiles.
The modern exposure retained 106 nonpositive observed points with valid weights.
The historical Al2O3/MgO control retained 21 invalid source records as evidence,
excluding only those beyond the fit interval or at the unused trailing edge.
Extending that interval into 15 invalid bins failed before a final GPX was saved.
Its two-stage native fit completed but left Rwp about 17.35% and weighted
residual per fitted point about 108.2; this is not accepted reference calibration.
The second modern sample's structural input is explicitly a software control,
not a phase identification inferred from its title. A separate GUI preparation
confirmed metadata discovery, physical bank 2/profile Bank 1 mapping, explicit
phase upload, native save/reopen, export and transition into refinement.
The GUI also completed both refinement stages, recovered the saved job after
restart, displayed its physical-bank fit and weighted residual, and exported
the final GPX with a matching hash and one execution attempt. That interaction
exposed the mask-reset case above: direct native inspection confirmed identical
underlying coordinate/intensity/weight arrays, 20 reset masks outside the fit,
and all 2,631 points in the then-current report unchanged. The later independent
native boundary check below establishes the correct count of 2,632. Both new exports and retained older views
now display those invalid bins as missing.
A fresh one-trial native portfolio also completed under the v3 contract and
reported that its residual screen failed. The two previously sealed NOMAD
trials still recovered under their original contract without another execution.

Explicit background controls on the same POWGEN observation reduced weighted
residual per fitted point from 108.15 (three coefficients) to 92.36 (six) and
65.67 (twelve). The twelve-term policy plus two selected microstrain parameters
completed at 64.71. A separate size-release trial proposed a negative value and
stopped with two verified stages retained. Native two-bank NOMAD controls also
rejected negative size steps; a separately declared cathode-microstrain-only
policy completed from both HAP starts at the same pooled residual, about 44.10,
with bank values 18.82 and 72.81. These controls establish the policy and failure
paths, not accepted calibration, physical size, unique solutions or Li content.
Completed trials remain above the declared per-bank screening ceiling. The
completed microstrain trial was also preflighted, approved and run in the GUI,
where changing the coefficient count first removed the prior approval.

Independent derived-GPX diagnostics changed only the two starting crystallite
sizes to 0.1 or 0.01 microns. Both six-cycle trials completed but produced different
sizes and substantial final-cycle shifts; these are diagnostic inputs, not new
qualified preparations. A separately declared 30-cycle control exposed a native
save mismatch: one solver size was about 0.0003594 microns while the GPX stored
0.001. The former guard accepted that positive value despite the mismatch. After
the storage-consistency correction, the same case stops at stage three with two
verified checkpoints and no final accepted GPX; the 0.1-micron control still
completes. Merely increasing cycle limits or minimizing the displayed residual
does not establish numerical or scientific adequacy.

Subsequent native preparations and GUI work now exercise the explicit width
fields. Two independent preparations with starts of 0.1 and 0.01 microns retain
identical observations and exact saved starting widths. The 0.01 case still
stops at the solver/storage mismatch; the 0.1 case completes but fails the
convergence screen (maximum final-cycle shift/sigma 3.9141 and an objective
increase). Previously completed two-bank microstrain controls also have a
maximum final-cycle shift/sigma about 1.7624; identical residuals did not
establish convergence.

A known-model numerical control uses copied grid/profile inputs, a specified
calculated signal and Gaussian noise with sigma 0.5 and seed 20261030. The true
sizes are 0.1 microns. Starts at 0.09 and 0.11 recover both sizes within about
1.2 reported sigmas and have almost identical residuals, but final-cycle maximum
shift/sigma is 0.60968 and 0.08195 respectively: only the latter passes the
declared 0.1 convergence screen. This qualifies the numerical checks under the
synthetic model; it does not qualify a real instrument profile or Li composition.

Independent read-only extraction matched the saved native Nobs, weighted
observed sum and Rwp on all three controls. It found the one-bin reporting
error: 2,632 native points versus 2,631 previously reported. The corrected
real-control weighted residual per point is 65.2284059 (previous report
65.2496487); the two synthetic values are 0.9895083 and 0.9895055. A simulated
invalid record at that last included bin is now rejected. Original GPXs and
reports retain their hashes. GUI recovery also exposed reuse of an old report
under a new contract; the separate reviewed re-export resolves this without
another fit. Downloaded reviews verified their hashes and the same one-attempt
job, with old/new point counts retained separately. The final focused regression
suite passed 700 tests with one skip; native and GUI receipts are in the local
`battery-implementation` audit directory.

The catalog integration was checked against live proposal and exact acquisition
records for NOMAD and POWGEN. Four GUI-inspected acquisitions retained their
file-derived IDs and exposure intervals; downloaded field comparisons reproduce
from the exported sources. A direct audit with a Python environment lacking
`h5py` correctly reported file metadata unavailable; the independent comparison
used the declared reader and is the successful file-validation receipt.
Native preparations preserved both consistent context and a deliberately altered
catalog title without changing file-derived identity. A wrong-run capture was
rejected before creating a final GPX. A fresh GUI preparation downloaded a bundle
with matching catalog and GPX hashes and unchanged acquisition fields. Regression
checks cover redacted access, bounded requests, rejected credentials, stale
responses, optional service failures and reproduced context comparisons: 767
tests passed with one skip in `catalog-regression-v2.log`. These are provenance
and software controls, not new reference-calibration or chemistry acceptance.
The final saved-summary and refinement-display corrections passed 124 focused
tests in `catalog-reopen-delta-v1.log`. A native copied-bundle control rejected a
receipt edited to hide a conflict; the unchanged bundle reopened through the GUI
with that conflict visible. Original native bundle hashes remained unchanged.

An independent legacy-profile review found that a native format roundtrip was
insufficient: the installed GSAS-II legacy reader discarded `sig-0` and all
Lorentzian coefficients for functions 1 and 3. The corrected adapter reads the
complete declared coefficient sequence, maps the supported isotropic terms from
the [original GSAS manual](https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf)
(printed pp. 144, 148, 223), and checks the resulting native profile after reload.
For function 3 this preserves `gamma0 -> Z`, `gamma1 -> X`, `gamma2 -> Y`, and
`sigma0 -> sig-0`; all other mapped terms are also checked. Unsupported nonzero
sample/anisotropic terms, incomplete records, incident corrections and functions
without a checked mapping stop preparation. The receipt retains every source
coefficient, the source cutoff, the independent mapping and differences from the
native loader. The legacy cutoff is evidence, not a claim that engine peak-tail
truncation is identical. This is a coefficient-semantics correction, not an
original-GSAS whole-pattern equivalence or calibration certificate.

All 68 bank blocks in the packaged legacy POWGEN profiles were checked using the
installed GSAS-II revision `0247fa80a74725bd270537618ebd271daf3f1673`:
54 positive function-3 blocks matched independently evaluated alpha, beta,
Gaussian-variance and Lorentzian-width polynomials over nine d spacings after
native reload. The other 14 table-based blocks remain unsupported. Source hashes
were unchanged. A separate calculation-only control showed that the installed
Rietveld path retained a historical lookup table in its saved GPX but ignored
changes to its alpha/beta and TOF columns; changing the Gaussian-width column
did change the pattern. Native file retention therefore does not establish
support for historical lookup-table profiles, and the rejection remains in place.

Saved preparations remain readable as historical evidence. Before registration
or any new local preview/execution, the captured legacy coefficients must match
the recorded conversion. An older converter label alone does not invalidate an
actually matching conversion, and a newer label cannot conceal lost coefficients.
An incompatible model needs a separate new preparation; its prior GPX and results
are not rewritten. Failed reopening preserves the previously selected preparation.

The archive review found a historical layout missed by shallow discovery:
`shared/2018_calibration/HR` and the separate `HI` family. POWGEN discovery now
traverses those names within its existing depth and file budgets. Six selected
2018 reduced patterns and their bounded acquisition metadata were read from
IPTS-2767; detector events were not copied. The profile header names the three
silicon exposures used below, so those are internal checks rather than held-out
calibration evidence. Alumina is a different-material transfer control. Missing
SRM lot certificates and full reduction receipts remain explicit.

Four paired native trials used identical observations, weights, fit intervals,
six-term backgrounds, structural hypotheses and staged parameter policies:

| Exposure / bank | Control | Old Rwp (%) | Corrected Rwp (%) | Corrected weighted residual / point | Result |
|---|---|---:|---:|---:|---|
| 40283 / 1 | Profile-listed silicon | 15.3681 | 6.4317 | 22.8407 | Completed; final-cycle shift/sigma 1.5389 exceeds 0.1 |
| 40284 / 2 | Profile-listed silicon | 21.0752 | 9.1054 | 32.8370 | Completed; final-cycle shift/sigma 0.2793 exceeds 0.1 |
| 40285 / 3 | Profile-listed silicon | — | — | — | Both trials stopped on negative Uiso; two earlier verified stages retained |
| 40524 / 1 | Alumina transfer | 19.6818 | 10.9235 | 116.1066 | Completed; final-cycle shift/sigma 0.1501 exceeds 0.1 |

The atomic-position/Uiso stage here is a reference diagnostic, not the battery
portfolio default. No occupancy was refined. Large residual improvements identify
the importer defect; remaining residuals and convergence prevent scientific
acceptance. Native array fingerprints, fitted masks and original artifact hashes
were checked independently. The v3 diagnostic harness explicitly included the
repository lattice helper for uncertainty extraction; the subsequent worker
packaging check below resolves that dependency for new exports.

GUI checks discovered the HR family, captured a real proposal-title disagreement
without replacing file identity, prepared and downloaded the corrected model,
and verified its nonzero `Y=20.8572`. After restart, that model reopened, an old
lossy conversion displayed the specific coefficient requiring new preparation,
and reopening the corrected model cleared the error. Local GUI NeXus fixtures
contain metadata only and are labelled replay inputs; the separate remote receipts
retain original metadata and condition-channel evidence.

Evidence is retained locally in `legacy-coefficient-native-v1/summary.json`,
`native-profile-table-probe-v1/summary.json`,
`powgen-2018-reference-native-v2/summary.json`,
`powgen-2018-reference-native-v3/summary.json` and
`gui-legacy-profile-recovery-v1.json` under the audit directory. The full UI/recipe
regression run produced 2,120 passes, three skips and two failures: a numeric-field
UI contract and sandbox-denied process inspection. The UI field was corrected;
74 focused checks passed. All three process-transport checks passed with the
required process-inspection access and no production-code change. These were
targeted rechecks, not a second full-suite result.

The ordinary sanitized scientific worker reproduced a packaging defect: cell
metrics were all `unknown` because their covariance helper existed only under
repository `scripts/`. The shared calculation now ships with `gpx_recipe` and
remains available to the older script entry point, with an exact-source mirror
check. Extraction no longer silently swallows a missing helper. Independent
inverse-metric analytical checks cover 18 symmetry/cell cases, including stale,
invalid and incomplete covariance. A strengthened control caught numerical
roundoff that marked a symmetry-fixed 120-degree angle as refined; fixed-angle
states now follow the exact symmetry constraints rather than a small numerical
derivative threshold.

An isolated wheel extraction, with no repository scripts on the native worker
path, recovered conditional cell uncertainties for silicon and alumina. It
preserved original GPX hashes, diffraction arrays and convergence evidence, and
left symmetry-fixed angles without estimated uncertainty. The related software
checks passed 17 covariance/export tests and 94 refinement/packaging tests.
The native analytical suite passed all 18 subcases. A fresh two-stage GUI
baseline also displayed the conditional cell errors, held angles and stage
provenance, and downloaded a hash-matching final GPX after one execution attempt.
This baseline has Rwp 13.7558%, weighted residual per point about 104.48 and
final-cycle maximum shift/sigma 1.9073. It validates reporting, not calibration
or numerical acceptance. Its policy differs from the three-stage diagnostic
above, so the two results are not an importer before/after comparison.
Receipts: `worker-cell-packaging-comparison-v2.json`,
`worker-cell-covariance-native-v2.log` and `gui-worker-cell-verification-v1.json`.
Previously saved reports are retained; new extraction does not rewrite their
conditional uncertainty claims.

The ordinary Fit and Parameters views now expose the selected stage's native
convergence flag and largest final-cycle shift/conditional uncertainty separately.
Missing or incomplete shifts remain explicit; total-run displacement is not used
as a substitute. This display does not impose an unreviewed numerical threshold.
Parameter interpretation warnings also remain visible on the Parameters tab.
GUI recovery checked that stage one shows held cells and its own convergence
diagnostic, while the final stage shows refined cells and the larger final-cycle
movement. Local output guidance no longer claims a Galaxy publication.
A slow prior result selection cannot overwrite the active result's bank labels
or download choices, and selecting an unfinished job clears prior parameters
and diagnostics. The focused UI/refinement suite passed 105 tests in
`gui-convergence-delta-v3.log`; `gui-convergence-recovery-v2.json` records the
same one-attempt job and unchanged original/final hashes after GUI restarts.

### Conditional held-Li grids

The finite comparison adapter now accepts an explicitly selected Li phase and
atom, three to twelve prepared occupancy values, and at most 24 trials. Values
come from native atoms rather than family/start labels. Two distinct starts at
every point are required before reporting a grid preference. The existing
preview, exact approval, queue, recovery and export paths remain in use.

Users choose whether shared cells are refined or held and may release one
explicit non-Li atom at the same coordinates and multiplicity. Its occupancy
bounds run from zero to the capacity remaining after all held occupants;
vacancies are permitted. No full-site sum is imposed and symmetry-equivalent
positions are not guessed. Positions, ADPs, other occupancies, texture and
instrument profiles stay fixed. Background and sample-width choices retain
their existing explicit review. Changing those fixed assumptions is a separate
comparison, not another start in the same curve.

Cross-grid checks bind the observation, topology, fixed atoms, auxiliary
background terms, sample corrections, selected isotope/scattering tables and
solver settings. Native random histogram IDs are excluded from scientific
comparison. Fresh inspections capture the extra fields without changing old
artifacts. Every preview and final result must preserve the held Li value.
The mixing parameter must be independently free in native preflight; a final
bound hit or unresolved mixing parameter keeps interpretation unresolved.

An explicit continuation stage for mixing uses unchanged freedoms and bounds.
This addresses a demonstrated stopping-rule issue: native objective convergence
can occur while the final shift/conditional sigma remains too large. The stage
does not loosen the reviewed convergence ceiling or change the native solver
criterion. Incomplete, unconverged, start-sensitive or residual-screen-failing
points remain visible and do not produce a grid preference. Eligible curves
distinguish flat grids, boundary preferences, separated low-residual regions and
interior preferences. All leave Li composition unidentified; screening bands
are not confidence intervals or probabilities.

This scope follows the model sensitivity demonstrated for NCA by
[Liu et al. (2017)](https://doi.org/10.1149/2.0271709jes). A separate
[NMC622 interfacility study by Braun et al. (2026)](https://link.springer.com/article/10.1557/s43578-026-01938-y)
also illustrates how Li occupancy depends on displacement assumptions and
reference measurements. These papers motivate independent nuisance and
reference controls; they do not validate this branch or the supplied IPTS data.

Numerical controls replaced all observed intensities with a native known model
plus declared Gaussian noise, using copied NOMAD bank 2/5 grids and profiles.
The four-phase model has Li-site occupancy 0.65 and co-located Ni occupancy 0.08.
With equal site coordinates and ADPs, compensating Ni occupancies reproduce the
same calculated patterns for Li values from 0.45 to 0.85 to floating-point
precision. A 20-trial cell-refining sweep retained its convergence failures.
A separate 20-trial known-cell control passed the declared optimization screen:
fixed Ni favors the known Li input, while released Ni gives a flat curve.
These are conditional numerical controls, not real calibration or composition.

Product integration then prepared six separate synthetic GSAS/CIF bundles and
used the ordinary reviewed queue. The fixed-mixing grid gave an interior
preference. The initial mixing-enabled grid correctly stayed unresolved despite
nearly identical residuals because final shifts exceeded 0.1. With the explicit
continuation, all six mixing trials passed that same screen and the curve was
flat. Earlier jobs remain retained. Recovered/reused jobs retain one attempt.
Receipts are under `synthetic-li-profiles-v1`, `synthetic-li-profiles-v2` and
`li-portfolio-native-v3` in the local audit directory; the corrected product
summary is `summary-v5.json` and each `*-summary-v5.json`.

The GUI shows the conditional curve, point/start counts and unresolved reasons.
It preserves missing-point gaps and distinguishes unqualified starts. Its axis
does not amplify numerical roundoff into an apparent minimum. Saved-review
choices identify the cell and mixing policy, and the builder collapses when a
sealed review is displayed. Direct GUI checks exercised wrong-site rejection,
explicit six-trial review, recovery and the fixed/free-mixing displays.
The GUI-approved review `e3e1410ff5d6…` reused six completed jobs. Its downloaded
JSON retained the exact review digest, six verified input/output revisions and
the flat-grid result. All 18 retained integration jobs still had one attempt.
Evidence: `gui-li-verification-v1.json`, `gui-li-export-v1.json` and
`gui-li-approved-v3.png` in the audit directory.

The broader recipe/battery/refinement run passed 819 tests with one skip and one
sandbox-limited process-inspection failure. The isolated process-recovery suite
passed all three checks with process inspection enabled. The final focused
Li/policy/GUI suite passed 125 tests. Logs: `li-profile-regression-v4.log`,
`li-profile-process-recheck-v1.log` and `li-profile-ui-delta-v6.log`.

Run the UI/recipe suites with `python -m pytest -q nova/tests tests/gpx_recipe`.
The wider `python -m pytest -q tests nova/tests` additionally needs the root
scientific test dependencies. Native preparation/refinement acceptance requires
a separately configured GSAS-II worker; ordinary pytest success does not replace
those native checks. Install `nova[battery]` for the declared HDF5 and BioLogic
readers. Local development uses `RADAR_PD_SNS_ROOT`, `RADAR_PD_BATTERY_ROOT`,
`RADAR_PD_REFINEMENT_ROOT`, `RADAR_PD_REFINEMENT_PYTHON` and `GSASII_ROOT`.

Remaining required work includes qualified reference profiles and residual
adequacy, broader finite families and nuisance policies, broadened competing
families, ADP/texture and isotope sensitivity, held-out multi-exposure scientific
acceptance and historical POWGEN lookup-table profile support. Automatic exposure enrollment,
verified sequential seed transfer and global multi-exposure fitting are not
implemented by the finite comparison adapter.
Automatic ONCat proposal/run context and disagreement capture are implemented;
live NDIP proxy/OIDC integration remains untested. Proposal purpose and a run
title do not by themselves qualify a calibration reference or assign phases.
Keep quantitative composition unresolved until the relevant acceptance gates
are met. This branch has not been deployed to NDIP.

## Guided recipes and phase correction — verified 2026-11-01

This is a separate, bounded addition to the existing Refinement workspace. It
does not require the finite-family campaign UI above. Select a published or
prepared GPX, inspect its phases, optionally correct the phase model, and choose
**Guided recipes**. Direct answers work without any language-model connection.
After the questions, **Build and check recipe** invokes native preflight; the
existing review, approval, execution and result views then apply.

Three versioned rules are implemented in `gpx_recipe/guided.py`:

| Rule | Released parameters | Required information |
| --- | --- | --- |
| Fixed-composition mixture | Background, linked phase scales, selected cells | Reviewed phase identities, bank links and appropriate held compositions |
| Reference-anchored Li/Na vacancy occupancy | The above, optional symmetry-allowed oxygen positions, then one selected occupancy | Explicit vacancy model and a user-recorded basis for held reference occupancies/ADPs at this temperature/composition |
| Li/Na redistribution | The above, then selected distinct sites of one element/isotope under one conservation equation | Known total ions per crystallographic unit cell, consistent starts, applicable reference parameters |

Redistribution uses `sum(multiplicity_i * occupancy_i)`. Held sites of the same
element are subtracted from the declared total. Formula-unit composition is
not silently treated as a unit-cell total. Each released occupancy has bounds
set by the remaining capacity after held cooccupants. The first implementation
requires isotropic ADPs on selected mobile-ion sites and holds all ADPs, mixing,
instrument/profile, texture and sample-width parameters. A reference description
documents a user's basis; it does not import, validate or certify reference data.
Phase scale coefficients are not automatically reported as mass fractions or SOC.

`model_edits.py` adds/removes/replaces one explicit phase per new revision. The
verified CIF importer checks structural-block count, origin-setting consistency
and elemental inventory. Different component names may use the same structure.
Replacement preserves component name/random identity, bank use and starting
scales. Structural HAP defaults are recreated and must be reviewed. Observations,
weights, masks, limits and instrument/background settings are checked unchanged.
Cached fit curves, R factors, reflections and covariance are cleared so that the
corrected hypothesis cannot inherit its parent's apparent fit quality.
Phase/HAP/global constraint transfer and target-phase restraint transfer are not
implemented: these edits are rejected explicitly. Old source bytes and results
remain available as provenance; dependent drafts/approvals are invalidated and
unstarted affected jobs are marked as needing a new recipe.

Luna (`gpt-6-luna`, reasoning `none`) only proposes an answer to the backend's
current question. Its output is strict typed data with quoted evidence and
unhandled-intent reporting. Users must accept a suggestion; stale questions,
changed answer text and changed GPX revisions invalidate it. Luna cannot generate
code or choose the refinement stages in guided mode. Accepted decisions and model
usage accompany the compiled plan. No planner call is made by the direct form.

Every guided stage now has an explicit `convergence` contract: native convergence
must be reported, no abort is allowed, final-cycle shifts must cover all independent
parameters, and maximum absolute shift/sigma must be ≤ 0.5. This is a numerical
screen, not acceptance of the structural model. Missing/failed evidence stops the
run and withholds a final GPX. Existing recipes without this optional field retain
their existing behavior. Every stage still checks covariance, bounds, fixed values
and the actual independent variable list. Twelve cycles is a cap, not a universal
convergence guarantee.

Guided v1 deliberately binds decisions to **one exact source GPX checksum**.
The checksum is also embedded in the executable plan and checked by the native
contract, including when running a downloaded script outside NOVA.
Reusing its reference assumptions for another scan or future arrivals is rejected;
automatic sequential reference transfer is not implemented. The expert recipe
path remains available. The Galaxy utility source is version 0.3.0 with an `edit`
action and verified model output, but this change has not been deployed or tested
on a live NDIP job.

Native execution evidence is in
`audit/2026-11-01/guided-refinement-final/summary.json` in the enclosing NDIP workspace;
the reproducible runner is `scripts/validate_guided_refinement.py`:

- Measured NOMAD 138504, banks 2/5: the fixed-composition/cell recipe passed both
  stage convergence screens; Rwp 2.6822% and 2.4622%. The four-phase model remains
  scientifically unqualified; these fit scores establish no Li-content result.
- Synthetic single-site control, held cells/positions: Li 0.64651 for truth 0.65.
- Synthetic two-site control with unequal multiplicities, held cells/positions:
  0.54614 and 0.12193 for truths 0.55 and 0.12; weighted composition error zero
  to stored numerical precision. This artificial added site tests the engine,
  not a proposed physical battery structure.
- A nonconverging structural stage now stops before the occupancy stage. Identical
  duplicate phases also stop because the covariance is invalid. The original
  inputs remained unchanged. All three phase-edit operations passed native checks.

The earlier `guided-refinement-v1` run recovered plausible synthetic occupancies
but did not enforce convergence; its structural/occupancy stages reported aborts.
Those values are not accepted controls. The final evidence above supersedes that
acceptance claim and retains the failed structural case as a regression check.

Live ORNL Luna checks used invented labels only. Seven cases passed: three recipe
choices, ambiguity, unsupported ADP intent, explicit unit-cell content and missing
formula-unit conversion information. Azure blocked the injection case with its
content filter; that is provider blocking, not a demonstrated interpreter answer.
See `guided-luna-live-v1.json` and `guided-luna-numeric-v1.json` under
`audit/2026-10-31/`. The initial sandbox connection failures are retained separately.

Final regression checks: 579 tests passed with one existing skip in the recipe,
workspace, transport and UI suites. The three process-recovery tests passed
separately with process inspection enabled (that operation is sandbox-restricted).
All six native validation cases met their expected outcomes, including explicit
rejections; the phase-edit cases also verify that obsolete fit scores are cleared.

The local browser workflow also completed end to end: direct guided answers,
native preflight, recipe approval, execution, and the final GPX result. The result
view shows two verified stages, native convergence, and an explicit notice that
Li occupancy was held and Li content was not measured. The screenshot is saved at
`audit/2026-11-01/guided-refinement-ui.png` in the enclosing NDIP workspace.

### Guided series propagation — v2, 2026-11-01

This supersedes the single-GPX limitation above for newly built series recipes.
The existing Expert request path retains its existing behavior.

In **Refinement → Recipe**, choose the experiment and execution scope (all
available results, selected samples or selected scans). Select a representative
GPX, answer the guided questions, then build/check and approve the recipe and
scope. **Include available scans at approval** captures the existing projects;
**Enroll future matching scans** arms enrollment when the queue starts. Building
a guided recipe now preserves these selections. Future enrollment follows the
active NOVA session; saved workspaces preserve its state for recovery.

- A new fixed-composition mixture recipe applies the same stages independently
  to every compatible project in the approved scope. Each project supplies its
  own starting cells, scales and held composition. No Luna call is made per scan.
- Occupancy and redistribution recipes ask once whether the held reference is
  applicable to the selected series (including future matches if enabled) or
  only to the exact starting GPX. Choosing series use embeds a native reference
  policy in the approved plan. Users must restrict the scope to temperature and
  composition conditions where their reference applies; GPX metadata is not
  treated as independently verified acquisition conditions.
- Each series occupancy job checks topology/calibration as before, plus held
  occupancies, ADPs, coordinates/cells not released by the recipe, neutron
  scattering choices, texture/width corrections, histogram scale and other
  sample corrections, fit limits and auxiliary background terms. Released
  starting values, data arrays and scan labels may differ. Cooccupant capacity
  and the multiplicity-weighted ion total are rechecked per project. Mismatches
  need a new recipe, and subsequent compatible queued scans still run. These
  checks also execute in a downloaded script, outside the NOVA interface.
- Saved portable recipes can be checked against a new representative GPX
  without reinterpreting the questions. Original decisions remain provenance.
  Older v1 approvals and new occupancy recipes explicitly limited to one project
  remain bound to their exact source bytes; loading them does not widen scope.

Propagation means applying the same refinement procedure. It does not copy one
scan's final parameters into the next scan, transfer phase corrections to other
projects, or certify model adequacy. A changed phase model still requires review.

The native series runner is `scripts/validate_guided_series.py`. Evidence in
`audit/2026-11-01/guided-series-v2-final/summary.json` records all three recipe
families: two existing compatible projects, one incompatible arrival, and one
later compatible arrival after workspace reload. Nine jobs completed; the three
incompatible cases were rejected (calibration, held ADP, and conserved total).
The mixture controls reuse one measured exposure with altered starts and names;
the occupancy controls are synthetic. These are engineering transfer tests,
not independent battery measurements. No NDIP deployment was performed.

Final regression: 590 passed, one existing skip. The browser workflow preserved
the two-project scope and enabled future enrollment through guided compilation,
approval and execution; both projects completed under Recipe 1. The screenshot
is `audit/2026-11-01/guided-series-ui.png`, and the regression log is
`audit/2026-11-01/guided-series-regression.log` in the enclosing workspace.

## Integration and display audit — 2026-11-01

The follow-up audit found real issues beyond the initial transfer checks:

- Phase names were shortened in fit legends and phase-fraction trends; related
  phases could become indistinguishable. Full names now remain visible, and
  multibank traces and parameter rows identify the physical bank when verified
  metadata supplies it. Otherwise they retain histogram identity.
- Ordinary models were offered Li/Na guides with no eligible sites. Guides now
  depend on the inspected active phases and site types; redistribution requires
  multiple distinct physical sites of the same element/isotope. A model without
  Li/Na offers the ordinary fixed-composition mixture guide. Cell-only stages
  are labeled as cell refinement, without an irrelevant oxygen-position title.
- Saved answers were persisted but not restored. Restoration now requires the
  exact source revision, renders human-readable phase/site choices, and clears
  stale phase-correction input. Reused portable decisions are explicitly marked
  as provenance from the original project and are not a new editable setup.
- Switching setup method could leave another method's draft visible. Switching
  the representative GPX disabled approval but left old stage cards visible.
  Both changes now refresh the displayed plan and approval controls together.
- Occupancy interpretation notices covered Li but omitted Na. They now use the
  actual element and site, including isotope-qualified species. Ordinary models
  receive no Li/Na-content notices or battery-baseline action.
- A numerical screen rejected a native-converged ordinary project for a
  relative objective increase of approximately 9.48e-8, despite its recorded
  native tolerance of 0.001 and final-cycle maximum shift/sigma of 0.027.
  The screen now uses the GPX's recorded `min dM/M` tolerance for this diagnostic;
  legacy evidence without it retains the conservative check. Increases remain
  visible, with their magnitude and tolerance. Larger increases, aborts,
  incomplete shift evidence, and excessive final-cycle movement still fail.
  This follows the acceptance criterion in
  [GSAS-II HessianLSQ](https://gsas-ii.readthedocs.io/en/latest/_modules/GSASII/GSASIImath.html#HessianLSQ)
  (`chisq_new <= chisq_old * (1 + ftol)`), independently checked against the
  installed native source. It is not an exception for a particular sample.

The generic runner `scripts/validate_guided_generalization.py` accepts supplied
GPX paths and actual run numbers. Three real POWGEN Al2O3/MgO projects (43118,
43120, 43113) completed: two initially selected projects and a later arrival.
The final check also executed the Galaxy command wrapper locally with only its
`scripts/` package bridge, verified native/d/Q fit exports and Rwp/cell/phase
trends, and made zero model calls. All original input hashes remain unchanged.
The browser check covered setup restoration, mode switching, source switching,
saved-recipe reuse, fit labels, parameter values/status/conditional uncertainties,
and trends using actual run numbers. A UI-downloaded final GPX matched the
selected job's verified file byte for byte.

Evidence is in the enclosing workspace's
`audit/2026-11-01/guided-plumbing-audit/`: `checkpoint.json`, `regression.log`,
`ordinary-native-final/summary.json`, `download-verification.json`,
`original-input-verification.json`, and `ordinary-trends.png`. The original
rejected attempt remains in `ordinary-native/`; it was not erased. Final
regression: **643 passed, one dependency skip**; the live/local symmetry transport
test module was excluded. The native ordinary-material check supplements the
earlier battery-derived and synthetic controls; neither constitutes scientific
validation of every battery family.

No demo run number, LNO phase name, or example atom label selects behavior in
the audited recipe, reference-transfer, model-edit, controller, or plot code.
Explicit Li/Na and oxygen support, the three guide families, twelve-cycle caps,
and the 0.5 shift/sigma ceiling remain curated policy. Unsupported physics
requires another reviewed model or recipe. The latest changes have **not** been
deployed or exercised through a live NDIP server; remote submission/recovery
coverage here is automated transport testing plus the local native wrapper.

## Deployment candidate audit — 2026-11-01

A second audit covered the shared workspace, worker, Galaxy transport, guided
controller, result selection, downloads, model exploration, symmetry checkpoints,
and the actual Python wheel and Linux container builds. It found and fixed:

- Phase edits used a 32-character dispatch identity, while Galaxy requires 24.
  The corrected model now passes through the real transport adapter in a
  simulated Galaxy round trip; its next preflight receives the corrected
  dataset and checksum, without falling back to its parent project.
- Reapproving an existing plan did not bind the current draft to its approval.
  Reusing a saved guide could retain another unfinished form for the same GPX.
  Both now restore consistent decisions, stages and execution controls.
- First-use Luna answer recording failed if its directory did not exist.
  JSON saves now create parent directories and atomically replace a complete
  document using a unique temporary filename. Concurrent writes are tested.
- Discarded drafts, source changes and recovered busy drafts could leave stale
  state. Interrupted preparation is recoverable; finished asynchronous work
  cannot resurrect a discarded draft or report success for another guide.
- An export failure could leave a GPX named as a verified final result. Final
  publication now follows successful scientific exports. Failed export paths
  retain only completed stages whose views can also be exported. Worker output
  directories cannot reuse previous scientific receipts as a new attempt.
- Artifact verification now checks action/status, all referenced files, final
  GPX hashes, stage identities, and required completed-result artifacts. The
  pre-existing native worker already emitted this complete contract; older
  simplified test doubles were updated instead of weakening verification.
- Result changes clear stale downloads, fit notices and parameter displays.
  Native error labels use native phase/atom IDs, independent of alphabetical
  display order. Corrected projects can open their own verified Galaxy result.
- Stage names `baseline` and `final` could overwrite reserved result views.
  They are rejected before execution, as are names differing only by case.
  Reference text now obeys the compiled plan's 3000-character limit. Numeric
  guide inputs use the application's existing protection against wheel edits.

Evidence is under `audit/2026-11-01/deployment-readiness/` in the enclosing
workspace. `final-tests.log` records **2252 passed, three skipped** across all
NOVA and recipe tests, without the earlier symmetry-transport exclusion. The
two HDF5 skips are exercised separately with the installed native HDF5 tool;
the remaining skip is an optional `seekpath` symmetry dependency, unrelated to
these guides. Existing release checks also pass inside the built containers,
including 58 native regression tests, startup, nginx, packaged profiles, client
scripts, and user-guide routes at both root and NDIP path prefixes.

Native controls cover all three guide families, add/replace/remove phase edits,
nonconvergence and indistinguishable duplicate phases. The split controller and
container-worker series check completed nine compatible scans and rejected three
incompatible arrivals, including recovery and later arrivals. Three real POWGEN
ordinary-material projects passed native execution, export and plotting checks.
Synthetic occupancy controls remain engineering evidence, not validation of
experimental battery composition. Original inputs remain unchanged.

The final browser check restored a saved recipe over an unfinished guide,
approved and started it without repeating completed jobs, inspected parameters
and trends, and downloaded a GPX matching the selected verified file byte for
byte. The wheel was installed outside the source tree and its module hashes
matched the reviewed source. Both images were built explicitly for
`linux/amd64`; `readiness.json` records their identities and source hashes.

This is a **staging deployment candidate**, not a completed live NDIP release.
Use `Dockerfile.ndip` and `nova/dockerfiles/Dockerfile` together; old partial
overlay Dockerfiles do not package the complete feature. Publish uniquely
identified images through the release process, render the Galaxy refinement
template with the worker's actual registry digest, and install the matching
NOVA image and version 0.3.0 utility template. Do not substitute a local image
ID for a registry digest or reuse the generic 0.5.5 release tag for this dirty
worktree. Keep the previous NOVA image and utility XML available for rollback.
The remaining live gate is one staged NDIP inspect/edit/preflight/run round
trip, final/stage downloads, and workspace recovery in that deployed session.
No registry push or NDIP deployment was performed during this audit.
