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
- Starting-curve construction reuses the exact fixed-cell native profile for
  displacement-only proposals. Cell changes still use complete native
  calculations. All starting points, contact checks and secant-step choices
  remain unchanged. A paired Linux/amd64 TOF control measured 52.45 → 33.38 s
  preparation (36% less), with all 69 curves agreeing to 1.8e-15 relative error.
  This is a preparation timing, not complete experimental-workflow latency.
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

`ReducedObservation` accepts unknown exposure. Its `mask=True` means usable;
the snapshot adapter explicitly accepts `invalid_mask=True`. Reduced intensity
and supplied variance are never converted into invented counts. Verified raw
counts retain a separate count/exposure contract.

The library is ready for an adapter under the existing WebMON acquisition owner.
This change does **not** expose structural analysis in WebMON's live UI or launch
a second polling service. Cross-job prepared-context transport and actual live
queue latency need their own acceptance tests before enabling that route.
Likewise, Method-3 catalog enumeration, calibrated structural posteriors,
counting advice and named-mode custom recipes remain separate work. Current
custom recipes accept nuclear projects; magnetic posterior checkpoints are
offered as verified GPX downloads for GSAS-II rather than sent to an incompatible
recipe inspector.

## Packaging and qualification

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
deployment. The consolidated local UI has not yet passed the experimental
Ba4SbRu3O12 recovery test and has not been deployed to NDIP-test.
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
The final five-case structural acceptance gate runs without skips, including
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
