# Prepared structural search (pilot)

The pilot adds a nonlinear structural backend to **Refinement → Model
exploration → Structural displacement modes**. It uses GSAS-II for diffraction
and instrumental profiles, and shares the existing campaign ledger, Galaxy
admission policy, process cancellation, independent audits and result viewer.
It does not replace RADAR phase discovery, custom nuclear recipes, magnetic
Hessian/grouped optimization or live magnetic posterior analysis.

## User workflow

1. Select the target RADAR or recipe GPX and inspect its parent.
2. Choose **Prepared nonlinear search · all modes · pilot**.
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

The native reference solver remains the default. It supports explicit selected
modes and signed starts; the pilot instead retains every independent physical
mode in each included child. Only demonstrated uniform-origin gauges are
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
  native fallback instead of relaxing the reconstruction tolerance.
- An independent saved-GPX calculation must agree within 1e-8 relative error
  and 0.01 supplied standard deviations. Export verifies active mode variables,
  origin holds, cell freedom and fixed instrumental values, with no stale
  covariance. Ordinary GUI continuation does not enforce the contact screen or
  the original search box; this limitation is recorded in its contract.
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
The pilot must not become the default based only on successful process exits,
excellent Rwp on a single case, or the research branch's local latency figures.
Qualification needs independent model-recovery controls, full exported-GPX
round trips and measured complete workflow latency in the actual deployment.

## Integration validation

The local integration was checked against the current NOVA, campaign, recipe
and CIF contract suites (5,189 passing tests; 22 environment-dependent skips).
Separate offline Linux controls use the deployed scientific image and the
built wheel. They exercise a separate parent reference, a strong excluded
peak, a retained secondary phase, a TOF child with 27 physical displacement
coordinates plus six cell coordinates, and two sequential reduced updates.
The latter verifies prepared-start reuse, restored-session duplicates and
stale-revision rejection. Each successful export is checked by an independent
GSAS-II calculation; these synthetic controls establish contract behavior,
not universal experimental model recovery or deployment latency.
The shared native-profile changes also pass 21 real GSAS-II magnetic solver
controls. A deliberately perturbed two-mode structural export was continued
with native least squares, recovering its original amplitudes with a finite
seven-variable covariance matrix while preserving the source checkpoint.
