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

| Check | Result | Limit |
| --- | --- | --- |
| Live ISODISTORT Method 2, public SrTiO3 parent, k=(1/2,1/2,1/2) | Five irreps, 22 OPDs cataloged; five representative children constructed | Seventeen other OPDs deferred; no exhaustive search |
| Native import into the pinned RADAR-PD Linux GSAS-II 5843 image | All five children imported with verified native displacement-mode labels, variables and normalization | Import test, not fitting or model-selection validation |
| Name-only CIF adapter negative controls | Incompatible operators and a verified origin-equivalent but differently situated model both rejected | No general origin/basis transformation of mode tables implemented |
| Bilbao authenticated API, public Pnma #62, k=(0,0,0) | 51 magnetic subgroup records parsed, retaining basis and origin | Group discovery only; no moments, magnetic GPX, or refinement tested |

The focused host checks passed 17 tests with one native-runtime skip; the
separate Docker audit exercised all five imports and both setting controls.
The imported mode-coordinate matrices were verified to be invertible. The
R3+ child illustrates another branching issue: it contains both a primary R3+
displacement mode and a secondary R1+ displacement mode. A child-CIF choice
does not by itself specify which modes should be freed at each stage.

The first unadapted native import failed for the full cubic H-M symbol. The
new `scripts/gsas_isodistort_import.py` changes only the H-M scalar, after
complete operator-set equality in the existing setting. It reparses and checks
that every non-name CIF value is unchanged and records hashes and the imported
mode inventory. It refuses an origin/basis change, magnetic modes, occupancy
modes and rotational modes. Strain metadata is retained but not compiled.
Original service CIFs are never overwritten. No runtime Studio import is added.

Local evidence is under `audit/2026-10-28/symmetry-exploration/` in the surrounding
development workspace. In particular: `live-isodistort/manifest.json`,
`native-import-v2/report.json`, and `bilbao-k0-parsed.json`. The live service
checks used public crystallographic fixtures, not private experiment structures.
No assumption about NDIP pod network access follows from these laptop checks.

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

Acceptance progresses from the completed import feasibility check to:

- Real native mode refinements with actual varying-parameter and GPX checks;
  parent equivalence, both signs, mode normalization and seed coverage audited.
- Blind structural benchmarks: Gamma-only distortion, single-k tilt, coupled
  irreps, near-parent null data, impurity/coexistence, wrong k, other settings,
  metric-only splitting and weak signals. Lock thresholds before holdout data.
- Magnetic benchmarks: k=0 and nonzero commensurate order, multiple magnetic
  sites, AFM/ferrimagnetic/canted alternatives, forbidden sites, and a nuclear
  control with no magnetic signal. Verify links and refined moment components.
- Corrupt catalog, interrupted external service, missing mode, source changes,
  budget exhaustion, no-network execution and incremental-scan replay.

No general structural/magnetic model-selection reliability claim is justified
by the five imports or 51 subgroup records above.

## Primary references

- [ISODISTORT help](https://iso.byu.edu/isodistorthelp.php)
- [ISODISTORT examples](https://iso.byu.edu/isodistortexample.php)
- [Bilbao k-SUBGROUPSMAG help](https://cryst.ehu.eus/cryst/help/k_Subgroupsmag1_help.html)
- [GSAS-II magnetic-structure workflow paper](https://journals.iucr.org/b/issues/2024/05/00/cam5001/cam5001.pdf)
