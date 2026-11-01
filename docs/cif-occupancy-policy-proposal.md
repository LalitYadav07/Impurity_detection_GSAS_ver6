# Explicit CIF occupancy-rounding policy

This implementation is an isolated, unreleased change. It requires matching
NOVA, Library Builder wrapper/backend, and Analyze wrapper/backend. It does not
modify `gpx_recipe` or its native protocol. Galaxy changes are review templates
and patches in `galaxy/`, not a deployment.

## User decision

The default is strict: no site total greater than one. A build may explicitly
allow a maximum excess of **0.001**, with a review note. Occupancies are preserved;
there is no normalization or error rescaling. Ordinary valid sources remain strict
members. Only sources with an actual excess receive a policy record.

Selecting an archive does not activate its `reviewed: true` declarations. NOVA
labels these as **unverified archive declarations**; the actual affected-site
count remains unknown until native validation. The user permits the bounded
policy once for a configuration and its series. Permission binds both the
archive SHA256 and source-policy hashes. Replacing the library invalidates it.
A source-selection key prevents delayed UI reads approving a replacement.

That UI key is not serialized as portable permission. Native consent compares
only acknowledgment, archive hash, policy-set hash and incomplete-library
decision. Re-reviewing identical content in a new History can reuse the portable
configuration; changed hashes require a derived configuration. Explicitly
choosing the built-in library clears inapplicable permission in a derived copy,
without modifying the saved original.

The configuration carries `library_review` with `acknowledged: true`,
`archive_sha256`, `policy_set_sha256`, and optional `allow_incomplete`. Direct
Galaxy use can provide that same explicit configuration field or the dedicated
activation JSON input. An archive's declarations are never permission.
NOVA checks registered input capabilities before sending a non-strict build or
analysis; old strict workflows remain available on older wrappers.

The separate partial-library feature handles incomplete-build selection. This
archive inspector exposes the same manifest to its classifier. Exact
producer-declared duplicates differ from all other failures; unrecognized
failures require explicit incomplete-library review. This does not certify
that a historical archive contains every originally intended source.

Historical augmented manifests retain their declared base/added counts without
mislabeling the added IDs as a verified total inventory. Archive hashes are
streamed; cached review documents are copied before delivery. The direct Galaxy
source-ZIP path rejects colliding CIF basenames before extraction rather than
silently omitting a file when flattening folders. Review tables are paginated.

## Native checks and provenance

`scripts/cif_occupancy.py` checks scalar tokens before parser sanitation,
including negative/nonfinite/unknown values and numerical underflow. It expands
complete symmetry support, reconstructs original occupancy components, and
checks exact preservation. It retains the parser's existing 1e-4 fractional-site
equivalence tolerance so rounded special positions do not create duplicate
near-coincident atoms. It does not loosen the occupancy limit. Zero components
stay zero. Named D/T isotopes and declared oxidation tables are preserved;
unsupported atom notation fails explicitly.

Each policy records source and canonical-newline hashes, atom/non-cell-model
signatures, affected sites/components/totals, source composition, builder runtime,
and a policy hash. Analyze recomputes that evidence from the exact source before
invoking a scientific pipeline. Fabricated counts or missing affected-site
evidence cannot pass native validation merely by having consistent hashes.

A process-local ledger permits only approved sources inside that analysis's
workspace. Database reads, full/rapid renders, lattice nudging, and native import
use the shared loader. Structure caches bind source, sidecar, and activation
content; original-JSON caches observe file revisions. Reviewed sources cannot
fall back to regenerated CIFs when originals are absent. Lattice derivatives
preserve atom data and non-cell fields, with explicit neutron element-label
neutralization. Changed symmetry cannot inherit a policy. GSAS import checks
absolute atom counts and per-site values as well as relative composition.

Result provenance carries the verified policy, runtime, and warning that a
rounded starting model does not certify specimen composition. Pattern
observation context retains this provenance for later analysis.

## Validation and limits

Repository tests use synthetic sources only. Experimental CIFs and the original
input end-to-end benchmark remain outside the repository in audit. Tests cover
strict/opt-in behavior, symmetry, rounded special positions, zero/tiny/malformed
values, stale hashes, forged evidence, cache revocation, native import,
configuration round trips, stale UI reads, capability gates, and wrapper quoting.
The frozen Linux runtime is checked without dependency upgrades as well as the
local development runtime.

Preserving isotope labels does not qualify isotope diffraction fingerprints;
that separate pre-existing calculator issue needs its own validation. A
converged fit or accepted policy is never a calibration or composition certificate.

An unchanged rounded mixed site can still fail the native recipe's independent
capacity check. An explicitly proposed replacement model, reviewed through the
existing phase-correction workflow, is a separate supported option. This is a
scientific model change, not something automatically generated or authorized by
the library occupancy allowance.
