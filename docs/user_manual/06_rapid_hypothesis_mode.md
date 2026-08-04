# Rapid Hypothesis Mode

Rapid Hypothesis Mode is designed for quick scientific triage. It asks: which small set of phases best explains the observed pattern or residual signal before spending a long time on full refinement?

## Rapid Stage Overview

| Stage | User-Friendly Name | Purpose |
| --- | --- | --- |
| Input prep | Signal preparation | Convert files, remove obvious background, build the working signal. |
| Coarse search | Broad hypothesis search | Search many candidate combinations using tolerant 64-bin profiles. |
| Lattice nudge | Lattice-aware peak alignment | Nudge unique phase cells with RADAR-style symmetry-respecting rules. |
| Pattern scoring | Higher-resolution pattern scoring | Rebuild nudged profiles at higher resolution and rank hypotheses. |
| Final refinement ranking | Focused refinement ranking | Run targeted GSAS-II checks for top hypotheses. |
| Solution inspector | Interactive solution review | Swap representative variants, remove phases, rerun targeted checks. |

## Why 64-Bin Then Higher Resolution

The broad 64-bin stage is intentionally tolerant. It can survive modest lattice mismatch and helps find phase families. The later higher-resolution stage uses nudged candidates to test whether the proposed phases explain sharper peak-position information.

## Representative Families

Many database entries are near-duplicates: same element set, same or comparable space group, very similar pattern, and close stoichiometry. The rapid pipeline groups these into phase families and prefers a cleaner representative formula when appropriate.

Example: `Cu31S16` and `Cu2S` can belong to the same near-stoichiometric family if the pattern, space group, and element ratios are sufficiently close. The preferred representative may be `Cu2S`, while `Cu31S16` remains selectable in the solution inspector.

## Solution Inspector

The solution inspector should allow you to:

- inspect the selected hypothesis,
- choose alternate members from a grouped family,
- remove a phase from a hypothesis,
- rerun a targeted refinement,
- compare the new result with the original ranking.

Targeted refinements should use the best available nudged or optimized parameters from previous rapid stages, not blindly restart from a stale database cell when a better staged estimate exists.

## Rapid Result Interpretation

The rapid results table should avoid internal names such as `sse64` or `gain64`. User-facing labels should describe what the number means:

- broad match,
- pattern match,
- unexplained signal,
- refinement quality,
- final rank,
- relative contribution.

## Good Use Cases

- Fast impurity triage.
- Comparing phase-combination hypotheses.
- Checking whether a supplied main phase is enough.
- Generating a shortlist before a longer Full RADAR-PD run.

## Common Mistakes

- Treating the first broad-search rank as the final answer.
- Ignoring whether the top peaks of a proposed phase are actually supported by observed intensity.
- Letting a phase with many overlapping main peaks steal main phase intensity.
- Forgetting to inspect alternate representatives inside the same phase family.

