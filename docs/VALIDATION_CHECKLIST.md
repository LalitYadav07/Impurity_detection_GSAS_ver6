# RADAR-PD Validation Checklist

Use this checklist whenever changing pipeline logic, GSAS-II refinement behavior, candidate filtering, lattice nudging, UI run controls, or deployment settings.

## Record The Run Context

- Branch, commit SHA, and whether the code is local, VM UX, production VM, or Hugging Face.
- Source root, active Python environment, and service port.
- Pipeline config path and any overrides.
- Candidate library: neutron or X-ray, built-in, augmented, or custom-only.
- Example name, input diffraction file, instrument parameter file, main CIF, allowed elements, and sample environment elements.
- Whether the run is a quick validation, one-pass validation, or full benchmark.

## Required Automated Checks

- Run the focused regression tests for touched areas.
- Run `py_compile` or equivalent import checks for changed pipeline modules.
- Confirm the Streamlit app starts without a top-level exception.
- Confirm no accidental changes are made to production/Hugging Face unless that is the explicit target.

## Required End-To-End Examples

- TBSSL: fast neutron CW sanity check with known main phase and impurity search.
- LK99: TOF example with more difficult CIF parsing behavior.
- Mystery Peaks: longer TOF stress case that exposes timing, refinement, and candidate-ranking issues.

For development changes, one discovery pass is usually enough. Before release or deployment, run the intended full configuration.

## Pipeline Pass Criteria

- Process exits with code 0.
- `pipeline_summary.json` exists and reports success.
- `Results/Summary_Fractions.csv` exists and has finite, plausible phase fractions.
- Final Rwp is finite and not an obvious collapse value.
- Weight fractions are not negative, not above 100 percent for a single phase, and total close to 100 percent unless there is a documented reason.
- Accepted phases and rejected phases are traceable in the logs and artifacts.
- No unexpected `Traceback`, `ERROR`, corrupted GPX, missing `.lst`, or stale refinement output.

## Refinement Safety Checks

- Failed trial refinements restore the last safe GPX state.
- A candidate is not accepted when Rwp explodes, phase fraction collapses, or GSAS-II output is stale.
- Commit/polish accepts only a scientifically safer state than the previous accepted model.
- If polishing fails, the result remains at the last accepted pre-polish state.
- The main phase fit image, accepted-model image, and final artifacts correspond to the final GPX.

## Lattice Nudger Checks

- Nudged CIFs obey their crystal-system constraints:
  - cubic: `a=b=c`, `alpha=beta=gamma=90`
  - tetragonal: `a=b`, `alpha=beta=gamma=90`
  - orthorhombic: `alpha=beta=gamma=90`
  - hexagonal/trigonal hex setting: `a=b`, `alpha=beta=90`, `gamma=120`
  - trigonal rhombohedral setting: `a=b=c`, `alpha=beta=gamma`
  - monoclinic: exactly the observed unique-axis angle is free
- Nudged CIFs are loadable by pymatgen/GSAS-II.
- `score_q_max`, sampled representative count, timing fields, and scoring mode are recorded.
- Near-tie behavior prefers the lattice closest to the starting CIF when Pearson scores are effectively equal.

## Timing Checks

- Capture stage timings from the pipeline summary.
- Compare changed stages against the previous baseline on the same example and config.
- For Mystery Peaks, always inspect separate timings for screening/nudging, compare/joint refinement, and commit/polish.
- A speedup is only accepted if scientific outputs remain plausible and safety checks still pass.

## UI Checks During A Run

- Starting a run does not freeze the whole page for long periods.
- Live logs update without forcing expensive full-page work on every refresh.
- Results/artifact tabs remain usable while the run is active.
- Failed input files, CIF parse failures, and GSAS-II refinement failures are visible to the user.
- Example mode does not expose private or previous user-uploaded files.
- Run names advance automatically and do not overwrite a previous run unless explicitly requested.

## Warnings Policy

- Known warnings can be accepted only when they are understood and do not affect the run result.
- Unexpected warnings around CIF parsing, phase attachment, stale `.lst`, failed GPX save/load, or refinement divergence need investigation.
- Record any accepted warning in the validation notes for that run.

## Deployment Checks

- Validate experimental UX work on the VM UX service/branch first.
- Keep the public production service and Hugging Face version unchanged unless deployment is explicitly requested.
- After restarting a VM service, verify the service status and the HTTP endpoint.
- Confirm the ORNL-network URL and local forwarded URL are pointing to the intended service.
