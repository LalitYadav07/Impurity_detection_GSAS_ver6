# Tutorials

The tutorials are written as scientific scenarios. The exact filenames can change as the hosted demo workspace is refreshed, but each scenario should be reproducible from a saved documentation run.

## Tutorial 1: TBSSL Neutron With Main Phase

Purpose: show a normal known-main-phase neutron workflow.

Inputs:

- neutron diffraction data from the TBSSL example,
- matching instrument parameter file,
- supplied main phase CIF,
- sample elements entered in Sample elements,
- can/environment elements entered separately if applicable.

Mode: Full RADAR-PD.

What to inspect:

- main phase fit,
- residual peaks,
- accepted impurity candidates,
- final phase fractions,
- publication plot and interactive plot.

## Tutorial 2: LK-99 Style Impurity Search

Purpose: demonstrate Rapid Hypothesis Mode and alternate phase-family representatives.

Run variants:

- without supplied main phase, to see whether the phase family is discovered,
- with supplied main phase, to focus on secondary phases.

Expected concepts:

- coarse search finds broad families,
- lattice nudging improves peak alignment,
- pattern scoring reranks hypotheses,
- solution inspector lets the user swap variants such as copper sulfide representatives.

## Tutorial 3: Four-Phase Oxide Mixture

Purpose: show multiple comparable phases where the best rapid histogram result and the best GSAS-II refinement may not be identical.

What to inspect:

- whether all known oxide families appear in the shortlist,
- whether phase fractions are realistic,
- whether strong peaks of each phase are supported,
- whether a phase is overfitting background or main-phase peaks.

## Tutorial 4: Mystery Peaks / Hard Neutron Case

Purpose: show a difficult, long-running dataset.

Use this tutorial to teach:

- why Full RADAR-PD can take much longer,
- where partial outputs appear,
- how to inspect live artifacts before the run completes,
- how to compare rapid and full expectations.

## Tutorial 5: Container / Environment Peaks

Purpose: show how can or environment peaks should be handled.

Demonstrate three policies:

- enter can elements under Sample can / environment,
- add ignored regions around known artifacts,
- enable any prebuilt can or reference mask option if available.

Key concept: container elements should not be freely mixed with sample chemistry unless the user intentionally allows that scientific interpretation.

## Tutorial 6: PXRD / XRDML Workflow

Purpose: show laboratory X-ray analysis.

Inputs:

- `.xrdml` or converted X-ray data,
- X-ray instrument information,
- optional main phase CIF.

What to discuss:

- X-ray mode,
- K-alpha doublet behavior,
- background sensitivity,
- peak overlap in 2-theta,
- why a neutron-style library or instrument file is not appropriate.

## Tutorial 7: Custom CIF Library

Purpose: show user-owned candidate libraries.

Steps:

1. Upload a small set of CIF files.
2. Build Built-in + my CIFs.
3. Build Only my CIFs.
4. Run the same data against each choice.
5. Compare whether the custom phases appear in the results.

## Tutorial 8: API / Batch Use

Purpose: show how an experiment script can submit scans as they appear.

Steps:

1. Configure a run in the web app.
2. Download API config.
3. Submit config, data, instrument, and optional main CIF from Python.
4. Poll status.
5. Download `results.zip`.
6. Repeat for a new scan file.

## Verified API Tutorial Evidence

The hosted API workflow was checked with an LK-99 rapid-mode submission using the public API base URL. The accepted job returned a job ID, completed with `state: complete`, reported `Complete Rapid` at 100 percent, and exposed the result archive through `results.zip`.

Files captured for maintainers:

- `docs/user_manual/examples/lk99_api_config.yaml`
- `docs/user_manual/examples/lk99_api_submit_response.json`
- `docs/user_manual/examples/lk99_api_status_latest.json`
- `docs/user_manual/examples/lk99_api_artifacts.json`

These files are not required for ordinary users, but they make the API tutorial reproducible when the hosted service changes.

