---
title: RADAR-PD
emoji: "\U0001F52C"
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# RADAR-PD

RADAR-PD, Residual-Aware Deep-learning-Assisted Refinement for Powder Diffraction, is a GSAS-II based phase-discovery workflow for powder diffraction. It combines machine-learning candidate screening, lattice nudging, and sequential Rietveld refinement to identify and quantify impurity phases.

The app supports neutron CW, neutron TOF, and laboratory X-ray workflows. It can run with a known main phase CIF or in no-main discovery mode, and it includes tools for sample-environment handling, manual ignored regions, and automatic Al/Cu/V reference peak masking.

## Quick Start On Hugging Face

Use the hosted Space:

`https://huggingface.co/spaces/Lalityadav07/phase_detection`

1. Choose `Neutron` or `X-ray`.
2. Select a database mode. Use `Original` for the built-in database, or choose a custom augmented/mini pack if one has been built.
3. Set `Example Mode` to a demo, or leave it as `None` and upload your own diffraction data.
4. Upload an instrument parameter file. For X-ray CW screening, you can use the built-in Cu K-alpha lab PXRD preset when appropriate.
5. Optionally upload a main CIF. If omitted, the pipeline starts from Stage 0 discovery.
6. Enter allowed chemistry and sample-environment elements.
7. Configure fit windows or ignored regions if needed.
8. Click `RUN PIPELINE`.

Outputs are written under `runs/<run_name>/` and can be browsed from the app after the run starts.

## Local Installation

RADAR-PD uses Pixi to provide a reproducible Python and scientific-computing environment.

```bash
git clone https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6.git
cd Impurity_detection_GSAS_ver6
```

Install Pixi from `https://pixi.sh`, then run setup:

```bash
# Linux
pixi run setup-linux

# Windows PowerShell
pixi run setup-win
```

The setup scripts install dependencies, prepare GSAS-II, and build or link the numerical binaries needed by GSAS-II.

## Run The GUI Locally

```bash
pixi run ui
```

The default Streamlit URL is:

`http://localhost:8501`

If you need a different port:

```bash
pixi run streamlit run app.py --server.port 8503 --server.address 0.0.0.0
```

## Run From The CLI

Validate the example configuration:

```bash
pixi run cli-test
```

Run all datasets from a config:

```bash
pixi run cli-run
```

Run a specific config and dataset:

```bash
pixi run python scripts/gsas_complete_pipeline_nomain.py \
  --config /path/to/pipeline_config.yaml \
  --dataset my_dataset
```

Dry-run a custom config:

```bash
pixi run python scripts/gsas_complete_pipeline_nomain.py \
  --config /path/to/pipeline_config.yaml \
  --dry-run
```

GUI-created runs save the generated config at:

`runs/<run_name>/pipeline_config.yaml`

That file can be reused directly from the CLI.

## Inputs

Required:

- Diffraction data: `.dat`, `.xye`, `.gsa`, or `.fxye`.
- Instrument parameters: GSAS-II `.instprm`, or a supported profile format that the app can normalize to `.instprm`.
- Allowed elements: host/sample chemistry used for database filtering.

Optional:

- Main CIF: if supplied, Stage 1 refines the main phase before impurity discovery.
- Hardware / SE elements: sample-environment elements such as `Al`, `Cu`, or `V`.
- Fit window: outer x-axis range to refine.
- Ignored regions: manual x-axis windows excluded from refinement and downstream residual screening.
- Reference/can masks: generated windows around known Al/Cu/V Bragg peaks.

Native x-axis units depend on instrument mode:

- CW and lab X-ray: `2theta` in degrees.
- TOF: microseconds.

## Database Modes

The app has separate neutron and X-ray database roots. The radiation selector controls which database is active.

Available modes:

- `Original`: built-in database for the selected radiation source.
- `Augmented Pack`: built-in database plus user-provided CIFs.
- `Mini Pack`: standalone user-provided CIF database.

Custom packs are stored under:

`data/user_db_packs/<pack_name>/<neutron_or_xray>/`

Use `Build Custom Pack` in the GUI to create a pack from CIF uploads. Successful builds are auto-selected for subsequent runs.

## Sample Environment And Peak Masks

`Hardware / SE` and `Auto-mask reference/can Bragg peaks` are related but separate.

`Hardware / SE` controls chemistry filtering. For example, putting `Al` there allows pure Al or allowed Al-oxide sample-environment phases without letting Al mix into host-chemistry candidates.

`Auto-mask reference/can Bragg peaks` controls data masking. It computes Bragg peak positions from bundled reference structures and the actual instrument file, then excludes those regions before refinement and ML residual screening.

Supported bundled presets:

- `Al_fcc`
- `Cu_fcc`
- `V_bcc`

Width controls:

- `Auto from instrument profile`: uses profile terms from the instrument file, plus tolerance for reference-cell, alloying, and zero-offset mismatch.
- `Fixed half-width`: uses the same user-provided half-width for every generated peak.
- `d tolerance (%)`: widens masks for cell mismatch or alloy shifts.
- `Zero tolerance`: widens masks for residual center-position mismatch.
- `Min/Max half-width`: clamps generated windows to avoid masks that are too narrow or too broad.
- `Also mask Cu K-beta positions`: useful for Cu-anode lab X-ray data.

The audit file is written to:

`runs/<run_name>/Technical/Logs/reference_phase_exclusions.json`

It records generated centers, windows, hkl values, instrument parameters, and warnings.

## Pipeline Stages

The exact path depends on whether a main CIF is provided, but the major steps are:

1. Prepare input files, instrument parameters, database, fit limits, and ignored regions.
2. Optionally perform light PXRD calibration for X-ray runs with a main CIF.
3. Refine the main phase, or run Stage 0 no-main discovery if no main CIF is supplied.
4. Extract residuals on the active, masked histogram.
5. Screen candidate phases with the ML histogram model.
6. Apply knee filtering to prioritize candidates.
7. Nudge candidate lattices and evaluate residual agreement.
8. Run sequential joint refinements and accept phases that improve the model.
9. Write plots, summaries, refined CIFs, and logs.

## Output Layout

Each run is self-contained:

```text
runs/<run_name>/
  pipeline_config.yaml
  pipeline.log
  inputs/
  Results/
    Summary_Fractions.csv
    Plots/
  Diagnostics/
    Screening_Histograms/
    Residual_Scanning/
  Models/
    Reference_CIFs/
    Refined_CIFs/
  Technical/
    GSAS_Projects/
    Logs/
```

Key files:

- `pipeline.log`: full text log for GUI and CLI runs.
- `Results/Summary_Fractions.csv`: final phase quantification table.
- `Results/Plots/main_phase_fit.png`: main-phase baseline fit when a main CIF is supplied.
- `Results/Plots/seq_pass*_accepted_model.png`: accepted model after each pass.
- `Diagnostics/Screening_Histograms/pass*/hist_grid.png`: ML screening diagnostics.
- `Technical/Logs/run_manifest.json`: structured run manifest.
- `Technical/Logs/reference_phase_exclusions.json`: generated reference/can mask audit when enabled.

## Configuration Reference

Important top-level config fields:

```yaml
instrument_mode: auto        # auto, cw, or tof
allowed_elements: [Ba, Er, Si, O]
max_passes: 3
min_impurity_percent: 0.5

background:
  mode: auto_fixed_points
  type: chebyschev-1
  terms: 12

element_filter:
  max_offlist_elements: 0
  wildcard_relation: same_family
  require_base: true
  sample_env:
    elements: [Al]
    allow_pure: true
    allow_with: [O]
    ban_cross_with_base: true
    ignore_in_budget: true

reference_phase_exclusions:
  enabled: true
  presets: [Al_fcc, Cu_fcc]
  window_mode: auto
  fwhm_factor: 6.0
  fractional_d_tolerance: 0.003
  zero_tolerance_deg: 0.05
  min_half_width_deg: 0.35
  max_half_width_deg: 2.00
  include_cu_kbeta: false
```

Dataset entries:

```yaml
datasets:
  - name: my_run
    mode: auto
    data_path: /abs/path/pattern.xye
    instprm_path: /abs/path/instrument.instprm
    main_cif: /abs/path/main_phase.cif   # optional
    fmthint: xye
    limits: [5.0, 125.0]
    exclude_regions:
      - [70.0, 71.3]
      - [83.0, 84.5]
```

The complete example config lives at:

`scripts/pipeline_config.yaml`

## Troubleshooting

### GSAS-II binaries are missing

If you see `Unable to find GSAS-II binaries`, rerun setup:

```bash
pixi run setup-linux
# or
pixi run setup-win
```

### Windows paths with spaces

Avoid installing the repo under paths with spaces, OneDrive sync folders, or deeply nested folders. GSAS-II build tools are more reliable under simple paths such as:

`C:\Coding\Impurity_detection_GSAS_ver6`

### Database missing

The built-in databases are not stored fully in Git. In the GUI, use the database download/install panel or build/select a custom pack. Locally, the expected built-in database roots are:

- `data/database_neutron/`
- `data/database_xray/`

### Reference/can peak still visible

If a can peak shoulder remains visible after masking:

1. Confirm the correct preset is selected, for example `Cu_fcc` for copper metal.
2. Open `Technical/Logs/reference_phase_exclusions.json` and verify generated centers.
3. Increase `d tolerance (%)`, `Zero tolerance`, or `Max half-width`.
4. For lab X-ray data, enable Cu K-beta masking if K-beta contamination is present.

### CLI logs are not where expected

GUI runs write `pipeline.log` inside `runs/<run_name>/`. CLI runs also use the dataset run directory when `work_root` or dataset paths are set through the generated GUI config. If running from a hand-written config, check `work_root`, `WORK_ROOT`, and dataset `name`.

## Development Notes

Useful checks:

```bash
python3 -m py_compile app.py scripts/gsas_complete_pipeline_nomain.py scripts/config_builder.py
python3 -m unittest tests.test_reference_phase_masks tests.test_db_pack_regressions
```

The Hugging Face Space is synchronized by GitHub Actions through `.github/workflows/hf_sync.yml`.
