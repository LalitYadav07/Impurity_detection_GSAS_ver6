# Reproducible Setup Guide for Impurity Detection GSAS Pipeline

This guide details the exact steps to recreate a fully functional environment ("one-click" setup) for the Impurity Detection project, based on rigorous analysis of the successful installation process.

## Prerequisites

- **OS**: Linux (tested on Ubuntu/Debian)
- **PackageManager**: `pixi` (will be installed automatically if missing)
- **Internet Connection**: Required for fetching packages and models.

## 1. Automated Setup (The "One-Click" Solution)

The project includes a `setup.sh` script that automates 90% of the work. However, some manual interventions were required during our "trial and error" phase. This guide synthesizes them into a robust workflow.

### Step-by-Step Execution

1.  **Clone the Repository** (Recursive clone is critical for submodules if they exist, though `setup.sh` handles `GSAS-II` explicitly)
    ```bash
    git clone --recurse-submodules <repo_url>
    cd Impurity_detection_GSAS_ver6
    ```

2.  **Run the Setup Script**
    This script installs Pixi, creates the environment, builds GSAS-II, and bridges the binaries.
    ```bash
    bash setup.sh
    ```
    *Note: If `setup.sh` fails at the validation step, verify that `gfortran` and `gcc` are available, although Pixi should handle dependencies.*

3.  **Critical Fixes (Pre-Applied)**
    The following fixes have been applied to the codebase and should persist in the repo:
    - **`app.py` NameError**: `IS_HF_SPACES` is now defined globally.
    - **Missing ML Models**: The `setup.sh` now attempts to run `pixi run install-models`.

4.  **Verify ML Components**
    Ensure the ranker model was downloaded correctly. If `setup.sh` skipped it or failed lightly:
    ```bash
    pixi run install-models
    ```

## 2. Manual Configuration Checklist

If the automated setup fails partialy, verify these specific items:

### A. Environment Variables
No manual export is needed if using `pixi run`. Pixi handles `PYTHONPATH`.
- `PYTHONPATH` must include `./GSAS-II`
- `MPLBACKEND` should be `Agg` for CLI runs.

### B. GSAS-II Compilation
The `setup.sh` compiles GSAS-II. If "Unable to load GSAS-II" errors persist:
1.  Enter the shell: `pixi shell`
2.  Navigate: `cd GSAS-II/pixi`
3.  Rebuild: `pixi run -e py312 install-editable` (Explicitly use py312 environment to match project python version)
4.  **Bridge Binaries**: Copy `.so` files from `GSAS-II/build` to `GSAS-II/GSASII/`.

### C. ML Ranker Model
Location: `ML_ranker/mlp_ranker_for_phase_detection-main/mlp_ranker.pt`
Size: ~30KB
Action: Run `pixi run install-models` if missing.

## 3. Launching the Application

### Web UI
```bash
pixi run ui
```

### CLI Pipeline
```bash
pixi run cli-run
```

## 4. Updates for Robustness
To make the `setup.sh` truly "one-click" without the discovered issues, I recommend the following updates to `setup.sh` (already verified in analysis):
- Ensure `pixi run install-models` is called with the correct python interpreter context.
- Explicitly check for `IS_HF_SPACES` logic compatibility (done in code).

---
**Status**: The current codebase, with the `app.py` fix and `setup.sh` workflow, is now fully replicable.
