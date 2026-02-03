# GSAS-II Pipeline Web UI

This application provides a user-friendly interface to configure, run, and monitor the GSAS-II impurity detection pipeline.

## 🚀 How to Run Locally

1.  **Ensure the environment is set up**:
    If you haven't already, run `.\setup.ps1` in the root directory.

2.  **Launch the UI**:
    Open a PowerShell terminal and run:
    ```powershell
    cd GSAS-II\pixi
    pixi run ui-run
    ```
    Alternatively, if you're already in the root:
    ```powershell
    cd GSAS-II\pixi
    pixi run streamlit run ../../app.py
    ```

3.  **Access the App**:
    The terminal will provide a local URL (usually `http://localhost:8501`). Open this in your browser.

## 🛠️ Usage Guide

### 1. Upload Mandatory Files
- **Diffraction Data**: Upload your experimental pattern (e.g., `HB2A_TbSSL.dat`).
- **Instrument Parameters**: Upload the `.instprm` file corresponding to the instrument (e.g., `HB2A_standard.instprm`).
- **Main Phase CIF**: (Optional) Upload the CIF for the primary phase. If omitted, the pipeline will attempt to "Bootstrap" and find the best match automatically.

### 2. Configure Run
- **Allowed Elements**: Select all elements that could potentially be in the sample.
- **Run Name**: Give your run a name (defaults to timestamp).
- **Settings**: Adjust sensitivity (Wt% threshold) and max passes.

### 3. Monitor Results
- **Run & Progress**: Watch the live logs and progress bar. Plots will refresh as they are generated.
- **Results**: See the final phase composition table and download artifacts like refined `.gpx` files and the summary CSV.
- **Explainability**: Understand why certain candidates were chosen via ML histogram and Pearson rankings.

## 📂 Output Structure
Every run creates a new folder under `runs/`:
```text
runs/
└── <run_name>/
    ├── config.yaml          # Generated config
    ├── run_events.jsonl     # Progress events
    ├── run_manifest.json    # Final artifact list
    ├── inputs/              # Uploaded files
    ├── joint/               # Refined GPX and CSVs
    └── plots/               # Fit pngs and ML histograms
```

## ❓ Troubleshooting
- **Missing Database**: Ensure the `data/database_aug` folder exists and contains the necessary files.
- **GSAS-II Import Errors**: Make sure `pixi run` is used to ensure the correct Python environment.
