---
title: GSAS-II Impurity Detector
emoji: 🔬
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

# 🔬 GSAS-II Impurity Detector (v7.0)

This platform leverages **GSAS-II** to automate the discovery and refinement of impurity phases in powder diffraction data. It combines machine learning candidate screening with rigorous Rietveld refinement to provide high-confidence phase identification.

---

## 🚀 Quick Start (Hugging Face / Cloud)

The easiest way to use the tool is via our hosted [Hugging Face Space](https://huggingface.co/spaces/Lalityadav07/phase_detection).

1. Upload your `.dat` or `.xye` file.
2. Upload your `.instprm` file.
3. Select an Example (LK-99 or TbSSL) for a quick demo.
4. Pulse the **Start Pipeline** button.

---

## 🛠️ Local Installation (Windows & Linux)

We use **Pixi** to manage all scientific dependencies and GSAS-II binaries in an isolated environment.

1. **Install Pixi**: [pixi.sh](https://pixi.sh)
2. **Clone the Repo**:

    ```bash
    git clone https://github.com/Lalityadav07/Impurity_detection_GSAS_version7.git
    cd Impurity_detection_GSAS_version7
    ```

3. **Run Setup**:
    - **Windows**: `.\setup.ps1`
    - **Linux**: `./setup.sh`

This will clone GSAS-II, build the necessary numerical binaries (`LATTIC`, `convcell`), and solve all Python dependencies (PyTorch, Pymatgen, etc.).

---

## 💻 Usage

### 🌐 Web UI (Streamlit)

The Web UI provides a premium dashboard for monitoring runs, viewing live logs, and exploring results.

```bash
pixi run ui
```

Access the dashboard at `http://localhost:8501`.

### ⌨️ Command Line Interface (CLI)

For batch processing or high-performance environments (HPC), use the CLI tool directly.

```bash
# Verify the environment with a dry-run
pixi run cli-test

# Run a full pipeline with a configuration file
pixi run cli-run -- --config my_pipeline_config.yaml
```

---

## 📂 Project Architecture

### Directory Structure

- `app.py`: Main Streamlit application.
- `scripts/`: Implementation of the discovery pipeline and ML screening.
- `GSAS-II/`: The core GSAS-II scientific engine (git-ignored submodule).
- `ML_components/`: Pre-trained models for Stage-3 screening.
- `data/database_aug/`: Crystallographic database (required for screening).
- `runs/`: Output directory for generated artifacts and logs.

### Key Components

- **ML Screening**: Rapid candidate selection using element family and spacegroup filters. Now includes detailed metadata (Compound Name, Space Group).
- **Lattice Nudging**: Adaptive lattice parameter optimization before Rietveld.
- **Sequential Passes**: Each pass identifies the next most likely impurity, refining the global model until convergence. Multi-pass results are now viewable in organized expanders.
- **Decision Engine**: Visualizes the internal "Knee Filter" logic used to prune candidates.

---

## 🤝 Contributing

Scientific contributions, bug reports, and database updates are welcome. Please ensure that all changes maintain compatibility with the Pixi environment.

---

## 🛠️ Troubleshooting & Technical Notes

### 1. Missing GSAS-II Binaries

If you see `*** ERROR: Unable to find GSAS-II binaries`, it usually means the Fortran/C extensions weren't compiled or linked correctly.

- **Solution**: Run `.\setup.ps1` (Windows) or `./setup.sh` (Linux) again. The scripts are designed to "bridge" the compiled binaries into the Python path.

### 2. Python Version

This project requires **Python 3.12**.

### 3. Database Download

The database is ~2.3GB. If the download fails in the UI, you can manually download it from the link provided in the logs and extract it to `data/database_aug/`.
