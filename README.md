# 🔬 GSAS-II Impurity Detector

This platform leverages **GSAS-II** to automate the discovery and refinement of impurity phases in powder diffraction data. It combines machine learning candidate screening with rigorous Rietveld refinement to provide high-confidence phase identification.

---

## 🚀 Quick Start (Hugging Face / Cloud)
The easiest way to use the tool is via our hosted [Hugging Face Space](https://huggingface.co/spaces/Lalityadav07/phase_detection). 
1. Upload your `.dat` or `.xye` file.
2. Upload your `.instprm` file.
3. Pulse the **Start Pipeline** button.

---

## 🛠️ Local Installation (Windows & Linux)

We use **Pixi** to manage all scientific dependencies and GSAS-II binaries in an isolated environment.

1.  **Install Pixi**: [pixi.sh](https://pixi.sh)
2.  **Clone the Repo**:
    ```bash
    git clone https://github.com/Lalityadav07/Impurity_detection_GSAS_ver6.git
    cd Impurity_detection_GSAS_ver6
    ```
3.  **Run Setup**:
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
- `data/database_aug/`: Crystallographic database (required for screening).
- `runs/`: Output directory for generated artifacts and logs.

### Key Components
- **ML Screening**: Rapid candidate selection using element family and spacegroup filters.
- **Lattice Nudging**: Adaptive lattice parameter optimization before Rietveld.
- **Sequential Passes**: Each pass identifies the next most likely impurity, refining the global model until convergence.
- **Decision Engine**: Visualizes the internal "Knee Filter" logic used to prune candidates.

---

## 🤝 Contributing
Scientific contributions, bug reports, and database updates are welcome. Please ensure that all changes maintain compatibility with the Pixi environment.
