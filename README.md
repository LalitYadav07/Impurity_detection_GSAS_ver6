# GSAS-II & Scientific Computing Environment

This project provides a robust, isolated Python environment for GSAS-II crystallography scripting combined with a modern scientific data analysis stack.

## 🚀 Quick Start

### Windows
1.  **Clone / Download** this repository.
2.  **Run Setup**:
    ```powershell
    .\setup.ps1
    ```
    *This will install Pixi (package manager), Git (if missing), and the GSAS-II environment.*
3.  **Activate & Use**:
    ```powershell
    .\activate.ps1
    ```
    *You are now inside the environment and can run `python` directly.*

### Linux
1.  **Run Setup**:
    ```bash
    bash setup.sh
    ```
2.  **Activate**:
    ```bash
    cd GSAS-II/pixi
    pixi shell
    ```

---

## 🛠️ What's Included

### 📊 Crystallography
- **GSAS-II**: Full installation in editable mode, ready for scripting via `GSASIIscriptable`.

### 🔬 Scientific Stack
- **Data Analysis**: `pandas`, `numpy`, `scipy`
- **Visualization**: `matplotlib`
- **Machine Learning**: `pytorch` (CPU-optimized), `scikit-learn`
- **Materials Science**: `pymatgen`
- **Configuration**: `pyyaml`, `json`

---

## 💡 Usage Examples

### Import Verification
Once the environment is activated, you can verify everything is working:
```python
import GSASII.GSASIIscriptable as G2sc
import torch
import pandas as pd
import pymatgen

print(f"GSAS-II located at: {G2sc.__file__}")
print(f"PyTorch version: {torch.__version__}")
```

### Running Scripts via Pixi
If you don't want to enter the shell, you can run scripts directly:
```powershell
# From the project root
.\GSAS-II\pixi\.pixi\envs\default\Scripts\python.exe your_script.py
```

---

## ⚡ Running the Pipeline

The impurity detection pipeline integrates GSAS-II refinement with ML-based candidate screening.

### Standard Execution (Windows)
To run the pipeline on the `cw_tbssl` dataset:
1.  **Open PowerShell** in the project root.
2.  **Run**:
    ```powershell
    cd GSAS-II\pixi
    pixi run python ..\..\scripts\gsas_complete_pipeline_nomain.py --config ..\..\scripts\pipeline_config.yaml --dataset cw_tbssl | Tee-Object -FilePath ..\..\run_cw_tbssl.log
    ```

### Command Arguments
- `--config`: Path to the `pipeline_config.yaml`.
- `--dataset`: The name of the dataset to process (e.g., `cw_tbssl`).
- `--dry-run`: (Optional) Validates configuration and database without running GSAS-II refinements.

---

## ❓ Troubleshooting

### Windows Encoding / `charmap` Errors
If you see errors like `'charmap' codec can't encode characters`, ensure you are using a terminal that supports **UTF-8**. 
> [!NOTE]
> The pipeline script has been updated to force UTF-8 for stdout, but some environments may still require manual terminal configuration (`chcp 65001`).

### "Main thread is not in main loop"
This error occurs if GSAS-II tries to initialize a GUI in a headless environment. 
> [!TIP]
> The pipeline script includes monkeypatches to force `GSASIIctrlGUI.haveGUI = False` and use the Matplotlib `Agg` backend. If this persists, ensure no `wx` or `matplotlib` imports occur before these patches.

### Permanent Path Update (Windows)
If `pixi` command is still not found in new windows after running `setup.ps1`, you may need to:
1.  Restart your terminal.
2.  Or manually add `%USERPROFILE%\.pixi\bin` to your system/user `Path` environment variable.

---

## 📜 Installation Details
This environment is managed by [Pixi](https://pixi.sh), a high-performance package manager. All dependencies are isolated within the `GSAS-II/pixi/.pixi` folder, ensuring no interference with your system Python.
