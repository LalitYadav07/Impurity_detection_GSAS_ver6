#!/bin/bash
# setup.sh for GSAS-II Environment (Linux)
# This script installs Pixi, Git, and GSAS-II in an isolated environment.

set -e

echo "--- Starting GSAS-II Setup ---"

# 1. Install Pixi (if not already present)
if ! command -v pixi &> /dev/null; then
    echo "Installing Pixi..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
else
    echo "Pixi found at $(which pixi)"
fi

# 2. Install Git (if not already present)
if ! command -v git &> /dev/null; then
    echo "Installing Git via Pixi..."
    pixi global install git
fi

# 3. Clone / Update GSAS-II Repo
if [ ! -d "GSAS-II" ]; then
    echo "Cloning GSAS-II repository..."
    git clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II.git GSAS-II
else
    echo "GSAS-II repository found. Ensuring submodules are initialized..."
fi

# 4. Create Root Environment and Install GSAS-II
echo "Creating root environment and installing GSAS-II (this may take several minutes)..."
# 4.1 Install GSAS-II binaries using its internal pixi config
echo "Configuring GSAS-II binaries and pinning dependencies..."
cd GSAS-II/pixi
pixi add psutil
pixi run install-editable
cd ../..

# --- NEW: Binary Bridge ---
echo "Bridging compiled binaries to GSASII package..."
# On Linux, these are usually .so files or executables
FIND_PYD=$(find GSAS-II/build -name "*.so" -o -name "*.pyd" 2>/dev/null || true)
FIND_EXE=$(find GSAS-II/build -executable -type f -not -path "*/.*" 2>/dev/null || true)

if [ -n "$FIND_PYD" ] || [ -n "$FIND_EXE" ]; then
    cp $FIND_PYD GSAS-II/GSASII/ 2>/dev/null || true
    cp $FIND_EXE GSAS-II/GSASII/ 2>/dev/null || true
    echo "Bridge complete: binaries copied to GSASII package source."
fi

# 4.2 Initialize root environment dependencies
echo "Solving root environment dependencies..."
pixi install

# 5. Download ML Models (LFS Recovery)
echo "Ensuring ML models are downloaded..."
pixi run python scripts/download_models.py

# 6. Validation
echo -e "\n--- Validating Installation ---"
export PYTHONPATH="$PWD/GSAS-II"
TEST_CMD="import GSASII.GSASIIscriptable as G2sc; import GSASII.pyspg; print('OK', G2sc.__file__)"
# Ensure we use the pixi-managed python
RESULT=$(pixi run python -c "$TEST_CMD" 2>&1)

if [[ $RESULT == *"OK"* ]]; then
    echo -e "\nSUCCESS: GSAS-II and binaries are confirmed active!"
    echo "$RESULT"
else
    echo -e "\nWARNING: Validation failed or binaries missing. Check gfortran/gcc installation."
    echo "Output:"
    echo "$RESULT"
fi

echo -e "\n--- SETUP COMPLETE ---"
echo "To use the project, always run via pixi tasks: 'pixi run ui' or 'pixi run cli-run'"
echo "Refer to README.md for more details."
