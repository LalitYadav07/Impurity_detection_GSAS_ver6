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

# 3. Clone GSAS-II Repo
if [ ! -d "GSAS-II" ]; then
    echo "Cloning GSAS-II repository..."
    git clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II.git GSAS-II
else
    echo "GSAS-II repository already exists."
fi

# 4. Create Environment and Install GSAS-II
echo "Creating environment and installing GSAS-II (this may take several minutes)..."
cd GSAS-II/pixi
# Add extra scientific dependencies
echo "Adding Scientific dependencies (pandas, pyyaml, pytorch-cpu, scikit-learn, pymatgen)..."
pixi add pandas pyyaml pytorch-cpu scikit-learn pymatgen

# On Linux we use install-editable
pixi run install-editable

# 5. Validation
echo -e "\n--- Validating Installation ---"
TEST_CMD="import GSASII.GSASIIscriptable as G2sc; print('OK', G2sc.__file__)"
RESULT=$(pixi run python -c "$TEST_CMD" 2>&1)

if [[ $RESULT == *"OK"* ]]; then
    echo -e "\nSUCCESS: GSAS-II is importable!"
    echo "$RESULT"
else
    echo -e "\nValidation failed. Output:"
    echo "$RESULT"
    exit 1
fi

echo -e "\nTo use the environment, run commands via: cd GSAS-II/pixi && pixi run python ..."
echo "Or activate the shell: cd GSAS-II/pixi && pixi shell"
