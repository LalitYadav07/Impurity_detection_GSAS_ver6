# setup.ps1 for GSAS-II Environment (Windows)
# This script installs Pixi, Git, and GSAS-II in an isolated environment.

$ErrorActionPreference = "Stop"

Write-Host "--- Starting GSAS-II Setup ---" -ForegroundColor Cyan

# 1. Install Pixi (if not already present)
$pixi_bin_dir = "$env:USERPROFILE\.pixi\bin"
$pixi_exe = Join-Path $pixi_bin_dir "pixi.exe"

if (-not (Test-Path $pixi_exe)) {
    Write-Host "Installing Pixi..."
    powershell -ExecutionPolicy ByPass -Command "irm -useb https://pixi.sh/install.ps1 | iex"
}
else {
    Write-Host "Pixi found at $pixi_exe"
}

# Add Pixi to current path for this session and refresh environment
if ($env:Path -notlike "*\.pixi\bin*") {
    $env:Path = "$pixi_bin_dir;" + $env:Path
}

# Alias pixi to the absolute path just in case
function pixi { & $pixi_exe @args }

# 2. Install Git via Pixi (if not already present)
if (-not (Test-Path (Join-Path $pixi_bin_dir "git.exe"))) {
    Write-Host "Installing Git via Pixi..."
    pixi global install git
}

# 3. Clone GSAS-II Repo
$repoDir = Join-Path $PWD "GSAS-II"
if (-not (Test-Path $repoDir)) {
    Write-Host "Cloning GSAS-II repository..."
    $git_exe = Join-Path $pixi_bin_dir "git.exe"
    & $git_exe clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II.git GSAS-II
}
else {
    Write-Host "GSAS-II repository already exists."
}

# 4. Create Root Environment and Install GSAS-II
Write-Host "Creating root environment and installing GSAS-II (this may take several minutes)..."
try {
    # 4.1 Install GSAS-II binaries using its internal pixi config
    Write-Host "Configuring GSAS-II binaries..."
    $g2PixiDir = Join-Path $repoDir "pixi"
    Push-Location $g2PixiDir
    pixi run install-editable-win
    Pop-Location

    # 4.2 Initialize root environment dependencies
    Write-Host "Solving root environment dependencies..."
    pixi install
}
catch {
    Write-Error "Environment setup failed: $_"
}

# 5. Validation
Write-Host "`n--- Validating Installation ---" -ForegroundColor Cyan
$testCmd = "import GSASII.GSASIIscriptable as G2sc; print('OK', G2sc.__file__)"
$result = pixi run python -c $testCmd

if ($result -match "OK") {
    Write-Host "`nSUCCESS: GSAS-II is importable!" -ForegroundColor Green
    Write-Host $result
}
else {
    Write-Error "Validation failed. Output: $result"
}

Write-Host "`n--- SETUP COMPLETE ---" -ForegroundColor Cyan
Write-Host "To start the environment, simply run: .\activate.ps1"
Write-Host "If 'pixi' command is still not found in new windows, please restart your computer."
