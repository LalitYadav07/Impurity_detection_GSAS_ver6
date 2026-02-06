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

# 3. Clone / Update GSAS-II Repo
$repoDir = Join-Path $PWD "GSAS-II"
$git_exe = Join-Path $pixi_bin_dir "git.exe"

if (-not (Test-Path $repoDir)) {
    Write-Host "Cloning GSAS-II repository..."
    & $git_exe clone --depth 1 https://github.com/AdvancedPhotonSource/GSAS-II.git GSAS-II
}
else {
    Write-Host "GSAS-II repository found. Ensuring submodules are initialized..."
    & $git_exe submodule update --init --recursive
}

# 4. Create Root Environment and Install GSAS-II
Write-Host "Creating root environment and installing GSAS-II (this may take several minutes)..."
try {
    # 4.1 Install GSAS-II binaries using its internal pixi config
    Write-Host "Configuring GSAS-II binaries and pinning dependencies..."
    $g2PixiDir = Join-Path $repoDir "pixi"
    Push-Location $g2PixiDir
    
    # Ensure numerical stability and missing dependencies in internal environment
    pixi add "numpy<2.0" psutil
    
    Write-Host "Building GSAS-II binaries (this requires a C++/Fortran compiler)..."
    pixi run install-editable-win
    Pop-Location

    # --- NEW: Binary Bridge ---
    # GSAS-II build artifacts are often deep in the build directory. 
    # We move them to the package source for immediate availability.
    Write-Host "Bridging compiled binaries to GSASII package..."
    $buildDir = Join-Path $repoDir "build"
    if (Test-Path $buildDir) {
        $pydFiles = Get-ChildItem -Path $buildDir -Recurse -Filter "*.pyd"
        $exeFiles = Get-ChildItem -Path $buildDir -Recurse -Filter "*.exe"
        $destDir = Join-Path $repoDir "GSASII"
        
        foreach ($f in $pydFiles) { Copy-Item $f.FullName $destDir -Force }
        foreach ($f in $exeFiles) { Copy-Item $f.FullName $destDir -Force }
        Write-Host "Bridge complete: $($pydFiles.Count) extensions and $($exeFiles.Count) binaries copied."
    }

    # 4.2 Initialize root environment dependencies
    Write-Host "Solving root environment dependencies..."
    pixi install
}
catch {
    Write-Error "Environment setup failed: $_"
}

# 5. Validation
Write-Host "`n--- Validating Installation ---" -ForegroundColor Cyan
# We set PYTHONPATH to include GSAS-II during validation to match production behavior
$env:PYTHONPATH = $repoDir
$testCmd = "import GSASII.GSASIIscriptable as G2sc; import GSASII.pyspg; print('OK', G2sc.__file__)"
$result = pixi run python -c $testCmd

if ($result -match "OK") {
    Write-Host "`nSUCCESS: GSAS-II and binaries are confirmed active!" -ForegroundColor Green
    Write-Host $result
}
else {
    Write-Host "`nWARNING: Validation failed or binaries missing. Check gfortran installation." -ForegroundColor Yellow
    Write-Host "Output: $result"
}

Write-Host "`n--- SETUP COMPLETE ---" -ForegroundColor Cyan
Write-Host "To use the project, always run via pixi tasks: 'pixi run ui' or 'pixi run cli-run'"
Write-Host "If binaries are still missing, refer to the Troubleshooting section in README.md."
