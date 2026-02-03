# activate.ps1 - GSAS-II Environment Activation
# Run this to enter the GSAS-II Python environment.

$pixi_exe = "$env:USERPROFILE\.pixi\bin\pixi.exe"
$pixi_bin_dir = "$env:USERPROFILE\.pixi\bin"

# 1. Update Path for this session
if ($env:Path -notlike "*\.pixi\bin*") {
    $env:Path = "$pixi_bin_dir;" + $env:Path
}

# 2. Check if Pixi exists
if (-not (Test-Path $pixi_exe)) {
    Write-Error "Pixi not found! Please run .\setup.ps1 first."
    exit 1
}

# 3. Navigate and shell
$pixiDir = Join-Path $PWD "GSAS-II/pixi"
if (-not (Test-Path $pixiDir)) {
    Write-Error "GSAS-II directory not found! Please run .\setup.ps1 first."
    exit 1
}

Write-Host "Entering GSAS-II environment..." -ForegroundColor Cyan
Push-Location $pixiDir
try {
    & $pixi_exe shell
}
finally {
    Pop-Location
}
