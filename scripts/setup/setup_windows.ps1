# UTTA-Med Windows setup — run from repository root
# Usage:  powershell -ExecutionPolicy Bypass -File .\SETUP.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Working directory: $(Get-Location)"
if (-not (Test-Path ".\requirements.txt")) {
    Write-Host "ERROR: requirements.txt not found here."
    Write-Host "Extract UTTA-Med-code.zip INTO this folder (next to .venv), then re-run."
    Get-ChildItem
    exit 1
}

if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}

& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

if (-not $env:CAMELYON17_ROOT) {
    Write-Error "CAMELYON17_ROOT is not set. Please set the CAMELYON17_ROOT environment variable (e.g. `$env:CAMELYON17_ROOT='C:\path\to\Camelyon17-Data') before running SETUP.ps1."
    exit 1
}

$root = $env:CAMELYON17_ROOT
Write-Host "Inspecting Camelyon17 at $root"
python scripts\inspect_local_data.py --data-root $root
