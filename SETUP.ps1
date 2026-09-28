# UTTA-Med Windows setup — run from C:\Users\avina\OneDrive\Desktop\UTTA-Med
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

$root = "C:\Users\avina\OneDrive\Desktop\Camelyon17-Data"
$env:CAMELYON17_ROOT = $root
Write-Host "Inspecting Camelyon17 at $root"
python scripts\inspect_local_data.py --data-root $root
