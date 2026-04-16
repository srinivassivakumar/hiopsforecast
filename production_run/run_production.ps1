$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

if (-not (Test-Path .venv)) {
    python -m venv .venv
}

. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_pipeline.py --config config.yaml
