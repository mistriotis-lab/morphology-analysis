# ==========================================================================
# This script creates a Python virtual environment at the top level
# of the project and installs dependencies from requirements.txt
# ==========================================================================

# Command prompt command to execute this script
# powershell -ExecutionPolicy Bypass -File setup_venv.ps1

# Save current directory
$curr_dir = Get-Location

# Get this script's directory
$this_dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $this_dir

# Path to venv at project root (one folder up from script)
$venv_path = Join-Path $this_dir "..\.venv"

# Show Python being used
Write-Host "Now creating the virtual environment from the python located at:"
Get-Command python

# Create venv
py -m venv $venv_path
if (!$?) { python -m venv $venv_path }

# Install dependencies into the venv
# & "$venv_path\Scripts\python.exe" -m pip install --upgrade pip
& "$venv_path\Scripts\python.exe" -m pip install -r requirements.txt

# Return to previous working directory
Set-Location $curr_dir