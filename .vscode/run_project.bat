@echo off
REM
cd /d "%~dp0"

REM Use the venv created by the setup scripts
set VENV_PYTHON=..\.venv\Scripts\python.exe

REM Check if venv exists, fall back to system if not
if not exist %VENV_PYTHON% (
    echo Virtual environment not found. Using system Python instead ...
    set VENV_PYTHON=python
)

REM Upgrade pip
%VENV_PYTHON% -m pip install --upgrade pip

REM Install required packages
%VENV_PYTHON% -m pip install -r requirements.txt

REM Run the main script at the project root
%VENV_PYTHON% ..\protrusionTracker.py

pause