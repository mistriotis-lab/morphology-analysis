#!/bin/bash

# ==========================================================================
# This is a script to create a Python virtual environment at the top level
# of this project which includes all packages and dependencies required.
# ==========================================================================

# Setup Paths:
# (1) Current Directory
curr_dir="$(pwd)"
# (2) This file's path
this_file="$(realpath "$0")"
this_dir="$(dirname "$this_file")"
cd "$this_dir"

# Create the venv
echo "Now creating the virtual environment from the python located at: "
which python3
python3 -m venv ../.venv

# Install this package, with dependencies, as editable
../.venv/bin/python -m pip install -r requirements.txt

# Return to previous working directory 
cd "$curr_dir"