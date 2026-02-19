[![DOI](https://zenodo.org/badge/1160645091.svg)](https://doi.org/10.5281/zenodo.18687992)

# Morphology Analysis Script

This repository contains a Python script for generating a time-lapsed image of morphological variation of a single cell.
This workflow was developed for and used in **Akinpelu et al. (2026)**.

## Overview

The primary script tracks the morphological boundaries of a cell over a series of microscopy images, producing a timelapsed morphological overlay of of the cell's boundary.
Users may select a ROI containing the desired cell to be analyzed, and `OpenCV` is used to segment and extract cell boundaries from each time frame.

## Repository Structure

The structure of the repository is outlined below in a tree diagram. 
The `.vscode` folder contains all of the necessary scripts for setting up a Python virtual environment on Windows or Unix systems for running this code (see [Usage](##Usage) below) for more info. 

```
morphology-analysis/
├── .gitignore
├── .vscode
│   ├── requirements.txt
│   ├── run_project.bat
│   ├── settings.json
│   ├── setup_venv.ps1
│   └── setup_venv.sh
├── LICENSE
├── README.MD
├── example_data
│   └── example_time_series.nd2
└── protrusionTracker.py
```

## Usage

The `.vscode` folder of this project is included to help users get this code up and running on both Windows and Unix systems.
A detailed description for each system type is provided below.
This guide assumes that you have downloaded and unpacked the repository into its own directory, and that you also have an existing installation of Python on your machine.

For any analysis performed, an output directory named `temp_out` will be generated at the root level of the repository folder.

For the selection of the ROI, users may click and drag to generate a rectangular selection region over the presented image.
If users are unhappy with the selected ROI, they may click anywhere and reselect the ROI, or press the `c` key to cancel/reset the ROI selection.
Users must then press `Enter` to run the analysis with their desired ROI selection.

### Windows Execution

To begin, Windows users may navigate to the downloaded repository through the file explorer.
Open a Powershell terminal at the project root, and run the following command:

`powershell -ExecutionPolicy Bypass -File ".\.vscode\setup_venv.ps1"`

You should find a Python virtual environment folder, `.venv` created at the root level of the project directory.

Windows users may now navigate back to the `.vscode` folder from the file explorer and run the `run_project.bat` file either by clicking on it or by running:

`.\.venv\Scripts\python.exe protrusionTracker.py`

A terminal will appear, prompting the user for the desired name of the output directory.
GUI prompts will carry the user through the rest of the script's execution.

### Unix Execution

From the root level of the project directory, you may run:

`bash .vscode/setup_venv.sh`

A virtual environment should now exist at the root level of the directory. 
The virtual environment can be activated, or you may run the following command from the root repository directory:

`./.venv/bin/python protrusionTracker.py`

A command line prompt will ask the user for the desired name of the output directory.
`Tkinter` GUI prompts will be used to guide the user through the remainder of the analysis.

## Licensing and Citations

This project is licensed under the MIT License - see the [LICENSE file](LICENSE) for details. 

If you use this repository for analysis, please cite:
**Akinpelu et al. 2026**
