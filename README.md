# Dynamic labyrinth  detection, analysis and matching

This repository contains a Python implementation for lego dynamic labyrinth detection and analysis.
Each folder contains README with deeper description of each process.

## Features

- **Synthetic data generation**: With provided Blender models of details renders 3d boards of several types and creates datasets in available formats
- **Pipeline**: provides simple way to run the full created pipeline - from board segmentation to json os details creation

## Prerequisites

### Python Libraries
To run all processes available in the project ensure to install all libraries from requirements.txt.

Install them via pip if necessary:
```bash
pip install -r requirements.txt
```

### Configs

- **Gen**: synthetic data generation requires 2 configs, templates of which are provided in gen/configs_example
  - **detail_config.json** contains general information about details, their classes and labels
  - **render_option.json** contains characteristics of environment where board is rendered
- **Pipeline** - pipeline/func/configs contains base_config with general template of json file to log details into

## Models

Pipeline segmentation and detection processes require models, which should be stored in pipeline/models folder.

## Usage

- **Gen**: can be run with main.py with ability to change generator, augmentor and formatter
- **Pipeline**: can be run with main.py

## Extending the Code
- **Add orientation detection**:
  For json formation add the detail rotation detection for path reconstruction
- **Correctness analysis**:
  Check constructed dynamic labyrinth for correct performance
