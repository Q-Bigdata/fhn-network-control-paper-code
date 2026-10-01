# FitzHugh–Nagumo Network Control

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23074655.svg)](https://doi.org/10.5281/zenodo.23074655)

This repository contains the reproducible Python implementation accompanying the paper on dimensionality reduction and collective control in FitzHugh–Nagumo networks.

## Overview

The program constructs directed benchmark networks, simulates the network and reduced FitzHugh–Nagumo models with fourth-order Runge–Kutta integration, estimates oscillation regimes, and evaluates control strategies based on driver-node selection. It produces the analyses and figures used in the paper.

## Repository contents

- `demo.py`: main simulation and figure-generation program.

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy
- NetworkX
- Matplotlib

Install the Python dependencies with:

```bash
pip install numpy scipy networkx matplotlib
```

## Running the analysis

Run the main program from the repository directory:

```bash
python demo.py
```

The default run writes figures and JSON summaries to `nature_output_U(0,0.01)+p0.02/`. The main analyses can be selected with the environment variables `RUN_FIG2` through `RUN_FIG6`; numerical convergence checks can be enabled with `RUN_CONVERGENCE_CHECKS=1`. Runtime, trial counts, and output settings are also configurable through the environment variables defined in `demo.py`.

## Reproducibility

The code records run parameters and uses fixed seeds for the shared benchmark networks. For a fast smoke test, reduce `N_TIME`, `N_WORKERS`, and the trial-count variables before running. For the full paper analysis, use the default settings and retain the generated JSON summaries alongside the figures.

## Citation

When using this code, please cite the accompanying paper and the versioned software record:

- GitHub repository: https://github.com/Q-Bigdata/fhn-network-control-paper-code
- Zenodo DOI: https://doi.org/10.5281/zenodo.23074655
- GitHub release: https://github.com/Q-Bigdata/fhn-network-control-paper-code/releases/tag/v1.0.1-publication
