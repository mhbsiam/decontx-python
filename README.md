# DecontX Python
[![PyPI version](https://badge.fury.io/py/decontx-python.svg)](https://badge.fury.io/py/decontx-python)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)

A Python implementation of DecontX for removing ambient RNA contamination from single-cell RNA-seq data, designed for seamless integration with scanpy workflows.

## Overview

DecontX is a Bayesian method to estimate and remove cross-contamination from ambient RNA in droplet-based single-cell RNA-seq data. This Python implementation provides near-perfect parity with the original R version (correlation > 0.999) while enabling pure Python workflows without R dependencies.

**Key Features:**
- Pure Python implementation (no R required)
- Seamless scanpy integration
- Numba-accelerated performance
- Bayesian contamination estimation per cell
- Validated against original R implementation

## Installation

### Latest version from GitHub

```bash
pip install git+https://github.com/NiRuff/decontx-python.git
```

This installs the current `main` branch directly. Use this if you need the newest changes (for example, the performance improvements or `compute_log_likelihood` described in [PERFORMANCE.md](PERFORMANCE.md)).

For development install instructions and how to run tests, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Quickstart

```python
import decontx

decontx.decontx(adata, cluster_key="leiden")
```

The function modifies `adata` in place unless `copy=True`. See [API.md](API.md) for the full API and [WORKFLOW.md](WORKFLOW.md) for how to slot it into a scanpy pipeline.

## Documentation

- [API reference](API.md)
- [Workflow guide](WORKFLOW.md)
- [Performance tips](PERFORMANCE.md)
- [Validation and R parity](VALIDATION.md)
- [Method comparison](METHODS.md)
- [Changelog](CHANGELOG.md)
- [Contributing and development](CONTRIBUTING.md)

## Citation

If you use DecontX in your research, please cite:

> Yang, S., Corbett, S.E., Koga, Y. et al. Decontamination of ambient RNA in single-cell RNA-seq with DecontX. Genome Biol 21, 57 (2020). https://doi.org/10.1186/s13059-020-1950-6

## License

MIT License
