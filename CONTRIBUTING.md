# Contributing

For the full validation workflow, see [AGENTS.md](AGENTS.md).

## Install locally

```bash
git clone https://github.com/NiRuff/decontx-python.git
cd decontx-python
pip install .
```

This command installs a snapshot into `site-packages`. Run it again after each source change.

For development, use `pip install -e .`. This makes Python import the local source directly. Edits take effect without reinstallation.

**Verify which copy you import**, especially if the package was installed from another path:

```bash
python -c "import decontx; print(decontx.__file__)"
```

A stale editable install (`.pth` file) can point at an old checkout. It will shadow your working copy in every directory except the repo root. The Numba kernels have no bounds checking. Running an outdated copy can cause heap corruption instead of a clean error. If the path is not what you expect, run `pip uninstall decontx-python` and reinstall.

## Tests and benchmarks

```bash
# Tests
python -m pytest tests/test_core.py tests/test_regression.py -v

# Standalone benchmark
PYTHONPATH=. python tests/benchmark.py

# Head-to-head comparison with the jjia1 implementation
# (requires a local checkout of jjia1/decontx-python)
DECONTX_JJIA1_PATH=/path/to/jjia1/decontx-python \
PYTHONPATH=. python tests/benchmark_compare.py
```

If you installed the package in editable mode, you do not need `PYTHONPATH=.`
