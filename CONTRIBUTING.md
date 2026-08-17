# Contributing

For the full validation workflow used before shipping changes, see [AGENTS.md](AGENTS.md).

## Local install

```bash
git clone https://github.com/NiRuff/decontx-python.git
cd decontx-python
pip install .
```

This installs a snapshot into site-packages. Re-run it after any source change.

For development, `pip install -e .` makes Python import the local source directly, so
edits take effect without reinstalling.

**Verify which copy you are importing**, especially if the package has ever been
installed from another path:

```bash
python -c "import decontx; print(decontx.__file__)"
```

A stale editable install (`.pth` file) pointing at an old checkout will shadow your
working copy from every directory except the repo root, and because the Numba kernels
have no bounds checking, running an outdated copy can surface as heap corruption rather
than a clean error. If the path is not what you expect, run `pip uninstall
decontx-python` and reinstall.

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

If the package is installed in editable mode, `PYTHONPATH=.` is not needed.
