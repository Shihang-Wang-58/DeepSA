# DeepSA PyPI Packaging and Publishing Guide

This document provides detailed steps for packaging and publishing DeepSA to PyPI.

## Preparation

1. Ensure necessary packaging tools are installed:

```bash
pip install build twine
```

2. Ensure model files are correctly placed:

Before publishing, you need to copy the DeepSA model files to the package directory to include them in the distribution package:

```bash
# Create model directory
mkdir -p deepsa/model

# Copy model files
cp -r DeepSA_model/* deepsa/model/
```

## Build Package

Use Python's build module to build the package:

```bash
python -m build
```

This will create source distribution package (.tar.gz) and wheel distribution package (.whl) in the `dist/` directory.

## Test Package

Before uploading to PyPI, it's recommended to test the installation locally:

```bash
# Create virtual environment
python -m venv test_env
source test_env/bin/activate  # Linux/Mac
# Or on Windows:
# test_env\Scripts\activate

# Install locally built package
pip install dist/deepsa-0.1.0-py3-none-any.whl

# Test import and basic functionality
python -c "from deepsa import predict_sa; print(predict_sa('CCO'))"
```

## Upload to TestPyPI (Optional)

Before official release, you can upload to TestPyPI for testing:

```bash
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

Then you can install from TestPyPI for testing:

```bash
pip install --index-url https://test.pypi.org/simple/ deepsa
```

## Upload to PyPI

After confirming everything is normal, upload to the official PyPI:

```bash
twine upload dist/*
```

You will be asked to enter your PyPI account and password. If you don't have a PyPI account, you need to register first at [PyPI website](https://pypi.org).

## Verify Installation

After publishing, verify the installation via pip:

```bash
pip install deepsa
```

## Update Version

When you need to update the package, modify the version number in the following files:

1. `version` in `setup.py`
2. `__version__` in `deepsa/__init__.py`
3. `version` in `pyproject.toml`

Then repeat the above build and upload steps.

## Notes

1. Model files are large, including them in the distribution package may result in a large package size. Consider whether you need to separate the model files and distribute them through other means.

2. If the model files are too large, consider the following alternatives:
   - Automatically download model files on first use
   - Provide a separate command for model download
   - Use external storage services to host model files

3. Ensure all dependencies are correctly specified in `setup.py` and `pyproject.toml`.

4. Check the encoding of all files before publishing to ensure there are no problems caused by non-ASCII characters.