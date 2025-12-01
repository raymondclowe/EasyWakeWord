# Publishing to TestPyPI

This guide shows how to build and publish EasyWakeWord to TestPyPI using `uv`.

## Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create TestPyPI account**: https://test.pypi.org/account/register/

3. **Generate API token**:
   - Go to https://test.pypi.org/manage/account/token/
   - Create a token with scope for this project
   - Save it securely

## Build the Package

```bash
# Create virtual environment (if not already done)
uv venv

# Install build dependencies
uv pip install build twine

# Build the distribution
uv run python -m build
```

This creates:
- `dist/easywakeword-0.1.0.tar.gz` (source distribution)
- `dist/easywakeword-0.1.0-py3-none-any.whl` (wheel distribution)

## Upload to TestPyPI

```bash
# Upload using twine
uv run twine upload --repository testpypi dist/*
```

When prompted:
- **Username**: `__token__`
- **Password**: Your TestPyPI API token (including `pypi-` prefix)

## Test Installation from TestPyPI

```bash
# Create a new test environment
uv venv test-env
source test-env/bin/activate  # On Windows: test-env\Scripts\activate

# Install from TestPyPI
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ easywakeword

# Test the installation
easywakeword-demo
```

Note: We use `--extra-index-url https://pypi.org/simple/` because dependencies (numpy, librosa, etc.) are on the main PyPI.

## Publish to Production PyPI

Once tested on TestPyPI:

```bash
# Upload to production PyPI
uv run twine upload dist/*
```

Use your production PyPI token when prompted.

## Version Updates

To publish a new version:

1. Update version in `pyproject.toml`
2. Commit changes: `git commit -am "Bump version to X.Y.Z"`
3. Create tag: `git tag vX.Y.Z`
4. Push: `git push && git push --tags`
5. Rebuild: `rm -rf dist/ && uv run python -m build`
6. Upload: `uv run twine upload dist/*`

## Quick Commands Reference

```bash
# Setup
uv venv
uv pip install -e .

# Build
uv pip install build twine
uv run python -m build

# Upload to TestPyPI
uv run twine upload --repository testpypi dist/*

# Upload to PyPI
uv run twine upload dist/*

# Clean build artifacts
rm -rf dist/ build/ *.egg-info
```

## .pypirc Configuration (Optional)

Create `~/.pypirc` to avoid entering credentials each time:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YourProductionTokenHere

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YourTestTokenHere
```

**Security Note**: Keep your tokens secure and never commit `.pypirc` to version control!
