# Quick Start Guide - EasyWakeWord Development

## Setup with uv

```bash
# Clone the repository
git clone https://github.com/raymondclowe/EasyWakeWord.git
cd EasyWakeWord

# Create virtual environment with uv
uv venv

# Activate (optional, uv commands work without activation)
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# Install dev dependencies
uv pip install -e ".[dev]"
```

## Run Demo

```bash
# First, ensure mini_transcriber is running
# See: https://github.com/raymondclowe/mini_transcriber

# Run the demo
easywakeword-demo

# Or run directly
uv run python -m easywakeword.demo
```

## Build and Test

```bash
# Install build tools
uv pip install build twine

# Build the package
uv run python -m build

# Check the distribution
uv run twine check dist/*
```

## Testing Locally

```bash
# In a new directory/environment
uv venv test-env
cd test-env

# Install from local wheel
uv pip install ../EasyWakeWord/dist/easywakeword-0.1.0-py3-none-any.whl

# Test import
uv run python -c "import easywakeword; print(easywakeword.__version__)"

# Test demo
easywakeword-demo
```

## Upload to TestPyPI

See PUBLISHING.md for detailed instructions.

```bash
# Quick upload
uv run twine upload --repository testpypi dist/*
```

## Git Workflow

```bash
# Make changes
git add .
git commit -m "Description of changes"

# Push to repository
git push origin newversion
```

## Common Commands

```bash
# List installed packages
uv pip list

# Check package info
uv pip show easywakeword

# Uninstall
uv pip uninstall easywakeword

# Clean build artifacts
rm -rf dist/ build/ *.egg-info easywakeword.egg-info
# Windows: Remove-Item -Recurse -Force dist,build,*.egg-info

# Reinstall after changes
uv pip install -e . --force-reinstall --no-deps
```

## Directory Structure

```
EasyWakeWord/
├── easywakeword/           # Main package
│   ├── __init__.py
│   ├── recogniser.py       # Main recognizer class
│   ├── matching.py         # MFCC matching
│   ├── silence.py          # Sound buffer
│   ├── transcription.py    # STT integration
│   ├── utils.py
│   ├── demo.py             # Demo script
│   └── examples/           # Example WAV files
├── demo.py                 # Root demo (legacy)
├── pyproject.toml          # Package configuration
├── README.md               # Documentation
├── LICENSE                 # MIT License
├── MANIFEST.in             # Include data files
├── PUBLISHING.md           # Publishing guide
└── .private/               # Private notes (not in distribution)
```
