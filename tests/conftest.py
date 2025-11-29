"""
Pytest configuration for EasyWakeWord tests.

This module handles environment detection and provides fixtures for
testing in various environments (Windows dev, GitHub Copilot CI, etc.)
without requiring real audio hardware.

Environment Compatibility:
- Windows dev: Usually has audio hardware available
- GitHub Copilot/Actions CI: May not have PortAudio installed
- Linux CI: May need libportaudio2 package installed

Usage:
- Tests marked with @pytest.mark.requires_portaudio are skipped if PortAudio unavailable
- Tests marked with @pytest.mark.requires_audio_device are skipped without real hardware
- Use fixtures `portaudio_available` and `in_ci_environment` for conditional logic
"""

import os
import sys
import platform

import pytest


# Detect if PortAudio/sounddevice is available
PORTAUDIO_AVAILABLE = False
try:
    import sounddevice as sd
    PORTAUDIO_AVAILABLE = True
except OSError:
    # PortAudio library not found - common in CI environments
    pass

# Detect CI environment
IN_CI_ENVIRONMENT = any([
    os.environ.get('GITHUB_ACTIONS') == 'true',
    os.environ.get('CI') == 'true',
    os.environ.get('CODESPACES') == 'true',
])

# Detect platform
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'


def pytest_configure(config):
    """Register custom markers for audio-dependent tests."""
    config.addinivalue_line(
        "markers", "requires_portaudio: mark test as requiring PortAudio library"
    )
    config.addinivalue_line(
        "markers", "requires_audio_device: mark test as requiring real audio hardware"
    )


def pytest_collection_modifyitems(config, items):
    """
    Skip tests that require unavailable audio hardware.
    
    This allows tests to be collected and run even when PortAudio
    is not available, skipping only the tests that actually need it.
    """
    if not PORTAUDIO_AVAILABLE:
        skip_portaudio = pytest.mark.skip(reason="PortAudio library not available")
        for item in items:
            if "requires_portaudio" in item.keywords:
                item.add_marker(skip_portaudio)
            if "requires_audio_device" in item.keywords:
                item.add_marker(skip_portaudio)


@pytest.fixture
def portaudio_available():
    """Fixture to check if PortAudio is available."""
    return PORTAUDIO_AVAILABLE


@pytest.fixture
def in_ci_environment():
    """Fixture to check if running in CI environment."""
    return IN_CI_ENVIRONMENT


@pytest.fixture
def is_windows():
    """Fixture to check if running on Windows."""
    return IS_WINDOWS


@pytest.fixture
def is_linux():
    """Fixture to check if running on Linux."""
    return IS_LINUX


@pytest.fixture
def is_macos():
    """Fixture to check if running on macOS."""
    return IS_MACOS
