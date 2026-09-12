"""Initial test to verify package discovery and imports."""

import enterprise_orchestrator


def test_package_version():
    """Verify the package version is defined."""
    assert enterprise_orchestrator.__version__ == "0.1.0"
