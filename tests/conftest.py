import pytest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Configure pytest
def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "asyncio: mark test as async")


@pytest.fixture(autouse=True)
def mock_sleep(monkeypatch):
    """Mock time.sleep to speed up tests."""

    def mock_sleep(*args, **kwargs):
        pass

    monkeypatch.setattr("time.sleep", mock_sleep)
