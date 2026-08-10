"""Single-cell mass spectrometry processing and analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("scMM")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
