from typing import TYPE_CHECKING

from .util.threadpool import configure_windows_threadpools

configure_windows_threadpools()

if TYPE_CHECKING:
    from .file.data import CyESIData

__all__ = ["CyESIData"]
__version__ = "0.3.0"


def __getattr__(name):
    """Load the pyOpenMS data layer only when the public class is requested."""
    if name == "CyESIData":
        from .file.data import CyESIData

        return CyESIData
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
