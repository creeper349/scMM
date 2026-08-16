"""Panel-based guided web interface for scMM."""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    """Create one web application session, importing optional UI dependencies lazily."""
    from .app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]
