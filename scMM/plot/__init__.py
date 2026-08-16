"""Plotting configuration and high-level plotting helpers.

Importing this module intentionally leaves the caller's Matplotlib settings
unchanged. Use :func:`configure_plotting` to opt into the scMM defaults.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager


def configure_plotting(font_path: str | Path | None = None) -> None:
    """Apply scMM's plotting defaults, optionally using a custom font file."""
    if font_path is not None:
        path = Path(font_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(path)
        font_manager.fontManager.addfont(path)
        family_name = font_manager.FontProperties(fname=path).get_name()
        matplotlib.rcParams.update(
            {
                "font.family": family_name,
                "mathtext.fontset": "custom",
                "mathtext.rm": family_name,
            }
        )

    matplotlib.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


__all__ = ["configure_plotting"]
