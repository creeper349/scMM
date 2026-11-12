"""Cross-platform plotting defaults for SCMM."""

import matplotlib


matplotlib.rcParams.update(
    {
        "svg.fonttype": "none",
        "font.family": ["Arial", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "dejavusans",
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)
