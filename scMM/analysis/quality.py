from pathlib import Path
import os

import matplotlib
if os.environ.get("SCMM_GUI_ACTIVE") != "1":
    matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def quality_metrics(data):
    """Return per-cell and per-file quality-control tables."""
    if hasattr(data, "ensure_cell_quality_metrics"):
        data.ensure_cell_quality_metrics()
    X = data.X.toarray() if hasattr(data.X, "toarray") else np.asarray(data.X)
    cell = data.obs.copy()
    cell["total_intensity"] = cell.get(
        "raw_total_intensity", pd.Series(np.nansum(X, axis=1), index=cell.index)
    )
    if "detected_features" not in cell:
        cell["detected_features"] = np.count_nonzero(X > 0, axis=1)
    if hasattr(data, "cell_keep_mask"):
        cell["cell_active"] = data.cell_keep_mask().astype(np.uint8)
    if "source_file" not in cell:
        cell["source_file"] = "unknown"

    numeric = [
        "total_intensity",
        "detected_features",
        *[name for name in cell.columns if name.startswith("reference_intensity_")],
    ]
    per_file = cell.groupby("source_file", observed=True)[numeric].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    return {"cell": cell, "file": per_file}


def render_quality_report(data, output_file):
    """Create a compact PDF report of core CyESI quality distributions."""
    from scMM.plot.ticks import tidy_numeric_ticks
    metrics = quality_metrics(data)
    cell = metrics["cell"]
    target = Path(output_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    reference_columns = [
        name for name in cell.columns if name.startswith("reference_intensity_")
    ]

    with PdfPages(target) as pdf:
        fig = Figure(figsize=(11, 4))
        axes = fig.subplots(1, 2)
        for source, group in cell.groupby("source_file", observed=True):
            axes[0].hist(group["total_intensity"], bins=40, alpha=0.5, label=source)
            axes[1].hist(group["detected_features"], bins=40, alpha=0.5, label=source)
        axes[0].set_title("Total intensity per cell")
        axes[1].set_title("Detected features per cell")
        for ax in axes:
            ax.set_ylabel("Cells")
            ax.legend(fontsize=8)
        tidy_numeric_ticks(fig)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.clear()

        for column in reference_columns:
            fig = Figure(figsize=(7, 4))
            ax = fig.subplots()
            values = [
                group[column].dropna().to_numpy()
                for _, group in cell.groupby("source_file", observed=True)
            ]
            labels = [
                source for source, _ in cell.groupby("source_file", observed=True)
            ]
            ax.boxplot(values, tick_labels=labels, showfliers=False)
            ax.set_title(column.replace("_", " "))
            ax.set_ylabel("Intensity")
            tidy_numeric_ticks(fig)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.clear()
    return target
