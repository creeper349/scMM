import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pickle
from typing import Optional, Tuple
from matplotlib.ticker import AutoMinorLocator

def eic(ax: plt.Axes, data: pd.DataFrame, mz: float, ppm_tol: float = 5.0, time: Optional[pd.Series] = None) -> plt.Axes:
   
    mz_axis = data.columns.astype(np.float64)
    sort_index = np.where(np.abs((mz_axis - mz) / mz * 1e6) <= ppm_tol)[0]
    if len(sort_index) == 0:
        raise ValueError("No m/z found within the specified tolerance.")
    eic_values = data.iloc[:, sort_index].sum(axis=1)
    if time is not None:
        ax.plot(time.values, eic_values, label=f"EIC at m/z= {mz} ± {ppm_tol}ppm", linewidth=1.5)
        ax.set_xlabel("Time")
    else:
        ax.plot(data.index, eic_values, label=f"EIC at m/z={mz} ± {ppm_tol}ppm", linewidth=1.5)
        ax.set_xlabel("Scan Index")
    ax.set_ylabel("Intensity")
    ax.legend()
    return ax, (time.values if time is not None else data.index, eic_values)


def tic(
    ax: plt.Axes,
    data: pd.DataFrame,
    time: Optional[pd.Series] = None,
) -> plt.Axes:
    """Plot a total-ion chromatogram from an aligned frame matrix."""
    values = data.sum(axis=1)
    x = time.to_numpy() if time is not None else np.asarray(data.index)
    ax.plot(x, values.to_numpy(), linewidth=1.2)
    ax.set_xlabel("Time" if time is not None else "Scan index")
    ax.set_ylabel("Total ion intensity")
    ax.set_title("TIC")
    return ax


def plot_cell_detection(
    data: pd.DataFrame,
    reference_mz,
    cell_mask,
    *,
    peak_frames=None,
    time=None,
    ppm_tol=5.0,
    output_file=None,
    ax=None,
):
    """Plot reference EICs and mark frames detected as cells."""
    references = np.atleast_1d(reference_mz).astype(float)
    cell_mask = np.asarray(cell_mask, dtype=bool)
    if len(cell_mask) != len(data):
        raise ValueError("cell_mask length must match the frame matrix")
    x = np.asarray(time) if time is not None else np.asarray(data.index)
    if len(x) != len(data):
        raise ValueError("time length must match the frame matrix")

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure
    mz_axis = data.columns.astype(float).to_numpy()
    for value in references:
        indices = np.flatnonzero(
            np.abs((mz_axis - value) / value * 1e6) <= ppm_tol
        )
        if len(indices) == 0:
            indices = np.asarray([int(np.abs(mz_axis - value).argmin())])
        signal = data.iloc[:, indices].sum(axis=1).to_numpy()
        line = ax.plot(x, signal, linewidth=1.2, label=f"m/z {value:g}")[0]
        ax.scatter(
            x[cell_mask],
            signal[cell_mask],
            s=8,
            color=line.get_color(),
            alpha=0.65,
        )
        if peak_frames is not None:
            peaks = np.asarray(peak_frames, dtype=int)
            ax.scatter(
                x[peaks],
                signal[peaks],
                s=24,
                marker="x",
                color="black",
            )
    ax.set_xlabel("Time" if time is not None else "Scan index")
    ax.set_ylabel("Intensity")
    ax.set_title("Reference EIC cell detection")
    ax.legend()
    from .ticks import tidy_numeric_ticks
    tidy_numeric_ticks(fig)
    fig.tight_layout()
    if output_file is not None:
        fig.savefig(output_file, bbox_inches="tight")
    return fig, ax

def _plot_cells(ax: plt.Axes, cell_mask:np.ndarray, data: Tuple[np.ndarray, np.ndarray]) -> plt.Axes:
    t, eic_values = data
    t, eic_values = t[cell_mask], eic_values[cell_mask]
    ax.scatter(t, eic_values, color='red', s=3, label='Cells')
    ax.legend()
    return ax

def plot_ms(ax: plt.Axes,
            data: pd.DataFrame,
            frame_range: Optional[Tuple[int, int]] = None) -> plt.Axes:
    if frame_range is None:
        selected = data
        title = "MS Spectrum for all frames"
    else:
        start, stop = frame_range
        selected = data.iloc[start:stop]
        title = f"MS Spectrum for frames {start} to {stop - 1}"
    mz_inten = selected.values.sum(axis=0)
    for mz, inten in zip(data.columns.astype(np.float64), mz_inten):
        ax.vlines(mz, 0, inten, colors='black')
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    return ax

def plot_hook(stage, data):
    if stage == "find_cells":
        os.makedirs(".tmp", exist_ok=True)
        fig, _ = plot_cell_detection(
            data["signal"],
            data["reference_mz"],
            data["cell_mask"],
            peak_frames=data["cell_idx"],
            output_file=f".tmp/{stage}.svg",
        )
        plt.close(fig)
            
def save_hook(stage, data):
    if not os.path.exists(".tmp"):
        os.mkdir(".tmp")
    save_dict = {
        "stage": stage,
        "data": dict(data)
    }
    with open(f".tmp/{stage}.pkl", "wb") as fp:
        pickle.dump(save_dict, fp)

def plot_spectrum(
        spec,
        top_n_labels: int = 0,
        mz_range: tuple[float, float] | None = None,
        intensity_range: tuple[float, float] | None = None,
        normalize: bool = False,
        exclusion_window: float = 10.0,
        label_fmt: str = "{:.4f}",
        title: str | None = None,
        figsize: tuple[float, float] = (10, 4),
        linewidth: float = 1.0,
        save_path: str | None = None,
        ax=None,
        **kwargs
    ):
    mz, inten = spec.get_peaks()
    mz = np.asarray(mz, dtype=float)
    inten = np.asarray(inten, dtype=float)

    order = np.argsort(mz)
    mz = mz[order]
    inten = inten[order]
    if mz_range is not None:
        mask = (mz >= mz_range[0]) & (mz <= mz_range[1])
        mz = mz[mask]
        inten = inten[mask]

    if normalize:
        max_i = np.max(inten)
        if max_i > 0:
            inten = inten / max_i

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        fig = ax.figure

    ax.plot(mz, inten, linewidth=linewidth, color = kwargs.get("color", "black"))
    ax.set_xlim(mz_range if mz_range is not None else (np.min(mz), np.max(mz)))
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity" if normalize else "Intensity")
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))

    if title is not None:
        ax.set_title(title)

    if intensity_range is not None:
        ax.set_ylim(*intensity_range)
    else:
        ax.set_ylim(0, np.max(inten) * 1.05)

    if top_n_labels and top_n_labels > 0:
        idx_sorted = np.argsort(inten)[::-1]
        selected = []

        for idx in idx_sorted:
            x = mz[idx]

            too_close = any(abs(x - mz[j]) < exclusion_window for j in selected)
            if too_close:
                continue

            selected.append(idx)
            if len(selected) >= top_n_labels:
                break

        selected = sorted(selected, key=lambda i: mz[i])

        y_max = np.max(inten)
        offset = 0.03 * y_max if y_max > 0 else 0.03

        for idx in selected:
            x = mz[idx]
            y = inten[idx]

            ax.annotate(
                label_fmt.format(x),
                xy=(x, y),
                xytext=(x, y + offset),
                textcoords="data",
                ha="center",
                fontsize=9,
                arrowprops=dict(arrowstyle="-", lw=0.8),
            )

    ax.margins(x=0.01)
    from .ticks import tidy_numeric_ticks
    tidy_numeric_ticks(fig)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if created_fig:
        plt.show()

    return fig, ax
