import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Optional, Tuple

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

def _plot_cells(ax: plt.Axes, cell_mask:np.ndarray, data: Tuple[np.ndarray, np.ndarray]) -> plt.Axes:
    t, eic_values = data
    t, eic_values = t[cell_mask], eic_values[cell_mask]
    ax.scatter(t, eic_values, color='red', s=3, label='Cells')
    ax.legend()
    return ax

def plot_ms(ax: plt.Axes,
            data: pd.DataFrame,
            frame_range: Optional[Tuple[int, int]] = None) -> plt.Axes:
    frame_range = data.index if frame_range is None else range(frame_range[0], frame_range[1])
    mz_inten = data.loc[frame_range].values.sum(axis=0)
    for mz, inten in zip(data.columns.astype(np.float64), mz_inten):
        ax.vlines(mz, 0, inten, colors='black')
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(f"MS Spectrum for frames {frame_range.start} to {frame_range.stop - 1}")
    return ax

def plot_hook(stage, data):
    if (stage == "peak_detection") or (stage == "denoised_signal"):
        fig, ax = plt.subplots(figsize=(10, 6))
        eic_ax, eic_data = eic(ax, data["data"], data["ref_mz"], ppm_tol=5.0)
        eic_ax = _plot_cells(eic_ax, data["cell_mask"], eic_data)
        eic_ax.set_title("EIC with Detected Cells")
        plt.savefig(f"debug_{stage}.svg")
        plt.close(fig)
    elif stage == "cell_signal":
        C = data["C"]
        ref_index = (np.abs(data["data"].columns.astype(np.float64) - data["ref_mz"])).argmin()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data["data"].index, C[:, ref_index], label="Cell Signal at ref m/z")
        ax.set_xlabel("Scan Index")
        ax.set_ylabel("Intensity")
        plt.savefig(f"debug_{stage}.svg")
        plt.close(fig)
    elif stage == "r1":
        r1 = data["r1"]
        if r1 is not None:
            a, b = r1
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax[0].plot(a)
            ax[0].set_xlabel("Index")
            ax[0].set_ylabel("Time loadings")
            ax[1].plot(b)
            ax[1].set_xlabel("Index")
            ax[1].set_ylabel("m/z loadings")
            plt.savefig(f"debug_{stage}.svg")
            plt.close(fig)