import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


def eic(
    ax: plt.Axes, data: pd.DataFrame, mz: float, ppm_tol: float = 5.0, time: pd.Series | None = None
) -> tuple[plt.Axes, tuple[np.ndarray | pd.Index, pd.Series]]:
    if not np.isfinite(mz) or mz <= 0:
        raise ValueError("mz must be a positive finite number")
    if ppm_tol < 0:
        raise ValueError("ppm_tol must be non-negative")
    if time is not None and len(time) != len(data):
        raise ValueError("time must contain one value per data row")
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


def _plot_cells(
    ax: plt.Axes, cell_mask: np.ndarray, data: tuple[np.ndarray, np.ndarray]
) -> plt.Axes:
    t, eic_values = data
    t, eic_values = t[cell_mask], eic_values[cell_mask]
    ax.scatter(t, eic_values, color="red", s=3, label="Cells")
    ax.legend()
    return ax


def plot_ms(
    ax: plt.Axes, data: pd.DataFrame, frame_range: tuple[int, int] | None = None
) -> plt.Axes:
    if data.empty:
        raise ValueError("data must not be empty")
    if frame_range is None:
        start, stop = 0, len(data)
    else:
        start, stop = frame_range
        start = max(0, start)
        stop = min(len(data), stop)
        if start >= stop:
            raise ValueError("frame_range must select at least one row")
    mz_inten = data.iloc[start:stop].values.sum(axis=0)
    for mz, inten in zip(data.columns.astype(np.float64), mz_inten, strict=True):
        ax.vlines(mz, 0, inten, colors="black")
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    ax.set_title(f"MS Spectrum for rows {start} to {stop - 1}")
    return ax


def plot_hook(stage, data):
    if stage == "find_cells":
        cell_idx = np.asarray(data["cell_idx"], dtype=int)
        if cell_idx.size == 0:
            return
        output_dir = Path(".tmp")
        output_dir.mkdir(exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        signal = data["signal"].sum(axis=1)
        ax.plot(data["signal"].index, signal, color="black", linewidth=1, label="Signal")
        ax.scatter(
            data["signal"].index[cell_idx], signal.iloc[cell_idx], color="red", s=8, label="Cells"
        )
        ax.set_xlabel("Scan Index")
        ax.set_ylabel("Total Intensity")
        ax.legend()
        fig.savefig(output_dir / f"{stage}.svg", bbox_inches="tight")
        plt.close(fig)


def save_hook(stage, data):
    output_dir = Path(".tmp")
    output_dir.mkdir(exist_ok=True)
    save_dict = {"stage": stage, "data": dict(data)}
    with (output_dir / f"{stage}.pkl").open("wb") as fp:
        pickle.dump(save_dict, fp)


def _prepare_spectrum_peaks(
    spec,
    mz_range: tuple[float, float] | None,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    mz, intensity = spec.get_peaks()
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    finite = np.isfinite(mz) & np.isfinite(intensity)
    mz = mz[finite]
    intensity = intensity[finite]

    order = np.argsort(mz)
    mz = mz[order]
    intensity = intensity[order]
    if mz_range is not None:
        if mz_range[0] >= mz_range[1]:
            raise ValueError("mz_range must be increasing")
        mask = (mz >= mz_range[0]) & (mz <= mz_range[1])
        mz = mz[mask]
        intensity = intensity[mask]
    if mz.size == 0:
        raise ValueError("Spectrum has no peaks in the requested range")

    if normalize:
        max_intensity = np.max(intensity)
        if max_intensity > 0:
            intensity = intensity / max_intensity
    return mz, intensity


def _spectrum_x_limits(
    mz: np.ndarray,
    mz_range: tuple[float, float] | None,
) -> tuple[float, float]:
    if mz_range is not None:
        return mz_range
    x_min, x_max = float(np.min(mz)), float(np.max(mz))
    padding = max(abs(x_min) * 1e-6, 1e-6) if x_min == x_max else 0.0
    return x_min - padding, x_max + padding


def _configure_spectrum_axes(
    ax: plt.Axes,
    mz: np.ndarray,
    intensity: np.ndarray,
    mz_range: tuple[float, float] | None,
    intensity_range: tuple[float, float] | None,
    normalize: bool,
    title: str | None,
) -> None:
    ax.set_xlim(_spectrum_x_limits(mz, mz_range))
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative Intensity" if normalize else "Intensity")
    ax.xaxis.set_minor_locator(AutoMinorLocator(10))
    if title is not None:
        ax.set_title(title)

    if intensity_range is not None:
        ax.set_ylim(*intensity_range)
    else:
        y_max = float(np.max(intensity))
        ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)


def _select_peak_labels(
    mz: np.ndarray,
    intensity: np.ndarray,
    top_n_labels: int,
    exclusion_window: float,
) -> list[int]:
    selected = []
    for index in np.argsort(intensity)[::-1]:
        if any(abs(mz[index] - mz[other]) < exclusion_window for other in selected):
            continue
        selected.append(index)
        if len(selected) >= top_n_labels:
            break
    return sorted(selected, key=lambda index: mz[index])


def _annotate_spectrum_peaks(
    ax: plt.Axes,
    mz: np.ndarray,
    intensity: np.ndarray,
    selected: list[int],
    label_fmt: str,
) -> None:
    y_max = np.max(intensity)
    offset = 0.03 * y_max if y_max > 0 else 0.03
    for index in selected:
        x = mz[index]
        y = intensity[index]
        ax.annotate(
            label_fmt.format(x),
            xy=(x, y),
            xytext=(x, y + offset),
            textcoords="data",
            ha="center",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "lw": 0.8},
        )


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
    save_path: str | Path | None = None,
    ax=None,
    **kwargs,
):
    if top_n_labels < 0:
        raise ValueError("top_n_labels must be non-negative")
    if exclusion_window < 0:
        raise ValueError("exclusion_window must be non-negative")
    mz, intensity = _prepare_spectrum_peaks(spec, mz_range, normalize)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(mz, intensity, linewidth=linewidth, color=kwargs.get("color", "black"))
    _configure_spectrum_axes(ax, mz, intensity, mz_range, intensity_range, normalize, title)

    if top_n_labels:
        selected = _select_peak_labels(mz, intensity, top_n_labels, exclusion_window)
        _annotate_spectrum_peaks(ax, mz, intensity, selected, label_fmt)

    ax.margins(x=0.01)

    if save_path is not None:
        output = Path(save_path).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")

    return fig, ax
