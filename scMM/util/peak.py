"""Spectrum denoising and cell-event peak extraction."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyopenms as oms
from joblib import Parallel, delayed
from scipy.ndimage import label, median_filter


@dataclass(frozen=True)
class _SpectrumFilterConfig:
    baseline_window: int
    noise_window: int
    baseline_quantile: float
    snr_threshold: float
    keep_negative: bool
    baseline_stride: int


@dataclass(frozen=True)
class _CellWindowInputs:
    values: np.ndarray
    baseline: np.ndarray
    reference_signal: np.ndarray
    labeled_mask: np.ndarray


@dataclass(frozen=True)
class _CellPeak:
    intensities: np.ndarray
    peak_frame: int
    window_range: tuple[int, int]


def filter_spectrum(
    spec: oms.MSSpectrum,
    baseline_window: int = 101,
    noise_window: int = 101,
    baseline_quantile: float = 0.1,
    snr_threshold: float = 3.0,
    keep_negative: bool = False,
    return_snr: bool = False,
    baseline_stride: int = 10,
):
    """Remove a local quantile baseline and suppress low-SNR intensities."""
    config = _spectrum_filter_config(
        baseline_window,
        noise_window,
        baseline_quantile,
        snr_threshold,
        keep_negative,
        baseline_stride,
    )
    mz, intensity = (np.asarray(array, dtype=np.float64) for array in spec.get_peaks())
    if intensity.size == 0:
        output = _new_spectrum(spec, mz, intensity, copy_optional_metadata=False)
        return (output, np.array([], dtype=np.float64)) if return_snr else output

    baseline = _local_quantile_baseline(
        intensity,
        config.baseline_window,
        config.baseline_quantile,
        config.baseline_stride,
    )
    signal = intensity - baseline
    noise = _local_mad_noise(signal, config.noise_window)
    if not config.keep_negative:
        signal[signal < 0] = 0.0
    snr = signal / noise
    filtered = signal.copy()
    filtered[snr < config.snr_threshold] = 0.0
    output = _new_spectrum(spec, mz, filtered, copy_optional_metadata=True)
    return (output, snr) if return_snr else output


def _spectrum_filter_config(
    baseline_window,
    noise_window,
    baseline_quantile,
    snr_threshold,
    keep_negative,
    baseline_stride,
) -> _SpectrumFilterConfig:
    if not 0 <= baseline_quantile <= 1:
        raise ValueError("baseline_quantile must be between 0 and 1")
    if snr_threshold < 0:
        raise ValueError("snr_threshold must be non-negative")
    return _SpectrumFilterConfig(
        baseline_window=_odd_window(baseline_window),
        noise_window=_odd_window(noise_window),
        baseline_quantile=baseline_quantile,
        snr_threshold=snr_threshold,
        keep_negative=keep_negative,
        baseline_stride=max(1, int(baseline_stride)),
    )


def _odd_window(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _local_quantile_baseline(
    intensity: np.ndarray,
    window: int,
    quantile: float,
    stride: int,
) -> np.ndarray:
    if stride == 1:
        return _quantiles_at_indices(intensity, np.arange(len(intensity)), window, quantile)
    anchors = np.arange(0, len(intensity), stride, dtype=np.int64)
    if anchors[-1] != len(intensity) - 1:
        anchors = np.append(anchors, len(intensity) - 1)
    anchor_values = _quantiles_at_indices(intensity, anchors, window, quantile)
    return np.interp(np.arange(len(intensity), dtype=float), anchors.astype(float), anchor_values)


def _quantiles_at_indices(
    values: np.ndarray,
    indices: np.ndarray,
    window: int,
    quantile: float,
) -> np.ndarray:
    half_window = window // 2
    result = np.empty(len(indices), dtype=np.float64)
    for output_index, value_index in enumerate(indices):
        left = max(0, value_index - half_window)
        right = min(len(values), value_index + half_window + 1)
        result[output_index] = np.quantile(values[left:right], quantile)
    return result


def _local_mad_noise(residual: np.ndarray, window: int) -> np.ndarray:
    half_window = window // 2
    noise = np.empty(len(residual), dtype=np.float64)
    for index in range(len(residual)):
        local = residual[max(0, index - half_window) : min(len(residual), index + half_window + 1)]
        median = np.median(local)
        mad = np.median(np.abs(local - median))
        noise[index] = max(1.4826 * mad, 1e-12)
    return noise


def _new_spectrum(
    source: oms.MSSpectrum,
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    copy_optional_metadata: bool,
) -> oms.MSSpectrum:
    output = oms.MSSpectrum()
    output.set_peaks((mz, intensity))
    output.setRT(source.getRT())
    output.setMSLevel(source.getMSLevel())
    if copy_optional_metadata:
        with suppress(Exception):
            output.setName(source.getName())
        with suppress(Exception):
            output.setDriftTime(source.getDriftTime())
    return output


def _filter(data: np.ndarray, size: int = 10, filter=median_filter, **filter_kwargs):
    """Apply a 1D frame-axis filter independently to every feature."""
    return filter(data, size=(size, 1), **filter_kwargs)


def find_cell_peaks(
    data: pd.DataFrame,
    ref_mz: float,
    baseline_filter=median_filter,
    baseline_filter_size: int = 15,
    cell_snr: float = 5.0,
    peak_snr: float = 3.0,
    dtype=np.float64,
    baseline_stat="median",
    max_zero_frac: float = 0.9,
    n_jobs: int = -1,
    **kwargs,
):
    """Identify cell-event windows and reduce each to one feature vector."""
    _validate_cell_peak_options(
        data,
        ref_mz,
        baseline_filter_size,
        cell_snr,
        peak_snr,
        max_zero_frac,
    )
    inputs, common, n_cells = _detect_cell_windows(
        data,
        ref_mz,
        dtype,
        baseline_filter,
        baseline_filter_size,
        cell_snr,
        kwargs,
    )
    if n_cells == 0:
        return _empty_cell_result(data, common)
    peaks = Parallel(n_jobs=n_jobs)(
        delayed(_process_cell_window)(label_id, inputs, baseline_stat, peak_snr)
        for label_id in range(1, n_cells + 1)
    )
    peaks = [peak for peak in peaks if peak is not None]
    cell_frame, zero_fraction, kept_columns = _assemble_cell_frame(
        data.columns,
        peaks,
        max_zero_frac,
    )
    return {
        **common,
        "cell_df": cell_frame,
        "peak_frames": np.asarray([peak.peak_frame for peak in peaks], dtype=int),
        "window_ranges": [peak.window_range for peak in peaks],
        "zero_frac": zero_fraction,
        "kept_columns": kept_columns,
    }


def _detect_cell_windows(
    data,
    ref_mz,
    dtype,
    baseline_filter,
    baseline_filter_size,
    cell_snr,
    filter_kwargs,
):
    values = data.values.astype(dtype)
    mz_values = data.columns.astype(dtype)
    baseline = _filter(
        values,
        size=baseline_filter_size,
        filter=baseline_filter,
        **filter_kwargs,
    )
    reference_index = np.abs(mz_values - ref_mz).argmin()
    reference_signal = values[:, reference_index]
    cell_mask = reference_signal > cell_snr * baseline[:, reference_index]
    labeled_mask, n_cells = label(cell_mask.astype(np.int8))
    common = _common_cell_result(
        data,
        cell_mask,
        labeled_mask,
        baseline,
        reference_index,
    )
    inputs = _CellWindowInputs(values, baseline, reference_signal, labeled_mask)
    return inputs, common, n_cells


def _validate_cell_peak_options(
    data,
    ref_mz,
    baseline_filter_size,
    cell_snr,
    peak_snr,
    max_zero_frac,
) -> None:
    if data.empty or data.shape[1] == 0:
        raise ValueError("data must be a non-empty DataFrame")
    if not np.isfinite(ref_mz) or ref_mz <= 0:
        raise ValueError("ref_mz must be a positive finite number")
    if baseline_filter_size < 1:
        raise ValueError("baseline_filter_size must be at least 1")
    if cell_snr < 0 or peak_snr < 0:
        raise ValueError("cell_snr and peak_snr must be non-negative")
    if not 0 <= max_zero_frac <= 1:
        raise ValueError("max_zero_frac must be between 0 and 1")


def _common_cell_result(data, cell_mask, labeled_mask, baseline, reference_index) -> dict:
    return {
        "cell_mask": cell_mask,
        "labeled_mask": labeled_mask,
        "baseline": baseline,
        "ref_idx": reference_index,
        "ref_mz_matched": data.columns[reference_index],
    }


def _empty_cell_result(data: pd.DataFrame, common: dict) -> dict:
    return {
        **common,
        "cell_df": pd.DataFrame(columns=data.columns),
        "peak_frames": np.array([], dtype=int),
        "window_ranges": [],
        "zero_frac": pd.Series(index=data.columns, data=np.nan),
        "kept_columns": pd.Series(index=data.columns, data=False),
    }


def _process_cell_window(
    label_id: int,
    inputs: _CellWindowInputs,
    baseline_stat: str,
    peak_snr: float,
) -> _CellPeak | None:
    frame_indices = np.where(inputs.labeled_mask == label_id)[0]
    if len(frame_indices) == 0:
        return None
    window_values = inputs.values[frame_indices]
    window_baseline = inputs.baseline[frame_indices]
    feature_maxima = window_values.max(axis=0)
    feature_baseline = _window_baseline(window_baseline, baseline_stat)
    intensities = np.where(feature_maxima > peak_snr * feature_baseline, feature_maxima, 0)
    local_reference = inputs.reference_signal[frame_indices]
    return _CellPeak(
        intensities=intensities,
        peak_frame=int(frame_indices[np.argmax(local_reference)]),
        window_range=(int(frame_indices[0]), int(frame_indices[-1])),
    )


def _window_baseline(baseline: np.ndarray, statistic: str) -> np.ndarray:
    reducers = {"max": np.max, "mean": np.mean, "median": np.median}
    try:
        reducer = reducers[statistic]
    except KeyError as exc:
        raise ValueError("baseline_stat must be one of: 'max', 'mean', 'median'") from exc
    return reducer(baseline, axis=0)


def _assemble_cell_frame(columns, peaks: list[_CellPeak], max_zero_frac: float):
    matrix = np.vstack([peak.intensities for peak in peaks])
    frame = pd.DataFrame(matrix, index=range(len(peaks)), columns=columns)
    zero_fraction = (frame == 0).mean(axis=0)
    kept_columns = zero_fraction <= max_zero_frac
    return frame.loc[:, kept_columns], zero_fraction, kept_columns
