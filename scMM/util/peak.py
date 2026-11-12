import numpy as np
import pandas as pd
import pyopenms as oms
from scipy.ndimage import median_filter, label, uniform_filter1d
from tqdm.auto import tqdm

def filter_spectrum(
    spec: oms.MSSpectrum,
    baseline_window: int = 101,
    noise_window: int = 101,
    baseline_quantile: float = 0.1,
    snr_threshold: float = 3.0,
    keep_negative: bool = False,
    return_snr: bool = False,
    baseline_stride: int = 10,  
    dtype=np.float32,
):
    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be a floating-point dtype.")

    mz, intensity = spec.get_peaks()

    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    if intensity.size == 0:
        out_spec = oms.MSSpectrum()
        out_spec.set_peaks((mz, intensity.astype(output_dtype, copy=False)))
        out_spec.setRT(spec.getRT())
        out_spec.setMSLevel(spec.getMSLevel())
        if return_snr:
            return out_spec, np.array([], dtype=np.float64)
        return out_spec

    def _make_odd(k: int) -> int:
        k = max(1, int(k))
        return k if k % 2 == 1 else k + 1

    baseline_window = _make_odd(baseline_window)
    noise_window = _make_odd(noise_window)
    baseline_stride = max(1, int(baseline_stride))

    n = intensity.size
    eps = 1e-12

    if baseline_stride == 1:
        baseline = np.empty(n, dtype=np.float64)
        hw_b = baseline_window // 2
        for i in range(n):
            left = max(0, i - hw_b)
            right = min(n, i + hw_b + 1)
            baseline[i] = np.quantile(intensity[left:right], baseline_quantile)
    else:
        anchor_idx = np.arange(0, n, baseline_stride, dtype=np.int64)
        if anchor_idx[-1] != n - 1:
            anchor_idx = np.append(anchor_idx, n - 1)

        anchor_baseline = np.empty(anchor_idx.size, dtype=np.float64)
        hw_b = baseline_window // 2

        for j, i in enumerate(anchor_idx):
            left = max(0, i - hw_b)
            right = min(n, i + hw_b + 1)
            anchor_baseline[j] = np.quantile(intensity[left:right], baseline_quantile)

        baseline = np.interp(
            np.arange(n, dtype=np.float64),
            anchor_idx.astype(np.float64),
            anchor_baseline
        )

    residual = intensity - baseline
    noise = np.empty(n, dtype=np.float64)
    hw_n = noise_window // 2

    for i in range(n):
        left = max(0, i - hw_n)
        right = min(n, i + hw_n + 1)
        local_res = residual[left:right]

        med = np.median(local_res)
        mad = np.median(np.abs(local_res - med))
        sigma = 1.4826 * mad
        noise[i] = max(sigma, eps)

    signal = residual.copy()
    if not keep_negative:
        signal[signal < 0] = 0.0

    snr = signal / noise

    filtered = signal.copy()
    filtered[snr < snr_threshold] = 0.0

    out_spec = oms.MSSpectrum()
    # Baseline/noise estimation is intentionally kept in float64 for numerical
    # stability.  Only the stored intensity vector is cast to the configurable
    # output dtype; float32 is the memory-efficient default.
    out_spec.set_peaks((mz, filtered.astype(output_dtype, copy=False)))
    out_spec.setRT(spec.getRT())
    out_spec.setMSLevel(spec.getMSLevel())

    try:
        out_spec.setName(spec.getName())
    except Exception:
        pass

    try:
        out_spec.setDriftTime(spec.getDriftTime())
    except Exception:
        pass

    if return_snr:
        return out_spec, snr
    return out_spec

def _filter(data:np.ndarray, size:int = 10, filter = median_filter, **filter_kwargs):
    return filter(data, size = (size, 1), **filter_kwargs)

def find_cell_peaks(
    data: pd.DataFrame,
    reference_mz,
    *,
    reference_mode: str = "union",
    ppm_tolerance: float = 10.0,
    baseline_filter=median_filter,
    baseline_window: int = 51,
    cell_signal_threshold: float = 5.0,
    feature_snr_threshold: float = 3.0,
    noise_window: int = 51,
    max_zero_fraction: float = 0.9,
    dtype=np.float32,
    n_jobs: int = 1,
    feature_block_size: int = 256,
    return_full_baseline: bool = False,
    show_progress: bool = True,
    progress_callback=None,
    **filter_kwargs,
):
    """Extract cells using reference EICs and quantify features with true SNR.

    Cell windows are detected from one or more reference EICs using a
    signal-to-baseline ratio. reference_mode combines their masks by union or
    intersection. Feature detection is deliberately different: local noise is
    estimated from frames outside every detected cell, and a feature is retained
    in a cell only when its baseline-corrected apex divided by that noise standard
    deviation exceeds feature_snr_threshold.
    """
    del n_jobs
    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be a floating-point dtype")
    if reference_mode not in {"union", "intersection"}:
        raise ValueError("reference_mode must be 'union' or 'intersection'")
    if not np.isfinite(ppm_tolerance) or ppm_tolerance <= 0:
        raise ValueError("ppm_tolerance must be a positive finite number")
    if feature_block_size < 1 or baseline_window < 1 or noise_window < 2:
        raise ValueError("window and block sizes must be positive")
    if not 0 <= max_zero_fraction <= 1:
        raise ValueError("max_zero_fraction must be between 0 and 1")

    references = np.atleast_1d(reference_mz).astype(np.float64)
    if (
        references.size == 0
        or not np.all(np.isfinite(references))
        or np.any(references <= 0)
    ):
        raise ValueError("reference_mz must contain at least one positive finite value")

    X = data.to_numpy(dtype=output_dtype, copy=False)
    mz_values = np.asarray(data.columns, dtype=np.float64)
    reference_indices = np.asarray(
        [int(np.abs(mz_values - value).argmin()) for value in references],
        dtype=np.int64,
    )
    matched_reference_mz = mz_values[reference_indices]
    ppm_errors = (
        np.abs(matched_reference_mz - references)
        / np.abs(references)
        * 1e6
    )
    missing = ppm_errors > float(ppm_tolerance)
    if np.any(missing):
        details = ", ".join(
            f"{requested:g} (nearest {matched:g}, {error:.3f} ppm)"
            for requested, matched, error in zip(
                references[missing],
                matched_reference_mz[missing],
                ppm_errors[missing],
            )
        )
        raise ValueError(
            "Reference m/z feature was not detected within "
            f"{float(ppm_tolerance):g} ppm: {details}"
        )

    reference_signal = X[:, reference_indices]
    reference_baseline = _filter(
        reference_signal,
        size=baseline_window,
        filter=baseline_filter,
        **filter_kwargs,
    )
    eps = np.finfo(output_dtype).eps
    reference_ratio = reference_signal / np.maximum(reference_baseline, eps)
    reference_masks = reference_ratio > float(cell_signal_threshold)
    cell_mask = (
        np.any(reference_masks, axis=1)
        if reference_mode == "union"
        else np.all(reference_masks, axis=1)
    )

    labeled_mask, n_cells = label(cell_mask.astype(np.int8))
    padded = np.concatenate(([False], cell_mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    window_slices = [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]
    if len(window_slices) != n_cells:
        raise RuntimeError("cell-window labeling produced inconsistent results")

    reference_score = np.mean(reference_ratio, axis=1)
    peak_frames = np.asarray(
        [slc.start + int(np.argmax(reference_score[slc])) for slc in window_slices],
        dtype=np.int64,
    )
    window_ranges = [(slc.start, slc.stop - 1) for slc in window_slices]

    _, n_features = X.shape
    baseline_full = np.empty_like(X) if return_full_baseline else None
    cell_matrix = np.zeros((n_cells, n_features), dtype=output_dtype)
    feature_snr = np.zeros((n_cells, n_features), dtype=output_dtype)
    non_cell = (~cell_mask).astype(np.float64)[:, None]

    progress = tqdm(
        range(0, n_features, feature_block_size),
        total=(n_features + feature_block_size - 1) // feature_block_size,
        desc="Cell feature extraction",
        unit="block",
        disable=not show_progress,
    )
    total_blocks = max(1, (n_features + feature_block_size - 1) // feature_block_size)
    for block_index, col_start in enumerate(progress):
        col_stop = min(col_start + feature_block_size, n_features)
        X_block = X[:, col_start:col_stop]
        baseline = _filter(
            X_block,
            size=baseline_window,
            filter=baseline_filter,
            **filter_kwargs,
        )
        residual = X_block - baseline
        if baseline_full is not None:
            baseline_full[:, col_start:col_stop] = baseline

        count = uniform_filter1d(
            non_cell, size=noise_window, axis=0, mode="constant", cval=0.0
        ) * noise_window
        sum_ = uniform_filter1d(
            residual * non_cell,
            size=noise_window,
            axis=0,
            mode="constant",
            cval=0.0,
        ) * noise_window
        sum_sq = uniform_filter1d(
            residual * residual * non_cell,
            size=noise_window,
            axis=0,
            mode="constant",
            cval=0.0,
        ) * noise_window
        mean = np.divide(sum_, count, out=np.zeros_like(sum_), where=count > 0)
        variance = (
            np.divide(sum_sq, count, out=np.zeros_like(sum_sq), where=count > 1)
            - mean * mean
        )
        noise = np.sqrt(np.maximum(variance, 0.0))

        background = residual[~cell_mask]
        if background.shape[0] > 1:
            global_noise = np.nanstd(background, axis=0, ddof=1)
        else:
            global_noise = np.nanstd(residual, axis=0)
        global_noise = np.maximum(global_noise, eps)
        noise = np.where(
            (count > 1) & (noise > eps), noise, global_noise[None, :]
        )

        for cell_i, slc in enumerate(window_slices):
            corrected_window = residual[slc]
            apex_local = np.argmax(corrected_window, axis=0)
            feature_index = np.arange(col_stop - col_start)
            corrected_apex = np.maximum(
                corrected_window[apex_local, feature_index], 0.0
            )
            absolute_apex = slc.start + apex_local
            local_noise = noise[absolute_apex, feature_index]
            snr = corrected_apex / np.maximum(local_noise, eps)
            detected = snr >= float(feature_snr_threshold)
            cell_matrix[cell_i, col_start:col_stop] = np.where(
                detected, corrected_apex, 0.0
            ).astype(output_dtype, copy=False)
            feature_snr[cell_i, col_start:col_stop] = snr.astype(
                output_dtype, copy=False
            )
        if progress_callback is not None:
            progress_callback(
                (block_index + 1) / total_blocks,
                f"特征提取 block {block_index + 1}/{total_blocks}",
            )

    if n_cells:
        zero_fraction_values = np.mean(cell_matrix == 0, axis=0)
    else:
        zero_fraction_values = np.ones(n_features, dtype=np.float64)
    keep_mask = zero_fraction_values <= max_zero_fraction
    cell_df = pd.DataFrame(
        cell_matrix[:, keep_mask],
        index=pd.RangeIndex(n_cells, name="cell"),
        columns=data.columns[keep_mask],
        copy=False,
    )

    return {
        "cell_df": cell_df,
        "cell_mask": cell_mask,
        "labeled_mask": labeled_mask,
        "baseline": baseline_full,
        "reference_baseline": reference_baseline,
        "reference_signal": reference_signal,
        "reference_ratio": reference_ratio,
        "reference_indices": reference_indices,
        "reference_mz_matched": matched_reference_mz,
        "reference_ppm_error": ppm_errors,
        "peak_frames": peak_frames,
        "window_ranges": window_ranges,
        "feature_snr": feature_snr[:, keep_mask],
        "zero_fraction": pd.Series(zero_fraction_values, index=data.columns),
        "kept_columns": pd.Series(keep_mask, index=data.columns),
    }
