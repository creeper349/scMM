import numpy as np
import pandas as pd
import pyopenms as oms
from .denoise import peak_recon, r1_decomposition
from scipy.ndimage import median_filter, label
from joblib import Parallel, delayed
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
    ref_mz: float,
    baseline_filter=median_filter,
    baseline_filter_size: int = 15,
    cell_snr: float = 5.0,
    peak_snr: float = 3.0,
    dtype=np.float32,
    baseline_stat="median",
    max_zero_frac: float = 0.9,
    n_jobs: int = 1,
    feature_block_size: int = 256,
    return_full_baseline: bool = True,
    show_progress: bool = True,
    **kwargs,
):
    """Detect cell windows and extract per-cell feature maxima with bounded RAM.

    The original cell/peak criteria are preserved.  The key memory optimization is
    that baseline filtering is performed in independent feature blocks.  Because the
    baseline filter is called with ``size=(baseline_filter_size, 1)``, each feature
    is filtered only along the frame axis and blockwise evaluation is mathematically
    equivalent to filtering the full frame-by-feature matrix at once.

    Parameters added for memory control
    -----------------------------------
    feature_block_size:
        Number of m/z features baseline-filtered at once.
    return_full_baseline:
        When False, do not retain an N x P baseline matrix.  ``ref_baseline`` is
        always returned for reference-channel SNR calculations.
    show_progress:
        Display a tqdm progress bar over feature blocks.
    """
    feature_block_size = int(feature_block_size)
    if feature_block_size <= 0:
        raise ValueError("feature_block_size must be a positive integer.")
    if baseline_stat not in {"max", "mean", "median"}:
        raise ValueError("baseline_stat must be one of: 'max', 'mean', 'median'")

    # ``dtype`` controls the large frame x feature / cell x feature intensity
    # arrays.  float32 is the default to halve RAM relative to float64.  When the
    # input DataFrame already has the requested dtype this remains a zero-copy view.
    output_dtype = np.dtype(dtype)
    if output_dtype.kind != "f":
        raise TypeError("dtype must be a floating-point dtype.")
    X = data.to_numpy(dtype=output_dtype, copy=False)

    mz_values = np.asarray(data.columns, dtype=np.float64)
    ref_idx = int(np.abs(mz_values - ref_mz).argmin())

    # Cell detection requires only the reference feature baseline.  Computing this
    # one column first avoids materializing the full baseline matrix before windows
    # are known.
    ref_signal = X[:, ref_idx]
    ref_baseline = _filter(
        X[:, ref_idx:ref_idx + 1],
        size=baseline_filter_size,
        filter=baseline_filter,
        **kwargs,
    )[:, 0]
    cell_mask = ref_signal > cell_snr * ref_baseline
    labeled_mask, n_cells = label(cell_mask.astype(np.int8))

    # In a 1-D connected-component mask each label corresponds to one contiguous
    # slice.  Using slices avoids repeated np.where scans and avoids copying X_win.
    padded = np.concatenate(([False], cell_mask, [False]))
    transitions = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    window_slices = [slice(int(s), int(e)) for s, e in zip(starts, stops)]

    if len(window_slices) != n_cells:
        raise RuntimeError(
            f"Internal cell-window mismatch: label() found {n_cells}, "
            f"but {len(window_slices)} contiguous windows were identified."
        )

    peak_frames = np.empty(n_cells, dtype=np.int64)
    window_ranges = []
    for i, slc in enumerate(window_slices):
        local_ref = ref_signal[slc]
        peak_frames[i] = slc.start + int(np.argmax(local_ref))
        window_ranges.append((slc.start, slc.stop - 1))

    _, n_features = X.shape

    # In the normal low-memory preprocessing path there is nothing else to compute
    # when no cells are detected; the reference baseline is already available.
    if n_cells == 0 and not return_full_baseline:
        empty_df = pd.DataFrame(columns=data.columns, dtype=X.dtype)
        return {
            "cell_df": empty_df,
            "cell_mask": cell_mask,
            "labeled_mask": labeled_mask,
            "baseline": None,
            "ref_baseline": ref_baseline,
            "ref_idx": ref_idx,
            "ref_mz_matched": data.columns[ref_idx],
            "peak_frames": peak_frames,
            "window_ranges": window_ranges,
            "zero_frac": pd.Series(index=data.columns, data=np.nan),
            "kept_columns": pd.Series(index=data.columns, data=False),
        }

    baseline_full = np.empty_like(X) if return_full_baseline else None
    cell_matrix = (
        np.empty((n_cells, n_features), dtype=X.dtype)
        if n_cells > 0
        else None
    )
    zero_counts = np.empty(n_features, dtype=np.int64) if n_cells > 0 else None

    block_starts = range(0, n_features, feature_block_size)
    progress = tqdm(
        block_starts,
        total=(n_features + feature_block_size - 1) // feature_block_size,
        desc="Cell peak extraction",
        unit="block",
        disable=not show_progress,
    )

    for col_start in progress:
        col_stop = min(col_start + int(feature_block_size), n_features)
        X_block = X[:, col_start:col_stop]
        B_block = _filter(
            X_block,
            size=baseline_filter_size,
            filter=baseline_filter,
            **kwargs,
        )

        if baseline_full is not None:
            baseline_full[:, col_start:col_stop] = B_block

        if n_cells == 0:
            del B_block
            continue

        def _process_one_window(cell_i):
            slc = window_slices[cell_i]
            X_win = X_block[slc, :]
            B_win = B_block[slc, :]

            feat_max = X_win.max(axis=0)
            if baseline_stat == "max":
                feat_baseline = B_win.max(axis=0)
            elif baseline_stat == "mean":
                feat_baseline = B_win.mean(axis=0)
            else:  # median
                feat_baseline = np.median(B_win, axis=0)

            valid = feat_max > peak_snr * feat_baseline
            return cell_i, np.where(valid, feat_max, 0).astype(X.dtype, copy=False)

        if n_jobs in (None, 1):
            block_results = (_process_one_window(i) for i in range(n_cells))
            for cell_i, feat_out in block_results:
                cell_matrix[cell_i, col_start:col_stop] = feat_out
        else:
            # Threads share X/B_block and therefore avoid process-level copies of the
            # large intensity matrix.  Results are only one feature block per cell.
            block_results = Parallel(
                n_jobs=n_jobs,
                prefer="threads",
                batch_size=1,
            )(delayed(_process_one_window)(i) for i in range(n_cells))
            for cell_i, feat_out in block_results:
                cell_matrix[cell_i, col_start:col_stop] = feat_out
            del block_results

        # Count zeros per block so we never materialize a full n_cells x n_features
        # boolean matrix solely for feature filtering.
        zero_counts[col_start:col_stop] = np.count_nonzero(
            cell_matrix[:, col_start:col_stop] == 0, axis=0
        )
        del B_block

    if n_cells == 0:
        empty_df = pd.DataFrame(columns=data.columns, dtype=X.dtype)
        return {
            "cell_df": empty_df,
            "cell_mask": cell_mask,
            "labeled_mask": labeled_mask,
            "baseline": baseline_full,
            "ref_baseline": ref_baseline,
            "ref_idx": ref_idx,
            "ref_mz_matched": data.columns[ref_idx],
            "peak_frames": peak_frames,
            "window_ranges": window_ranges,
            "zero_frac": pd.Series(index=data.columns, data=np.nan),
            "kept_columns": pd.Series(index=data.columns, data=False),
        }

    # Zero counts were accumulated blockwise above, avoiding an additional full
    # boolean matrix or DataFrame.
    zero_frac_values = zero_counts.astype(np.float64) / float(n_cells)
    keep_mask = zero_frac_values <= max_zero_frac
    zero_frac = pd.Series(zero_frac_values, index=data.columns)
    keep_cols = pd.Series(keep_mask, index=data.columns)

    kept_matrix = cell_matrix if np.all(keep_mask) else cell_matrix[:, keep_mask]
    kept_columns = data.columns[keep_mask]
    cell_df = pd.DataFrame(
        kept_matrix,
        index=np.arange(n_cells),
        columns=kept_columns,
        copy=False,
    )

    return {
        "cell_df": cell_df,
        "cell_mask": cell_mask,
        "labeled_mask": labeled_mask,
        "baseline": baseline_full,
        "ref_baseline": ref_baseline,
        "ref_idx": ref_idx,
        "ref_mz_matched": data.columns[ref_idx],
        "peak_frames": peak_frames,
        "window_ranges": window_ranges,
        "zero_frac": zero_frac,
        "kept_columns": keep_cols,
    }
