import numpy as np
import pandas as pd
import pyopenms as oms
from .denoise import peak_recon, r1_decomposition
from scipy.ndimage import median_filter, label
from joblib import Parallel, delayed

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
    mz, intensity = spec.get_peaks()

    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)

    if intensity.size == 0:
        out_spec = oms.MSSpectrum()
        out_spec.set_peaks((mz, intensity))
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
    out_spec.set_peaks((mz, filtered))
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

def find_cell_peaks(data: pd.DataFrame, ref_mz: float, baseline_filter=median_filter, baseline_filter_size: int = 15,
                    cell_snr: float = 5.0, peak_snr: float = 3.0, dtype=np.float64, baseline_stat="median",
                    max_zero_frac: float = 0.9, n_jobs: int = -1, **kwargs):
    
    X = data.values.astype(dtype)
    mz_values = data.columns.astype(dtype)
    B = _filter(X, size=baseline_filter_size, filter=baseline_filter, **kwargs)
    ref_idx = np.abs(mz_values - ref_mz).argmin()

    ref_signal = X[:, ref_idx]
    ref_baseline = B[:, ref_idx]
    cell_mask = ref_signal > cell_snr * ref_baseline

    labeled_mask, n_cells = label(cell_mask.astype(np.int8))

    if n_cells == 0:
        empty_df = pd.DataFrame(columns=data.columns)
        return {
            "cell_df": empty_df,
            "cell_mask": cell_mask,
            "labeled_mask": labeled_mask,
            "baseline": B,
            "ref_idx": ref_idx,
            "ref_mz_matched": data.columns[ref_idx],
            "peak_frames": np.array([], dtype=int),
            "window_ranges": [],
            "zero_frac": pd.Series(index=data.columns, data=np.nan),
            "kept_columns": pd.Series(index=data.columns, data=False),
        }

    def _process_one_label(lab):
        idx = np.where(labeled_mask == lab)[0]
        if len(idx) == 0:
            return None

        X_win = X[idx, :]
        B_win = B[idx, :]

        feat_max = X_win.max(axis=0)

        if baseline_stat == "max":
            feat_baseline = B_win.max(axis=0)
        elif baseline_stat == "mean":
            feat_baseline = B_win.mean(axis=0)
        elif baseline_stat == "median":
            feat_baseline = np.median(B_win, axis=0)
        else:
            raise ValueError("baseline_stat must be one of: 'max', 'mean', 'median'")

        valid = feat_max > peak_snr * feat_baseline
        feat_out = np.where(valid, feat_max, 0)

        local_ref = ref_signal[idx]
        peak_frame = idx[np.argmax(local_ref)]
        window_range = (idx[0], idx[-1])

        return feat_out, peak_frame, window_range

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_label)(lab) for lab in range(1, n_cells + 1)
    )

    results = [r for r in results if r is not None]

    cell_rows = [r[0] for r in results]
    peak_frames = [r[1] for r in results]
    window_ranges = [r[2] for r in results]

    cell_matrix = np.vstack(cell_rows)
    cell_index = list(range(cell_matrix.shape[0]))
    cell_df = pd.DataFrame(cell_matrix, index=cell_index, columns=data.columns)

    zero_frac = (cell_df == 0).mean(axis=0)
    keep_cols = zero_frac <= max_zero_frac
    cell_df = cell_df.loc[:, keep_cols]

    return {
        "cell_df": cell_df,
        "cell_mask": cell_mask,
        "labeled_mask": labeled_mask,
        "baseline": B,
        "ref_idx": ref_idx,
        "ref_mz_matched": data.columns[ref_idx],
        "peak_frames": np.array(peak_frames, dtype=int),
        "window_ranges": window_ranges,
        "zero_frac": zero_frac,
        "kept_columns": keep_cols,
    }