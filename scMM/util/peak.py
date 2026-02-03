import numpy as np
import pandas as pd
from .denoise import _filter, peak_recon, r1_decomposition
from scipy.ndimage import median_filter
from typing import Optional, Literal
from joblib import Parallel, delayed

def peak_detection_recon(data:pd.DataFrame, 
                         baseline_filter = median_filter, 
                         baseline_filter_size:int = 10, 
                         ref_mz: Optional[float] = None, 
                         peak_lam:float = 0.5, 
                         peak_sigma_min:float = 1e-3, 
                         tau:float = 2.0,
                         n_jobs:int = -1, 
                         dtype=np.float64, 
                         **kwargs):
    """
    Docstring for peak_detection_recon
    
    :param data: Description
    :type data: pd.DataFrame
    :param baseline_filter: Description
    :param baseline_filter_size: Description
    :type baseline_filter_size: int
    :param ref_mz: Description
    :type ref_mz: Optional[float]
    :param peak_lam: Description
    :type peak_lam: float
    :param peak_sigma_min: Description
    :type peak_sigma_min: float
    :param peak_threshold: Description
    :type peak_threshold: float
    :param n_jobs: Description
    :type n_jobs: int
    :param dtype: Description
    :param kwargs: Description
    """
    S = data.values.astype(dtype)
    B = _filter(S, size=baseline_filter_size, filter=baseline_filter, **kwargs)
    C, sigma = peak_recon(S, B, lam=peak_lam, sigma_min=peak_sigma_min, tau=tau, n_jobs=n_jobs, dtype=dtype)
    peak_mask = (C > 0)
    if ref_mz is not None:
        ref_index = (np.abs(data.columns.astype(dtype) - ref_mz)).argmin()
        cell_mask = (C[:, ref_index] > 0)
    else:
        a, b = r1_decomposition(C, dtype=dtype)
        cell_mask = (a > 0)
    return cell_mask, peak_mask, C, B, sigma, (a, b) if ref_mz is None else None

def _cell_mask_compute(cell_mask:np.ndarray, peak_mask:np.ndarray):
    combined_mask = peak_mask & cell_mask.reshape(-1, 1)
    return combined_mask

def _find_peaks(cell_mask:np.ndarray):
    mask_int = cell_mask.astype(np.int8)
    d = np.diff(mask_int, prepend=0, append=0)

    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0] - 1 
    return list(zip(starts, ends))

def _peak_pooling(data:np.ndarray, cell_peaks: list[tuple[int, int]], peak_mask:np.ndarray, method:Literal['max', 'integration'] = 'max', dtype = np.float64):
    pooled_values = np.empty(len(cell_peaks), dtype=dtype)
    for i, (start, end) in enumerate(cell_peaks):
        if (peak_mask[start:end + 1] == 0).all():
            pooled_values[i] = 0.0
            continue
        if method == 'integration':
            pooled_values[i] = np.sum(data[start:end + 1], dtype=dtype)
        elif method == 'max':
            pooled_values[i] = np.max(data[start:end + 1])
    return pooled_values

def _peak_width(peaks:list[tuple[int, int]], time: np.ndarray = None, dtype = np.float64):
    peak_width = np.empty(len(peaks), dtype=dtype)
    for i, (start, end) in enumerate(peaks):
        if time is not None:
            widths = time[min(end + 1, len(time) - 1)] - time[max(start - 1, 0)]
        else:
            widths = end - start + 2
        peak_width[i] = widths
    return peak_width

def _peak_symmetry(data:pd.Series, peaks:list[tuple[int, int]], dtype = np.float64):
    sym = np.empty(len(peaks), dtype=dtype)
    for i, (start, end) in enumerate(peaks):
        peak_data = data.values[start:end + 1].astype(dtype)
        if peak_data.size == 0:
            sym[i] = 0.0
            continue
        mid = peak_data.argmax()

        left_area  = np.sum(peak_data[:mid + 1], dtype=dtype)
        right_area = np.sum(peak_data[mid:],     dtype=dtype)

        denom = max(left_area, right_area)
        sym[i] = min(left_area, right_area) / denom if denom > 0 else 0.0
    return np.array(sym, dtype=dtype)

def _peak_rt(data:pd.Series, peaks:list[tuple[int, int]], time: np.ndarray = None, dtype = np.float64):
    peak_rt = np.empty(len(peaks), dtype=dtype)
    for i, (start, end) in enumerate(peaks):
        peak_data = data.values[start:end + 1].astype(dtype)
        mid = peak_data.argmax()
        if time is not None:
            peak_rt[i] = time[start + mid]
        else:
            peak_rt[i] = start + mid
    return peak_rt

def peak_profiling(signal:pd.DataFrame, 
                   baseline: np.ndarray,
                   cell_mask:np.ndarray, 
                   peak_mask:np.ndarray, 
                   ref_mz: Optional[float] = None,
                   C: Optional[np.ndarray] = None,
                   pooling:Literal['max', 'integration'] = 'max',
                   profiling: list = ['rt', 'width', 'symmetry'],
                   time: Optional[np.ndarray] = None,
                   n_jobs:int = -1,
                   subtract_baseline: bool = False,
                   dtype = np.float64):
    if (ref_mz is None) and (C is None):
        raise ValueError("Either ref_mz or C must be provided for peak profiling.")
    signal_denoise = pd.DataFrame(
        (signal.values.astype(dtype) - baseline.astype(dtype)) if subtract_baseline\
        else signal.values.astype(dtype),
        index=signal.index,
        columns=signal.columns,
        dtype=dtype
    )
    cell_peaks = _find_peaks(cell_mask)
    combined_mask = _cell_mask_compute(cell_mask, peak_mask)
    _peaks_pooled = Parallel(n_jobs=n_jobs)(
        delayed(_peak_pooling)(
            signal_denoise.values[:, i],
            cell_peaks,
            combined_mask[:, i],
            method=pooling,
            dtype=dtype
        ) for i in range(signal_denoise.shape[1])
    )
    peaks_pooled = pd.DataFrame(
        np.concatenate([p[:, np.newaxis] for p in _peaks_pooled], axis=1),
        columns=signal_denoise.columns,
        dtype=dtype
    )
    if ref_mz is not None:
        ref_index = (np.abs(signal_denoise.columns.astype(dtype) - ref_mz)).argmin()
    peaks_profiling = pd.DataFrame(index=peaks_pooled.index, dtype=dtype)
    if 'rt' in profiling:
        _peak_rts = _peak_rt(signal_denoise.iloc[:, ref_index], 
                             _find_peaks(combined_mask[:, ref_index]), 
                             time=time, dtype=dtype) if ref_mz is not None else _peak_rt(
            C[:, ref_index], 
            _find_peaks(combined_mask[:, ref_index]), 
            time=time, dtype=dtype)
        peaks_profiling['rt'] = _peak_rts
    if 'width' in profiling:
        _peak_widths = _peak_width(_find_peaks(combined_mask[:, ref_index]), 
                                   time=time, dtype=dtype) if ref_mz is not None else _peak_width(
                                    _find_peaks(combined_mask[:, ref_index]), 
                                    time=time, dtype=dtype)
        peaks_profiling['width'] = _peak_widths
    if 'symmetry' in profiling:
        _peak_symmetries = _peak_symmetry(signal_denoise.iloc[:, ref_index], 
                                          _find_peaks(combined_mask[:, ref_index]), dtype=dtype) if ref_mz is not None else _peak_symmetry(pd.Series(
            C[:, ref_index]), 
            _find_peaks(combined_mask[:, ref_index]), dtype=dtype)
        peaks_profiling['symmetry'] = _peak_symmetries
    return peaks_pooled, peaks_profiling