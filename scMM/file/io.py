from typing import Literal, Tuple, Dict, Any
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
from collections.abc import Sequence
import pyopenms as oms
import pandas as pd
import numpy as np
import os
import logging

def load_single_file(
    path: str,
    format: Literal['auto', 'mzML', 'mzXML'] = 'mzML'
) -> Tuple[oms.MSExperiment, Dict[str, Any]]:
    
    exp = oms.MSExperiment()
    if format == 'auto':
        format = 'mzML' if path.lower().endswith('.mzml') else 'mzXML'
    
    if format == 'mzML':
        oms.MzMLFile().load(path, exp)
    elif format == 'mzXML':
        oms.MzXMLFile().load(path, exp)

    metadata = {}
    metadata["name"], _ = os.path.splitext(os.path.basename(path))
    metadata["timestamp"] = datetime.strptime(exp.getDateTime().get(), "%Y-%m-%d %H:%M:%S").timestamp()
    metadata["instrument"] = exp.getInstrument().getName()
    logging.info(f"Loaded MS file from: {path}")

    return exp, metadata

def orbitrap_resolution_at_mz(mz: float, resolution_200: float) -> float:
    return resolution_200 * np.sqrt(200.0 / mz)

def orbitrap_fwhm_at_mz(mz: float, resolution_200: float) -> float:
    return mz / orbitrap_resolution_at_mz(mz, resolution_200)

def build_orbitrap_grid(
    mz_range=(100.0, 1000.0),
    resolution_200: float = 70000.0,
    points_per_fwhm: float = 5.0,
) -> np.ndarray:

    mz_min, mz_max = map(float, mz_range)
    if mz_min <= 0 or mz_max <= mz_min:
        raise ValueError("Invalid mz_range.")
    if resolution_200 <= 0:
        raise ValueError("resolution_200 must be positive.")
    if points_per_fwhm <= 0:
        raise ValueError("points_per_fwhm must be positive.")

    grid = [mz_min]
    mz = mz_min
    while mz < mz_max:
        dm = orbitrap_fwhm_at_mz(mz, resolution_200) / points_per_fwhm
        if dm <= 0 or not np.isfinite(dm):
            raise ValueError("Invalid grid spacing encountered.")
        mz = mz + dm
        grid.append(mz)

    grid = np.asarray(grid, dtype=np.float64)
    if grid[-1] > mz_max:
        grid[-1] = mz_max
    elif grid[-1] < mz_max:
        grid = np.append(grid, mz_max)

    return grid


def _prepare_sorted_unique_peaks(mz: np.ndarray, inten: np.ndarray):

    mz = np.asarray(mz, dtype=np.float64)
    inten = np.asarray(inten, dtype=np.float64)

    valid = np.isfinite(mz) & np.isfinite(inten)
    mz = mz[valid]
    inten = inten[valid]

    if mz.size == 0:
        return mz, inten

    if np.any(np.diff(mz) < 0):
        order = np.argsort(mz)
        mz = mz[order]
        inten = inten[order]

    if mz.size > 1:
        uniq_mz, inverse = np.unique(mz, return_inverse=True)
        if uniq_mz.size != mz.size:
            new_inten = np.zeros_like(uniq_mz, dtype=np.float64)
            np.add.at(new_inten, inverse, inten)
            mz = uniq_mz
            inten = new_inten

    return mz, inten

def sum_spec(
    exp: oms.MSExperiment,
    mz_range=(100.0, 1000.0),
    resolution_200: float = 35000.0,
    points_per_fwhm: float = 5.0,
    ms_level: int = 1,
    normalize: bool = False,
    zero_outside: bool = True,
    intensity_dtype=np.float32,
):

    mz_min, mz_max = map(float, mz_range)

    mz_grid = build_orbitrap_grid(
        mz_range=(mz_min, mz_max),
        resolution_200=resolution_200,
        points_per_fwhm=points_per_fwhm,
    )

    acc = np.zeros_like(mz_grid, dtype=np.float64)
    total_spectra = 0

    for spec in exp:
        if spec.getMSLevel() != ms_level:
            continue

        mz, inten = spec.get_peaks()
        mz, inten = _prepare_sorted_unique_peaks(mz, inten)

        if mz.size == 0:
            continue

        if mz[-1] < mz_min or mz[0] > mz_max:
            continue

        if zero_outside:
            interp_inten = np.interp(
                mz_grid,
                mz,
                inten,
                left=0.0,
                right=0.0
            )
        else:
            interp_inten = np.interp(mz_grid, mz, inten)

        acc += interp_inten
        total_spectra += 1

    if total_spectra == 0:
        raise ValueError("No spectra found.")

    out_intensity = acc / total_spectra if normalize else acc

    spec_out = oms.MSSpectrum()
    spec_out.setMSLevel(ms_level)
    spec_out.setRT(0.0)
    intensity_dtype = np.dtype(intensity_dtype)
    if intensity_dtype.kind != "f":
        raise TypeError("intensity_dtype must be a floating-point dtype.")

    spec_out.set_peaks((
        mz_grid.astype(np.float64, copy=False),
        out_intensity.astype(intensity_dtype, copy=False)
    ))

    spec_out.setMetaValue("n_summed_spectra", int(total_spectra))
    spec_out.setMetaValue("resolution_200", float(resolution_200))
    spec_out.setMetaValue("points_per_fwhm", float(points_per_fwhm))
    spec_out.setMetaValue("mz_min", float(mz_min))
    spec_out.setMetaValue("mz_max", float(mz_max))
    spec_out.setMetaValue("grid_type", "orbitrap_variable_fwhm")
    spec_out.setMetaValue("interpolation", "linear")
    spec_out.setMetaValue("normalized", int(bool(normalize)))

    return spec_out

def sum_spectrum_from_file(
        path: str,
        ms_level: int = 1,
        resolution_200: float = 35000.0,
        points_per_fwhm: float = 5.0,
        intensity_dtype=np.float32,
    ) -> tuple[oms.MSSpectrum, int]:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path, exp)
    return sum_spec(
        exp,
        ms_level=ms_level,
        resolution_200=resolution_200,
        points_per_fwhm=points_per_fwhm,
        intensity_dtype=intensity_dtype,
    )

def extract_peaks(
    spec: oms.MSSpectrum,
    dtype=np.float32,
    mz_dtype=np.float64,
    prominence_ratio: float = None,
    distance: int = 3,
    method: str = "centroid",    
    resolution_200: float = 35000.0, 
    window_fwhm_factor: float = 1.0,   
    centroid_intensity_mode: str = "apex" 
) -> Tuple[np.ndarray, np.ndarray]:

    # m/z values remain float64 by default because centroid locations and ppm
    # matching are precision-sensitive.  ``dtype`` controls intensity storage and
    # defaults to float32, which is the large-memory path in raw-MS processing.
    intensity_dtype = np.dtype(dtype)
    mz_dtype = np.dtype(mz_dtype)
    if intensity_dtype.kind != "f" or mz_dtype.kind != "f":
        raise TypeError("dtype and mz_dtype must be floating-point dtypes.")

    mz = np.asarray(spec.get_peaks()[0], dtype=mz_dtype)
    # Peak localization/centroiding is kept in float64 to preserve the previous
    # numerical behavior near ppm-matching boundaries.  Only the returned peak
    # intensity vector is down-cast to the configurable storage dtype.
    intensity = np.asarray(spec.get_peaks()[1], dtype=np.float64)

    if mz.size == 0:
        return np.array([], dtype=mz_dtype), np.array([], dtype=intensity_dtype)

    # Ensure sorted
    if mz.size > 1 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]

    prom = None
    if prominence_ratio is not None:
        if intensity.size == 0 or np.max(intensity) <= 0:
            return np.array([], dtype=mz_dtype), np.array([], dtype=intensity_dtype)
        prom = np.max(intensity) * prominence_ratio

    peak_idx, _ = find_peaks(
        intensity,
        prominence=prom,
        distance=distance
    )

    if peak_idx.size == 0:
        return np.array([], dtype=mz_dtype), np.array([], dtype=intensity_dtype)

    peak_mz_out = []
    peak_int_out = []

    n = mz.size

    for i in peak_idx:
        mz0 = float(mz[i])

        fwhm = orbitrap_fwhm_at_mz(mz0, resolution_200)

        if i == 0:
            dm_local = float(mz[1] - mz[0]) if n > 1 else np.nan
        elif i == n - 1:
            dm_local = float(mz[-1] - mz[-2])
        else:
            dm_local = float((mz[i + 1] - mz[i - 1]) / 2.0)

        if (not np.isfinite(dm_local)) or dm_local <= 0:
            peak_mz_out.append(mz[i])
            peak_int_out.append(intensity[i])
            continue

        half_window_pts = max(
            1,
            int(np.ceil(window_fwhm_factor * fwhm / dm_local))
        )

        left = max(0, i - half_window_pts)
        right = min(n, i + half_window_pts + 1)

        mz_win = mz[left:right]
        int_win = intensity[left:right]

        if mz_win.size == 0:
            peak_mz_out.append(mz[i])
            peak_int_out.append(intensity[i])
            continue

        if method == "centroid":
            s = np.sum(int_win)
            if s <= 0:
                peak_mz = mz[i]
            else:
                peak_mz = np.sum(mz_win * int_win) / s

            if centroid_intensity_mode == "apex":
                peak_int = intensity[i]
            elif centroid_intensity_mode == "sum":
                peak_int = s
            else:
                raise ValueError("centroid_intensity_mode must be 'apex' or 'sum'")

            peak_mz_out.append(peak_mz)
            peak_int_out.append(peak_int)
            continue

        elif method == "parabola":
            if i == 0 or i == n - 1:
                peak_mz_out.append(mz[i])
                peak_int_out.append(intensity[i])
                continue

            x1, x2, x3 = float(mz[i - 1]), float(mz[i]), float(mz[i + 1])
            y1, y2, y3 = float(intensity[i - 1]), float(intensity[i]), float(intensity[i + 1])

            x1l = x1 - x2
            x2l = 0.0
            x3l = x3 - x2

            denom = (x1l - x2l) * (x1l - x3l) * (x2l - x3l)
            if denom == 0:
                peak_mz_out.append(mz[i])
                peak_int_out.append(intensity[i])
                continue

            a = (x3l * (y2 - y1) + x2l * (y1 - y3) + x1l * (y3 - y2)) / denom
            b = (x3l**2 * (y1 - y2) + x2l**2 * (y3 - y1) + x1l**2 * (y2 - y3)) / denom
            c = y2

            if a >= 0:
                peak_mz_out.append(mz[i])
                peak_int_out.append(intensity[i])
                continue

            x_peak_local = -b / (2.0 * a)

            if x_peak_local < x1l or x_peak_local > x3l:
                peak_mz_out.append(mz[i])
                peak_int_out.append(intensity[i])
                continue

            peak_mz = x2 + x_peak_local
            peak_int = a * x_peak_local**2 + b * x_peak_local + c

            peak_mz_out.append(peak_mz)
            peak_int_out.append(peak_int)
            continue

        else:
            raise ValueError("method must be 'centroid' or 'parabola'")

    return (
        np.asarray(peak_mz_out, dtype=mz_dtype),
        np.asarray(peak_int_out, dtype=intensity_dtype),
    )

def align_frame(
        exp: oms.MSExperiment,
        mz_list,
        ppm: float = 10.0,
        ms_level: int = 1,
        aggregate: str = "max",   # "sum" | "max"
        dtype=np.float32,
        mz_dtype=np.float64,
        out=None,
        **kwargs
    ):
    """Align MS1 frames to a target m/z list.

    Parameters
    ----------
    exp : pyopenms.MSExperiment
        Input experiment.
    mz_list : array-like
        Target m/z values. Their original order defines output columns.
    ppm : float, default=10.0
        Maximum nearest-target mass error in ppm.
    ms_level : int, default=1
        MS level to align.
    aggregate : {"sum", "max"}, default="max"
        Rule used when multiple extracted peaks map to the same target.
    dtype : numpy dtype, default=np.float32
        Intensity/output dtype.
    mz_dtype : numpy dtype, default=np.float64
        m/z computation dtype.
    out : numpy.ndarray, optional
        Writable preallocated array with shape ``(n_frames, n_targets)``.
        When supplied, alignment is written directly into this buffer and no
        separate frame x feature intensity matrix is allocated. The buffer is
        zero-filled before alignment.

    Returns
    -------
    data : pandas.DataFrame
        View-like DataFrame wrapper around the aligned output array.
    peak_meta : pandas.DataFrame
        Retention-time metadata indexed by original frame id.

    Notes
    -----
    Providing ``out`` changes only storage, not matching or aggregation logic.
    The returned DataFrame preserves the historical API.
    """

    intensity_dtype = np.dtype(dtype)
    mz_dtype = np.dtype(mz_dtype)
    if intensity_dtype.kind != "f" or mz_dtype.kind != "f":
        raise TypeError("dtype and mz_dtype must be floating-point dtypes.")

    targets = np.asarray(mz_list, dtype=mz_dtype)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("mz_list must be a non-empty 1D array-like.")

    order = np.argsort(targets)
    targets_sorted = targets[order]

    n_targets = targets_sorted.size

    spectra = []
    rt_list = []
    frame_ids = []

    for i, spec in enumerate(exp):
        if spec.getMSLevel() != ms_level:
            continue
        spectra.append((i, spec))
        rt_list.append(spec.getRT())
        frame_ids.append(i)

    n_frames = len(spectra)
    if n_frames == 0:
        raise ValueError("No spectra found.")

    if out is None:
        X = np.zeros((n_frames, n_targets), dtype=intensity_dtype)
    else:
        if not isinstance(out, np.ndarray):
            raise TypeError("out must be a writable numpy.ndarray or None.")
        if out.shape != (n_frames, n_targets):
            raise ValueError(
                "out has incompatible shape: "
                f"expected {(n_frames, n_targets)}, got {out.shape}."
            )
        if not out.flags.writeable:
            raise ValueError("out must be writable.")
        if np.dtype(out.dtype) != intensity_dtype:
            raise TypeError(
                "out dtype must match dtype exactly: "
                f"expected {intensity_dtype.name}, got {out.dtype}."
            )
        X = out
        # ``aligned_matrix`` is allocated with np.empty in the multi-file path.
        # Zero the assigned row slice before sparse peak writes so unmatched
        # frame-feature entries retain the same value as the historical
        # np.zeros-based implementation.
        X.fill(0)

    for row_idx, (frame_id, spec) in enumerate(spectra):

        mz, inten = extract_peaks(
            spec,
            dtype=dtype,
            mz_dtype=mz_dtype,
            prominence_ratio=kwargs.get("prominence_ratio", None),
            distance=kwargs.get("distance", 3),
            method=kwargs.get("method", "centroid"),
            resolution_200=kwargs.get("resolution_200", 70000.0),
            window_fwhm_factor=kwargs.get("window_fwhm_factor", 1.0),
            centroid_intensity_mode=kwargs.get("centroid_intensity_mode", "apex"),
        )

        if mz.size == 0:
            continue

        mz = np.asarray(mz, dtype=mz_dtype)
        inten = np.asarray(inten, dtype=intensity_dtype)

        if mz.size >= 2 and np.any(np.diff(mz) < 0):
            idx = np.argsort(mz)
            mz = mz[idx]
            inten = inten[idx]

        # For each extracted peak, find nearest target
        pos = np.searchsorted(targets_sorted, mz)

        left_idx = pos - 1
        right_idx = pos

        left_valid = left_idx >= 0
        right_valid = right_idx < n_targets

        left_ppm = np.full(mz.shape, np.inf, dtype=np.float64)
        right_ppm = np.full(mz.shape, np.inf, dtype=np.float64)

        if np.any(left_valid):
            t_left = targets_sorted[left_idx[left_valid]]
            left_ppm[left_valid] = np.abs(mz[left_valid] - t_left) / t_left * 1e6

        if np.any(right_valid):
            t_right = targets_sorted[right_idx[right_valid]]
            right_ppm[right_valid] = np.abs(mz[right_valid] - t_right) / t_right * 1e6

        choose_left = left_ppm <= right_ppm
        best_idx = np.where(choose_left, left_idx, right_idx)
        best_ppm = np.where(choose_left, left_ppm, right_ppm)

        matched = (best_idx >= 0) & (best_idx < n_targets) & (best_ppm <= ppm)
        if not np.any(matched):
            continue

        # ``best_idx`` indexes the sorted target array.  Map matched indices back
        # to the original target-column order *before* writing.  This is
        # mathematically equivalent to the previous final ``X[:, inv_order]``
        # operation, but avoids allocating a second full frame x feature matrix.
        tgt_idx = order[best_idx[matched]]
        tgt_int = inten[matched].astype(intensity_dtype, copy=False)

        if aggregate == "sum":
            np.add.at(X[row_idx], tgt_idx, tgt_int)

        elif aggregate == "max":
            s = np.argsort(tgt_idx)
            tgt_idx_s = tgt_idx[s]
            tgt_int_s = tgt_int[s]

            uniq, start = np.unique(tgt_idx_s, return_index=True)
            max_vals = np.maximum.reduceat(tgt_int_s, start)
            X[row_idx, uniq] = np.maximum(X[row_idx, uniq], max_vals)

        else:
            raise ValueError("aggregate must be 'sum' or 'max'")

    df = pd.DataFrame(X, index=frame_ids, columns=targets)
    df.index.name = "frame"

    rt_df = pd.DataFrame({"rt": rt_list}, index=frame_ids)

    return df, rt_df

def pack_specs(
        spec_list,
        reset_rt=True,
        rt_step=1.0
    ):
    exp = oms.MSExperiment()

    for i, spec in enumerate(spec_list):
        spec_copy = oms.MSSpectrum(spec)

        if reset_rt:
            spec_copy.setRT(i * rt_step)

        exp.addSpectrum(spec_copy)

    return exp

def save_spectra(spectra, output_path: str):
    exp = oms.MSExperiment()
    if isinstance(spectra, oms.MSSpectrum):
        exp.addSpectrum(spectra)
    elif isinstance(spectra, Sequence) and not isinstance(spectra, (str, bytes)):
        for i, spec in enumerate(spectra):
            exp.addSpectrum(spec)

    oms.MzMLFile().store(output_path, exp)