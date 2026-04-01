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
    spec_out.set_peaks((
        mz_grid.astype(np.float64),
        out_intensity.astype(np.float32)
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
    ) -> tuple[oms.MSSpectrum, int]:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path, exp)
    return sum_spec(
        exp,
        ms_level=ms_level,
        resolution_200=resolution_200,
        points_per_fwhm=points_per_fwhm
    )

def extract_peaks(
    spec: oms.MSSpectrum,
    dtype=np.float64,
    prominence_ratio: float = None,
    distance: int = 3,
    method: str = "centroid",    
    resolution_200: float = 35000.0, 
    window_fwhm_factor: float = 1.0,   
    centroid_intensity_mode: str = "apex" 
) -> Tuple[np.ndarray, np.ndarray]:

    mz = np.asarray(spec.get_peaks()[0], dtype=dtype)
    intensity = np.asarray(spec.get_peaks()[1], dtype=dtype)

    if mz.size == 0:
        return np.array([], dtype=dtype), np.array([], dtype=dtype)

    # Ensure sorted
    if mz.size > 1 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]

    prom = None
    if prominence_ratio is not None:
        if intensity.size == 0 or np.max(intensity) <= 0:
            return np.array([], dtype=dtype), np.array([], dtype=dtype)
        prom = np.max(intensity) * prominence_ratio

    peak_idx, _ = find_peaks(
        intensity,
        prominence=prom,
        distance=distance
    )

    if peak_idx.size == 0:
        return np.array([], dtype=dtype), np.array([], dtype=dtype)

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

    return np.asarray(peak_mz_out, dtype=dtype), np.asarray(peak_int_out, dtype=dtype)

def align_frame(
        exp: oms.MSExperiment,
        mz_list,
        ppm: float = 10.0,
        ms_level: int = 1,
        aggregate: str = "max",   # "sum" | "max"
        dtype=np.float64,
        **kwargs
    ):

    targets = np.asarray(mz_list, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("mz_list must be a non-empty 1D array-like.")

    order = np.argsort(targets)
    targets_sorted = targets[order]
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

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

    X = np.zeros((n_frames, n_targets), dtype=np.float32)

    for row_idx, (frame_id, spec) in enumerate(spectra):

        mz, inten = extract_peaks(
            spec,
            dtype=dtype,
            prominence_ratio=kwargs.get("prominence_ratio", None),
            distance=kwargs.get("distance", 3),
            method=kwargs.get("method", "centroid"),
            resolution_200=kwargs.get("resolution_200", 70000.0),
            window_fwhm_factor=kwargs.get("window_fwhm_factor", 1.0),
            centroid_intensity_mode=kwargs.get("centroid_intensity_mode", "apex"),
        )

        if mz.size == 0:
            continue

        mz = np.asarray(mz, dtype=np.float64)
        inten = np.asarray(inten, dtype=np.float64)

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

        tgt_idx = best_idx[matched]
        tgt_int = inten[matched].astype(np.float32)

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

    X = X[:, inv_order]

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