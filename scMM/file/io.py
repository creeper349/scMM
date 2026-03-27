from typing import Literal, Tuple, Dict, Any
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
import pyopenms as oms
import pandas as pd
import numpy as np
import os

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

    return exp, metadata

def sum_spec(
        exp: oms.MSExperiment,
        mz_range=(100.0, 1000.0),
        ppm=5.0,
        ms_level=1,
    ):

    mz_min, mz_max = mz_range
    ratio = 1.0 + ppm * 1e-6
    n_bins = int(np.floor(np.log(mz_max / mz_min) / np.log(ratio))) + 1
    mz_grid = mz_min * (ratio ** np.arange(n_bins, dtype=np.float64))

    acc = np.zeros_like(mz_grid, dtype=np.float64)
    total_spectra = 0

    for spec in exp:
        if spec.getMSLevel() != ms_level:
            continue

        mz, inten = spec.get_peaks()
        mz = np.asarray(mz, dtype=np.float64)
        inten = np.asarray(inten, dtype=np.float64)

        if mz.size == 0:
            continue

        mask = (mz >= mz_min) & (mz <= mz_max)
        if not np.any(mask):
            continue

        mz = mz[mask]
        inten = inten[mask]

        if np.any(np.diff(mz) < 0):
            order = np.argsort(mz)
            mz = mz[order]
            inten = inten[order]

        if mz.size >= 2:
            uniq = np.empty(mz.size, dtype=bool)
            uniq[0] = True
            uniq[1:] = mz[1:] > mz[:-1]
            mz = mz[uniq]
            inten = inten[uniq]

        if mz.size == 0:
            continue
        idx = np.rint(np.log(mz / mz_min) / np.log(ratio)).astype(np.int64)
        valid = (idx >= 0) & (idx < acc.size)
        if np.any(valid):
            np.add.at(acc, idx[valid], inten[valid])

        total_spectra += 1

    if total_spectra == 0:
        raise ValueError("No spectra found.")

    avg_intensity = acc
    spec_out = oms.MSSpectrum()
    spec_out.setMSLevel(ms_level)
    spec_out.setRT(0.0)

    spec_out.set_peaks((
        mz_grid.astype(np.float64),
        avg_intensity.astype(np.float32)
    ))
    spec_out.setMetaValue("n_averaged_spectra", int(total_spectra))
    spec_out.setMetaValue("ppm", float(ppm))
    spec_out.setMetaValue("mz_min", float(mz_min))
    spec_out.setMetaValue("mz_max", float(mz_max))

    return spec_out

def sum_spectrum_from_file(
        path: str,
        ms_level: int = 1,
        mz_binning_width: float = 5.0,
    ) -> tuple[oms.MSSpectrum, int]:
    exp = oms.MSExperiment()
    oms.MzMLFile().load(path, exp)
    return sum_spec(
        exp,
        ms_level=ms_level,
        ppm=mz_binning_width
    )

def extract_peaks(index:int, spec: oms.MSSpectrum, dtype = np.float64,
                  prominence_ratio: float = None, distance:int = 3) -> Tuple[np.ndarray, np.ndarray]:
    mz = np.array(spec.get_peaks()[0], dtype=dtype)
    intensity = np.array(spec.get_peaks()[1], dtype=dtype)
    
    if len(mz) == 0:
        return spec.getRT(), np.array([]), np.array([])

    peak_idx, _ = find_peaks(intensity, 
                    prominence=np.max(intensity) * prominence_ratio if prominence_ratio is not None else None,
                    distance=distance)
    peak_mz = mz[peak_idx]
    peak_intensity = intensity[peak_idx]

    return index, spec.getRT(), peak_mz, peak_intensity

def align_frame(
    exp: oms.MSExperiment,
    mz_list,
    ppm: float = 10.0,
    ms_level: int = 1,
    aggregate: str = "sum",
    dtype=np.float64,
    **kwargs
):

    targets = np.asarray(mz_list, dtype=np.float64)
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

    for row_idx, (i, spec) in enumerate(spectra):

        _, rt, mz, inten = extract_peaks(i, spec, dtype=dtype,
            prominence_ratio=kwargs.get("prominence_ratio", None), distance=kwargs.get("distance", 3))

        if mz.size == 0:
            continue

        if mz.size >= 2 and np.any(np.diff(mz) < 0):
            idx = np.argsort(mz)
            mz = mz[idx]
            inten = inten[idx]

        pos = np.searchsorted(targets_sorted, mz)

        left_idx = pos - 1
        right_idx = pos

        left_valid = left_idx >= 0
        right_valid = right_idx < n_targets

        left_ppm = np.full(mz.shape, np.inf)
        right_ppm = np.full(mz.shape, np.inf)

        if np.any(left_valid):
            t = targets_sorted[left_idx[left_valid]]
            left_ppm[left_valid] = np.abs(mz[left_valid] - t) / t * 1e6

        if np.any(right_valid):
            t = targets_sorted[right_idx[right_valid]]
            right_ppm[right_valid] = np.abs(mz[right_valid] - t) / t * 1e6

        choose_left = left_ppm <= right_ppm
        best_idx = np.where(choose_left, left_idx, right_idx)
        best_ppm = np.where(choose_left, left_ppm, right_ppm)

        matched = (best_idx >= 0) & (best_idx < n_targets) & (best_ppm <= ppm)
        if not np.any(matched):
            continue

        tgt_idx = best_idx[matched]
        tgt_int = inten[matched]

        if aggregate == "sum":
            np.add.at(X[row_idx], tgt_idx, tgt_int.astype(np.float32))

        elif aggregate == "max":
            s = np.argsort(tgt_idx)
            tgt_idx_s = tgt_idx[s]
            tgt_int_s = tgt_int[s]

            uniq, start = np.unique(tgt_idx_s, return_index=True)
            max_vals = np.maximum.reduceat(tgt_int_s, start)
            X[row_idx, uniq] = np.maximum(X[row_idx, uniq], max_vals.astype(np.float32))

        else:
            raise ValueError("aggregate must be 'sum' or 'max'")

    X = X[:, inv_order]

    df = pd.DataFrame(X, index=frame_ids, columns=targets)
    df.index.name = "frame"

    rt_df = pd.DataFrame({
        "rt": rt_list
    }, index=frame_ids)

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