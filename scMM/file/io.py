from typing import Literal, Tuple, Dict, Any
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from tqdm import tqdm
import pyopenms as oms
import pandas as pd
import numpy as np
import os

def load_file(
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

    return exp, metadata

def _extract_peaks(index:int, spec: oms.MSSpectrum, dtype = np.float64) -> Tuple[np.ndarray, np.ndarray]:
    mz = np.array(spec.get_peaks()[0], dtype=dtype)
    intensity = np.array(spec.get_peaks()[1], dtype=dtype)
    
    if len(mz) == 0:
        return spec.getRT(), np.array([]), np.array([])

    peak_idx, _ = find_peaks(intensity)
    peak_mz = mz[peak_idx]
    peak_intensity = intensity[peak_idx]

    return index, spec.getRT(), peak_mz, peak_intensity

def _merge_mz(all_mz:list, all_intensity:list, ppm_tol:int, dtype=np.float64) -> pd.DataFrame:
    n_frame = len(all_mz)

    frame_idx = np.concatenate([i*np.ones_like(mz, dtype=np.int32) for i, mz in enumerate(all_mz)])
    all_mz = np.concatenate(all_mz).astype(dtype)
    all_intensity = np.concatenate(all_intensity).astype(dtype)

    order = np.argsort(all_mz)
    all_mz = all_mz[order]
    all_intensity = all_intensity[order]
    frame_idx = frame_idx[order]

    groups = np.zeros_like(all_mz, dtype=np.int32)
    group_id = 0
    group_sum = all_mz[0]
    group_count = 1

    for i in range(1, len(all_mz)):
        avg = group_sum / group_count
        ppm = (all_mz[i] - avg) / avg * 1e6
        if ppm <= ppm_tol:
            group_sum += all_mz[i]
            group_count += 1
        else:
            group_id += 1
            group_sum = all_mz[i]
            group_count = 1
        groups[i] = group_id

    n_groups = groups.max() + 1

    merged_mz = np.bincount(groups, weights=all_mz) / np.bincount(groups)
    merged_mz = merged_mz.astype(dtype)

    intensity_matrix = np.zeros((n_frame, n_groups), dtype=dtype)

    for g in range(n_groups):
        mask = groups == g
        frames = frame_idx[mask]
        ints = all_intensity[mask]
        for f, val in zip(frames, ints):
            intensity_matrix[f, g] += val

    return intensity_matrix, merged_mz

def _post_merge_features(
    intensity: np.ndarray,
    mz: np.ndarray, 
    ppm_tol: float = 10,
    comp_thresh: float = 0.9,
    dtype=np.float64,
):
    n_feat = len(mz)
    used = np.zeros(n_feat, dtype=bool)

    new_mz = []
    new_intensity = []

    for i in range(n_feat):
        if used[i]:
            continue

        group = [i]
        used[i] = True

        for j in range(i + 1, n_feat):
            if used[j]:
                continue

            ppm = abs(mz[i] - mz[j]) / ((mz[i] + mz[j]) / 2) * 1e6
            if ppm > ppm_tol:
                break

            nz_i = intensity[:, i] > 0
            nz_j = intensity[:, j] > 0

            union = np.sum(nz_i | nz_j)
            if union == 0:
                continue

            intersection = np.sum(nz_i & nz_j)
            complementarity = 1 - intersection / union

            if complementarity >= comp_thresh:
                group.append(j)
                used[j] = True

        group_int = intensity[:, group]
        total_int = group_int.sum(axis=0)
        mz_weighted = np.sum(mz[group] * total_int) / np.sum(total_int)

        new_mz.append(mz_weighted)
        new_intensity.append(group_int.sum(axis=1))

    new_intensity = np.vstack(new_intensity).T.astype(dtype)
    new_mz = np.array(new_mz, dtype=dtype)

    return new_intensity, new_mz

def frame_concat(exp: oms.MSExperiment, ppm_tol: int = 10, zero_filter: float = 0.99, dtype = np.float64) -> pd.DataFrame: 
    
    results = []
    for i, spec in tqdm(enumerate(exp), desc="Loading spectra"):
        results.append(_extract_peaks(i, spec, dtype=dtype))

    results = sorted(results, key=lambda x: x[0])
    frame_index = np.array([res[0] for res in results], dtype=int)
    rts = np.array([res[1] for res in results], dtype=dtype)
    all_mz = [res[2] for res in results]
    all_int = [res[3] for res in results]
    
    intensity_matrix, merged_mz = _merge_mz(all_mz, all_int, ppm_tol, dtype=dtype)
    intensity_matrix, merged_mz = _post_merge_features(intensity_matrix, merged_mz, ppm_tol, comp_thresh=0.9, dtype=dtype)
    zero_ratio = np.sum(intensity_matrix == 0, axis=0) / intensity_matrix.shape[0]
    keep_mask = zero_ratio < zero_filter
    intensity_matrix = intensity_matrix[:, keep_mask]
    merged_mz = merged_mz[keep_mask]
    df = pd.DataFrame(intensity_matrix, columns=merged_mz, index=frame_index)
    
    return df, pd.DataFrame({"RT": rts })