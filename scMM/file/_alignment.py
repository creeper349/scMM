"""Alignment of centroided spectrum peaks to a shared target m/z axis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyopenms as oms

from ._spectrum import extract_peaks


def align_frame(
    exp: oms.MSExperiment,
    mz_list,
    ppm: float = 10.0,
    ms_level: int = 1,
    aggregate: str = "max",
    dtype=np.float64,
    **kwargs,
):
    """Align each selected scan to target m/z values using nearest ppm matches."""
    _validate_alignment_options(ppm, aggregate)
    targets = np.asarray(mz_list, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0:
        raise ValueError("mz_list must be a non-empty 1D array-like.")
    sorted_targets, restore_order = _sorted_targets(targets)
    spectra, frame_ids, retention_times = _collect_spectra(exp, ms_level)
    if not spectra:
        raise ValueError("No spectra found.")
    values = np.zeros((len(spectra), len(targets)), dtype=np.float32)
    peak_options = _peak_options(kwargs)
    for row, spectrum in enumerate(spectra):
        mz, intensity = extract_peaks(spectrum, dtype=dtype, **peak_options)
        if mz.size == 0:
            continue
        mz, intensity = _sort_extracted_peaks(mz, intensity)
        target_indices, matched_intensity = _match_target_peaks(
            sorted_targets,
            mz,
            intensity,
            ppm,
        )
        _aggregate_target_intensity(values[row], target_indices, matched_intensity, aggregate)
    frame = pd.DataFrame(values[:, restore_order], index=frame_ids, columns=targets)
    frame.index.name = "frame"
    metadata = pd.DataFrame({"rt": retention_times}, index=frame_ids)
    return frame, metadata


def _validate_alignment_options(ppm: float, aggregate: str) -> None:
    if ppm < 0:
        raise ValueError("ppm must be non-negative")
    if aggregate not in {"sum", "max"}:
        raise ValueError("aggregate must be 'sum' or 'max'")


def _sorted_targets(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(targets)
    restore_order = np.empty_like(order)
    restore_order[order] = np.arange(order.size)
    return targets[order], restore_order


def _collect_spectra(experiment, ms_level: int):
    spectra, frame_ids, retention_times = [], [], []
    for frame_id, spectrum in enumerate(experiment):
        if spectrum.getMSLevel() != ms_level:
            continue
        spectra.append(spectrum)
        frame_ids.append(frame_id)
        retention_times.append(spectrum.getRT())
    return spectra, frame_ids, retention_times


def _peak_options(kwargs: dict) -> dict:
    return {
        "prominence_ratio": kwargs.get("prominence_ratio"),
        "distance": kwargs.get("distance", 3),
        "method": kwargs.get("method", "centroid"),
        "resolution_200": kwargs.get("resolution_200", 70000.0),
        "window_fwhm_factor": kwargs.get("window_fwhm_factor", 1.0),
        "centroid_intensity_mode": kwargs.get("centroid_intensity_mode", "apex"),
    }


def _sort_extracted_peaks(mz, intensity) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.size >= 2 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz)
        return mz[order], intensity[order]
    return mz, intensity


def _match_target_peaks(
    targets: np.ndarray,
    mz: np.ndarray,
    intensity: np.ndarray,
    ppm: float,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(targets, mz)
    left_indices, right_indices = positions - 1, positions
    left_ppm = _candidate_ppm(targets, mz, left_indices)
    right_ppm = _candidate_ppm(targets, mz, right_indices)
    choose_left = left_ppm <= right_ppm
    best_indices = np.where(choose_left, left_indices, right_indices)
    best_ppm = np.where(choose_left, left_ppm, right_ppm)
    matched = (best_indices >= 0) & (best_indices < len(targets)) & (best_ppm <= ppm)
    return best_indices[matched], intensity[matched].astype(np.float32)


def _candidate_ppm(targets: np.ndarray, mz: np.ndarray, indices: np.ndarray) -> np.ndarray:
    valid = (indices >= 0) & (indices < len(targets))
    errors = np.full(mz.shape, np.inf, dtype=np.float64)
    if np.any(valid):
        candidate_targets = targets[indices[valid]]
        errors[valid] = np.abs(mz[valid] - candidate_targets) / candidate_targets * 1e6
    return errors


def _aggregate_target_intensity(
    output: np.ndarray,
    target_indices: np.ndarray,
    intensity: np.ndarray,
    aggregate: str,
) -> None:
    if target_indices.size == 0:
        return
    if aggregate == "sum":
        np.add.at(output, target_indices, intensity)
        return
    order = np.argsort(target_indices)
    sorted_targets, sorted_intensity = target_indices[order], intensity[order]
    unique_targets, starts = np.unique(sorted_targets, return_index=True)
    output[unique_targets] = np.maximum(
        output[unique_targets],
        np.maximum.reduceat(sorted_intensity, starts),
    )
