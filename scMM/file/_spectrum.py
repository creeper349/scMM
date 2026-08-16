"""Orbitrap spectrum grids, accumulation, and peak refinement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyopenms as oms
from scipy.signal import find_peaks


@dataclass(frozen=True)
class _PeakConfig:
    method: str
    resolution_200: float
    window_fwhm_factor: float
    centroid_intensity_mode: str


def orbitrap_resolution_at_mz(mz: float, resolution_200: float) -> float:
    if mz <= 0 or resolution_200 <= 0:
        raise ValueError("mz and resolution_200 must be positive")
    return resolution_200 * np.sqrt(200.0 / mz)


def orbitrap_fwhm_at_mz(mz: float, resolution_200: float) -> float:
    return mz / orbitrap_resolution_at_mz(mz, resolution_200)


def build_orbitrap_grid(
    mz_range=(100.0, 1000.0),
    resolution_200: float = 70000.0,
    points_per_fwhm: float = 5.0,
) -> np.ndarray:
    """Build a variable-width grid at a fixed number of points per FWHM."""
    mz_min, mz_max = map(float, mz_range)
    if mz_min <= 0 or mz_max <= mz_min:
        raise ValueError("Invalid mz_range.")
    if resolution_200 <= 0:
        raise ValueError("resolution_200 must be positive.")
    if points_per_fwhm <= 0:
        raise ValueError("points_per_fwhm must be positive.")
    grid = [mz_min]
    current_mz = mz_min
    while current_mz < mz_max:
        spacing = orbitrap_fwhm_at_mz(current_mz, resolution_200) / points_per_fwhm
        if spacing <= 0 or not np.isfinite(spacing):
            raise ValueError("Invalid grid spacing encountered.")
        current_mz += spacing
        grid.append(current_mz)
    result = np.asarray(grid, dtype=np.float64)
    if result[-1] > mz_max:
        result[-1] = mz_max
    elif result[-1] < mz_max:
        result = np.append(result, mz_max)
    return result


def _prepare_sorted_unique_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove non-finite peaks, sort by m/z, and sum duplicate positions."""
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    valid = np.isfinite(mz) & np.isfinite(intensity)
    mz, intensity = mz[valid], intensity[valid]
    if mz.size == 0:
        return mz, intensity
    if np.any(np.diff(mz) < 0):
        order = np.argsort(mz)
        mz, intensity = mz[order], intensity[order]
    unique_mz, inverse = np.unique(mz, return_inverse=True)
    if unique_mz.size != mz.size:
        combined = np.zeros_like(unique_mz, dtype=np.float64)
        np.add.at(combined, inverse, intensity)
        return unique_mz, combined
    return mz, intensity


def sum_spec(
    exp: oms.MSExperiment,
    mz_range=(100.0, 1000.0),
    resolution_200: float = 35000.0,
    points_per_fwhm: float = 5.0,
    ms_level: int = 1,
    normalize: bool = False,
    zero_outside: bool = True,
):
    """Interpolate and sum spectra on a variable-resolution Orbitrap grid."""
    if ms_level < 1:
        raise ValueError("ms_level must be at least 1")
    mz_min, mz_max = map(float, mz_range)
    grid = build_orbitrap_grid(
        mz_range=(mz_min, mz_max),
        resolution_200=resolution_200,
        points_per_fwhm=points_per_fwhm,
    )
    intensity, count = _accumulate_spectra(exp, grid, mz_min, mz_max, ms_level, zero_outside)
    if count == 0:
        raise ValueError("No spectra found.")
    if normalize:
        intensity = intensity / count
    return _summed_spectrum(
        grid,
        intensity,
        count,
        ms_level,
        resolution_200,
        points_per_fwhm,
        mz_min,
        mz_max,
        normalize,
    )


def _accumulate_spectra(
    experiment,
    grid: np.ndarray,
    mz_min: float,
    mz_max: float,
    ms_level: int,
    zero_outside: bool,
) -> tuple[np.ndarray, int]:
    accumulated = np.zeros_like(grid, dtype=np.float64)
    count = 0
    for spectrum in experiment:
        if spectrum.getMSLevel() != ms_level:
            continue
        mz, intensity = _prepare_sorted_unique_peaks(*spectrum.get_peaks())
        if mz.size == 0 or mz[-1] < mz_min or mz[0] > mz_max:
            continue
        if zero_outside:
            interpolated = np.interp(grid, mz, intensity, left=0.0, right=0.0)
        else:
            interpolated = np.interp(grid, mz, intensity)
        accumulated += interpolated
        count += 1
    return accumulated, count


def _summed_spectrum(
    grid,
    intensity,
    count: int,
    ms_level: int,
    resolution_200: float,
    points_per_fwhm: float,
    mz_min: float,
    mz_max: float,
    normalized: bool,
) -> oms.MSSpectrum:
    spectrum = oms.MSSpectrum()
    spectrum.setMSLevel(ms_level)
    spectrum.setRT(0.0)
    spectrum.set_peaks((grid.astype(np.float64), intensity.astype(np.float32)))
    metadata = {
        "n_summed_spectra": int(count),
        "resolution_200": float(resolution_200),
        "points_per_fwhm": float(points_per_fwhm),
        "mz_min": float(mz_min),
        "mz_max": float(mz_max),
        "grid_type": "orbitrap_variable_fwhm",
        "interpolation": "linear",
        "normalized": int(bool(normalized)),
    }
    for key, value in metadata.items():
        spectrum.setMetaValue(key, value)
    return spectrum


def extract_peaks(
    spec: oms.MSSpectrum,
    dtype=np.float64,
    prominence_ratio: float | None = None,
    distance: int = 3,
    method: str = "centroid",
    resolution_200: float = 35000.0,
    window_fwhm_factor: float = 1.0,
    centroid_intensity_mode: str = "apex",
) -> tuple[np.ndarray, np.ndarray]:
    """Detect local maxima and refine their m/z and intensity estimates."""
    config = _PeakConfig(method, resolution_200, window_fwhm_factor, centroid_intensity_mode)
    _validate_peak_options(config, prominence_ratio, distance)
    mz, intensity = spec.get_peaks()
    mz, intensity = _sorted_spectrum_arrays(mz, intensity, dtype)
    if mz.size == 0:
        return _empty_peaks(dtype)
    prominence = _peak_prominence(intensity, prominence_ratio)
    if prominence is False:
        return _empty_peaks(dtype)
    peak_indices, _ = find_peaks(intensity, prominence=prominence, distance=distance)
    if peak_indices.size == 0:
        return _empty_peaks(dtype)
    refined = [_refine_peak(mz, intensity, int(index), config) for index in peak_indices]
    peak_mz, peak_intensity = zip(*refined, strict=True)
    return np.asarray(peak_mz, dtype=dtype), np.asarray(peak_intensity, dtype=dtype)


def _validate_peak_options(config: _PeakConfig, prominence_ratio, distance: int) -> None:
    if config.method not in {"centroid", "parabola"}:
        raise ValueError("method must be 'centroid' or 'parabola'")
    if config.centroid_intensity_mode not in {"apex", "sum"}:
        raise ValueError("centroid_intensity_mode must be 'apex' or 'sum'")
    if distance < 1:
        raise ValueError("distance must be at least 1")
    if prominence_ratio is not None and prominence_ratio < 0:
        raise ValueError("prominence_ratio must be non-negative")
    if config.window_fwhm_factor <= 0:
        raise ValueError("window_fwhm_factor must be positive")


def _sorted_spectrum_arrays(mz, intensity, dtype) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(mz, dtype=dtype)
    intensity = np.asarray(intensity, dtype=dtype)
    if mz.size > 1 and np.any(np.diff(mz) < 0):
        order = np.argsort(mz)
        mz, intensity = mz[order], intensity[order]
    return mz, intensity


def _peak_prominence(intensity: np.ndarray, ratio: float | None):
    if ratio is None:
        return None
    if intensity.size == 0 or np.max(intensity) <= 0:
        return False
    return np.max(intensity) * ratio


def _empty_peaks(dtype) -> tuple[np.ndarray, np.ndarray]:
    return np.array([], dtype=dtype), np.array([], dtype=dtype)


def _refine_peak(
    mz: np.ndarray,
    intensity: np.ndarray,
    index: int,
    config: _PeakConfig,
) -> tuple[float, float]:
    peak_window = _local_peak_window(mz, intensity, index, config)
    if peak_window is None:
        return mz[index], intensity[index]
    window_mz, window_intensity = peak_window
    if config.method == "centroid":
        return _centroid_peak(
            mz[index],
            intensity[index],
            window_mz,
            window_intensity,
            config.centroid_intensity_mode,
        )
    return _parabolic_peak(mz, intensity, index)


def _local_peak_window(mz, intensity, index: int, config: _PeakConfig):
    local_spacing = _local_mz_spacing(mz, index)
    if not np.isfinite(local_spacing) or local_spacing <= 0:
        return None
    fwhm = orbitrap_fwhm_at_mz(float(mz[index]), config.resolution_200)
    half_width = max(1, int(np.ceil(config.window_fwhm_factor * fwhm / local_spacing)))
    left = max(0, index - half_width)
    right = min(len(mz), index + half_width + 1)
    if left == right:
        return None
    return mz[left:right], intensity[left:right]


def _local_mz_spacing(mz: np.ndarray, index: int) -> float:
    if index == 0:
        return float(mz[1] - mz[0]) if len(mz) > 1 else np.nan
    if index == len(mz) - 1:
        return float(mz[-1] - mz[-2])
    return float((mz[index + 1] - mz[index - 1]) / 2.0)


def _centroid_peak(apex_mz, apex_intensity, mz, intensity, intensity_mode: str):
    total = np.sum(intensity)
    centroid_mz = apex_mz if total <= 0 else np.sum(mz * intensity) / total
    centroid_intensity = apex_intensity if intensity_mode == "apex" else total
    return centroid_mz, centroid_intensity


def _parabolic_peak(mz: np.ndarray, intensity: np.ndarray, index: int):
    if index == 0 or index == len(mz) - 1:
        return mz[index], intensity[index]
    x1, x2, x3 = float(mz[index - 1]), float(mz[index]), float(mz[index + 1])
    y1, y2, y3 = (float(intensity[position]) for position in (index - 1, index, index + 1))
    left, right = x1 - x2, x3 - x2
    denominator = left * (left - right) * -right
    if denominator == 0:
        return mz[index], intensity[index]
    curvature = (right * (y2 - y1) + left * (y3 - y2)) / denominator
    slope = (right**2 * (y1 - y2) + left**2 * (y2 - y3)) / denominator
    if curvature >= 0:
        return mz[index], intensity[index]
    vertex = -slope / (2.0 * curvature)
    if vertex < left or vertex > right:
        return mz[index], intensity[index]
    return x2 + vertex, curvature * vertex**2 + slope * vertex + y2
