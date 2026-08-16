"""Processed and raw-file loading workflows for :class:`CyESIData`."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..util.peak import filter_spectrum
from .io import (
    align_frame,
    extract_peaks,
    load_single_file,
    pack_specs,
    sum_spec,
    sum_spectrum_from_file,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetState:
    """Frames and provenance needed to initialize a dataset container."""

    data: pd.DataFrame
    peak_meta: pd.DataFrame
    file_meta: dict[str, Any]
    ref_mz: float | None
    feature_meta: pd.DataFrame | None = None


@dataclass(frozen=True)
class _RawLoadConfig:
    ref_mz: float
    dtype: Any
    ppm_tol: int
    resolution: float
    points_per_fwhm: float
    snr_threshold: float
    prominence_ratio: float | None
    distance: int


def load_processed_dataset(result_dir: str | Path) -> DatasetState:
    """Read a processed dataset, preferring pickle frames over CSV."""
    result_path = Path(result_dir).expanduser()
    with (result_path / ".meta").open(encoding="utf-8") as handle:
        file_meta = json.load(handle)
    data, peak_meta, feature_meta = _read_processed_frames(result_path)
    if data is None:
        raise FileNotFoundError(f"No processed data found in {result_path}")
    if peak_meta is None:
        peak_meta = pd.DataFrame(index=data.index.copy())
    feature_meta = _normalize_feature_metadata(data, feature_meta)
    return DatasetState(data, peak_meta, file_meta, file_meta.get("ref_mz"), feature_meta)


def _read_processed_frames(path: Path):
    if (path / "data.pkl").exists():
        return (
            pd.read_pickle(path / "data.pkl"),
            _read_optional_pickle(path / "peak_meta.pkl"),
            _read_optional_pickle(path / "feature_meta.pkl"),
        )
    return (
        _read_optional_csv(path / "data.csv"),
        _read_optional_csv(path / "peak_meta.csv"),
        _read_optional_csv(path / "feature_meta.csv"),
    )


def _read_optional_pickle(path: Path):
    return pd.read_pickle(path) if path.exists() else None


def _read_optional_csv(path: Path):
    return pd.read_csv(path, index_col=0) if path.exists() else None


def _normalize_feature_metadata(
    data: pd.DataFrame,
    feature_meta: pd.DataFrame | None,
) -> pd.DataFrame:
    if feature_meta is None:
        return make_feature_metadata(data)
    if len(feature_meta) != data.shape[1]:
        raise ValueError(
            "feature_meta row count does not match the number of data features: "
            f"{len(feature_meta)} != {data.shape[1]}"
        )
    result = feature_meta.copy()
    result.index = data.columns.copy()
    result.index.name = "feature_id"
    return result


def make_feature_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """Create the minimal feature table for an aligned data frame."""
    metadata = pd.DataFrame({"mz": data.columns.astype(float)}, index=data.columns.copy())
    metadata.index.name = "feature_id"
    return metadata


def load_raw_file(
    file_path: str | Path,
    ref_mz: float,
    *,
    dtype=np.float64,
    ppm_tol: int = 10,
    resolution: float = 35000,
    resample_points_per_fwhm: float = 5.0,
    ms_peak_snr_threshold: float = 10.0,
    prominence_ratio: float | None = None,
    distance: int = 3,
) -> DatasetState:
    """Load and align one raw file without performing cell preprocessing."""
    config = _raw_config(
        ref_mz,
        dtype,
        ppm_tol,
        resolution,
        resample_points_per_fwhm,
        ms_peak_snr_threshold,
        prominence_ratio,
        distance,
    )
    experiment, file_meta = load_single_file(str(file_path), format="auto")
    targets = _pick_common_targets(experiment, config, dtype=config.dtype)
    data, peak_meta = align_frame(experiment, targets, config.ppm_tol, dtype=config.dtype)
    _annotate_single_file_frames(peak_meta, file_meta)
    file_meta["ref_mz"] = ref_mz
    return DatasetState(data, peak_meta, file_meta, ref_mz)


def load_raw_directory(
    dir_path: str | Path,
    ref_mz: float,
    *,
    dtype=np.float64,
    ppm_tol: int = 10,
    resolution: float = 35000,
    resample_points_per_fwhm: float = 5.0,
    ms_peak_snr_threshold: float = 10.0,
    prominence_ratio: float | None = None,
    n_jobs: int = -1,
    distance: int = 3,
) -> DatasetState:
    """Load and align a directory of raw files without cell preprocessing."""
    config = _raw_config(
        ref_mz,
        dtype,
        ppm_tol,
        resolution,
        resample_points_per_fwhm,
        ms_peak_snr_threshold,
        prominence_ratio,
        distance,
    )
    directory = Path(dir_path).expanduser()
    files = discover_ms_files(directory)
    logger.info("Detected %d MS files in %s", len(files), directory)
    targets = _directory_targets(files, config, n_jobs)
    alignments = Parallel(n_jobs=n_jobs)(
        delayed(_align_frame_from_file)(path, targets, config.ppm_tol, config.dtype)
        for path in files
    )
    return combine_aligned_files(directory.name, ref_mz, alignments)


def _raw_config(
    ref_mz,
    dtype,
    ppm_tol,
    resolution,
    points_per_fwhm,
    snr_threshold,
    prominence_ratio,
    distance,
) -> _RawLoadConfig:
    if not np.isfinite(ref_mz) or ref_mz <= 0:
        raise ValueError("ref_mz must be a positive finite number")
    return _RawLoadConfig(
        ref_mz,
        dtype,
        ppm_tol,
        resolution,
        points_per_fwhm,
        snr_threshold,
        prominence_ratio,
        distance,
    )


def discover_ms_files(directory: Path) -> list[str]:
    """Return sorted direct mzML/mzXML children of a directory."""
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    files = sorted(
        str(path)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".mzml", ".mzxml"}
    )
    if not files:
        raise FileNotFoundError(f"No mzML or mzXML files found in {directory}")
    return files


def _pick_common_targets(experiment, config: _RawLoadConfig, *, dtype=None) -> np.ndarray:
    summed = sum_spec(
        experiment,
        resolution_200=config.resolution,
        points_per_fwhm=config.points_per_fwhm,
    )
    denoised = filter_spectrum(summed, snr_threshold=config.snr_threshold)
    peak_options = {
        "prominence_ratio": config.prominence_ratio,
        "distance": config.distance,
    }
    if dtype is not None:
        peak_options["dtype"] = dtype
    targets, _ = extract_peaks(denoised, **peak_options)
    return targets


def _directory_targets(files: list[str], config: _RawLoadConfig, n_jobs: int) -> np.ndarray:
    logger.info(
        "Summing spectra (resolution=%s, points_per_fwhm=%s)",
        config.resolution,
        config.points_per_fwhm,
    )
    spectra = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(sum_spectrum_from_file)(
            path,
            resolution_200=config.resolution,
            points_per_fwhm=config.points_per_fwhm,
        )
        for path in files
    )
    return _pick_common_targets(pack_specs(spectra), config)


def _align_frame_from_file(file_path: str, targets, ppm_tol: int, dtype) -> dict:
    experiment, file_meta = load_single_file(file_path, format="auto")
    logger.info("Aligning frames from MS file %s", file_path)
    data, peak_meta = align_frame(experiment, targets, ppm_tol, dtype=dtype)
    return {"file_meta": file_meta, "data": data, "peak_meta": peak_meta}


def _annotate_single_file_frames(peak_meta: pd.DataFrame, file_meta: dict) -> None:
    max_rt = float(peak_meta["rt"].max())
    peak_meta["time"] = peak_meta["rt"] / max_rt if max_rt > 0 else 0.0
    peak_meta["label"] = file_meta["name"].split(".")[0]


def combine_aligned_files(
    directory_name: str,
    ref_mz: float,
    alignments: list[dict],
) -> DatasetState:
    """Order aligned files by acquisition time and concatenate their frames."""
    alignments = sorted(alignments, key=lambda item: item["file_meta"]["timestamp"])
    start, elapsed = _acquisition_span(alignments)
    data_frames, metadata_frames, source_metadata = [], [], []
    for alignment in alignments:
        data = alignment["data"].copy()
        peak_meta = alignment["peak_meta"].copy()
        file_meta = dict(alignment["file_meta"])
        peak_meta["time"] = (peak_meta["rt"] + file_meta["timestamp"] - start) / elapsed
        peak_meta["label"] = file_meta["name"].split(".")[0]
        data_frames.append(data)
        metadata_frames.append(peak_meta)
        source_metadata.append(file_meta)
    return DatasetState(
        data=pd.concat(data_frames, axis=0),
        peak_meta=pd.concat(metadata_frames, axis=0),
        file_meta={
            "name": directory_name,
            "ref_mz": ref_mz,
            "per_file_meta": source_metadata,
        },
        ref_mz=ref_mz,
    )


def _acquisition_span(alignments: list[dict]) -> tuple[float, float]:
    start = alignments[0]["file_meta"]["timestamp"]
    last = alignments[-1]
    end = last["file_meta"]["timestamp"] + last["peak_meta"]["rt"].iloc[-1]
    elapsed = end - start
    if elapsed <= 0:
        raise ValueError("Acquisition timestamps must span a positive duration")
    return start, elapsed
