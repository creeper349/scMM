"""Public CyESI dataset container assembled from focused capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Self

import numpy as np

from ._dataset_interop import DatasetInteropMixin
from ._dataset_loading import (
    DatasetState,
    load_processed_dataset,
    load_raw_directory,
    load_raw_file,
    make_feature_metadata,
)
from ._dataset_processing import DatasetProcessingMixin


class CyESIData(DatasetProcessingMixin, DatasetInteropMixin):
    """Processed CyESI dataset and its observation/feature metadata.

    Construction lives here, while processing and interoperability methods are
    grouped into private mixins.  The public class and method-chain API remain
    stable for callers.
    """

    def __init__(self, result_dir: str | Path):
        self._apply_state(load_processed_dataset(result_dir))

    @classmethod
    def load_from_processed(cls, result_dir: str | Path) -> Self:
        """Load a dataset previously written by :meth:`save`."""
        return cls(result_dir)

    @classmethod
    def load_from_file(
        cls,
        file_path: str | Path,
        ref_mz: float,
        dtype=np.float64,
        ppm_tol: int = 10,
        resolution: float = 35000,
        resample_points_per_fwhm: float = 5.0,
        ms_peak_snr_threshold: float = 10.0,
        prominence_ratio: float | None = None,
        distance: int = 3,
        **preprocess_kwds,
    ) -> Self:
        """Load, align, and preprocess one mzML/mzXML file."""
        state = load_raw_file(
            file_path,
            ref_mz,
            dtype=dtype,
            ppm_tol=ppm_tol,
            resolution=resolution,
            resample_points_per_fwhm=resample_points_per_fwhm,
            ms_peak_snr_threshold=ms_peak_snr_threshold,
            prominence_ratio=prominence_ratio,
            distance=distance,
        )
        return cls._from_raw_state(state, preprocess_kwds)

    @classmethod
    def load_from_filelist(
        cls,
        dir_path: str | Path,
        ref_mz: float,
        dtype=np.float64,
        ppm_tol: int = 10,
        resolution: float = 35000,
        resample_points_per_fwhm: float = 5.0,
        ms_peak_snr_threshold: float = 10.0,
        prominence_ratio: float | None = None,
        n_jobs: int = -1,
        distance: int = 3,
        **preprocess_kwds,
    ) -> Self:
        """Load and combine all direct mzML/mzXML children of a directory."""
        state = load_raw_directory(
            dir_path,
            ref_mz,
            dtype=dtype,
            ppm_tol=ppm_tol,
            resolution=resolution,
            resample_points_per_fwhm=resample_points_per_fwhm,
            ms_peak_snr_threshold=ms_peak_snr_threshold,
            prominence_ratio=prominence_ratio,
            n_jobs=n_jobs,
            distance=distance,
        )
        return cls._from_raw_state(state, preprocess_kwds)

    @classmethod
    def _from_raw_state(cls, state: DatasetState, preprocess_kwds: dict) -> Self:
        obj = object.__new__(cls)
        obj._apply_state(state)
        obj.preprocess(**preprocess_kwds)
        obj.feature_meta = make_feature_metadata(obj.data)
        return obj

    def _apply_state(self, state: DatasetState) -> None:
        self.data = state.data
        self.peak_meta = state.peak_meta
        self.file_meta = state.file_meta
        self.ref_mz = state.ref_mz
        if state.feature_meta is not None:
            self.feature_meta = state.feature_meta
