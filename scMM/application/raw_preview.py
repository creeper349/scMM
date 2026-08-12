"""Raw mass-spectrometry summaries for interactive previews."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms

from scMM.file.io import load_single_file, sum_spec

from .storage import StorageCatalog


@dataclass(frozen=True)
class RawFileSummary:
    """Compact metadata shown before a processing workflow starts."""

    name: str
    path: Path
    size_bytes: int
    scan_count: int
    scans_by_ms_level: tuple[tuple[int, int], ...]
    rt_min_seconds: float | None
    rt_max_seconds: float | None
    mz_min: float | None
    mz_max: float | None
    instrument: str


class RawFilePreview:
    """A loaded raw file with reusable TIC, EIC, and spectrum calculations."""

    def __init__(
        self,
        path: Path,
        experiment: oms.MSExperiment,
        metadata: dict[str, object],
    ) -> None:
        self.path = path
        self.experiment = experiment
        self.metadata = metadata
        self.summary = _summarize(path, experiment, metadata)

    @property
    def ms_levels(self) -> tuple[int, ...]:
        """Return the MS levels represented in the file."""
        return tuple(level for level, _ in self.summary.scans_by_ms_level)

    def total_ion_chromatogram(
        self,
        *,
        ms_level: int = 1,
        rt_range: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        """Return total ion intensity for each selected scan."""
        _validate_ms_level(ms_level)
        rows = []
        for scan_index, spectrum in _selected_spectra(self.experiment, ms_level, rt_range):
            _, intensity = spectrum.get_peaks()
            rows.append(
                {
                    "scan_index": scan_index,
                    "rt_seconds": float(spectrum.getRT()),
                    "intensity": float(np.nansum(intensity)),
                }
            )
        return pd.DataFrame(rows, columns=["scan_index", "rt_seconds", "intensity"])

    def extracted_ion_chromatogram(
        self,
        target_mz: float,
        *,
        ppm_tolerance: float = 5.0,
        ms_level: int = 1,
        rt_range: tuple[float, float] | None = None,
    ) -> pd.DataFrame:
        """Sum signal inside a target-centered ppm window for each scan."""
        if not np.isfinite(target_mz) or target_mz <= 0:
            raise ValueError("target_mz must be a positive finite number")
        if not np.isfinite(ppm_tolerance) or ppm_tolerance <= 0:
            raise ValueError("ppm_tolerance must be a positive finite number")
        _validate_ms_level(ms_level)
        delta = target_mz * ppm_tolerance * 1e-6
        lower, upper = target_mz - delta, target_mz + delta
        rows = []
        for scan_index, spectrum in _selected_spectra(self.experiment, ms_level, rt_range):
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, dtype=np.float64)
            intensity = np.asarray(intensity, dtype=np.float64)
            valid = np.isfinite(mz) & np.isfinite(intensity)
            selected_intensity = intensity[valid & (mz >= lower) & (mz <= upper)]
            rows.append(
                {
                    "scan_index": scan_index,
                    "rt_seconds": float(spectrum.getRT()),
                    "intensity": float(selected_intensity.sum()),
                }
            )
        return pd.DataFrame(rows, columns=["scan_index", "rt_seconds", "intensity"])

    def summed_spectrum(
        self,
        *,
        mz_range: tuple[float, float],
        resolution_200: float = 35_000.0,
        points_per_fwhm: float = 3.0,
        ms_level: int = 1,
        rt_range: tuple[float, float] | None = None,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Return a summed or average profile spectrum over selected scans."""
        _validate_ms_level(ms_level)
        selected = oms.MSExperiment()
        for _, spectrum in _selected_spectra(self.experiment, ms_level, rt_range):
            selected.addSpectrum(spectrum)
        spectrum = sum_spec(
            selected,
            mz_range=mz_range,
            resolution_200=resolution_200,
            points_per_fwhm=points_per_fwhm,
            ms_level=ms_level,
            normalize=normalize,
        )
        mz, intensity = spectrum.get_peaks()
        return pd.DataFrame(
            {
                "mz": np.asarray(mz, dtype=np.float64),
                "intensity": np.asarray(intensity, dtype=np.float64),
            }
        )

    def binned_spectrum(
        self,
        *,
        mz_range: tuple[float, float],
        bins: int = 20_000,
        ms_level: int = 1,
        rt_range: tuple[float, float] | None = None,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Return a fast fixed-bin preview spectrum without high-resolution interpolation."""
        _validate_ms_level(ms_level)
        mz_min, mz_max = map(float, mz_range)
        if not np.isfinite(mz_min) or not np.isfinite(mz_max) or mz_max <= mz_min:
            raise ValueError("mz_range must contain finite values in ascending order")
        if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 1_000_000:
            raise ValueError("bins must be an integer between 2 and 1000000")
        accumulated = np.zeros(bins, dtype=np.float64)
        selected_count = 0
        scale = bins / (mz_max - mz_min)
        for _, spectrum in _selected_spectra(self.experiment, ms_level, rt_range):
            selected_count += 1
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, dtype=np.float64)
            intensity = np.asarray(intensity, dtype=np.float64)
            valid = np.isfinite(mz) & np.isfinite(intensity) & (mz >= mz_min) & (mz <= mz_max)
            selected_mz = mz[valid]
            if selected_mz.size == 0:
                continue
            indices = np.floor((selected_mz - mz_min) * scale).astype(np.int64)
            np.minimum(indices, bins - 1, out=indices)
            np.add.at(accumulated, indices, intensity[valid])
        if selected_count == 0:
            raise ValueError("No spectra found.")
        if normalize:
            accumulated /= selected_count
        width = (mz_max - mz_min) / bins
        centers = mz_min + (np.arange(bins, dtype=np.float64) + 0.5) * width
        return pd.DataFrame({"mz": centers, "intensity": accumulated})


class RawPreviewService:
    """Open raw files only after validating them against a storage catalog."""

    def __init__(self, storage: StorageCatalog) -> None:
        self.storage = storage

    def open(self, root_label: str, selected_path: str | Path) -> RawFilePreview:
        """Load one validated raw file for an interactive session."""
        path = self.storage.resolve_raw_file(root_label, selected_path)
        experiment, metadata = load_single_file(path)
        return RawFilePreview(path, experiment, metadata)


def _summarize(
    path: Path,
    experiment: oms.MSExperiment,
    metadata: dict[str, object],
) -> RawFileSummary:
    levels: Counter[int] = Counter()
    retention_times: list[float] = []
    mz_min: float | None = None
    mz_max: float | None = None
    for spectrum in experiment:
        levels[int(spectrum.getMSLevel())] += 1
        retention_times.append(float(spectrum.getRT()))
        mz, _ = spectrum.get_peaks()
        finite_mz = np.asarray(mz, dtype=np.float64)
        finite_mz = finite_mz[np.isfinite(finite_mz)]
        if finite_mz.size:
            local_min, local_max = float(finite_mz.min()), float(finite_mz.max())
            mz_min = local_min if mz_min is None else min(mz_min, local_min)
            mz_max = local_max if mz_max is None else max(mz_max, local_max)
    return RawFileSummary(
        name=str(metadata.get("name", path.stem)),
        path=path,
        size_bytes=path.stat().st_size,
        scan_count=int(experiment.getNrSpectra()),
        scans_by_ms_level=tuple(sorted(levels.items())),
        rt_min_seconds=min(retention_times) if retention_times else None,
        rt_max_seconds=max(retention_times) if retention_times else None,
        mz_min=mz_min,
        mz_max=mz_max,
        instrument=str(metadata.get("instrument", "")),
    )


def _validate_ms_level(ms_level: int) -> None:
    if ms_level < 1:
        raise ValueError("ms_level must be at least 1")


def _selected_spectra(
    experiment: oms.MSExperiment,
    ms_level: int,
    rt_range: tuple[float, float] | None,
):
    if rt_range is not None:
        rt_min, rt_max = map(float, rt_range)
        if not np.isfinite(rt_min) or not np.isfinite(rt_max) or rt_max < rt_min:
            raise ValueError("rt_range must contain finite values in ascending order")
    else:
        rt_min = rt_max = None
    for scan_index, spectrum in enumerate(experiment):
        rt = float(spectrum.getRT())
        if spectrum.getMSLevel() != ms_level:
            continue
        if rt_min is not None and not (rt_min <= rt <= rt_max):
            continue
        yield scan_index, spectrum
