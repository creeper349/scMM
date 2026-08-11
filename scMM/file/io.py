"""Mass-spectrometry file boundaries and stable public I/O exports."""

import logging
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pyopenms as oms

from ._alignment import align_frame
from ._spectrum import (
    _prepare_sorted_unique_peaks,
    build_orbitrap_grid,
    extract_peaks,
    orbitrap_fwhm_at_mz,
    orbitrap_resolution_at_mz,
    sum_spec,
)

logger = logging.getLogger(__name__)


def load_single_file(
    path: str | Path,
    format: Literal["auto", "mzML", "mzXML"] = "mzML",
) -> tuple[oms.MSExperiment, dict[str, Any]]:
    """Load an mzML/mzXML experiment and its acquisition metadata."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_format = _resolve_ms_format(path, format)
    experiment = oms.MSExperiment()
    if resolved_format == "mzML":
        oms.MzMLFile().load(str(path), experiment)
    else:
        oms.MzXMLFile().load(str(path), experiment)
    metadata = _file_metadata(path, experiment)
    logger.info("Loaded MS file from %s", path)
    return experiment, metadata


def _resolve_ms_format(path: Path, requested: str) -> str:
    if requested == "auto":
        formats = {".mzml": "mzML", ".mzxml": "mzXML"}
        try:
            return formats[path.suffix.lower()]
        except KeyError as exc:
            raise ValueError(f"Cannot infer MS format from extension: {path.suffix}") from exc
    if requested not in {"mzML", "mzXML"}:
        raise ValueError("format must be 'auto', 'mzML', or 'mzXML'")
    return requested


def _file_metadata(path: Path, experiment: oms.MSExperiment) -> dict[str, Any]:
    acquisition_time = experiment.getDateTime().get()
    try:
        timestamp = datetime.fromisoformat(acquisition_time).timestamp()
    except (TypeError, ValueError):
        timestamp = path.stat().st_mtime
        logger.warning(
            "Missing or invalid acquisition time in %s; using file modification time",
            path,
        )
    return {
        "name": path.stem,
        "timestamp": timestamp,
        "instrument": experiment.getInstrument().getName(),
    }


def sum_spectrum_from_file(
    path: str | Path,
    ms_level: int = 1,
    resolution_200: float = 35000.0,
    points_per_fwhm: float = 5.0,
) -> oms.MSSpectrum:
    """Load one MS file and return its summed spectrum."""
    experiment, _ = load_single_file(path, format="auto")
    return sum_spec(
        experiment,
        ms_level=ms_level,
        resolution_200=resolution_200,
        points_per_fwhm=points_per_fwhm,
    )


def pack_specs(spec_list, reset_rt=True, rt_step=1.0):
    """Copy spectra into an experiment, optionally assigning sequential RTs."""
    if rt_step <= 0:
        raise ValueError("rt_step must be positive")
    experiment = oms.MSExperiment()
    for index, spectrum in enumerate(spec_list):
        try:
            spectrum_copy = oms.MSSpectrum(spectrum)
        except (TypeError, ValueError) as exc:
            raise TypeError("spec_list must contain only MSSpectrum-compatible objects") from exc
        if reset_rt:
            spectrum_copy.setRT(index * rt_step)
        experiment.addSpectrum(spectrum_copy)
    return experiment


def save_spectra(spectra, output_path: str | Path) -> Path:
    """Save a spectrum or sequence of spectra as mzML."""
    experiment = oms.MSExperiment()
    if hasattr(spectra, "get_peaks") and hasattr(spectra, "getMSLevel"):
        _add_spectrum(experiment, spectra, "spectra must be MSSpectrum-compatible")
    elif isinstance(spectra, Sequence) and not isinstance(spectra, (str, bytes)):
        for spectrum in spectra:
            _add_spectrum(
                experiment,
                spectrum,
                "Every item in spectra must be MSSpectrum-compatible",
            )
    else:
        raise TypeError("spectra must be an MSSpectrum or a sequence of MSSpectrum objects")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    oms.MzMLFile().store(str(output), experiment)
    return output


def _add_spectrum(experiment: oms.MSExperiment, spectrum, error_message: str) -> None:
    try:
        experiment.addSpectrum(spectrum)
    except (TypeError, ValueError) as exc:
        raise TypeError(error_message) from exc


__all__ = [
    "_prepare_sorted_unique_peaks",
    "align_frame",
    "build_orbitrap_grid",
    "extract_peaks",
    "load_single_file",
    "orbitrap_fwhm_at_mz",
    "orbitrap_resolution_at_mz",
    "pack_specs",
    "save_spectra",
    "sum_spec",
    "sum_spectrum_from_file",
]
