"""Mass-spectrometry file boundaries and stable public I/O exports."""

import logging
import re
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

_XML_SCAN_BYTES = 64 * 1024
_MZML_SPECTRUM_TAG = re.compile(rb"<spectrum(?:\s|>)")
_MZXML_SCAN_TAG = re.compile(rb"<scan(?:\s|>)")


class InvalidMSFileError(ValueError):
    """Raised when a mass-spectrometry XML file cannot provide spectra."""


def load_single_file(
    path: str | Path,
    format: Literal["auto", "mzML", "mzXML"] = "auto",
) -> tuple[oms.MSExperiment, dict[str, Any]]:
    """Load an mzML/mzXML experiment and its acquisition metadata."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_format = _resolve_ms_format(path, format)
    validate_ms_file(path, format=resolved_format)
    experiment = oms.MSExperiment()
    try:
        if resolved_format == "mzML":
            oms.MzMLFile().load(str(path), experiment)
        else:
            oms.MzXMLFile().load(str(path), experiment)
    except RuntimeError as exc:
        raise InvalidMSFileError(
            f"无法读取 {path.name}: {resolved_format} 解析失败, 文件可能已截断或转换不完整; "
            "请从对应的原始数据重新转换后再试。"
        ) from exc
    if experiment.getNrSpectra() == 0:
        raise InvalidMSFileError(
            f"无法读取 {path.name}: 文件中没有质谱扫描; 请检查转换设置, "
            "并从对应的原始数据重新转换。"
        )
    metadata = _file_metadata(path, experiment)
    logger.info("Loaded MS file from %s", path)
    return experiment, metadata


def validate_ms_file(
    path: str | Path,
    format: Literal["auto", "mzML", "mzXML"] = "auto",
) -> None:
    """Quickly reject truncated XML files or files without spectrum records.

    The check reads only the document header, tail, and data up to the first
    spectrum tag. It is intended for interactive preflight; OpenMS remains the
    authoritative parser during :func:`load_single_file`.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved_format = _resolve_ms_format(path, format)
    try:
        with path.open("rb") as handle:
            header = handle.read(_XML_SCAN_BYTES)
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _XML_SCAN_BYTES))
            tail = handle.read()
    except OSError as exc:
        raise InvalidMSFileError(f"无法读取 {path.name}: {exc}") from exc

    if resolved_format == "mzML":
        indexed = re.search(rb"<indexedmzML(?:\s|>)", header) is not None
        root_tag = b"<indexedmzML" if indexed else b"<mzML"
        closing_tag = b"</indexedmzML>" if indexed else b"</mzML>"
        data_tag = _MZML_SPECTRUM_TAG
    else:
        root_tag = b"<mzXML"
        closing_tag = b"</mzXML>"
        data_tag = _MZXML_SCAN_TAG

    if root_tag not in header:
        raise InvalidMSFileError(
            f"无法读取 {path.name}: 内容不是有效的 {resolved_format} XML 文档。"
        )
    if closing_tag not in tail:
        raise InvalidMSFileError(
            f"无法读取 {path.name}: XML 文档不完整 (缺少 {closing_tag.decode()}); "
            "文件很可能在转换或复制时被截断, 请从对应的原始数据重新转换。"
        )
    if not _contains_data_tag(path, data_tag):
        raise InvalidMSFileError(
            f"无法读取 {path.name}: 文件中没有质谱扫描; 请检查转换设置, "
            "并从对应的原始数据重新转换。"
        )


def _contains_data_tag(path: Path, pattern: re.Pattern[bytes]) -> bool:
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(_XML_SCAN_BYTES):
            combined = overlap + chunk
            if pattern.search(combined):
                return True
            overlap = combined[-32:]
    return False


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
    "InvalidMSFileError",
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
    "validate_ms_file",
]
