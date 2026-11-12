from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

import numpy as np
import pyopenms as oms


_DATA_DLL = "ThermoFisher.CommonCore.Data.dll"
_READER_DLL = "ThermoFisher.CommonCore.RawFileReader.dll"


def _assembly_directory() -> Path:
    configured = os.environ.get("SCMM_THERMO_RAW_DLL_DIR")
    directory = (
        Path(configured).expanduser()
        if configured
        else Path(sys.prefix) / "Library" / "bin" / "ThermoRawFileReader"
    )
    missing = [name for name in (_DATA_DLL, _READER_DLL) if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            "Thermo RawFileReader assemblies are unavailable. Run "
            "setup_thermo_raw.ps1 from the scmm Conda environment, or set "
            "SCMM_THERMO_RAW_DLL_DIR to the directory containing the official DLLs."
        )
    return directory.resolve()


def _load_api():
    directory = _assembly_directory()
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

    import clr

    clr.AddReference(str(directory / _DATA_DLL))
    clr.AddReference(str(directory / _READER_DLL))

    from ThermoFisher.CommonCore.Data.Business import Device
    from ThermoFisher.CommonCore.RawFileReader import RawFileReaderAdapter

    return RawFileReaderAdapter, Device


def _timestamp(value) -> float:
    try:
        return datetime(
            int(value.Year),
            int(value.Month),
            int(value.Day),
            int(value.Hour),
            int(value.Minute),
            int(value.Second),
        ).timestamp()
    except (AttributeError, TypeError, ValueError, OSError):
        return float("nan")


def load_thermo_raw(path: str, *, centroid_ms2=False):
    """Read a Thermo RAW file scan by scan with Thermo RawFileReader."""
    RawFileReaderAdapter, Device = _load_api()
    source = str(Path(path).resolve())
    raw_file = RawFileReaderAdapter.FileFactory(source)
    try:
        if not raw_file.IsOpen or raw_file.IsError:
            raise RuntimeError(f"Thermo RawFileReader could not open: {source}")
        if raw_file.InAcquisition:
            raise RuntimeError(f"Thermo RAW file is still being acquired: {source}")

        raw_file.SelectInstrument(Device.MS, 1)
        instrument = raw_file.GetInstrumentData()
        first_scan = int(raw_file.RunHeaderEx.FirstSpectrum)
        last_scan = int(raw_file.RunHeaderEx.LastSpectrum)
        exp = oms.MSExperiment()

        for scan_number in range(first_scan, last_scan + 1):
            scan_filter = raw_file.GetFilterForScanNumber(scan_number)
            if not str(scan_filter):
                continue
            statistics = raw_file.GetScanStatsForScanNumber(scan_number)
            is_centroid = bool(statistics.IsCentroidScan)
            use_ms2_centroids = centroid_ms2 and int(scan_filter.MSOrder) == 2
            if use_ms2_centroids or (is_centroid and str(statistics.SpectrumPacketType) == "FtCentroid"):
                stream = raw_file.GetCentroidStream(scan_number, False)
                mz = np.asarray(stream.Masses, dtype=np.float64)
                intensity = np.asarray(stream.Intensities, dtype=np.float32)
                if mz.size:
                    is_centroid = True
                elif use_ms2_centroids:
                    stream = raw_file.GetSegmentedScanFromScanNumber(scan_number, statistics)
                    mz = np.asarray(stream.Positions, dtype=np.float64)
                    intensity = np.asarray(stream.Intensities, dtype=np.float32)
            else:
                stream = raw_file.GetSegmentedScanFromScanNumber(scan_number, statistics)
                mz = np.asarray(stream.Positions, dtype=np.float64)
                intensity = np.asarray(stream.Intensities, dtype=np.float32)

            spectrum = oms.MSSpectrum()
            spectrum.setMSLevel(int(scan_filter.MSOrder))
            spectrum.setType(oms.SpectrumSettings.SpectrumType.CENTROID if is_centroid
                             else oms.SpectrumSettings.SpectrumType.PROFILE)
            if int(scan_filter.MSOrder) > 1:
                reaction = raw_file.GetScanEventForScanNumber(scan_number).GetReaction(0)
                precursor = oms.Precursor()
                precursor.setMZ(float(reaction.PrecursorMass))
                precursor.setActivationEnergy(float(reaction.CollisionEnergy))
                spectrum.setPrecursors([precursor])
            spectrum.setRT(float(raw_file.RetentionTimeFromScanNumber(scan_number)) * 60.0)
            spectrum.setNativeID(
                f"controllerType=0 controllerNumber=1 scan={scan_number}"
            )
            spectrum.setMetaValue("thermo_scan_number", scan_number)
            spectrum.set_peaks((mz, intensity))
            exp.addSpectrum(spectrum)

        model = str(instrument.Model or "")
        name = str(instrument.Name or "")
        exp_instrument = exp.getInstrument()
        exp_instrument.setName(model or name)
        exp.setInstrument(exp_instrument)
        metadata = {
            "timestamp": _timestamp(raw_file.FileHeader.CreationDate),
            "instrument": model or name,
            "raw_reader": "Thermo RawFileReader",
            "thermo_first_scan": first_scan,
            "thermo_last_scan": last_scan,
        }
        return exp, metadata
    finally:
        raw_file.Dispose()
