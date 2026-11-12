from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def chromatogram_preview(
    path,
    reference_mz=(),
    *,
    ppm=10.0,
    msconvert_path=None,
    progress_callback=None,
):
    """Return TIC and EIC traces from all MS1 spectra in one raw-MS file."""
    from scMM.file.io import load_single_file

    exp, metadata = load_single_file(
        str(path), format="auto", msconvert_path=msconvert_path
    )
    spectra = [
        (frame_id, spec)
        for frame_id, spec in enumerate(exp)
        if spec.getMSLevel() == 1
    ]
    references = np.atleast_1d(reference_mz).astype(float)
    rows = []
    total = max(1, len(spectra))
    for index, (frame_id, spec) in enumerate(spectra):
        mz, intensity = spec.get_peaks()
        mz = np.asarray(mz, dtype=float)
        intensity = np.asarray(intensity, dtype=float)
        if mz.size > 1 and np.any(np.diff(mz) < 0):
            order = np.argsort(mz)
            mz, intensity = mz[order], intensity[order]
        row = {
            "scan": index,
            "frame_id": frame_id,
            "rt": float(spec.getRT()),
            "tic": float(np.nansum(intensity)),
        }
        for target in references:
            tolerance = abs(float(target)) * float(ppm) * 1e-6
            left = int(np.searchsorted(mz, target - tolerance, side="left"))
            right = int(np.searchsorted(mz, target + tolerance, side="right"))
            row[f"eic_{target:g}"] = float(np.nansum(intensity[left:right]))
        rows.append(row)
        if progress_callback is not None and (index % max(1, total // 100) == 0):
            progress_callback((index + 1) / total, f"读取 MS1 帧 {index + 1}/{total}")
    return pd.DataFrame(rows), metadata


def spectrum_preview(path, scan_index=0, *, msconvert_path=None):
    """Return one MS1 spectrum and its retention time."""
    from scMM.file.io import load_single_file

    exp, metadata = load_single_file(
        str(path), format="auto", msconvert_path=msconvert_path
    )
    spectra = [spec for spec in exp if spec.getMSLevel() == 1]
    if not spectra:
        raise ValueError("No MS1 spectra were found")
    index = min(max(0, int(scan_index)), len(spectra) - 1)
    mz, intensity = spectra[index].get_peaks()
    frame = pd.DataFrame(
        {"mz": np.asarray(mz, dtype=float), "intensity": np.asarray(intensity, dtype=float)}
    )
    return frame, float(spectra[index].getRT()), len(spectra), metadata


def cell_detection_preview(
    path,
    reference_mz,
    *,
    reference_mode="union",
    msconvert_path=None,
    progress_callback=None,
    **kwargs,
):
    """Run the production cell detector and return reference traces for preview."""
    from scMM.file.data import CyESIData

    captured = {}

    def hook(stage, payload):
        if stage != "find_cells":
            return
        signal = payload["signal"]
        references = np.asarray(payload["reference_mz"], dtype=float)
        columns = np.asarray(signal.columns, dtype=float)
        frame = pd.DataFrame(index=signal.index)
        frame_obs = payload.get("frame_obs")
        frame["x"] = (
            frame_obs.loc[signal.index, "rt"].to_numpy(dtype=float)
            if frame_obs is not None and "rt" in frame_obs
            else np.arange(len(signal), dtype=float)
        )
        for target in references:
            column = int(np.abs(columns - target).argmin())
            frame[f"m/z {target:g}"] = signal.iloc[:, column].to_numpy(dtype=float)
        frame["cell_window"] = np.asarray(payload["cell_mask"], dtype=bool)
        frame["cell_apex"] = False
        frame.iloc[np.asarray(payload["cell_idx"], dtype=int), frame.columns.get_loc("cell_apex")] = True
        captured["frame"] = frame

    data = CyESIData.load_from_file(
        Path(path),
        reference_mz=reference_mz,
        reference_mode=reference_mode,
        msconvert_path=msconvert_path,
        debug_hook=hook,
        show_progress=False,
        progress_callback=progress_callback,
        **kwargs,
    )
    if "frame" not in captured:
        raise RuntimeError("Cell preview was not produced")
    return captured["frame"], data
