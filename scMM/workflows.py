from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


ProgressCallback = Callable[[float, str], None]


def parse_reference_mz(text):
    values = [
        float(value.strip())
        for value in str(text).replace(";", ",").split(",")
        if value.strip()
    ]
    if not values:
        raise ValueError("At least one reference m/z is required")
    return values


def load_metadata_csv(path):
    if not path:
        return None
    table = pd.read_csv(path)
    if "source_file" not in table:
        raise ValueError("Metadata CSV must contain source_file")
    return {
        str(row["source_file"]): {
            key: value
            for key, value in row.items()
            if key != "source_file" and pd.notna(value)
        }
        for _, row in table.iterrows()
    }


@dataclass
class ProcessingConfig:
    input_path: str
    output_path: str
    reference_mz: list[float]
    reference_mode: str = "union"
    feature_strategy: str = "shared"
    feature_merge_ppm: float = 10.0
    ppm_tol: float = 10.0
    mz_min: float = 100.0
    mz_max: float = 1000.0
    resolution: float = 35000.0
    resample_points_per_fwhm: float = 5.0
    ms_peak_snr_threshold: float = 10.0
    prominence_ratio: float | None = None
    peak_distance: int = 3
    cell_threshold: float = 5.0
    feature_snr: float = 3.0
    baseline_window: int = 51
    noise_window: int = 51
    max_zero_fraction: float = 0.9
    feature_block_size: int = 256
    metadata_path: str = ""
    frame_metadata: tuple[str, ...] = ()
    create_qc: bool = True
    msconvert_path: str = ""


def run_processing(
    config: ProcessingConfig,
    *,
    progress_callback: ProgressCallback | None = None,
):
    """Extract a CyESI dataset and report GUI-friendly progress."""
    from .analysis import render_quality_report
    from .file.data import CyESIData

    def report(fraction, message):
        if progress_callback is not None:
            progress_callback(float(fraction), str(message))

    source = Path(config.input_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(source)
    mz_min = float(config.mz_min)
    mz_max = float(config.mz_max)
    if not (0 < mz_min < mz_max):
        raise ValueError("m/z range must satisfy 0 < mz_min < mz_max")
    outside = [
        float(value)
        for value in config.reference_mz
        if not mz_min < float(value) < mz_max
    ]
    if outside:
        raise ValueError(
            "Reference m/z must lie strictly inside the processing range "
            f"({mz_min:g}, {mz_max:g}); outside/boundary values: {outside}"
        )
    report(0.01, f"检查输入路径: {source}")
    kwargs = {
        "reference_mz": config.reference_mz,
        "reference_mode": config.reference_mode,
        "frame_metadata": list(config.frame_metadata),
        "metadata_by_file": load_metadata_csv(config.metadata_path),
        "ppm_tol": config.ppm_tol,
        "mz_range": (mz_min, mz_max),
        "resolution": config.resolution,
        "resample_points_per_fwhm": config.resample_points_per_fwhm,
        "ms_peak_snr_threshold": config.ms_peak_snr_threshold,
        "prominence_ratio": config.prominence_ratio,
        "distance": config.peak_distance,
        "cell_signal_threshold": config.cell_threshold,
        "feature_snr_threshold": config.feature_snr,
        "baseline_window": config.baseline_window,
        "noise_window": config.noise_window,
        "max_zero_fraction": config.max_zero_fraction,
        "feature_block_size": config.feature_block_size,
        "msconvert_path": config.msconvert_path or None,
        "progress_callback": lambda value, message: report(
            0.05 + 0.8 * value, message
        ),
    }
    if source.is_dir() and source.suffix.lower() != ".raw":
        data = CyESIData.load_from_directory(
            source,
            feature_strategy=config.feature_strategy,
            feature_merge_ppm=config.feature_merge_ppm,
            **kwargs,
        )
    else:
        data = CyESIData.load_from_file(source, **kwargs)
    report(0.88, "写入 H5AD 结果")
    output = data.save(config.output_path)
    qc_output = None
    if config.create_qc:
        report(0.94, "生成质量控制 PDF")
        qc_output = output.with_name(f"{output.stem}_quality.pdf")
        render_quality_report(data, qc_output)
    report(1.0, f"处理完成: {data.n_obs} cells × {data.n_vars} features")
    return data, output, qc_output
