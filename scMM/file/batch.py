"""Batch workflows for processed scMM datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from joblib import Parallel, delayed

from .data import CyESIData

logger = logging.getLogger(__name__)


def batch_process(
    root_dir: str | Path,
    save_root: str | Path,
    n_jobs: int = -1,
    prefer: str | None = None,
    *,
    overwrite: bool = False,
    **kwargs,
) -> list[Path]:
    """Process each mzML/mzXML file in a directory and save each dataset."""
    root = Path(root_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)
    filelist = sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".mzml", ".mzxml"}
    )
    if not filelist:
        raise FileNotFoundError(f"No mzML or mzXML files found in {root}")
    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(CyESIData.load_from_file)(file, **kwargs) for file in filelist
    )
    return [result.save(save_root, overwrite=overwrite) for result in results]


def concat(
    root_dir: str | Path,
    save_path: str | Path,
    ppm_tol: float = 5.0,
    ref_idx: int = 0,
    mz_merge_options: Literal["union", "ref"] = "union",
    *,
    overwrite: bool = False,
) -> CyESIData:
    """Concatenate all saved datasets directly below ``root_dir``."""
    root = Path(root_dir).expanduser()
    if not root.is_dir():
        raise NotADirectoryError(root)

    dataset_dirs = sorted(
        sub for sub in root.iterdir() if sub.is_dir() and (sub / ".meta").is_file()
    )
    if not dataset_dirs:
        raise FileNotFoundError(f"No processed scMM datasets found in {root}")
    results = [CyESIData.load_from_processed(sub) for sub in dataset_dirs]
    if not -len(results) <= ref_idx < len(results):
        raise IndexError(f"ref_idx {ref_idx} is out of range for {len(results)} datasets")
    data = results[ref_idx]

    for i, result in enumerate(results):
        if i == ref_idx:
            continue
        data.alignwith(result, ppm_tol=ppm_tol, mz_merge_options=mz_merge_options)

    data.save(save_path, overwrite=overwrite)
    logger.info("Concatenated %d datasets", len(results))
    return data
