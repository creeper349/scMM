"""Memory-bounded batch entry points for scMM raw-MS processing.

The current batch architecture builds one shared feature axis, then processes raw
files strictly one at a time and concatenates only cell-level results.  The
implementation lives in :meth:`CyESIData.load_from_filelist` so the returned
object remains fully compatible with the current plotting and downstream API.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Literal
import logging

from .data import CyESIData


def batch_process(root_dir, save_root=None, n_jobs=1, **kwargs):
    """Process a directory using the shared-axis/single-file batch workflow.

    Parameters
    ----------
    root_dir : path-like
        Directory containing mzML/mzXML files.
    save_root : path-like, optional
        If provided, save the combined processed :class:`CyESIData` using its
        existing storage API after processing.
    n_jobs : int, default=1
        Retained for compatibility. Raw-file loading/alignment is always
        sequential; values greater than one may be used by blockwise cell peak
        extraction.
    **kwargs
        Forwarded to :meth:`CyESIData.load_from_filelist`.

    Returns
    -------
    CyESIData
        Combined cell-level dataset compatible with the current plotting and
        downstream processing modules.
    """
    obj = CyESIData.load_from_filelist(
        str(root_dir),
        n_jobs=n_jobs,
        **kwargs,
    )
    if save_root is not None:
        Path(save_root).mkdir(parents=True, exist_ok=True)
        obj.save(str(save_root))
    return obj


def concat(
    root_dir,
    save_path=None,
    ppm_tol: float = 5.0,
    ref_idx: int = 0,
    mz_merge_options: Literal["union", "ref"] = "union",
):
    """Concatenate already processed CyESIData directories.

    This compatibility helper keeps the current ``CyESIData.alignwith`` logic.
    It is not used by the raw-MS batch workflow, which now concatenates at cell
    level internally before returning.
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(root_dir)

    results: List[CyESIData] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not (sub / ".meta").exists():
            continue
        try:
            results.append(CyESIData(str(sub)))
        except Exception as exc:
            logging.warning("Skipping processed directory %s: %s", sub, exc)

    if not results:
        raise FileNotFoundError(f"No processed CyESIData directories found in {root_dir}")
    if not (0 <= ref_idx < len(results)):
        raise IndexError("ref_idx is out of range.")

    result = results[ref_idx]
    for i, other in enumerate(results):
        if i == ref_idx:
            continue
        result.alignwith(other, ppm_tol=ppm_tol, mz_merge_options=mz_merge_options)

    if save_path is not None:
        target = Path(save_path)
        target.mkdir(parents=True, exist_ok=True)
        result.save(str(target))
    return result
