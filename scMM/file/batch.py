"""Memory-bounded batch entry points for scMM raw-MS processing.

The current batch architecture builds one shared feature axis, then processes raw
files strictly one at a time and concatenates only cell-level results.  The
implementation lives in :meth:`CyESIData.load_from_filelist` so the returned
object remains fully compatible with the current plotting and downstream API.
"""
from __future__ import annotations

from pathlib import Path

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
        save_root = Path(save_root)
        if save_root.suffix.lower() == ".h5ad":
            target = save_root
        else:
            save_root.mkdir(parents=True, exist_ok=True)
            target = save_root / f"{Path(root_dir).name}.h5ad"
        obj.save(target)
    return obj
