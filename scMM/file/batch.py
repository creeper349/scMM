from .data import CyESIData
from joblib import Parallel, delayed
from typing import Literal, List
from pathlib import Path
from ..file.data import CyESIData
import os
import logging

def batch_process(root_dir, save_root, n_jobs = -1, prefer = None, **kwargs):
    files = os.listdir(root_dir)
    filelist = []
    for file in files:
        full_path = os.path.join(root_dir, file)
        if (not os.path.isdir(full_path)) and file.lower().endswith((".mzml", ".mzxml")):
            filelist.append(full_path)
    results = Parallel(n_jobs=n_jobs, prefer=prefer)(
        delayed(CyESIData.load_from_file)(file, **kwargs) for file in filelist)
    for result, file in zip(results, filelist):
        save_path = os.path.join(save_root)
        result.save(save_path)
        
def concat(root_dir, save_path, ppm_tol:float = 5.0, ref_idx = 0, mz_merge_options: Literal["union", "ref"] = "union"):
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(root_dir)

    results: List[CyESIData] = []
    names: List[str] = []

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue

        try:
            obj = CyESIData.load_from_processed(str(sub))
            results.append(obj)
            names.append(sub.name)
        except Exception as e:
            logging.info(f"Failed to load directory {sub}: {e}")
            continue
        
    data = results[ref_idx]

    for i, result in enumerate(results):
        if i == ref_idx:
            continue
        data.alignwith(result, ppm_tol, mz_merge_options)

    data.save(save_path)