import queue
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from .data import CyESIData
from typing import Literal, List, Union
from pathlib import Path

def _reader_worker(file_path:str, q_in:queue.Queue, **data_kwargs):
    data = CyESIData(file_path, **data_kwargs)
    q_in.put(data)

def _writer_worker(q_out:queue.Queue, result_dir = None, binary:bool = False):
    while True:
        item = q_out.get()
        if item is None:
            break
        data = item
        data.save(result_dir=result_dir, binary=binary)
        q_out.task_done()
        
def _mt(files, result_dir: str, data_kwargs: dict, preprocess_kwargs: dict, n_readers:int = 2, n_writers:int = 2, binary: bool = False):
    q_in = queue.Queue(maxsize=2 * n_readers)
    q_out = queue.Queue(maxsize=2 * n_writers)

    writers = []
    for _ in range(n_writers):
        t = ThreadPoolExecutor(max_workers=1)
        t.submit(_writer_worker, q_out, result_dir, binary)
        writers.append(t)

    with ThreadPoolExecutor(max_workers=n_readers) as readers:
        for f in files:
            readers.submit(_reader_worker, f, q_in, **data_kwargs)

        processed = 0
        total = len(files)
        while processed < total:
            data = q_in.get()
            data.preprocess(**preprocess_kwargs)
            q_out.put(data)
            processed += 1

    for _ in writers:
        q_out.put(None)
        
def _seq(files, result_dir: str, data_kwargs: dict, preprocess_kwargs: dict, binary: bool):
    for f in files:
        data = CyESIData(f, **data_kwargs)
        data.preprocess(**preprocess_kwargs)
        data.save(result_dir, binary)
        
def run_batch(
    input_dir: str, 
    result_dir: str,
    data_kwargs: dict,
    preprocess_kwargs: dict,
    method: Literal["sequential", "multithreading"] = "sequential",
    mt_kwargs: dict = None,
    binary:bool = False
):
    files = [
        os.path.join(input_dir, f) 
        for f in os.listdir(input_dir) 
        if (f.lower().endswith(".mzml")) or (f.lower().endswith(".mzxml"))
    ]
    if not files:
        raise ValueError(f"No mzML files found in {input_dir}")

    if method == "multithreading":
        if mt_kwargs is None: mt_kwargs = {}
        _mt(files, result_dir, data_kwargs, preprocess_kwargs, mt_kwargs.get("n_readers", 2),
            mt_kwargs.get("n_writers", 2), binary)
    elif method == "sequential":
        _seq(files, result_dir, data_kwargs, preprocess_kwargs, binary)
        
def align_batch(root_dir:str, result_dir:str, ppm_tol:int = 10, 
                unmatched:Literal['del', 'pad'] = 'del', 
                binary:bool = False,
                base: Union[int, str, None] = None):
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

    if len(results) < 2:
        raise ValueError("Need at least two valid directories to perform alignment")

    if base is None:
        base_idx = 0

    elif isinstance(base, int):
        if not (0 <= base < len(results)):
            raise IndexError(f"base index {base} out of range (0 ~ {len(results)-1})")
        base_idx = base

    elif isinstance(base, str):
        if base not in names:
            raise ValueError(f"base directory '{base}' not found in {names}")
        base_idx = names.index(base)

    else:
        raise TypeError("base must be int, str, or None")

    data = results[base_idx]

    for i, result in enumerate(results):
        if i == base_idx:
            continue
        data.alignwith(result, ppm_tol, unmatched)

    data.save(result_dir, binary)