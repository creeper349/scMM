import numpy as np
import os
import json
import pandas as pd
import pyopenms as oms
import logging
import math
import gc
import ctypes
from pathlib import Path
from joblib import Parallel, delayed, effective_n_jobs
from tqdm.auto import tqdm
from anndata import AnnData

from .io import (load_single_file, 
                 sum_spec, 
                 extract_peaks, 
                 align_frame)
from ..util.peak import filter_spectrum, find_cell_peaks
from ..util.normalize import normalize
from ..util.annotation import SDFMzSearcher, DEFAULT_ADDUCTS_NEG, DEFAULT_ADDUCTS_POS

from typing import Callable, Optional, Dict, Any, Self, Literal, Hashable
from scipy.ndimage import median_filter, grey_opening
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import IsolationForest

DebugHook = Callable[[str, Dict[str, Any]], None]


def _log_memory_usage(message: str) -> None:
    """Log process RSS/peak RSS and Linux cgroup memory usage.

    This helper intentionally has no external dependency such as ``psutil``.  It
    reads Linux ``/proc`` and cgroup-v2 files directly and silently degrades when
    those files are unavailable.
    """

    def _to_gib(value: int) -> float:
        return value / (1024 ** 3)

    rss_bytes = None
    hwm_bytes = None
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss_bytes = int(line.split()[1]) * 1024
                elif line.startswith("VmHWM:"):
                    hwm_bytes = int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass

    cgroup_current = None
    cgroup_limit = None
    try:
        path = Path("/sys/fs/cgroup/memory.current")
        if path.exists():
            cgroup_current = int(path.read_text().strip())
        else:
            # cgroup v1 fallback
            path = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            if path.exists():
                cgroup_current = int(path.read_text().strip())
    except (OSError, ValueError):
        pass

    try:
        path = Path("/sys/fs/cgroup/memory.max")
        if path.exists():
            value = path.read_text().strip()
            if value != "max":
                cgroup_limit = int(value)
        else:
            # cgroup v1 fallback
            path = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            if path.exists():
                cgroup_limit = int(path.read_text().strip())

        # Some cgroup-v1 systems encode "unlimited" as a huge integer.
        if cgroup_limit is not None and cgroup_limit >= (1 << 60):
            cgroup_limit = None
    except (OSError, ValueError):
        pass

    parts = [f"[MEM] {message}"]
    if rss_bytes is not None:
        parts.append(f"process RSS={_to_gib(rss_bytes):.2f} GiB")
    if hwm_bytes is not None:
        parts.append(f"process peak={_to_gib(hwm_bytes):.2f} GiB")
    if cgroup_current is not None:
        parts.append(f"cgroup current={_to_gib(cgroup_current):.2f} GiB")
    if cgroup_limit is not None:
        parts.append(f"cgroup limit={_to_gib(cgroup_limit):.2f} GiB")
        if cgroup_current is not None and cgroup_limit > 0:
            parts.append(f"cgroup usage={100.0 * cgroup_current / cgroup_limit:.1f}%")

    logging.info(" | ".join(parts))


def _release_memory_to_os() -> None:
    """Release Python garbage and return releasable glibc heap pages on Linux.

    The call is best-effort and has no effect on the numerical workflow.  It is
    particularly useful after destroying large pyOpenMS/native objects between
    sequential raw-file processing steps.
    """
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _validate_float_dtype(dtype):
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        raise TypeError("dtype must be a floating-point dtype.")
    return dtype

def _align_frame_from_file(
    file_path: str,
    mz_list,
    ppm_tol: int = 10,
    dtype=np.float32,
    log_memory: bool = True,
    out=None,
):
    """Align one raw file, optionally directly into a caller-owned buffer.

    When ``out`` is provided, ``align_frame`` writes directly into that array.
    Only frame indices and retention-time metadata are returned, so the caller
    does not retain a second file-sized aligned matrix.
    """
    dtype = _validate_float_dtype(dtype)
    exp, file_meta = load_single_file(file_path, format="auto")
    if log_memory:
        _log_memory_usage(f"After loading {Path(file_path).name} for alignment")
    logging.info(f"Aligning frames from MS file {file_path}...")
    data, peak_meta = align_frame(
        exp, mz_list, ppm_tol, dtype=dtype, out=out
    )
    if log_memory:
        stage = "directly into global matrix" if out is not None else "into local matrix"
        _log_memory_usage(f"After aligning {Path(file_path).name} {stage}")

    # The DataFrame returned by align_frame is only a lightweight wrapper around
    # ``out`` in direct-write mode. Keep only its frame index before releasing it.
    data_index = np.asarray(data.index).copy()
    if out is not None:
        del data

    # Explicitly release the native OpenMS experiment before returning metadata.
    del exp
    gc.collect()
    if log_memory:
        _log_memory_usage(f"After releasing raw experiment {Path(file_path).name}")

    result = {
        "file_meta": file_meta,
        "data_index": data_index,
        "peak_meta": peak_meta,
    }
    if out is None:
        result["data"] = data
    return result

class CyESIData:
    def __init__(self, result_dir:str):
        with open(os.path.join(result_dir, ".meta"), 'r') as fp:
            file_meta = json.load(fp)

        data_path_pkl = os.path.join(result_dir, "data.pkl")
        peak_path_pkl = os.path.join(result_dir, "peak_meta.pkl")
        feature_path_pkl = os.path.join(result_dir, "feature_meta.pkl")
        data = None
        peak_meta = None
        feature_meta = None
        if os.path.exists(data_path_pkl):
            data = pd.read_pickle(data_path_pkl)
            if os.path.exists(peak_path_pkl):
                peak_meta = pd.read_pickle(peak_path_pkl)
            if os.path.exists(feature_path_pkl):
                feature_meta = pd.read_pickle(feature_path_pkl)
        else:
            data_path_csv = os.path.join(result_dir, "data.csv")
            peak_path_csv = os.path.join(result_dir, "peak_meta.csv")
            feature_path_csv = os.path.join(result_dir, "feature_meta.csv")
            if os.path.exists(data_path_csv):
                data = pd.read_csv(data_path_csv, index_col=0)
            if os.path.exists(peak_path_csv):
                peak_meta = pd.read_csv(peak_path_csv, index_col=0)
            if os.path.exists(feature_path_csv):
                feature_meta = pd.read_csv(feature_path_csv, index_col=0)

        if data is None:
            raise FileNotFoundError(f"No processed data found in {result_dir}")
        self.data = data
        self.peak_meta = peak_meta
        self.file_meta = file_meta
        self.feature_meta = feature_meta

    @classmethod
    def load_from_file(cls, file_path:str,
                    ref_mz: Optional[float] = None, 
                    dtype = np.float32,
                    ppm_tol: int = 10,
                    resolution: float = 35000,
                    resample_points_per_fwhm: float = 5.0,
                    ms_peak_snr_threshold: float = 10.0,
                    prominence_ratio: float = None,
                    distance:int = 3,
                    log_memory: bool = True,
                    **preprocess_kwds):
        dtype = _validate_float_dtype(dtype)
        obj = object.__new__(cls)
        exp, obj.file_meta = load_single_file(file_path, format='auto')
        if log_memory:
            _log_memory_usage(f"After loading {Path(file_path).name}")

        sum_ = sum_spec(
            exp,
            resolution_200=resolution,
            points_per_fwhm=resample_points_per_fwhm,
            intensity_dtype=dtype,
        )
        if log_memory:
            _log_memory_usage(f"After summing {Path(file_path).name}")
        obj.file_meta["ref_mz"] = ref_mz
        
        sum_ = filter_spectrum(sum_, snr_threshold=ms_peak_snr_threshold, dtype=dtype)
        mz_list, _ = extract_peaks(sum_, dtype=dtype, prominence_ratio=prominence_ratio, distance=distance)
        obj.data, obj.peak_meta = align_frame(exp, mz_list, ppm_tol, dtype=dtype)
        if log_memory:
            _log_memory_usage(f"After aligning {Path(file_path).name}")

        # Raw spectra and the summed spectrum are no longer needed during cell
        # extraction; releasing them here lowers the preprocessing peak RSS.
        del exp, sum_
        gc.collect()
        if log_memory:
            _log_memory_usage(f"After releasing raw MS objects {Path(file_path).name}")

        obj.peak_meta["time"] = obj.peak_meta["rt"] / np.max(obj.peak_meta["rt"])
        obj.peak_meta["label"] = [obj.file_meta["name"].split(".")[0]] * len(obj.peak_meta)
        obj.ref_mz = ref_mz
        preprocess_kwds.setdefault("dtype", dtype)
        obj.preprocess(**preprocess_kwds)
        if log_memory:
            _log_memory_usage(f"After cell peak extraction {Path(file_path).name}")
        obj.feature_meta = pd.DataFrame({
            "mz": obj.data.columns.astype(float)
        })
        return obj
        
    @classmethod
    def load_from_filelist(cls, dir_path:str,
                    ref_mz: Optional[float] = None,
                    dtype = np.float32,
                    ppm_tol: int = 10,
                    resolution: float = 35000,
                    resample_points_per_fwhm: float = 5.0,
                    ms_peak_snr_threshold: float = 10.0,
                    prominence_ratio: float = None,
                    n_jobs:int = 1,
                    distance:int = 3,
                    show_progress: bool = True,
                    log_memory: bool = True,
                    **preprocess_kwds):
        """Process multiple raw MS files with a memory-bounded batch architecture.

        The mass-spectrometry workflow uses two sequential passes.  Pass 1 reads
        one raw file at a time and builds the shared summed-spectrum feature axis.
        Pass 2 again reads exactly one raw file at a time, aligns that file to the
        shared axis, extracts cells locally, retains only the much smaller
        cell-level result, and releases the raw/frame-level objects before opening
        the next file.  Cell-level results are concatenated only after all files
        have been processed.

        This deliberately avoids the historical global ``all_frames x features``
        matrix while preserving the existing CyESIData return type, plotting
        modules, and downstream analysis API.

        Acquisition timestamps are preserved across the batch.  Each detected
        cell retains local ``rt`` plus ``acquisition_timestamp``, ``absolute_time``
        (Unix seconds), globally normalized ``time``, ``label``, and
        ``source_file``.

        ``n_jobs`` is retained for API compatibility but raw-file processing is
        intentionally sequential.  It is forwarded only to blockwise cell peak
        extraction unless that routine is explicitly configured via
        ``preprocess_kwds['n_jobs']``.
        """
        dtype = _validate_float_dtype(dtype)

        files = []
        for name in os.listdir(dir_path):
            full_path = os.path.join(dir_path, name)
            if (not os.path.isdir(full_path)) and name.lower().endswith((".mzml", ".mzxml")):
                files.append(full_path)
        files.sort()
        if not files:
            raise FileNotFoundError(f"No mzML/mzXML files found in {dir_path}")

        logging.info(f"Detected files in targeted directory: {files}")
        logging.info(
            "Batch raw-MS workflow: shared feature-axis pass followed by "
            "strictly sequential single-file alignment/cell extraction."
        )
        if n_jobs not in (None, 1):
            logging.info(
                "Raw-file alignment remains sequential for bounded memory; "
                "n_jobs=%s is used only for cell peak extraction unless overridden.",
                n_jobs,
            )

        # ------------------------------------------------------------------
        # PASS 1: build a shared feature axis with one raw file in memory.
        # ------------------------------------------------------------------
        logging.info(
            "Building shared summed spectrum, resolution=%s, "
            "resample points per FWHM=%s...",
            resolution,
            resample_points_per_fwhm,
        )
        total_mz = None
        total_intensity = None
        file_records = []

        for fp in tqdm(
            files,
            desc="Shared axis: summing files",
            unit="file",
            disable=not show_progress,
        ):
            exp, file_meta = load_single_file(fp, format="auto")
            if log_memory:
                _log_memory_usage(f"After loading {Path(fp).name} for shared-axis summing")

            n_ms1 = 0
            last_ms1_rt = 0.0
            for spec in exp:
                if spec.getMSLevel() == 1:
                    n_ms1 += 1
                    last_ms1_rt = float(spec.getRT())

            summed = sum_spec(
                exp,
                resolution_200=resolution,
                points_per_fwhm=resample_points_per_fwhm,
                intensity_dtype=dtype,
            )
            mz_grid, summed_intensity = summed.get_peaks()
            mz_grid = np.asarray(mz_grid, dtype=np.float64)
            # Accumulation remains float64 so the shared feature axis is not
            # changed by the low-memory storage dtype.
            summed_intensity = np.asarray(summed_intensity, dtype=np.float64)

            if total_mz is None:
                total_mz = mz_grid.copy()
                total_intensity = summed_intensity.copy()
            else:
                if total_mz.shape != mz_grid.shape or not np.array_equal(total_mz, mz_grid):
                    summed_intensity = np.interp(
                        total_mz, mz_grid, summed_intensity, left=0.0, right=0.0
                    )
                total_intensity += summed_intensity

            file_records.append({
                "path": fp,
                "file_meta": dict(file_meta),
                "n_frames": int(n_ms1),
                "last_rt": float(last_ms1_rt),
            })

            del exp, summed, mz_grid, summed_intensity
            _release_memory_to_os()
            if log_memory:
                _log_memory_usage(f"After summing and releasing {Path(fp).name}")

        total_sum_spec = oms.MSSpectrum()
        total_sum_spec.setMSLevel(1)
        total_sum_spec.setRT(0.0)
        total_sum_spec.set_peaks((
            total_mz.astype(np.float64, copy=False),
            total_intensity.astype(dtype, copy=False),
        ))

        logging.info(
            "Performing shared summed-MS denoising, snr_threshold=%s...",
            ms_peak_snr_threshold,
        )
        total_sum_spec = filter_spectrum(
            total_sum_spec,
            snr_threshold=ms_peak_snr_threshold,
            dtype=dtype,
        )
        logging.info("Performing shared summed-MS peak picking...")
        mz_list, _ = extract_peaks(
            total_sum_spec,
            dtype=dtype,
            mz_dtype=np.float64,
            prominence_ratio=prominence_ratio,
            distance=distance,
            resolution_200=resolution,
        )
        mz_list = np.asarray(mz_list, dtype=np.float64)
        if mz_list.size == 0:
            raise ValueError("No peaks were detected from the shared summed spectrum.")

        logging.info("Shared feature axis contains %d m/z features.", mz_list.size)
        if log_memory:
            _log_memory_usage("After shared summed-spectrum peak picking")

        del total_sum_spec, total_mz, total_intensity
        _release_memory_to_os()

        # ------------------------------------------------------------------
        # Preserve the v3 acquisition-timestamp logic across independent files.
        # ------------------------------------------------------------------
        file_records.sort(key=lambda x: x["file_meta"]["timestamp"])
        timestamp_start = float(file_records[0]["file_meta"]["timestamp"])
        timestamp_end = max(
            float(record["file_meta"]["timestamp"]) + float(record["last_rt"])
            for record in file_records
        )
        time_span = timestamp_end - timestamp_start
        if not np.isfinite(time_span) or time_span <= 0:
            time_span = 1.0

        # ``find_cell_peaks`` can still parallelize small cell/window operations,
        # but the expensive raw-file load/alignment itself always remains serial.
        local_preprocess_kwds = dict(preprocess_kwds)
        local_preprocess_kwds.setdefault("dtype", dtype)
        local_preprocess_kwds.setdefault("show_progress", show_progress)
        if n_jobs not in (None, 1):
            local_preprocess_kwds.setdefault("n_jobs", n_jobs)

        cell_data_parts = []
        cell_meta_parts = []
        per_file_meta = []

        # ------------------------------------------------------------------
        # PASS 2: one raw file -> local frames -> local cells -> release.
        # ------------------------------------------------------------------
        iterator = tqdm(
            file_records,
            desc="Processing raw files",
            unit="file",
            disable=not show_progress,
        )
        for record in iterator:
            fp = record["path"]
            file_name = Path(fp).name
            file_meta = dict(record["file_meta"])

            exp, loaded_meta = load_single_file(fp, format="auto")
            # Use metadata from the actual second-pass load while preserving the
            # timestamp/order established in pass 1.
            file_meta.update(loaded_meta)
            if log_memory:
                _log_memory_usage(f"After loading {file_name} for local alignment")

            logging.info("Aligning frames from MS file %s to shared feature axis...", fp)
            frame_data, frame_meta = align_frame(
                exp,
                mz_list,
                ppm_tol,
                dtype=dtype,
                mz_dtype=np.float64,
                prominence_ratio=prominence_ratio,
                distance=distance,
                resolution_200=resolution,
            )
            if log_memory:
                _log_memory_usage(f"After local alignment {file_name}")

            # Timestamp fields are created at frame level so local cell extraction
            # automatically carries the correct values to each detected peak frame.
            acquisition_timestamp = float(file_meta["timestamp"])
            frame_meta = frame_meta.copy()
            frame_meta["acquisition_timestamp"] = acquisition_timestamp
            frame_meta["absolute_time"] = acquisition_timestamp + frame_meta["rt"].to_numpy(dtype=np.float64)
            frame_meta["time"] = (
                frame_meta["absolute_time"] - timestamp_start
            ) / time_span
            label = file_meta["name"].split(".")[0]
            frame_meta["label"] = label
            frame_meta["source_file"] = label

            # Raw spectra are no longer needed.  Release the native object before
            # running the blockwise cell extraction on the local frame matrix.
            del exp
            _release_memory_to_os()
            if log_memory:
                _log_memory_usage(f"After releasing raw experiment {file_name}")

            local_obj = object.__new__(cls)
            local_obj.data = frame_data
            local_obj.peak_meta = frame_meta
            local_obj.file_meta = file_meta
            local_obj.file_meta["ref_mz"] = ref_mz
            local_obj.ref_mz = ref_mz

            logging.info("Extracting cells from %s before opening the next raw file...", file_name)
            local_obj.preprocess(**local_preprocess_kwds)
            if log_memory:
                _log_memory_usage(f"After local cell extraction {file_name}")

            # Keep only cell-level objects.  Prefixing the temporary index prevents
            # collisions; a compact global cell index is assigned after concat.
            local_cells = local_obj.data.copy(deep=False)
            local_meta = local_obj.peak_meta.copy(deep=False)
            local_index = pd.Index(
                [f"{label}__cell_{i:07d}" for i in range(len(local_cells))],
                name="cell_id",
            )
            local_cells.index = local_index
            local_meta.index = local_index
            cell_data_parts.append(local_cells)
            cell_meta_parts.append(local_meta)

            file_meta["n_cells"] = int(len(local_cells))
            file_meta["n_frames"] = int(record["n_frames"])
            file_meta["last_rt"] = float(record["last_rt"])
            per_file_meta.append(file_meta)

            # Drop the local frame-level matrix before processing the next file.
            del frame_data, frame_meta, local_obj, local_cells, local_meta
            _release_memory_to_os()
            if log_memory:
                _log_memory_usage(f"After releasing local frame matrix {file_name}")

        # ------------------------------------------------------------------
        # Cell-level concat.  Local feature filtering may have removed different
        # shared-axis columns, so outer-concat and zero-fill reproduce the batch
        # architecture without ever concatenating raw frames.
        # ------------------------------------------------------------------
        logging.info("Concatenating %d processed files at cell level...", len(cell_data_parts))
        if cell_data_parts:
            combined_data = pd.concat(
                cell_data_parts,
                axis=0,
                join="outer",
                sort=False,
                copy=False,
            ).fillna(0)
            # Columns originate from the same shared feature axis.  Restore that
            # axis order for all features that survived at least one local filter.
            present = set(combined_data.columns)
            ordered_cols = [mz for mz in mz_list if mz in present]
            combined_data = combined_data.loc[:, ordered_cols]
            combined_data = combined_data.astype(dtype, copy=False)
            combined_meta = pd.concat(cell_meta_parts, axis=0, copy=False)
        else:
            combined_data = pd.DataFrame(dtype=dtype)
            combined_meta = pd.DataFrame()

        # Compact, deterministic global cell ids while retaining source_file/label.
        global_index = pd.Index(
            [f"cell_{i:08d}" for i in range(len(combined_data))],
            name="cell_id",
        )
        combined_data.index = global_index
        combined_meta.index = global_index

        obj = object.__new__(cls)
        obj.data = combined_data
        obj.peak_meta = combined_meta
        obj.file_meta = {
            "name": os.path.basename(os.path.normpath(dir_path)),
            "ref_mz": ref_mz,
            "per_file_meta": per_file_meta,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "time_span": time_span,
            "batch_processing": {
                "architecture": "shared_axis_singlefile_cell_concat",
                "n_files": int(len(file_records)),
                "raw_file_parallelism": 1,
                "shared_feature_count": int(mz_list.size),
                "dtype": dtype.name,
            },
        }
        obj.ref_mz = ref_mz
        obj.feature_meta = pd.DataFrame({
            "mz": obj.data.columns.astype(float)
        })

        del cell_data_parts, cell_meta_parts
        _release_memory_to_os()
        if log_memory:
            _log_memory_usage("After final cell-level concatenation")

        logging.info(
            "Batch processing complete: %d cells x %d features from %d files.",
            obj.data.shape[0],
            obj.data.shape[1],
            len(file_records),
        )
        return obj

    def preprocess(self, baseline_filter = median_filter, 
                         baseline_filter_size:int = 50,
                         cell_snr:float = 5.0,
                         peak_snr:float = 3.0,
                         max_zero_frac:float = 0.9,
                         dtype=np.float32,
                         debug_hook: Optional[DebugHook] = None,
                         **kwargs):
        dtype = _validate_float_dtype(dtype)
        
        def emit(stage: str, **payload):
            if debug_hook is not None:
                debug_hook(stage, payload)
                
        # The normal preprocessing path does not retain the full frame x feature
        # baseline matrix.  A full baseline is only requested when a debug hook is
        # present and may need it for visualization/inspection.
        data = find_cell_peaks(
            self.data,
            self.ref_mz,
            baseline_filter=baseline_filter,
            baseline_filter_size=baseline_filter_size,
            cell_snr=cell_snr,
            peak_snr=peak_snr,
            max_zero_frac=max_zero_frac,
            dtype=dtype,
            return_full_baseline=(debug_hook is not None),
            **kwargs,
        )
        emit(
            "find_cells",
            time=self.peak_meta["time"],
            signal=self.data,
            baseline=data["baseline"],
            cell_idx=data["peak_frames"],
        )
        # Compute reference m/z SNR for each detected cell (peak frame).
        peak_frames = data.get("peak_frames", np.array([], dtype=int))
        ref_baseline = data.get("ref_baseline", None)
        ref_idx = data.get("ref_idx", None)
        snr_values = []
        eps = 1e-12
        # Use original data matrix to get intensities at peak frames
        X_orig = self.data.values
        for pf in peak_frames:
            try:
                inten = float(X_orig[pf, ref_idx])
            except Exception:
                inten = float('nan')
            try:
                bval = float(ref_baseline[pf])
            except Exception:
                bval = 0.0
            if bval is None or (isinstance(bval, float) and math.isnan(bval)):
                bval = 0.0
            if bval <= 0:
                snr = float('nan') if math.isnan(inten) else (inten / eps)
            else:
                snr = inten / bval
            snr_values.append(snr)

        # Subset data and peak_meta to detected cells, then attach SNRs
        self.data, self.peak_meta = (data["cell_df"], 
            pd.DataFrame(self.peak_meta.iloc[data["peak_frames"], :], index=self.peak_meta.index[data["peak_frames"]]))
        try:
            # align snr_values order with the new peak_meta rows
            self.peak_meta["ref_snr"] = snr_values
        except Exception:
            # fallback: ensure column exists even if empty
            self.peak_meta["ref_snr"] = pd.Series(index=self.peak_meta.index, dtype=float)
        self.file_meta['length'] = self.data.shape[0]
        return self
            
    def impute(self, method:str = 'knn', missing_values = 0, **kwargs):
        logging.info(f"Run data imputing on {self.get_name()}, method:{method}")
        if method == 'knn':
            imputer = KNNImputer(missing_values=missing_values, **kwargs)
        else:
            imputer = SimpleImputer(missing_values=missing_values, strategy=method, **kwargs)
            
        self.data = pd.DataFrame(
            imputer.fit_transform(self.data),
            columns = self.data.columns,
            dtype = self.data.values.dtype
        )
        return self
    
    def remove_outlier(self, **kwargs):
        iso = IsolationForest(**kwargs)
        inlier_id = (iso.fit_predict(self.data) == 1)
        self.data = self.data.iloc[inlier_id, :]
        self.peak_meta = self.peak_meta.iloc[inlier_id, :]
        return self
                    
    def normalize(self, method:str = "total", **norm_kwargs):
        logging.info(f"Run normalization on {self.file_meta['name']}, method:{method}")
        self.data = pd.DataFrame(
            normalize(self.data.values, method, norm_kwargs),
            columns = self.data.columns,
            dtype = self.data.values.dtype
        )
        return self
    
    def alignwith(self, other:Self, ppm_tol:float = 5.0, mz_merge_options: Literal["union", "ref"] = "union"):
        
        df1, df2 = self.data, other.data
        mz1, mz2 = df1.columns.values.astype(self.data.values.dtype), df2.columns.values.astype(self.data.values.dtype)
        idx2_aligned = np.full(len(mz1), -1, dtype=int)

        j = 0
        for i, m in enumerate(mz1):
            while j < len(mz2) and mz2[j] < m * (1 - ppm_tol * 1e-6):
                j += 1
            if j < len(mz2) and abs(mz2[j] - m) / m * 1e6 <= ppm_tol:
                idx2_aligned[i] = j

        keep = idx2_aligned >= 0
        mz_aligned = df1.columns[keep]
        df1_aligned = df1.loc[:, mz_aligned]
        df2_aligned = df2.iloc[:, idx2_aligned[keep]]
        df2_aligned.columns = mz_aligned

        merged_df = pd.concat([df1_aligned, df2_aligned], axis=0, ignore_index=True)

        if mz_merge_options == "union":
            mask_new = np.ones(len(mz2), dtype=bool)
            mask_new[idx2_aligned[keep]] = False
            new_mz = mz2[mask_new]
            if len(new_mz) > 0:
                df2_new = df2.loc[:, new_mz].copy()
                df2_new[:] = 0
                merged_df = pd.concat([merged_df, df2_new], axis=1)
                
        self.data = merged_df
        self.peak_meta = pd.concat([self.peak_meta, other.peak_meta], axis = 0, ignore_index=True)
        if not self._concat_flag:
            self.file_meta = [self.file_meta]
        per_file_meta = []
        if "per_file_meta" in self.file_meta:
            if "per_file_meta" not in other.file_meta:
                per_file_meta.append(other.file_meta)
            else:
                per_file_meta.extend(other.file_meta.get("per_file_meta", []))
        else:
            per_file_meta = [self.file_meta, other.file_meta]
        self.file_meta = {
            "name": f"{self.file_meta['name']}+{other.file_meta['name']}",
            "per_file_meta": per_file_meta
        }

        return self
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, key):
        key = float(key)
        idx = (np.abs(self.data.columns.values.astype(float) - key)).argmin()
        return self.data.iloc[:, idx].values
    
    def save(self, root_path:str):
        dir_name = os.path.join(root_path, self.file_meta["name"])
        os.mkdir(dir_name)
        logging.info(f"Saving processed data to {dir_name}...")
        with open(os.path.join(dir_name, ".meta"), 'w') as fp:
            file_meta = json.dump(self.file_meta, fp)
        self.data.to_csv(os.path.join(dir_name, "data.csv"))
        self.peak_meta.to_csv(os.path.join(dir_name, "peak_meta.csv"))
        self.feature_meta.to_csv(os.path.join(dir_name, "feature_meta.csv"))
        
    def to_anndata(self):
        obs_df = pd.DataFrame({
            "cell_id": self.peak_meta.index,
        })
        var_df = pd.DataFrame({
            "feature_id": self.feature_meta.index
        })
        for col in self.peak_meta.columns:
            obs_df[col] = self.peak_meta[col].values
        for col in self.feature_meta.columns:
            var_df[col] = self.feature_meta[col].values

        adata = AnnData(
            X=self.data.values,
            obs=obs_df.set_index("cell_id"),
            var=var_df.set_index("feature_id")
        )
        adata.raw = adata.copy()
        return adata
    
    def deisotope(
        self,
        isotope_diff: float = 1.003355,
        ppm_tol: float = 1.0,
        max_isotope_order: int = 3,
        r_square_threshold: float = 0.95,
        carbon13_abundance: float = 0.0109,
        intensity_threshold: float = 0.0,
        safety_factor: float = 1.0,
        missing_func=lambda x: x == 0,
        merge_mode: str = "keep_parent",
        remove: bool = True,
        inplace: bool = True,
    ):

        if merge_mode not in {"keep_parent", "sum"}:
            raise ValueError("merge_mode must be either 'keep_parent' or 'sum'.")

        df = self.data
        mz = df.columns.astype(float).to_numpy()
        n_features = len(mz)

        # ---------- ensure feature_meta ----------
        if not hasattr(self, "feature_meta") or self.feature_meta is None:
            self.feature_meta = pd.DataFrame(index=df.columns)

        self.feature_meta = self.feature_meta.reindex(df.columns)

        if "mz" not in self.feature_meta.columns:
            self.feature_meta["mz"] = mz

        # ---------- Step 1: m/z candidate search ----------
        dmz = mz[None, :] - mz[:, None]

        isotope_order = np.rint(dmz / isotope_diff).astype(int)
        expected_dmz = isotope_order * isotope_diff

        ppm_error = np.abs(dmz - expected_dmz) / mz[:, None] * 1e6

        candidate_mask = (
            (isotope_order >= 1)
            & (isotope_order <= max_isotope_order)
            & (dmz > 0)
            & (ppm_error <= ppm_tol)
        )

        candidate_rows, candidate_cols = np.where(candidate_mask)

        candidate_table = pd.DataFrame({
            "parent_index": candidate_rows,
            "isotope_index": candidate_cols,
            "parent_feature": df.columns[candidate_rows],
            "isotope_feature": df.columns[candidate_cols],
            "parent_mz": mz[candidate_rows],
            "isotope_mz": mz[candidate_cols],
            "isotope_order": isotope_order[candidate_rows, candidate_cols],
            "ppm_error": ppm_error[candidate_rows, candidate_cols],
        })

        candidate_map = {
            str(df.columns[i]): [str(x) for x in df.columns[candidate_mask[i]]]
            for i in range(n_features)
            if np.any(candidate_mask[i])
        }

        # ---------- Step 2: through-origin regression ----------
        X_raw = df.to_numpy(dtype=float)

        missing = missing_func(X_raw)
        missing = np.asarray(missing, dtype=bool)

        if missing.shape != X_raw.shape:
            raise ValueError(
                "missing_func must return a boolean array with the same shape as input data."
            )

        missing = missing | np.isnan(X_raw) | np.isinf(X_raw)

        if intensity_threshold > 0:
            missing = missing | (X_raw <= intensity_threshold)

        valid = ~missing

        X = X_raw.copy()
        X[missing] = 0.0

        G = X.T @ X
        X2 = X ** 2
        V = valid.astype(float)

        ss_x_pair = X2.T @ V
        ss_y_pair = V.T @ X2

        with np.errstate(divide="ignore", invalid="ignore"):
            A = G / ss_x_pair
            R = (G ** 2) / (ss_x_pair * ss_y_pair)

        A[~np.isfinite(A)] = np.nan
        R[~np.isfinite(R)] = np.nan

        A_df = pd.DataFrame(A, index=df.columns, columns=df.columns)
        R_df = pd.DataFrame(R, index=df.columns, columns=df.columns)

        # ---------- Step 3: R^2 filtering ----------
        r2_mask = candidate_mask & (R >= r_square_threshold)

        # ---------- Step 4: isotope upper-bound filtering ----------
        nC_max = np.floor(mz / 12.0).astype(int)
        q = carbon13_abundance / (1.0 - carbon13_abundance)

        ratio_limit = np.full_like(A, np.nan, dtype=float)

        for k in range(1, max_isotope_order + 1):
            limits_k = np.zeros(n_features, dtype=float)

            for i in range(n_features):
                if nC_max[i] >= k:
                    limits_k[i] = math.comb(int(nC_max[i]), k) * (q ** k)
                else:
                    limits_k[i] = 0.0

            mask_k = isotope_order == k
            ratio_limit[mask_k] = np.broadcast_to(
                limits_k[:, None],
                ratio_limit.shape
            )[mask_k]

        ratio_limit = ratio_limit * safety_factor
        ratio_mask = A <= ratio_limit

        final_mask_raw = r2_mask & ratio_mask

        # ---------- Greedy low-m/z assignment ----------
        final_mask = np.zeros_like(final_mask_raw, dtype=bool)
        removed_indices = set()

        for i in np.argsort(mz):
            if i in removed_indices:
                continue

            js = np.where(final_mask_raw[i])[0]
            js = sorted(js, key=lambda j: (isotope_order[i, j], mz[j]))

            for j in js:
                if j not in removed_indices:
                    final_mask[i, j] = True
                    removed_indices.add(j)

        final_rows, final_cols = np.where(final_mask)

        final_table = pd.DataFrame({
            "parent_index": final_rows,
            "isotope_index": final_cols,
            "parent_feature": df.columns[final_rows],
            "isotope_feature": df.columns[final_cols],
            "parent_mz": mz[final_rows],
            "isotope_mz": mz[final_cols],
            "isotope_order": isotope_order[final_rows, final_cols],
            "ppm_error": ppm_error[final_rows, final_cols],
            "slope_A": A[final_rows, final_cols],
            "r_square": R[final_rows, final_cols],
            "max_allowed_ratio": ratio_limit[final_rows, final_cols],
        })

        isotope_features = df.columns[sorted(removed_indices)].tolist()
        parent_features = df.columns[sorted(set(final_rows))].tolist()

        # ---------- write isotope distribution into feature_meta ----------
        fm = self.feature_meta.copy()

        fm["deisotope_role"] = "unique"
        fm.loc[parent_features, "deisotope_role"] = "parent"
        fm.loc[isotope_features, "deisotope_role"] = "isotope"

        fm["isotope_parent"] = pd.NA
        fm["isotope_order"] = pd.NA
        fm["isotope_slope_A"] = np.nan
        fm["isotope_r_square"] = np.nan
        fm["isotope_ppm_error"] = np.nan

        fm["isotope_children"] = "[]"
        fm["n_isotope_children"] = 0

        for k in range(1, max_isotope_order + 1):
            fm[f"M{k}_mz"] = np.nan
            fm[f"M{k}_feature"] = pd.NA
            fm[f"M{k}_slope_A"] = np.nan
            fm[f"M{k}_r_square"] = np.nan
            fm[f"M{k}_ppm_error"] = np.nan
            fm[f"M{k}_max_allowed_ratio"] = np.nan

        for _, row in final_table.iterrows():
            parent = row["parent_feature"]
            iso = row["isotope_feature"]
            k = int(row["isotope_order"])

            fm.loc[iso, "isotope_parent"] = parent
            fm.loc[iso, "isotope_order"] = k
            fm.loc[iso, "isotope_slope_A"] = row["slope_A"]
            fm.loc[iso, "isotope_r_square"] = row["r_square"]
            fm.loc[iso, "isotope_ppm_error"] = row["ppm_error"]

            child_info = {
                "feature": str(iso),
                "mz": float(row["isotope_mz"]),
                "order": k,
                "slope_A": float(row["slope_A"]),
                "r_square": float(row["r_square"]),
                "ppm_error": float(row["ppm_error"]),
                "max_allowed_ratio": float(row["max_allowed_ratio"]),
            }

            old_children = json.loads(fm.loc[parent, "isotope_children"])
            old_children.append(child_info)
            fm.loc[parent, "isotope_children"] = json.dumps(old_children, ensure_ascii=False)
            fm.loc[parent, "n_isotope_children"] = len(old_children)

            fm.loc[parent, f"M{k}_mz"] = row["isotope_mz"]
            fm.loc[parent, f"M{k}_feature"] = iso
            fm.loc[parent, f"M{k}_slope_A"] = row["slope_A"]
            fm.loc[parent, f"M{k}_r_square"] = row["r_square"]
            fm.loc[parent, f"M{k}_ppm_error"] = row["ppm_error"]
            fm.loc[parent, f"M{k}_max_allowed_ratio"] = row["max_allowed_ratio"]

        # ---------- Merge isotope intensities ----------
        processed_data = df.copy()

        if merge_mode == "sum":
            for parent_idx, isotope_idx in zip(final_rows, final_cols):
                parent_col = df.columns[parent_idx]
                isotope_col = df.columns[isotope_idx]

                processed_data[parent_col] = (
                    processed_data[parent_col].fillna(0)
                    + df[isotope_col].fillna(0)
                )

        if remove:
            processed_data = processed_data.drop(columns=isotope_features)
            fm = fm.loc[processed_data.columns].copy()

        result = {
            "candidate_map": candidate_map,
            "candidate_table": candidate_table,
            "A": A_df,
            "R": R_df,
            "ratio_limit": pd.DataFrame(
                ratio_limit,
                index=df.columns,
                columns=df.columns
            ),
            "final_table": final_table,
            "isotope_features": isotope_features,
            "parent_features": parent_features,
            "processed_data": processed_data,
            "feature_meta": fm,
            "params": {
                "isotope_diff": isotope_diff,
                "ppm_tol": ppm_tol,
                "max_isotope_order": max_isotope_order,
                "r_square_threshold": r_square_threshold,
                "carbon13_abundance": carbon13_abundance,
                "intensity_threshold": intensity_threshold,
                "safety_factor": safety_factor,
                "merge_mode": merge_mode,
                "remove": remove,
            }
        }

        if inplace:
            self.deisotope_result = result
            self.data = processed_data
            self.feature_meta = fm

            if not hasattr(self, "file_meta") or self.file_meta is None:
                self.file_meta = {}

            self.file_meta["deisotope"] = {
                "params": result["params"],
                "n_candidate_pairs": int(len(candidate_table)),
                "n_final_isotope_pairs": int(len(final_table)),
                "n_removed_features": int(len(isotope_features)),
                "merge_mode": merge_mode,
            }

            return self

        return result
    
    def get_annotation(self, sdf_path:str, ppm_tol:float, search_mode: Literal["pos", "neg", "both"] = "pos",
                 adducts_pos:dict = DEFAULT_ADDUCTS_POS, 
                 adducts_neg:dict = DEFAULT_ADDUCTS_NEG, **kwargs):
        searcher = SDFMzSearcher(sdf_path=sdf_path, adducts_pos=adducts_pos, adducts_neg=adducts_neg)
        res = searcher.search(mz=self.data.columns.astype(float), ppm_tol=ppm_tol, mode=search_mode, **kwargs)
        return res