import numpy as np
import json
import pandas as pd
import pyopenms as oms
import logging
import math
import gc
import ctypes
from pathlib import Path
from tqdm.auto import tqdm
from anndata import AnnData, read_h5ad

from .io import (load_single_file, 
                 sum_spec, 
                 extract_peaks, 
                 align_frame)
from ..util.peak import filter_spectrum, find_cell_peaks
from ..util.normalize import normalize
from ..util.annotation import SDFMzSearcher, DEFAULT_ADDUCTS_NEG, DEFAULT_ADDUCTS_POS

from typing import Literal


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

class CyESIData(AnnData):
    """CyESI cell-by-feature data backed by the AnnData data model."""

    CELL_FILTER_PREFIX = "cell_keep_"
    FEATURE_FILTER_PREFIX = "feature_keep_"

    def __init__(
        self,
        X=None,
        obs=None,
        var=None,
        uns=None,
        layers=None,
        **kwargs,
    ):
        super().__init__(
            X=X,
            obs=obs,
            var=var,
            uns=uns,
            layers=layers,
            **kwargs,
        )
        if X is not None and self.n_obs:
            self.ensure_cell_quality_metrics()

    @classmethod
    def read(cls, path):
        """Read a processed SCMM H5AD file."""
        adata = read_h5ad(path)
        result = cls(
            X=adata.X,
            obs=adata.obs.copy(),
            var=adata.var.copy(),
            uns=dict(adata.uns),
            obsm=adata.obsm.copy(),
            varm=adata.varm.copy(),
            obsp=adata.obsp.copy(),
            varp=adata.varp.copy(),
            layers=adata.layers.copy(),
        )
        result.ensure_cell_quality_metrics()
        return result

    def ensure_cell_quality_metrics(self, *, overwrite=False):
        """Store immutable per-cell metrics derived from the processed raw matrix."""
        matrix = self.X.toarray() if hasattr(self.X, "toarray") else np.asarray(self.X)
        if overwrite or "detected_features" not in self.obs:
            self.obs["detected_features"] = np.count_nonzero(matrix > 0, axis=1).astype(
                np.uint32
            )
        if overwrite or "raw_total_intensity" not in self.obs:
            self.obs["raw_total_intensity"] = np.nansum(matrix, axis=1).astype(
                np.float64, copy=False
            )
        return self

    def ensure_matrix_layers(self):
        """Preserve the initial X and track which layer supplies the working X."""
        active = self.uns.get("active_layer")
        if active in self.layers:
            return self
        name = "raw" if "raw" not in self.layers else self._available_layer_name("input_X")
        self.layers[name] = self.X
        self.uns["active_layer"] = name
        self.uns.setdefault("matrix_layers", {})[name] = {"operation": "input"}
        return self

    def _available_layer_name(self, name):
        candidate = str(name).strip() or "processed"
        base, index = candidate, 2
        while candidate in self.layers:
            candidate = f"{base}_{index}"
            index += 1
        return candidate

    def set_active_layer(self, name):
        """Make the selected layer the working X used by all downstream consumers."""
        self.ensure_matrix_layers()
        self.X = self.layers[str(name)]
        self.uns["active_layer"] = str(name)
        return self

    def _store_processed_layer(self, matrix, name, operation, params):
        source = str(self.uns["active_layer"])
        destination = self._available_layer_name(name)
        self.layers[destination] = matrix
        self.uns.setdefault("matrix_layers", {})[destination] = {
            "operation": operation,
            "source": source,
            "params": dict(params),
        }
        self.set_active_layer(destination)
        return destination

    def cell_filter_columns(self):
        return [name for name in self.obs if str(name).startswith(self.CELL_FILTER_PREFIX)]

    def feature_filter_columns(self):
        return [name for name in self.var if str(name).startswith(self.FEATURE_FILTER_PREFIX)]

    def set_active_cell_filters(self, columns):
        available = set(self.cell_filter_columns())
        self.uns["active_cell_filters"] = [name for name in columns if name in available]
        return self

    def set_active_feature_filters(self, columns):
        available = set(self.feature_filter_columns())
        self.uns["active_feature_filters"] = [name for name in columns if name in available]
        return self

    def cell_keep_mask(self, columns=None):
        available = self.cell_filter_columns()
        if columns is None:
            configured = self.uns.get("active_cell_filters")
            columns = available if configured is None else list(configured)
        columns = [name for name in columns if name in available]
        if not columns:
            return np.ones(self.n_obs, dtype=bool)
        return np.all(self.obs[columns].to_numpy(dtype=np.uint8) == 1, axis=1)

    def feature_keep_mask(self, columns=None):
        available = self.feature_filter_columns()
        if columns is None:
            configured = self.uns.get("active_feature_filters")
            columns = available if configured is None else list(configured)
        columns = [name for name in columns if name in available]
        if not columns:
            return np.ones(self.n_vars, dtype=bool)
        return np.all(self.var[columns].to_numpy(dtype=np.uint8) == 1, axis=1)

    def set_cell_filter(self, name, keep):
        column = str(name)
        if not column.startswith(self.CELL_FILTER_PREFIX):
            column = f"{self.CELL_FILTER_PREFIX}{column}"
        values = np.asarray(keep, dtype=np.uint8)
        if values.shape != (self.n_obs,):
            raise ValueError("cell filter must contain one value per cell")
        self.obs[column] = values
        active = list(self.uns.get("active_cell_filters", self.cell_filter_columns()))
        if column not in active:
            active.append(column)
        self.set_active_cell_filters(active)
        return column

    def restore_cell_filter(self, name):
        if name in self.obs:
            self.obs[name] = np.ones(self.n_obs, dtype=np.uint8)
        return self

    def set_feature_filter(self, name, keep):
        column = str(name)
        if not column.startswith(self.FEATURE_FILTER_PREFIX):
            column = f"{self.FEATURE_FILTER_PREFIX}{column}"
        values = np.asarray(keep, dtype=np.uint8)
        if values.shape != (self.n_vars,):
            raise ValueError("feature filter must contain one value per feature")
        self.var[column] = values
        active = list(self.uns.get("active_feature_filters", self.feature_filter_columns()))
        if column not in active:
            active.append(column)
        self.set_active_feature_filters(active)
        return column

    def restore_feature_filter(self, name):
        if name in self.var:
            self.var[name] = np.ones(self.n_vars, dtype=np.uint8)
        return self

    def active_view(self, *, copy=True):
        view = self[self.cell_keep_mask(), self.feature_keep_mask()]
        return view.copy() if copy else view

    def label_features_by_detection(
        self,
        threshold,
        *,
        source_file=None,
        key="detection",
    ):
        cell_mask = self.cell_keep_mask()
        if source_file is not None:
            if "source_file" not in self.obs:
                raise KeyError("obs['source_file'] is required for per-file detection")
            cell_mask &= self.obs["source_file"].astype(str).eq(str(source_file)).to_numpy()
        if not cell_mask.any():
            raise ValueError("no cells are available for feature detection-rate filtering")
        matrix = self.X[cell_mask]
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        rate = np.count_nonzero(np.asarray(matrix) > 0, axis=0) / int(cell_mask.sum())
        rate_column = "detection_rate_total" if source_file is None else "detection_rate_selected_file"
        self.var[rate_column] = rate.astype(np.float32)
        column = self.set_feature_filter(key, rate >= float(threshold))
        return column, rate

    @property
    def data(self):
        """Compatibility view of X as a DataFrame."""
        return self.to_df()

    @data.setter
    def data(self, value):
        frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
        columns = pd.Index(frame.columns.astype(str))
        if frame.shape == self.shape and columns.equals(self.var_names):
            self.X = frame.to_numpy()
            return

        positions = self.var_names.get_indexer(columns)
        if len(frame) == self.n_obs and np.all(positions >= 0):
            self._inplace_subset_var(positions)
            self.X = frame.to_numpy()
            self.var_names = columns
            return

        obs = self.obs.copy() if len(frame) == self.n_obs else pd.DataFrame(index=frame.index)
        var = pd.DataFrame(index=columns)
        replacement = AnnData(
            X=frame.to_numpy(),
            obs=obs,
            var=var,
            uns=dict(self.uns),
        )
        self._init_as_actual(replacement)

    @property
    def peak_meta(self):
        return self.obs

    @peak_meta.setter
    def peak_meta(self, value):
        value = pd.DataFrame(value).copy()
        if len(value) != self.n_obs:
            raise ValueError("peak_meta row count must match n_obs")
        value.index = self.obs_names
        self.obs = value

    @property
    def feature_meta(self):
        return self.var

    @feature_meta.setter
    def feature_meta(self, value):
        value = pd.DataFrame(value).copy()
        if len(value) != self.n_vars:
            raise ValueError("feature_meta row count must match n_vars")
        value.index = self.var_names
        self.var = value

    @property
    def file_meta(self):
        return self.uns.setdefault("file_meta", {})

    @file_meta.setter
    def file_meta(self, value):
        self.uns["file_meta"] = dict(value)

    @staticmethod
    def _source_metadata(metadata_by_file, file_path):
        if not metadata_by_file:
            return {}
        path = Path(file_path)
        if all(not isinstance(value, dict) for value in metadata_by_file.values()):
            return dict(metadata_by_file)
        for key in (str(path), path.name, path.stem):
            if key in metadata_by_file:
                return dict(metadata_by_file[key])
        return {}

    @staticmethod
    def _frame_obs(frame_meta, file_path, file_meta, frame_metadata, metadata_by_file):
        frame_metadata = tuple(frame_metadata or ())
        obs = pd.DataFrame(index=frame_meta.index)
        obs["source_file"] = Path(file_path).name
        # Always retain the original experiment-frame position.  This is not
        # the compact MS1-only row number: when MS2 scans are interleaved it
        # still points back to the original frame in the source file.
        obs["frame_id"] = frame_meta.index.to_numpy(dtype=np.int64)

        for field in frame_metadata:
            if field == "frame_id":
                continue
            if field in frame_meta.columns:
                obs[field] = frame_meta[field].to_numpy()
            elif field == "acquisition_timestamp":
                obs[field] = float(file_meta.get("timestamp", np.nan))
            elif field == "acquisition_time":
                timestamp = float(file_meta.get("timestamp", np.nan))
                if "rt" not in frame_meta:
                    raise KeyError("rt is required to compute acquisition_time")
                obs[field] = timestamp + frame_meta["rt"].to_numpy(dtype=float)
            else:
                raise KeyError(f"Unknown frame metadata field: {field}")

        for key, value in CyESIData._source_metadata(
            metadata_by_file, file_path
        ).items():
            obs[key] = value
        return obs

    @classmethod
    def _from_aligned_frames(
        cls,
        frame_data,
        frame_obs,
        *,
        reference_mz,
        reference_mode,
        source_file,
        file_meta,
        dtype,
        debug_hook=None,
        progress_callback=None,
        **preprocess_kwargs,
    ):
        debug_full_baseline = bool(
            preprocess_kwargs.pop("debug_full_baseline", False)
        )
        result = find_cell_peaks(
            frame_data,
            reference_mz,
            reference_mode=reference_mode,
            dtype=dtype,
            return_full_baseline=debug_hook is not None and debug_full_baseline,
            progress_callback=progress_callback,
            **preprocess_kwargs,
        )
        if debug_hook is not None:
            debug_hook(
                "find_cells",
                {
                    "signal": frame_data,
                    "baseline": result["baseline"],
                    "cell_mask": result["cell_mask"],
                    "cell_idx": result["peak_frames"],
                    "reference_mz": result["reference_mz_matched"],
                    "frame_obs": frame_obs,
                },
            )

        peak_frames = result["peak_frames"]
        obs = frame_obs.iloc[peak_frames].copy()
        obs.index = pd.Index(
            [f"{Path(source_file).stem}__cell_{i:07d}" for i in range(len(obs))],
            name="cell_id",
        )

        for i, matched_mz in enumerate(result["reference_mz_matched"]):
            label = f"{float(matched_mz):.6f}".rstrip("0").rstrip(".")
            obs[f"reference_intensity_{label}"] = result["reference_signal"][
                peak_frames, i
            ]
            obs[f"reference_ratio_{label}"] = result["reference_ratio"][
                peak_frames, i
            ]

        cell_df = result["cell_df"].copy()
        cell_df.index = obs.index
        var = pd.DataFrame(
            {
                "mz": cell_df.columns.astype(float),
                "zero_fraction": result["zero_fraction"].loc[
                    cell_df.columns
                ].to_numpy(),
            },
            index=pd.Index(cell_df.columns.astype(str), name="feature_id"),
        )
        cell_df.columns = var.index

        processing = {
            "reference_mz_requested": np.atleast_1d(reference_mz).astype(float).tolist(),
            "reference_mz_matched": result["reference_mz_matched"].astype(float).tolist(),
            "reference_ppm_error": result["reference_ppm_error"].astype(float).tolist(),
            "reference_mode": reference_mode,
            "n_frames": int(len(frame_data)),
            "n_cells": int(len(cell_df)),
        }
        uns = {
            "files": {Path(source_file).name: dict(file_meta)},
            "file_meta": {
                "name": Path(source_file).stem,
                "reference_mz": processing["reference_mz_requested"],
            },
            "processing": processing,
        }
        return cls(
            X=cell_df.to_numpy(dtype=dtype, copy=False),
            obs=obs,
            var=var,
            uns=uns,
            layers={
                "feature_snr": np.asarray(result["feature_snr"], dtype=dtype)
            },
        )

    @classmethod
    def load_from_file(
        cls,
        file_path,
        reference_mz=None,
        *,
        ref_mz=None,
        reference_mode="union",
        frame_metadata=None,
        metadata_by_file=None,
        dtype=np.float32,
        ppm_tol=10,
        mz_range=(100.0, 1000.0),
        resolution=35000,
        resample_points_per_fwhm=5.0,
        ms_peak_snr_threshold=10.0,
        prominence_ratio=None,
        distance=3,
        log_memory=True,
        debug_hook=None,
        progress_callback=None,
        msconvert_path=None,
        **preprocess_kwargs,
    ):
        """Load one mzML/mzXML/vendor RAW file and return cell-level AnnData."""
        reference_mz = reference_mz if reference_mz is not None else ref_mz
        if reference_mz is None:
            raise ValueError("reference_mz is required")
        dtype = _validate_float_dtype(dtype)

        def report(fraction, message):
            if progress_callback is not None:
                progress_callback(float(fraction), str(message))

        report(0.05, f"读取质谱文件: {Path(file_path).name}")
        exp, file_meta = load_single_file(
            str(file_path), format="auto", msconvert_path=msconvert_path
        )
        file_meta = dict(file_meta)
        file_meta["source_file"] = Path(file_path).name
        file_meta["path"] = str(Path(file_path).resolve())
        if log_memory:
            _log_memory_usage(f"After loading {Path(file_path).name}")

        report(0.25, "叠加质谱图")
        summed = sum_spec(
            exp,
            mz_range=mz_range,
            resolution_200=resolution,
            points_per_fwhm=resample_points_per_fwhm,
            ms_level=1,
            intensity_dtype=dtype,
        )
        summed_mz, summed_intensity = summed.get_peaks()
        summed_mz = np.asarray(summed_mz, dtype=np.float64).copy()
        summed_intensity = np.asarray(summed_intensity, dtype=dtype).copy()
        filtered = filter_spectrum(
            summed,
            snr_threshold=ms_peak_snr_threshold,
            dtype=dtype,
        )
        mz_list, _ = extract_peaks(
            filtered,
            dtype=dtype,
            mz_dtype=np.float64,
            prominence_ratio=prominence_ratio,
            distance=distance,
            resolution_200=resolution,
        )
        if len(mz_list) == 0:
            raise ValueError("No m/z features were detected in the summed spectrum")

        report(0.50, f"总谱寻峰: 检测到 {len(mz_list)} 个候选特征")
        frame_data, frame_meta = align_frame(
            exp,
            mz_list,
            ppm_tol,
            ms_level=1,
            dtype=dtype,
            mz_dtype=np.float64,
            prominence_ratio=prominence_ratio,
            distance=distance,
            resolution_200=resolution,
        )
        frame_obs = cls._frame_obs(
            frame_meta,
            file_path,
            file_meta,
            frame_metadata,
            metadata_by_file,
        )
        del exp, summed, filtered
        _release_memory_to_os()

        report(0.75, "细胞检测与特征定量")
        obj = cls._from_aligned_frames(
            frame_data,
            frame_obs,
            reference_mz=reference_mz,
            reference_mode=reference_mode,
            source_file=file_path,
            file_meta=file_meta,
            dtype=dtype,
            debug_hook=debug_hook,
            progress_callback=lambda value, message: report(
                0.75 + 0.2 * value, message
            ),
            ppm_tolerance=ppm_tol,
            **preprocess_kwargs,
        )
        obj.uns["summed_spectra"] = {
            Path(file_path).name: {
                "mz": summed_mz,
                "intensity": summed_intensity,
                "candidate_mz": np.asarray(mz_list, dtype=np.float64),
                "retained_feature_mz": obj.var["mz"].to_numpy(dtype=np.float64),
                "ms_level": 1,
                "aggregation": "sum",
                "mz_range": [float(mz_range[0]), float(mz_range[1])],
            }
        }
        obj.uns["processing"].update(
            {
                "ppm_tol": float(ppm_tol),
                "resolution": float(resolution),
                "ms_peak_snr_threshold": float(ms_peak_snr_threshold),
                "ms_level": 1,
                "mz_range": [float(mz_range[0]), float(mz_range[1])],
            }
        )
        if log_memory:
            _log_memory_usage(f"After cell extraction {Path(file_path).name}")
        report(1.0, f"完成提取: {obj.n_obs} cells × {obj.n_vars} features")
        return obj

    @classmethod
    def load_from_filelist(
        cls,
        dir_path,
        reference_mz=None,
        *,
        ref_mz=None,
        reference_mode="union",
        frame_metadata=None,
        metadata_by_file=None,
        dtype=np.float32,
        ppm_tol=10,
        mz_range=(100.0, 1000.0),
        resolution=35000,
        resample_points_per_fwhm=5.0,
        ms_peak_snr_threshold=10.0,
        prominence_ratio=None,
        n_jobs=1,
        distance=3,
        show_progress=True,
        progress_callback=None,
        log_memory=True,
        msconvert_path=None,
        **preprocess_kwargs,
    ):
        """Process a directory using a shared feature axis and bounded memory."""
        reference_mz = reference_mz if reference_mz is not None else ref_mz
        if reference_mz is None:
            raise ValueError("reference_mz is required")
        dtype = _validate_float_dtype(dtype)
        root = Path(dir_path)
        files = sorted(
            path for path in root.iterdir()
            if (
                path.suffix.lower() in {".mzml", ".mzxml", ".raw"}
                and (path.is_file() or path.suffix.lower() == ".raw")
            )
        )
        if not files:
            raise FileNotFoundError(f"No mzML/mzXML/RAW files found in {root}")

        total_mz = None
        total_intensity = None
        file_records = []
        for file_index, path in enumerate(tqdm(
            files,
            desc="Shared axis",
            unit="file",
            disable=not show_progress,
        )):
            exp, file_meta = load_single_file(
                str(path), format="auto", msconvert_path=msconvert_path
            )
            file_meta = dict(file_meta)
            file_meta["source_file"] = path.name
            file_meta["path"] = str(path.resolve())
            summed = sum_spec(
                exp,
                mz_range=mz_range,
                resolution_200=resolution,
                points_per_fwhm=resample_points_per_fwhm,
                ms_level=1,
                intensity_dtype=dtype,
            )
            mz_grid, intensity = summed.get_peaks()
            mz_grid = np.asarray(mz_grid, dtype=np.float64)
            intensity = np.asarray(intensity, dtype=np.float64)
            if total_mz is None:
                total_mz = mz_grid.copy()
                total_intensity = intensity.copy()
            else:
                if total_mz.shape != mz_grid.shape or not np.array_equal(total_mz, mz_grid):
                    intensity = np.interp(
                        total_mz, mz_grid, intensity, left=0.0, right=0.0
                    )
                total_intensity += intensity
            file_records.append((path, file_meta))
            del exp, summed, mz_grid, intensity
            _release_memory_to_os()
            if progress_callback is not None:
                progress_callback(
                    0.4 * (file_index + 1) / len(files),
                    f"叠加质谱图: {file_index + 1}/{len(files)} files",
                )

        shared = oms.MSSpectrum()
        shared.setMSLevel(1)
        shared.setRT(0.0)
        shared.set_peaks(
            (
                total_mz.astype(np.float64, copy=False),
                total_intensity.astype(dtype, copy=False),
            )
        )
        shared = filter_spectrum(
            shared,
            snr_threshold=ms_peak_snr_threshold,
            dtype=dtype,
        )
        mz_list, _ = extract_peaks(
            shared,
            dtype=dtype,
            mz_dtype=np.float64,
            prominence_ratio=prominence_ratio,
            distance=distance,
            resolution_200=resolution,
        )
        if len(mz_list) == 0:
            raise ValueError("No m/z features were detected in the shared spectrum")
        shared_spectrum_mz = total_mz.astype(np.float64, copy=True)
        shared_spectrum_intensity = total_intensity.astype(dtype, copy=True)
        del shared, total_mz, total_intensity
        _release_memory_to_os()

        local_kwargs = dict(preprocess_kwargs)
        local_kwargs.setdefault("show_progress", show_progress)
        local_kwargs.setdefault("n_jobs", n_jobs)
        objects = []
        for file_index, (path, file_meta) in enumerate(tqdm(
            file_records,
            desc="Extracting cells",
            unit="file",
            disable=not show_progress,
        )):
            exp, loaded_meta = load_single_file(
                str(path), format="auto", msconvert_path=msconvert_path
            )
            loaded_meta["source_file"] = path.name
            loaded_meta["path"] = str(path.resolve())
            file_meta.update(loaded_meta)
            frame_data, frame_meta = align_frame(
                exp,
                mz_list,
                ppm_tol,
                ms_level=1,
                dtype=dtype,
                mz_dtype=np.float64,
                prominence_ratio=prominence_ratio,
                distance=distance,
                resolution_200=resolution,
            )
            frame_obs = cls._frame_obs(
                frame_meta,
                path,
                file_meta,
                frame_metadata,
                metadata_by_file,
            )
            del exp
            _release_memory_to_os()
            objects.append(
                cls._from_aligned_frames(
                    frame_data,
                    frame_obs,
                    reference_mz=reference_mz,
                    reference_mode=reference_mode,
                    source_file=path,
                    file_meta=file_meta,
                    dtype=dtype,
                    progress_callback=(
                        (lambda value, message, index=file_index: progress_callback(
                            0.4
                            + 0.55
                            * (index + min(max(float(value), 0.0), 1.0))
                            / len(file_records),
                            f"{path.name}: {message}",
                        ))
                        if progress_callback is not None
                        else None
                    ),
                    ppm_tolerance=ppm_tol,
                    **local_kwargs,
                )
            )
            file_meta["n_cells"] = int(objects[-1].n_obs)
            del frame_data, frame_meta
            _release_memory_to_os()
            if progress_callback is not None:
                progress_callback(
                    0.4 + 0.55 * (file_index + 1) / len(file_records),
                    f"文件级细胞提取: {file_index + 1}/{len(file_records)} files",
                )

        all_features = pd.Index(
            sorted(
                {name for obj in objects for name in obj.var_names},
                key=float,
            ),
            name="feature_id",
        )
        X_parts = []
        snr_parts = []
        obs_parts = []
        for obj in objects:
            aligned = pd.DataFrame(
                0.0,
                index=obj.obs_names,
                columns=all_features,
                dtype=dtype,
            )
            aligned.loc[:, obj.var_names] = obj.X
            aligned_snr = aligned.copy()
            aligned_snr.loc[:, :] = 0.0
            aligned_snr.loc[:, obj.var_names] = obj.layers["feature_snr"]
            X_parts.append(aligned)
            snr_parts.append(aligned_snr)
            obs_parts.append(obj.obs.copy())

        combined = pd.concat(X_parts, axis=0)
        combined_snr = pd.concat(snr_parts, axis=0)
        obs = pd.concat(obs_parts, axis=0)
        obs.index = pd.Index(
            [f"cell_{i:08d}" for i in range(len(obs))],
            name="cell_id",
        )
        combined.index = obs.index
        combined_snr.index = obs.index
        zero_fraction = (
            np.mean(combined.to_numpy() == 0, axis=0)
            if len(combined)
            else np.ones(len(all_features))
        )
        var = pd.DataFrame(
            {
                "mz": all_features.astype(float),
                "zero_fraction": zero_fraction,
            },
            index=all_features,
        )
        result = cls(
            X=combined.to_numpy(dtype=dtype, copy=False),
            obs=obs,
            var=var,
            uns={
                "files": {
                    path.name: dict(meta) for path, meta in file_records
                },
                "file_meta": {
                    "name": root.name,
                    "reference_mz": np.atleast_1d(reference_mz).astype(float).tolist(),
                },
                "processing": {
                    "architecture": "shared_axis_sequential",
                    "n_files": len(files),
                    "reference_mode": reference_mode,
                    "ppm_tol": float(ppm_tol),
                    "resolution": float(resolution),
                    "shared_feature_count": int(len(mz_list)),
                    "ms_level": 1,
                    "mz_range": [float(mz_range[0]), float(mz_range[1])],
                },
            },
            layers={
                "feature_snr": combined_snr.to_numpy(dtype=dtype, copy=False)
            },
        )
        result.uns["summed_spectra"] = {
            "shared_all_files": {
                "mz": shared_spectrum_mz,
                "intensity": shared_spectrum_intensity,
                "candidate_mz": np.asarray(mz_list, dtype=np.float64),
                "retained_feature_mz": result.var["mz"].to_numpy(dtype=np.float64),
                "ms_level": 1,
                "aggregation": "sum_across_files",
                "source_files": [path.name for path in files],
                "mz_range": [float(mz_range[0]), float(mz_range[1])],
            }
        }
        if progress_callback is not None:
            progress_callback(
                1.0,
                f"合并结果: {result.n_obs} cells × {result.n_vars} features",
            )
        return result

    @classmethod
    def load_from_directory(
        cls,
        dir_path,
        *,
        feature_strategy="shared",
        feature_merge_ppm=10.0,
        **kwargs,
    ):
        """Process a directory with a shared or per-file feature axis.

        shared builds one summed spectrum across every file. independent performs
        summing and peak picking per file, then merges all retained features by
        ppm while preserving features unique to individual files.
        """
        if feature_strategy == "shared":
            return cls.load_from_filelist(dir_path, **kwargs)
        if feature_strategy == "independent":
            return cls.load_from_filelist_independent(
                dir_path,
                feature_merge_ppm=feature_merge_ppm,
                **kwargs,
            )
        raise ValueError("feature_strategy must be 'shared' or 'independent'")

    @classmethod
    def load_from_filelist_independent(
        cls,
        dir_path,
        *,
        feature_merge_ppm=10.0,
        show_progress=True,
        progress_callback=None,
        **kwargs,
    ):
        """Process every file independently and union features by ppm."""
        root = Path(dir_path)
        files = sorted(
            path for path in root.iterdir()
            if (
                path.suffix.lower() in {".mzml", ".mzxml", ".raw"}
                and (path.is_file() or path.suffix.lower() == ".raw")
            )
        )
        if not files:
            raise FileNotFoundError(f"No supported MS files found in {root}")
        objects = []
        for file_index, path in enumerate(tqdm(
            files,
            desc="Independent peak picking",
            unit="file",
            disable=not show_progress,
        )):
            objects.append(
                cls.load_from_file(
                    path,
                    show_progress=False,
                    progress_callback=(
                        (lambda value, message, index=file_index: progress_callback(
                            0.9
                            * (index + min(max(float(value), 0.0), 1.0))
                            / len(files),
                            f"{path.name}: {message}",
                        ))
                        if progress_callback is not None
                        else None
                    ),
                    **kwargs,
                )
            )
            if progress_callback is not None:
                progress_callback(
                    0.9 * (file_index + 1) / len(files),
                    f"独立质量轴提取: {file_index + 1}/{len(files)} files",
                )
        result = cls._merge_independent_objects(
            objects,
            ppm_tolerance=feature_merge_ppm,
            dataset_name=root.name,
        )
        result.uns["processing"].update(
            {
                "feature_strategy": "independent",
                "n_files": len(files),
                "feature_merge_ppm": float(feature_merge_ppm),
                "ms_level": 1,
                "mz_range": [
                    float(kwargs.get("mz_range", (100.0, 1000.0))[0]),
                    float(kwargs.get("mz_range", (100.0, 1000.0))[1]),
                ],
            }
        )
        if progress_callback is not None:
            progress_callback(
                1.0,
                f"按 ppm 合并结果: {result.n_obs} cells × {result.n_vars} features",
            )
        return result

    @classmethod
    def _merge_independent_objects(
        cls,
        objects,
        *,
        ppm_tolerance=10.0,
        dataset_name="merged",
    ):
        if not objects:
            raise ValueError("objects must contain at least one CyESIData")
        entries = sorted(
            (
                (float(mz), object_index, feature_index)
                for object_index, obj in enumerate(objects)
                for feature_index, mz in enumerate(obj.var["mz"].to_numpy(dtype=float))
            ),
            key=lambda item: item[0],
        )
        clusters = []
        for mz, object_index, feature_index in entries:
            if not clusters:
                clusters.append({"mz": [mz], "members": [(object_index, feature_index)]})
                continue
            center = float(np.median(clusters[-1]["mz"]))
            ppm_error = abs(mz - center) / center * 1e6
            if ppm_error <= ppm_tolerance:
                clusters[-1]["mz"].append(mz)
                clusters[-1]["members"].append((object_index, feature_index))
            else:
                clusters.append({"mz": [mz], "members": [(object_index, feature_index)]})

        centers = np.asarray(
            [np.median(cluster["mz"]) for cluster in clusters], dtype=float
        )
        local_maps = [dict() for _ in objects]
        source_counts = np.zeros(len(clusters), dtype=int)
        for global_index, cluster in enumerate(clusters):
            sources = set()
            for object_index, feature_index in cluster["members"]:
                local_maps[object_index].setdefault(global_index, []).append(feature_index)
                sources.add(object_index)
            source_counts[global_index] = len(sources)

        X_parts = []
        snr_parts = []
        obs_parts = []
        files = {}
        summed_spectra = {}
        for object_index, obj in enumerate(objects):
            local_X = obj.X.toarray() if hasattr(obj.X, "toarray") else np.asarray(obj.X)
            local_snr = np.asarray(
                obj.layers.get("feature_snr", np.zeros_like(local_X))
            )
            merged_X = np.zeros((obj.n_obs, len(clusters)), dtype=local_X.dtype)
            merged_snr = np.zeros_like(merged_X)
            for global_index, local_indices in local_maps[object_index].items():
                merged_X[:, global_index] = np.max(local_X[:, local_indices], axis=1)
                merged_snr[:, global_index] = np.max(local_snr[:, local_indices], axis=1)
            X_parts.append(merged_X)
            snr_parts.append(merged_snr)
            obs_parts.append(obj.obs.copy())
            files.update(dict(obj.uns.get("files", {})))
            summed_spectra.update(dict(obj.uns.get("summed_spectra", {})))

        X = np.vstack(X_parts)
        snr = np.vstack(snr_parts)
        obs = pd.concat(obs_parts, axis=0)
        obs.index = pd.Index(
            [f"cell_{i:08d}" for i in range(len(obs))], name="cell_id"
        )
        feature_ids = pd.Index(
            [f"{value:.8f}" for value in centers], name="feature_id"
        )
        var = pd.DataFrame(
            {
                "mz": centers,
                "zero_fraction": np.mean(X == 0, axis=0),
                "source_file_count": source_counts,
            },
            index=feature_ids,
        )
        result = cls(
            X=X,
            obs=obs,
            var=var,
            layers={"feature_snr": snr},
            uns={
                "files": files,
                "file_meta": {"name": dataset_name},
                "processing": {
                    "architecture": "independent_axis_ppm_union",
                    "merged_feature_count": len(clusters),
                },
            },
        )
        result.uns["summed_spectra"] = summed_spectra
        return result

    def assign_source_metadata(self, metadata_by_file, *, overwrite=False):
        """Assign observation metadata by source_file."""
        if "source_file" not in self.obs:
            raise KeyError("obs['source_file'] is required")
        for source, metadata in metadata_by_file.items():
            source_name = Path(source).name
            source_stem = Path(source).stem
            mask = self.obs["source_file"].astype(str).map(
                lambda value: value == source_name or Path(value).stem == source_stem
            )
            if not mask.any():
                continue
            for key, value in metadata.items():
                if key not in self.obs:
                    self.obs[key] = pd.NA
                if overwrite:
                    self.obs.loc[mask, key] = value
                else:
                    empty = self.obs[key].isna()
                    self.obs.loc[mask & empty, key] = value
        return self

    def impute(self, method="knn", missing_values=0, *, layer_key=None, **kwargs):
        from sklearn.impute import KNNImputer, SimpleImputer

        self.ensure_matrix_layers()
        if method == "knn":
            imputer = KNNImputer(missing_values=missing_values, **kwargs)
        else:
            imputer = SimpleImputer(
                missing_values=missing_values,
                strategy=method,
                **kwargs,
            )
        cell_mask = self.cell_keep_mask()
        feature_mask = self.feature_keep_mask()
        cell_index = np.flatnonzero(cell_mask)
        feature_index = np.flatnonzero(feature_mask)
        matrix = self.X.toarray() if hasattr(self.X, "toarray") else np.array(self.X, copy=True)
        selected = matrix[np.ix_(cell_index, feature_index)]
        transformed = imputer.fit_transform(selected)
        if transformed.shape != selected.shape:
            raise ValueError("imputation removed features; adjust imputer settings")
        matrix[np.ix_(cell_index, feature_index)] = transformed.astype(
            matrix.dtype, copy=False
        )
        self._store_processed_layer(matrix, layer_key or f"imputed_{method}", "impute", {"method": method, "missing_values": missing_values, **kwargs})
        return self

    def remove_outlier(self, key="isolation_forest", **kwargs):
        """Label Isolation Forest inliers without deleting observations."""
        from sklearn.ensemble import IsolationForest

        current = self.cell_keep_mask()
        features = self.feature_keep_mask()
        cell_index = np.flatnonzero(current)
        feature_index = np.flatnonzero(features)
        matrix = self.X.toarray() if hasattr(self.X, "toarray") else np.asarray(self.X)
        inlier = np.ones(self.n_obs, dtype=np.uint8)
        inlier[cell_index] = (
            IsolationForest(**kwargs)
            .fit_predict(matrix[np.ix_(cell_index, feature_index)])
            .astype(np.int8)
            == 1
        ).astype(np.uint8)
        self.set_cell_filter(key, inlier)
        return self

    def normalize(self, method="total", *, layer_key=None, **norm_kwargs):
        self.ensure_matrix_layers()
        cell_mask = self.cell_keep_mask()
        feature_mask = self.feature_keep_mask()
        cell_index = np.flatnonzero(cell_mask)
        feature_index = np.flatnonzero(feature_mask)
        matrix = self.X.toarray() if hasattr(self.X, "toarray") else np.array(self.X, copy=True)
        selected = normalize(
            matrix[np.ix_(cell_index, feature_index)], method, norm_kwargs
        ).astype(matrix.dtype, copy=False)
        matrix[np.ix_(cell_index, feature_index)] = selected
        output_layer = self._store_processed_layer(matrix, layer_key or f"normalized_{method}", "normalize", {"method": method, **norm_kwargs})
        self.uns["normalization"] = {
            "method": method,
            "params": dict(norm_kwargs),
            "layer": output_layer,
        }
        return self

    def get_feature(self, mz):
        """Return the intensity vector for the closest m/z feature."""
        feature_mz = self.var["mz"].to_numpy(dtype=float)
        return np.asarray(self.X[:, int(np.abs(feature_mz - float(mz)).argmin())]).ravel()

    def save(self, path, *, compression="gzip"):
        """Write one self-contained H5AD file and return its path."""
        if self.X is not None:
            self.ensure_matrix_layers()
        target = Path(path)
        if target.suffix.lower() != ".h5ad":
            target = target.with_suffix(".h5ad")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.write_h5ad(target, compression=compression)
        return target

    def export_csv(self, directory):
        """Export X, obs and var as interoperable CSV files."""
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        self.to_df().to_csv(target / "X.csv")
        self.obs.to_csv(target / "obs.csv")
        self.var.to_csv(target / "var.csv")
        return target


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
