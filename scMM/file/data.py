import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.ndimage import median_filter
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer

from ..util.annotation import SDFMzSearcher
from ..util.normalize import normalize
from ..util.peak import filter_spectrum, find_cell_peaks
from ._deisotope import DeisotopeParams, analyze_isotopes, zero_is_missing
from .io import (
    align_frame,
    extract_peaks,
    load_single_file,
    pack_specs,
    sum_spec,
    sum_spectrum_from_file,
)

DebugHook = Callable[[str, dict[str, Any]], None]
logger = logging.getLogger(__name__)


def _align_frame_from_file(file_path: str, mz_list, ppm_tol: int = 10, dtype=np.float64):
    exp, file_meta = load_single_file(file_path, format="auto")
    logger.info("Aligning frames from MS file %s", file_path)
    data, peak_meta = align_frame(exp, mz_list, ppm_tol, dtype=dtype)
    return {
        "file_meta": file_meta,
        "data": data,
        "peak_meta": peak_meta,
    }


class CyESIData:
    """Processed CyESI dataset and its observation/feature metadata."""

    def __init__(self, result_dir: str | Path):
        result_path = Path(result_dir).expanduser()
        with (result_path / ".meta").open(encoding="utf-8") as fp:
            file_meta = json.load(fp)

        data_path_pkl = result_path / "data.pkl"
        peak_path_pkl = result_path / "peak_meta.pkl"
        feature_path_pkl = result_path / "feature_meta.pkl"
        data = None
        peak_meta = None
        feature_meta = None
        if data_path_pkl.exists():
            data = pd.read_pickle(data_path_pkl)
            if peak_path_pkl.exists():
                peak_meta = pd.read_pickle(peak_path_pkl)
            if feature_path_pkl.exists():
                feature_meta = pd.read_pickle(feature_path_pkl)
        else:
            data_path_csv = result_path / "data.csv"
            peak_path_csv = result_path / "peak_meta.csv"
            feature_path_csv = result_path / "feature_meta.csv"
            if data_path_csv.exists():
                data = pd.read_csv(data_path_csv, index_col=0)
            if peak_path_csv.exists():
                peak_meta = pd.read_csv(peak_path_csv, index_col=0)
            if feature_path_csv.exists():
                feature_meta = pd.read_csv(feature_path_csv, index_col=0)

        if data is None:
            raise FileNotFoundError(f"No processed data found in {result_path}")
        self.data = data
        self.peak_meta = (
            peak_meta if peak_meta is not None else pd.DataFrame(index=self.data.index.copy())
        )
        self.file_meta = file_meta
        self.ref_mz = file_meta.get("ref_mz")

        if feature_meta is None:
            feature_meta = pd.DataFrame(
                {"mz": self.data.columns.astype(float)}, index=self.data.columns.copy()
            )
        elif len(feature_meta) == self.data.shape[1]:
            feature_meta = feature_meta.copy()
            feature_meta.index = self.data.columns.copy()
        else:
            raise ValueError(
                "feature_meta row count does not match the number of data features: "
                f"{len(feature_meta)} != {self.data.shape[1]}"
            )
        feature_meta.index.name = "feature_id"
        self.feature_meta = feature_meta

    @classmethod
    def load_from_processed(cls, result_dir: str | Path) -> Self:
        """Load a dataset previously written by :meth:`save`."""
        return cls(result_dir)

    @classmethod
    def load_from_file(
        cls,
        file_path: str | Path,
        ref_mz: float,
        dtype=np.float64,
        ppm_tol: int = 10,
        resolution: float = 35000,
        resample_points_per_fwhm: float = 5.0,
        ms_peak_snr_threshold: float = 10.0,
        prominence_ratio: float | None = None,
        distance: int = 3,
        **preprocess_kwds,
    ):
        if not np.isfinite(ref_mz) or ref_mz <= 0:
            raise ValueError("ref_mz must be a positive finite number")
        obj = object.__new__(cls)
        exp, obj.file_meta = load_single_file(str(file_path), format="auto")
        sum_ = sum_spec(exp, resolution_200=resolution, points_per_fwhm=resample_points_per_fwhm)
        obj.file_meta["ref_mz"] = ref_mz

        sum_ = filter_spectrum(sum_, snr_threshold=ms_peak_snr_threshold)
        mz_list, _ = extract_peaks(
            sum_, dtype=dtype, prominence_ratio=prominence_ratio, distance=distance
        )
        obj.data, obj.peak_meta = align_frame(exp, mz_list, ppm_tol, dtype=dtype)
        max_rt = float(obj.peak_meta["rt"].max())
        obj.peak_meta["time"] = obj.peak_meta["rt"] / max_rt if max_rt > 0 else 0.0
        obj.peak_meta["label"] = [obj.file_meta["name"].split(".")[0]] * len(obj.peak_meta)
        obj.ref_mz = ref_mz
        obj.preprocess(**preprocess_kwds)
        obj.feature_meta = pd.DataFrame(
            {"mz": obj.data.columns.astype(float)}, index=obj.data.columns.copy()
        )
        obj.feature_meta.index.name = "feature_id"
        return obj

    @classmethod
    def load_from_filelist(
        cls,
        dir_path: str | Path,
        ref_mz: float,
        dtype=np.float64,
        ppm_tol: int = 10,
        resolution: float = 35000,
        resample_points_per_fwhm: float = 5.0,
        ms_peak_snr_threshold: float = 10.0,
        prominence_ratio: float | None = None,
        n_jobs: int = -1,
        distance: int = 3,
        **preprocess_kwds,
    ):
        if not np.isfinite(ref_mz) or ref_mz <= 0:
            raise ValueError("ref_mz must be a positive finite number")
        directory = Path(dir_path).expanduser()
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        filelist = sorted(
            str(path)
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".mzml", ".mzxml"}
        )
        if not filelist:
            raise FileNotFoundError(f"No mzML or mzXML files found in {directory}")
        logger.info("Detected %d MS files in %s", len(filelist), directory)
        logger.info(
            "Summing spectra (resolution=%s, points_per_fwhm=%s)",
            resolution,
            resample_points_per_fwhm,
        )
        _sum_specs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(sum_spectrum_from_file)(
                f, resolution_200=resolution, points_per_fwhm=resample_points_per_fwhm
            )
            for f in filelist
        )
        total_sum_spec = sum_spec(
            pack_specs(_sum_specs),
            resolution_200=resolution,
            points_per_fwhm=resample_points_per_fwhm,
        )

        logger.info("Denoising summed spectrum (snr_threshold=%s)", ms_peak_snr_threshold)
        total_sum_spec = filter_spectrum(total_sum_spec, snr_threshold=ms_peak_snr_threshold)
        logger.info("Picking peaks from summed spectrum")
        mz_list, _ = extract_peaks(
            total_sum_spec, prominence_ratio=prominence_ratio, distance=distance
        )
        align_results = Parallel(n_jobs=n_jobs)(
            delayed(_align_frame_from_file)(fp, mz_list, ppm_tol=ppm_tol, dtype=dtype)
            for fp in filelist
        )

        logger.info("Building data container")
        align_results = sorted(align_results, key=lambda x: x["file_meta"]["timestamp"])
        obj = cls.__new__(cls)
        obj.data = None
        obj.peak_meta = None
        timestamp_start = align_results[0]["file_meta"]["timestamp"]
        timestamp_end = (
            align_results[-1]["file_meta"]["timestamp"]
            + align_results[-1]["peak_meta"]["rt"].iloc[-1]
        )
        elapsed = timestamp_end - timestamp_start
        if elapsed <= 0:
            raise ValueError("Acquisition timestamps must span a positive duration")

        per_file_meta = []

        for r in align_results:
            data = r["data"]
            peak_meta = r["peak_meta"]
            file_meta = dict(r["file_meta"])

            if isinstance(data, pd.DataFrame):
                data = data.copy()
                peak_meta["time"] = (
                    peak_meta["rt"] + file_meta["timestamp"] - timestamp_start
                ) / elapsed
                peak_meta["label"] = [file_meta["name"].split(".")[0]] * len(peak_meta)
                if obj.data is None:
                    obj.data = data
                else:
                    obj.data = pd.concat([obj.data, data], axis=0)

            if isinstance(peak_meta, pd.DataFrame):
                peak_meta = peak_meta.copy()

                if obj.peak_meta is None:
                    obj.peak_meta = peak_meta
                else:
                    obj.peak_meta = pd.concat([obj.peak_meta, peak_meta], axis=0)

            per_file_meta.append(file_meta)

        obj.file_meta = {
            "name": directory.name,
            "ref_mz": ref_mz,
            "per_file_meta": per_file_meta,
        }
        obj.ref_mz = ref_mz

        logger.info("Denoising aligned data and picking cells")
        obj.preprocess(**preprocess_kwds)
        obj.feature_meta = pd.DataFrame(
            {"mz": obj.data.columns.astype(float)}, index=obj.data.columns.copy()
        )
        obj.feature_meta.index.name = "feature_id"
        return obj

    def preprocess(
        self,
        baseline_filter=median_filter,
        baseline_filter_size: int = 50,
        cell_snr: float = 5.0,
        peak_snr: float = 3.0,
        max_zero_frac: float = 0.9,
        debug_hook: DebugHook | None = None,
        **kwargs,
    ):
        if self.ref_mz is None or not np.isfinite(self.ref_mz) or self.ref_mz <= 0:
            raise ValueError("A positive finite ref_mz is required for preprocessing")

        def emit(stage: str, **payload):
            if debug_hook is not None:
                debug_hook(stage, payload)

        data = find_cell_peaks(
            self.data,
            self.ref_mz,
            baseline_filter=baseline_filter,
            baseline_filter_size=baseline_filter_size,
            cell_snr=cell_snr,
            peak_snr=peak_snr,
            max_zero_frac=max_zero_frac,
            **kwargs,
        )
        emit(
            "find_cells",
            signal=self.data,
            baseline=data["baseline"],
            cell_mask=data["cell_mask"],
            cell_idx=data["peak_frames"],
        )
        self.data, self.peak_meta = (
            data["cell_df"],
            pd.DataFrame(
                self.peak_meta.iloc[data["peak_frames"], :],
                index=self.peak_meta.index[data["peak_frames"]],
            ),
        )
        self.file_meta["length"] = self.data.shape[0]
        return self

    def impute(self, method: str = "knn", missing_values=0, **kwargs):
        logger.info("Imputing %s with method %s", self.get_name(), method)
        if method == "knn":
            imputer = KNNImputer(missing_values=missing_values, **kwargs)
        else:
            kwargs.setdefault("keep_empty_features", True)
            imputer = SimpleImputer(missing_values=missing_values, strategy=method, **kwargs)

        self.data = pd.DataFrame(
            imputer.fit_transform(self.data),
            index=self.data.index,
            columns=self.data.columns,
        )
        return self

    def remove_outlier(self, **kwargs):
        iso = IsolationForest(**kwargs)
        inlier_id = iso.fit_predict(self.data) == 1
        self.data = self.data.iloc[inlier_id, :]
        self.peak_meta = self.peak_meta.iloc[inlier_id, :]
        return self

    def normalize(self, method: str = "total", **norm_kwargs):
        logger.info("Normalizing %s with method %s", self.get_name(), method)
        self.data = pd.DataFrame(
            normalize(self.data.values, method, norm_kwargs),
            index=self.data.index,
            columns=self.data.columns,
        )
        return self

    def alignwith(
        self, other: Self, ppm_tol: float = 5.0, mz_merge_options: Literal["union", "ref"] = "union"
    ):
        """Append another dataset after aligning its features by m/z.

        ``union`` retains unmatched features from both datasets and fills their
        absent measurements with zero. ``ref`` retains the current dataset's
        feature axis only.
        """
        if mz_merge_options not in {"union", "ref"}:
            raise ValueError("mz_merge_options must be either 'union' or 'ref'")
        if ppm_tol < 0:
            raise ValueError("ppm_tol must be non-negative")
        if self.data.columns.has_duplicates or other.data.columns.has_duplicates:
            raise ValueError("Feature columns must be unique before alignment")

        left = self.data.copy()
        right = other.data.copy()
        left_mz = left.columns.to_numpy(dtype=float)
        right_mz = right.columns.to_numpy(dtype=float)

        matched_right: set[int] = set()
        rename: dict[object, object] = {}
        right_order = np.argsort(right_mz)
        right_sorted = right_mz[right_order]
        right_start = 0
        for left_idx in np.argsort(left_mz):
            target = left_mz[left_idx]
            lower = target * (1 - ppm_tol * 1e-6)
            upper = target * (1 + ppm_tol * 1e-6)
            while right_start < len(right_sorted) and right_sorted[right_start] < lower:
                right_start += 1
            stop = int(np.searchsorted(right_sorted, upper, side="right"))
            candidates = [
                int(right_order[pos])
                for pos in range(right_start, stop)
                if int(right_order[pos]) not in matched_right
            ]
            if candidates:
                right_idx = min(candidates, key=lambda idx: abs(right_mz[idx] - target))
                matched_right.add(right_idx)
                rename[right.columns[right_idx]] = left.columns[left_idx]

        right = right.rename(columns=rename)
        if mz_merge_options == "ref":
            aligned_right = pd.DataFrame(0.0, index=right.index, columns=left.columns)
            for column in rename.values():
                aligned_right[column] = right[column].to_numpy()
            right = aligned_right

        merged_df = pd.concat([left, right], axis=0, ignore_index=True, sort=False).fillna(0)
        if mz_merge_options == "union":
            unmatched_columns = [
                other.data.columns[idx]
                for idx in range(other.data.shape[1])
                if idx not in matched_right
            ]
            merged_df = merged_df.loc[:, [*left.columns, *unmatched_columns]]

        name = f"{self.get_name()}+{other.get_name()}"
        left_sources = self.file_meta.get("per_file_meta", [self.file_meta])
        right_sources = other.file_meta.get("per_file_meta", [other.file_meta])

        feature_meta = self.feature_meta.reindex(left.columns).copy()
        if mz_merge_options == "union":
            other_unmatched = other.feature_meta.reindex(unmatched_columns).copy()
            feature_meta = pd.concat([feature_meta, other_unmatched], axis=0)
        feature_meta = feature_meta.reindex(merged_df.columns)
        feature_meta.index.name = "feature_id"

        self.data = merged_df
        self.peak_meta = pd.concat(
            [self.peak_meta, other.peak_meta], axis=0, ignore_index=True, sort=False
        )
        self.feature_meta = feature_meta
        self.file_meta = {
            "name": name,
            "ref_mz": self.ref_mz,
            "per_file_meta": [*left_sources, *right_sources],
        }
        return self

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, key):
        if self.data.shape[1] == 0:
            raise KeyError("Dataset has no features")
        key = float(key)
        idx = (np.abs(self.data.columns.values.astype(float) - key)).argmin()
        return self.data.iloc[:, idx].values

    def get_name(self) -> str:
        """Return the human-readable dataset name."""
        return str(self.file_meta.get("name", "unnamed"))

    def get_labels(self, mapping: dict | None = None) -> np.ndarray:
        """Return per-cell labels, optionally replacing them with ``mapping``."""
        if "label" not in self.peak_meta:
            raise KeyError("peak_meta does not contain a 'label' column")
        labels = self.peak_meta["label"]
        if mapping is not None:
            labels = labels.map(lambda value: mapping.get(value, value))
        return labels.to_numpy()

    def save(self, root_path: str | Path, *, overwrite: bool = False) -> Path:
        """Save the dataset below ``root_path`` and return its directory."""
        root = Path(root_path).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(self.get_name()).name
        if dataset_name in {"", ".", ".."}:
            raise ValueError(f"Invalid dataset name: {dataset_name!r}")
        result_path = root / dataset_name
        result_path.mkdir(exist_ok=overwrite)
        logger.info("Saving processed data to %s", result_path)

        with (result_path / ".meta").open("w", encoding="utf-8") as fp:
            json.dump(self.file_meta, fp, ensure_ascii=False, indent=2)

        for name, frame in (
            ("data", self.data),
            ("peak_meta", self.peak_meta),
            ("feature_meta", self.feature_meta),
        ):
            frame.to_pickle(result_path / f"{name}.pkl")
            frame.to_csv(result_path / f"{name}.csv")
        return result_path

    def to_anndata(self):
        if len(self.peak_meta) != len(self.data):
            raise ValueError("peak_meta row count must match data row count")
        if len(self.feature_meta) != self.data.shape[1]:
            raise ValueError("feature_meta row count must match data column count")

        obs_df = self.peak_meta.copy()
        obs_df.insert(0, "source_index", self.peak_meta.index.astype(str))
        obs_df.index = pd.Index([f"cell_{idx}" for idx in range(len(obs_df))], name="cell_id")
        var_df = self.feature_meta.reindex(self.data.columns).copy()
        var_df.index = pd.Index(self.data.columns.astype(str), name="feature_id")

        adata = AnnData(
            X=self.data.values,
            obs=obs_df,
            var=var_df,
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
        missing_func: Callable[[np.ndarray], np.ndarray] = zero_is_missing,
        merge_mode: str = "keep_parent",
        remove: bool = True,
        inplace: bool = True,
    ):
        """Detect isotope pairs and optionally apply the processed feature set.

        ``inplace=False`` returns a complete audit dictionary without changing
        the dataset.  Numerical detection and metadata assembly live in the
        dedicated ``_deisotope`` module; this method only manages container
        state and provenance.
        """
        params = DeisotopeParams(
            isotope_diff=isotope_diff,
            ppm_tol=ppm_tol,
            max_isotope_order=max_isotope_order,
            r_square_threshold=r_square_threshold,
            carbon13_abundance=carbon13_abundance,
            intensity_threshold=intensity_threshold,
            safety_factor=safety_factor,
            merge_mode=merge_mode,
            remove=remove,
        )
        result = analyze_isotopes(
            self.data,
            getattr(self, "feature_meta", None),
            params,
            missing_func,
        )

        if inplace:
            self.deisotope_result = result
            self.data = result["processed_data"]
            self.feature_meta = result["feature_meta"]

            if not hasattr(self, "file_meta") or self.file_meta is None:
                self.file_meta = {}

            self.file_meta["deisotope"] = {
                "params": result["params"],
                "n_candidate_pairs": len(result["candidate_table"]),
                "n_final_isotope_pairs": len(result["final_table"]),
                "n_removed_features": len(result["isotope_features"]),
                "merge_mode": merge_mode,
            }

            return self

        return result

    def get_annotation(
        self,
        sdf_path: str | Path,
        ppm_tol: float,
        search_mode: Literal["pos", "neg", "both"] = "pos",
        adducts_pos: dict | None = None,
        adducts_neg: dict | None = None,
        **kwargs,
    ):
        searcher = SDFMzSearcher(
            sdf_path=sdf_path, adducts_pos=adducts_pos, adducts_neg=adducts_neg
        )
        res = searcher.search(
            mz=self.data.columns.astype(float), ppm_tol=ppm_tol, mode=search_mode, **kwargs
        )
        return res
