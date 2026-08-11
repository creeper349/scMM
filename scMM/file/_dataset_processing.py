"""Mutable processing capabilities composed into :class:`CyESIData`."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer, SimpleImputer

from ..util.normalize import normalize
from ..util.peak import find_cell_peaks
from ._dataset_loading import DatasetState
from ._deisotope import DeisotopeParams, analyze_isotopes, zero_is_missing

DebugHook = Callable[[str, dict[str, Any]], None]
logger = logging.getLogger(__name__)


class DatasetProcessingMixin:
    """Provide preprocessing, transformation, alignment, and deisotoping."""

    def preprocess(
        self,
        baseline_filter=median_filter,
        baseline_filter_size: int = 50,
        cell_snr: float = 5.0,
        peak_snr: float = 3.0,
        max_zero_frac: float = 0.9,
        debug_hook: DebugHook | None = None,
        **kwargs,
    ) -> Self:
        if self.ref_mz is None or not np.isfinite(self.ref_mz) or self.ref_mz <= 0:
            raise ValueError("A positive finite ref_mz is required for preprocessing")
        result = find_cell_peaks(
            self.data,
            self.ref_mz,
            baseline_filter=baseline_filter,
            baseline_filter_size=baseline_filter_size,
            cell_snr=cell_snr,
            peak_snr=peak_snr,
            max_zero_frac=max_zero_frac,
            **kwargs,
        )
        if debug_hook is not None:
            debug_hook(
                "find_cells",
                {
                    "signal": self.data,
                    "baseline": result["baseline"],
                    "cell_mask": result["cell_mask"],
                    "cell_idx": result["peak_frames"],
                },
            )
        self.data = result["cell_df"]
        frames = result["peak_frames"]
        self.peak_meta = pd.DataFrame(
            self.peak_meta.iloc[frames, :],
            index=self.peak_meta.index[frames],
        )
        self.file_meta["length"] = self.data.shape[0]
        return self

    def impute(self, method: str = "knn", missing_values=0, **kwargs) -> Self:
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

    def remove_outlier(self, **kwargs) -> Self:
        inliers = IsolationForest(**kwargs).fit_predict(self.data) == 1
        self.data = self.data.iloc[inliers, :]
        self.peak_meta = self.peak_meta.iloc[inliers, :]
        return self

    def normalize(self, method: str = "total", **norm_kwargs) -> Self:
        logger.info("Normalizing %s with method %s", self.get_name(), method)
        self.data = pd.DataFrame(
            normalize(self.data.values, method, norm_kwargs),
            index=self.data.index,
            columns=self.data.columns,
        )
        return self

    def alignwith(
        self,
        other: Self,
        ppm_tol: float = 5.0,
        mz_merge_options: Literal["union", "ref"] = "union",
    ) -> Self:
        """Append another dataset after aligning its features by m/z."""
        state = merge_dataset_states(self, other, ppm_tol, mz_merge_options)
        self.data = state.data
        self.peak_meta = state.peak_meta
        self.feature_meta = state.feature_meta
        self.file_meta = state.file_meta
        self.ref_mz = state.ref_mz
        return self

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
        """Detect isotope pairs and optionally apply the processed feature set."""
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
        if not inplace:
            return result
        self._apply_deisotope_result(result, merge_mode)
        return self

    def _apply_deisotope_result(self, result: dict, merge_mode: str) -> None:
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


def merge_dataset_states(left, right, ppm_tol: float, merge_mode: str) -> DatasetState:
    """Return merged frames and provenance without mutating either dataset."""
    _validate_dataset_merge(left, right, ppm_tol, merge_mode)
    rename, matched_right = _match_feature_columns(left.data.columns, right.data.columns, ppm_tol)
    right_data = _align_right_data(right.data, left.data.columns, rename, merge_mode)
    merged_data, unmatched_columns = _merge_data_frames(
        left.data,
        right_data,
        right.data.columns,
        matched_right,
        merge_mode,
    )
    feature_meta = _merge_feature_metadata(
        left,
        right,
        merged_data.columns,
        unmatched_columns,
        merge_mode,
    )
    return DatasetState(
        data=merged_data,
        peak_meta=pd.concat(
            [left.peak_meta, right.peak_meta],
            axis=0,
            ignore_index=True,
            sort=False,
        ),
        feature_meta=feature_meta,
        file_meta={
            "name": f"{left.get_name()}+{right.get_name()}",
            "ref_mz": left.ref_mz,
            "per_file_meta": [*_source_metadata(left), *_source_metadata(right)],
        },
        ref_mz=left.ref_mz,
    )


def _validate_dataset_merge(left, right, ppm_tol: float, merge_mode: str) -> None:
    if merge_mode not in {"union", "ref"}:
        raise ValueError("mz_merge_options must be either 'union' or 'ref'")
    if ppm_tol < 0:
        raise ValueError("ppm_tol must be non-negative")
    if left.data.columns.has_duplicates or right.data.columns.has_duplicates:
        raise ValueError("Feature columns must be unique before alignment")


def _match_feature_columns(left_columns, right_columns, ppm_tol: float):
    left_mz = left_columns.to_numpy(dtype=float)
    right_mz = right_columns.to_numpy(dtype=float)
    right_order = np.argsort(right_mz)
    right_sorted = right_mz[right_order]
    matched_right: set[int] = set()
    rename: dict[object, object] = {}
    right_start = 0
    for left_index in np.argsort(left_mz):
        target = left_mz[left_index]
        lower, upper = target * (1 - ppm_tol * 1e-6), target * (1 + ppm_tol * 1e-6)
        while right_start < len(right_sorted) and right_sorted[right_start] < lower:
            right_start += 1
        stop = int(np.searchsorted(right_sorted, upper, side="right"))
        candidates = [
            int(right_order[position])
            for position in range(right_start, stop)
            if int(right_order[position]) not in matched_right
        ]
        if candidates:
            match = min(candidates, key=lambda index: abs(right_mz[index] - target))
            matched_right.add(match)
            rename[right_columns[match]] = left_columns[left_index]
    return rename, matched_right


def _align_right_data(right, left_columns, rename: dict, merge_mode: str) -> pd.DataFrame:
    renamed = right.copy().rename(columns=rename)
    if merge_mode == "union":
        return renamed
    aligned = pd.DataFrame(0.0, index=right.index, columns=left_columns)
    for column in rename.values():
        aligned[column] = renamed[column].to_numpy()
    return aligned


def _merge_data_frames(left, right, original_right_columns, matched_right, merge_mode: str):
    merged = pd.concat([left.copy(), right], axis=0, ignore_index=True, sort=False).fillna(0)
    unmatched = [
        original_right_columns[index]
        for index in range(len(original_right_columns))
        if index not in matched_right
    ]
    if merge_mode == "union":
        merged = merged.loc[:, [*left.columns, *unmatched]]
    return merged, unmatched


def _merge_feature_metadata(left, right, columns, unmatched, merge_mode: str) -> pd.DataFrame:
    metadata = left.feature_meta.reindex(left.data.columns).copy()
    if merge_mode == "union":
        metadata = pd.concat([metadata, right.feature_meta.reindex(unmatched).copy()], axis=0)
    metadata = metadata.reindex(columns)
    metadata.index.name = "feature_id"
    return metadata


def _source_metadata(dataset) -> list[dict]:
    return dataset.file_meta.get("per_file_meta", [dataset.file_meta])
