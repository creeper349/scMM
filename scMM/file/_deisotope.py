"""Pure helpers for isotope-pair detection and feature processing.

This module keeps the numerical deisotoping workflow separate from the mutable
``CyESIData`` container.  The container owns API state; the helpers below own
candidate detection, regression, assignment, and metadata construction.
"""

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MissingValueDetector = Callable[[np.ndarray], np.ndarray]


def zero_is_missing(values: np.ndarray) -> np.ndarray:
    """Return the default missing-value mask used during regression."""
    return values == 0


@dataclass(frozen=True)
class DeisotopeParams:
    """Validated parameters that control isotope-pair detection."""

    isotope_diff: float = 1.003355
    ppm_tol: float = 1.0
    max_isotope_order: int = 3
    r_square_threshold: float = 0.95
    carbon13_abundance: float = 0.0109
    intensity_threshold: float = 0.0
    safety_factor: float = 1.0
    merge_mode: str = "keep_parent"
    remove: bool = True

    def validate(self, data: pd.DataFrame) -> None:
        """Reject unsupported data shapes and parameter ranges."""
        if data.shape[1] == 0:
            raise ValueError("Cannot deisotope a dataset without features")
        if self.isotope_diff <= 0:
            raise ValueError("isotope_diff must be positive")
        if self.ppm_tol < 0:
            raise ValueError("ppm_tol must be non-negative")
        if self.max_isotope_order < 1:
            raise ValueError("max_isotope_order must be at least 1")
        if not 0 <= self.r_square_threshold <= 1:
            raise ValueError("r_square_threshold must be between 0 and 1")
        if not 0 < self.carbon13_abundance < 1:
            raise ValueError("carbon13_abundance must be between 0 and 1")
        if self.safety_factor <= 0:
            raise ValueError("safety_factor must be positive")
        if self.merge_mode not in {"keep_parent", "sum"}:
            raise ValueError("merge_mode must be either 'keep_parent' or 'sum'.")

    def to_dict(self) -> dict[str, float | int | str | bool]:
        """Return the serializable parameter record stored with results."""
        return asdict(self)


@dataclass(frozen=True)
class CandidatePairs:
    """Mass-based isotope candidates and their pairwise measurements."""

    mz: np.ndarray
    isotope_order: np.ndarray
    ppm_error: np.ndarray
    mask: np.ndarray
    table: pd.DataFrame
    feature_map: dict[str, list[str]]


@dataclass(frozen=True)
class RegressionMatrices:
    """Through-origin slopes and coefficients of determination."""

    slope: np.ndarray
    r_square: np.ndarray


@dataclass(frozen=True)
class IsotopeAssignments:
    """Greedily selected parent/isotope relationships."""

    parent_indices: np.ndarray
    isotope_indices: np.ndarray
    table: pd.DataFrame
    isotope_features: list
    parent_features: list


def analyze_isotopes(
    data: pd.DataFrame,
    feature_meta: pd.DataFrame | None,
    params: DeisotopeParams,
    missing_func: MissingValueDetector = zero_is_missing,
) -> dict:
    """Run deisotoping without mutating the source data or metadata."""
    params.validate(data)
    candidates = _find_mass_candidates(data, params)
    regression = _fit_pairwise_regressions(data, missing_func, params.intensity_threshold)
    ratio_limit = _calculate_ratio_limits(candidates, params)
    assignments = _select_assignments(data, candidates, regression, ratio_limit, params)
    metadata = _build_feature_metadata(
        data,
        feature_meta,
        assignments,
        params.max_isotope_order,
    )
    processed_data, metadata = _process_feature_data(data, metadata, assignments, params)

    columns = data.columns
    return {
        "candidate_map": candidates.feature_map,
        "candidate_table": candidates.table,
        "A": pd.DataFrame(regression.slope, index=columns, columns=columns),
        "R": pd.DataFrame(regression.r_square, index=columns, columns=columns),
        "ratio_limit": pd.DataFrame(ratio_limit, index=columns, columns=columns),
        "final_table": assignments.table,
        "isotope_features": assignments.isotope_features,
        "parent_features": assignments.parent_features,
        "processed_data": processed_data,
        "feature_meta": metadata,
        "params": params.to_dict(),
    }


def _find_mass_candidates(data: pd.DataFrame, params: DeisotopeParams) -> CandidatePairs:
    mz = data.columns.astype(float).to_numpy()
    mass_delta = mz[None, :] - mz[:, None]
    isotope_order = np.rint(mass_delta / params.isotope_diff).astype(int)
    expected_delta = isotope_order * params.isotope_diff
    ppm_error = np.abs(mass_delta - expected_delta) / mz[:, None] * 1e6
    mask = (
        (isotope_order >= 1)
        & (isotope_order <= params.max_isotope_order)
        & (mass_delta > 0)
        & (ppm_error <= params.ppm_tol)
    )
    parent_indices, isotope_indices = np.where(mask)
    table = pd.DataFrame(
        {
            "parent_index": parent_indices,
            "isotope_index": isotope_indices,
            "parent_feature": data.columns[parent_indices],
            "isotope_feature": data.columns[isotope_indices],
            "parent_mz": mz[parent_indices],
            "isotope_mz": mz[isotope_indices],
            "isotope_order": isotope_order[parent_indices, isotope_indices],
            "ppm_error": ppm_error[parent_indices, isotope_indices],
        }
    )
    feature_map = {
        str(data.columns[index]): [str(feature) for feature in data.columns[mask[index]]]
        for index in range(len(mz))
        if np.any(mask[index])
    }
    return CandidatePairs(mz, isotope_order, ppm_error, mask, table, feature_map)


def _fit_pairwise_regressions(
    data: pd.DataFrame,
    missing_func: MissingValueDetector,
    intensity_threshold: float,
) -> RegressionMatrices:
    raw_values = data.to_numpy(dtype=float)
    missing = np.asarray(missing_func(raw_values), dtype=bool)
    if missing.shape != raw_values.shape:
        raise ValueError(
            "missing_func must return a boolean array with the same shape as input data."
        )

    missing |= ~np.isfinite(raw_values)
    if intensity_threshold > 0:
        missing |= raw_values <= intensity_threshold

    values = raw_values.copy()
    values[missing] = 0.0
    valid = (~missing).astype(float)
    cross_products = values.T @ values
    squared = values**2
    parent_sums = squared.T @ valid
    isotope_sums = valid.T @ squared

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = cross_products / parent_sums
        r_square = cross_products**2 / (parent_sums * isotope_sums)
    slope[~np.isfinite(slope)] = np.nan
    r_square[~np.isfinite(r_square)] = np.nan
    return RegressionMatrices(slope, r_square)


def _calculate_ratio_limits(
    candidates: CandidatePairs,
    params: DeisotopeParams,
) -> np.ndarray:
    max_carbon_atoms = np.floor(candidates.mz / 12.0).astype(int)
    abundance_ratio = params.carbon13_abundance / (1.0 - params.carbon13_abundance)
    ratio_limit = np.full(candidates.mask.shape, np.nan, dtype=float)

    for order in range(1, params.max_isotope_order + 1):
        limits = np.array(
            [
                math.comb(int(atom_count), order) * abundance_ratio**order
                if atom_count >= order
                else 0.0
                for atom_count in max_carbon_atoms
            ]
        )
        order_mask = candidates.isotope_order == order
        expanded_limits = np.broadcast_to(limits[:, None], ratio_limit.shape)
        ratio_limit[order_mask] = expanded_limits[order_mask]

    return ratio_limit * params.safety_factor


def _select_assignments(
    data: pd.DataFrame,
    candidates: CandidatePairs,
    regression: RegressionMatrices,
    ratio_limit: np.ndarray,
    params: DeisotopeParams,
) -> IsotopeAssignments:
    accepted = (
        candidates.mask
        & (params.r_square_threshold <= regression.r_square)
        & (ratio_limit >= regression.slope)
    )
    selected = np.zeros_like(accepted, dtype=bool)
    assigned_isotopes: set[int] = set()

    for parent_index in np.argsort(candidates.mz):
        if parent_index in assigned_isotopes:
            continue
        isotope_indices = sorted(
            np.where(accepted[parent_index])[0],
            key=lambda index: (
                candidates.isotope_order[parent_index, index],
                candidates.mz[index],
            ),
        )
        for isotope_index in isotope_indices:
            if isotope_index not in assigned_isotopes:
                selected[parent_index, isotope_index] = True
                assigned_isotopes.add(isotope_index)

    parent_indices, isotope_indices = np.where(selected)
    table = _build_assignment_table(
        data,
        candidates,
        regression,
        ratio_limit,
        parent_indices,
        isotope_indices,
    )
    return IsotopeAssignments(
        parent_indices=parent_indices,
        isotope_indices=isotope_indices,
        table=table,
        isotope_features=data.columns[sorted(assigned_isotopes)].tolist(),
        parent_features=data.columns[sorted(set(parent_indices))].tolist(),
    )


def _build_assignment_table(
    data: pd.DataFrame,
    candidates: CandidatePairs,
    regression: RegressionMatrices,
    ratio_limit: np.ndarray,
    parent_indices: np.ndarray,
    isotope_indices: np.ndarray,
) -> pd.DataFrame:
    pair = (parent_indices, isotope_indices)
    return pd.DataFrame(
        {
            "parent_index": parent_indices,
            "isotope_index": isotope_indices,
            "parent_feature": data.columns[parent_indices],
            "isotope_feature": data.columns[isotope_indices],
            "parent_mz": candidates.mz[parent_indices],
            "isotope_mz": candidates.mz[isotope_indices],
            "isotope_order": candidates.isotope_order[pair],
            "ppm_error": candidates.ppm_error[pair],
            "slope_A": regression.slope[pair],
            "r_square": regression.r_square[pair],
            "max_allowed_ratio": ratio_limit[pair],
        }
    )


def _build_feature_metadata(
    data: pd.DataFrame,
    feature_meta: pd.DataFrame | None,
    assignments: IsotopeAssignments,
    max_isotope_order: int,
) -> pd.DataFrame:
    metadata = _initialize_feature_metadata(data, feature_meta, max_isotope_order)
    metadata.loc[assignments.parent_features, "deisotope_role"] = "parent"
    metadata.loc[assignments.isotope_features, "deisotope_role"] = "isotope"

    for row in assignments.table.itertuples(index=False):
        _record_isotope_relationship(metadata, row)
    return metadata


def _initialize_feature_metadata(
    data: pd.DataFrame,
    feature_meta: pd.DataFrame | None,
    max_isotope_order: int,
) -> pd.DataFrame:
    metadata = (
        pd.DataFrame(index=data.columns)
        if feature_meta is None
        else feature_meta.reindex(data.columns)
    ).copy()
    if "mz" not in metadata:
        metadata["mz"] = data.columns.astype(float).to_numpy()

    metadata["deisotope_role"] = "unique"
    metadata["isotope_parent"] = pd.NA
    metadata["isotope_order"] = pd.NA
    metadata["isotope_slope_A"] = np.nan
    metadata["isotope_r_square"] = np.nan
    metadata["isotope_ppm_error"] = np.nan
    metadata["isotope_children"] = "[]"
    metadata["n_isotope_children"] = 0
    for order in range(1, max_isotope_order + 1):
        metadata[f"M{order}_mz"] = np.nan
        metadata[f"M{order}_feature"] = pd.NA
        metadata[f"M{order}_slope_A"] = np.nan
        metadata[f"M{order}_r_square"] = np.nan
        metadata[f"M{order}_ppm_error"] = np.nan
        metadata[f"M{order}_max_allowed_ratio"] = np.nan
    return metadata


def _record_isotope_relationship(metadata: pd.DataFrame, row) -> None:
    parent = row.parent_feature
    isotope = row.isotope_feature
    order = int(row.isotope_order)
    metadata.loc[isotope, "isotope_parent"] = parent
    metadata.loc[isotope, "isotope_order"] = order
    metadata.loc[isotope, "isotope_slope_A"] = row.slope_A
    metadata.loc[isotope, "isotope_r_square"] = row.r_square
    metadata.loc[isotope, "isotope_ppm_error"] = row.ppm_error

    children = json.loads(metadata.loc[parent, "isotope_children"])
    children.append(
        {
            "feature": str(isotope),
            "mz": float(row.isotope_mz),
            "order": order,
            "slope_A": float(row.slope_A),
            "r_square": float(row.r_square),
            "ppm_error": float(row.ppm_error),
            "max_allowed_ratio": float(row.max_allowed_ratio),
        }
    )
    metadata.loc[parent, "isotope_children"] = json.dumps(children, ensure_ascii=False)
    metadata.loc[parent, "n_isotope_children"] = len(children)
    metadata.loc[parent, f"M{order}_mz"] = row.isotope_mz
    metadata.loc[parent, f"M{order}_feature"] = isotope
    metadata.loc[parent, f"M{order}_slope_A"] = row.slope_A
    metadata.loc[parent, f"M{order}_r_square"] = row.r_square
    metadata.loc[parent, f"M{order}_ppm_error"] = row.ppm_error
    metadata.loc[parent, f"M{order}_max_allowed_ratio"] = row.max_allowed_ratio


def _process_feature_data(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    assignments: IsotopeAssignments,
    params: DeisotopeParams,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = data.copy()
    if params.merge_mode == "sum":
        for parent_index, isotope_index in zip(
            assignments.parent_indices,
            assignments.isotope_indices,
            strict=True,
        ):
            parent = data.columns[parent_index]
            isotope = data.columns[isotope_index]
            processed[parent] = processed[parent].fillna(0) + data[isotope].fillna(0)

    if params.remove:
        processed = processed.drop(columns=assignments.isotope_features)
        metadata = metadata.loc[processed.columns].copy()
    return processed, metadata
