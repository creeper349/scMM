"""Pseudotime, windowed trajectory, velocity, and trend computations."""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import numpy as np
import palantir
import pandas as pd
from scipy.stats import spearmanr

from ._trend_clustering import trend_cluster


@dataclass(frozen=True)
class _PalantirConfig:
    n_diff_components: int
    knn: int
    num_waypoints: int
    scale_components: bool
    max_iterations: int
    seed: int
    use_early_cell_as_start: bool

    def to_params(self) -> dict:
        return {
            "n_diff_components": int(self.n_diff_components),
            "knn": int(self.knn),
            "num_waypoints": int(self.num_waypoints),
            "scale_components": bool(self.scale_components),
            "max_iterations": int(self.max_iterations),
            "seed": int(self.seed),
            "use_early_cell_as_start": bool(self.use_early_cell_as_start),
        }


@dataclass
class _TrajectoryBuffers:
    points: np.ndarray
    time: np.ndarray
    counts: np.ndarray
    weight_sum: np.ndarray

    @classmethod
    def create(cls, n_branches: int, n_windows: int, n_dimensions: int):
        shape = (n_branches, n_windows)
        return cls(
            points=np.full((*shape, n_dimensions), np.nan, dtype=float),
            time=np.full(shape, np.nan, dtype=float),
            counts=np.zeros(shape, dtype=int),
            weight_sum=np.full(shape, np.nan, dtype=float),
        )


def _dense_matrix(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("adata.X must be a 2D matrix")
    return matrix


def _window_starts(n_obs: int, window_size: int, step_size: int) -> list[int]:
    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if step_size < 1:
        raise ValueError("step_size must be at least 1")
    effective_window = min(window_size, n_obs)
    starts = list(range(0, max(1, n_obs - effective_window + 1), step_size)) or [0]
    final_start = max(0, n_obs - effective_window)
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def run_palantir(
    adata: ad.AnnData,
    start_idx: int,
    terminal_states: list[int | str] | None = None,
    n_diff_components: int = 10,
    knn: int = 30,
    num_waypoints: int = 500,
    scale_components: bool = True,
    max_iterations: int = 25,
    seed: int = 42,
    use_early_cell_as_start: bool = True,
    store_prefix: str = "palantir",
):
    """Run Palantir and store aligned results and provenance in ``adata``."""
    _validate_palantir_input(adata, start_idx)
    data = pd.DataFrame(_dense_matrix(adata), index=adata.obs_names, columns=adata.var_names)
    internal_index = data.index
    early_cell = internal_index[start_idx]
    terminal_states = _resolve_terminal_states(terminal_states, internal_index)
    config = _PalantirConfig(
        n_diff_components=n_diff_components,
        knn=knn,
        num_waypoints=num_waypoints,
        scale_components=scale_components,
        max_iterations=max_iterations,
        seed=seed,
        use_early_cell_as_start=use_early_cell_as_start,
    )
    multiscale, result = _run_palantir_analysis(
        data,
        early_cell,
        terminal_states,
        config,
    )
    _store_palantir_outputs(adata, multiscale, result, internal_index, store_prefix)
    adata.uns[f"{store_prefix}_params"] = _palantir_params(start_idx, early_cell, config)
    return adata


def _validate_palantir_input(adata: ad.AnnData, start_idx: int) -> None:
    if adata.n_obs < 2 or adata.n_vars < 1:
        raise ValueError("Palantir requires at least two observations and one feature")
    if not 0 <= start_idx < adata.n_obs:
        raise IndexError(f"start_idx out of range: {start_idx}")


def _resolve_terminal_states(
    terminal_states: list[int | str] | None,
    internal_index: pd.Index,
) -> list | None:
    if terminal_states is None:
        return None
    resolved = []
    for state in terminal_states:
        if isinstance(state, int):
            if not 0 <= state < len(internal_index):
                raise IndexError(f"terminal state index out of range: {state}")
            resolved.append(internal_index[state])
            continue
        matches = np.where(internal_index == state)[0]
        if len(matches) == 0:
            raise KeyError(f"terminal state name not found: {state}")
        resolved.append(internal_index[matches[0]])
    return resolved


def _run_palantir_analysis(
    data: pd.DataFrame,
    early_cell,
    terminal_states,
    config: _PalantirConfig,
):
    diffusion = palantir.utils.run_diffusion_maps(
        data,
        n_components=config.n_diff_components,
        knn=config.knn,
        seed=config.seed,
    )
    n_eigenvectors = min(config.n_diff_components + 1, diffusion["EigenVectors"].shape[1])
    if n_eigenvectors < 2:
        raise ValueError("Too few diffusion eigenvectors were produced.")
    multiscale = palantir.utils.determine_multiscale_space(diffusion, n_eigs=n_eigenvectors)
    result = palantir.core.run_palantir(
        multiscale,
        early_cell=early_cell,
        terminal_states=terminal_states,
        knn=config.knn,
        num_waypoints=config.num_waypoints,
        scale_components=config.scale_components,
        max_iterations=config.max_iterations,
        use_early_cell_as_start=config.use_early_cell_as_start,
        seed=config.seed,
    )
    return multiscale, result


def _store_palantir_outputs(
    adata: ad.AnnData,
    multiscale: pd.DataFrame,
    result,
    internal_index: pd.Index,
    prefix: str,
) -> None:
    adata.obs[f"{prefix}_pseudotime"] = result.pseudotime.reindex(internal_index).to_numpy()
    entropy = getattr(result, "entropy", None)
    if entropy is not None:
        adata.obs[f"{prefix}_entropy"] = entropy.reindex(internal_index).to_numpy()

    branch_probs = result.branch_probs.reindex(internal_index)
    adata.obsm[f"{prefix}_branch_probs"] = branch_probs.to_numpy()
    adata.uns[f"{prefix}_branch_names"] = list(branch_probs.columns)
    adata.obsm[f"{prefix}_ms_data"] = multiscale.reindex(internal_index).to_numpy()
    adata.uns[f"{prefix}_ms_columns"] = list(multiscale.columns)
    adata.uns[f"{prefix}_cell_index_map"] = pd.DataFrame(
        {
            "internal_id": internal_index.astype(str),
            "obs_name": adata.obs_names.astype(str),
            "obs_pos": np.arange(adata.n_obs),
        }
    )
    _store_optional_palantir_sequence(adata, result, prefix, "terminal_states")
    _store_optional_palantir_sequence(adata, result, prefix, "waypoints")


def _store_optional_palantir_sequence(adata, result, prefix: str, attribute: str) -> None:
    values = getattr(result, attribute, None)
    if values is not None:
        adata.uns[f"{prefix}_{attribute}_internal"] = list(values)


def _palantir_params(
    start_idx: int,
    early_cell,
    config: _PalantirConfig,
) -> dict:
    return {
        "source": "X",
        "start_idx": int(start_idx),
        "early_cell_internal": str(early_cell),
        **config.to_params(),
    }


def resample_trajectory(
    adata: ad.AnnData,
    window_size: int = 100,
    step_size: int = 50,
    cell_dist_key: str = "X_umap",
    parameterization_key: str = "palantir_pseudotime",
    branch_prob_key: str = "palantir_branch_probs",
    store_key: str = "trajectory",
    min_cells_per_window: int = 5,
    **kwargs,
):
    """Pool low-dimensional cell positions over ordered sliding windows."""
    points, time, branch_prob = _prepare_trajectory_inputs(
        adata,
        cell_dist_key,
        parameterization_key,
        branch_prob_key,
        min_cells_per_window,
    )
    order = np.argsort(time)
    starts = _window_starts(adata.n_obs, window_size, step_size)
    effective_window = min(window_size, adata.n_obs)
    buffers = _pool_trajectory_windows(
        points[order],
        time[order],
        branch_prob[order],
        starts,
        effective_window,
        min_cells_per_window,
    )
    adata.uns[store_key] = buffers.points
    adata.uns[f"{store_key}_metadata"] = {
        "time": buffers.time,
        "counts": buffers.counts,
        "weight_sum": buffers.weight_sum,
        "window_starts": np.asarray(starts, dtype=int),
        "window_size": int(effective_window),
        "step_size": int(step_size),
    }
    return adata


def _prepare_trajectory_inputs(
    adata: ad.AnnData,
    cell_dist_key: str,
    parameterization_key: str,
    branch_prob_key: str,
    min_cells_per_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if adata.n_obs < 1:
        raise ValueError("Trajectory resampling requires at least one observation")
    if cell_dist_key not in adata.obsm:
        raise KeyError(f"{cell_dist_key} not found in adata.obsm")
    if parameterization_key not in adata.obs:
        raise KeyError(f"{parameterization_key} not found in adata.obs")
    if min_cells_per_window < 1:
        raise ValueError("min_cells_per_window must be at least 1")

    points = np.asarray(adata.obsm[cell_dist_key], dtype=float)
    time = np.asarray(adata.obs[parameterization_key].to_numpy(), dtype=float)
    if points.ndim != 2 or points.shape[0] != adata.n_obs:
        raise ValueError(f"{cell_dist_key} must be a 2D array with one row per observation")
    if not np.isfinite(time).all():
        raise ValueError(f"{parameterization_key} must contain only finite values")
    return points, time, _prepare_branch_probabilities(adata, branch_prob_key)


def _prepare_branch_probabilities(adata: ad.AnnData, key: str) -> np.ndarray:
    probabilities = adata.obsm.get(key)
    if probabilities is None:
        return np.ones((adata.n_obs, 1), dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities[:, None]
    if probabilities.ndim != 2 or probabilities.shape[0] != adata.n_obs:
        raise ValueError(f"{key} must have one row per observation")
    if np.any(probabilities[np.isfinite(probabilities)] < 0):
        raise ValueError(f"{key} cannot contain negative probabilities")
    return probabilities


def _pool_trajectory_windows(
    points: np.ndarray,
    time: np.ndarray,
    probabilities: np.ndarray,
    starts: list[int],
    window_size: int,
    min_cells: int,
) -> _TrajectoryBuffers:
    buffers = _TrajectoryBuffers.create(
        probabilities.shape[1],
        len(starts),
        points.shape[1],
    )

    for window_index, start in enumerate(starts):
        window = slice(start, min(start + window_size, len(time)))
        if points[window].shape[0] < min_cells:
            continue
        for branch in range(probabilities.shape[1]):
            _pool_trajectory_branch(
                points[window],
                time[window],
                probabilities[window, branch],
                branch,
                window_index,
                min_cells,
                buffers,
            )
    return buffers


def _pool_trajectory_branch(
    points,
    time,
    weights,
    branch: int,
    window_index: int,
    min_cells: int,
    buffers: _TrajectoryBuffers,
) -> None:
    valid = np.isfinite(weights) & np.isfinite(time) & np.isfinite(points).all(axis=1)
    if valid.sum() < min_cells:
        return
    weights = weights[valid]
    total_weight = weights.sum()
    if total_weight <= 0:
        return
    buffers.points[branch, window_index] = np.average(points[valid], axis=0, weights=weights)
    buffers.time[branch, window_index] = np.average(time[valid], weights=weights)
    buffers.counts[branch, window_index] = valid.sum()
    buffers.weight_sum[branch, window_index] = total_weight


def metabolic_velocity_field(
    adata: ad.AnnData,
    window_size: int = 100,
    step_size: int = 50,
    parameterization_key: str = "time",
):
    """Estimate local feature slopes and their vector speed over time."""
    matrix, time = _prepare_velocity_inputs(adata, parameterization_key, window_size, step_size)
    order = np.argsort(time)
    starts = _window_starts(adata.n_obs, window_size, step_size)
    result = _calculate_velocity_windows(
        matrix[order],
        time[order],
        starts,
        min(window_size, adata.n_obs),
    )
    result["window_starts"] = np.asarray(starts, dtype=int)
    adata.uns["metabolic_velocity"] = result
    return adata


def _prepare_velocity_inputs(
    adata: ad.AnnData,
    parameterization_key: str,
    window_size: int,
    step_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if parameterization_key not in adata.obs:
        raise KeyError(f"{parameterization_key} not found in adata.obs")
    matrix = _dense_matrix(adata)
    time = np.asarray(adata.obs[parameterization_key].to_numpy(), dtype=float)
    if adata.n_obs < 2:
        raise ValueError("Need at least 2 observations to compute velocity")
    if not np.isfinite(time).all() or not np.isfinite(matrix).all():
        raise ValueError("Data and parameterization values must be finite")
    if window_size < 2:
        raise ValueError("window_size must be >= 2")
    if step_size < 1:
        raise ValueError("step_size must be >= 1")
    return matrix, time


def _calculate_velocity_windows(
    matrix: np.ndarray,
    time: np.ndarray,
    starts: list[int],
    window_size: int,
) -> dict[str, np.ndarray]:
    n_windows, n_features = len(starts), matrix.shape[1]
    state_centers = np.full((n_windows, n_features), np.nan, dtype=float)
    velocity = np.full((n_windows, n_features), np.nan, dtype=float)
    time_centers = np.full(n_windows, np.nan, dtype=float)
    speeds = np.full(n_windows, np.nan, dtype=float)
    counts = np.zeros(n_windows, dtype=int)
    for index, start in enumerate(starts):
        window = slice(start, min(start + window_size, len(time)))
        values, window_time = matrix[window], time[window]
        if len(window_time) < 2:
            continue
        center_time = window_time.mean()
        time_delta = window_time - center_time
        denominator = np.sum(time_delta**2)
        if denominator <= 0:
            continue
        center = values.mean(axis=0)
        slope = (time_delta[:, None] * (values - center[None, :])).sum(axis=0) / denominator
        state_centers[index], velocity[index] = center, slope
        time_centers[index], speeds[index] = center_time, np.linalg.norm(slope)
        counts[index] = len(window_time)
    return {
        "state_centers": state_centers,
        "velocity_field": velocity,
        "speeds": speeds,
        "time_centers": time_centers,
        "counts": counts,
    }


def metabolite_trends(
    adata: ad.AnnData,
    parameterization_key: str = "time",
    window_size: int = 100,
    step_size: int = 50,
    kernel_stat: str = "median",
    feature_name_key: str | None = None,
):
    """Pool feature intensities over time and rank monotonic trends."""
    matrix, time, kernel_stat = _prepare_trend_inputs(
        adata,
        parameterization_key,
        window_size,
        step_size,
        kernel_stat,
    )
    order = np.argsort(time)
    starts = _window_starts(adata.n_obs, window_size, step_size)
    effective_window = min(window_size, adata.n_obs)
    pooled, time_centers, counts = _pool_feature_windows(
        matrix[order],
        time[order],
        starts,
        effective_window,
        kernel_stat,
    )
    rho, p_values = _trend_significance(pooled, time_centers)
    q_values = _bh_fdr(p_values)
    adata.uns["metabolite_trends"] = {
        "pooled": pooled,
        "time_centers": time_centers,
        "counts": counts,
        "rho": rho,
        "pval": p_values,
        "qval": q_values,
        "feature_names": _feature_names(adata, feature_name_key),
        "window_size": effective_window,
        "step_size": step_size,
        "kernel_stat": kernel_stat,
        "parameterization_key": parameterization_key,
        "sorted_obs_order": order,
        "rank_idx": _rank_trends(q_values, rho),
    }
    return adata


def _prepare_trend_inputs(
    adata: ad.AnnData,
    parameterization_key: str,
    window_size: int,
    step_size: int,
    kernel_stat: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    if parameterization_key not in adata.obs:
        raise KeyError(f"{parameterization_key} not found in adata.obs")
    matrix = _dense_matrix(adata)
    time = np.asarray(adata.obs[parameterization_key].to_numpy(), dtype=float)
    if adata.n_obs < 2:
        raise ValueError("Need at least 2 observations")
    if not np.isfinite(time).all():
        raise ValueError(f"{parameterization_key} must contain only finite values")
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if step_size < 1:
        raise ValueError("step_size must be >= 1")
    kernel_stat = kernel_stat.lower()
    if kernel_stat not in {"median", "mean", "sum"}:
        raise ValueError("kernel_stat must be one of {'median', 'mean', 'sum'}")
    return matrix, time, kernel_stat


def _pool_feature_windows(
    matrix: np.ndarray,
    time: np.ndarray,
    starts: list[int],
    window_size: int,
    kernel_stat: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pooled = np.full((len(starts), matrix.shape[1]), np.nan, dtype=float)
    time_centers = np.full(len(starts), np.nan, dtype=float)
    counts = np.zeros(len(starts), dtype=int)
    reducers = {"median": np.nanmedian, "mean": np.nanmean, "sum": np.nansum}
    reducer = reducers[kernel_stat]
    for index, start in enumerate(starts):
        window = slice(start, min(start + window_size, len(time)))
        values, window_time = matrix[window], time[window]
        if values.shape[0] == 0:
            continue
        pooled[index] = reducer(values, axis=0)
        time_centers[index] = np.nanmean(window_time)
        counts[index] = values.shape[0]
    return pooled, time_centers, counts


def _trend_significance(
    pooled: np.ndarray, time_centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    rho = np.full(pooled.shape[1], np.nan, dtype=float)
    p_values = np.full(pooled.shape[1], np.nan, dtype=float)
    valid_time = np.isfinite(time_centers)
    for feature in range(pooled.shape[1]):
        values = pooled[:, feature]
        valid = valid_time & np.isfinite(values)
        if valid.sum() < 3 or np.nanstd(values[valid]) == 0:
            continue
        rho[feature], p_values[feature] = spearmanr(time_centers[valid], values[valid])
    return rho, p_values


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    valid = np.isfinite(p_values)
    if valid.sum() == 0:
        return q_values
    valid_values = p_values[valid]
    order = np.argsort(valid_values)
    ranked = valid_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(valid_values)
    restored[order] = adjusted
    q_values[valid] = restored
    return q_values


def _feature_names(adata: ad.AnnData, feature_name_key: str | None) -> np.ndarray:
    if feature_name_key is not None and feature_name_key in adata.var.columns:
        return np.asarray(adata.var[feature_name_key].astype(str))
    return np.asarray([f"feature_{index}" for index in range(adata.n_vars)])


def _rank_trends(q_values: np.ndarray, rho: np.ndarray) -> np.ndarray:
    score = np.full(len(q_values), np.inf, dtype=float)
    valid = np.isfinite(q_values) & np.isfinite(rho)
    score[valid] = q_values[valid] - 1e-6 * np.abs(rho[valid])
    return np.argsort(score)


__all__ = [
    "metabolic_velocity_field",
    "metabolite_trends",
    "resample_trajectory",
    "run_palantir",
    "trend_cluster",
]
