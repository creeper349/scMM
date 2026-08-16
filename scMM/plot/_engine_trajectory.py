"""Trajectory-analysis and trend-plotting capabilities for :class:`PlotEngine`."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt

from ._trajectory import (
    metabolic_velocity_field,
    metabolite_trends,
    resample_trajectory,
    trend_cluster,
)
from ._trajectory import (
    run_palantir as run_palantir_analysis,
)


class TrajectoryMixin:
    """Provide pseudotime, trajectory, velocity, and trend analysis methods."""

    def run_palantir(
        self,
        start_idx: int,
        plotting: bool = False,
        cmap: str = "viridis",
        use_obsm: str = "X_umap",
        s=1,
        **kwargs,
    ):
        """Run Palantir and optionally save a pseudotime scatter plot."""
        self.adata = run_palantir_analysis(adata=self.adata, start_idx=start_idx, **kwargs)
        if plotting:
            coordinates = _require_obsm(self.adata, use_obsm)
            _save_palantir_plot(
                coordinates,
                self.adata.obs["palantir_pseudotime"],
                self.path / "palantir_pseudotime.svg",
                cmap,
                s,
            )
        return self

    def compute_trajectory(
        self,
        window_size: int = 100,
        step_size: int = 50,
        cell_dist_key: str = "X_umap",
        parameterization_key: str = "palantir_pseudotime",
        branch_prob_key: str | None = "palantir_branch_probs",
        store_key: str = "trajectory",
        min_cells_per_window: int = 5,
        plotting: bool = False,
        cmap: str = "viridis",
        s: int = 1,
        traj_linewidth: float = 1.0,
        traj_points: int = 10,
        title=None,
        **kwargs,
    ):
        """Resample branch trajectories and optionally save their UMAP overlay."""
        self.adata = resample_trajectory(
            adata=self.adata,
            window_size=window_size,
            step_size=step_size,
            cell_dist_key=cell_dist_key,
            parameterization_key=parameterization_key,
            branch_prob_key=branch_prob_key,
            store_key=store_key,
            min_cells_per_window=min_cells_per_window,
            **kwargs,
        )
        if plotting:
            if traj_points < 1:
                raise ValueError("traj_points must be at least 1")
            _save_trajectory_plot(
                _require_obsm(self.adata, cell_dist_key),
                self.adata.obs[parameterization_key],
                self.adata.uns[store_key],
                self.path / "trajectory.svg",
                cmap,
                s,
                traj_linewidth,
                traj_points,
                title,
            )
        return self

    def metabolic_velocity(
        self,
        window_size: int = 100,
        step_size: int = 50,
        parameterization_key: str = "time",
        plot=True,
        linewidth=1,
        **kwargs,
    ):
        """Compute metabolic velocity and optionally save its speed profile."""
        self.adata = metabolic_velocity_field(
            adata=self.adata,
            window_size=window_size,
            step_size=step_size,
            parameterization_key=parameterization_key,
        )
        if plot:
            result = self.adata.uns["metabolic_velocity"]
            _save_velocity_plot(
                result["time_centers"],
                result["speeds"],
                self.path / "metabolic_velocity_speed.svg",
                parameterization_key,
                linewidth,
            )
        return self

    def plot_metabolite_trends(
        self,
        parameterization_key: str = "time",
        window_size: int = 100,
        step_size: int = 50,
        kernel_stat: str = "median",
        feature_name_key: str | None = None,
        plot_top_n: int | None = None,
        cmap: str = "viridis",
        **kwargs,
    ):
        """Compute metabolite trends and optionally save a top-feature heatmap."""
        self.adata = metabolite_trends(
            adata=self.adata,
            window_size=window_size,
            step_size=step_size,
            parameterization_key=parameterization_key,
            kernel_stat=kernel_stat,
            feature_name_key=feature_name_key,
        )
        if plot_top_n is not None:
            top_indices = _select_top_trend_indices(self.adata, feature_name_key, plot_top_n)
            _save_trend_heatmap(
                self.adata.uns["metabolite_trends"],
                top_indices,
                self.path / f"metabolite_trends_top{len(top_indices)}.svg",
                cmap,
                kwargs.get("xlabel", parameterization_key),
                kwargs.get("ylabel", "Metabolite"),
            )
        return self

    def plot_trend_clusters(
        self,
        metric: str = "correlation",
        cluster_method: str = "leiden",
        linewidth: float = 1.0,
        top_k: int | None = None,
        **kwargs,
    ):
        """Cluster normalized feature trends and save one panel per cluster."""
        result = self.adata.uns["metabolite_trends"]
        trends = _select_and_standardize_trends(result, top_k)
        labels = trend_cluster(trends, metric=metric, cluster_method=cluster_method, **kwargs)
        _save_trend_cluster_plot(
            trends,
            labels,
            result["time_centers"],
            self.path / f"trend_clusters_{cluster_method}_{metric}.svg",
            linewidth,
            kwargs.get("xlabel", "Time"),
            kwargs.get("ylabel", "Relative intensity"),
        )
        return self


def _require_obsm(adata, key: str) -> np.ndarray:
    coordinates = adata.obsm.get(key)
    if coordinates is None:
        raise KeyError(f"{key} not found in obsm")
    return np.asarray(coordinates)


def _save_palantir_plot(coordinates, pseudotime, output_path, cmap: str, size) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=pseudotime,
        cmap=cmap,
        s=size,
    )
    fig.colorbar(scatter, ax=ax, label="Pseudotime")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_trajectory_plot(
    coordinates,
    pseudotime,
    trajectories,
    output_path,
    cmap: str,
    size: float,
    linewidth: float,
    n_points: int,
    title,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=pseudotime,
        cmap=cmap,
        s=size,
    )
    point_step = max(1, trajectories.shape[1] // n_points)
    point_indices = np.arange(0, trajectories.shape[1], point_step)
    for branch in trajectories:
        ax.plot(branch[:, 0], branch[:, 1], color="black", linewidth=linewidth)
        ax.scatter(
            branch[point_indices, 0], branch[point_indices, 1], color="black", s=2 * n_points
        )
    fig.colorbar(scatter, ax=ax, label="Pseudotime")
    ax.set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
    if title is not None:
        ax.set_title(title, size=16)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _save_velocity_plot(time_centers, speeds, output_path, xlabel: str, linewidth: float) -> None:
    fig, ax = plt.subplots()
    ax.plot(time_centers, speeds, color="black", linewidth=linewidth)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Metabolic velocity")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _select_top_trend_indices(adata, feature_name_key: str | None, plot_top_n: int) -> list[int]:
    ranked = list(adata.uns["metabolite_trends"]["rank_idx"])
    if feature_name_key is not None:
        feature_values = adata.var[feature_name_key]
        valid = feature_values.notna() & (feature_values.astype(str).str.strip() != "")
        valid_indices = set(np.where(valid.to_numpy())[0])
        ranked = [index for index in ranked if index in valid_indices]
    selected = ranked[: min(plot_top_n, len(ranked))]
    if not selected:
        raise ValueError("No features are available for trend plotting")
    return selected


def _standardize_rows(values: np.ndarray) -> np.ndarray:
    row_mean = np.nanmean(values, axis=1, keepdims=True)
    row_std = np.nanstd(values, axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0
    return (values - row_mean) / row_std


def _save_trend_heatmap(
    result: dict,
    top_indices: list[int],
    output_path,
    cmap: str,
    xlabel: str,
    ylabel: str,
) -> None:
    values = _standardize_rows(result["pooled"][:, top_indices].T.copy())
    fig, ax = plt.subplots(
        figsize=(max(6, 0.35 * values.shape[1]), max(4, 0.25 * len(top_indices)))
    )
    image = ax.imshow(values, aspect="auto", interpolation="nearest", cmap=cmap)
    ax.set_yticks(np.arange(len(top_indices)), [result["feature_names"][i] for i in top_indices])
    ax.set_xticks(
        np.arange(len(result["time_centers"])),
        [f"{value:.2f}" if np.isfinite(value) else "" for value in result["time_centers"]],
        rotation=90,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax, label="Row-wise z-scored pooled intensity")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _select_and_standardize_trends(result: dict, top_k: int | None) -> np.ndarray:
    trends = result["pooled"]
    indices = result["rank_idx"][:top_k] if top_k is not None else np.arange(trends.shape[1])
    selected = trends[:, indices]
    mean = np.nanmean(selected, axis=0, keepdims=True)
    std = np.nanstd(selected, axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (selected - mean) / std


def _save_trend_cluster_plot(
    trends: np.ndarray,
    labels: np.ndarray,
    time_centers: np.ndarray,
    output_path,
    linewidth: float,
    xlabel: str,
    ylabel: str,
) -> None:
    unique_labels = np.unique(labels)
    if unique_labels.size == 0:
        raise ValueError("No trend clusters were produced")
    fig, axes = plt.subplots(nrows=unique_labels.size, figsize=(6, 6 * unique_labels.size))
    axes = [axes] if unique_labels.size == 1 else axes
    for axis, label in zip(axes, unique_labels, strict=True):
        cluster_trends = trends.T[labels == label]
        axis.plot(
            time_centers, np.nanmean(cluster_trends, axis=0), color="black", linewidth=linewidth
        )
        for trend in cluster_trends:
            axis.plot(time_centers, trend, color="gray", alpha=0.5, linewidth=0.5)
        axis.set_title(f"Cluster {label} (n={cluster_trends.shape[0]})")
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
