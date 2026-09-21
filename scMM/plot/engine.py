from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd
import anndata as ad
import palantir
import umap
import seaborn as sns
import os
import warnings
import networkx as nx
import re

from sklearn.covariance import GraphicalLasso
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph
from ._trajectory import (run_palantir, resample_trajectory, metabolic_velocity_field, 
                          metabolite_trends, trend_cluster)

class PlotEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        fig_path_dir: str,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
    ):
        self.adata = ad.AnnData(
            X=df.values,
            obs=obs.copy() if obs is not None else pd.DataFrame(index=df.index),
            var=var.copy() if var is not None else pd.DataFrame(index=df.columns),
        )
        self.adata.obs_names_make_unique()
        self.adata.var_names_make_unique()
        self.path = fig_path_dir

    @classmethod
    def from_adata(cls, adata: ad.AnnData, fig_path_dir: str):
        obj = object.__new__(cls)
        obj.adata = adata.copy()
        obj.adata.obs_names_make_unique()
        obj.adata.var_names_make_unique()
        obj.path = fig_path_dir
        return obj

    def _get_X(self) -> np.ndarray:
        X = self.adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        return np.asarray(X, dtype=float)

    def _get_internal_cell_index(self) -> pd.Index:
        return pd.Index([f"cell_{i}" for i in range(self.adata.n_obs)], name="cell_id")

    def _get_X_df(self) -> pd.DataFrame:
        X = self._get_X()
        index = self._get_internal_cell_index()

        if X.shape[1] == self.adata.n_vars:
            columns = self.adata.var_names.copy()
        else:
            columns = [f"feature_{i}" for i in range(X.shape[1])]

        return pd.DataFrame(X, index=index, columns=columns)

    def _sliding_window_slices(
        self,
        parameterization_key: str,
        window_size: int,
        step_size: int,
        min_cells_per_window: int = 5,
    ):
        """Return sorted observation order and reusable sliding-window slices."""
        if parameterization_key not in self.adata.obs.columns:
            raise KeyError(f"{parameterization_key!r} not found in self.adata.obs")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if step_size < 1:
            raise ValueError("step_size must be >= 1")
        if min_cells_per_window < 1:
            raise ValueError("min_cells_per_window must be >= 1")

        t = pd.to_numeric(self.adata.obs[parameterization_key], errors="coerce").to_numpy(dtype=float)
        valid_obs = np.isfinite(t)
        if valid_obs.sum() < min_cells_per_window:
            raise ValueError(
                f"Fewer than {min_cells_per_window} finite observations were found in "
                f"adata.obs[{parameterization_key!r}]."
            )

        valid_idx = np.flatnonzero(valid_obs)
        order_local = np.argsort(t[valid_obs], kind="mergesort")
        order = valid_idx[order_local]
        t_sorted = t[order]
        n_obs = len(order)

        starts = list(range(0, max(1, n_obs - window_size + 1), step_size))
        if not starts:
            starts = [0]
        if starts[-1] + window_size < n_obs:
            starts.append(max(0, n_obs - window_size))

        windows = []
        for start in starts:
            end = min(start + window_size, n_obs)
            if end - start < min_cells_per_window:
                continue
            obs_idx = order[start:end]
            windows.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "obs_idx": obs_idx,
                    "time_center": float(np.nanmean(t[obs_idx])),
                    "time_min": float(np.nanmin(t[obs_idx])),
                    "time_max": float(np.nanmax(t[obs_idx])),
                    "count": int(len(obs_idx)),
                }
            )

        if not windows:
            raise ValueError("No sliding window satisfies min_cells_per_window")
        return t, order, windows

    def _resolve_feature_index(
        self,
        feature: Union[int, str],
        feature_name_key: Optional[str] = None,
    ) -> int:
        """Resolve a feature index from var position, var_name, or a var annotation column."""
        if isinstance(feature, (int, np.integer)):
            idx = int(feature)
            if idx < 0 or idx >= self.adata.n_vars:
                raise IndexError(f"feature index out of range: {idx}")
            return idx

        feature = str(feature)
        if feature in self.adata.var_names:
            return int(self.adata.var_names.get_loc(feature))

        if feature_name_key is not None:
            if feature_name_key not in self.adata.var.columns:
                raise KeyError(f"{feature_name_key!r} not found in self.adata.var")
            values = self.adata.var[feature_name_key].astype(str).to_numpy()
            matches = np.flatnonzero(values == feature)
            if len(matches) == 0:
                raise KeyError(
                    f"Feature {feature!r} not found in adata.var_names or "
                    f"adata.var[{feature_name_key!r}]"
                )
            if len(matches) > 1:
                warnings.warn(
                    f"Feature name {feature!r} matched {len(matches)} features in "
                    f"adata.var[{feature_name_key!r}]; using the first match.",
                    RuntimeWarning,
                )
            return int(matches[0])

        raise KeyError(
            f"Feature {feature!r} not found in adata.var_names. "
            "Provide feature_name_key to match against a column in adata.var."
        )

    def pca(
        self,
        n_components: int = 50,
        scale: bool = True,
        zero_center: bool = True,
        random_state: int = 42,
        store_key: str = "X_pca",
        return_model: bool = False,
    ):
        X = self._get_X()

        X_proc = X.copy()

        if scale or zero_center:
            scaler = StandardScaler(with_mean=zero_center, with_std=scale)
            X_proc = scaler.fit_transform(X_proc)

        n_components = min(n_components, X_proc.shape[0], X_proc.shape[1])
        if n_components < 1:
            raise ValueError("n_components must be >= 1")

        model = PCA(n_components=n_components, random_state=random_state)
        X_pca = model.fit_transform(X_proc)

        self.adata.obsm[store_key] = X_pca
        self.adata.uns[f"{store_key}_params"] = {
            "source": "X",
            "n_components": int(n_components),
            "scale": bool(scale),
            "zero_center": bool(zero_center),
            "random_state": int(random_state),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        }

        if return_model:
            return X_pca, model
        return X_pca

    def umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 30,
        min_dist: float = 0.3,
        metric: str = "euclidean",
        random_state: int = 42,
        store_key: str = "X_umap",
        use_pca: bool = False,
        pca_key: str = "X_pca",
        pca_n_components: int = 30,
        scale_before_pca: bool = True,
    ):
        if use_pca:
            if pca_key in self.adata.obsm:
                X_umap_input = np.asarray(self.adata.obsm[pca_key], dtype=float)
                actual_source = f"obsm:{pca_key}"
            else:
                X = self._get_X().copy()
                if scale_before_pca:
                    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

                n_pca = min(pca_n_components, X.shape[0], X.shape[1])
                if n_pca < 1:
                    raise ValueError("pca_n_components must be >= 1")

                X_umap_input = PCA(
                    n_components=n_pca,
                    random_state=random_state
                ).fit_transform(X)
                actual_source = f"X->PCA({n_pca})"
        else:
            X_umap_input = self._get_X()
            actual_source = "X"

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        X_umap = reducer.fit_transform(X_umap_input)

        self.adata.obsm[store_key] = X_umap
        self.adata.uns[f"{store_key}_params"] = {
            "source": actual_source,
            "n_components": int(n_components),
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": metric,
            "random_state": int(random_state),
        }

        return X_umap

    def run_palantir(self, start_idx: int, plotting: bool = False, cmap: str = "viridis", 
                     use_obsm: str = "X_umap", s = 1, **kwargs):
        self.adata = run_palantir(adata=self.adata, start_idx=start_idx, **kwargs)
        if plotting:
            obsm = self.adata.obsm.get(use_obsm, None)
            assert obsm is not None, f"{use_obsm} not found in obsm"
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(obsm[:, 0], obsm[:, 1], c=self.adata.obs["palantir_pseudotime"], cmap=cmap, s=s)
            plt.colorbar(sc, label="Pseudotime")
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.savefig(f"{self.path}/palantir_pseudotime.svg", bbox_inches="tight")
            plt.close()
        return self
    
    def compute_trajectory(self, window_size: int = 100, step_size: int = 50,
        cell_dist_key: str = "X_umap", parameterization_key: str = "palantir_pseudotime",
        branch_prob_key: str = "palantir_branch_probs", store_key: str = "trajectory",
        min_cells_per_window: int = 5, plotting: bool = False, cmap: str = "viridis", s: int = 1, 
        traj_linewidth: float = 1.0, traj_points: int = 10, title = None, **kwargs):
        self.adata = resample_trajectory(
            adata=self.adata,
            window_size=window_size,
            step_size=step_size,
            cell_dist_key=cell_dist_key,
            parameterization_key=parameterization_key,
            branch_prob_key=branch_prob_key,
            store_key=store_key,
            min_cells_per_window=min_cells_per_window,
            **kwargs
        )
        if plotting:
            obsm = self.adata.obsm.get(cell_dist_key, None)
            assert obsm is not None, f"{cell_dist_key} not found in obsm"
            traj = self.adata.uns[store_key]
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(obsm[:, 0], obsm[:, 1], c=self.adata.obs[parameterization_key], cmap=cmap, s=s)
            for b in range(traj.shape[0]):
                traj_points_mask = np.arange(0, traj.shape[1], traj.shape[1] // traj_points)
                plt.plot(traj[b, :, 0], traj[b, :, 1], color="black", linewidth=traj_linewidth)
                plt.scatter(traj[b, traj_points_mask, 0], traj[b, traj_points_mask, 1], color = "black", s=2 * traj_points)
            plt.colorbar(sc, label="Pseudotime")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.xticks([])
            plt.yticks([])
            if title is not None: plt.title(title, size=16)
            plt.savefig(f"{self.path}/trajectory.svg", bbox_inches="tight")
            plt.close()
            
    def metabolic_velocity(self, window_size: int = 100, step_size: int = 50,
                           parameterization_key: str = "time", plot = True,
                           linewidth = 1, **kwargs):
        self.adata = metabolic_velocity_field(
            adata=self.adata,
            window_size=window_size,
            step_size=step_size,
            parameterization_key=parameterization_key)
        time_centers = self.adata.uns["metabolic_velocity"]["time_centers"]
        speeds = self.adata.uns["metabolic_velocity"]["speeds"]
        if plot:
            plt.plot(time_centers, speeds, color="black", linewidth=linewidth)
            plt.xlabel(parameterization_key)
            plt.ylabel("Metabolic velocity")
            plt.savefig(f"{self.path}/metabolic_velocity_speed.svg", bbox_inches="tight")
            plt.close()
            
    def plot_metabolite_trends(
        self,
        parameterization_key: str = "time",
        window_size: int = 100,
        step_size: int = 50,
        kernel_stat: str = "median",
        feature_name_key: Optional[str] = None,
        plot_top_n: Optional[int] = None,
        feature_indices: Optional[Union[list, np.ndarray, pd.Index]] = None,
        cmap: str = "viridis",
        recompute: bool = True,
        output_file: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs
    ):
        
        if recompute or "metabolite_trends" not in self.adata.uns:
            self.adata = metabolite_trends(
                adata=self.adata,
                window_size=window_size,
                step_size=step_size,
                parameterization_key=parameterization_key,
                kernel_stat=kernel_stat,
                feature_name_key=feature_name_key
            )

        trend_info = self.adata.uns["metabolite_trends"]
        rank_idx = list(trend_info["rank_idx"])

        if feature_indices is None:
            if plot_top_n is None:
                return self

            selected_idx = rank_idx[:min(plot_top_n, len(rank_idx))]
        else:
            selected_idx = [int(i) for i in np.asarray(feature_indices).ravel()]

            # Keep selected features in the same ranking order when possible.
            rank_order = {int(idx): pos for pos, idx in enumerate(rank_idx)}
            selected_idx = sorted(
                selected_idx,
                key=lambda idx: rank_order.get(int(idx), len(rank_order))
            )

            if plot_top_n is not None:
                selected_idx = selected_idx[:min(plot_top_n, len(selected_idx))]

        # Remove features with empty feature_name_key / empty plotted name.
        if feature_name_key is not None and feature_name_key in self.adata.var.columns:
            feature_values = self.adata.var[feature_name_key]
            valid_mask = (
                feature_values.notna()
                & (feature_values.astype(str).str.strip() != "")
            )
            selected_idx = [idx for idx in selected_idx if bool(valid_mask.iloc[idx])]

        if len(selected_idx) == 0:
            return self

        M = trend_info["pooled"][:, selected_idx].T

        # Row-wise z-score for visualization.
        M_plot = M.copy()
        row_mean = np.nanmean(M_plot, axis=1, keepdims=True)
        row_std = np.nanstd(M_plot, axis=1, keepdims=True)
        row_std[row_std == 0] = 1.0
        M_plot = (M_plot - row_mean) / row_std

        fig_h = max(4, 0.1 * len(selected_idx))
        fig_w = max(4, 0.1 * M_plot.shape[1])

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(
            M_plot,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
        )

        feature_names = trend_info["feature_names"]
        ylabels = [feature_names[j] for j in selected_idx]

        ax.set_yticks(np.arange(len(selected_idx)))
        ax.set_yticklabels(ylabels)

        time_centers = np.asarray(trend_info["time_centers"], dtype=float)
        n_time = M_plot.shape[1]

        if len(time_centers) != n_time:
            raise ValueError(
                f"Length mismatch: time_centers has length {len(time_centers)}, "
                f"but heatmap has {n_time} time windows."
            )

        # Integer ticks in the real time scale.
        tick_step = kwargs.get("xtick_step", 2)
        time_min = int(np.ceil(np.nanmin(time_centers)))
        time_max = int(np.floor(np.nanmax(time_centers)))

        tick_labels = np.arange(time_min, time_max + 1, tick_step)

        # Map real time values to heatmap column positions.
        tick_pos = np.interp(
            tick_labels,
            time_centers,
            np.arange(n_time)
        )

        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(int(x)) for x in tick_labels])

        ax.set_xlabel(kwargs.get("xlabel", parameterization_key))
        ax.set_ylabel(kwargs.get("ylabel", "Metabolite"))

        if title is not None:
            ax.set_title(title)

        fig.colorbar(im, ax=ax, label="Row-wise z-scored pooled intensity")
        fig.tight_layout()

        if output_file is None:
            output_file = f"{self.path}/metabolite_trends_top{len(selected_idx)}.svg"

        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)

        return self

    def plot_trend_clusters(
        self,
        metric: str = "correlation",
        cluster_method: str = "leiden",
        linewidth: float = 1.0,
        top_k: int = None,
        plot_cluster_heatmaps: bool = True,
        max_metabolites_per_cluster: Optional[int] = None,
        named_only: bool = True,
        feature_name_key: Optional[str] = None,
        heatmap_cmap: str = "viridis",
        heatmap_prefix: Optional[str] = None,
        **kwargs
    ):
        if "metabolite_trends" not in self.adata.uns:
            raise KeyError(
                "metabolite_trends not found in self.adata.uns. "
                "Please run plot_metabolite_trends() first."
            )

        trends_all = self.adata.uns["metabolite_trends"]["pooled"]
        time_centers = self.adata.uns["metabolite_trends"]["time_centers"]

        if top_k is not None:
            rank_idx = self.adata.uns["metabolite_trends"]["rank_idx"]
            top_k_idx = np.asarray(rank_idx[:min(top_k, len(rank_idx))], dtype=int)
        else:
            top_k_idx = np.arange(trends_all.shape[1], dtype=int)

        trends = trends_all[:, top_k_idx]

        # z-score each feature across time
        mean = np.nanmean(trends, axis=0, keepdims=True)
        std = np.nanstd(trends, axis=0, keepdims=True)
        std[std == 0] = 1.0
        trends = (trends - mean) / std

        cluster_labels = trend_cluster(
            trends,
            metric=metric,
            cluster_method=cluster_method,
            **kwargs
        )

        label_unique = np.unique(cluster_labels)

        self.adata.uns["trend_clusters"] = {
            "cluster_labels": np.asarray(cluster_labels),
            "feature_indices": np.asarray(top_k_idx),
            "metric": metric,
            "cluster_method": cluster_method,
            "top_k": top_k,
        }

        var_cluster_key = f"trend_clusters_{cluster_method}_{metric}"
        cluster_series = pd.Series(index=self.adata.var_names, dtype="object")
        for local_i, feature_i in enumerate(top_k_idx):
            cluster_series.iloc[feature_i] = str(cluster_labels[local_i])
        self.adata.var[var_cluster_key] = pd.Categorical(cluster_series)

        fig, ax = plt.subplots(
            nrows=label_unique.size,
            figsize=(6, 6 * label_unique.size)
        )

        if label_unique.size == 1:
            ax = [ax]

        if heatmap_prefix is None:
            heatmap_prefix = f"trend_cluster_{cluster_method}_{metric}"

        for i, label in enumerate(label_unique):
            cluster_local_idx = np.where(cluster_labels == label)[0]
            cluster_trends = trends.T[cluster_local_idx]

            mean_trend = np.nanmean(cluster_trends, axis=0)
            ax[i].plot(
                time_centers,
                mean_trend,
                color="red",
                linewidth=linewidth
            )

            for j in cluster_local_idx:
                ax[i].plot(
                    time_centers,
                    trends.T[j],
                    color="gray",
                    alpha=0.5,
                    linewidth=0.5
                )

            ax[i].set_title(f"Cluster {label} (n={cluster_trends.shape[0]})")
            ax[i].set_xlabel(kwargs.get("xlabel", "Time"))
            ax[i].set_ylabel(kwargs.get("ylabel", "Relative intensity"))

            if plot_cluster_heatmaps:
                cluster_feature_idx = top_k_idx[cluster_local_idx]

                if named_only:
                    if feature_name_key is None:
                        raise ValueError(
                            "named_only=True requires feature_name_key to be provided."
                        )
                    if feature_name_key not in self.adata.var.columns:
                        raise KeyError(
                            f"{feature_name_key!r} not found in self.adata.var."
                        )

                    feature_values = self.adata.var[feature_name_key]
                    valid_mask = (
                        feature_values.notna()
                        & (feature_values.astype(str).str.strip() != "")
                    )

                    cluster_feature_idx = np.asarray(
                        [
                            idx for idx in cluster_feature_idx
                            if valid_mask.iloc[idx]
                        ],
                        dtype=int
                    )

                if max_metabolites_per_cluster is not None:
                    if max_metabolites_per_cluster < 1:
                        raise ValueError(
                            "max_metabolites_per_cluster must be >= 1 or None."
                        )
                    cluster_feature_idx = cluster_feature_idx[
                        :max_metabolites_per_cluster
                    ]

                if len(cluster_feature_idx) == 0:
                    continue

                self.plot_metabolite_trends(
                    feature_name_key=feature_name_key,
                    feature_indices=cluster_feature_idx,
                    recompute=False,
                    cmap=heatmap_cmap,
                    output_file=(
                        f"{self.path}/"
                        f"{heatmap_prefix}_cluster{label}_"
                        f"top{len(cluster_feature_idx)}_heatmap.svg"
                    ),
                    title=f"Cluster {label} metabolite trends",
                    xlabel=kwargs.get("xlabel", "Time"),
                    ylabel=kwargs.get("heatmap_ylabel", "Metabolite"),
                )

        plt.tight_layout()
        plt.savefig(
            f"{self.path}/trend_clusters_{cluster_method}_{metric}.svg",
            bbox_inches="tight"
        )
        plt.close()

    def plot_metabolite_entropy(
        self,
        feature: Union[int, str],
        parameterization_key: str = "time",
        window_size: int = 100,
        step_size: int = 50,
        bins: Union[int, str, np.ndarray, list] = "fd",
        feature_name_key: Optional[str] = None,
        normalized: bool = True,
        include_zero: bool = True,
        min_cells_per_window: int = 20,
        linewidth: float = 1.5,
        output_file: Optional[str] = None,
        title: Optional[str] = None,
        ratio_to: Optional[Union[int, str]] = None,
        ratio_log2: bool = True,
        ratio_pseudocount: float = 0.0,
        ratio_require_both_detected: bool = True,
        **kwargs,
    ):
        """Plot sliding-window Shannon entropy for one metabolite or metabolite ratio.

        For a single feature, entropy is estimated from its observed single-cell
        intensity distribution.  When ``ratio_to`` is provided, the analyzed
        quantity is ``feature / ratio_to`` (numerator / denominator), optionally
        transformed to log2 ratio with ``ratio_log2=True``.  By default, ratio
        entropy uses only cells in which both metabolites are detected, matching
        the convention used by the time-resolved metabolite-ratio analyses.

        Histogram bin edges are fixed from the complete usable distribution and
        reused for all time windows so entropy values are comparable over time.
        With ``normalized=True`` entropy is divided by log(number of bins).

        Parameters specific to ratios
        -----------------------------
        ratio_to
            Denominator feature.  It is resolved using the same rules and
            ``feature_name_key`` as ``feature``.  If None, ordinary single-
            metabolite entropy is calculated.
        ratio_log2
            If True (default), calculate log2((feature + pseudocount) /
            (ratio_to + pseudocount)).  If False, use the raw ratio.
        ratio_pseudocount
            Non-negative pseudocount added to numerator and denominator before
            ratio calculation.  Default is 0, so no pseudocount is introduced.
        ratio_require_both_detected
            If True (default), only cells with non-zero numerator and denominator
            signals are eligible for ratio analysis.
        """
        feature_idx = self._resolve_feature_index(feature, feature_name_key)
        X = self._get_X()
        numerator = np.asarray(X[:, feature_idx], dtype=float)

        if ratio_pseudocount < 0:
            raise ValueError("ratio_pseudocount must be >= 0")

        ratio_idx = None
        ratio_display_name = None
        is_ratio = ratio_to is not None

        if feature_name_key is not None and feature_name_key in self.adata.var.columns:
            display_name = str(self.adata.var.iloc[feature_idx][feature_name_key])
            if not display_name.strip() or display_name.lower() == "nan":
                display_name = str(self.adata.var_names[feature_idx])
        else:
            display_name = str(self.adata.var_names[feature_idx])

        if is_ratio:
            ratio_idx = self._resolve_feature_index(ratio_to, feature_name_key)
            denominator = np.asarray(X[:, ratio_idx], dtype=float)

            if feature_name_key is not None and feature_name_key in self.adata.var.columns:
                ratio_display_name = str(self.adata.var.iloc[ratio_idx][feature_name_key])
                if not ratio_display_name.strip() or ratio_display_name.lower() == "nan":
                    ratio_display_name = str(self.adata.var_names[ratio_idx])
            else:
                ratio_display_name = str(self.adata.var_names[ratio_idx])

            base_valid = np.isfinite(numerator) & np.isfinite(denominator)
            if ratio_require_both_detected:
                base_valid &= (numerator != 0) & (denominator != 0)

            num_adj = numerator + float(ratio_pseudocount)
            den_adj = denominator + float(ratio_pseudocount)
            base_valid &= den_adj != 0
            if ratio_log2:
                base_valid &= (num_adj > 0) & (den_adj > 0)

            values = np.full(self.adata.n_obs, np.nan, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                ratio_values = num_adj[base_valid] / den_adj[base_valid]
                if ratio_log2:
                    ratio_values = np.log2(ratio_values)
            values[base_valid] = ratio_values
            quantity_name = (
                f"log2({display_name}/{ratio_display_name})"
                if ratio_log2
                else f"{display_name}/{ratio_display_name}"
            )
        else:
            values = numerator
            quantity_name = display_name

        valid_global = np.isfinite(values)
        if not include_zero:
            valid_global &= values != 0
        global_values = values[valid_global]
        if global_values.size < min_cells_per_window:
            raise ValueError(
                f"{quantity_name!r} has fewer than {min_cells_per_window} usable values."
            )

        if isinstance(bins, str):
            bin_edges = np.histogram_bin_edges(global_values, bins=bins)
        elif np.isscalar(bins):
            n_bins_requested = int(bins)
            if n_bins_requested < 2:
                raise ValueError("bins must be >= 2")
            bin_edges = np.histogram_bin_edges(global_values, bins=n_bins_requested)
        else:
            bin_edges = np.asarray(bins, dtype=float)
            if bin_edges.ndim != 1 or len(bin_edges) < 3:
                raise ValueError(
                    "Explicit bins must contain at least 3 monotonically increasing edges"
                )
            if not np.all(np.diff(bin_edges) > 0):
                raise ValueError("Explicit bin edges must be strictly increasing")

        # Constant features/ratios can make numpy return a degenerate range.
        if len(bin_edges) < 3 or not np.all(np.isfinite(bin_edges)):
            center_value = float(np.nanmedian(global_values))
            eps = max(abs(center_value) * 1e-9, 1e-12)
            bin_edges = np.array(
                [center_value - eps, center_value, center_value + eps], dtype=float
            )

        _, _, windows = self._sliding_window_slices(
            parameterization_key=parameterization_key,
            window_size=window_size,
            step_size=step_size,
            min_cells_per_window=min_cells_per_window,
        )

        time_centers = []
        entropy_values = []
        counts = []
        n_bins = len(bin_edges) - 1
        entropy_scale = np.log(n_bins) if normalized and n_bins > 1 else 1.0

        for window in windows:
            vw = values[window["obs_idx"]]
            valid = np.isfinite(vw)
            if not include_zero:
                valid &= vw != 0
            vw = vw[valid]
            if vw.size < min_cells_per_window:
                continue

            hist, _ = np.histogram(vw, bins=bin_edges)
            total = hist.sum()
            if total == 0:
                continue
            p = hist[hist > 0].astype(float) / float(total)
            h = -np.sum(p * np.log(p))
            if normalized and entropy_scale > 0:
                h /= entropy_scale

            time_centers.append(window["time_center"])
            entropy_values.append(float(h))
            counts.append(int(vw.size))

        if len(time_centers) == 0:
            raise ValueError("No sliding window contained enough usable values")

        time_centers = np.asarray(time_centers, dtype=float)
        entropy_values = np.asarray(entropy_values, dtype=float)
        counts = np.asarray(counts, dtype=int)

        store_key = kwargs.pop("store_key", "metabolite_entropy")
        self.adata.uns[store_key] = {
            "feature_index": int(feature_idx),
            "feature_name": display_name,
            "ratio_to_index": None if ratio_idx is None else int(ratio_idx),
            "ratio_to_name": ratio_display_name,
            "quantity_name": quantity_name,
            "is_ratio": bool(is_ratio),
            "ratio_log2": bool(ratio_log2) if is_ratio else None,
            "ratio_pseudocount": float(ratio_pseudocount) if is_ratio else None,
            "ratio_require_both_detected": (
                bool(ratio_require_both_detected) if is_ratio else None
            ),
            "time_centers": time_centers,
            "entropy": entropy_values,
            "counts": counts,
            "bin_edges": np.asarray(bin_edges, dtype=float),
            "normalized": bool(normalized),
            "include_zero": bool(include_zero),
            "window_size": int(window_size),
            "step_size": int(step_size),
            "parameterization_key": parameterization_key,
        }

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))
        ax.plot(time_centers, entropy_values, color="black", linewidth=linewidth)
        if kwargs.pop("show_points", False):
            ax.scatter(
                time_centers,
                entropy_values,
                s=kwargs.pop("point_size", 12),
                color="black",
            )
        ax.set_xlabel(kwargs.pop("xlabel", parameterization_key))
        ax.set_ylabel(
            kwargs.pop(
                "ylabel",
                "Normalized Shannon entropy" if normalized else "Shannon entropy",
            )
        )
        if title is None:
            title = f"{quantity_name} heterogeneity"
        if title:
            ax.set_title(title)
        for spine in kwargs.pop("hide_spines", ("top", "right")):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

        if output_file is None:
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", quantity_name).strip("_")
            if not safe_name:
                safe_name = f"feature_{feature_idx}"
            output_file = f"{self.path}/metabolite_entropy_{safe_name}.svg"
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return self

    def plot_cell_state_dispersion(
        self,
        parameterization_key: str = "time",
        use_rep: str = "X_pca",
        n_components: Optional[int] = None,
        window_size: int = 100,
        step_size: int = 50,
        center: str = "median",
        statistic: str = "median",
        min_cells_per_window: int = 20,
        linewidth: float = 1.5,
        output_file: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs,
    ):
        """Plot sliding-window multivariate cell-state dispersion.

        For each time-ordered window, cells are represented in ``use_rep`` space
        (typically PCA), their distance to the window center is calculated, and a
        robust summary of those distances is reported as cell-state dispersion.
        """
        if use_rep in {"X", "x", None}:
            Z = self._get_X()
            rep_name = "X"
        else:
            if use_rep not in self.adata.obsm:
                raise KeyError(
                    f"{use_rep!r} not found in self.adata.obsm. "
                    "Run pca()/umap() first or set use_rep='X'."
                )
            Z = np.asarray(self.adata.obsm[use_rep], dtype=float)
            rep_name = use_rep

        if Z.ndim != 2 or Z.shape[0] != self.adata.n_obs:
            raise ValueError(f"Representation {rep_name!r} must have shape (n_obs, n_dimensions)")
        if n_components is not None:
            if n_components < 1:
                raise ValueError("n_components must be >= 1 or None")
            Z = Z[:, :min(int(n_components), Z.shape[1])]

        center = center.lower()
        statistic = statistic.lower()
        if center not in {"mean", "median"}:
            raise ValueError("center must be 'mean' or 'median'")
        if statistic not in {"mean", "median", "rms"}:
            raise ValueError("statistic must be one of {'mean', 'median', 'rms'}")

        _, order, windows = self._sliding_window_slices(
            parameterization_key=parameterization_key,
            window_size=window_size,
            step_size=step_size,
            min_cells_per_window=min_cells_per_window,
        )

        time_centers = []
        dispersion = []
        q25 = []
        q75 = []
        counts = []

        for window in windows:
            Zw = Z[window["obs_idx"]]
            valid = np.isfinite(Zw).all(axis=1)
            Zw = Zw[valid]
            if Zw.shape[0] < min_cells_per_window:
                continue

            if center == "median":
                centroid = np.nanmedian(Zw, axis=0)
            else:
                centroid = np.nanmean(Zw, axis=0)

            distances = np.linalg.norm(Zw - centroid[None, :], axis=1)
            if statistic == "median":
                d = np.nanmedian(distances)
            elif statistic == "mean":
                d = np.nanmean(distances)
            else:
                d = np.sqrt(np.nanmean(distances ** 2))

            time_centers.append(window["time_center"])
            dispersion.append(float(d))
            q25.append(float(np.nanquantile(distances, 0.25)))
            q75.append(float(np.nanquantile(distances, 0.75)))
            counts.append(int(len(distances)))

        if len(time_centers) == 0:
            raise ValueError("No sliding window contained enough finite cells")

        time_centers = np.asarray(time_centers, dtype=float)
        dispersion = np.asarray(dispersion, dtype=float)
        q25 = np.asarray(q25, dtype=float)
        q75 = np.asarray(q75, dtype=float)
        counts = np.asarray(counts, dtype=int)

        store_key = kwargs.pop("store_key", "cell_state_dispersion")
        self.adata.uns[store_key] = {
            "time_centers": time_centers,
            "dispersion": dispersion,
            "q25": q25,
            "q75": q75,
            "counts": counts,
            "use_rep": rep_name,
            "n_components": int(Z.shape[1]),
            "center": center,
            "statistic": statistic,
            "window_size": int(window_size),
            "step_size": int(step_size),
            "parameterization_key": parameterization_key,
        }

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (6, 4)))
        ax.plot(time_centers, dispersion, color="black", linewidth=linewidth)
        if kwargs.pop("show_iqr", True):
            ax.fill_between(time_centers, q25, q75, alpha=0.18, color="black", linewidth=0)
        ax.set_xlabel(kwargs.pop("xlabel", parameterization_key))
        ax.set_ylabel(kwargs.pop("ylabel", "Cell-state dispersion"))
        if title is not None:
            ax.set_title(title)
        for spine in kwargs.pop("hide_spines", ("top", "right")):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

        if output_file is None:
            output_file = f"{self.path}/cell_state_dispersion.svg"
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return self

    def plot_cluster_time_distribution(
        self,
        cluster_key: str = "clusters",
        parameterization_key: str = "time",
        window_size: int = 100,
        step_size: int = 50,
        normalize: bool = True,
        min_cells_per_window: int = 20,
        kind: str = "stack",
        cmap: str = "Set2",
        linewidth: float = 1.5,
        output_file: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs,
    ):
        """Plot the temporal occupancy/distribution of precomputed cell clusters."""
        if cluster_key not in self.adata.obs.columns:
            raise KeyError(
                f"{cluster_key!r} not found in self.adata.obs. "
                "Run cluster_cells(..., key_added=cluster_key) first."
            )
        kind = kind.lower()
        if kind not in {"stack", "line"}:
            raise ValueError("kind must be 'stack' or 'line'")

        labels_raw = self.adata.obs[cluster_key]
        labels = labels_raw.astype(str).to_numpy()
        if isinstance(labels_raw.dtype, pd.CategoricalDtype):
            cluster_names = [str(x) for x in labels_raw.cat.categories]
        else:
            cluster_names = sorted(pd.unique(labels).tolist())

        _, order, windows = self._sliding_window_slices(
            parameterization_key=parameterization_key,
            window_size=window_size,
            step_size=step_size,
            min_cells_per_window=min_cells_per_window,
        )

        time_centers = []
        counts = []
        totals = []
        for window in windows:
            lw = labels[window["obs_idx"]]
            row = np.array([(lw == lab).sum() for lab in cluster_names], dtype=float)
            time_centers.append(window["time_center"])
            counts.append(row)
            totals.append(len(lw))

        time_centers = np.asarray(time_centers, dtype=float)
        counts = np.vstack(counts)
        totals = np.asarray(totals, dtype=int)
        values = counts / totals[:, None] if normalize else counts.copy()

        store_key = kwargs.pop("store_key", f"cluster_time_distribution_{cluster_key}")
        self.adata.uns[store_key] = {
            "time_centers": time_centers,
            "cluster_names": np.asarray(cluster_names, dtype=str),
            "counts": counts,
            "values": values,
            "totals": totals,
            "normalize": bool(normalize),
            "window_size": int(window_size),
            "step_size": int(step_size),
            "parameterization_key": parameterization_key,
            "cluster_key": cluster_key,
        }

        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4.5)))
        palette = sns.color_palette(cmap, n_colors=len(cluster_names))
        if kind == "stack":
            ax.stackplot(
                time_centers,
                values.T,
                labels=cluster_names,
                colors=palette,
                alpha=kwargs.pop("alpha", 0.9),
            )
        else:
            for i, lab in enumerate(cluster_names):
                ax.plot(
                    time_centers,
                    values[:, i],
                    label=str(lab),
                    color=palette[i],
                    linewidth=linewidth,
                )

        ax.set_xlabel(kwargs.pop("xlabel", parameterization_key))
        ax.set_ylabel(kwargs.pop("ylabel", "Cluster fraction" if normalize else "Cell count"))
        if normalize:
            ax.set_ylim(0, 1)
        if title is None:
            title = "Cluster distribution over time"
        if title:
            ax.set_title(title)
        ax.legend(title=kwargs.pop("legend_title", "Cluster"), frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
        for spine in kwargs.pop("hide_spines", ("top", "right")):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()

        if output_file is None:
            output_file = f"{self.path}/{cluster_key}_time_distribution.svg"
        fig.savefig(output_file, bbox_inches="tight")
        plt.close(fig)
        return self

    def cluster_cells(
        self,
        method: str = "leiden",
        key_added: str = "clusters",
        n_neighbors: int = 15,
        resolution: float = 1.0,
        random_state: int = 0,
        figsize=(6, 6),
        s: float = 8,
        cmap: str = "Set2",
        plot_time_distribution: bool = False,
        parameterization_key: str = "time",
        time_window_size: int = 100,
        time_step_size: int = 50,
        time_distribution_kind: str = "stack",
        time_distribution_normalize: bool = True,
        **kwargs
    ):

        method = method.lower()
        if method not in ["leiden", "louvain"]:
            raise ValueError("method must be 'leiden' or 'louvain'")

        X_cluster = self.adata.obsm.get("X_pca", None)
        if X_cluster is None:
            raise KeyError("X_pca not found in self.adata.obsm")

        X_plot = self.adata.obsm.get("X_umap", None)
        if X_plot is None:
            raise KeyError("X_umap not found in self.adata.obsm")
        if X_plot.shape[1] < 2:
            raise ValueError("X_umap must contain at least 2 dimensions")

        knn = kneighbors_graph(
            X_cluster,
            n_neighbors=n_neighbors,
            mode="connectivity",
            include_self=False
        )

        sources, targets = knn.nonzero()
        edges = list(zip(sources.tolist(), targets.tolist()))

        if method == "leiden":
            try:
                import igraph as ig
                import leidenalg
            except ImportError:
                raise ImportError(
                    "Leiden clustering requires igraph and leidenalg. "
                    "Please install them via 'pip install igraph leidenalg'."
                )

            g = ig.Graph(n=X_cluster.shape[0], edges=edges, directed=False)
            g.simplify()

            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=random_state,
                **kwargs
            )
            labels = np.array(partition.membership, dtype=int)

        elif method == "louvain":
            try:
                import networkx as nx
                from networkx.algorithms.community import louvain_communities
            except ImportError:
                raise ImportError(
                    "Louvain clustering requires networkx. "
                    "Please install it via 'pip install networkx'."
                )

            G = nx.Graph()
            G.add_nodes_from(range(X_cluster.shape[0]))
            G.add_edges_from(edges)

            communities = louvain_communities(
                G,
                resolution=resolution,
                seed=random_state,
                **kwargs
            )

            labels = np.empty(X_cluster.shape[0], dtype=int)
            for i, comm in enumerate(communities):
                for node in comm:
                    labels[node] = i

        self.adata.obs[key_added] = pd.Categorical(labels.astype(str))
        self.adata.uns[f"{key_added}_params"] = {
            "method": method,
            "source": "X_pca",
            "n_neighbors": int(n_neighbors),
            "resolution": float(resolution),
            "random_state": int(random_state),
        }

        if figsize is None or figsize == "auto":
            figsize = (6, 6)

        fig, ax = plt.subplots(figsize=figsize)

        uniq = np.unique(labels)
        palette = sns.color_palette(cmap, n_colors=len(uniq))
        color_map = {lab: palette[i] for i, lab in enumerate(uniq)}

        for i, lab in enumerate(uniq):
            idx = labels == lab
            ax.scatter(
                X_plot[idx, 0],
                X_plot[idx, 1],
                s=s,
                color=color_map[lab],
                linewidths=0,
                alpha=0.85
            )

            x_center = np.median(X_plot[idx, 0])
            y_center = np.median(X_plot[idx, 1])
            ax.text(
                x_center,
                y_center,
                str(lab),
                fontsize=12,
                ha="center",
                va="center",
                weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
            )

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"{method.capitalize()} clustering")
        plt.tight_layout()
        plt.savefig(f"{self.path}/{method}_{key_added}_umap.svg", bbox_inches="tight")
        plt.close(fig)

        if plot_time_distribution:
            self.plot_cluster_time_distribution(
                cluster_key=key_added,
                parameterization_key=parameterization_key,
                window_size=time_window_size,
                step_size=time_step_size,
                normalize=time_distribution_normalize,
                kind=time_distribution_kind,
                cmap=cmap,
            )
        return self
        
    def feature_network(
        self,
        name_key: str = None,
        class_key: str = None,
        metric: str = "glasso",
        target_label: str = None,
        label_key: str = None,
        obs_key: str = None,
        obs_range: tuple = None,
        save_path: str = None,
        title: str = None,
        alpha: float = 0.3,
        top_nodes: int = 50,
        edge_threshold: float = 0.01,
        figsize=(10, 8),
        layout_seed: int = 0,
        min_class_size: int = 1,
        other_label: str = "Other",
        annotated_only: bool = True,
        colorbar_vmin: float = None,
        colorbar_vmax: float = None,
        show_node_size_legend: bool = True,
        node_size_legend_title: str = "intensity",
        node_size_legend_values=None,
        node_size_legend_range: tuple = None,
        node_size_legend_n: int = 3,

        # ------------------------------------------------------------------
        # Numbered metabolite markers
        # ------------------------------------------------------------------
        numbered_nodes=None,
        numbered_key: str = None,
        numbered_exact: bool = True,
        numbered_case_sensitive: bool = False,
        force_include_numbered: bool = True,
        preserve_numbered_isolates: bool = True,
        numbered_size_multiplier: float = 1.35,
        number_font_size: int = 8,
        number_font_weight: str = "bold",
        number_font_color: str = "black",
        number_bbox: bool = True,
        number_legend_title: str = "Marked metabolites",
        number_legend_loc: str = "lower left",
        number_legend_bbox_to_anchor: tuple = (0.76, 0.02),
        **kwargs
    ):
        metabolite_name_key = kwargs.pop("metabolite_name_key", name_key)
        metabolite_label_key = kwargs.pop("metabolite_label_key", class_key)

        method = kwargs.pop("method", metric)
        method = str(method).lower()

        if method in {"correlation", "corr"}:
            method = "pearson"

        if method not in {"glasso", "pearson"}:
            raise ValueError("metric/method must be 'glasso' or 'pearson'.")

        if numbered_nodes is None:
            numbered_nodes = []

        if isinstance(numbered_nodes, str):
            numbered_nodes = [numbered_nodes]

        numbered_nodes = list(numbered_nodes)

        # ------------------------------------------------------------------
        # 1. Select samples
        # ------------------------------------------------------------------
        X_all = self._get_X()

        if X_all.shape[0] != self.adata.n_obs:
            raise ValueError(
                f"X sample number mismatch: X has {X_all.shape[0]} rows, "
                f"but self.adata has {self.adata.n_obs} observations."
            )

        selection_mode = "global"
        selection_label = "global"

        if obs_key is not None or obs_range is not None:
            if obs_key is None or obs_range is None:
                raise ValueError(
                    "obs_key and obs_range must be provided together. "
                    "Example: obs_key='true_time', obs_range=(0, 6)."
                )

            if obs_key not in self.adata.obs.columns:
                raise KeyError(f"{obs_key!r} not found in self.adata.obs.")

            if not isinstance(obs_range, (tuple, list)) or len(obs_range) != 2:
                raise ValueError(
                    "obs_range must be a tuple/list of length 2, e.g. (0, 6)."
                )

            obs_min, obs_max = obs_range

            if obs_min is None or obs_max is None:
                raise ValueError("obs_range values cannot be None.")

            if obs_min >= obs_max:
                raise ValueError(
                    f"obs_range must satisfy min < max, got {obs_range}."
                )

            obs_values = pd.to_numeric(
                self.adata.obs[obs_key],
                errors="coerce"
            ).values

            include_left = kwargs.get("include_left", True)
            include_right = kwargs.get("include_right", False)

            if include_left:
                left_mask = obs_values >= obs_min
            else:
                left_mask = obs_values > obs_min

            if include_right:
                right_mask = obs_values <= obs_max
            else:
                right_mask = obs_values < obs_max

            obs_mask = (
                np.isfinite(obs_values)
                & left_mask
                & right_mask
            )

            selection_mode = "obs_range"

            obs_min_str = str(obs_min).replace(".", "p").replace("-", "m")
            obs_max_str = str(obs_max).replace(".", "p").replace("-", "m")
            selection_label = f"{obs_key}_{obs_min_str}_{obs_max_str}"

        elif target_label is not None and label_key is not None:
            if label_key not in self.adata.obs.columns:
                raise KeyError(f"{label_key!r} not found in self.adata.obs.")

            obs_mask = np.asarray(self.adata.obs[label_key] == target_label)

            selection_mode = "label"
            selection_label = str(target_label)

        else:
            obs_mask = np.ones(self.adata.n_obs, dtype=bool)

        obs_mask = np.asarray(obs_mask, dtype=bool)

        if obs_mask.ndim != 1:
            raise ValueError("obs_mask must be one-dimensional.")

        if obs_mask.shape[0] != self.adata.n_obs:
            raise ValueError(
                f"obs_mask length mismatch: got {obs_mask.shape[0]}, "
                f"expected {self.adata.n_obs}."
            )

        n_selected_samples = int(np.sum(obs_mask))

        if n_selected_samples == 0:
            raise ValueError("No samples are available for the selected condition.")

        min_samples = kwargs.get("min_samples", 3)

        if n_selected_samples < min_samples:
            raise ValueError(
                f"Too few samples for network construction: "
                f"{n_selected_samples} selected, but min_samples={min_samples}."
            )

        X = X_all[obs_mask, :]
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("self.adata.X must be a 2D matrix: samples x features.")

        # ------------------------------------------------------------------
        # 2. Keep only annotated metabolites if requested
        # ------------------------------------------------------------------
        if annotated_only:
            if metabolite_name_key is None:
                raise ValueError(
                    "annotated_only=True requires name_key or metabolite_name_key."
                )

            if metabolite_name_key not in self.adata.var.columns:
                raise KeyError(
                    f"{metabolite_name_key!r} not found in self.adata.var."
                )

            raw_names = self.adata.var[metabolite_name_key]
            name_string = raw_names.astype(str).str.strip()
            name_lower = name_string.str.lower()

            annotated_feature_mask = (
                raw_names.notna()
                & (name_string != "")
                & (~name_lower.isin(
                    {
                        "nan",
                        "none",
                        "na",
                        "n/a",
                        "unknown",
                        "unannotated",
                        "not annotated",
                        "undefined",
                        "null",
                    }
                ))
            ).values

            if not np.any(annotated_feature_mask):
                raise ValueError(
                    f"No annotated metabolites found using {metabolite_name_key!r}."
                )

            feature_original_indices = np.where(annotated_feature_mask)[0]
            X = X[:, annotated_feature_mask]
            var_for_network = self.adata.var.iloc[annotated_feature_mask].copy()

        else:
            feature_original_indices = np.arange(self.adata.n_vars)
            var_for_network = self.adata.var.copy()

        n_samples, n_features = X.shape

        if n_features != var_for_network.shape[0]:
            raise ValueError(
                f"Feature number mismatch: X has {n_features} columns, "
                f"but var_for_network has {var_for_network.shape[0]} rows."
            )

        # ------------------------------------------------------------------
        # 3. Basic cleaning
        # ------------------------------------------------------------------
        zero_tol = kwargs.get("zero_tol", 0.0)
        var_tol = kwargs.get("var_tol", 1e-12)
        keep_zero_features = kwargs.get("keep_zero_features", False)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if kwargs.get("clip_negative_to_zero", True):
            X = np.where(X < 0, 0.0, X)

        X_log = np.log1p(X)

        if zero_tol > 0:
            all_zero_feature_mask = np.all(np.abs(X) <= zero_tol, axis=0)
        else:
            all_zero_feature_mask = np.all(X == 0, axis=0)

        feature_var = np.nanvar(X_log, axis=0)
        zero_variance_mask = feature_var <= var_tol

        model_feature_mask = ~zero_variance_mask

        valid_feature_indices = np.where(model_feature_mask)[0]
        invalid_feature_indices = np.where(~model_feature_mask)[0]

        # ------------------------------------------------------------------
        # 4. Feature names
        # ------------------------------------------------------------------
        if metabolite_name_key is not None and metabolite_name_key in var_for_network.columns:
            raw_feature_names = var_for_network[metabolite_name_key].values
            feature_names = pd.Series(raw_feature_names).astype(str).str.strip().values

            empty_name_mask = (
                pd.isna(raw_feature_names)
                | (pd.Series(raw_feature_names).astype(str).str.strip() == "").values
                | (pd.Series(raw_feature_names).astype(str).str.lower().str.strip() == "nan").values
                | (pd.Series(raw_feature_names).astype(str).str.lower().str.strip() == "none").values
            )

            feature_names[empty_name_mask] = [
                f"feature_{feature_original_indices[i]}"
                for i in np.where(empty_name_mask)[0]
            ]

        else:
            feature_names = np.asarray(var_for_network.index).astype(str)

        feature_names_index = pd.Index(feature_names).astype(str)

        if not feature_names_index.is_unique:
            counts = {}
            unique_names = []

            for x in feature_names_index:
                if x not in counts:
                    counts[x] = 0
                    unique_names.append(x)
                else:
                    counts[x] += 1
                    unique_names.append(f"{x}_{counts[x]}")

            feature_names = np.asarray(unique_names, dtype=str)
        else:
            feature_names = np.asarray(feature_names_index, dtype=str)

        index_to_node = {
            i: feature_names[i]
            for i in range(n_features)
        }

        node_to_index = {
            feature_names[i]: i
            for i in range(n_features)
        }

        # ------------------------------------------------------------------
        # 4.5 Resolve numbered metabolites
        # ------------------------------------------------------------------
        def _normalize_numbered_value(x):
            x = str(x).strip()
            if not numbered_case_sensitive:
                x = x.lower()
            return x

        numbered_query_values = {
            _normalize_numbered_value(x)
            for x in numbered_nodes
        }

        numbered_node_set = set()
        numbered_node_order = []
        numbered_match_table_records = []

        if len(numbered_query_values) > 0:
            if numbered_key is not None:
                if numbered_key not in var_for_network.columns:
                    raise KeyError(
                        f"numbered_key={numbered_key!r} not found in self.adata.var "
                        f"after annotation filtering."
                    )

                values_to_match = (
                    var_for_network[numbered_key]
                    .astype(str)
                    .str.strip()
                    .values
                )

                for query_raw in numbered_nodes:
                    query_norm = _normalize_numbered_value(query_raw)

                    for i, raw_value in enumerate(values_to_match):
                        candidate = _normalize_numbered_value(raw_value)

                        if numbered_exact:
                            matched = candidate == query_norm
                        else:
                            matched = query_norm in candidate

                        if matched:
                            node_name = index_to_node[i]

                            if node_name not in numbered_node_set:
                                numbered_node_set.add(node_name)
                                numbered_node_order.append(node_name)

                            numbered_match_table_records.append({
                                "requested_value": query_raw,
                                "matched_value": raw_value,
                                "matched_node": node_name,
                                "matched_by": numbered_key,
                                "feature_index_in_network": i,
                                "feature_index_in_adata": int(feature_original_indices[i])
                            })

                            if numbered_exact:
                                break

            else:
                for query_raw in numbered_nodes:
                    query_norm = _normalize_numbered_value(query_raw)

                    for i, node_name in index_to_node.items():
                        candidate = _normalize_numbered_value(node_name)

                        if numbered_exact:
                            matched = candidate == query_norm
                        else:
                            matched = query_norm in candidate

                        if matched:
                            if node_name not in numbered_node_set:
                                numbered_node_set.add(node_name)
                                numbered_node_order.append(node_name)

                            numbered_match_table_records.append({
                                "requested_value": query_raw,
                                "matched_value": node_name,
                                "matched_node": node_name,
                                "matched_by": "node_name",
                                "feature_index_in_network": i,
                                "feature_index_in_adata": int(feature_original_indices[i])
                            })

                            if numbered_exact:
                                break

        numbered_match_table = pd.DataFrame(numbered_match_table_records)

        node_to_number = {
            node: i + 1
            for i, node in enumerate(numbered_node_order)
        }

        # ------------------------------------------------------------------
        # 5. Metabolite classes; small classes -> Other only if requested
        # ------------------------------------------------------------------
        node_to_class = {}
        class_values = None
        class_counts_before_merging = None
        class_counts_after_merging = None

        if metabolite_label_key is not None and metabolite_label_key in var_for_network.columns:
            raw_class = var_for_network[metabolite_label_key]

            class_series = raw_class.copy()
            class_series = class_series.where(class_series.notna(), other_label)
            class_series = class_series.astype(str).str.strip()
            class_series = class_series.replace(
                {
                    "": other_label,
                    "nan": other_label,
                    "NaN": other_label,
                    "None": other_label,
                    "none": other_label,
                    "NA": other_label,
                    "na": other_label,
                    "N/A": other_label,
                    "n/a": other_label,
                    "Unknown": other_label,
                    "unknown": other_label,
                    "Unannotated": other_label,
                    "unannotated": other_label,
                }
            )

            class_counts_before_merging = class_series.value_counts(dropna=False)

            if min_class_size is not None and min_class_size > 1:
                small_classes = set(
                    class_counts_before_merging[
                        class_counts_before_merging < min_class_size
                    ].index.astype(str)
                )

                class_series = class_series.where(
                    ~class_series.isin(small_classes),
                    other_label
                )

            class_counts_after_merging = class_series.value_counts(dropna=False)
            class_values = class_series.values.astype(str)

            node_to_class = {
                feature_names[i]: class_values[i]
                for i in range(n_features)
            }

            network_class_key = f"{metabolite_label_key}_network"

            self.adata.var[network_class_key] = pd.Series(
                other_label,
                index=self.adata.var.index,
                dtype="object"
            )

            self.adata.var.iloc[
                feature_original_indices,
                self.adata.var.columns.get_loc(network_class_key)
            ] = class_values

        # ------------------------------------------------------------------
        # 6. Initialize full matrices
        # ------------------------------------------------------------------
        A = np.zeros((n_features, n_features), dtype=float)
        partial_corr = np.zeros((n_features, n_features), dtype=float)

        # ------------------------------------------------------------------
        # 7. Build model network
        # ------------------------------------------------------------------
        if len(valid_feature_indices) < 2:
            warnings.warn(
                "Fewer than 2 non-constant features are available. "
                "Returning an empty network."
            )

            G = nx.Graph()

            if keep_zero_features:
                for i in range(n_features):
                    G.add_node(index_to_node[i])

        else:
            X_model = X_log[:, valid_feature_indices]

            scaler = StandardScaler()
            X_std = scaler.fit_transform(X_model)

            X_std = np.nan_to_num(
                X_std,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            if method == "glasso":
                model = GraphicalLasso(
                    alpha=alpha,
                    max_iter=kwargs.get("max_iter", 500),
                    tol=kwargs.get("tol", 1e-4)
                )

                try:
                    model.fit(X_std)
                    theta_sub = model.precision_

                    d = np.sqrt(np.diag(theta_sub))
                    d[d <= var_tol] = np.nan

                    partial_corr_sub = -theta_sub / np.outer(d, d)
                    partial_corr_sub = np.nan_to_num(
                        partial_corr_sub,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0
                    )

                    np.fill_diagonal(partial_corr_sub, 0.0)

                except Exception as e:
                    if kwargs.get("fallback_to_pearson", True):
                        warnings.warn(
                            f"GraphicalLasso failed: {e}. "
                            "Falling back to Pearson correlation."
                        )

                        corr_sub = np.corrcoef(X_std, rowvar=False)
                        corr_sub = np.nan_to_num(
                            corr_sub,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0
                        )
                        np.fill_diagonal(corr_sub, 0.0)
                        partial_corr_sub = corr_sub
                    else:
                        raise e

            elif method == "pearson":
                corr_sub = np.corrcoef(X_std, rowvar=False)
                corr_sub = np.nan_to_num(
                    corr_sub,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0
                )
                np.fill_diagonal(corr_sub, 0.0)
                partial_corr_sub = corr_sub

            A_sub = np.abs(partial_corr_sub)
            A_sub[A_sub < kwargs.get("matrix_zero_threshold", 1e-6)] = 0.0

            row_idx, col_idx = np.ix_(valid_feature_indices, valid_feature_indices)

            partial_corr[row_idx, col_idx] = partial_corr_sub
            A[row_idx, col_idx] = A_sub

            G = nx.from_numpy_array(A)
            G = nx.relabel_nodes(G, index_to_node)

            if not keep_zero_features:
                invalid_nodes = [
                    index_to_node[i]
                    for i in invalid_feature_indices
                ]

                if preserve_numbered_isolates:
                    invalid_nodes = [
                        n for n in invalid_nodes
                        if n not in numbered_node_set
                    ]

                G.remove_nodes_from(invalid_nodes)

            weak_edges = [
                (u, v)
                for u, v, data in G.edges(data=True)
                if abs(data.get("weight", 0.0)) < edge_threshold
            ]
            G.remove_edges_from(weak_edges)

            if kwargs.get("remove_isolates_before_top", False):
                isolates = list(nx.isolates(G))

                if preserve_numbered_isolates:
                    isolates = [
                        n for n in isolates
                        if n not in numbered_node_set
                    ]

                G.remove_nodes_from(isolates)

        # ------------------------------------------------------------------
        # 8. Centrality
        # ------------------------------------------------------------------
        if G.number_of_nodes() == 0:
            centrality = {
                "degree": {},
                "betweenness": {},
                "eigenvector": {},
                "pagerank": {}
            }

            H = G.copy()

        else:
            centrality = {}

            try:
                centrality["degree"] = nx.degree_centrality(G)
            except Exception:
                centrality["degree"] = {n: 0.0 for n in G.nodes()}

            try:
                centrality["betweenness"] = nx.betweenness_centrality(
                    G,
                    weight="weight"
                )
            except Exception:
                centrality["betweenness"] = {n: 0.0 for n in G.nodes()}

            try:
                if G.number_of_edges() > 0 and G.number_of_nodes() > 1:
                    centrality["eigenvector"] = nx.eigenvector_centrality_numpy(
                        G,
                        weight="weight"
                    )
                else:
                    centrality["eigenvector"] = {n: 0.0 for n in G.nodes()}
            except Exception:
                centrality["eigenvector"] = {n: 0.0 for n in G.nodes()}

            try:
                centrality["pagerank"] = nx.pagerank(G, weight="weight")
            except Exception:
                centrality["pagerank"] = {
                    n: 1.0 / G.number_of_nodes()
                    for n in G.nodes()
                }

            keep = sorted(
                centrality["pagerank"],
                key=centrality["pagerank"].get,
                reverse=True
            )[:top_nodes]

            if force_include_numbered:
                keep = list(dict.fromkeys(keep + numbered_node_order))

            keep = [
                n for n in keep
                if n in G.nodes()
            ]

            H = G.subgraph(keep).copy()

        # ------------------------------------------------------------------
        # 9. Prepare edges for plotting
        # ------------------------------------------------------------------
        edges = []
        edge_colors = []
        edge_widths = []

        for u, v in H.edges():
            i = node_to_index.get(u, None)
            j = node_to_index.get(v, None)

            if i is None or j is None:
                continue

            w = partial_corr[i, j]

            if not np.isfinite(w):
                continue

            if abs(w) < edge_threshold:
                continue

            edges.append((u, v))
            edge_colors.append(w)
            edge_widths.append(
                max(
                    kwargs.get("edge_width_min", 0.15),
                    abs(w) * kwargs.get("edge_width_multiplier", 1.5)
                )
            )

        H.remove_edges_from(list(H.edges()))
        H.add_edges_from(edges)

        if kwargs.get("remove_isolates", True):
            isolates = list(nx.isolates(H))

            if preserve_numbered_isolates:
                isolates = [
                    n for n in isolates
                    if n not in numbered_node_set
                ]

            H.remove_nodes_from(isolates)

        numbered_nodes_in_H = [
            n for n in numbered_node_order
            if n in H.nodes()
        ]

        # ------------------------------------------------------------------
        # 10. Node size
        # ------------------------------------------------------------------
        mean_intensity = np.nanmean(X_log, axis=0)

        node_size_base = kwargs.get("node_size_base", 70)
        node_size_multiplier = kwargs.get("node_size_multiplier", 25)

        node_sizes = []

        for n in H.nodes():
            i = node_to_index.get(n, None)

            if i is None:
                size = node_size_base
            else:
                size = (
                    node_size_base
                    + node_size_multiplier * mean_intensity[i]
                )

            if not np.isfinite(size):
                size = node_size_base

            if n in numbered_node_set:
                size = size * numbered_size_multiplier

            node_sizes.append(size)

        # ------------------------------------------------------------------
        # 11. Layout
        # ------------------------------------------------------------------
        if H.number_of_nodes() > 0:
            pos = nx.spring_layout(
                H,
                seed=layout_seed,
                weight="weight",
                k=kwargs.get("layout_k", 0.9),
                iterations=kwargs.get("layout_iterations", 300),
                scale=kwargs.get("layout_scale", 1.0)
            )

            if kwargs.get("compact_layout", True) and len(pos) > 0:
                coords = np.asarray(list(pos.values()), dtype=float)
                center = np.mean(coords, axis=0)

                radial_compress = kwargs.get("radial_compress", 0.55)

                pos = {
                    node: center + radial_compress * (coord - center)
                    for node, coord in pos.items()
                }

        else:
            pos = {}

        # ------------------------------------------------------------------
        # 12. Node colors by metabolite class
        # ------------------------------------------------------------------
        node_colors = "lightgray"
        category_color_map = None

        if class_values is not None:
            unique_cat = sorted(
                {
                    node_to_class.get(n, other_label)
                    for n in H.nodes()
                }
            )

            if len(unique_cat) > 0:
                cmap_obj = plt.get_cmap(
                    kwargs.get("category_cmap", "tab20"),
                    len(unique_cat)
                )

                category_color_map = {
                    c: cmap_obj(i)
                    for i, c in enumerate(unique_cat)
                }

                node_colors = [
                    category_color_map.get(
                        node_to_class.get(n, other_label),
                        "lightgray"
                    )
                    for n in H.nodes()
                ]

        node_edgecolors = kwargs.get("node_edgecolors", "black")
        node_linewidths = kwargs.get("node_linewidths", 0.7)

        # ------------------------------------------------------------------
        # 13. Resolve edge and colorbar scale
        # ------------------------------------------------------------------
        edge_vmin = None
        edge_vmax = None
        resolved_colorbar_vmin = None
        resolved_colorbar_vmax = None

        if edge_colors:
            auto_vmin = float(min(edge_colors))
            auto_vmax = float(max(edge_colors))

            edge_vmin = kwargs.get("edge_vmin", colorbar_vmin)
            edge_vmax = kwargs.get("edge_vmax", colorbar_vmax)

            resolved_colorbar_vmin = colorbar_vmin
            resolved_colorbar_vmax = colorbar_vmax

            if edge_vmin is None:
                edge_vmin = auto_vmin

            if edge_vmax is None:
                edge_vmax = auto_vmax

            if resolved_colorbar_vmin is None:
                resolved_colorbar_vmin = edge_vmin

            if resolved_colorbar_vmax is None:
                resolved_colorbar_vmax = edge_vmax

            edge_vmin = float(edge_vmin)
            edge_vmax = float(edge_vmax)
            resolved_colorbar_vmin = float(resolved_colorbar_vmin)
            resolved_colorbar_vmax = float(resolved_colorbar_vmax)

            if not np.isfinite(edge_vmin) or not np.isfinite(edge_vmax):
                raise ValueError(
                    f"edge_vmin/edge_vmax must be finite numbers, "
                    f"got edge_vmin={edge_vmin}, edge_vmax={edge_vmax}."
                )

            if not np.isfinite(resolved_colorbar_vmin) or not np.isfinite(resolved_colorbar_vmax):
                raise ValueError(
                    f"colorbar_vmin/colorbar_vmax must be finite numbers, "
                    f"got colorbar_vmin={resolved_colorbar_vmin}, "
                    f"colorbar_vmax={resolved_colorbar_vmax}."
                )

            if edge_vmin >= edge_vmax:
                raise ValueError(
                    f"edge_vmin must be smaller than edge_vmax, "
                    f"got edge_vmin={edge_vmin}, edge_vmax={edge_vmax}."
                )

            if resolved_colorbar_vmin >= resolved_colorbar_vmax:
                raise ValueError(
                    f"colorbar_vmin must be smaller than colorbar_vmax, "
                    f"got colorbar_vmin={resolved_colorbar_vmin}, "
                    f"colorbar_vmax={resolved_colorbar_vmax}."
                )

        # ------------------------------------------------------------------
        # 14. Plot
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        if H.number_of_nodes() == 0:
            ax.text(
                0.5,
                0.5,
                "No valid network\nall features are zero/constant or no edges pass threshold",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=kwargs.get("empty_text_fontsize", 11)
            )

        else:
            if edges:
                edge_collection = nx.draw_networkx_edges(
                    H,
                    pos,
                    edge_color=edge_colors,
                    edge_cmap=plt.cm.coolwarm,
                    width=edge_widths,
                    edge_vmin=edge_vmin,
                    edge_vmax=edge_vmax,
                    alpha=kwargs.get("edge_alpha", 0.28),
                    ax=ax
                )

                if edge_collection is not None:
                    if isinstance(edge_collection, list):
                        for ec in edge_collection:
                            ec.set_zorder(1)
                    else:
                        edge_collection.set_zorder(1)

            nodes = nx.draw_networkx_nodes(
                H,
                pos,
                node_size=node_sizes,
                node_color=node_colors,
                edgecolors=node_edgecolors,
                linewidths=node_linewidths,
                ax=ax
            )

            nodes.set_zorder(3)

            if kwargs.get("show_labels", False):
                labels = nx.draw_networkx_labels(
                    H,
                    pos,
                    font_size=kwargs.get("label_fontsize", 8),
                    ax=ax
                )

                for text in labels.values():
                    text.set_zorder(4)

            # --------------------------------------------------------------
            # Draw numbers on selected nodes
            # --------------------------------------------------------------
            if len(numbered_nodes_in_H) > 0:
                number_labels = {
                    n: str(node_to_number[n])
                    for n in numbered_nodes_in_H
                    if n in node_to_number
                }

                if number_bbox:
                    bbox = dict(
                        boxstyle="circle,pad=0.18",
                        facecolor="white",
                        edgecolor="none",
                        alpha=0.75
                    )
                else:
                    bbox = None

                number_texts = nx.draw_networkx_labels(
                    H,
                    pos,
                    labels=number_labels,
                    font_size=number_font_size,
                    font_weight=number_font_weight,
                    font_color=number_font_color,
                    bbox=bbox,
                    ax=ax
                )

                for text in number_texts.values():
                    text.set_zorder(6)

        # ------------------------------------------------------------------
        # 15. Figure layout
        # ------------------------------------------------------------------
        ax.axis("off")

        if title is not None:
            ax.set_title(title)
        else:
            if selection_mode == "obs_range":
                ax.set_title(
                    f"{obs_key}: {obs_range[0]} to {obs_range[1]} "
                    f"n={n_selected_samples}"
                )
            elif selection_mode == "label":
                ax.set_title(
                    f"{label_key}: {target_label} "
                    f"n={n_selected_samples}"
                )

        fig.subplots_adjust(
            left=kwargs.get("subplots_left", 0.02),
            right=kwargs.get("subplots_right", 0.72),
            top=kwargs.get("subplots_top", 0.95),
            bottom=kwargs.get("subplots_bottom", 0.05)
        )

        # ------------------------------------------------------------------
        # 16. Edge colorbar + node-size legend
        # ------------------------------------------------------------------
        if edge_colors and kwargs.get("colorbar", True):
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.coolwarm,
                norm=plt.Normalize(
                    resolved_colorbar_vmin,
                    resolved_colorbar_vmax
                )
            )
            sm.set_array([])

            colorbar_axes = kwargs.get(
                "colorbar_axes",
                [0.76, 0.15, 0.035, 0.7]
            )
            cbar_ax = fig.add_axes(colorbar_axes)

            fig.colorbar(
                sm,
                cax=cbar_ax,
                label=kwargs.get(
                    "colorbar_label",
                    "Partial correlation" if method == "glasso" else "Pearson correlation"
                )
            )

            # --------------------------------------------------------------
            # Node-size legend: intensity
            # The legend uses the same size mapping as ordinary network nodes:
            # node_size = node_size_base + node_size_multiplier * intensity.
            # The extra numbered_size_multiplier is intentionally not included.
            # --------------------------------------------------------------
            if show_node_size_legend and H.number_of_nodes() > 0:
                # Resolve and validate an optional hard legend range first.
                legend_range = None
                if node_size_legend_range is not None:
                    if (
                        not isinstance(node_size_legend_range, (tuple, list))
                        or len(node_size_legend_range) != 2
                    ):
                        raise ValueError(
                            "node_size_legend_range must be a tuple/list "
                            "of length 2, e.g. (0, 0.12)."
                        )

                    legend_min = float(node_size_legend_range[0])
                    legend_max = float(node_size_legend_range[1])

                    if not np.isfinite(legend_min) or not np.isfinite(legend_max):
                        raise ValueError(
                            "node_size_legend_range values must be finite."
                        )

                    if legend_min > legend_max:
                        raise ValueError(
                            "node_size_legend_range must satisfy min <= max, "
                            f"got {node_size_legend_range}."
                        )

                    legend_range = (legend_min, legend_max)

                try:
                    n_size_legend = max(1, int(node_size_legend_n))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"node_size_legend_n must be a positive integer, "
                        f"got {node_size_legend_n!r}."
                    )

                if node_size_legend_values is None:
                    if legend_range is not None:
                        # When a range is explicitly supplied, legend entries
                        # come ONLY from that range. Data outside it cannot
                        # introduce extra legend labels.
                        intensity_min, intensity_max = legend_range

                        if np.isclose(intensity_min, intensity_max):
                            size_legend_values = [intensity_min]
                        else:
                            size_legend_values = np.linspace(
                                intensity_min,
                                intensity_max,
                                n_size_legend
                            ).tolist()
                    else:
                        plotted_intensities = []

                        for n in H.nodes():
                            i = node_to_index.get(n, None)
                            if i is not None and np.isfinite(mean_intensity[i]):
                                plotted_intensities.append(float(mean_intensity[i]))

                        if len(plotted_intensities) > 0:
                            intensity_min = float(np.min(plotted_intensities))
                            intensity_max = float(np.max(plotted_intensities))

                            if np.isclose(intensity_min, intensity_max):
                                size_legend_values = [intensity_min]
                            else:
                                size_legend_values = np.linspace(
                                    intensity_min,
                                    intensity_max,
                                    n_size_legend
                                ).tolist()
                        else:
                            size_legend_values = []
                else:
                    size_legend_values = [
                        float(v)
                        for v in node_size_legend_values
                        if np.isfinite(float(v))
                    ]

                    # node_size_legend_range is a HARD boundary, even when
                    # explicit legend values are provided.
                    if legend_range is not None:
                        legend_min, legend_max = legend_range
                        tol = 1e-12 * max(1.0, abs(legend_min), abs(legend_max))
                        size_legend_values = [
                            v for v in size_legend_values
                            if (legend_min - tol) <= v <= (legend_max + tol)
                        ]

                # Final defensive filter: nothing outside the requested hard
                # range is ever allowed to reach matplotlib's legend.
                if legend_range is not None and len(size_legend_values) > 0:
                    legend_min, legend_max = legend_range
                    tol = 1e-12 * max(1.0, abs(legend_min), abs(legend_max))
                    size_legend_values = [
                        v for v in size_legend_values
                        if (legend_min - tol) <= v <= (legend_max + tol)
                    ]

                if len(size_legend_values) > 0:
                    size_legend_handles = []
                    label_format = kwargs.get("node_size_legend_format", ".2f")

                    for intensity_value in size_legend_values:
                        legend_size = (
                            node_size_base
                            + node_size_multiplier * intensity_value
                        )
                        legend_size = max(float(legend_size), 1.0)

                        size_legend_handles.append(
                            ax.scatter(
                                [],
                                [],
                                s=legend_size,
                                facecolors=kwargs.get(
                                    "node_size_legend_facecolor",
                                    "lightgray"
                                ),
                                edgecolors=kwargs.get(
                                    "node_size_legend_edgecolor",
                                    node_edgecolors
                                ),
                                linewidths=kwargs.get(
                                    "node_size_legend_linewidth",
                                    node_linewidths
                                ),
                                label=format(intensity_value, label_format)
                            )
                        )

                    default_size_legend_anchor = (
                        colorbar_axes[0] + colorbar_axes[2] / 2,
                        colorbar_axes[1] + colorbar_axes[3] + 0.015
                    )

                    fig.legend(
                        handles=size_legend_handles,
                        title=node_size_legend_title,
                        loc=kwargs.get("node_size_legend_loc", "lower center"),
                        bbox_to_anchor=kwargs.get(
                            "node_size_legend_bbox_to_anchor",
                            default_size_legend_anchor
                        ),
                        ncol=kwargs.get(
                            "node_size_legend_ncol",
                            len(size_legend_handles)
                        ),
                        frameon=kwargs.get("node_size_legend_frameon", False),
                        fontsize=kwargs.get("node_size_legend_fontsize", 8),
                        title_fontsize=kwargs.get(
                            "node_size_legend_title_fontsize",
                            9
                        ),
                        handletextpad=kwargs.get(
                            "node_size_legend_handletextpad",
                            0.5
                        ),
                        columnspacing=kwargs.get(
                            "node_size_legend_columnspacing",
                            0.8
                        ),
                        borderaxespad=0.0,
                    )

        # ------------------------------------------------------------------
        # 17. Category legend
        # ------------------------------------------------------------------
        if category_color_map is not None and H.number_of_nodes() > 0:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markeredgecolor="black",
                    markersize=kwargs.get("legend_marker_size", 8),
                    linestyle="None",
                    label=str(cat_name)
                )
                for cat_name, color in category_color_map.items()
            ]

            fig.legend(
                handles=handles,
                title=metabolite_label_key if metabolite_label_key is not None else "Class",
                loc=kwargs.get("legend_loc", "center left"),
                bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (0.84, 0.5)),
                frameon=kwargs.get("legend_frameon", False),
                fontsize=kwargs.get("legend_fontsize", 9),
                title_fontsize=kwargs.get("legend_title_fontsize", 10),
                borderaxespad=kwargs.get("legend_borderaxespad", 0.0),
            )

        # ------------------------------------------------------------------
        # 17.5 Numbered metabolite legend
        # ------------------------------------------------------------------
        if len(numbered_nodes_in_H) > 0:
            numbered_handles = []

            for n in numbered_nodes_in_H:
                number = node_to_number.get(n, None)

                if number is None:
                    continue

                numbered_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="",
                        color="none",
                        linestyle="None",
                        label=f"{number}: {n}"
                    )
                )

            if len(numbered_handles) > 0:
                fig.legend(
                    handles=numbered_handles,
                    title=number_legend_title,
                    loc=number_legend_loc,
                    bbox_to_anchor=number_legend_bbox_to_anchor,
                    frameon=kwargs.get("number_legend_frameon", False),
                    fontsize=kwargs.get("number_legend_fontsize", 9),
                    title_fontsize=kwargs.get("number_legend_title_fontsize", 10),
                    borderaxespad=kwargs.get("number_legend_borderaxespad", 0.0),
                )

        # ------------------------------------------------------------------
        # 18. Save figure
        # ------------------------------------------------------------------
        network_key = selection_label

        if save_path is None:
            save_path = f"{self.path}/feature_network_{network_key}_{method}.svg"

        save_dir = os.path.dirname(save_path)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        plt.savefig(
            save_path,
            dpi=kwargs.get("dpi", 300),
            bbox_inches=kwargs.get("bbox_inches", "tight")
        )
        plt.close(fig)

        # ------------------------------------------------------------------
        # 19. Network metrics
        # ------------------------------------------------------------------
        network_metrics = {}

        network_metrics["selection_mode"] = selection_mode
        network_metrics["selection_label"] = selection_label
        network_metrics["obs_key"] = obs_key
        network_metrics["obs_range"] = obs_range
        network_metrics["target_label"] = target_label
        network_metrics["label_key"] = label_key
        network_metrics["include_left"] = kwargs.get("include_left", True)
        network_metrics["include_right"] = kwargs.get("include_right", False)

        network_metrics["colorbar_vmin"] = resolved_colorbar_vmin
        network_metrics["colorbar_vmax"] = resolved_colorbar_vmax
        network_metrics["edge_vmin"] = edge_vmin
        network_metrics["edge_vmax"] = edge_vmax

        network_metrics["n_nodes"] = H.number_of_nodes()
        network_metrics["n_edges"] = H.number_of_edges()
        network_metrics["n_samples"] = n_samples
        network_metrics["n_selected_samples"] = n_selected_samples
        network_metrics["n_features_total_in_adata"] = int(self.adata.n_vars)
        network_metrics["n_features_after_annotation_filter"] = int(n_features)
        network_metrics["n_features_removed_unannotated"] = int(
            self.adata.n_vars - len(feature_original_indices)
        )
        network_metrics["n_all_zero_features"] = int(np.sum(all_zero_feature_mask))
        network_metrics["n_zero_variance_features"] = int(np.sum(zero_variance_mask))
        network_metrics["n_features_used_for_model"] = int(len(valid_feature_indices))
        network_metrics["n_features_excluded_from_model"] = int(len(invalid_feature_indices))
        network_metrics["annotated_only"] = bool(annotated_only)
        network_metrics["min_class_size"] = min_class_size
        network_metrics["other_label"] = other_label
        network_metrics["layout_k"] = kwargs.get("layout_k", 0.9)
        network_metrics["layout_iterations"] = kwargs.get("layout_iterations", 300)
        network_metrics["layout_scale"] = kwargs.get("layout_scale", 1.0)
        network_metrics["compact_layout"] = kwargs.get("compact_layout", True)
        network_metrics["radial_compress"] = kwargs.get("radial_compress", 0.55)

        network_metrics["numbered_nodes_requested"] = numbered_nodes
        network_metrics["numbered_key"] = numbered_key
        network_metrics["numbered_exact"] = numbered_exact
        network_metrics["numbered_case_sensitive"] = numbered_case_sensitive
        network_metrics["numbered_nodes_matched"] = numbered_node_order
        network_metrics["numbered_nodes_in_graph"] = numbered_nodes_in_H
        network_metrics["n_numbered_nodes_matched"] = int(len(numbered_node_set))
        network_metrics["n_numbered_nodes_in_graph"] = int(len(numbered_nodes_in_H))
        network_metrics["node_to_number"] = node_to_number

        if class_counts_before_merging is not None:
            network_metrics["class_counts_before_merging"] = (
                class_counts_before_merging.to_dict()
            )

        if class_counts_after_merging is not None:
            network_metrics["class_counts_after_merging"] = (
                class_counts_after_merging.to_dict()
            )

        try:
            network_metrics["density"] = nx.density(H)
        except Exception:
            network_metrics["density"] = np.nan

        try:
            network_metrics["average_degree"] = np.mean([d for _, d in H.degree()])
        except Exception:
            network_metrics["average_degree"] = np.nan

        try:
            network_metrics["clustering_coefficient"] = nx.average_clustering(H)
        except Exception:
            network_metrics["clustering_coefficient"] = np.nan

        try:
            if H.number_of_nodes() > 0 and H.number_of_edges() > 0:
                communities = list(
                    nx.algorithms.community.greedy_modularity_communities(H)
                )

                modularity = nx.algorithms.community.modularity(
                    H,
                    communities
                )

                network_metrics["modularity"] = modularity
            else:
                communities = []
                network_metrics["modularity"] = np.nan

        except Exception:
            communities = []
            network_metrics["modularity"] = np.nan

        # ------------------------------------------------------------------
        # 20. Participation coefficient
        # ------------------------------------------------------------------
        participation = {}

        if class_values is not None:
            for node in H.nodes():
                neighbors = list(H.neighbors(node))
                k_i = len(neighbors)

                if k_i == 0:
                    participation[node] = 0.0
                    continue

                neighbor_class_counts = {}

                for nb in neighbors:
                    metabolite_class = node_to_class.get(nb, other_label)
                    neighbor_class_counts[metabolite_class] = (
                        neighbor_class_counts.get(metabolite_class, 0) + 1
                    )

                sum_sq = sum(
                    (v / k_i) ** 2
                    for v in neighbor_class_counts.values()
                )

                participation[node] = 1 - sum_sq

            try:
                nx.set_node_attributes(
                    H,
                    {
                        n: node_to_class.get(n, other_label)
                        for n in H.nodes()
                    },
                    "metabolite_class"
                )

                if H.number_of_edges() > 0:
                    assortativity = nx.attribute_assortativity_coefficient(
                        H,
                        "metabolite_class"
                    )
                else:
                    assortativity = np.nan

                network_metrics["class_assortativity"] = assortativity

            except Exception:
                network_metrics["class_assortativity"] = np.nan

        else:
            participation = {
                n: np.nan
                for n in H.nodes()
            }

            network_metrics["class_assortativity"] = np.nan

        # ------------------------------------------------------------------
        # 21. Output tables
        # ------------------------------------------------------------------
        degree_dict = dict(H.degree())

        bridge_df = pd.DataFrame({
            "node": list(H.nodes()),
            "degree": [
                degree_dict.get(n, 0)
                for n in H.nodes()
            ],
            "betweenness": [
                centrality["betweenness"].get(n, 0)
                for n in H.nodes()
            ],
            "pagerank": [
                centrality["pagerank"].get(n, 0)
                for n in H.nodes()
            ],
            "participation": [
                participation.get(n, np.nan)
                for n in H.nodes()
            ],
            "numbered": [
                n in numbered_node_set
                for n in H.nodes()
            ],
            "number": [
                node_to_number.get(n, np.nan)
                for n in H.nodes()
            ]
        })

        if class_values is not None:
            bridge_df["class"] = [
                node_to_class.get(n, other_label)
                for n in bridge_df["node"]
            ]

        for col in ["degree", "betweenness", "pagerank", "participation"]:
            if bridge_df.shape[0] == 0:
                bridge_df[col + "_scaled"] = []
                continue

            x = bridge_df[col].values.astype(float)
            finite_mask = np.isfinite(x)

            if not np.any(finite_mask):
                bridge_df[col + "_scaled"] = 0.0
            elif np.nanmax(x) == np.nanmin(x):
                bridge_df[col + "_scaled"] = 0.0
            else:
                bridge_df[col + "_scaled"] = (
                    (x - np.nanmin(x))
                    /
                    (np.nanmax(x) - np.nanmin(x))
                )

        if bridge_df.shape[0] > 0:
            bridge_df["bridge_score"] = (
                0.4 * bridge_df["betweenness_scaled"]
                +
                0.4 * bridge_df["participation_scaled"]
                +
                0.2 * bridge_df["degree_scaled"]
            )

            bridge_df = bridge_df.sort_values(
                "bridge_score",
                ascending=False
            )
        else:
            bridge_df["bridge_score"] = []

        node_table = pd.DataFrame({
            "node": list(H.nodes()),
            "degree": [
                degree_dict.get(n, 0)
                for n in H.nodes()
            ],
            "degree_centrality": [
                centrality["degree"].get(n, 0)
                for n in H.nodes()
            ],
            "betweenness": [
                centrality["betweenness"].get(n, 0)
                for n in H.nodes()
            ],
            "eigenvector": [
                centrality["eigenvector"].get(n, 0)
                for n in H.nodes()
            ],
            "pagerank": [
                centrality["pagerank"].get(n, 0)
                for n in H.nodes()
            ],
            "participation": [
                participation.get(n, np.nan)
                for n in H.nodes()
            ],
            "numbered": [
                n in numbered_node_set
                for n in H.nodes()
            ],
            "number": [
                node_to_number.get(n, np.nan)
                for n in H.nodes()
            ]
        })

        if class_values is not None:
            node_table["class"] = [
                node_to_class.get(n, other_label)
                for n in node_table["node"]
            ]

        edge_table_records = []

        for u, v in H.edges():
            i = node_to_index.get(u, None)
            j = node_to_index.get(v, None)

            if i is None or j is None:
                continue

            edge_table_records.append({
                "source": u,
                "target": v,
                "weight": partial_corr[i, j],
                "abs_weight": abs(partial_corr[i, j]),
                "source_numbered": u in numbered_node_set,
                "target_numbered": v in numbered_node_set,
                "source_number": node_to_number.get(u, np.nan),
                "target_number": node_to_number.get(v, np.nan)
            })

        edge_table = pd.DataFrame(edge_table_records)

        excluded_feature_table = pd.DataFrame({
            "feature_index_in_network": invalid_feature_indices,
            "feature_index_in_adata": [
                feature_original_indices[i]
                for i in invalid_feature_indices
            ],
            "feature_name": [
                index_to_node[i]
                for i in invalid_feature_indices
            ],
            "all_zero_in_selected_group": [
                bool(all_zero_feature_mask[i])
                for i in invalid_feature_indices
            ],
            "zero_variance_after_log1p": [
                bool(zero_variance_mask[i])
                for i in invalid_feature_indices
            ],
            "variance_after_log1p": [
                feature_var[i]
                for i in invalid_feature_indices
            ],
            "numbered": [
                index_to_node[i] in numbered_node_set
                for i in invalid_feature_indices
            ],
            "number": [
                node_to_number.get(index_to_node[i], np.nan)
                for i in invalid_feature_indices
            ]
        })

        # ------------------------------------------------------------------
        # 22. Save to self.adata.uns
        # ------------------------------------------------------------------
        if "network" not in self.adata.uns:
            self.adata.uns["network"] = {}

        self.adata.uns["network"][network_key] = {
            "method": method,
            "selection_mode": selection_mode,
            "selection_label": selection_label,
            "obs_key": obs_key,
            "obs_range": obs_range,
            "obs_mask": obs_mask,
            "target_label": target_label,
            "label_key": label_key,
            "name_key": metabolite_name_key,
            "class_key": metabolite_label_key,
            "annotated_only": annotated_only,
            "min_class_size": min_class_size,
            "other_label": other_label,
            "feature_original_indices": feature_original_indices,
            "var_for_network": var_for_network,
            "adjacency": A,
            "correlation_matrix": partial_corr,
            "graph": G,
            "subgraph": H,
            "centrality": centrality,
            "metrics": network_metrics,
            "communities": communities,
            "participation": participation,
            "bridge_table": bridge_df,
            "node_table": node_table,
            "edge_table": edge_table,
            "excluded_feature_table": excluded_feature_table,
            "valid_feature_indices": valid_feature_indices,
            "excluded_feature_indices": invalid_feature_indices,
            "all_zero_feature_mask": all_zero_feature_mask,
            "zero_variance_mask": zero_variance_mask,
            "layout": pos,
            "save_path": save_path,

            "numbered_nodes_requested": numbered_nodes,
            "numbered_nodes_matched": numbered_node_order,
            "numbered_nodes_in_graph": numbered_nodes_in_H,
            "numbered_match_table": numbered_match_table,
            "node_to_number": node_to_number,
        }

        return H, network_metrics, bridge_df, node_table, edge_table


    @staticmethod
    def _safe_filename(value) -> str:
        """Convert an arbitrary label into a filesystem-safe token."""
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"

    def _match_feature_indices(
        self,
        name_key: str,
        query_name: str,
        exact_match: bool = True,
        aggregate_duplicates: str = "mean",
    ) -> np.ndarray:
        """Find feature indices in self.adata.var by a display-name column."""
        if name_key not in self.adata.var.columns:
            raise KeyError(f"{name_key!r} not found in self.adata.var.")

        aggregate_duplicates = str(aggregate_duplicates).lower()
        if aggregate_duplicates not in {"mean", "median", "first", "error"}:
            raise ValueError(
                "aggregate_duplicates must be one of: "
                "'mean', 'median', 'first', 'error'."
            )

        names = self.adata.var[name_key].astype(str).str.strip()
        query = str(query_name).strip()

        if exact_match:
            mask = names == query
        else:
            mask = names.str.contains(
                re.escape(query),
                case=False,
                na=False,
                regex=True,
            )

        idx = np.where(mask.values)[0]

        if len(idx) == 0:
            preview = names.dropna().unique()[:10]
            raise ValueError(
                f"No feature matched {query_name!r} using name_key={name_key!r}. "
                f"First available names include: {list(preview)}"
            )

        if len(idx) > 1 and aggregate_duplicates == "error":
            matched_names = names.iloc[idx].tolist()
            raise ValueError(
                f"Multiple features matched {query_name!r}: {matched_names}. "
                "Use aggregate_duplicates='mean', 'median', or 'first'."
            )

        return idx.astype(int)

    def _extract_feature_vector(
        self,
        indices: Union[list, np.ndarray, pd.Index],
        aggregate_duplicates: str = "mean",
    ) -> np.ndarray:
        """Extract one observation-level vector from one or multiple feature columns."""
        aggregate_duplicates = str(aggregate_duplicates).lower()
        if aggregate_duplicates not in {"mean", "median", "first", "error"}:
            raise ValueError(
                "aggregate_duplicates must be one of: "
                "'mean', 'median', 'first', 'error'."
            )

        indices = np.asarray(indices, dtype=int).ravel()
        if len(indices) == 0:
            raise ValueError("indices is empty.")

        X = self._get_X()
        block = np.asarray(X[:, indices], dtype=float)

        if block.ndim == 1 or block.shape[1] == 1:
            return np.asarray(block).ravel().astype(float)

        if aggregate_duplicates == "error":
            raise ValueError("Multiple columns were provided while aggregate_duplicates='error'.")
        if aggregate_duplicates == "first":
            return block[:, 0].astype(float)
        if aggregate_duplicates == "median":
            return np.nanmedian(block, axis=1).astype(float)
        return np.nanmean(block, axis=1).astype(float)

    @staticmethod
    def _gaussian_kernel_regression(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_grid: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        """Nadaraya-Watson Gaussian kernel regression."""
        x_train = np.asarray(x_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        x_grid = np.asarray(x_grid, dtype=float)
        bandwidth = float(bandwidth)

        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError("bandwidth must be a finite positive number.")

        pred = np.full(len(x_grid), np.nan, dtype=float)

        for i, x0 in enumerate(x_grid):
            z = (x_train - x0) / bandwidth
            weights = np.exp(-0.5 * z ** 2)
            weight_sum = np.sum(weights)

            if weight_sum > 0 and np.isfinite(weight_sum):
                pred[i] = np.sum(weights * y_train) / weight_sum

        return pred

    def _compute_point_density(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = "kde",
        log_density: bool = True,
        bw_method=None,
        sample_size: Optional[int] = 5000,
        bins: int = 120,
        random_state: int = 0,
    ) -> np.ndarray:
        """Compute per-point 2D local density for density-colored scatter plots."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same length.")

        method = str(method).lower()
        if method not in {"kde", "hist2d"}:
            raise ValueError("density method must be either 'kde' or 'hist2d'.")

        valid = np.isfinite(x) & np.isfinite(y)
        density = np.full(x.shape[0], np.nan, dtype=float)

        if np.sum(valid) < 3:
            density[valid] = 1.0
            return density

        xv = x[valid]
        yv = y[valid]
        method_used = method

        if method_used == "kde":
            try:
                from scipy.stats import gaussian_kde

                n_valid = len(xv)
                if sample_size is not None and n_valid > int(sample_size):
                    rng = np.random.default_rng(random_state)
                    sample_idx = rng.choice(n_valid, size=int(sample_size), replace=False)
                    kde_x = xv[sample_idx]
                    kde_y = yv[sample_idx]
                else:
                    kde_x = xv
                    kde_y = yv

                kde = gaussian_kde(np.vstack([kde_x, kde_y]), bw_method=bw_method)
                density_valid = kde(np.vstack([xv, yv]))

            except Exception:
                method_used = "hist2d"

        if method_used == "hist2d":
            hist, x_edges, y_edges = np.histogram2d(xv, yv, bins=bins)
            x_bin = np.searchsorted(x_edges, xv, side="right") - 1
            y_bin = np.searchsorted(y_edges, yv, side="right") - 1
            x_bin = np.clip(x_bin, 0, hist.shape[0] - 1)
            y_bin = np.clip(y_bin, 0, hist.shape[1] - 1)
            density_valid = hist[x_bin, y_bin].astype(float)

        if log_density:
            density_valid = np.log1p(density_valid)

        density[valid] = density_valid
        return density


    @staticmethod
    def _compute_y_axis_kde_grid(
        x: np.ndarray,
        y: np.ndarray,
        x_bins: Union[int, np.ndarray] = 80,
        y_grid_size: int = 200,
        y_bandwidth: Optional[float] = None,
        x_range: Optional[tuple] = None,
        y_range: Optional[tuple] = None,
        min_count: int = 1,
        scale: str = "count",
    ):
        """
        Compute a time/window-resolved one-dimensional KDE along the y axis.

        This is different from a 2D x-y KDE. The x axis is only used to split
        cells into vertical windows. Within each x window, density is estimated
        only from the y values in that window.

        Parameters
        ----------
        x, y
            Paired observation vectors.
        x_bins
            Number of x windows, or explicit x-bin edges.
        y_grid_size
            Number of y-axis grid points.
        y_bandwidth
            Gaussian KDE bandwidth in the y-value scale. If None, use a robust
            Silverman-style bandwidth separately for each x window.
        x_range, y_range
            Optional plotting/density ranges.
        min_count
            Windows with fewer than this number of cells are left as NaN.
        scale
            "count": density integrates approximately to the number of cells in
            each x window. This is usually the best choice for single-cell
            distribution plots.
            "probability": each x window integrates approximately to 1.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        if x.size == 0:
            raise ValueError("No finite x/y values are available for y-axis KDE density.")

        if x_range is None:
            x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
        else:
            x_min, x_max = map(float, x_range)

        if y_range is None:
            y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
        else:
            y_min, y_max = map(float, y_range)

        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
            raise ValueError("Invalid x range for y-axis KDE density.")
        if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
            raise ValueError("Invalid y range for y-axis KDE density.")

        if np.isscalar(x_bins):
            x_edges = np.linspace(x_min, x_max, int(x_bins) + 1)
        else:
            x_edges = np.asarray(x_bins, dtype=float).ravel()
            if x_edges.size < 2:
                raise ValueError("x_bins must contain at least two edges.")

        y_grid = np.linspace(y_min, y_max, int(y_grid_size))
        if y_grid.size < 2:
            raise ValueError("y_grid_size must be >= 2.")

        density = np.full((y_grid.size, x_edges.size - 1), np.nan, dtype=float)
        counts = np.zeros(x_edges.size - 1, dtype=int)
        scale = str(scale).lower()
        if scale not in {"count", "probability"}:
            raise ValueError("scale must be either 'count' or 'probability'.")

        dy = float(np.nanmedian(np.diff(y_grid)))
        min_count = int(min_count) if min_count is not None else 1

        for j in range(x_edges.size - 1):
            if j == x_edges.size - 2:
                mask = (x >= x_edges[j]) & (x <= x_edges[j + 1])
            else:
                mask = (x >= x_edges[j]) & (x < x_edges[j + 1])

            yj = y[mask]
            yj = yj[np.isfinite(yj)]
            n = int(yj.size)
            counts[j] = n

            if n < min_count:
                continue

            if y_bandwidth is None:
                if n < 2:
                    bw = max((y_max - y_min) / 50.0, np.finfo(float).eps)
                else:
                    std = float(np.nanstd(yj, ddof=1))
                    iqr = float(np.nanpercentile(yj, 75) - np.nanpercentile(yj, 25))
                    sigma = min(std, iqr / 1.349) if iqr > 0 else std
                    if not np.isfinite(sigma) or sigma <= 0:
                        sigma = max((y_max - y_min) / 50.0, np.finfo(float).eps)
                    bw = 1.06 * sigma * (n ** (-1 / 5))
                    bw = max(float(bw), max((y_max - y_min) / 200.0, np.finfo(float).eps))
            else:
                bw = float(y_bandwidth)
                if bw <= 0 or not np.isfinite(bw):
                    raise ValueError("density_y_bandwidth must be a finite positive number.")

            z = (y_grid[:, None] - yj[None, :]) / bw
            kde = np.exp(-0.5 * z ** 2).sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))

            if scale == "count":
                # Approximate expected cell count per y-grid step in this x window.
                kde = kde * n * dy

            density[:, j] = kde

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        return x_edges, x_centers, y_grid, density, counts


    @staticmethod
    def _finite_quantile_range(
        values: np.ndarray,
        qrange: tuple = (0.005, 0.995),
        pad_fraction: float = 0.03,
        hard_range: Optional[tuple] = None,
    ) -> tuple:
        """Resolve a robust plotting range from finite values.

        This is mainly used for distribution-density plots, where a few extreme
        outliers can make the visible density band look artificially flattened.
        """
        values = np.asarray(values, dtype=float).ravel()
        values = values[np.isfinite(values)]

        if values.size == 0:
            raise ValueError("Cannot resolve plotting range from an empty vector.")

        if hard_range is not None:
            if len(hard_range) != 2:
                raise ValueError("hard_range must be a tuple/list of length 2.")
            lo, hi = hard_range
            lo = float(np.nanmin(values)) if lo is None else float(lo)
            hi = float(np.nanmax(values)) if hi is None else float(hi)
        else:
            if qrange is None:
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
            else:
                if len(qrange) != 2:
                    raise ValueError("qrange must be a tuple/list of length 2.")
                q0, q1 = float(qrange[0]), float(qrange[1])
                # Accept either fractions, e.g. 0.005, or percent values, e.g. 0.5.
                if q1 > 1.0:
                    q0 /= 100.0
                    q1 /= 100.0
                if not (0.0 <= q0 < q1 <= 1.0):
                    raise ValueError("qrange must satisfy 0 <= low < high <= 1.")
                lo, hi = np.nanquantile(values, [q0, q1]).astype(float)

        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = float(np.nanmin(values)), float(np.nanmax(values))

        if lo == hi:
            delta = max(abs(lo) * 0.05, 1.0)
            lo -= delta
            hi += delta

        pad = float(pad_fraction) * (hi - lo)
        if np.isfinite(pad) and pad > 0:
            lo -= pad
            hi += pad

        return float(lo), float(hi)

    @staticmethod
    def _centers_to_edges(centers: np.ndarray) -> np.ndarray:
        """Convert monotonically increasing grid centers to bin edges."""
        centers = np.asarray(centers, dtype=float).ravel()
        if centers.size < 2:
            raise ValueError("At least two centers are required to infer edges.")
        mids = 0.5 * (centers[:-1] + centers[1:])
        edges = np.empty(centers.size + 1, dtype=float)
        edges[1:-1] = mids
        edges[0] = centers[0] - (mids[0] - centers[0])
        edges[-1] = centers[-1] + (centers[-1] - mids[-1])
        return edges

    @staticmethod
    def _resolve_auto_figsize(
        x_range: tuple,
        y_range: tuple,
        density_color: bool = False,
        colorbar: bool = False,
        width: Optional[float] = None,
        min_height: float = 3.8,
        max_height: float = 6.5,
    ) -> tuple:
        """Choose a readable figure size from the visible x/y ranges.

        The data-unit aspect is square-root compressed because time and intensity
        usually have unrelated units; using the raw y_range / x_range ratio often
        makes time-course density plots look too flat.
        """
        x0, x1 = map(float, x_range)
        y0, y1 = map(float, y_range)
        xr = max(abs(x1 - x0), np.finfo(float).eps)
        yr = max(abs(y1 - y0), np.finfo(float).eps)

        if width is None:
            width = 7.2 if density_color else 6.0
            if colorbar:
                width += 0.6

        ratio = np.sqrt(yr / xr)
        height = float(width) * float(np.clip(ratio, 0.55, 0.85))
        height = float(np.clip(height, min_height, max_height))
        return (float(width), height)

    @staticmethod
    def _auto_hexbin_gridsize(
        ax,
        x: np.ndarray,
        y: np.ndarray,
        nx: int = 70,
        x_range: Optional[tuple] = None,
        y_range: Optional[tuple] = None,
    ) -> tuple:
        """Infer a near-regular hexbin (nx, ny) for the current axes.

        Matplotlib accepts either an integer gridsize (mostly controlling the
        x direction) or a tuple (nx, ny). When the axes panel is not square or
        when x/y data ranges differ strongly, a scalar gridsize often produces
        visually flattened hexagons. This helper uses the final on-screen axes
        size together with the visible data ranges to infer a better ny.
        """
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]

        nx = max(int(nx), 1)
        if x.size == 0 or y.size == 0:
            return (nx, max(1, int(round(nx / np.sqrt(3.0)))))

        if x_range is None:
            x0, x1 = float(np.nanmin(x)), float(np.nanmax(x))
        else:
            x0, x1 = map(float, x_range)

        if y_range is None:
            y0, y1 = float(np.nanmin(y)), float(np.nanmax(y))
        else:
            y0, y1 = map(float, y_range)

        xr = max(abs(x1 - x0), np.finfo(float).eps)
        yr = max(abs(y1 - y0), np.finfo(float).eps)

        fig = ax.figure
        fig.canvas.draw()
        bbox = ax.get_window_extent()
        ax_w = max(float(bbox.width), 1.0)
        ax_h = max(float(bbox.height), 1.0)

        ratio = np.sqrt(3.0) * (ax_w * yr) / (ax_h * xr)
        if not np.isfinite(ratio) or ratio <= 0:
            ny = max(1, int(round(nx / np.sqrt(3.0))))
        else:
            ny = max(1, int(round(nx / ratio)))

        return (nx, ny)

    def plot_xy_scatter(
        self,
        x,
        y,
        x_name: str = "x",
        y_name: str = "y",
        color_key: Optional[str] = None,
        color_values=None,
        color_name: Optional[str] = None,
        color_is_numeric: Optional[bool] = None,
        density_color: bool = False,
        density_method: str = "hexbin",
        density_cmap: str = "viridis",
        density_colorbar: bool = True,
        density_colorbar_label: str = "Cell count",
        density_log: bool = False,
        density_bw_method=None,
        density_sample_size: int = 5000,
        density_bins: int = 120,
        density_gridsize: Union[int, tuple] = 75,
        density_min_count: int = 1,
        density_extent: Optional[tuple] = None,
        linear_regression: bool = False,
        kernel_regression: bool = False,
        bandwidth: Optional[float] = None,
        grid_size: int = 200,
        ci: float = 95.0,
        n_bootstrap: int = 500,
        ignore_missing: bool = True,
        zero_as_missing: bool = False,
        log1p: bool = False,
        log1p_x: Optional[bool] = None,
        log1p_y: Optional[bool] = None,
        require_positive_x: bool = False,
        require_positive_y: bool = False,
        sort_by_x: bool = False,
        scatter: bool = True,
        scatter_size: float = 12,
        scatter_alpha: float = 0.35,
        line_width: float = 2.0,
        line_color: str = "black",
        kernel_line_color: str = "black",
        linear_line_color: str = "black",
        ci_alpha: float = 0.25,
        figsize=(5, 5),
        palette: str = "tab10",
        cmap: str = "viridis",
        show_colorbar: bool = True,
        show_legend: bool = True,
        legend: bool = True,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        output_file: Optional[str] = None,
        store_uns_key: Optional[str] = None,
        store_key: Optional[str] = None,
        random_state: int = 0,
        return_result: bool = False,
        **kwargs,
    ):
        """
        Unified backend for scatter plots.

        It accepts arbitrary external x/y vectors and supports:
        1) plain scatter;
        2) obs-based continuous/categorical coloring through color_key;
        3) explicit color_values;
        4) binned density coloring without drawing individual points;
        5) optional linear regression with bootstrap CI;
        6) optional Gaussian kernel regression with bootstrap CI.
        """
        x_values = np.asarray(x, dtype=float).ravel()
        y_values = np.asarray(y, dtype=float).ravel()

        if x_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError("x and y must be one-dimensional vectors.")
        if x_values.shape[0] != y_values.shape[0]:
            raise ValueError(
                f"x/y length mismatch: got len(x)={x_values.shape[0]}, "
                f"len(y)={y_values.shape[0]}."
            )
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2.")
        if n_bootstrap < 0:
            raise ValueError("n_bootstrap must be >= 0.")
        if not (0 < float(ci) < 100):
            raise ValueError("ci must be between 0 and 100.")

        if log1p_x is None:
            log1p_x = bool(log1p)
        if log1p_y is None:
            log1p_y = bool(log1p)

        if color_key is not None and color_values is not None:
            raise ValueError("Use only one of color_key or color_values, not both.")
        if density_color and (color_key is not None or color_values is not None):
            raise ValueError("density_color cannot be combined with obs/external color values in one scatter layer.")

        density_method = str(density_method).lower()
        if density_method in {"hex", "hex_bin", "hex-bin", "binned", "density"}:
            density_method = "hexbin"
        if density_method in {"kde", "point", "points", "point_kde", "xy_kde"}:
            raise ValueError(
                "x-y KDE / point-density coloring is not used here. "
                "Use density_method='hexbin' for the same logic as the reference code, "
                "or density_method='y_kde' if you explicitly want KDE(y | x-bin)."
            )
        if density_method in {"ykde", "y-axis-kde", "y_axis_kde"}:
            density_method = "y_kde"
        if density_method in {"yhist", "y_histogram", "y-axis-hist", "y_axis_hist"}:
            density_method = "y_hist"
        if density_method not in {"hexbin", "hist2d", "y_kde", "y_hist"}:
            raise ValueError(
                "When density_color=True, density_method must be 'hexbin', 'hist2d', "
                "'y_kde', or 'y_hist'. The default is 'hexbin'."
            )
        if density_min_count is not None and int(density_min_count) < 0:
            raise ValueError("density_min_count must be >= 0 or None.")

        resolved_color_name = color_name
        resolved_color_values = None
        resolved_color_is_numeric = False

        if color_key is not None:
            if color_key not in self.adata.obs.columns:
                raise KeyError(f"{color_key!r} not found in self.adata.obs.")
            if self.adata.n_obs != x_values.shape[0]:
                raise ValueError(
                    f"color_key={color_key!r} comes from self.adata.obs with length "
                    f"{self.adata.n_obs}, but x/y length is {x_values.shape[0]}."
                )
            color_series = self.adata.obs[color_key].copy()
            resolved_color_name = color_key if resolved_color_name is None else resolved_color_name
        elif color_values is not None:
            if isinstance(color_values, pd.Series):
                color_series = color_values.copy()
            else:
                color_series = pd.Series(np.asarray(color_values), name=color_name)
            if len(color_series) != x_values.shape[0]:
                raise ValueError(
                    f"color_values length mismatch: got {len(color_series)}, "
                    f"expected {x_values.shape[0]}."
                )
            resolved_color_name = "color" if resolved_color_name is None else resolved_color_name
        else:
            color_series = None

        if color_series is not None:
            if color_is_numeric is None:
                color_numeric = pd.to_numeric(color_series, errors="coerce")
                n_non_missing = int(pd.Series(color_series).notna().sum())
                n_numeric = int(color_numeric.notna().sum())
                resolved_color_is_numeric = n_non_missing > 0 and n_numeric == n_non_missing
            else:
                resolved_color_is_numeric = bool(color_is_numeric)
                color_numeric = pd.to_numeric(color_series, errors="coerce")

            if resolved_color_is_numeric:
                resolved_color_values = color_numeric.values.astype(float)
            else:
                resolved_color_values = (
                    pd.Series(color_series)
                    .astype("object")
                    .where(pd.Series(color_series).notna(), "Missing")
                    .astype(str)
                    .values
                )

        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if zero_as_missing:
            valid_mask = valid_mask & (x_values != 0) & (y_values != 0)
        if require_positive_x:
            valid_mask = valid_mask & (x_values > 0)
        if require_positive_y:
            valid_mask = valid_mask & (y_values > 0)
        if resolved_color_values is not None and resolved_color_is_numeric:
            valid_mask = valid_mask & np.isfinite(resolved_color_values)

        if ignore_missing:
            n_removed_missing = int(np.sum(~valid_mask))
            x_values = x_values[valid_mask]
            y_values = y_values[valid_mask]
            if resolved_color_values is not None:
                resolved_color_values = resolved_color_values[valid_mask]
        else:
            if not np.all(valid_mask):
                raise ValueError(
                    f"Found {int(np.sum(~valid_mask))} invalid samples. "
                    "Set ignore_missing=True to remove them."
                )
            n_removed_missing = 0

        if len(x_values) < 3:
            raise ValueError("At least 3 valid paired samples are required for scatter/regression plotting.")

        if log1p_x:
            if np.nanmin(x_values) < -1:
                raise ValueError("log1p_x=True but x contains values < -1.")
            x_values = np.log1p(x_values)
        if log1p_y:
            if np.nanmin(y_values) < -1:
                raise ValueError("log1p_y=True but y contains values < -1.")
            y_values = np.log1p(y_values)

        if sort_by_x:
            order = np.argsort(x_values)
            x_values = x_values[order]
            y_values = y_values[order]
            if resolved_color_values is not None:
                resolved_color_values = resolved_color_values[order]

        pearson_r = np.nan
        spearman_r = np.nan
        try:
            pearson_r = float(pd.Series(x_values).corr(pd.Series(y_values), method="pearson"))
        except Exception:
            pass
        try:
            spearman_r = float(pd.Series(x_values).corr(pd.Series(y_values), method="spearman"))
        except Exception:
            pass

        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        if x_min == x_max and (linear_regression or kernel_regression):
            raise ValueError("All x values are identical; cannot fit a regression/trend line.")

        # ------------------------------------------------------------------
        # Resolve visible plotting ranges before density computation.
        # For y-axis density mode, robust y-limits prevent a few extreme cells
        # from compressing the main distribution into a visually flat band.
        # ------------------------------------------------------------------
        density_auto_ylim = bool(kwargs.get("density_auto_ylim", density_color and density_method in {"y_kde", "y_hist"}))
        density_auto_xlim = bool(kwargs.get("density_auto_xlim", False))
        density_quantile_range = kwargs.get("density_quantile_range", None)
        density_y_quantile_range = kwargs.get("density_y_quantile_range", (0.005, 0.995))
        density_x_quantile_range = kwargs.get("density_x_quantile_range", (0.0, 1.0))
        density_range_pad_fraction = float(kwargs.get("density_range_pad_fraction", 0.03))

        if density_quantile_range is not None:
            density_y_quantile_range = density_quantile_range

        resolved_xlim = xlim
        resolved_ylim = ylim

        if density_auto_xlim and resolved_xlim is None:
            resolved_xlim = self._finite_quantile_range(
                x_values,
                qrange=density_x_quantile_range,
                pad_fraction=density_range_pad_fraction,
            )

        if density_auto_ylim and resolved_ylim is None:
            resolved_ylim = self._finite_quantile_range(
                y_values,
                qrange=density_y_quantile_range,
                pad_fraction=density_range_pad_fraction,
            )

        visible_x_range = resolved_xlim if resolved_xlim is not None else (x_min, x_max)
        visible_y_range = resolved_ylim if resolved_ylim is not None else (
            float(np.nanmin(y_values)),
            float(np.nanmax(y_values)),
        )

        x_grid = np.linspace(x_min, x_max, int(grid_size))
        rng = np.random.default_rng(random_state)
        alpha = (100.0 - float(ci)) / 2.0

        linear_fit = None
        if linear_regression:
            coef = np.polyfit(x_values, y_values, deg=1)
            linear_pred = np.polyval(coef, x_grid)
            linear_lower = None
            linear_upper = None
            linear_boot = None

            if n_bootstrap > 0:
                linear_boot = np.full((int(n_bootstrap), len(x_grid)), np.nan, dtype=float)
                n = len(x_values)
                for b in range(int(n_bootstrap)):
                    sample_idx = rng.integers(0, n, size=n)
                    xb = x_values[sample_idx]
                    yb = y_values[sample_idx]
                    if len(np.unique(xb[np.isfinite(xb)])) < 2:
                        continue
                    try:
                        cb = np.polyfit(xb, yb, deg=1)
                        linear_boot[b, :] = np.polyval(cb, x_grid)
                    except Exception:
                        continue
                linear_lower = np.nanpercentile(linear_boot, alpha, axis=0)
                linear_upper = np.nanpercentile(linear_boot, 100.0 - alpha, axis=0)

            linear_fit = {
                "coef": coef,
                "grid": x_grid,
                "pred": linear_pred,
                "ci_lower": linear_lower,
                "ci_upper": linear_upper,
                "boot_curves": linear_boot,
            }

        kernel_fit = None
        if kernel_regression:
            if bandwidth is None:
                n = len(x_values)
                x_std = float(np.nanstd(x_values))
                x_range = x_max - x_min
                bandwidth = 1.06 * x_std * (n ** (-1 / 5))
                if not np.isfinite(bandwidth) or bandwidth <= 0:
                    bandwidth = 0.15 * x_range
                min_bandwidth = kwargs.get("min_bandwidth_fraction", 0.03) * x_range
                bandwidth = max(float(bandwidth), float(min_bandwidth))

            bandwidth = float(bandwidth)
            if bandwidth <= 0 or not np.isfinite(bandwidth):
                raise ValueError("bandwidth must be a finite positive number.")

            kernel_pred = self._gaussian_kernel_regression(x_values, y_values, x_grid, bandwidth)
            kernel_lower = None
            kernel_upper = None
            kernel_boot = None

            if n_bootstrap > 0:
                kernel_boot = np.full((int(n_bootstrap), len(x_grid)), np.nan, dtype=float)
                n = len(x_values)
                for b in range(int(n_bootstrap)):
                    sample_idx = rng.integers(0, n, size=n)
                    xb = x_values[sample_idx]
                    yb = y_values[sample_idx]
                    valid = np.isfinite(xb) & np.isfinite(yb)
                    if np.sum(valid) < 3:
                        continue
                    kernel_boot[b, :] = self._gaussian_kernel_regression(
                        xb[valid], yb[valid], x_grid, bandwidth
                    )
                kernel_lower = np.nanpercentile(kernel_boot, alpha, axis=0)
                kernel_upper = np.nanpercentile(kernel_boot, 100.0 - alpha, axis=0)

            kernel_fit = {
                "bandwidth": bandwidth,
                "grid": x_grid,
                "pred": kernel_pred,
                "ci_lower": kernel_lower,
                "ci_upper": kernel_upper,
                "boot_curves": kernel_boot,
            }

        # Density mode draws an aggregate distribution layer only: no individual points are drawn.
        # Preferred mode is y_kde: x windows + one-dimensional KDE along the y axis.
        point_density = None
        scatter_order = np.arange(len(x_values))

        if isinstance(figsize, str):
            if figsize.lower() != "auto":
                raise ValueError("figsize must be a tuple like (6, 4) or the string 'auto'.")
            resolved_figsize = self._resolve_auto_figsize(
                x_range=visible_x_range,
                y_range=visible_y_range,
                density_color=bool(density_color),
                colorbar=bool((density_color and density_colorbar) or (resolved_color_values is not None and resolved_color_is_numeric and show_colorbar)),
                width=kwargs.get("auto_fig_width", None),
                min_height=kwargs.get("auto_fig_min_height", 3.8),
                max_height=kwargs.get("auto_fig_max_height", 6.5),
            )
        else:
            resolved_figsize = figsize

        fig, ax = plt.subplots(figsize=resolved_figsize)

        if resolved_xlim is not None:
            ax.set_xlim(resolved_xlim[0], resolved_xlim[1])
        if resolved_ylim is not None:
            ax.set_ylim(resolved_ylim[0], resolved_ylim[1])

        if scatter:
            if density_color:
                if density_method in {"y_kde", "y_hist"}:
                    # Preferred density mode for time-dependent distribution plots:
                    # x only defines vertical windows; density is computed along y within each x window.
                    import matplotlib.colors as mcolors

                    density_x_bins = kwargs.get("density_x_bins", density_gridsize)
                    density_y_grid_size = kwargs.get("density_y_grid_size", density_bins)
                    density_y_bandwidth = kwargs.get("density_y_bandwidth", kwargs.get("y_bandwidth", None))
                    density_y_scale = kwargs.get("density_y_scale", "count")

                    x_range_for_density = None
                    if density_extent is not None:
                        if len(density_extent) != 4:
                            raise ValueError("density_extent must be (xmin, xmax, ymin, ymax).")
                        x_range_for_density = (density_extent[0], density_extent[1])
                        y_range_for_density = (density_extent[2], density_extent[3])
                    else:
                        y_range_for_density = resolved_ylim
                        x_range_for_density = resolved_xlim

                    if density_method == "y_kde":
                        x_edges, x_centers, y_grid, D, x_counts = self._compute_y_axis_kde_grid(
                            x=x_values,
                            y=y_values,
                            x_bins=density_x_bins,
                            y_grid_size=density_y_grid_size,
                            y_bandwidth=density_y_bandwidth,
                            x_range=x_range_for_density,
                            y_range=y_range_for_density,
                            min_count=density_min_count,
                            scale=density_y_scale,
                        )
                        density_plot_label = kwargs.get(
                            "density_colorbar_label",
                            density_colorbar_label if density_colorbar_label is not None else "Y-axis density",
                        )
                    else:
                        # Histogram alternative: also only bins y within each x window.
                        x_edges = (
                            np.linspace(float(np.nanmin(x_values)), float(np.nanmax(x_values)), int(density_x_bins) + 1)
                            if np.isscalar(density_x_bins)
                            else np.asarray(density_x_bins, dtype=float)
                        )
                        y_min = float(np.nanmin(y_values)) if resolved_ylim is None else float(resolved_ylim[0])
                        y_max = float(np.nanmax(y_values)) if resolved_ylim is None else float(resolved_ylim[1])
                        y_edges = np.linspace(y_min, y_max, int(density_y_grid_size) + 1)
                        H, _, _ = np.histogram2d(x_values, y_values, bins=[x_edges, y_edges])
                        D = H.T.astype(float)
                        D[D < int(density_min_count)] = np.nan
                        y_grid = 0.5 * (y_edges[:-1] + y_edges[1:])
                        x_counts = np.nansum(H, axis=1).astype(int)
                        density_plot_label = kwargs.get("density_colorbar_label", density_colorbar_label)

                    # Convert y centers to edges so the density layer becomes a continuous
                    # vertical distribution field rather than center-like cells.
                    y_edges_for_mesh = self._centers_to_edges(y_grid)

                    if kwargs.get("density_mask_zeros", True):
                        D = np.where(np.isfinite(D) & (D > 0), D, np.nan)

                    if density_log:
                        positive = D[np.isfinite(D) & (D > 0)]
                        norm = mcolors.LogNorm(vmin=np.nanmin(positive), vmax=np.nanmax(positive)) if positive.size > 0 else None
                    else:
                        finite_density = D[np.isfinite(D)]
                        vmin = kwargs.get("density_vmin", None)
                        vmax = kwargs.get("density_vmax", None)
                        if finite_density.size > 0:
                            if vmin is None:
                                vmin = float(np.nanmin(finite_density))
                            if vmax is None:
                                density_vmax_quantile = kwargs.get("density_vmax_quantile", None)
                                if density_vmax_quantile is not None:
                                    q = float(density_vmax_quantile)
                                    if q > 1.0:
                                        q = q / 100.0
                                    vmax = float(np.nanquantile(finite_density, q))
                                else:
                                    vmax = float(np.nanmax(finite_density))
                        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if finite_density.size > 0 else None

                    mesh = ax.pcolormesh(
                        x_edges,
                        y_edges_for_mesh,
                        D,
                        cmap=density_cmap,
                        norm=norm,
                        shading=kwargs.get("density_shading", "auto"),
                        alpha=kwargs.get("density_alpha", 1.0),
                        rasterized=kwargs.get("rasterized", True),
                        zorder=2,
                    )

                    if density_colorbar:
                        cbar = fig.colorbar(
                            mesh,
                            ax=ax,
                            pad=kwargs.get("density_colorbar_pad", 0.02),
                            fraction=kwargs.get("density_colorbar_fraction", 0.046),
                        )
                        cbar.set_label(density_plot_label)

                elif density_method == "hexbin":
                    # Reference-code logic: direct matplotlib hexbin layer.
                    # No point-wise KDE and no automatic (nx, ny) distortion correction.
                    # This preserves the visual behavior of:
                    # ax.hexbin(x, y, gridsize=75, mincnt=1, cmap="viridis")
                    hb = ax.hexbin(
                        x_values,
                        y_values,
                        gridsize=density_gridsize,
                        mincnt=density_min_count,
                        cmap=density_cmap,
                        extent=density_extent,
                        bins=("log" if density_log else None),
                        linewidths=kwargs.get("density_linewidths", 0.0),
                        edgecolors=kwargs.get("density_edgecolors", "none"),
                        alpha=kwargs.get("density_alpha", 1.0),
                        rasterized=kwargs.get("rasterized", True),
                        zorder=2,
                    )
                    if density_colorbar:
                        cb = fig.colorbar(hb, ax=ax)
                        cb.set_label(density_colorbar_label)

                elif density_method == "hist2d":
                    import matplotlib.colors as mcolors

                    norm = None
                    if density_log:
                        norm = mcolors.LogNorm()

                    hist = ax.hist2d(
                        x_values,
                        y_values,
                        bins=density_bins,
                        range=kwargs.get("density_range", None),
                        cmap=density_cmap,
                        norm=norm,
                        cmin=density_min_count,
                        alpha=kwargs.get("density_alpha", 1.0),
                        rasterized=kwargs.get("rasterized", True),
                        zorder=2,
                    )
                    if density_colorbar:
                        cbar = fig.colorbar(
                            hist[3],
                            ax=ax,
                            pad=kwargs.get("density_colorbar_pad", 0.02),
                            fraction=kwargs.get("density_colorbar_fraction", 0.046),
                        )
                        cbar.set_label(density_colorbar_label)

            elif resolved_color_values is None:
                ax.scatter(
                    x_values,
                    y_values,
                    s=scatter_size,
                    alpha=scatter_alpha,
                    linewidths=0,
                    rasterized=kwargs.get("rasterized", True),
                    label=kwargs.get("scatter_label", "Observed"),
                    zorder=2,
                )

            elif resolved_color_is_numeric:
                sc = ax.scatter(
                    x_values,
                    y_values,
                    c=resolved_color_values.astype(float),
                    cmap=cmap,
                    s=scatter_size,
                    alpha=scatter_alpha,
                    linewidths=0,
                    rasterized=kwargs.get("rasterized", True),
                    zorder=2,
                )
                if show_colorbar:
                    cbar = fig.colorbar(sc, ax=ax)
                    cbar.solids.set_alpha(1)
                    cbar.ax.set_alpha(1)
                    cbar.set_label(resolved_color_name)

            else:
                plot_df = pd.DataFrame({
                    "x": x_values,
                    "y": y_values,
                    "color": resolved_color_values,
                })
                sns.scatterplot(
                    data=plot_df,
                    x="x",
                    y="y",
                    hue="color",
                    palette=palette,
                    s=scatter_size,
                    alpha=scatter_alpha,
                    linewidth=0,
                    ax=ax,
                    legend=show_legend,
                    zorder=2,
                )
                if show_legend and ax.get_legend() is not None:
                    ax.legend(
                        title=resolved_color_name,
                        bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (1.05, 1)),
                        loc=kwargs.get("legend_loc", "upper left"),
                        frameon=kwargs.get("legend_frameon", False),
                    )

        if linear_fit is not None:
            if linear_fit["ci_lower"] is not None and linear_fit["ci_upper"] is not None:
                ax.fill_between(
                    linear_fit["grid"],
                    linear_fit["ci_lower"],
                    linear_fit["ci_upper"],
                    alpha=ci_alpha,
                    linewidth=0,
                    label=kwargs.get("linear_ci_label", f"Linear {ci:g}% CI"),
                    zorder=3,
                )
            ax.plot(
                linear_fit["grid"],
                linear_fit["pred"],
                color=kwargs.get("linear_line_color", linear_line_color or line_color),
                linewidth=line_width,
                label=kwargs.get("linear_label", "Linear regression"),
                zorder=4,
            )

        if kernel_fit is not None:
            if kernel_fit["ci_lower"] is not None and kernel_fit["ci_upper"] is not None:
                ax.fill_between(
                    kernel_fit["grid"],
                    kernel_fit["ci_lower"],
                    kernel_fit["ci_upper"],
                    alpha=ci_alpha,
                    linewidth=0,
                    label=kwargs.get("kernel_ci_label", f"Kernel {ci:g}% CI"),
                    zorder=3,
                )
            ax.plot(
                kernel_fit["grid"],
                kernel_fit["pred"],
                color=kwargs.get("kernel_line_color", kernel_line_color),
                linewidth=line_width,
                label=kwargs.get("kernel_label", "Kernel regression"),
                zorder=5,
            )

        x_label = xlabel if xlabel is not None else str(x_name)
        y_label = ylabel if ylabel is not None else str(y_name)
        if log1p_x:
            x_label = f"log1p({x_label})"
        if log1p_y:
            y_label = f"log1p({y_label})"

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        if resolved_xlim is not None:
            if not isinstance(resolved_xlim, (tuple, list)) or len(resolved_xlim) != 2:
                raise ValueError("xlim must be a tuple/list of length 2, e.g. xlim=(0, 10).")
            if resolved_xlim[0] is not None and resolved_xlim[1] is not None and resolved_xlim[0] >= resolved_xlim[1]:
                raise ValueError(f"xlim must satisfy min < max, got xlim={resolved_xlim}.")
            ax.set_xlim(resolved_xlim[0], resolved_xlim[1])

        if resolved_ylim is not None:
            if not isinstance(resolved_ylim, (tuple, list)) or len(resolved_ylim) != 2:
                raise ValueError("ylim must be a tuple/list of length 2, e.g. ylim=(0, 10).")
            if resolved_ylim[0] is not None and resolved_ylim[1] is not None and resolved_ylim[0] >= resolved_ylim[1]:
                raise ValueError(f"ylim must satisfy min < max, got ylim={resolved_ylim}.")
            ax.set_ylim(resolved_ylim[0], resolved_ylim[1])

        if title is None:
            title = (
                f"{x_name} vs {y_name}\n"
                f"Pearson r={pearson_r:.3g}, Spearman r={spearman_r:.3g}, n={len(x_values)}"
            )
        ax.set_title(title)

        if legend and (linear_fit is not None or kernel_fit is not None or (scatter and resolved_color_values is None)):
            handles, labels = ax.get_legend_handles_labels()
            if len(labels) > 0:
                ax.legend(
                    frameon=kwargs.get("legend_frameon", False),
                    fontsize=kwargs.get("legend_fontsize", 9),
                    loc=kwargs.get("legend_loc", "best"),
                )

        if kwargs.get("despine", False):
            sns.despine(ax=ax)

        fig.tight_layout()

        if output_file is not None:
            save_dir = os.path.dirname(output_file)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            fig.savefig(
                output_file,
                dpi=kwargs.get("dpi", 300),
                bbox_inches=kwargs.get("bbox_inches", "tight"),
            )

        plt.close(fig)

        result = {
            "x_name": x_name,
            "y_name": y_name,
            "x_values_used": x_values,
            "y_values_used": y_values,
            "n_valid_samples": int(len(x_values)),
            "n_removed_missing": int(n_removed_missing),
            "ignore_missing": bool(ignore_missing),
            "zero_as_missing": bool(zero_as_missing),
            "log1p_x": bool(log1p_x),
            "log1p_y": bool(log1p_y),
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "density_color": bool(density_color),
            "density_method": density_method,
            "density_bins": density_bins,
            "density_gridsize": density_gridsize,
            "density_min_count": density_min_count,
            "density_extent": density_extent,
            "point_density": point_density,
            "color_key": color_key,
            "color_name": resolved_color_name,
            "color_is_numeric": resolved_color_is_numeric if resolved_color_values is not None else None,
            "linear_fit": linear_fit,
            "kernel_fit": kernel_fit,
            "ci": float(ci),
            "n_bootstrap": int(n_bootstrap),
            "output_file": output_file,
            "figsize": resolved_figsize,
        }

        if store_uns_key is not None:
            if store_uns_key not in self.adata.uns:
                self.adata.uns[store_uns_key] = {}
            resolved_store_key = store_key or f"{x_name}|{y_name}"
            self.adata.uns[store_uns_key][resolved_store_key] = result

        if return_result:
            return result
        return self

    def plot_single_metabolite_trend(
        self,
        name_key: Optional[str] = None,
        metabolite_name: Optional[str] = None,
        time_key: str = "time",
        bandwidth: Optional[float] = None,
        grid_size: int = 200,
        n_bootstrap: int = 500,
        ci: float = 95.0,
        ignore_missing: bool = True,
        aggregate_duplicates: str = "mean",
        exact_match: bool = True,
        log1p: bool = False,
        scatter: bool = True,
        scatter_alpha: float = 0.35,
        scatter_size: float = 12,
        line_width: float = 2.0,
        ci_alpha: float = 0.25,
        figsize=(5, 5),
        random_state: int = 0,
        output_file: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        physical_values=None,
        physical_key: Optional[str] = None,
        physical_name: Optional[str] = None,
        density_color: bool = False,
        density_method: str = "hexbin",
        density_cmap: str = "viridis",
        density_colorbar: bool = True,
        density_colorbar_label: str = "Cell count",
        density_log: bool = False,
        density_bw_method=None,
        density_sample_size: int = 5000,
        density_bins: int = 120,
        density_gridsize: Union[int, tuple] = 75,
        density_min_count: int = 1,
        density_extent: Optional[tuple] = None,
        linear_regression: bool = False,
        kernel_regression: bool = True,
        **kwargs,
    ):
        """Plot one metabolite/physical quantity against an obs time-like axis."""
        if time_key not in self.adata.obs.columns:
            raise KeyError(f"{time_key!r} not found in self.adata.obs.")

        using_physical_values = physical_values is not None
        using_physical_key = physical_key is not None
        if using_physical_values and using_physical_key:
            raise ValueError("Use only one of physical_values or physical_key, not both.")

        x_values = pd.to_numeric(self.adata.obs[time_key], errors="coerce").values.astype(float)
        matched_idx = np.array([], dtype=int)
        matched_var_names = []
        y_source = None

        if using_physical_values:
            y_values = np.asarray(physical_values, dtype=float).ravel()
            if y_values.shape[0] != self.adata.n_obs:
                raise ValueError(
                    f"physical_values length mismatch: got {y_values.shape[0]}, "
                    f"expected self.adata.n_obs={self.adata.n_obs}."
                )
            y_source = "physical_values"
            if physical_name is None:
                physical_name = kwargs.get("physical_values_name", "Physical quantity")

        elif using_physical_key:
            if physical_key not in self.adata.obs.columns:
                raise KeyError(f"{physical_key!r} not found in self.adata.obs.")
            y_values = pd.to_numeric(self.adata.obs[physical_key], errors="coerce").values.astype(float)
            y_source = f"adata.obs[{physical_key!r}]"
            if physical_name is None:
                physical_name = physical_key

        else:
            if name_key is None:
                raise ValueError("name_key is required when physical_values and physical_key are both None.")
            if metabolite_name is None:
                raise ValueError("metabolite_name is required when physical_values and physical_key are both None.")
            matched_idx = self._match_feature_indices(
                name_key=name_key,
                query_name=metabolite_name,
                exact_match=exact_match,
                aggregate_duplicates=aggregate_duplicates,
            )
            y_values = self._extract_feature_vector(matched_idx, aggregate_duplicates=aggregate_duplicates)
            matched_var_names = self.adata.var_names[matched_idx].tolist()
            y_source = "adata.X"
            if physical_name is None:
                physical_name = str(metabolite_name)

        if output_file is None:
            safe_name = self._safe_filename(physical_name)
            output_file = f"{self.path}/metabolite_trend_{safe_name}_{self._safe_filename(time_key)}.svg"

        if xlabel is None:
            xlabel = time_key
        if ylabel is None:
            ylabel = str(physical_name)

        if title is None:
            title = f"{physical_name} trend over {time_key}"

        result = self.plot_xy_scatter(
            x=x_values,
            y=y_values,
            x_name=time_key,
            y_name=str(physical_name),
            density_color=density_color,
            density_method=density_method,
            density_cmap=density_cmap,
            density_colorbar=density_colorbar,
            density_colorbar_label=density_colorbar_label,
            density_log=density_log,
            density_bw_method=density_bw_method,
            density_sample_size=density_sample_size,
            density_bins=density_bins,
            density_gridsize=density_gridsize,
            density_min_count=density_min_count,
            density_extent=density_extent,
            linear_regression=linear_regression,
            kernel_regression=kernel_regression,
            bandwidth=bandwidth,
            grid_size=grid_size,
            ci=ci,
            n_bootstrap=n_bootstrap,
            ignore_missing=ignore_missing,
            log1p_y=log1p,
            scatter=scatter,
            scatter_alpha=scatter_alpha,
            scatter_size=scatter_size,
            line_width=line_width,
            ci_alpha=ci_alpha,
            figsize=figsize,
            random_state=random_state,
            output_file=output_file,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            sort_by_x=True,
            store_uns_key=None,
            return_result=True,
            **kwargs,
        )

        if "single_metabolite_trends" not in self.adata.uns:
            self.adata.uns["single_metabolite_trends"] = {}

        if using_physical_values or using_physical_key:
            default_store_key = f"{physical_name}|{time_key}|external"
        else:
            default_store_key = f"{metabolite_name}|{time_key}|{name_key}"

        result_key = kwargs.get("store_key", default_store_key)
        result.update({
            "name_key": name_key,
            "metabolite_name": metabolite_name,
            "time_key": time_key,
            "matched_indices": matched_idx,
            "matched_var_names": matched_var_names,
            "aggregate_duplicates": aggregate_duplicates,
            "exact_match": exact_match,
            "y_source": y_source,
            "using_external_physical_quantity": bool(using_physical_values or using_physical_key),
            "physical_key": physical_key,
            "physical_name": physical_name,
            "bandwidth": None if result.get("kernel_fit") is None else result["kernel_fit"].get("bandwidth"),
            "grid": None if result.get("kernel_fit") is None else result["kernel_fit"].get("grid"),
            "trend": None if result.get("kernel_fit") is None else result["kernel_fit"].get("pred"),
            "ci_lower": None if result.get("kernel_fit") is None else result["kernel_fit"].get("ci_lower"),
            "ci_upper": None if result.get("kernel_fit") is None else result["kernel_fit"].get("ci_upper"),
        })
        self.adata.uns["single_metabolite_trends"][result_key] = result
        return self

    def plot_feature_pair_scatter(
        self,
        name_key: str,
        feature_a: str,
        feature_b: str,
        aggregate_duplicates: str = "mean",
        exact_match: bool = True,
        ignore_missing: bool = True,
        zero_as_missing: bool = True,
        log1p: bool = False,
        ci: int = 95,
        n_boot: int = 1000,
        figsize=(5, 5),
        scatter_size: float = 12,
        scatter_alpha: float = 0.35,
        line_color: str = "black",
        color_key: Optional[str] = None,
        palette: str = "tab10",
        cmap: str = "viridis",
        show_colorbar: bool = True,
        show_legend: bool = True,
        output_file: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xlim: Optional[tuple] = None,
        ylim: Optional[tuple] = None,
        regression: bool = True,
        kernel_regression: bool = False,
        density_color: bool = False,
        density_method: str = "hexbin",
        density_cmap: str = "viridis",
        density_colorbar: bool = True,
        density_colorbar_label: str = "Local density",
        random_state: int = 0,
        **kwargs,
    ):
        """Plot scatter/correlation between two named metabolite features."""
        idx_a = self._match_feature_indices(
            name_key=name_key,
            query_name=feature_a,
            exact_match=exact_match,
            aggregate_duplicates=aggregate_duplicates,
        )
        idx_b = self._match_feature_indices(
            name_key=name_key,
            query_name=feature_b,
            exact_match=exact_match,
            aggregate_duplicates=aggregate_duplicates,
        )

        x_values = self._extract_feature_vector(idx_a, aggregate_duplicates=aggregate_duplicates)
        y_values = self._extract_feature_vector(idx_b, aggregate_duplicates=aggregate_duplicates)

        if output_file is None:
            safe_a = self._safe_filename(feature_a)
            safe_b = self._safe_filename(feature_b)
            if color_key is None and not density_color:
                output_file = f"{self.path}/feature_pair_scatter_{safe_a}_vs_{safe_b}.svg"
            elif density_color:
                output_file = f"{self.path}/feature_pair_scatter_{safe_a}_vs_{safe_b}_by_density.svg"
            else:
                safe_color = self._safe_filename(color_key)
                output_file = f"{self.path}/feature_pair_scatter_{safe_a}_vs_{safe_b}_by_{safe_color}.svg"

        result = self.plot_xy_scatter(
            x=x_values,
            y=y_values,
            x_name=str(feature_a),
            y_name=str(feature_b),
            color_key=color_key,
            density_color=density_color,
            density_method=density_method,
            density_cmap=density_cmap,
            density_colorbar=density_colorbar,
            density_colorbar_label=density_colorbar_label,
            linear_regression=regression,
            kernel_regression=kernel_regression,
            ci=ci,
            n_bootstrap=n_boot,
            ignore_missing=ignore_missing,
            zero_as_missing=zero_as_missing,
            log1p=log1p,
            scatter_size=scatter_size,
            scatter_alpha=scatter_alpha,
            line_color=line_color,
            linear_line_color=line_color,
            figsize=figsize,
            palette=palette,
            cmap=cmap,
            show_colorbar=show_colorbar,
            show_legend=show_legend,
            output_file=output_file,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
            random_state=random_state,
            store_uns_key=None,
            return_result=True,
            **kwargs,
        )

        if "feature_pair_scatter" not in self.adata.uns:
            self.adata.uns["feature_pair_scatter"] = {}

        store_key = kwargs.get(
            "store_key",
            (
                f"{feature_a}|{feature_b}|{name_key}"
                if color_key is None
                else f"{feature_a}|{feature_b}|{name_key}|color={color_key}"
            ),
        )

        result.update({
            "name_key": name_key,
            "feature_a": feature_a,
            "feature_b": feature_b,
            "matched_indices_a": idx_a,
            "matched_indices_b": idx_b,
            "aggregate_duplicates": aggregate_duplicates,
            "exact_match": exact_match,
        })
        self.adata.uns["feature_pair_scatter"][store_key] = result
        return self
    
    @staticmethod
    def _build_piecewise_axis_compressor(
        ylim: tuple = (0.0, 1.0),
        compress_regions: Optional[list] = None,
    ):
        """
        Build a piecewise-linear y-axis compression function.

        Parameters
        ----------
        ylim
            Original y-axis range, usually (0, 1) for composition plots.

        compress_regions
            List of tuples: [(lo, hi, factor), ...].

            Example:
                [(0.0, 0.55, 0.25)]

            This means the original y range 0.0–0.55 is visually compressed
            to 25% of its original height. Regions not listed are kept at factor=1.

        Returns
        -------
        transform
            Function mapping original y values to compressed y values.

        inverse
            Function mapping compressed y values back to original y values.

        transformed_ylim
            y-limit in compressed coordinates.

        segments
            Internal segment table:
            [(orig_lo, orig_hi, factor, transformed_lo, transformed_hi), ...]
        """
        y0, y1 = map(float, ylim)

        if y0 >= y1:
            raise ValueError(f"ylim must satisfy low < high, got {ylim}.")

        if compress_regions is None or len(compress_regions) == 0:
            def identity(values):
                return np.asarray(values, dtype=float)

            return identity, identity, (y0, y1), [(y0, y1, 1.0, y0, y1)]

        regions = []

        for region in compress_regions:
            if len(region) != 3:
                raise ValueError(
                    "Each compression region must be a tuple/list: "
                    "(lo, hi, factor)."
                )

            lo, hi, factor = map(float, region)

            if lo < y0 or hi > y1:
                raise ValueError(
                    f"Compression region {(lo, hi)} is outside ylim={ylim}."
                )

            if lo >= hi:
                raise ValueError(
                    f"Compression region must satisfy lo < hi, got {(lo, hi)}."
                )

            if factor <= 0 or not np.isfinite(factor):
                raise ValueError(
                    f"Compression factor must be a finite positive value, got {factor}."
                )

            regions.append((lo, hi, factor))

        regions = sorted(regions, key=lambda x: x[0])

        for i in range(1, len(regions)):
            if regions[i][0] < regions[i - 1][1]:
                raise ValueError(
                    f"Compression regions overlap: {regions[i - 1]} and {regions[i]}."
                )

        segments_raw = []
        cursor = y0

        for lo, hi, factor in regions:
            if lo > cursor:
                segments_raw.append((cursor, lo, 1.0))

            segments_raw.append((lo, hi, factor))
            cursor = hi

        if cursor < y1:
            segments_raw.append((cursor, y1, 1.0))

        segments = []
        transformed_cursor = y0

        for lo, hi, factor in segments_raw:
            transformed_lo = transformed_cursor
            transformed_hi = transformed_lo + (hi - lo) * factor

            segments.append(
                (
                    float(lo),
                    float(hi),
                    float(factor),
                    float(transformed_lo),
                    float(transformed_hi),
                )
            )

            transformed_cursor = transformed_hi

        transformed_ylim = (segments[0][3], segments[-1][4])

        def transform(values):
            values = np.asarray(values, dtype=float)
            out = np.full_like(values, np.nan, dtype=float)

            for lo, hi, factor, tlo, thi in segments:
                mask = (values >= lo) & (values <= hi)
                out[mask] = tlo + (values[mask] - lo) * factor

            below = values < y0
            above = values > y1

            if np.any(below):
                lo, hi, factor, tlo, thi = segments[0]
                out[below] = tlo + (values[below] - lo) * factor

            if np.any(above):
                lo, hi, factor, tlo, thi = segments[-1]
                out[above] = tlo + (values[above] - lo) * factor

            return out

        def inverse(values):
            values = np.asarray(values, dtype=float)
            out = np.full_like(values, np.nan, dtype=float)

            for lo, hi, factor, tlo, thi in segments:
                mask = (values >= tlo) & (values <= thi)
                out[mask] = lo + (values[mask] - tlo) / factor

            below = values < transformed_ylim[0]
            above = values > transformed_ylim[1]

            if np.any(below):
                lo, hi, factor, tlo, thi = segments[0]
                out[below] = lo + (values[below] - tlo) / factor

            if np.any(above):
                lo, hi, factor, tlo, thi = segments[-1]
                out[above] = lo + (values[above] - tlo) / factor

            return out

        return transform, inverse, transformed_ylim, segments

    def plot_class_composition_over_time(
        self,
        time_key: str = "time",
        class_key: str = "lipid_class",
        bin_width: Optional[float] = None,
        n_bins: Optional[int] = 72,
        time_range: Optional[tuple] = None,
        normalize_per_cell: bool = False,
        total_sum: float = 1e6,
        log1p: bool = False,
        clip_negative_to_zero: bool = True,
        min_class_total: float = 0.0,
        top_n_classes: Optional[int] = None,
        include_classes: Optional[list] = None,
        exclude_classes: Optional[list] = None,
        renormalize_excluding_classes: Optional[list] = None,

        # ------------------------------------------------------------------
        # New: merge classes with too few features into Other
        # ------------------------------------------------------------------
        min_features_per_class: Optional[int] = None,
        small_class_label: str = "Other",

        unknown_labels: tuple = (
            "",
            "nan",
            "none",
            "na",
            "n/a",
            "unknown",
            "unannotated",
        ),
        other_label: Optional[str] = None,
        order_by: str = "total_intensity",
        descending: bool = True,
        colors: Optional[Union[list, dict]] = None,
        cmap: str = "tab20",
        event_time: Optional[float] = None,
        event_color: str = "black",
        event_linestyle: str = "--",
        event_linewidth: float = 1.2,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: str = "Relative class composition",
        figsize=(8, 5),
        legend: bool = True,
        legend_mode: str = "top",
        legend_ncol: Optional[int] = None,
        legend_fontsize: float = 9,
        legend_title: Optional[str] = None,
        legend_frameon: bool = False,
        ylim: tuple = (0.0, 1.0),

        # ------------------------------------------------------------------
        # Optional visual y-axis compression
        # ------------------------------------------------------------------
        y_compress_regions: Optional[list] = None,
        y_compress_tick_values: Optional[list] = None,
        y_compress_show_boundaries: bool = True,
        y_compress_boundary_color: str = "gray",
        y_compress_boundary_linestyle: str = ":",
        y_compress_boundary_linewidth: float = 0.8,

        output_file: Optional[str] = None,
        output_prefix: Optional[str] = None,
        store_uns_key: str = "class_composition_over_time",
        return_result: bool = False,
        **kwargs,
    ):
        """
        Plot a stacked area chart showing time-resolved class composition.

        Classes are ordered from bottom to top by total class intensity.
        By default, the highest-abundance class is drawn at the bottom,
        followed upward by progressively lower-abundance classes.

        New option
        ----------
        min_features_per_class
            If not None, classes containing fewer than this number of features are
            merged into small_class_label, usually "Other".

            Example:
                min_features_per_class=3

            means classes represented by only 1 or 2 features will be merged into
            "Other" before top-N selection and plotting.

        small_class_label
            Label used for classes merged because of min_features_per_class.
        """

        # ------------------------------------------------------------------
        # 1. Check inputs
        # ------------------------------------------------------------------
        if time_key not in self.adata.obs.columns:
            raise KeyError(f"{time_key!r} not found in self.adata.obs.")

        if class_key not in self.adata.var.columns:
            raise KeyError(f"{class_key!r} not found in self.adata.var.")

        if bin_width is None and n_bins is None:
            raise ValueError("Provide either bin_width or n_bins.")

        if bin_width is not None and float(bin_width) <= 0:
            raise ValueError("bin_width must be positive.")

        if n_bins is not None and int(n_bins) < 1:
            raise ValueError("n_bins must be >= 1.")

        if min_features_per_class is not None:
            min_features_per_class = int(min_features_per_class)
            if min_features_per_class < 1:
                raise ValueError("min_features_per_class must be >= 1 or None.")

        order_by = str(order_by).lower()

        if order_by not in {"total_intensity", "alphabetical"}:
            raise ValueError("order_by must be 'total_intensity' or 'alphabetical'.")

        legend_mode = str(legend_mode).lower()

        if legend_mode not in {"top", "right", "bottom", "inside"}:
            raise ValueError("legend_mode must be 'top', 'right', 'bottom', or 'inside'.")

        if ylim is None:
            ylim = (0.0, 1.0)

        if len(ylim) != 2:
            raise ValueError("ylim must be a tuple/list of length 2.")

        ylim = (float(ylim[0]), float(ylim[1]))

        if ylim[0] >= ylim[1]:
            raise ValueError(f"ylim must satisfy low < high, got {ylim}.")

        # ------------------------------------------------------------------
        # 2. Extract and preprocess matrix
        # ------------------------------------------------------------------
        X = self._get_X().astype(float)

        if X.shape[0] != self.adata.n_obs or X.shape[1] != self.adata.n_vars:
            raise ValueError(
                f"X shape mismatch: got {X.shape}, expected "
                f"({self.adata.n_obs}, {self.adata.n_vars})."
            )

        if clip_negative_to_zero:
            X = np.where(X < 0, 0.0, X)

        if log1p:
            if np.nanmin(X) < -1:
                raise ValueError("log1p=True but X contains values < -1.")
            X = np.log1p(X)

        if normalize_per_cell:
            cell_sum = np.nansum(X, axis=1, keepdims=True)
            cell_sum[cell_sum == 0] = np.nan
            X = X / cell_sum * float(total_sum)

        time_values = pd.to_numeric(
            self.adata.obs[time_key],
            errors="coerce",
        ).values.astype(float)

        valid_cell_mask = np.isfinite(time_values)

        if not np.any(valid_cell_mask):
            raise ValueError(f"No finite values found in obs[{time_key!r}].")

        X = X[valid_cell_mask, :]
        time_values = time_values[valid_cell_mask]

        # ------------------------------------------------------------------
        # 3. Time range and bins
        # ------------------------------------------------------------------
        if time_range is None:
            t_min = float(np.nanmin(time_values))
            t_max = float(np.nanmax(time_values))
        else:
            if len(time_range) != 2:
                raise ValueError("time_range must be a tuple/list of length 2.")
            t_min, t_max = map(float, time_range)

        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min >= t_max:
            raise ValueError(f"Invalid time_range: {(t_min, t_max)}.")

        range_mask = (time_values >= t_min) & (time_values <= t_max)
        X = X[range_mask, :]
        time_values = time_values[range_mask]

        if X.shape[0] == 0:
            raise ValueError("No observations remain after applying time_range.")

        if bin_width is not None:
            bin_width = float(bin_width)
            bin_edges = np.arange(t_min, t_max + bin_width, bin_width)

            if bin_edges[-1] < t_max:
                bin_edges = np.append(bin_edges, t_max)
            else:
                bin_edges[-1] = t_max
        else:
            bin_edges = np.linspace(t_min, t_max, int(n_bins) + 1)

        if bin_edges.size < 2:
            raise ValueError("Time binning produced fewer than two bin edges.")

        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        n_windows = bin_edges.size - 1

        # ------------------------------------------------------------------
        # 4. Resolve valid classes
        # ------------------------------------------------------------------
        class_series = self.adata.var[class_key].copy()
        class_series = class_series.where(class_series.notna(), "")
        class_series = class_series.astype(str).str.strip()

        unknown_norm = {str(x).strip().lower() for x in unknown_labels}
        valid_feature_mask = ~class_series.str.lower().isin(unknown_norm)

        if include_classes is not None:
            include_set = {str(x) for x in include_classes}
            valid_feature_mask &= class_series.isin(include_set)

        if exclude_classes is not None:
            exclude_set = {str(x) for x in exclude_classes}
            valid_feature_mask &= ~class_series.isin(exclude_set)

        if renormalize_excluding_classes is not None:
            renorm_exclude_set = {str(x) for x in renormalize_excluding_classes}
            valid_feature_mask &= ~class_series.isin(renorm_exclude_set)

        class_series_selected = class_series[valid_feature_mask].copy()
        X_selected = X[:, valid_feature_mask.values]

        if X_selected.shape[1] == 0:
            raise ValueError("No features remain after class filtering.")

        # ------------------------------------------------------------------
        # 4.5 Merge classes with too few features into Other
        # ------------------------------------------------------------------
        small_class_feature_counts = None
        small_classes_merged = []

        if min_features_per_class is not None:
            small_class_feature_counts = (
                class_series_selected
                .value_counts(dropna=False)
                .to_dict()
            )

            small_classes = {
                cls
                for cls, count in small_class_feature_counts.items()
                if int(count) < min_features_per_class
            }

            # Do not repeatedly classify the target Other label as small.
            small_classes.discard(str(small_class_label))

            if len(small_classes) > 0:
                small_classes_merged = sorted([str(x) for x in small_classes])

                class_series_selected = class_series_selected.where(
                    ~class_series_selected.isin(small_classes),
                    str(small_class_label),
                )

        # ------------------------------------------------------------------
        # 5. Class totals after small-class merging
        # ------------------------------------------------------------------
        class_totals_raw = {
            cls: float(np.nansum(X_selected[:, class_series_selected.values == cls]))
            for cls in pd.unique(class_series_selected)
        }

        selected_classes = [
            cls
            for cls, total in class_totals_raw.items()
            if np.isfinite(total) and total >= float(min_class_total)
        ]

        if len(selected_classes) == 0:
            raise ValueError("No classes pass min_class_total.")

        if order_by == "total_intensity":
            selected_classes = sorted(
                selected_classes,
                key=lambda c: class_totals_raw.get(c, 0.0),
                reverse=bool(descending),
            )
        else:
            selected_classes = sorted(
                selected_classes,
                reverse=bool(descending),
            )

        if top_n_classes is not None:
            top_n_classes = int(top_n_classes)

            if top_n_classes < 1:
                raise ValueError("top_n_classes must be >= 1 or None.")

            kept_classes = selected_classes[:top_n_classes]
        else:
            kept_classes = selected_classes

        # ------------------------------------------------------------------
        # 6. Merge / drop non-kept classes
        # ------------------------------------------------------------------
        if other_label is not None:
            class_values_for_plot = class_series_selected.copy()
            keep_set = set(kept_classes)

            class_values_for_plot = class_values_for_plot.where(
                class_values_for_plot.isin(keep_set),
                str(other_label),
            )

            plot_classes = list(kept_classes)

            if np.any(class_values_for_plot.values == str(other_label)):
                if str(other_label) not in plot_classes:
                    plot_classes.append(str(other_label))
        else:
            keep_mask = class_series_selected.isin(set(kept_classes)).values

            X_selected = X_selected[:, keep_mask]
            class_values_for_plot = class_series_selected.iloc[
                np.where(keep_mask)[0]
            ].copy()

            plot_classes = list(kept_classes)

        if X_selected.shape[1] == 0:
            raise ValueError(
                "No features remain for plotting after top_n_classes filtering."
            )

        # ------------------------------------------------------------------
        # 7. Aggregate class intensity in each time bin
        # ------------------------------------------------------------------
        class_sum = pd.DataFrame(
            0.0,
            index=np.arange(n_windows),
            columns=plot_classes,
            dtype=float,
        )

        for i in range(n_windows):
            if i == n_windows - 1:
                cell_mask = (
                    (time_values >= bin_edges[i])
                    & (time_values <= bin_edges[i + 1])
                )
            else:
                cell_mask = (
                    (time_values >= bin_edges[i])
                    & (time_values < bin_edges[i + 1])
                )

            if not np.any(cell_mask):
                class_sum.iloc[i, :] = np.nan
                continue

            Xi = X_selected[cell_mask, :]

            for cls in plot_classes:
                feature_mask = class_values_for_plot.values == cls

                if np.any(feature_mask):
                    class_sum.loc[i, cls] = float(np.nansum(Xi[:, feature_mask]))
                else:
                    class_sum.loc[i, cls] = 0.0

        total_per_bin = class_sum.sum(axis=1).replace(0, np.nan)
        class_fraction = class_sum.div(total_per_bin, axis=0)

        # ------------------------------------------------------------------
        # 8. Reorder classes by plotted total intensity
        #    bottom -> top = high -> low by default
        # ------------------------------------------------------------------
        plotted_totals = class_sum.sum(axis=0, skipna=True).to_dict()

        if order_by == "total_intensity":
            plot_classes = sorted(
                plot_classes,
                key=lambda c: plotted_totals.get(c, 0.0),
                reverse=bool(descending),
            )
        else:
            plot_classes = sorted(
                plot_classes,
                reverse=bool(descending),
            )

        class_fraction = class_fraction[plot_classes]
        class_sum = class_sum[plot_classes]

        # ------------------------------------------------------------------
        # 9. Resolve colors
        # ------------------------------------------------------------------
        if colors is None:
            cmap_obj = plt.get_cmap(cmap, max(len(plot_classes), 1))
            color_list = [cmap_obj(i) for i in range(len(plot_classes))]
        elif isinstance(colors, dict):
            fallback = plt.get_cmap(cmap, max(len(plot_classes), 1))

            color_list = [
                colors.get(cls, fallback(i))
                for i, cls in enumerate(plot_classes)
            ]
        else:
            color_list = list(colors)

            if len(color_list) < len(plot_classes):
                raise ValueError(
                    f"colors has length {len(color_list)}, "
                    f"but {len(plot_classes)} classes are plotted."
                )

            color_list = color_list[:len(plot_classes)]

        # ------------------------------------------------------------------
        # 10. Plot
        # ------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=figsize)

        alpha_area = kwargs.get("alpha", 0.95)
        linewidth_area = kwargs.get("linewidth", 0.0)

        use_y_compression = (
            y_compress_regions is not None
            and len(y_compress_regions) > 0
        )

        if use_y_compression:
            y_transform, y_inverse, transformed_ylim, y_segments = (
                self._build_piecewise_axis_compressor(
                    ylim=ylim,
                    compress_regions=y_compress_regions,
                )
            )

            cumulative_lower = np.zeros(len(bin_centers), dtype=float)

            for cls, color in zip(plot_classes, color_list):
                y = class_fraction[cls].values.astype(float)
                y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

                cumulative_upper = cumulative_lower + y

                y_lower_t = y_transform(cumulative_lower)
                y_upper_t = y_transform(cumulative_upper)

                ax.fill_between(
                    bin_centers,
                    y_lower_t,
                    y_upper_t,
                    color=color,
                    alpha=alpha_area,
                    linewidth=linewidth_area,
                    label=cls,
                )

                cumulative_lower = cumulative_upper

            ax.set_ylim(*transformed_ylim)

            if y_compress_tick_values is None:
                y_compress_tick_values = np.linspace(ylim[0], ylim[1], 6)

            y_tick_values = np.asarray(y_compress_tick_values, dtype=float)
            y_tick_values = y_tick_values[
                (y_tick_values >= ylim[0]) & (y_tick_values <= ylim[1])
            ]

            ax.set_yticks(y_transform(y_tick_values))
            ax.set_yticklabels([f"{v:g}" for v in y_tick_values])

            if y_compress_show_boundaries:
                boundary_values = []

                for lo, hi, factor in y_compress_regions:
                    boundary_values.extend([float(lo), float(hi)])

                boundary_values = sorted(set(boundary_values))

                for yb in boundary_values:
                    if ylim[0] < yb < ylim[1]:
                        ax.axhline(
                            y_transform(np.asarray([yb]))[0],
                            color=y_compress_boundary_color,
                            linestyle=y_compress_boundary_linestyle,
                            linewidth=y_compress_boundary_linewidth,
                            alpha=kwargs.get("y_compress_boundary_alpha", 0.7),
                            zorder=5,
                        )

            if kwargs.get("append_compressed_to_ylabel", True):
                ylabel_for_plot = f"{ylabel} (compressed y-axis)"
            else:
                ylabel_for_plot = ylabel

        else:
            y_segments = None
            transformed_ylim = ylim

            y_arrays = [
                class_fraction[cls].values.astype(float)
                for cls in plot_classes
            ]

            ax.stackplot(
                bin_centers,
                y_arrays,
                labels=plot_classes,
                colors=color_list,
                alpha=alpha_area,
                linewidth=linewidth_area,
            )

            if ylim is not None:
                ax.set_ylim(*ylim)

            ylabel_for_plot = ylabel

        if event_time is not None:
            ax.axvline(
                float(event_time),
                color=event_color,
                linestyle=event_linestyle,
                linewidth=event_linewidth,
            )

        ax.set_xlim(t_min, t_max)
        ax.set_xlabel(xlabel if xlabel is not None else str(time_key))
        ax.set_ylabel(ylabel_for_plot)

        if title is not None:
            ax.set_title(title)

        if kwargs.get("despine", True):
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # ------------------------------------------------------------------
        # 11. Legend layout
        # ------------------------------------------------------------------
        handles, labels = ax.get_legend_handles_labels()

        if legend and len(labels) > 0:
            n_classes = len(labels)

            if legend_ncol is None:
                if legend_mode in {"top", "bottom"}:
                    legend_ncol_resolved = min(
                        max(1, n_classes),
                        kwargs.get("legend_max_ncol", 6),
                    )
                elif legend_mode == "right":
                    legend_ncol_resolved = 1 if n_classes <= 12 else 2
                else:
                    legend_ncol_resolved = min(n_classes, 3)
            else:
                legend_ncol_resolved = int(legend_ncol)

            if legend_mode == "top":
                fig.legend(
                    handles,
                    labels,
                    title=legend_title,
                    loc="upper center",
                    bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (0.5, 1.02)),
                    ncol=legend_ncol_resolved,
                    frameon=legend_frameon,
                    fontsize=legend_fontsize,
                    title_fontsize=kwargs.get(
                        "legend_title_fontsize",
                        legend_fontsize,
                    ),
                    columnspacing=kwargs.get("legend_columnspacing", 1.0),
                    handlelength=kwargs.get("legend_handlelength", 1.4),
                    handletextpad=kwargs.get("legend_handletextpad", 0.4),
                    borderaxespad=kwargs.get("legend_borderaxespad", 0.0),
                )

                layout_rect = kwargs.get("tight_layout_rect", (0.0, 0.0, 1.0, 0.84))

            elif legend_mode == "bottom":
                fig.legend(
                    handles,
                    labels,
                    title=legend_title,
                    loc="lower center",
                    bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (0.5, -0.04)),
                    ncol=legend_ncol_resolved,
                    frameon=legend_frameon,
                    fontsize=legend_fontsize,
                    title_fontsize=kwargs.get(
                        "legend_title_fontsize",
                        legend_fontsize,
                    ),
                    columnspacing=kwargs.get("legend_columnspacing", 1.0),
                    handlelength=kwargs.get("legend_handlelength", 1.4),
                    handletextpad=kwargs.get("legend_handletextpad", 0.4),
                    borderaxespad=kwargs.get("legend_borderaxespad", 0.0),
                )

                layout_rect = kwargs.get("tight_layout_rect", (0.0, 0.14, 1.0, 1.0))

            elif legend_mode == "right":
                fig.legend(
                    handles,
                    labels,
                    title=legend_title,
                    loc="center left",
                    bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (0.82, 0.5)),
                    ncol=legend_ncol_resolved,
                    frameon=legend_frameon,
                    fontsize=legend_fontsize,
                    title_fontsize=kwargs.get(
                        "legend_title_fontsize",
                        legend_fontsize,
                    ),
                    columnspacing=kwargs.get("legend_columnspacing", 1.0),
                    handlelength=kwargs.get("legend_handlelength", 1.4),
                    handletextpad=kwargs.get("legend_handletextpad", 0.4),
                    borderaxespad=kwargs.get("legend_borderaxespad", 0.0),
                )

                layout_rect = kwargs.get("tight_layout_rect", (0.0, 0.0, 0.78, 1.0))

            else:
                ax.legend(
                    title=legend_title,
                    loc=kwargs.get("legend_loc", "best"),
                    ncol=legend_ncol_resolved,
                    frameon=legend_frameon,
                    fontsize=legend_fontsize,
                    title_fontsize=kwargs.get(
                        "legend_title_fontsize",
                        legend_fontsize,
                    ),
                )

                layout_rect = kwargs.get("tight_layout_rect", None)

            if layout_rect is None:
                fig.tight_layout()
            else:
                fig.tight_layout(rect=layout_rect)
        else:
            fig.tight_layout()

        # ------------------------------------------------------------------
        # 12. Save figure
        # ------------------------------------------------------------------
        if output_file is None:
            if output_prefix is None:
                output_prefix = (
                    f"class_composition_"
                    f"{self._safe_filename(class_key)}_by_"
                    f"{self._safe_filename(time_key)}"
                )

                if renormalize_excluding_classes is not None:
                    excluded_token = "_without_" + "_".join(
                        self._safe_filename(x)
                        for x in renormalize_excluding_classes
                    )
                    output_prefix += excluded_token

                if min_features_per_class is not None:
                    output_prefix += f"_minfeat{min_features_per_class}"

            output_file = os.path.join(
                self.path,
                f"{output_prefix}.svg",
            )

        save_dir = os.path.dirname(output_file)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        fig.savefig(
            output_file,
            dpi=kwargs.get("dpi", 300),
            bbox_inches=kwargs.get("bbox_inches", "tight"),
        )

        plt.close(fig)

        # ------------------------------------------------------------------
        # 13. Store result
        # ------------------------------------------------------------------
        result = {
            "time_key": time_key,
            "class_key": class_key,
            "bin_edges": bin_edges,
            "bin_centers": bin_centers,
            "classes": plot_classes,
            "class_sum": class_sum,
            "class_fraction": class_fraction,
            "class_totals": plotted_totals,
            "normalize_per_cell": bool(normalize_per_cell),
            "log1p": bool(log1p),
            "time_range": (t_min, t_max),
            "event_time": event_time,
            "output_file": output_file,
            "legend_mode": legend_mode,
            "legend_ncol": legend_ncol,
            "exclude_classes": exclude_classes,
            "renormalize_excluding_classes": renormalize_excluding_classes,
            "min_features_per_class": min_features_per_class,
            "small_class_label": small_class_label,
            "small_class_feature_counts": small_class_feature_counts,
            "small_classes_merged": small_classes_merged,
            "y_compress_regions": y_compress_regions,
            "y_compress_segments": y_segments,
            "transformed_ylim": transformed_ylim,
        }

        if store_uns_key is not None:
            self.adata.uns[store_uns_key] = result

        if return_result:
            return result

        return self