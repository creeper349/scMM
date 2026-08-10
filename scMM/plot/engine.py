from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

from ._trajectory import (
    metabolic_velocity_field,
    metabolite_trends,
    resample_trajectory,
    run_palantir,
    trend_cluster,
)


class PlotEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        fig_path_dir: str,
        obs: pd.DataFrame | None = None,
        var: pd.DataFrame | None = None,
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

                X_umap_input = PCA(n_components=n_pca, random_state=random_state).fit_transform(X)
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

    def run_palantir(
        self,
        start_idx: int,
        plotting: bool = False,
        cmap: str = "viridis",
        use_obsm: str = "X_umap",
        s=1,
        **kwargs,
    ):
        self.adata = run_palantir(adata=self.adata, start_idx=start_idx, **kwargs)
        if plotting:
            obsm = self.adata.obsm.get(use_obsm, None)
            assert obsm is not None, f"{use_obsm} not found in obsm"
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(
                obsm[:, 0], obsm[:, 1], c=self.adata.obs["palantir_pseudotime"], cmap=cmap, s=s
            )
            plt.colorbar(sc, label="Pseudotime")
            for spine in plt.gca().spines.values():
                spine.set_visible(False)
            plt.savefig(f"{self.path}/palantir_pseudotime.svg", bbox_inches="tight")
            plt.close()
        return self

    def compute_trajectory(
        self,
        window_size: int = 100,
        step_size: int = 50,
        cell_dist_key: str = "X_umap",
        parameterization_key: str = "palantir_pseudotime",
        branch_prob_key: str = "palantir_branch_probs",
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
            obsm = self.adata.obsm.get(cell_dist_key, None)
            assert obsm is not None, f"{cell_dist_key} not found in obsm"
            traj = self.adata.uns[store_key]
            plt.figure(figsize=(6, 6))
            sc = plt.scatter(
                obsm[:, 0], obsm[:, 1], c=self.adata.obs[parameterization_key], cmap=cmap, s=s
            )
            for b in range(traj.shape[0]):
                traj_points_mask = np.arange(0, traj.shape[1], traj.shape[1] // traj_points)
                plt.plot(traj[b, :, 0], traj[b, :, 1], color="black", linewidth=traj_linewidth)
                plt.scatter(
                    traj[b, traj_points_mask, 0],
                    traj[b, traj_points_mask, 1],
                    color="black",
                    s=2 * traj_points,
                )
            plt.colorbar(sc, label="Pseudotime")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.xticks([])
            plt.yticks([])
            if title is not None:
                plt.title(title, size=16)
            plt.savefig(f"{self.path}/trajectory.svg", bbox_inches="tight")
            plt.close()

    def metabolic_velocity(
        self,
        window_size: int = 100,
        step_size: int = 50,
        parameterization_key: str = "time",
        plot=True,
        linewidth=1,
        **kwargs,
    ):
        self.adata = metabolic_velocity_field(
            adata=self.adata,
            window_size=window_size,
            step_size=step_size,
            parameterization_key=parameterization_key,
        )
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
        feature_name_key: str | None = None,
        plot_top_n: int | None = None,
        cmap: str = "viridis",
        **kwargs,
    ):
        self.adata = metabolite_trends(
            adata=self.adata,
            window_size=window_size,
            step_size=step_size,
            parameterization_key=parameterization_key,
            kernel_stat=kernel_stat,
            feature_name_key=feature_name_key,
        )

        if plot_top_n is not None:
            rank_idx = self.adata.uns["metabolite_trends"]["rank_idx"]

            # remove features with empty feature_name_key
            if feature_name_key is not None:
                feature_values = self.adata.var[feature_name_key]
                valid_mask = feature_values.notna() & (feature_values.astype(str).str.strip() != "")
                valid_idx = set(np.where(valid_mask.values)[0])
                rank_idx = [idx for idx in rank_idx if idx in valid_idx]

            top_idx = rank_idx[: min(plot_top_n, len(rank_idx))]

            M = self.adata.uns["metabolite_trends"]["pooled"][:, top_idx].T

            # row-wise z-score for visualization
            M_plot = M.copy()
            row_mean = np.nanmean(M_plot, axis=1, keepdims=True)
            row_std = np.nanstd(M_plot, axis=1, keepdims=True)
            row_std[row_std == 0] = 1.0
            M_plot = (M_plot - row_mean) / row_std

            fig_h = max(4, 0.25 * len(top_idx))
            fig_w = max(6, 0.35 * M_plot.shape[1])

            plt.figure(figsize=(fig_w, fig_h))
            im = plt.imshow(
                M_plot,
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
            )

            feature_names = self.adata.uns["metabolite_trends"]["feature_names"]
            ylabels = [feature_names[j] for j in top_idx]

            plt.yticks(np.arange(len(top_idx)), ylabels)
            plt.xticks(
                np.arange(len(self.adata.uns["metabolite_trends"]["time_centers"])),
                [
                    f"{x:.2f}" if np.isfinite(x) else ""
                    for x in self.adata.uns["metabolite_trends"]["time_centers"]
                ],
                rotation=90,
            )
            plt.xlabel(kwargs.get("xlabel", parameterization_key))
            plt.ylabel(kwargs.get("ylabel", "Metabolite"))
            plt.colorbar(im, label="Row-wise z-scored pooled intensity")
            plt.tight_layout()
            plt.savefig(f"{self.path}/metabolite_trends_top{len(top_idx)}.svg", bbox_inches="tight")
            plt.close()

    def plot_trend_clusters(
        self,
        metric: str = "correlation",
        cluster_method: str = "leiden",
        linewidth: float = 1.0,
        top_k: int | None = None,
        **kwargs,
    ):
        trends = self.adata.uns["metabolite_trends"]["pooled"]
        top_k_idx = (
            self.adata.uns["metabolite_trends"]["rank_idx"][:top_k]
            if top_k is not None
            else np.arange(trends.shape[1])
        )
        trends = trends[:, top_k_idx]

        # z-score each feature across time
        mean = np.nanmean(trends, axis=0, keepdims=True)
        std = np.nanstd(trends, axis=0, keepdims=True)
        std[std == 0] = 1.0
        trends = (trends - mean) / std

        cluster_labels = trend_cluster(
            trends, metric=metric, cluster_method=cluster_method, **kwargs
        )
        label_unique = np.unique(cluster_labels)
        _fig, ax = plt.subplots(nrows=label_unique.size, figsize=(6, 6 * label_unique.size))

        if label_unique.size == 1:
            ax = [ax]

        for i, label in enumerate(label_unique):
            cluster_trends = trends.T[cluster_labels == label]
            mean_trend = np.nanmean(cluster_trends, axis=0)
            ax[i].plot(
                self.adata.uns["metabolite_trends"]["time_centers"],
                mean_trend,
                color="black",
                linewidth=linewidth,
            )
            cluster_idx = np.where(cluster_labels == label)
            for j in cluster_idx[0]:
                ax[i].plot(
                    self.adata.uns["metabolite_trends"]["time_centers"],
                    trends.T[j],
                    color="gray",
                    alpha=0.5,
                    linewidth=0.5,
                )
            ax[i].set_title(f"Cluster {label} (n={cluster_trends.shape[0]})")
            ax[i].set_xlabel(kwargs.get("xlabel", "Time"))
            ax[i].set_ylabel(kwargs.get("ylabel", "Relative intensity"))

        plt.tight_layout()
        plt.savefig(
            f"{self.path}/trend_clusters_{cluster_method}_{metric}.svg", bbox_inches="tight"
        )
        plt.close()

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
        **kwargs,
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
            X_cluster, n_neighbors=n_neighbors, mode="connectivity", include_self=False
        )

        sources, targets = knn.nonzero()
        edges = list(zip(sources.tolist(), targets.tolist(), strict=True))

        if method == "leiden":
            try:
                import igraph as ig
                import leidenalg
            except ImportError as exc:
                raise ImportError(
                    "Leiden clustering requires igraph and leidenalg. "
                    "Install the Conda packages python-igraph and leidenalg."
                ) from exc

            g = ig.Graph(n=X_cluster.shape[0], edges=edges, directed=False)
            g.simplify()

            partition = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=resolution,
                seed=random_state,
                **kwargs,
            )
            labels = np.array(partition.membership, dtype=int)

        elif method == "louvain":
            try:
                import networkx as nx
                from networkx.algorithms.community import louvain_communities
            except ImportError as exc:
                raise ImportError(
                    "Louvain clustering requires the Conda package networkx."
                ) from exc

            G = nx.Graph()
            G.add_nodes_from(range(X_cluster.shape[0]))
            G.add_edges_from(edges)

            communities = louvain_communities(G, resolution=resolution, seed=random_state, **kwargs)

            labels = np.empty(X_cluster.shape[0], dtype=int)
            for i, comm in enumerate(communities):
                for node in comm:
                    labels[node] = i

        self.adata.obs[key_added] = pd.Categorical(labels.astype(str))
        fig, ax = plt.subplots(figsize=figsize)

        uniq = np.unique(labels)
        palette = sns.color_palette(cmap, n_colors=len(uniq))
        color_map = {lab: palette[i] for i, lab in enumerate(uniq)}

        for lab in uniq:
            idx = labels == lab
            ax.scatter(
                X_plot[idx, 0], X_plot[idx, 1], s=s, color=color_map[lab], linewidths=0, alpha=0.85
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
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
            )

        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_title(f"{method.capitalize()} clustering")
        plt.tight_layout()
        plt.savefig(f"{self.path}/{method}_{key_added}_umap.svg", bbox_inches="tight")
        plt.close(fig)

    def feature_network(
        self,
        name_key: str | None = None,
        class_key: str | None = None,
        metric: str = "pearson",
        **kwargs,
    ):
        if name_key is not None and name_key in self.adata.var.columns:
            feature_names = self.adata.var[name_key].values

            nan_mask = (
                pd.isna(feature_names)
                | (pd.Series(feature_names).astype(str).str.strip() == "").values
            )

            feature_names[nan_mask] = [f"feature_{i}" for i in np.where(nan_mask)[0]]
            feature_arr = self.adata.X[:, ~nan_mask]
            feature_names = feature_names[~nan_mask]
        else:
            feature_names = np.array([f"feature_{i}" for i in range(self.adata.n_vars)])
            feature_arr = self.adata.X

        if metric == "pearson":
            corr_matrix = np.corrcoef(feature_arr.T)
            np.fill_diagonal(corr_matrix, 1)
            dist = np.sqrt(2 * (1 - np.abs(corr_matrix)))
        elif metric == "euclidean":
            from sklearn.metrics import pairwise_distances

            corr_matrix = pairwise_distances(feature_arr.T, metric="euclidean")

        feature_emb = umap.UMAP(
            metric="precomputed",
            n_neighbors=kwargs.get("n_neighbors", 15),
            min_dist=kwargs.get("min_dist", 0.1),
        ).fit_transform(dist)

        if np.issubdtype(np.array(feature_names).dtype, np.number):
            colors = feature_names.astype(float)
            cmap = kwargs.get("cmap", "viridis")
        else:
            if class_key is not None and class_key in self.adata.var.columns:
                colors = self.adata.var.loc[~nan_mask, class_key].astype("category").cat.codes
            else:
                colors = np.zeros(len(feature_names))

            cmap = kwargs.get("cmap", "tab20")

        plt.figure(figsize=kwargs.get("figsize", (6, 6)))
        plt.scatter(feature_emb[:, 0], feature_emb[:, 1], s=10, c=colors, cmap=cmap)

        for i in range(feature_emb.shape[0]):
            plt.text(
                feature_emb[i, 0],
                feature_emb[i, 1],
                str(feature_names[i]),
                fontsize=kwargs.get("fontsize", 8),
                alpha=kwargs.get("alpha", 0.7),
            )

        plt.savefig(f"{self.path}/feature_network_{metric}.svg", bbox_inches="tight")

        return feature_emb
