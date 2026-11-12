from __future__ import annotations

from pathlib import Path
from typing import Optional

import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import umap
from scipy import stats
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def _bh_fdr(p_values):
    p = np.asarray(p_values, dtype=float)
    result = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return result
    values = p[valid]
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


class PlotEngine:
    """Focused analysis and visualization methods for cell-by-feature AnnData."""

    def __init__(self, data: ad.AnnData, fig_path_dir=".", *, copy=True):
        if not isinstance(data, ad.AnnData):
            raise TypeError("data must be an AnnData object")
        self.adata = data.copy() if copy else data
        self.path = Path(fig_path_dir)
        self.path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_adata(cls, adata, fig_path_dir=".", *, copy=True):
        return cls(adata, fig_path_dir=fig_path_dir, copy=copy)

    def _get_X(self):
        X = self.adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        return X[np.ix_(self._cell_index(), self._feature_index_array())]

    def _cell_mask(self):
        if hasattr(self.adata, "cell_keep_mask"):
            return self.adata.cell_keep_mask()
        columns = [name for name in self.adata.obs if str(name).startswith("cell_keep_")]
        if not columns:
            return np.ones(self.adata.n_obs, dtype=bool)
        return np.all(self.adata.obs[columns].to_numpy(dtype=np.uint8) == 1, axis=1)

    def _feature_mask(self):
        if hasattr(self.adata, "feature_keep_mask"):
            return self.adata.feature_keep_mask()
        columns = [name for name in self.adata.var if str(name).startswith("feature_keep_")]
        if not columns:
            return np.ones(self.adata.n_vars, dtype=bool)
        return np.all(self.adata.var[columns].to_numpy(dtype=np.uint8) == 1, axis=1)

    def _cell_index(self):
        return np.flatnonzero(self._cell_mask())

    def _feature_index_array(self):
        return np.flatnonzero(self._feature_mask())

    def _get_rep(self, key):
        return np.asarray(self.adata.obsm[key], dtype=float)[self._cell_mask()]

    def _store_embedding(self, key, values):
        values = np.asarray(values, dtype=float)
        full = np.full((self.adata.n_obs, values.shape[1]), np.nan, dtype=float)
        full[self._cell_mask()] = values
        self.adata.obsm[key] = full
        self.adata.uns[f"{key}_source_layer"] = str(self.adata.uns.get("active_layer", "X"))
        return values

    def pca(
        self,
        n_components=50,
        *,
        scale=True,
        zero_center=True,
        random_state=42,
        store_key="X_pca",
    ):
        X = self._get_X()
        if scale or zero_center:
            X = StandardScaler(
                with_mean=zero_center,
                with_std=scale,
            ).fit_transform(X)
        n_components = min(int(n_components), X.shape[0], X.shape[1])
        if n_components < 1:
            raise ValueError("PCA requires at least one cell and feature")
        model = PCA(n_components=n_components, random_state=random_state)
        embedding = model.fit_transform(X)
        self._store_embedding(store_key, embedding)
        self.adata.uns[f"{store_key}_params"] = {
            "n_components": n_components,
            "scale": bool(scale),
            "zero_center": bool(zero_center),
            "explained_variance_ratio": model.explained_variance_ratio_,
        }
        return embedding

    def umap(
        self,
        *,
        use_rep="X_pca",
        n_components=2,
        n_neighbors=30,
        min_dist=0.3,
        metric="euclidean",
        random_state=42,
        store_key="X_umap",
    ):
        if use_rep is None:
            X = self._get_X()
        else:
            if use_rep not in self.adata.obsm:
                if use_rep == "X_pca":
                    self.pca()
                else:
                    raise KeyError(f"{use_rep!r} not found in adata.obsm")
            X = self._get_rep(use_rep)
        model = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, max(2, len(X) - 1)),
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        embedding = model.fit_transform(X)
        self._store_embedding(store_key, embedding)
        self.adata.uns[f"{store_key}_params"] = {
            "source": "X" if use_rep is None else use_rep,
            "n_components": int(n_components),
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": metric,
        }
        return embedding

    def reduce_dimension(
        self,
        method="pca",
        *,
        use_rep=None,
        n_components=2,
        scale=True,
        random_state=42,
        store_key=None,
        **kwargs,
    ):
        """Run a supported embedding and store it in ``adata.obsm``."""
        method = str(method).lower()
        store_key = store_key or f"X_{method}"
        if method == "pca":
            return self.pca(
                n_components=n_components,
                scale=scale,
                random_state=random_state,
                store_key=store_key,
            )
        if method == "umap":
            return self.umap(
                use_rep=use_rep,
                n_components=n_components,
                random_state=random_state,
                store_key=store_key,
                **kwargs,
            )
        if use_rep is None:
            X = self._get_X()
        else:
            if use_rep not in self.adata.obsm:
                raise KeyError(f"{use_rep!r} not found in adata.obsm")
            X = self._get_rep(use_rep)
        if scale:
            X = StandardScaler().fit_transform(X)
        components = min(int(n_components), X.shape[1], max(1, X.shape[0] - 1))
        if method in {"tsne", "t-sne"}:
            perplexity = min(float(kwargs.pop("perplexity", 30.0)), X.shape[0] - 1)
            if components > 3:
                kwargs.setdefault("method", "exact")
            model = TSNE(
                n_components=components,
                perplexity=max(1.0, perplexity),
                random_state=random_state,
                **kwargs,
            )
        elif method == "isomap":
            neighbors = min(int(kwargs.pop("n_neighbors", 15)), X.shape[0] - 1)
            model = Isomap(
                n_components=components,
                n_neighbors=max(1, neighbors),
                **kwargs,
            )
        elif method in {"lle", "locally_linear"}:
            neighbors = min(int(kwargs.pop("n_neighbors", 15)), X.shape[0] - 1)
            neighbors = min(len(X) - 1, max(components + 1, neighbors))
            model = LocallyLinearEmbedding(
                n_components=components,
                n_neighbors=neighbors,
                random_state=random_state,
                **kwargs,
            )
        else:
            raise ValueError("method must be pca, umap, tsne, isomap or lle")
        embedding = model.fit_transform(X)
        self._store_embedding(store_key, embedding)
        self.adata.uns[f"{store_key}_params"] = {
            "method": method,
            "source": "X" if use_rep is None else use_rep,
            "n_components": components,
            "scale": bool(scale),
            **kwargs,
        }
        return embedding

    @staticmethod
    def _normalize_pseudotime(values):
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        result = np.full(len(values), np.nan, dtype=float)
        if not finite.any():
            return result
        low, high = np.min(values[finite]), np.max(values[finite])
        result[finite] = 0.0 if high == low else (values[finite] - low) / (high - low)
        return result

    @staticmethod
    def _knn(X, n_neighbors):
        """Query Euclidean neighbors without sklearn/threadpoolctl DLL probing."""
        values = np.ascontiguousarray(X, dtype=np.float64)
        neighbors = min(max(1, int(n_neighbors)), len(values) - 1)
        distances, indices = cKDTree(values).query(
            values,
            k=neighbors + 1,
            workers=1,
        )
        return (
            np.asarray(distances, dtype=float)[:, 1:],
            np.asarray(indices, dtype=np.int64)[:, 1:],
        )

    def _root_index(self, X, *, root_cell=None, root_obs=None, root_value=None):
        active_obs = self.adata.obs.iloc[self._cell_index()]
        if root_cell is not None and str(root_cell):
            if str(root_cell) not in active_obs.index:
                raise KeyError(f"root cell {root_cell!r} was not found")
            return int(active_obs.index.get_loc(str(root_cell)))
        if root_obs and root_obs in active_obs and root_value is not None:
            mask = active_obs[root_obs].astype(str).eq(str(root_value)).to_numpy()
            if not mask.any():
                raise ValueError(f"no cells match {root_obs}={root_value!r}")
            candidates = np.flatnonzero(mask)
            center = np.mean(X[candidates], axis=0)
            return int(candidates[np.argmin(np.linalg.norm(X[candidates] - center, axis=1))])
        return 0

    def trajectory(
        self,
        method="diffusion_pseudotime",
        *,
        use_rep="X_pca",
        key_added="pseudotime",
        root_cell=None,
        root_obs=None,
        root_value=None,
        n_neighbors=15,
        n_components=10,
        cluster_key="cluster",
        random_state=42,
    ):
        """Infer pseudotime and write it to ``obs``.

        Diffusion components are additionally stored in ``obsm['X_diffmap']``.
        """
        if use_rep is None:
            X = self._get_X()
        else:
            if use_rep not in self.adata.obsm:
                if use_rep == "X_pca":
                    self.pca(n_components=max(2, n_components))
                else:
                    raise KeyError(f"{use_rep!r} not found in adata.obsm")
            X = self._get_rep(use_rep)
        if len(X) < 3:
            raise ValueError("trajectory analysis requires at least three cells")
        root = self._root_index(
            X,
            root_cell=root_cell,
            root_obs=root_obs,
            root_value=root_value,
        )
        neighbors = min(max(2, int(n_neighbors)), len(X) - 1)
        method = str(method).lower()
        if method in {"diffusion_pseudotime", "dpt", "diffusion"}:
            dimensions = min(int(n_components), X.shape[1], len(X) - 2)
            embedding = SpectralEmbedding(
                n_components=max(1, dimensions),
                affinity="nearest_neighbors",
                n_neighbors=neighbors,
                random_state=random_state,
            ).fit_transform(X)
            self._store_embedding("X_diffmap", embedding)
            distance = np.linalg.norm(embedding - embedding[root], axis=1)
        elif method in {"graph_pseudotime", "knn_geodesic"}:
            distances, indices = self._knn(X, neighbors)
            rows = np.repeat(np.arange(len(X)), neighbors)
            cols = indices.reshape(-1)
            weights = distances.reshape(-1)
            graph = csr_matrix((weights, (rows, cols)), shape=(len(X), len(X)))
            graph = graph.maximum(graph.T)
            distance = shortest_path(graph, directed=False, indices=root)
        elif method in {"pca_pseudotime", "pc_pseudotime"}:
            score = PCA(n_components=1, random_state=random_state).fit_transform(
                StandardScaler().fit_transform(X)
            )[:, 0]
            distance = np.abs(score - score[root])
        elif method in {"paga_pseudotime", "cluster_graph"}:
            if cluster_key not in self.adata.obs:
                raise KeyError(f"{cluster_key!r} not found in adata.obs")
            labels = self.adata.obs.iloc[self._cell_index()][cluster_key].astype(str).to_numpy()
            groups = sorted(np.unique(labels))
            centers = np.vstack([np.mean(X[labels == label], axis=0) for label in groups])
            center_distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
            tree = minimum_spanning_tree(center_distance)
            tree = tree.maximum(tree.T)
            root_group = groups.index(labels[root])
            group_distance = shortest_path(tree, directed=False, indices=root_group)
            distance = np.asarray([group_distance[groups.index(label)] for label in labels])
            local = np.asarray(
                [np.linalg.norm(X[index] - centers[groups.index(label)]) for index, label in enumerate(labels)]
            )
            distance = distance + local
        else:
            raise ValueError(
                "method must be diffusion_pseudotime, graph_pseudotime, "
                "pca_pseudotime or paga_pseudotime"
            )
        pseudotime = self._normalize_pseudotime(distance)
        full_pseudotime = np.full(self.adata.n_obs, np.nan, dtype=float)
        full_pseudotime[self._cell_mask()] = pseudotime
        self.adata.obs[key_added] = full_pseudotime
        self.adata.uns[f"{key_added}_params"] = {
            "method": method,
            "source": "X" if use_rep is None else use_rep,
            "root_cell": str(self.adata.obs_names[self._cell_index()[root]]),
            "n_neighbors": neighbors,
            "cluster_key": cluster_key,
        }
        return pd.Series(full_pseudotime, index=self.adata.obs_names, name=key_added)

    def cluster_cells(
        self,
        method="kmeans",
        *,
        use_rep="X_pca",
        key_added="cluster",
        random_state=42,
        **kwargs,
    ):
        if use_rep is None:
            X = self._get_X()
        else:
            if use_rep not in self.adata.obsm:
                if use_rep == "X_pca":
                    self.pca()
                else:
                    raise KeyError(f"{use_rep!r} not found in adata.obsm")
            X = self._get_rep(use_rep)

        graph_methods = {"leiden", "louvain"}
        if method in graph_methods:
            n_neighbors = int(kwargs.pop("n_neighbors", 15))
            resolution = float(kwargs.pop("resolution", 1.0))
            if kwargs:
                unknown = ", ".join(sorted(kwargs))
                raise TypeError(f"unexpected {method} parameters: {unknown}")
            labels = self._community_labels(
                X,
                method=method,
                n_neighbors=n_neighbors,
                resolution=resolution,
                random_state=random_state,
            )
        elif method == "kmeans":
            model = KMeans(random_state=random_state, **kwargs)
        elif method == "dbscan":
            model = DBSCAN(**kwargs)
        elif method in {"agglomerative", "hierarchical"}:
            model = AgglomerativeClustering(**kwargs)
        else:
            raise ValueError(
                "method must be kmeans, dbscan, agglomerative, leiden or louvain"
            )
        if method not in graph_methods:
            labels = np.asarray(model.fit_predict(X)).astype(str)
        else:
            labels = np.asarray(labels).astype(str)
        full_labels = np.full(self.adata.n_obs, None, dtype=object)
        full_labels[self._cell_mask()] = labels
        self.adata.obs[key_added] = pd.Categorical(full_labels)

        unique = np.unique(labels)
        valid = len(unique) > 1 and len(unique) < len(labels)
        qc = {
            "method": method,
            "n_clusters": int(len(unique)),
            "n_cells": int(len(labels)),
        }
        if valid:
            qc.update(
                {
                    "silhouette": float(silhouette_score(X, labels)),
                    "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X, labels)),
                }
            )
        self.adata.uns[f"{key_added}_qc"] = qc
        return pd.Series(full_labels, index=self.adata.obs_names, name=key_added)

    @staticmethod
    def _community_labels(
        X,
        *,
        method,
        n_neighbors=15,
        resolution=1.0,
        random_state=42,
    ):
        """Cluster a weighted kNN graph with Leiden or Louvain."""
        try:
            import igraph as ig
        except ImportError as exc:
            raise ImportError(
                "Leiden/Louvain clustering requires the 'cluster' extra: "
                "pip install -e .[cluster]"
            ) from exc
        if len(X) < 2:
            raise ValueError("graph clustering requires at least two cells")
        neighbors = min(max(1, int(n_neighbors)), len(X) - 1)
        distances, indices = PlotEngine._knn(X, neighbors)
        positive = distances[distances > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        edges = {}
        for source in range(len(X)):
            for target, distance in zip(indices[source], distances[source]):
                edge = tuple(sorted((source, int(target))))
                weight = float(np.exp(-float(distance) / scale))
                edges[edge] = max(weight, edges.get(edge, 0.0))
        graph = ig.Graph(n=len(X), edges=list(edges), directed=False)
        graph.es["weight"] = list(edges.values())
        if method == "leiden":
            try:
                import leidenalg
            except ImportError as exc:
                raise ImportError(
                    "Leiden clustering requires leidenalg: "
                    "pip install -e .[cluster]"
                ) from exc
            partition = leidenalg.find_partition(
                graph,
                leidenalg.RBConfigurationVertexPartition,
                weights="weight",
                resolution_parameter=float(resolution),
                seed=int(random_state),
            )
        else:
            import random

            ig.set_random_number_generator(random.Random(int(random_state)))
            partition = graph.community_multilevel(
                weights="weight",
                resolution=float(resolution),
            )
        return np.asarray(partition.membership, dtype=int)

    def differential_features(
        self,
        group_key,
        group_a,
        group_b,
        *,
        method="mannwhitney",
        min_present=3,
        include_zeros=True,
    ):
        if group_key not in self.adata.obs:
            raise KeyError(f"{group_key!r} not found in adata.obs")
        cell_index = self._cell_index()
        feature_index = self._feature_index_array()
        groups = self.adata.obs.iloc[cell_index][group_key].astype(str)
        mask_a = (groups == str(group_a)).to_numpy()
        mask_b = (groups == str(group_b)).to_numpy()
        if mask_a.sum() < min_present or mask_b.sum() < min_present:
            raise ValueError("each group must contain at least min_present cells")
        X = self._get_X()
        rows = []
        eps = np.finfo(float).eps
        for local_index, index in enumerate(feature_index):
            a = X[mask_a, local_index]
            b = X[mask_b, local_index]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if not include_zeros:
                a, b = a[a != 0], b[b != 0]
            if len(a) < min_present or len(b) < min_present:
                statistic, p_value = np.nan, np.nan
            elif method == "mannwhitney":
                statistic, p_value = stats.mannwhitneyu(
                    a, b, alternative="two-sided"
                )
            elif method in {"welch", "ttest"}:
                statistic, p_value = stats.ttest_ind(
                    a, b, equal_var=False, nan_policy="omit"
                )
            else:
                raise ValueError("method must be mannwhitney or welch")
            mean_a = float(np.mean(a)) if len(a) else np.nan
            mean_b = float(np.mean(b)) if len(b) else np.nan
            rows.append(
                {
                    "feature_id": self.adata.var_names[index],
                    "mz": self.adata.var.iloc[index].get("mz", np.nan),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "log2_fold_change": float(
                        np.log2((mean_a + eps) / (mean_b + eps))
                    ),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "n_a": len(a),
                    "n_b": len(b),
                }
            )
        result = pd.DataFrame(rows)
        result["fdr"] = _bh_fdr(result["p_value"])
        return result.sort_values(["fdr", "p_value"], ignore_index=True)

    def marker_features(self, cluster_key="cluster", **kwargs):
        if cluster_key not in self.adata.obs:
            raise KeyError(f"{cluster_key!r} not found in adata.obs")
        markers = []
        active = self._cell_mask()
        labels = self.adata.obs[cluster_key]
        temporary_key = "__scmm_marker_group"
        for cluster in sorted(labels.iloc[np.flatnonzero(active)].dropna().astype(str).unique()):
            self.adata.obs[temporary_key] = np.where(
                labels.astype(str) == cluster, cluster, "__rest__"
            )
            table = self.differential_features(
                temporary_key, cluster, "__rest__", **kwargs
            )
            table.insert(0, "cluster", cluster)
            markers.append(table)
        del self.adata.obs[temporary_key]
        return pd.concat(markers, ignore_index=True)

    def feature_correlation_network(
        self,
        *,
        method="pearson",
        min_abs_correlation=0.7,
        top_features=None,
    ):
        X = self._get_X()
        feature_names = self.adata.var_names[self._feature_index_array()]
        if top_features is not None and top_features < X.shape[1]:
            variance = np.nanvar(X, axis=0)
            selected = np.argsort(variance)[-int(top_features):]
        else:
            selected = np.arange(X.shape[1])
        frame = pd.DataFrame(X[:, selected], columns=feature_names[selected])
        corr = frame.corr(method=method)
        graph = nx.Graph()
        for name in corr.columns:
            graph.add_node(name)
        for i, left in enumerate(corr.columns):
            for j in range(i + 1, len(corr.columns)):
                right = corr.columns[j]
                value = float(corr.iloc[i, j])
                if np.isfinite(value) and abs(value) >= min_abs_correlation:
                    graph.add_edge(left, right, correlation=value, weight=abs(value))
        return graph, corr

    def plot_feature_scatter(
        self,
        feature_a,
        feature_b,
        *,
        color_key=None,
        regression=True,
        output_file=None,
    ):
        idx_a = self._feature_index(feature_a)
        idx_b = self._feature_index(feature_b)
        X = self._get_X()
        active_features = self._feature_index_array()
        if idx_a not in active_features or idx_b not in active_features:
            raise ValueError("selected scatter features are disabled by active filters")
        local_a = int(np.flatnonzero(active_features == idx_a)[0])
        local_b = int(np.flatnonzero(active_features == idx_b)[0])
        x, y = X[:, local_a], X[:, local_b]
        fig, ax = plt.subplots(figsize=(5, 5))
        color = (
            self.adata.obs.iloc[self._cell_index()][color_key]
            if color_key
            else None
        )
        scatter = ax.scatter(x, y, c=color, s=8, alpha=0.7, cmap="viridis")
        if color_key and pd.api.types.is_numeric_dtype(color):
            fig.colorbar(scatter, ax=ax, label=color_key)
        if regression:
            finite = np.isfinite(x) & np.isfinite(y)
            if finite.sum() >= 2:
                slope, intercept, r_value, p_value, _ = stats.linregress(
                    x[finite], y[finite]
                )
                order = np.argsort(x[finite])
                xx = x[finite][order]
                ax.plot(xx, intercept + slope * xx, color="red")
                ax.set_title(f"R²={r_value ** 2:.3f}, p={p_value:.2g}")
        ax.set_xlabel(str(feature_a))
        ax.set_ylabel(str(feature_b))
        from .ticks import tidy_numeric_ticks
        tidy_numeric_ticks(fig)
        fig.tight_layout()
        target = Path(output_file) if output_file else self.path / "feature_scatter.svg"
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return target

    def _feature_index(self, feature):
        if isinstance(feature, (int, np.integer)):
            return int(feature)
        text = str(feature)
        if text in self.adata.var_names:
            return int(self.adata.var_names.get_loc(text))
        if "mz" in self.adata.var:
            mz = self.adata.var["mz"].to_numpy(dtype=float)
            try:
                return int(np.abs(mz - float(feature)).argmin())
            except ValueError:
                pass
        raise KeyError(f"feature {feature!r} was not found")
