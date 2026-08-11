"""Cell clustering and cluster-plot helpers for :class:`PlotEngine`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import kneighbors_graph


class CellClusteringMixin:
    """Provide graph-based cell clustering for a PlotEngine-like object."""

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
        """Cluster PCA coordinates and save a labeled UMAP figure."""
        method = method.lower()
        if method not in {"leiden", "louvain"}:
            raise ValueError("method must be 'leiden' or 'louvain'")
        cluster_values, plot_values = _get_cluster_inputs(self.adata, n_neighbors)
        edges = _build_neighbor_edges(cluster_values, n_neighbors)
        labels = _cluster_graph(
            method,
            cluster_values.shape[0],
            edges,
            resolution,
            random_state,
            kwargs,
        )
        self.adata.obs[key_added] = pd.Categorical(labels.astype(str))
        _save_cluster_plot(
            plot_values,
            labels,
            self.path / f"{method}_{key_added}_umap.svg",
            method,
            figsize,
            s,
            cmap,
        )
        return self


def _get_cluster_inputs(adata, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    cluster_values = adata.obsm.get("X_pca")
    if cluster_values is None:
        raise KeyError("X_pca not found in self.adata.obsm")
    plot_values = adata.obsm.get("X_umap")
    if plot_values is None:
        raise KeyError("X_umap not found in self.adata.obsm")
    if plot_values.shape[1] < 2:
        raise ValueError("X_umap must contain at least 2 dimensions")
    if not 1 <= n_neighbors < adata.n_obs:
        raise ValueError("n_neighbors must be between 1 and n_obs - 1")
    return np.asarray(cluster_values), np.asarray(plot_values)


def _build_neighbor_edges(values: np.ndarray, n_neighbors: int) -> list[tuple[int, int]]:
    graph = kneighbors_graph(
        values,
        n_neighbors=n_neighbors,
        mode="connectivity",
        include_self=False,
    )
    sources, targets = graph.nonzero()
    return list(zip(sources.tolist(), targets.tolist(), strict=True))


def _cluster_graph(
    method: str,
    n_cells: int,
    edges: list[tuple[int, int]],
    resolution: float,
    random_state: int,
    kwargs: dict,
) -> np.ndarray:
    if method == "leiden":
        return _leiden_labels(n_cells, edges, resolution, random_state, kwargs)
    return _louvain_labels(n_cells, edges, resolution, random_state, kwargs)


def _leiden_labels(
    n_cells: int,
    edges: list[tuple[int, int]],
    resolution: float,
    random_state: int,
    kwargs: dict,
) -> np.ndarray:
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise ImportError(
            "Leiden clustering requires igraph and leidenalg. "
            "Install the Conda packages python-igraph and leidenalg."
        ) from exc

    graph = ig.Graph(n=n_cells, edges=edges, directed=False)
    graph.simplify()
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=random_state,
        **kwargs,
    )
    return np.array(partition.membership, dtype=int)


def _louvain_labels(
    n_cells: int,
    edges: list[tuple[int, int]],
    resolution: float,
    random_state: int,
    kwargs: dict,
) -> np.ndarray:
    try:
        import networkx as nx
        from networkx.algorithms.community import louvain_communities
    except ImportError as exc:
        raise ImportError("Louvain clustering requires the Conda package networkx.") from exc

    graph = nx.Graph()
    graph.add_nodes_from(range(n_cells))
    graph.add_edges_from(edges)
    communities = louvain_communities(
        graph,
        resolution=resolution,
        seed=random_state,
        **kwargs,
    )
    labels = np.empty(n_cells, dtype=int)
    for label, community in enumerate(communities):
        for node in community:
            labels[node] = label
    return labels


def _save_cluster_plot(
    values: np.ndarray,
    labels: np.ndarray,
    output_path,
    method: str,
    figsize,
    marker_size: float,
    cmap: str,
) -> None:
    try:
        import seaborn as sns
    except ImportError as exc:
        raise ImportError("Cluster plotting requires the optional seaborn package.") from exc

    fig, ax = plt.subplots(figsize=figsize)
    unique_labels = np.unique(labels)
    palette = sns.color_palette(cmap, n_colors=len(unique_labels))
    for color, label in zip(palette, unique_labels, strict=True):
        mask = labels == label
        ax.scatter(
            values[mask, 0],
            values[mask, 1],
            s=marker_size,
            color=color,
            linewidths=0,
            alpha=0.85,
        )
        ax.text(
            np.median(values[mask, 0]),
            np.median(values[mask, 1]),
            str(label),
            fontsize=12,
            ha="center",
            va="center",
            weight="bold",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.7},
        )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"{method.capitalize()} clustering")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
