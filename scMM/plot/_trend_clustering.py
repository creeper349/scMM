"""Clustering algorithms for feature-by-time trend matrices."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, KMeans

SUPPORTED_METRICS = {"cosine", "correlation", "euclidean", "sqeuclidean"}
SUPPORTED_METHODS = {"leiden", "louvain", "agglomerative", "kmeans"}


def trend_cluster(
    trends: np.ndarray,
    metric: str = "correlation",
    cluster_method: str = "leiden",
    **kwargs,
) -> np.ndarray:
    """Cluster feature trends while marking non-finite features with ``-1``."""
    trends = np.asarray(trends, dtype=float)
    if trends.ndim != 2:
        raise ValueError("trends must be a 2D array of shape (n_points, n_features)")
    total_features = trends.shape[1]
    if total_features < 2:
        return np.zeros(total_features, dtype=int)

    features, valid_rows, early_result = _prepare_feature_trends(trends)
    if early_result is not None:
        return early_result
    metric, cluster_method = _validate_clustering_options(metric, cluster_method)
    distances = _pairwise_distances(features, metric)
    labels = _cluster_valid_features(features, distances, metric, cluster_method, kwargs)
    result = np.full(total_features, -1, dtype=int)
    result[valid_rows] = labels
    return result


def _prepare_feature_trends(
    trends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    features = trends.T.copy()
    valid_rows = np.isfinite(features).all(axis=1)
    if valid_rows.sum() == 0:
        raise ValueError("No valid feature trends found")
    if valid_rows.sum() == 1:
        result = np.full(features.shape[0], -1, dtype=int)
        result[valid_rows] = 0
        return features[valid_rows], valid_rows, result
    return features[valid_rows], valid_rows, None


def _validate_clustering_options(metric: str, method: str) -> tuple[str, str]:
    metric = metric.lower()
    method = method.lower()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"Unsupported metric: {metric}")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported cluster_method: {method}")
    return metric, method


def _pairwise_distances(features: np.ndarray, metric: str) -> np.ndarray:
    if metric == "sqeuclidean":
        distances = squareform(pdist(features, metric="euclidean")) ** 2
    else:
        distances = squareform(pdist(features, metric=metric))
    distances = np.asarray(distances, dtype=float)
    finite = np.isfinite(distances)
    distances[~finite] = np.nanmax(distances[finite]) if finite.any() else 1.0
    np.fill_diagonal(distances, 0.0)
    return distances


def _cluster_valid_features(
    features: np.ndarray,
    distances: np.ndarray,
    metric: str,
    method: str,
    kwargs: dict,
) -> np.ndarray:
    if method in {"leiden", "louvain"}:
        return _graph_cluster(distances, method, kwargs)
    if method == "agglomerative":
        return _agglomerative_cluster(features, distances, metric, kwargs)
    return _kmeans_cluster(features, metric, kwargs)


def _graph_cluster(distances: np.ndarray, method: str, kwargs: dict) -> np.ndarray:
    try:
        import igraph as ig
    except ImportError as exc:
        raise ImportError("igraph is required for leiden/louvain clustering") from exc

    n_features = distances.shape[0]
    n_neighbors = max(1, min(int(kwargs.get("n_neighbors", 10)), n_features - 1))
    edges = _nearest_neighbor_edges(distances, n_neighbors)
    if not edges:
        return np.zeros(n_features, dtype=int)
    weights = _distance_weights(distances, edges)
    graph = ig.Graph(n=n_features, edges=edges, directed=False)
    graph.es["weight"] = weights
    if method == "leiden":
        return _leiden_cluster(graph, float(kwargs.get("resolution", 1.0)))
    partition = graph.community_multilevel(weights=graph.es["weight"])
    return np.array(partition.membership, dtype=int)


def _nearest_neighbor_edges(distances: np.ndarray, n_neighbors: int) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for index in range(distances.shape[0]):
        neighbors = np.argsort(distances[index])[1 : n_neighbors + 1]
        edges.update(tuple(sorted((index, int(neighbor)))) for neighbor in neighbors)
    return sorted(edges)


def _distance_weights(distances: np.ndarray, edges: list[tuple[int, int]]) -> list[float]:
    values = np.array([distances[left, right] for left, right in edges], dtype=float)
    positive = values[values > 0]
    scale = np.median(positive) if len(positive) > 0 else 1.0
    if scale <= 0:
        scale = 1.0
    return np.exp(-values / scale).tolist()


def _leiden_cluster(graph, resolution: float) -> np.ndarray:
    try:
        import leidenalg
    except ImportError as exc:
        raise ImportError("leidenalg is required for leiden clustering") from exc
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=graph.es["weight"],
        resolution_parameter=resolution,
    )
    return np.array(partition.membership, dtype=int)


def _agglomerative_cluster(
    features: np.ndarray,
    distances: np.ndarray,
    metric: str,
    kwargs: dict,
) -> np.ndarray:
    n_clusters = _validated_cluster_count(kwargs, features.shape[0])
    linkage = kwargs.get("linkage", "average")
    if linkage not in {"average", "complete", "single", "ward"}:
        raise ValueError("linkage must be one of {'average', 'complete', 'single', 'ward'}")
    if linkage == "ward":
        if metric not in {"euclidean", "sqeuclidean"}:
            raise ValueError("ward linkage requires euclidean-like metric")
        return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(features)
    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage=linkage,
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage=linkage,
        )
    return model.fit_predict(distances)


def _kmeans_cluster(features: np.ndarray, metric: str, kwargs: dict) -> np.ndarray:
    n_clusters = _validated_cluster_count(kwargs, features.shape[0])
    normalized = features.copy()
    if bool(kwargs.get("normalize", metric in {"cosine", "correlation"})):
        if metric == "correlation":
            normalized -= normalized.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(normalized, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized /= norms
    return KMeans(
        n_clusters=n_clusters,
        random_state=int(kwargs.get("random_state", 0)),
        n_init=20,
    ).fit_predict(normalized)


def _validated_cluster_count(kwargs: dict, n_features: int) -> int:
    n_clusters = int(kwargs.get("n_clusters", 5))
    if not 1 <= n_clusters <= n_features:
        raise ValueError("n_clusters must be between 1 and the number of valid features")
    return n_clusters
