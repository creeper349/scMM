"""Feature-network computation and rendering for :class:`PlotEngine`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances


class FeatureNetworkMixin:
    """Provide feature-network embedding for a PlotEngine-like object."""

    def feature_network(
        self,
        name_key: str | None = None,
        class_key: str | None = None,
        metric: str = "pearson",
        **kwargs,
    ):
        """Embed pairwise feature distances and save a labeled network plot."""
        if metric not in {"pearson", "euclidean"}:
            raise ValueError("metric must be 'pearson' or 'euclidean'")
        values, names, feature_mask = _select_named_features(self.adata, self._get_X(), name_key)
        if values.shape[1] < 3:
            raise ValueError("At least three named features are required for a feature network")
        distances = _feature_distances(values, metric)
        embedding = _embed_feature_distances(distances, kwargs)
        colors, cmap = _feature_colors(self.adata, names, feature_mask, class_key, kwargs)
        _save_feature_network(
            embedding,
            names,
            colors,
            cmap,
            self.path / f"feature_network_{metric}.svg",
            kwargs,
        )
        return embedding


def _select_named_features(adata, values: np.ndarray, name_key: str | None):
    feature_mask = np.ones(adata.n_vars, dtype=bool)
    if name_key is None or name_key not in adata.var.columns:
        return values, adata.var_names.to_numpy(copy=True), feature_mask

    names = adata.var[name_key].to_numpy(copy=True)
    invalid = pd.isna(names) | (pd.Series(names).astype(str).str.strip() == "").values
    feature_mask = ~invalid
    return values[:, feature_mask], names[feature_mask], feature_mask


def _feature_distances(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == "pearson":
        correlations = np.corrcoef(values.T)
        np.fill_diagonal(correlations, 1)
        distances = np.sqrt(2 * (1 - np.abs(correlations)))
    else:
        distances = pairwise_distances(values.T, metric="euclidean")
    return np.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0)


def _embed_feature_distances(distances: np.ndarray, kwargs: dict) -> np.ndarray:
    import umap

    n_features = distances.shape[0]
    n_neighbors = int(kwargs.get("n_neighbors", min(15, n_features - 1)))
    if not 2 <= n_neighbors < n_features:
        raise ValueError("n_neighbors must be between 2 and the number of features minus 1")
    return umap.UMAP(
        metric="precomputed",
        n_neighbors=n_neighbors,
        min_dist=kwargs.get("min_dist", 0.1),
        random_state=kwargs.get("random_state", 42),
    ).fit_transform(distances)


def _feature_colors(adata, names, feature_mask, class_key: str | None, kwargs: dict):
    if np.issubdtype(np.asarray(names).dtype, np.number):
        return np.asarray(names, dtype=float), kwargs.get("cmap", "viridis")
    if class_key is not None and class_key in adata.var.columns:
        colors = adata.var.loc[feature_mask, class_key].astype("category").cat.codes
    else:
        colors = np.zeros(len(names))
    return colors, kwargs.get("cmap", "tab20")


def _save_feature_network(
    embedding: np.ndarray,
    names,
    colors,
    cmap: str,
    output_path,
    kwargs: dict,
) -> None:
    fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
    ax.scatter(embedding[:, 0], embedding[:, 1], s=10, c=colors, cmap=cmap)
    for index, name in enumerate(names):
        ax.text(
            embedding[index, 0],
            embedding[index, 1],
            str(name),
            fontsize=kwargs.get("fontsize", 8),
            alpha=kwargs.get("alpha", 0.7),
        )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
