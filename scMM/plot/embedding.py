import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding

from ..file.data import CyESIData

DIM_REGISTRY = {}


def register_dim(name):
    def wrapper(func):
        if name in DIM_REGISTRY:
            raise ValueError(f"Dimension reduction method already registered: {name}")
        DIM_REGISTRY[name] = func
        return func

    return wrapper


@register_dim("umap")
def run_umap(X, dim):
    from umap import UMAP

    params = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42,
    }
    params.update(dim)

    model = UMAP(**params)
    return model.fit_transform(X)


@register_dim("pca")
def run_pca(X, dim):

    params = {"n_components": 2, "svd_solver": "auto"}
    params.update(dim)

    return PCA(**params).fit_transform(X)


@register_dim("isomap")
def run_isomap(X, dim):

    params = {"n_components": 2, "n_neighbors": 15, "metric": "euclidean", "path_method": "auto"}
    params.update(dim)

    return Isomap(**params).fit_transform(X)


@register_dim("tsne")
def run_tsne(X, dim):

    params = {
        "n_components": 2,
        "perplexity": 30,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": 42,
    }
    params.update(dim)

    return TSNE(**params).fit_transform(X)


@register_dim("lle")
def run_LLE(X, dim):

    params = {"n_components": 2, "n_neighbors": 15, "method": "standard"}
    params.update(dim)

    return LocallyLinearEmbedding(**params).fit_transform(X)


def _run_dim_reduction(X, method, dim):
    method = method.lower()
    if method not in DIM_REGISTRY:
        raise ValueError(f"Unknown method: {method}")

    return DIM_REGISTRY[method](X, dim)


def _as_feature_matrix(data: CyESIData | np.ndarray) -> np.ndarray:
    values = data.data if isinstance(data, CyESIData) else data
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError("data must be a non-empty 2D matrix")
    return matrix


def _validate_embedding(embedding, n_observations: int) -> np.ndarray:
    embedding = np.asarray(embedding)
    if embedding.ndim != 2 or embedding.shape[0] != n_observations or embedding.shape[1] < 2:
        raise ValueError("Dimension reduction must return at least two components per observation")
    return embedding


def _resolve_categorical_classes(
    data: CyESIData | np.ndarray,
    matrix: np.ndarray,
    color: str,
    categorical_mapping: dict | None,
    cluster_kwargs: dict | None,
) -> np.ndarray:
    if color == "categorical":
        if not isinstance(data, CyESIData):
            raise TypeError("categorical coloring requires a CyESIData instance")
        classes = data.get_labels(categorical_mapping)
    elif color == "cluster":
        options = dict(cluster_kwargs or {})
        cluster_method = options.pop("method", DBSCAN)
        model = cluster_method(**options) if isinstance(cluster_method, type) else cluster_method
        classes = model.fit_predict(matrix)
    else:
        raise ValueError("color must be 'categorical', 'cluster', an array, or None")

    classes = np.asarray(classes)
    if classes.ndim != 1 or len(classes) != len(matrix):
        raise ValueError("categorical colors must contain one label per observation")
    return classes


def _scatter_categories(
    ax: plt.Axes,
    embedding: np.ndarray,
    classes: np.ndarray,
    plot_kwargs: dict,
) -> None:
    cmap = plt.get_cmap(plot_kwargs.get("palette", "tab10"))
    for index, label in enumerate(np.unique(classes)):
        mask = classes == label
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=plot_kwargs.get("s", 8),
            alpha=plot_kwargs.get("alpha", 0.8),
            label=str(label),
            color=cmap(index % cmap.N),
        )

    if plot_kwargs.get("legend", True):
        ax.legend(title=plot_kwargs.get("legend_title", "Class"), markerscale=2, frameon=False)


def _scatter_continuous(
    ax: plt.Axes,
    embedding: np.ndarray,
    color: np.ndarray,
    plot_kwargs: dict,
) -> None:
    if color.ndim != 1 or len(color) != len(embedding):
        raise ValueError("color array must have one value per observation")
    points = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=color,
        s=plot_kwargs.get("s", 8),
        alpha=plot_kwargs.get("alpha", 0.8),
        cmap=plot_kwargs.get("cmap", "viridis"),
        vmin=plot_kwargs.get("vmin"),
        vmax=plot_kwargs.get("vmax"),
    )
    if plot_kwargs.get("colorbar", True):
        plt.colorbar(points, ax=ax)


def _plot_embedding(
    ax: plt.Axes,
    data: CyESIData | np.ndarray,
    matrix: np.ndarray,
    embedding: np.ndarray,
    color: np.ndarray | str | None,
    categorical_mapping: dict | None,
    cluster_kwargs: dict | None,
    plot_kwargs: dict,
) -> None:
    if isinstance(color, str):
        classes = _resolve_categorical_classes(
            data, matrix, color, categorical_mapping, cluster_kwargs
        )
        _scatter_categories(ax, embedding, classes, plot_kwargs)
    elif isinstance(color, np.ndarray):
        _scatter_continuous(ax, embedding, color, plot_kwargs)
    else:
        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            s=plot_kwargs.get("s", 8),
            alpha=plot_kwargs.get("alpha", 0.8),
        )


def _configure_embedding_axes(ax: plt.Axes, method: str, plot_kwargs: dict) -> None:
    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")
    if "title" in plot_kwargs:
        ax.set_title(plot_kwargs["title"])
    ax.set_aspect("equal", adjustable="datalim")


def dimension_reduction(
    data: CyESIData | np.ndarray,
    method: str = "pca",
    ax: plt.Axes | None = None,
    reduce_kwargs: dict | None = None,
    color: np.ndarray | str | None = "categorical",
    categorical_mapping: dict | None = None,
    cluster_kwargs: dict | None = None,
    plot_kwargs: dict | None = None,
):
    """Reduce a feature matrix and draw its first two embedding components.

    ``data`` may be a :class:`CyESIData` instance or a non-empty 2D NumPy-compatible
    matrix. Built-in methods are ``pca``, ``umap``, ``tsne``, ``isomap``, and ``lle``;
    additional methods can be added with :func:`register_dim`.

    ``color`` accepts a one-dimensional NumPy array for continuous values,
    ``"categorical"`` for dataset labels, ``"cluster"`` for DBSCAN or a compatible
    custom clusterer, and ``None`` for a single color. The returned dictionary contains
    the embedding, normalized method name, parameter mappings, and Matplotlib axis.
    """
    reduce_kwargs = reduce_kwargs or {}
    plot_kwargs = plot_kwargs or {}
    X = _as_feature_matrix(data)
    method = method.lower()
    emb = _validate_embedding(_run_dim_reduction(X, method, reduce_kwargs), len(X))

    if ax is None:
        _fig, ax = plt.subplots(figsize=plot_kwargs.get("figsize", (6, 6)))

    _plot_embedding(
        ax,
        data,
        X,
        emb,
        color,
        categorical_mapping,
        cluster_kwargs,
        plot_kwargs,
    )
    _configure_embedding_axes(ax, method, plot_kwargs)

    return {
        "X_emb": emb,
        "method": method,
        "reduce_params": reduce_kwargs,
        "plot_params": plot_kwargs,
        "ax": ax,
    }
