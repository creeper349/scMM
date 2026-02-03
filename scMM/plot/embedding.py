import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.cluster import DBSCAN
from umap import UMAP
from typing import Optional
from ..file.data import CyESIData

DIM_REGISTRY = {}

def register_dim(name):
    def wrapper(func):
        DIM_REGISTRY[name] = func
        return func
    return wrapper

@register_dim("umap")
def run_umap(X, dim):

    params = {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42
    }
    params.update(dim)

    model = UMAP(**params)
    return model.fit_transform(X)

@register_dim("pca")
def run_pca(X, dim):

    params = {
        "n_components": 2,
        "svd_solver": "auto"
    }
    params.update(dim)

    return PCA(**params).fit_transform(X)

@register_dim("isomap")
def run_isomap(X, dim):

    params = {
        "n_components": 2,
        "n_neighbors": 15,
        "metric": "euclidean",
        "path_method": "auto"
    }
    params.update(dim)

    return Isomap(**params).fit_transform(X)

@register_dim("tsne")
def run_tsne(X, dim):

    params = {
        "n_components": 2,
        "perplexity": 30,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": 42
    }
    params.update(dim)

    return TSNE(**params).fit_transform(X)

@register_dim("lle")
def run_LLE(X, dim):

    params = {
        "n_components": 2,
        "n_neighbors": 15,
        "method": "standard"
    }
    params.update(dim)

    return LocallyLinearEmbedding(**params).fit_transform(X)

def _run_dim_reduction(X, method, dim):
    if method not in DIM_REGISTRY:
        raise ValueError(f"Unknown method: {method}")

    return DIM_REGISTRY[method](X, dim)

def dimension_reduction(data:CyESIData, 
                        method:str = "pca",
                        ax: plt.Axes = None,
                        reduce_kwargs:dict = None,
                        color:np.ndarray | str | None = "categorical",
                        categorical_mapping: Optional[dict] = None,
                        cluster_kwargs:dict = None, 
                        plot_kwargs:dict = None):
    """
    A function interface for unsupervised dimension reduction and visualization.
    
    :param data: Processed CyESI single cell metabolism data
    :type data: CyESIData
    :param method: Dimension reduction method, could be "pca", "umap", "tsne", "isomap", "lle".
    if other methods are used, use @register_dim(method:str) to register first.
    :type method: str
    :param ax: Matplotlib figure axis for drawing. if None, a new one will be generated.
    :type ax: plt.Axes
    :param reduce_kwargs: Parameters for dimension reduction.
    :type reduce_kwargs: dict
    :param color: Method of coloring points. Using np.ndarray for continous values, "categorical" for
    coloring with file names, "cluster" for unsupervised clustering and None for single color.
    :type color: np.ndarray | str | None
    :param categorical_mapping: Mapping to transfer file names to class labels, 
    e.g. {"file_1":positive, "file_2":negative}
    :type categorical_mapping: Optional[dict]
    :param cluster_kwargs: Parameters for unsupervised clustering. Use only when color = "cluster".
    :type cluster_kwargs: dict
    :param plot_kwargs: Parameters for plotting.
    :type plot_kwargs: dict
    """
    reduce_kwargs = reduce_kwargs or {}
    plot_kwargs = plot_kwargs or {}
    X = data.data
    emb = _run_dim_reduction(X, method, reduce_kwargs)
    
    if ax is None:
        fig, ax = plt.subplots(
            figsize=plot_kwargs.get("figsize", (6, 6))
        )

    x = emb[:, 0]
    y = emb[:, 1]
    s = plot_kwargs.get("s", 8)
    alpha = plot_kwargs.get("alpha", 0.8)
    if isinstance(color, str):
        if color == "categorical":
            classes = data.get_labels(categorical_mapping)
            classes = np.asarray(classes)
        else:
            cluster_kwargs = cluster_kwargs or {}
            classes = cluster_kwargs.get("method", DBSCAN)(**cluster_kwargs).fit_predict(X)
            classes = np.asarray(classes)

        uniq = np.unique(classes)
        cmap = plt.get_cmap(plot_kwargs.get("palette", "tab10"))

        for i, u in enumerate(uniq):
            mask = classes == u
            ax.scatter(
                x[mask],
                y[mask],
                s=s,
                alpha=alpha,
                label=str(u),
                color=cmap(i % cmap.N)
            )

        if plot_kwargs.get("legend", True):
            ax.legend(
                title=plot_kwargs.get("legend_title", "Class"),
                markerscale=2,
                frameon=False
            )

    elif isinstance(color, np.ndarray):
        cmap = plot_kwargs.get("cmap", "viridis")

        sc = ax.scatter(
            x,
            y,
            c=color,
            s=s,
            alpha=alpha,
            cmap=cmap,
            vmin=plot_kwargs.get("vmin"),
            vmax=plot_kwargs.get("vmax")
        )

        if plot_kwargs.get("colorbar", True):
            plt.colorbar(sc, ax=ax)

    else:
        ax.scatter(x, y, s=s, alpha=alpha)

    ax.set_xlabel(f"{method.upper()}-1")
    ax.set_ylabel(f"{method.upper()}-2")

    if "title" in plot_kwargs:
        ax.set_title(plot_kwargs["title"])

    ax.set_aspect("equal", adjustable="datalim")

    return {
        "X_emb": emb,
        "method": method,
        "reduce_params": reduce_kwargs,
        "plot_params": plot_kwargs,
        "ax": ax
    }