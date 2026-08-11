"""PCA and UMAP capabilities used by :class:`PlotEngine`."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class EmbeddingMixin:
    """Provide dimensionality-reduction methods for a PlotEngine-like object."""

    def pca(
        self,
        n_components: int = 50,
        scale: bool = True,
        zero_center: bool = True,
        random_state: int = 42,
        store_key: str = "X_pca",
        return_model: bool = False,
    ):
        """Compute PCA and store both coordinates and reproducibility metadata."""
        values = self._get_X().copy()
        if scale or zero_center:
            values = StandardScaler(with_mean=zero_center, with_std=scale).fit_transform(values)

        n_components = min(n_components, values.shape[0], values.shape[1])
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        model = PCA(n_components=n_components, random_state=random_state)
        embedding = model.fit_transform(values)
        self.adata.obsm[store_key] = embedding
        self.adata.uns[f"{store_key}_params"] = {
            "source": "X",
            "n_components": int(n_components),
            "scale": bool(scale),
            "zero_center": bool(zero_center),
            "random_state": int(random_state),
            "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
        }
        return (embedding, model) if return_model else embedding

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
        """Compute UMAP from raw values or an existing/generated PCA embedding."""
        import umap

        if n_components < 1:
            raise ValueError("n_components must be at least 1")
        if not 2 <= n_neighbors < self.adata.n_obs:
            raise ValueError("n_neighbors must be between 2 and n_obs - 1")
        values, source = self._select_umap_input(
            use_pca,
            pca_key,
            pca_n_components,
            scale_before_pca,
            random_state,
        )
        embedding = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        ).fit_transform(values)
        self.adata.obsm[store_key] = embedding
        self.adata.uns[f"{store_key}_params"] = {
            "source": source,
            "n_components": int(n_components),
            "n_neighbors": int(n_neighbors),
            "min_dist": float(min_dist),
            "metric": metric,
            "random_state": int(random_state),
        }
        return embedding

    def _select_umap_input(
        self,
        use_pca: bool,
        pca_key: str,
        pca_n_components: int,
        scale_before_pca: bool,
        random_state: int,
    ) -> tuple[np.ndarray, str]:
        if not use_pca:
            return self._get_X(), "X"
        if pca_key in self.adata.obsm:
            return np.asarray(self.adata.obsm[pca_key], dtype=float), f"obsm:{pca_key}"

        values = self._get_X().copy()
        if scale_before_pca:
            values = StandardScaler(with_mean=True, with_std=True).fit_transform(values)
        n_components = min(pca_n_components, values.shape[0], values.shape[1])
        if n_components < 1:
            raise ValueError("pca_n_components must be >= 1")
        embedding = PCA(n_components=n_components, random_state=random_state).fit_transform(values)
        return embedding, f"X->PCA({n_components})"
