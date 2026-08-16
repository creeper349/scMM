"""High-level AnnData container that composes focused analysis capabilities."""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from ._engine_clustering import CellClusteringMixin
from ._engine_embedding import EmbeddingMixin
from ._engine_network import FeatureNetworkMixin
from ._engine_trajectory import TrajectoryMixin


class PlotEngine(EmbeddingMixin, TrajectoryMixin, CellClusteringMixin, FeatureNetworkMixin):
    """AnnData-backed entry point for analysis and figure generation.

    The public methods are grouped into private mixins by domain.  This class
    owns only shared data conversion and output-path state.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fig_path_dir: str | Path,
        obs: pd.DataFrame | None = None,
        var: pd.DataFrame | None = None,
    ):
        obs_frame = obs.copy() if obs is not None else pd.DataFrame(index=df.index)
        var_frame = var.copy() if var is not None else pd.DataFrame(index=df.columns)
        obs_frame.index = obs_frame.index.astype(str)
        var_frame.index = var_frame.index.astype(str)
        self.adata = ad.AnnData(X=df.values, obs=obs_frame, var=var_frame)
        self._initialize_state(fig_path_dir)

    @classmethod
    def from_adata(cls, adata: ad.AnnData, fig_path_dir: str | Path):
        """Build an engine around a defensive copy of an existing AnnData."""
        obj = object.__new__(cls)
        obj.adata = adata.copy()
        obj._initialize_state(fig_path_dir)
        return obj

    def _initialize_state(self, fig_path_dir: str | Path) -> None:
        self.adata.obs_names_make_unique()
        self.adata.var_names_make_unique()
        self.path = Path(fig_path_dir).expanduser()
        self.path.mkdir(parents=True, exist_ok=True)

    def _get_X(self) -> np.ndarray:
        values = self.adata.X
        if hasattr(values, "toarray"):
            values = values.toarray()
        return np.asarray(values, dtype=float)

    def _get_internal_cell_index(self) -> pd.Index:
        return pd.Index([f"cell_{index}" for index in range(self.adata.n_obs)], name="cell_id")

    def _get_X_df(self) -> pd.DataFrame:
        values = self._get_X()
        columns = (
            self.adata.var_names.copy()
            if values.shape[1] == self.adata.n_vars
            else [f"feature_{index}" for index in range(values.shape[1])]
        )
        return pd.DataFrame(values, index=self._get_internal_cell_index(), columns=columns)
