import os
import palantir
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from typing import Optional, Literal
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from contextlib import contextmanager
from .embedding import dimension_reduction
from ..util.normalize import normalize
from ..file._anndata import to_anndata
from ..file.data import CyESIData

class PseudotimeEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        fig_path_dir:str,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
    ):
        self.adata = ad.AnnData(
            X=df.values,
            obs=obs.copy() if obs is not None else pd.DataFrame(index=df.index),
            var=var.copy() if var is not None else pd.DataFrame(index=df.columns)
        )
        self._init_core(fig_path_dir)
        
    @classmethod
    def from_adata(cls, hdf_path:str, fig_path_dir:str):
        obj = object.__new__(cls)
        obj.adata = ad.read_h5ad(hdf_path)
        obj._init_core(fig_path_dir)
        return obj
    
    @classmethod
    def from_CyESIData(cls, data:CyESIData, fig_path_dir:str):
        obj = object.__new__(cls)
        obj.adata = to_anndata(data)
        obj._init_core(fig_path_dir)
        return obj
    
    def _init_core(self, fig_path_dir:str):
        self.path = fig_path_dir
        self.rep_key = "X"
        self.layer_key = None
        
    def use_layer(self, key: str):
        if key not in self.adata.layers:
            raise KeyError(f"Layer {key} not found")
        self.layer_key = key
        self.rep_key = "X"
        return self
    
    def use_rep(self, key: str):
        if key not in self.adata.obsm:
            raise KeyError(f"Rep {key} not found in obsm")
        self.rep_key = key
        self.layer_key = None
        return self
    
    @contextmanager
    def _swap_X(self):
        if self.layer_key is None:
            yield
            return
        X_orig = self.adata.X
        self.adata.X = self.adata.layers[self.layer_key]
        try:
            yield
        finally:
            self.adata.X = X_orig
            
    def normalize(self, method = "total", norm_kwargs: dict = None):
        X_norm_results = normalize(
            self.adata.X,
            method=method,
            norm_kwargs=norm_kwargs,
            return_params=True
        )
        self.adata.layers[f"norm_{method}"] = X_norm_results["X_norm"]
        self.adata.uns[f"{method}_norm_params"] = X_norm_results["norm_params"]
        return self
    
    def decomposition(self,
                      method:str = "pca",
                      ax: plt.Axes = None,
                      reduce_kwargs:dict = None,
                      color_key:str = None,
                      plot_kwargs:dict = None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        X = (
            self.adata.obsm[self.rep_key]
            if self.rep_key != "X"
            else self.adata.X
        )
        result_dict = dimension_reduction(X, method=method, ax=ax, 
                                          reduce_kwargs=reduce_kwargs, 
                                          color=self.adata.obs[color_key].values if color_key else None, 
                                          plot_kwargs=plot_kwargs)
        self.adata.obsm[f"X_{method}"] = result_dict["X_emb"]
        self.adata.uns[f"{method}_params"] = result_dict["reduce_params"]
        plt.savefig(os.path.join(self.path, f"{method}_decomposition.svg"))
        plt.close()
        return self
        
    def set_root_by_index(self, k, key="is_root"):
        self.adata.obs[key] = False
        self.adata.obs.iloc[k, self.adata.obs.columns.get_loc(key)] = True
        return self

    def build_graph(self, n_neighbors=50, metric="euclidean"):
        with self._swap_X():
            sc.pp.neighbors(
                self.adata,
                n_neighbors=n_neighbors,
                use_rep=None if self.rep_key == "X" else self.rep_key,
                metric=metric,
                random_state=42
            )
        return self
    
    def diffmap(self, n_comps = 10):
        with self._swap_X():
            sc.tl.diffmap(self.adata, n_comps=n_comps)
        sc.pl.diffmap(self.adata, components="1,2", color="palantir_pseudotime", cmap = "Spectral")
        plt.savefig(os.path.join(self.path, "diffmap.svg"))
        return self
        
    def highlight_cells(self, cells:dict):
        palantir.plot.highlight_cells_on_umap(self.adata, cells)
        plt.savefig(os.path.join(self.path, "highlighted_cells.svg"))
        plt.close()

    def run_palantir(
        self,
        root_key=None,
        root_value=None,
        key="palantir_pseudotime"
    ):

        if root_key:
            start_cell = self.adata.obs.index[
                self.adata.obs[root_key] == root_value
            ][0]
        else:
            start_cell = self.adata.obs.index[0]
            
        with self._swap_X():
            dm_res = palantir.utils.run_diffusion_maps(self.adata, n_components=10)
            ms_data = palantir.utils.determine_multiscale_space(self.adata)
            palantir.plot.plot_diffusion_components(self.adata, s=2)
            plt.savefig(os.path.join(self.path, "palantir_diffmap.svg"))

            pr_res = palantir.core.run_palantir(
                self.adata,
                start_cell
            )
            self.adata.obs['palantir_pseudotime'] = pr_res.pseudotime
            
            palantir.plot.plot_palantir_results(self.adata, pr_res, s=2)
            plt.savefig(os.path.join(self.path, "umap_palantir.svg"))
            plt.close()

            self.adata.obs[key] = pr_res.pseudotime
            self.adata.obsm["palantir_branch_probs"] = pr_res.branch_probs
        return self

    def plot_trajectory(self, branch_name:str, **kwargs):
        with self._swap_X():
            palantir.plot.plot_trajectory(self.adata, branch=branch_name, 
                                          cell_color = "palantir_pseudotime",
                                          color = kwargs.get("color", "red"),
                                          scanpy_kwargs=dict(cmap=kwargs.get("cmap", "viridis")))
        plt.savefig(os.path.join(self.path, f"trajectory_branch_{branch_name}.svg"))
        plt.close()
        return self

    def plot_branches(self, save_name="umap_trajectory.svg", eps = 0.05, q = 0.01):

        branch_probs = self.adata.obsm["palantir_branch_probs"]

        if "palantir_fate_probabilities" not in self.adata.obsm:
            fate_columns = [f"Branch {i}" for i in range(branch_probs.shape[1])]
            fate_probs_df = pd.DataFrame(
                branch_probs,
                index=self.adata.obs_names,
                columns=fate_columns
            )
            self.adata.obsm["palantir_fate_probabilities"] = fate_probs_df
        else:
            fate_columns = list(self.adata.obsm["palantir_fate_probabilities"].columns)

        if "branch_masks" not in self.adata.obsm:
            masks = palantir.presults.select_branch_cells(self.adata, eps=eps, q=q)

            masks_df = pd.DataFrame(
                masks,
                index=self.adata.obs_names,
                columns=fate_columns 
            )

            self.adata.obsm["branch_masks"] = masks_df
            self.adata.uns["branch_masks_columns"] = fate_columns

        plt.figure(figsize=(6, 6))
        palantir.plot.plot_branch_selection(self.adata)
        plt.title("Palantir Pseudotime Trajectory")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, save_name))
        plt.close()
        return self
        
    def plot_trend_clusters(
        self,
        branch_name:str,
        trends_key:str = "gene_trends",
        cluster_key:str = "feature_cluster"
    ):
        with self._swap_X():
            palantir.presults.compute_gene_trends(self.adata, gene_trend_key=trends_key)
            clust = palantir.presults.cluster_gene_trends(self.adata, branch_name)
            self.adata.var[cluster_key] = clust
            palantir.plot.plot_gene_trend_clusters(self.adata, branch_name=branch_name)
            plt.savefig(os.path.join(self.path, f"trends_clust_branch_{branch_name}.svg"))
            plt.close()
        return self
    
    def plot_trends(self, feature_names:list, heatmap:False):      
        with self._swap_X():
            if heatmap:
                palantir.plot.plot_gene_trend_heatmaps(self.adata, feature_names)
                axs = plt.gcf().get_axes()
                for ax in axs:
                    ax.yaxis.grid(False)
            else:
                palantir.plot.plot_gene_trends(self.adata, feature_names)
        plt.savefig(os.path.join(self.path, "feature_trends.svg"))
        plt.close()
        return self
    
    def save_adata(self, path:str):
        self.adata.write(path)