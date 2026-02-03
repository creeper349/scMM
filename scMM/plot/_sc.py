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

class PseudotimeEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        fig_path_dir:str,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        name="pseudotime"
    ):
        self.adata = ad.AnnData(
            X=df.values,
            obs=obs.copy() if obs is not None else pd.DataFrame(index=df.index),
            var=var.copy() if var is not None else pd.DataFrame(index=df.columns)
        )
        self.adata.uns["name"] = name
        self.path = fig_path_dir
        
    def set_root_by_index(self, k, key="is_root"):
        self.adata.obs[key] = False
        self.adata.obs.iloc[k, self.adata.obs.columns.get_loc(key)] = True
        return self

    def build_graph(
        self,
        n_neighbors=80,
        n_pcs = 30,
        metric="euclidean"
    ):
        sc.tl.pca(self.adata, n_comps=n_pcs)
        sc.pp.neighbors(
            self.adata,
            n_neighbors=n_neighbors,
            use_rep="X",
            metric=metric,
            random_state=42
        )
        sc.tl.umap(self.adata, min_dist=0.1, spread=1.0, random_state=42)
        return self
    
    def run_diffmap(
        self,
        n_comps = 10
    ):
        sc.tl.diffmap(self.adata, n_comps=n_comps)
        sc.pl.diffmap(self.adata, components="1,2", color="palantir_pseudotime", cmap = "Spectral")
        plt.savefig(os.path.join(self.path, "diffmap.svg"))
        
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

    def run_paga(
        self,
        cluster_key="leiden",
        resolution=1.0
    ):
        sc.tl.leiden(self.adata, resolution=resolution, key_added=cluster_key)
        sc.tl.paga(self.adata, groups=cluster_key)
        sc.pl.paga(self.adata)
        plt.savefig(os.path.join(self.path, "paga.svg"))
        return self

    def plot_branches(self, eps = 0.05, q = 0.01):
        masks = palantir.presults.select_branch_cells(self.adata, eps = eps, q = q)
        palantir.plot.plot_branch_selection(self.adata)
        plt.savefig(os.path.join(self.path, "palantir_branches.svg"))
        
    def plot_umap(self, key:str, color):
        sc.tl.umap(self.adata)
        self.adata.obs[key] = color
        sc.pl.umap(self.adata, color = key, legend_loc = 'on data')
        plt.savefig(os.path.join(self.path, "umap.svg"))

    def plot_trajectory(self, save_name="umap_trajectory.svg"):

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
            masks = palantir.presults.select_branch_cells(self.adata, eps=0.01, q=0.01)

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
        
    def plot_trend_clusters(
        self,
        n_clusters=6,
        pseudotime_key="palantir_pseudotime"
    ):
        
        gene_trends = palantir.presults.compute_gene_trends(self.adata)
        clust = palantir.presults.cluster_gene_trends(self.adata, "2073")
        palantir.plot.plot_gene_trend_clusters(self.adata, branch_name="2073")
        plt.savefig(os.path.join(self.path, "trends_clust.svg"))
        plt.close()

    def get_anndata(self):
        return self.adata