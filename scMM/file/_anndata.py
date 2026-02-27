import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import networkx as nx
from anndata import AnnData
from typing import Optional
from scipy.stats import zscore
from .data import CyESIData
from sklearn.covariance import GraphicalLasso
from sklearn.preprocessing import StandardScaler

def to_anndata(data:CyESIData):
    obs_df = pd.DataFrame({
        "cell_id": data.data.index,
        "labels": data.get_labels(),
        "time": data.get_time(),
        "width": data.peak_meta['width'].values,
        "symmetry": data.peak_meta['symmetry'].values
    })
    
    var_df = pd.DataFrame({
        "mz": data.data.columns
    })
    
    adata = AnnData(
        X=data.data.values,
        obs=obs_df.set_index("cell_id"),
        var=var_df.set_index("mz")
    )
    adata.raw = adata.copy()
    return adata

def _soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0)

class MetaboData(AnnData):
    def __init__(self, 
                 X: np.ndarray,
                 obs: Optional[pd.DataFrame] = None,
                 var: Optional[pd.DataFrame] = None,
                 is_inten: Optional[pd.DataFrame] = None,
                 process_X = lambda x: x,
                 **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(X=process_X(X), obs=obs, var=var, **kwargs)
        assert (is_inten is None) or (is_inten.shape[0] == X.shape[0])
        self.is_inten = is_inten
        
    @property
    def n_observations(self):
            return self.X.shape[0]
    @property
    def n_features(self):
            return self.X.shape[1]
        
    """
    def is_calibration(self, reg:float = 0):
        log_is = np.log1p(self.is_inten.values)
        is_central = log_is - log_is.mean(axis=0)
        proj_is = is_central @ \
            np.linalg.inv(is_central.T @ is_central + reg * np.eye(is_central.shape[1]))\
            @ is_central.T
        self.layers["X_calibrated"] = (np.eye(proj_is.shape[0]) - proj_is) @ self.X
        self.X = self.layers["X_calibrated"]
        self.uns["X_rep"] = "X_calibrated"
        return self
    """
    
    def gmis_calibration(
        self,
        n_pc: int = 1,
        reg: float = 1e-6,
        shrinkage: float = 1.0,
        is_cv_threshold:float = 0.2
    ):
        is_inten_cv = self.is_inten.std(axis=0) / (self.is_inten.mean(axis=0) + 1e-12)
        valid_is = is_inten_cv < is_cv_threshold
        logging.info(f"Using {valid_is.sum()} out of {len(valid_is)} IS for calibration (CV < {is_cv_threshold})")
        
        Z = np.log1p(self.is_inten.values[:, valid_is]).astype(np.float64)
        Z = Z - Z.mean(axis=0, keepdims=True)

        U, S, Vt = np.linalg.svd(Z, full_matrices=False)

        r = min(n_pc, U.shape[1])
        Z_sub = U[:, :r] * S[:r]

        P = Z_sub @ np.linalg.inv(
            Z_sub.T @ Z_sub + reg * np.eye(r)
        ) @ Z_sub.T

        X = np.log1p(self.X)
        #X_corr = X - shrinkage * (P @ X)
        X_corr = np.exp(X - shrinkage * (P @ X)) - 1
        
        self.layers["X_calibrated"] = X_corr
        self.X = X_corr
        self.uns["X_rep"] = "X_calibrated"

        return self
    
    def is_pqn_normalization(
        self,
        ref: str = "median", 
        eps: float = 1e-12
    ):

        Z = self.is_inten.values.astype(np.float64)

        if ref == "median":
            Z_ref = np.median(Z, axis=0, keepdims=True)
        elif ref == "mean":
            Z_ref = np.mean(Z, axis=0, keepdims=True)
        else:
            raise ValueError("ref must be 'median' or 'mean'")

        ratios = Z_ref / (Z_ref + eps)
        f = np.median(ratios, axis=1, keepdims=True)

        f = f + eps

        X_corr = self.X / f

        self.layers["X_pqn"] = X_corr
        self.obs["pqn_factor"] = f.squeeze()

        self.X = X_corr
        self.uns["X_rep"] = "X_pqn"

        return self
    
    def mz_calibration(self, mz_is_true: list|np.ndarray, 
                       mz_is_obs: list|np.ndarray = None):
        assert "mz" in self.var.columns
        mz_obs_is = np.asarray(mz_is_obs, dtype=float) if mz_is_obs is not None\
            else self.is_inten.columns.astype(float)
        mz_theory_is = np.asarray(mz_is_true, dtype=float)
        
        if mz_obs_is.shape != mz_theory_is.shape:
            raise ValueError("Observed and theoretical m/z must have same shape")
        
        A = np.vstack([mz_theory_is, np.ones_like(mz_theory_is)]).T
        a, b = np.linalg.lstsq(A, mz_obs_is, rcond=None)[0]
        
        mz_obs_all = self.var["mz"].values.astype(float)
        mz_calibrated = (mz_obs_all - b) / a
        self.var["mz_calibrated"] = mz_calibrated
        self.uns["mz_calibration_params"] = {"a": a, "b": b,
            "ppm_res": (mz_obs_is - (a * mz_theory_is + b)) / mz_theory_is * 1e6}
        return self
    
    def add_omics(self, omics_key:str, omics_data:pd.DataFrame):
        assert omics_data.shape[0] == self.n_observations, "Omics data must have same number of observations"
        self.uns[f"omics_{omics_key}"] = omics_data
        return self
    
    def max_cov_subspace(self, omics_key:str, n_components=2, 
                        lam_x=0.2, lam_y=0.2, n_iter=200):
        X = zscore(self.X, axis=0)
        Y = zscore(self.uns[f"omics_{omics_key}"], axis=0)
        n, p = X.shape
        _, q = Y.shape
        
        Wx = np.zeros((p, n_components))
        Wy = np.zeros((q, n_components))

        X_res = X.copy()
        Y_res = Y.copy()
        
        for comp in range(n_components):

            wx = np.random.randn(p)
            wx /= np.linalg.norm(wx)

            for _ in range(n_iter):

                wy = Y_res.T @ (X_res @ wx)
                wy = _soft_threshold(wy, lam_y)
                if np.linalg.norm(wy) > 0:
                    wy /= np.linalg.norm(wy)

                wx = X_res.T @ (Y_res @ wy)
                wx = _soft_threshold(wx, lam_x)
                if np.linalg.norm(wx) > 0:
                    wx /= np.linalg.norm(wx)

            Wx[:, comp] = wx
            Wy[:, comp] = wy

            X_res -= np.outer(X_res @ wx, wx)
            Y_res -= np.outer(Y_res @ wy, wy)

        Zx = X @ Wx
        Zy = Y @ Wy

        return Wx, Wy, Zx, Zy
    
    def plot_pca(self, n_components = 2, color_key:str = None, 
                 draw_confidence_ellipse:bool = False,
                 confidence_ellipse_scale:float = 2.447,
                 title:str = None, 
                 save_path:str = None, 
                 decomp_obj = None,
                 **kwargs):
        from sklearn.decomposition import PCA
        from matplotlib.patches import Ellipse
        pca = PCA(n_components=n_components)
        if decomp_obj is not None:
            assert decomp_obj.shape[0] == self.n_observations
            Z = pca.fit_transform(decomp_obj)
        else:
            Z = pca.fit_transform(self.X)
        plt.figure(figsize=(8, 6))
        if color_key is not None and color_key in self.obs.columns:
            groups = self.obs[color_key].unique()
            colors = plt.cm.get_cmap('tab20', len(groups))
            for i, group in enumerate(groups):
                idx = self.obs[color_key] == group
                plt.scatter(Z[idx, 0], Z[idx, 1], label=group, color=colors(i))
                if draw_confidence_ellipse:
                    cov = np.cov(Z[idx, :2].T)
                    mean = np.mean(Z[idx, :2], axis=0)
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    order = eigenvalues.argsort()[::-1]
                    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
                    angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                    width, height = 2 * confidence_ellipse_scale * np.sqrt(eigenvalues)
                    ellipse = Ellipse(xy=mean, width=width, height=height,
                                      angle=np.degrees(angle), edgecolor=colors(i),
                                      facecolor=colors(i), alpha = kwargs.get("alpha", 0.3))
                    plt.gca().add_patch(ellipse)
        else:
            plt.scatter(Z[:, 0], Z[:, 1])
        if title is not None: plt.title(title)
        if color_key is not None: plt.legend() 
        plt.savefig(save_path) if save_path is not None else plt.show()
        
    def volcano_plot(self, label_key:str, labelx:str, labely:str, 
                    title:str = None, save_path:str = None, name_key:str = None,
                    p_thr:float = 0.05, log2_fc_cutoff:float = 1.0, label_topk:int = 10, **kwargs):
        from scipy.stats import ttest_ind
        from adjustText import adjust_text
        data_x = self.X[self.obs[label_key] == labelx, :]
        data_y = self.X[self.obs[label_key] == labely, :]
        
        log2_fc = np.log2(np.mean(data_x, axis=0) / np.mean(data_y, axis=0))
        pvals = np.array([ttest_ind(data_x[:, i], data_y[:, i], equal_var=False).pvalue for i in range(data_x.shape[1])])
        neg_log10_p = -np.log10(pvals)
        significant_up = (log2_fc > log2_fc_cutoff) & (pvals < p_thr)
        significant_down = (log2_fc < -log2_fc_cutoff) & (pvals < p_thr)
        nonsig = ~(significant_up | significant_down)
        
        plt.figure(figsize=(9,7))

        plt.scatter(log2_fc[nonsig], neg_log10_p[nonsig], color="gray", alpha=0.5)
        plt.scatter(log2_fc[significant_up], neg_log10_p[significant_up], color="red", alpha=0.8, label="Upregulated")
        plt.scatter(log2_fc[significant_down], neg_log10_p[significant_down], color="blue", alpha=0.8, label="Downregulated")

        plt.axvline(log2_fc_cutoff, color="black", linestyle="--", alpha=0.5)
        plt.axvline(-log2_fc_cutoff, color="black", linestyle="--", alpha=0.5)
        plt.axhline(-np.log10(p_thr), color="black", linestyle="--", alpha=0.5)

        top_idx = np.argsort(pvals)[:label_topk]
        
        if name_key is not None and name_key in self.var.columns:
            texts = []
            for i in top_idx:
                texts.append(
                    plt.text(log2_fc[i], neg_log10_p[i], self.var[name_key][i], fontsize=10, color='black')
                )
                logging.info(f"Plot text of feature {self.var[name_key][i]} at ({log2_fc[i]:.2f}, {neg_log10_p[i]:.2f})")
            
            adjust_text(texts, 
                        arrowprops=dict(arrowstyle='-', color='black', lw=0.7),
                        only_move={'points':'y', 'text':'y'}) 
            
        plt.xlabel(f"log2 Fold Change ({labely} / {labelx})")
        plt.ylabel("-log10(p-value)")
        if title is not None: plt.title(title)
        plt.legend() 
        plt.savefig(save_path) if save_path is not None else plt.show()
        
    def plot_features(self, color_key:str, title:str = None, save_path:str = None, **kwargs):
        feature_mean = self.X.mean(axis=0)
        feature_cv = self.X.std(axis=0) / (feature_mean + 1e-12)
        plt.figure(figsize=(8, 6))
        unique_labels = self.var[color_key].unique()
        for label in unique_labels:
            idx = self.var[color_key] == label
            plt.scatter(feature_mean[idx], feature_cv[idx], label=label, alpha=kwargs.get("alpha", 0.7))
            
        plt.xlabel("Mean Ion Intensity")
        plt.ylabel("Coefficient of Variation")
        if title is not None: plt.title(title)
        plt.legend()
        plt.savefig(save_path) if save_path is not None else plt.show()

    def metabolite_network_plot(
        self,
        target_label: str = None,
        label_key: str = None,
        metabolite_name_key: str = None,
        metabolite_label_key: str = None,
        save_path: str = None,
        title: str = None,
        alpha=0.3,
        top_nodes=50,
        edge_threshold=0.01,
        figsize=(8, 8),
        layout_seed=0,
        method="glasso",
        **kwargs
    ):

        if target_label is None or label_key is None:
            X = self.X
        else:
            idx = self.obs[label_key] == target_label
            X = self.X[idx, :]

        X_log = np.log1p(X)
        X_std = StandardScaler().fit_transform(X_log)

        if method == "glasso":
            model = GraphicalLasso(alpha=alpha, max_iter=kwargs.get("max_iter", 500))
            model.fit(X_std)
            Theta = model.precision_

            d = np.sqrt(np.diag(Theta))
            partial_corr = -Theta / np.outer(d, d)
            np.fill_diagonal(partial_corr, 0)

            A = np.abs(partial_corr)
            A[A < 1e-6] = 0

        elif method == "pearson":
            corr = np.corrcoef(X_std, rowvar=False)
            np.fill_diagonal(corr, 0)
            partial_corr = corr
            A = np.abs(corr)

        else:
            raise ValueError("method must be 'glasso' or 'pearson'")
        G = nx.from_numpy_array(A)

        if metabolite_name_key and metabolite_name_key in self.var.columns:
            names = self.var[metabolite_name_key].values
            G = nx.relabel_nodes(G, dict(enumerate(names)))

        centrality = {
            "degree": nx.degree_centrality(G),
            "betweenness": nx.betweenness_centrality(G, weight="weight"),
            "eigenvector": nx.eigenvector_centrality_numpy(G, weight="weight"),
            "pagerank": nx.pagerank(G, weight="weight")
        }

        keep = sorted(
            centrality["pagerank"],
            key=centrality["pagerank"].get,
            reverse=True
        )[:top_nodes]

        H = G.subgraph(keep).copy()

        idx_map = {n: i for i, n in enumerate(G.nodes())}

        edges = []
        edge_colors = []
        edge_widths = []

        for u, v in H.edges():
            i, j = idx_map[u], idx_map[v]
            w = partial_corr[i, j]
            if abs(w) < edge_threshold:
                continue
            edges.append((u, v))
            edge_colors.append(w)
            edge_widths.append(abs(w) * kwargs.get("edge_width_multiplier", 20))

        H.remove_edges_from(list(H.edges()))
        H.add_edges_from(edges)

        mean_intensity = X_log.mean(axis=0)
        node_sizes = [
            kwargs.get("node_size_base", 10) +
            kwargs.get("node_size_multiplier", 20) *
            mean_intensity[idx_map[n]]
            for n in H.nodes()
        ]

        pos = nx.spring_layout(H, seed=layout_seed, weight="weight")

        node_colors = "lightgray"
        category_color_map = None

        if metabolite_label_key and metabolite_label_key in self.var.columns:

            cat = self.var[metabolite_label_key].values

            if metabolite_name_key and metabolite_name_key in self.var.columns:
                names = self.var[metabolite_name_key].values
                name_to_cat = dict(zip(names, cat))
            else:
                name_to_cat = dict(enumerate(cat))

            unique_cat = sorted(set(cat))
            cmap = plt.get_cmap("tab20", len(unique_cat))
            category_color_map = {c: cmap(i) for i, c in enumerate(unique_cat)}

            node_colors = [
                category_color_map.get(name_to_cat[n], "lightgray")
                for n in H.nodes()
            ]

        fig, ax = plt.subplots(figsize=figsize)

        if edges:
            nx.draw_networkx_edges(
                H, pos,
                edge_color=edge_colors,
                edge_cmap=plt.cm.coolwarm,
                width=edge_widths,
                edge_vmin=min(edge_colors),
                edge_vmax=max(edge_colors),
                alpha=0.8,
                ax=ax
            )

        nx.draw_networkx_nodes(
            H, pos,
            node_size=node_sizes,
            node_color=node_colors,
            edgecolors="black",
            linewidths=0.5,
            ax=ax
        )

        if category_color_map is not None:
            import matplotlib.patches as mpatches
            handles = [
                mpatches.Patch(color=color, label=str(cat))
                for cat, color in category_color_map.items()
            ]
            ax.legend(
                handles=handles,
                title=metabolite_label_key,
                bbox_to_anchor=kwargs.get("legend_bbox_to_anchor", (1.05, 0.5)),
                loc="upper left"
            )
            
        if edge_colors:
            vmin, vmax = min(edge_colors), max(edge_colors)
            if vmin == vmax:
                vmin, vmax = -abs(vmin), abs(vmax)

            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.coolwarm,
                norm=plt.Normalize(vmin, vmax)
            )
            sm.set_array([])
            if kwargs.get("colorbar", True):
                fig.colorbar(sm, ax=ax, label="Partial correlation")

        ax.axis("off")
        if title:
            ax.set_title(title)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        return G, centrality, partial_corr
    
    def metabolite_correlation_heatmap(
        self,
        target_label: str = None,
        label_key: str = None,
        metabolite_name_key: str = None,
        save_path: str = None,
        title: str = None,
        mode: str = "feature",          # "feature" or "sample"
        cmap: str = "coolwarm",
        triangle: str = "full",         # "full" or "lower"
        figsize=(8, 8),
        vmin: float = None,
        vmax: float = None,
        **kwargs
    ):

        if target_label is None or label_key is None:
            X = self.X
        else:
            idx = self.obs[label_key] == target_label
            X = self.X[idx, :]

        if mode == "feature":
            corr = np.corrcoef(X, rowvar=False)

            if metabolite_name_key and metabolite_name_key in self.var.columns:
                labels = self.var[metabolite_name_key].values
            else:
                labels = np.arange(corr.shape[0])

        elif mode == "sample":
            corr = np.corrcoef(X, rowvar=True)
            labels = np.arange(corr.shape[0])

        else:
            raise ValueError("mode must be 'feature' or 'sample'")

        if triangle == "lower":
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            corr = corr.copy()
            corr[mask] = np.nan

        elif triangle != "full":
            raise ValueError("triangle must be 'full' or 'lower'")

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            corr,
            cmap=cmap,
            vmin=vmin if vmin is not None else np.nanmin(corr),
            vmax=vmax if vmax is not None else np.nanmax(corr)
        )
        
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(
            left=False,
            bottom=False,
            labelleft=False if not kwargs.get("show_ticks", False) else True,
            labelbottom=False if not kwargs.get("show_ticks", False) else True
        )

        if kwargs.get("show_ticks", False):
            ax.set_xticks(np.arange(len(labels)))
            ax.set_yticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if kwargs.get("colorbar", True):
            fig.colorbar(im, ax=ax, label="Pearson correlation")

        if title:
            ax.set_title(title)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        return corr
    
    def feature_scatter(self, key:str, feature_x_name:str, feature_y_name:str, color_key:str = None,
                        title:str = None, save_path:str = None, **kwargs):
        from sklearn.linear_model import LinearRegression
        from scipy.stats import pearsonr
        x_idx = np.flatnonzero(self.var[key] == feature_x_name)[0]
        y_idx = np.flatnonzero(self.var[key] == feature_y_name)[0]
        
        x = self.X[:, x_idx]
        y = self.X[:, y_idx]

        plt.figure(figsize=(8, 6))
        alpha = kwargs.get("alpha", 0.7)
        lw = kwargs.get("lw", 1)

        def fit_and_plot(x_sub, y_sub, color=None, label_prefix=None):
            mask = ~np.isnan(x_sub) & ~np.isnan(y_sub)
            x_clean = x_sub[mask]
            y_clean = y_sub[mask]

            if len(x_clean) < 2:
                return

            X_reshape = x_clean.reshape(-1, 1)

            model = LinearRegression()
            model.fit(X_reshape, y_clean)
            y_pred = model.predict(X_reshape)

            r, p = pearsonr(x_clean, y_clean)

            plt.scatter(x_clean, y_clean, 
                        alpha=alpha, 
                        color=color,
                        label=f"{label_prefix} (R={r:.3f})" if label_prefix else f"R={r:.3f}")

            order = np.argsort(x_clean)
            plt.plot(x_clean[order], 
                    y_pred[order], 
                    color=color, 
                    linewidth=lw,
                    linestyle='--')

        if color_key and color_key in self.obs.columns:
            categories = self.obs[color_key].unique()
            cmap = plt.cm.get_cmap('tab20', len(categories))

            for i, cat in enumerate(categories):
                idx = self.obs[color_key] == cat
                fit_and_plot(
                    x[idx], 
                    y[idx], 
                    color=cmap(i), 
                    label_prefix=str(cat)
                )

            plt.legend()
        else:
            fit_and_plot(x, y, color="steelblue")

        plt.xlabel(feature_x_name)
        plt.ylabel(feature_y_name)

        if title:
            plt.title(title)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()
            
            plt.xlabel(feature_x_name)
            plt.ylabel(feature_y_name)
            if title: plt.title(title)
            if save_path: plt.savefig(save_path, dpi=300, bbox_inches="tight")
            else: plt.show()
        
    def _t_stats(self, label_key:str, labelx:str, labely:str, name_key:str, save_csv:str = None):
        from scipy.stats import ttest_ind
        data_x = self.X[self.obs[label_key] == labelx, :]
        data_y = self.X[self.obs[label_key] == labely, :]
        results = ttest_ind(data_x, data_y, axis=0, equal_var=False)
        if save_csv:
            df = pd.DataFrame({
                "feature": self.var[name_key].values,
                "t_stat": results.statistic
            })
            df.to_csv(save_csv, index=False)
        return results.statistic, results.pvalue