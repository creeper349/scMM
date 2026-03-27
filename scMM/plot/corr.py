import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from anndata import AnnData
from typing import Dict
from os.path import join, exists

class CorrAnalEngine:
    def __init__(self, save_path: str, **omics: Dict[str, AnnData]):
        self.save_path = save_path
        self.omics = omics
        self.uns = {}
        
    def corr_heatmap(self, omic1: str, omic2: str,
                 feature_name_key1: str, feature_name_key2: str,
                 method: str = "pearson",
                 figsize=(10,10),
                 metric="euclidean",
                 threshold=0.0,
                 **kwargs):

        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist
        from scipy.stats import spearmanr

        adata1 = self.omics[omic1]
        adata2 = self.omics[omic2]

        X = adata1.X
        Y = adata2.X

        if not isinstance(X, np.ndarray):
            X = X.toarray()
        if not isinstance(Y, np.ndarray):
            Y = Y.toarray()

        feature_names1 = adata1.var[feature_name_key1].values
        feature_names2 = adata2.var[feature_name_key2].values

        corr = np.zeros((X.shape[1], Y.shape[1]))

        if method == "pearson":

            Xc = X - X.mean(axis=0)
            Yc = Y - Y.mean(axis=0)

            corr = (Xc.T @ Yc) / (
                np.sqrt((Xc**2).sum(axis=0))[:,None] *
                np.sqrt((Yc**2).sum(axis=0))[None,:]
            )

        elif method == "spearman":

            from scipy.stats import rankdata

            Xr = np.apply_along_axis(rankdata, 0, X)
            Yr = np.apply_along_axis(rankdata, 0, Y)

            Xc = Xr - Xr.mean(axis=0)
            Yc = Yr - Yr.mean(axis=0)

            corr = (Xc.T @ Yc) / (
                np.sqrt((Xc**2).sum(axis=0))[:,None] *
                np.sqrt((Yc**2).sum(axis=0))[None,:]
            )

        corr = pd.DataFrame(
            corr,
            index=feature_names1,
            columns=feature_names2
        )
        
        mask = np.abs(corr.values) > threshold

        keep_rows = mask.any(axis=1)
        keep_cols = mask.any(axis=0)

        corr = corr.loc[keep_rows, keep_cols]

        feature_names1 = corr.index.values
        feature_names2 = corr.columns.values

        row_link = linkage(pdist(corr.values, metric=metric), method="average")
        col_link = linkage(pdist(corr.values.T, metric=metric), method="average")

        n_row_clusters = kwargs.get("n_row_clusters", 6)
        n_col_clusters = kwargs.get("n_col_clusters", 6)

        row_clusters = fcluster(row_link, n_row_clusters, criterion="maxclust")
        col_clusters = fcluster(col_link, n_col_clusters, criterion="maxclust")

        row_palette = sns.color_palette("tab10", n_row_clusters)
        col_palette = sns.color_palette("tab10", n_col_clusters)

        row_colors = [row_palette[i-1] for i in row_clusters]
        col_colors = [col_palette[i-1] for i in col_clusters]

        g = sns.clustermap(
            corr,
            row_linkage=row_link,
            col_linkage=col_link,
            cmap=kwargs.get("cmap", "viridis"),
            vmin=kwargs.get("vmin", corr.values.min()),
            vmax=kwargs.get("vmax", corr.values.max()),
            figsize=figsize,
            row_cluster=kwargs.get("cluster_rows", True),
            col_cluster=kwargs.get("cluster_cols", True),
            row_colors=row_colors,
            col_colors=col_colors,
            xticklabels=False,
            yticklabels=False,
            dendrogram_ratio=kwargs.get("dendrogram_ratio",(0.06,0.06)),
            cbar_pos=None
        )
        
        cax = g.fig.add_axes(kwargs.get("cax", [1.02, 0.3, 0.02, 0.6]))
        plt.colorbar(
            g.ax_heatmap.collections[0],
            cax=cax,
            label=f"{method.capitalize()} correlation"
        )

        plt.savefig(join(self.save_path, f"{omic1}_{omic2}_corr_heatmap.svg"), bbox_inches="tight")
        row_module = pd.Series(
            row_clusters,
            index=feature_names1,
            name=f"{omic1}_module"
        )

        col_module = pd.Series(
            col_clusters,
            index=feature_names2,
            name=f"{omic2}_module"
        )

        self.uns[f"{omic1}_{omic2}_corr"] = corr
        self.uns[f"{omic1}_cluster"] = row_module
        self.uns[f"{omic2}_cluster"] = col_module
        
    def kw_enrichment_wordcloud(self,
                         data_key: str,
                         word_include: list = None,
                         word_exclude: list = None,
                         top_n: int = 10,
                         **kwargs):

        from wordcloud import WordCloud
        from collections import Counter

        obj = self.uns[data_key]

        if isinstance(obj, pd.Series):
            cluster_map = obj
        elif isinstance(obj, pd.DataFrame):
            cluster_map = obj.iloc[:, 0]
        elif isinstance(obj, dict):
            cluster_map = pd.Series(obj)
        else:
            raise ValueError("Unsupported clustering result format.")

        cluster_map.index = cluster_map.index.astype(str)

        if word_exclude:
            word_exclude = set(word_exclude)
        else:
            word_exclude = set()

        if word_include:
            word_include = set(word_include)

        def tokenize(name):
            tokens = name.lower().replace("-", "_").split("_")
            tokens = [t for t in tokens if len(t) > 2]
            tokens = [t for t in tokens if t not in word_exclude]
            if word_include:
                tokens = [t for t in tokens if t in word_include]
            return tokens

        clusters = sorted(cluster_map.unique())
        palette = sns.color_palette("tab10", len(clusters))

        for i, c in enumerate(clusters):

            features = cluster_map[cluster_map == c].index

            tokens = []
            for f in features:
                tokens.extend(tokenize(f))

            if not tokens:
                continue

            freq = Counter(tokens)
            max_freq = max(freq.values())
            base_color = np.array(palette[i])
            
            def color_func(word, font_size, position, orientation, font_path, random_state):
                intensity = freq[word] / max_freq
                color = base_color * intensity + (1 - intensity) * np.array([1, 1, 1])
                r, g, b = (color * 255).astype(int)
                return f"rgb({r},{g},{b})"

            rgb = tuple(int(255*v) for v in palette[i])
            color_str = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

            wc = WordCloud(
                width=800,
                height=800,
                background_color=kwargs.get("background_color", "white"),
                max_words=top_n,
                collocations=False,
                font_path=kwargs.get("font_path", None)
            ).generate_from_frequencies(freq)

            wc = wc.recolor(color_func=color_func)

            plt.figure(figsize=kwargs.get("figsize", (4,4)))
            plt.imshow(wc)
            plt.axis("off")
            for spine in plt.gca().spines.values():
                spine.set_visible(kwargs.get("spines_visible", True))

            save_file = join(
                self.save_path,
                f"{data_key}_cluster{c}.svg"
            )

            plt.savefig(save_file)
            plt.close()
            
    def plot_corr_dist(self, omics1: str, omics2: str,
                   show_random_corr: bool = True, **kwargs):

        key = f"{omics1}_{omics2}_corr"

        if key not in self.uns:
            raise KeyError(f"{key} not found in self.uns")

        corr_df = self.uns[key]
        real_vals = np.abs(corr_df.values.ravel())
        real_vals = real_vals[np.isfinite(real_vals)]

        method = kwargs.get("method", "pearson")
        seed = kwargs.get("random_seed", 0)
        figsize = kwargs.get("figsize", (6, 4))

        rand_vals = None

        if show_random_corr:

            rng = np.random.default_rng(seed)

            ad1 = self.omics[omics1]
            ad2 = self.omics[omics2]

            X = ad1.X
            Y = ad2.X

            if not isinstance(X, np.ndarray):
                X = X.toarray()
            if not isinstance(Y, np.ndarray):
                Y = Y.toarray()
            perm = rng.permutation(X.shape[0])
            Y_perm = Y[perm, :]

            def pearson_corr(A, B):
                Ac = A - A.mean(axis=0)
                Bc = B - B.mean(axis=0)
                return (Ac.T @ Bc) / (
                    np.sqrt((Ac**2).sum(axis=0))[:, None] *
                    np.sqrt((Bc**2).sum(axis=0))[None, :]
                )

            if method == "pearson":
                rand_corr = pearson_corr(X, Y_perm)

            elif method == "spearman":
                Xr = pd.DataFrame(X).rank(axis=0).values
                Yr = pd.DataFrame(Y_perm).rank(axis=0).values
                rand_corr = pearson_corr(Xr, Yr)

            else:
                raise ValueError("method must be 'pearson' or 'spearman'")

            rand_vals = np.abs(rand_corr.ravel())
            rand_vals = rand_vals[np.isfinite(rand_vals)]
        plt.figure(figsize=figsize)

        sns.kdeplot(real_vals, label="Observed", linewidth=2)

        if rand_vals is not None:
            sns.kdeplot(rand_vals, label="Permutation", linewidth=2)

        plt.xlabel(f"|{method} R|")
        plt.ylabel("Density")
        plt.xlim(0, 1)
        plt.legend(frameon=False)
        plt.savefig(join(self.save_path, f"{omics1}_{omics2}_corr_dist.svg"), bbox_inches="tight")
        plt.close()

        return real_vals, rand_vals
    
    def plot_gini_dist(self, omics1: str, omics2: str, method = "spearman", metric = "gini", **kwargs):
        X = self.omics[omics1].X
        Y = self.omics[omics2].X

        if not isinstance(X, np.ndarray):
            X = X.toarray()
        if not isinstance(Y, np.ndarray):
            Y = Y.toarray()

        def pearson_corr(A, B):
            Ac = A - A.mean(axis=0)
            Bc = B - B.mean(axis=0)
            return (Ac.T @ Bc) / (
                np.sqrt((Ac**2).sum(axis=0))[:, None] *
                np.sqrt((Bc**2).sum(axis=0))[None, :]
            )

        if method == "pearson":
            corr_11 = pearson_corr(X, X)
            corr_12 = pearson_corr(X, Y)
            corr_22 = pearson_corr(Y, Y)

        elif method == "spearman":
            Xr = pd.DataFrame(X).rank(axis=0).values
            Yr = pd.DataFrame(Y).rank(axis=0).values
            corr_11 = pearson_corr(Xr, Xr)
            corr_12 = pearson_corr(Xr, Yr)
            corr_22 = pearson_corr(Yr, Yr)

        else:
            raise ValueError("method must be 'pearson' or 'spearman'")
        
        label_11 = f"{omics1}-{omics1}"
        label_12 = f"{omics1}-{omics2}"
        label_22 = f"{omics2}-{omics2}"

        if metric == "abs_r":

            v11 = np.abs(corr_11.ravel())
            v12 = np.abs(corr_12.ravel())
            v22 = np.abs(corr_22.ravel())

            df = pd.DataFrame({
                "Value": np.concatenate([v11, v12, v22]),
                "Type": ([label_11]*len(v11) +
                        [label_12]*len(v12) +
                        [label_22]*len(v22))
            })

            ylabel = f"|{method} R|"

        elif metric == "gini":

            def gini(vec):
                vec = np.abs(vec[np.isfinite(vec)])
                if vec.size == 0 or np.sum(vec)==0:
                    return 0
                vec = np.sort(vec)
                n = len(vec)
                return (2*np.sum(np.arange(1,n+1)*vec) /
                        (n*np.sum(vec))) - (n+1)/n

            g11 = [gini(corr_11[i,:]) for i in range(corr_11.shape[0])]
            g12 = [gini(corr_12[i,:]) for i in range(corr_12.shape[0])]
            g22 = [gini(corr_22[i,:]) for i in range(corr_22.shape[0])]

            df = pd.DataFrame({
                "Value": g11 + g12 + g22,
                "Type": ([label_11]*len(g11) +
                        [label_12]*len(g12) +
                        [label_22]*len(g22))
            })

            ylabel = "Gini Index"

        plt.figure(figsize=kwargs.get("figsize", (6, 4)))

        sns.violinplot(
            data=df,
            x="Type",
            y="Value",
            inner="box",
            palette="Set2"
        )

        plt.ylabel(ylabel)
        plt.xlabel("")
        plt.tight_layout()
        plt.savefig(join(self.save_path, f"{omics1}_{omics2}_{metric}_dist.svg"), bbox_inches="tight")
        plt.close()

        return df
    
    def plot_feature_scatter(self, omics1:str, omics2:str, feature1:str, feature2:str, 
                             name_key1:str, name_key2:str, reg:bool=True, **kwargs):
        
        from scipy.stats import spearmanr, pearsonr
        x = self.omics[omics1].X
        y = self.omics[omics2].X
        x_idx = (self.omics[omics1].var[name_key1] == feature1).to_numpy()
        y_idx = (self.omics[omics2].var[name_key2] == feature2).to_numpy()
        feature_x = x[:, x_idx].ravel()
        feature_y = y[:, y_idx].ravel()
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6,6)))
        sns.scatterplot(x=feature_x, y=feature_y, alpha=kwargs.get("alpha", 0.9), color=kwargs.get("dot_color", "black"))
        if reg:
            sns.regplot(x=feature_x, y=feature_y, scatter=False,
                        color=kwargs.get("line_color", "red"), line_kws={"linewidth":1}, ci=kwargs.get("ci", None))
        method = kwargs.get("corr_method", "pearson")
        if method == "spearman":
            r, p = spearmanr(feature_x, feature_y)
            label = f"Spearman r = {r:.3f}\np = {p:.2e}"
        else:
            r, p = pearsonr(feature_x, feature_y)
            label = f"Pearson r = {r:.3f}\np = {p:.2e}"
        ax.text(
            0.05, 0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10
        )
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.tight_layout()
        plt.savefig(join(self.save_path, f"{omics1}_{omics2}_{feature1}_vs_{feature2}_scatter.svg"), bbox_inches="tight")
        plt.close()
        
    def plot_feature_corr_dist(self, omics1:str, omics2:str, feature_name:str, **kwargs):
        if f"{omics1}_{omics2}_corr" not in self.uns:
            raise KeyError(f"{omics1}_{omics2}_corr not found in self.uns")
        
        corr_df = self.uns[f"{omics1}_{omics2}_corr"]
        dist = corr_df.loc[feature_name, :].values.ravel()
        sns.displot(dist, kde=True, height=kwargs.get("height", 4), aspect=kwargs.get("aspect", 1.5))
        plt.xlabel(f"{omics2} correlation with {feature_name}")
        plt.savefig(join(self.save_path, f"{omics1}_{omics2}_{feature_name}_corr_dist.svg"), bbox_inches="tight")
        plt.close()
        
        return corr_df.columns[np.argsort(dist)[::-1]], dist[np.argsort(dist)[::-1]]