"""Cell/file expression matrices and reproducible hierarchical heatmaps."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


def expression_matrix(data, features, *, pseudobulk=False, aggregate="mean", obs_key=None):
    cells = data.cell_keep_mask()
    indices = data.var_names.get_indexer(features)
    matrix = data.X[np.flatnonzero(cells)][:, indices]
    matrix = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    frame = pd.DataFrame(matrix, index=data.obs_names[cells], columns=features)
    obs = data.obs.loc[cells]
    if pseudobulk:
        if "source_file" not in obs:
            raise ValueError("Pseudobulk 需要 obs['source_file']。")
        files = obs["source_file"].astype(str)
        frame = frame.groupby(files, sort=False).agg(aggregate)
        if obs_key:
            annotations = obs.groupby(files, sort=False)[obs_key].agg(
                lambda values: str(values.dropna().iloc[0]) if values.dropna().astype(str).nunique() == 1 else "mixed"
            ).reindex(frame.index)
        else:
            annotations = pd.Series(frame.index, index=frame.index)
    else:
        annotations = obs[obs_key].astype(str) if obs_key else pd.Series(frame.index, index=frame.index)
    return frame, annotations


def cluster_order(values, *, method="average", metric="euclidean"):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.arange(len(values)), None
    distances = pdist(values, metric=metric)
    distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)
    tree = linkage(distances, method=method, optimal_ordering=True)
    return np.asarray(dendrogram(tree, no_plot=True)["leaves"]), tree


def heatmap_figure(frame, *, title, row_labels=None, column_labels=None,
                   show_rows=True, show_columns=True, cluster_rows=True, cluster_columns=True,
                   method="average", metric="euclidean", cmap="Viridis", annotate=False,
                   value_range=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    values = np.asarray(frame, float)
    if not values.size:
        raise ValueError("热图需要有效细胞和特征。")
    # Undefined correlations stay NaN in the heatmap; clustering uses a neutral fill.
    cluster_values = np.nan_to_num(values)
    rows, row_tree = cluster_order(cluster_values, method=method, metric=metric) if cluster_rows else (np.arange(len(values)), None)
    columns, column_tree = cluster_order(cluster_values.T, method=method, metric=metric) if cluster_columns else (np.arange(values.shape[1]), None)
    values = values[np.ix_(rows, columns)]
    row_labels = np.asarray(row_labels if row_labels is not None else frame.index, str)[rows]
    column_labels = np.asarray(column_labels if column_labels is not None else frame.columns, str)[columns]
    row_ids, column_ids = np.asarray(frame.index, str)[rows], np.asarray(frame.columns, str)[columns]
    figure = make_subplots(rows=2, cols=2, row_heights=[.18, .82], column_widths=[.16, .84],
                           horizontal_spacing=.02, vertical_spacing=.025,
                           specs=[[None, {}], [{}, {}]])
    if column_tree is not None:
        branches = dendrogram(column_tree, no_plot=True)
        for x, y in zip(branches["icoord"], branches["dcoord"]):
            figure.add_trace(go.Scatter(x=(np.asarray(x) - 5) / 10, y=y, mode="lines", line=dict(color="#555", width=1), showlegend=False, hoverinfo="skip"), row=1, col=2)
    if row_tree is not None:
        branches = dendrogram(row_tree, no_plot=True)
        for y, x in zip(branches["icoord"], branches["dcoord"]):
            figure.add_trace(go.Scatter(x=x, y=(np.asarray(y) - 5) / 10, mode="lines", line=dict(color="#555", width=1), showlegend=False, hoverinfo="skip"), row=2, col=1)
    custom = [[[row_ids[r], column_ids[c]] for c in range(len(columns))] for r in range(len(rows))]
    figure.add_trace(go.Heatmap(z=values, x=np.arange(len(columns)), y=np.arange(len(rows)), colorscale=cmap,
                                zmin=value_range[0] if value_range else None, zmax=value_range[1] if value_range else None,
                                customdata=custom, texttemplate="%{z:.2f}" if annotate else None,
                                hovertemplate="%{customdata[0]} / %{customdata[1]}<br>%{z:.5g}<extra></extra>"), row=2, col=2)
    figure.update_xaxes(visible=False, row=1, col=2, range=[-.5, len(columns) - .5])
    figure.update_yaxes(visible=False, row=1, col=2)
    figure.update_xaxes(visible=False, autorange="reversed", row=2, col=1)
    figure.update_yaxes(visible=False, range=[len(rows) - .5, -.5], row=2, col=1)
    figure.update_xaxes(tickmode="array", tickvals=np.arange(len(columns)), ticktext=column_labels,
                        showticklabels=show_columns, tickangle=-60, range=[-.5, len(columns) - .5], row=2, col=2)
    figure.update_yaxes(tickmode="array", tickvals=np.arange(len(rows)), ticktext=row_labels,
                        showticklabels=show_rows, side="right", range=[len(rows) - .5, -.5], row=2, col=2)
    figure.update_layout(title=title, height=max(650, min(1200, len(rows) * 18 + 200)),
                          margin=dict(l=20, r=180 if show_rows else 30, b=160 if show_columns else 30))
    return figure
