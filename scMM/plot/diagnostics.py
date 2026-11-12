"""Shared numeric data for interactive and exported diagnostic plots."""

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def color_coordinates(values, scale="linear", linthresh=1.0):
    values = np.asarray(values, float)
    if scale == "log10":
        result = np.full(values.shape, np.nan)
        valid = np.isfinite(values) & (values > 0)
        result[valid] = np.log10(values[valid])
        return result
    if scale == "symlog":
        from matplotlib.scale import SymmetricalLogTransform
        return SymmetricalLogTransform(base=10, linthresh=linthresh, linscale=1).transform(values)
    return values.copy()


def point_density(x, y, *, bins=128, smoothing=2.0):
    """Gaussian-smoothed 2D histogram density, evaluated at each cell."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    valid = np.isfinite(x) & np.isfinite(y)
    result = np.full(len(x), np.nan)
    if not valid.any():
        return result
    values = np.column_stack([x[valid], y[valid]])
    lower = values.min(axis=0)
    span = np.ptp(values, axis=0)
    span[span == 0] = 1
    coordinates = (values - lower) / span * (bins - 1)
    counts, _, _ = np.histogram2d(coordinates[:, 0], coordinates[:, 1], bins=bins,
                                   range=[[-.5, bins - .5], [-.5, bins - .5]])
    smooth = gaussian_filter(counts, sigma=smoothing)
    result[valid] = map_coordinates(smooth, coordinates.T, order=1, mode="nearest")
    return result


def spectrum_curve(mz, intensity, max_points, markers=()):
    """Peak-preserving min/max reduction with exact marker interpolation anchors."""
    mz, intensity = np.asarray(mz, float), np.asarray(intensity, float)
    if len(mz) > max_points:
        edges = np.linspace(0, len(mz), max(2, max_points // 2 + 1), dtype=int)
        indices = [0, len(mz) - 1]
        for left, right in zip(edges[:-1], edges[1:]):
            if right > left:
                block = intensity[left:right]
                indices.extend([left + int(np.argmin(block)), left + int(np.argmax(block))])
        selected = mz[np.unique(indices)]
    else:
        selected = mz
    markers = np.asarray(markers, float)
    markers = markers[np.isfinite(markers) & (markers >= mz[0]) & (markers <= mz[-1])]
    plot_mz = np.unique(np.concatenate([selected, markers]))
    return plot_mz, np.interp(plot_mz, mz, intensity)


def qc_distribution(metrics, metric, *, bins=40):
    """Build exact shared-bin counts plus a horizontal box panel."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotly.colors import qualitative

    values = np.asarray(metrics[metric], float)
    finite = values[np.isfinite(values)]
    edges = np.histogram_bin_edges(finite, bins=bins) if len(finite) else np.linspace(0, 1, bins + 1)
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.23, .77], vertical_spacing=.06)
    sources = metrics.groupby("source_file", observed=True, sort=False) if "source_file" in metrics else [("cells", metrics)]
    for index, (source, group) in enumerate(sources):
        group_values = np.asarray(group[metric], float)
        group_values = group_values[np.isfinite(group_values)]
        color = qualitative.Plotly[index % len(qualitative.Plotly)]
        counts, _ = np.histogram(group_values, bins=edges)
        figure.add_trace(go.Bar(x=(edges[:-1] + edges[1:]) / 2, y=counts, width=np.diff(edges),
                                marker_color=color, opacity=.55, name=str(source), legendgroup=str(source)), row=2, col=1)
        figure.add_trace(go.Box(x=group_values, orientation="h", name=str(source),
                                marker_color=color, line_color=color, boxpoints=False,
                                legendgroup=str(source), showlegend=False), row=1, col=1)
    figure.update_layout(height=600, barmode="overlay", title=f"QC: {metric}")
    figure.update_xaxes(title_text=metric, row=2, col=1)
    figure.update_yaxes(title_text="Cells", row=2, col=1)
    return figure
