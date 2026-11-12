"""Annotation mirror spectra with a common mass axis."""
import numpy as np
import plotly.graph_objects as go


def mirror_spectrum(observed, reference, title="MS/MS mirror spectrum"):
    figure = go.Figure()
    for peaks, sign, name, color in [(observed, 1, "Observed MS/MS", "#2676b8"),
                                      (reference, -1, "Library MS/MS", "#df8130")]:
        peaks = np.asarray(peaks, dtype=float).reshape(-1, 2)
        if not len(peaks):
            continue
        intensities = peaks[:, 1] / max(peaks[:, 1].max(), 1e-30) * 100 * sign
        x = np.column_stack([peaks[:, 0], peaks[:, 0], np.full(len(peaks), np.nan)]).ravel()
        y = np.column_stack([np.zeros(len(peaks)), intensities, np.full(len(peaks), np.nan)]).ravel()
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=1.3)))
    figure.add_hline(y=0, line_color="gray", line_width=.6)
    figure.update_layout(title=title, xaxis_title="m/z", yaxis_title="Relative intensity (%)",
                         template="plotly_white", height=430, yaxis=dict(range=[-110, 110]))
    return figure


def cell_spectrum_figure(data, spectrum, feature, *, half_window=2., max_points=30000):
    from scMM.plot.diagnostics import spectrum_curve

    mz = np.asarray(spectrum["mz"], float)
    intensity = np.asarray(spectrum["intensity"], float)
    center = float(data.var.at[feature, "mz"])
    parent = str(data.var.at[feature, "isotope_parent"]) if "isotope_parent" in data.var else feature
    parent = parent if parent in data.var_names else feature
    related = [parent]
    if "isotope_parent" in data.var:
        children = data.var_names[data.var["isotope_parent"].astype(str).eq(parent)]
        related.extend(children.tolist())
    related = list(dict.fromkeys([feature, *related]))
    markers = []
    for key in related:
        target = float(data.var.at[key, "mz"])
        if target < mz[0] or target > mz[-1]:
            continue
        resolution = spectrum["resolution_200"] * np.sqrt(200. / target)
        tolerance = max(target * 5e-6, target / resolution)
        left, right = np.searchsorted(mz, [target - tolerance, target + tolerance])
        if right <= left:
            continue
        apex = left + int(np.argmax(intensity[left:right]))
        markers.append((key, mz[apex], intensity[apex], left, right))
    plot_mz, plot_intensity = spectrum_curve(mz, intensity, max_points, [item[1] for item in markers])
    figure = go.Figure(go.Scattergl(x=plot_mz, y=plot_intensity, mode="lines", name="Summed cell-frame MS1",
                                   line=dict(color="#728295", width=1.)))
    for key, peak_mz, peak_intensity, left, right in markers:
        color = "#d62728" if key == feature else "#ef8a16"
        figure.add_trace(go.Scatter(x=mz[left:right], y=intensity[left:right], mode="lines",
                                    line=dict(color=color, width=2.5), showlegend=False))
        figure.add_trace(go.Scatter(x=[peak_mz], y=[peak_intensity], mode="markers+text", name=str(key),
                                    text=[str(key)], textposition="top center",
                                    marker=dict(color=color, size=9), cliponaxis=True,
                                    hovertemplate=f"{key}<br>m/z=%{{x:.6f}}<br>intensity=%{{y:.6g}}<extra></extra>"))
    masses = [float(data.var.at[key, "mz"]) for key in related]
    lower = max(mz[0], min(center - half_window, min(masses) - .2))
    upper = min(mz[-1], max(center + half_window, max(masses) + .2))
    if lower >= upper:
        lower, upper = mz[0], mz[-1]
    visible = intensity[(mz >= lower) & (mz <= upper)]
    maximum = float(visible.max()) if len(visible) else 0.
    figure.update_layout(title=f"Cell-frame MS1 spectrum · {feature} · m/z {center:.6f}", template="plotly_white", height=560,
                         xaxis=dict(title="m/z", range=[lower, upper]),
                         yaxis=dict(title="Summed intensity", range=[0., maximum * 1.15 or 1.]),
                         uirevision=f"{id(spectrum)}:{feature}:{half_window}")
    return figure
