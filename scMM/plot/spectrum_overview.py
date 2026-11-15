"""A fixed full-spectrum overview linked to the interactive detail viewport."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_LINK_VIEWPORT = """
const graph = document.getElementById('{plot_id}');
graph.on('plotly_relayout', function(event) {
    if (!('xaxis2.range[0]' in event || 'xaxis2.range' in event || 'xaxis2.autorange' in event)) return;
    const full = graph.layout.xaxis.range;
    const range = event['xaxis2.autorange'] ? full :
        (event['xaxis2.range'] || [event['xaxis2.range[0]'], event['xaxis2.range[1]']]);
    const lower = Math.max(full[0], Number(range[0]));
    const upper = Math.min(full[1], Number(range[1]));
    if (!(lower < upper)) return;
    const trace = graph.data[0];
    let maximum = 0;
    for (let i = 0; i < trace.x.length; i++) {
        if (trace.x[i] >= lower && trace.x[i] <= upper) maximum = Math.max(maximum, trace.y[i]);
    }
    Plotly.relayout(graph, {
        'shapes[0].x0': lower, 'shapes[0].x1': upper,
        'yaxis2.range': [0, maximum * 1.15 || 1]
    });
});
"""


def spectrum_overview_figure(detail):
    figure = make_subplots(rows=2, cols=1, row_heights=[.23, .77], vertical_spacing=.12)
    full = detail.data[0]
    figure.add_trace(go.Scatter(x=list(full.x), y=list(full.y), mode="lines", showlegend=False,
                                line=dict(color="#728295", width=1), hoverinfo="skip"), row=1, col=1)
    for trace in detail.data:
        figure.add_trace(trace, row=2, col=1)
    lower, upper = detail.layout.xaxis.range
    minimum, maximum = min(full.x), max(full.x)
    figure.add_shape(type="rect", x0=lower, x1=upper, y0=0, y1=1, xref="x", yref="y domain",
                     fillcolor="rgba(214,39,40,.18)", line=dict(color="#d62728", width=1))
    figure.update_xaxes(range=[minimum, maximum], fixedrange=True, row=1, col=1)
    figure.update_yaxes(range=[0, max(full.y) * 1.08 or 1], fixedrange=True,
                       title_text="Total", row=1, col=1)
    figure.update_xaxes(range=[lower, upper], minallowed=minimum, maxallowed=maximum,
                       title_text="m/z", row=2, col=1)
    figure.update_yaxes(range=list(detail.layout.yaxis.range), minallowed=0,
                       title_text="Summed intensity", row=2, col=1)
    figure.update_layout(template="plotly_white", height=740, title=detail.layout.title.text,
                         margin=dict(l=75, r=30, t=65, b=50), dragmode="zoom",
                         uirevision=detail.layout.uirevision, legend=dict(orientation="h", y=-.12))
    return figure


def spectrum_overview_html(detail):
    from scMM.i18n import localize_figure
    figure = spectrum_overview_figure(localize_figure(detail))
    return figure.to_html(include_plotlyjs=True, full_html=False, post_script=_LINK_VIEWPORT,
                          config={"displaylogo": False, "scrollZoom": True, "responsive": True})
