from types import SimpleNamespace

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.figure import Figure

from scMM.file.data import CyESIData
from scMM.plot.diagnostics import color_coordinates
from scMM.plot.export import render_figure
from scMM import webapp


def test_umap_log_colors_preserve_raw_ticks_and_export_norm(monkeypatch):
    data = CyESIData(X=np.asarray([[0.], [1.], [10.], [100.], [-5.], [np.nan]], dtype=np.float32),
                    obs=pd.DataFrame(index=list("abcdef")), var=pd.DataFrame({"mz": [100.]}, index=["f1"]))
    data.obsm["X_umap"] = np.arange(12, dtype=float).reshape(6, 2)
    captured = []
    monkeypatch.setattr(webapp, "_plotly_with_export", lambda st, figure, key: captured.append(figure))
    st = SimpleNamespace(caption=lambda message: None)
    webapp._plot_embedding(st, data, "X_umap", "feature", None, "f1", "Viridis", "Plotly", color_scale="log10")
    figure = captured[0]
    np.testing.assert_allclose(figure.data[0].marker.color, [0., 1., 2.])
    np.testing.assert_allclose(figure.data[0].meta["scmm_color_values"], [1., 10., 100.])
    assert len(figure.data[1].x) == 3
    assert figure.layout.coloraxis.colorbar.ticktext[0] == "1"
    assert figure.layout.coloraxis.colorbar.ticktext[-1] == "100"
    canvases = []
    original = Figure.savefig

    def record(self, *args, **kwargs):
        canvases.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", record)
    svg = render_figure(figure, file_format="svg")
    assert b"<text" in svg
    collection = canvases[0].axes[0].collections[0]
    assert isinstance(collection.norm, LogNorm)
    np.testing.assert_allclose(collection.get_array(), [1., 10., 100.])


def test_symlog_matches_matplotlib_color_mapping():
    values = np.asarray([-100., -1., -.1, 0., .1, 1., 100.])
    coordinates = color_coordinates(values, "symlog", linthresh=.5)
    scaled = (coordinates - coordinates.min()) / np.ptp(coordinates)
    np.testing.assert_allclose(scaled, SymLogNorm(.5, vmin=-100, vmax=100)(values))
