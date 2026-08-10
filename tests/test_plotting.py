import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyopenms as oms
import pytest

from scMM.plot.embedding import dimension_reduction
from scMM.plot.engine import PlotEngine
from scMM.plot.msplot import plot_ms, plot_spectrum


def test_plot_ms_supports_default_frame_range() -> None:
    data = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=[100.0, 200.0])
    fig, ax = plt.subplots()

    result = plot_ms(ax, data)

    assert result is ax
    assert ax.get_title() == "MS Spectrum for rows 0 to 1"
    plt.close(fig)


def test_plot_spectrum_validates_empty_range() -> None:
    spectrum = oms.MSSpectrum()
    spectrum.set_peaks((np.array([100.0]), np.array([1.0], dtype=np.float32)))

    with pytest.raises(ValueError, match="no peaks"):
        plot_spectrum(spectrum, mz_range=(200.0, 300.0))


def test_dimension_reduction_cluster_mode_does_not_forward_method() -> None:
    class Clusterer:
        def __init__(self, expected: int):
            self.expected = expected

        def fit_predict(self, matrix):
            assert self.expected == len(matrix)
            return np.arange(len(matrix)) % 2

    data = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0], [3.0, 1.0]])

    result = dimension_reduction(
        data,
        method="pca",
        color="cluster",
        cluster_kwargs={"method": Clusterer, "expected": 4},
    )

    assert result["X_emb"].shape == (4, 2)
    plt.close(result["ax"].figure)


def test_plot_engine_creates_output_and_runs_pca(tmp_path) -> None:
    frame = pd.DataFrame(np.arange(20, dtype=float).reshape(5, 4))

    engine = PlotEngine(frame, tmp_path / "figures")
    embedding = engine.pca(n_components=3)

    assert (tmp_path / "figures").is_dir()
    assert embedding.shape == (5, 3)
    assert "X_pca_params" in engine.adata.uns


def test_feature_network_euclidean_branch(tmp_path, monkeypatch) -> None:
    class FakeUmap:
        def __init__(self, **kwargs):
            assert kwargs["metric"] == "precomputed"

        def fit_transform(self, distances):
            assert distances.shape == (3, 3)
            return np.column_stack([np.arange(3), np.arange(3)[::-1]])

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUmap))
    frame = pd.DataFrame(np.arange(18, dtype=float).reshape(6, 3), columns=["a", "b", "c"])
    engine = PlotEngine(frame, tmp_path)

    result = engine.feature_network(metric="euclidean", n_neighbors=2)

    assert result.shape == (3, 2)
    assert (tmp_path / "feature_network_euclidean.svg").is_file()
