import numpy as np
import pandas as pd

from scMM.file.data import CyESIData
from scMM.plot.engine import PlotEngine


def _data():
    rng = np.random.default_rng(4)
    X = np.vstack(
        [rng.normal(1, 0.1, (15, 5)), rng.normal(4, 0.1, (15, 5))]
    ).astype(np.float32)
    return CyESIData(
        X=X,
        obs=pd.DataFrame(
            {"group": ["a"] * 15 + ["b"] * 15},
            index=[f"cell_{i}" for i in range(30)],
        ),
        var=pd.DataFrame(
            {"mz": np.arange(100, 105, dtype=float)},
            index=[str(x) for x in range(100, 105)],
        ),
    )


def test_reduction_clustering_and_differential(tmp_path):
    engine = PlotEngine(_data(), tmp_path)
    assert engine.pca(n_components=3).shape == (30, 3)
    labels = engine.cluster_cells(n_clusters=2)
    assert labels.nunique() == 2
    result = engine.differential_features("group", "a", "b")
    assert {"p_value", "fdr", "log2_fold_change"} <= set(result.columns)
    assert np.all(result["fdr"].between(0, 1))


def test_graph_clustering_writes_labels_to_obs(tmp_path):
    for method in ("leiden", "louvain"):
        data = _data()
        engine = PlotEngine(data, tmp_path, copy=False)
        engine.pca(n_components=3)
        key = f"{method}_cluster"
        labels = engine.cluster_cells(
            method,
            key_added=key,
            n_neighbors=5,
            resolution=1.0,
        )
        assert key in data.obs
        assert labels.index.equals(data.obs_names)
        assert labels.nunique() >= 2


def test_dimension_reduction_and_clustering_ignore_tagged_cells(tmp_path):
    data = _data()
    data.obs["cell_keep_manual"] = np.ones(data.n_obs, dtype=np.uint8)
    data.obs.iloc[[1, 4], data.obs.columns.get_loc("cell_keep_manual")] = 0
    data.var["feature_keep_detection"] = np.ones(data.n_vars, dtype=np.uint8)
    data.var.iloc[-1, data.var.columns.get_loc("feature_keep_detection")] = 0
    engine = PlotEngine.from_adata(data, tmp_path, copy=False)

    embedding = engine.pca(n_components=2)
    labels = engine.cluster_cells("kmeans", use_rep="X_pca", n_clusters=2)

    assert embedding.shape == (data.n_obs - 2, 2)
    assert np.isnan(data.obsm["X_pca"][[1, 4]]).all()
    assert labels.iloc[[1, 4]].isna().all()


def test_umap_current_x_uses_selected_layer_even_with_existing_pca(tmp_path, monkeypatch):
    data = _data()
    data.obsm["X_pca"] = np.zeros((data.n_obs, 2))
    data.normalize("total", scale=1.)
    captured = []

    class FakeUMAP:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, values):
            captured.append(values.copy())
            return values[:, :2]

    monkeypatch.setattr("scMM.plot.engine.umap.UMAP", FakeUMAP)
    PlotEngine(data, tmp_path, copy=False).reduce_dimension("umap", use_rep=None)
    np.testing.assert_allclose(captured[0], data.layers["normalized_total"])


def test_differential_can_exclude_zero_values(tmp_path):
    data = _data()
    data.X[:5, 0] = 0
    engine = PlotEngine(data, tmp_path, copy=False)
    included = engine.differential_features("group", "a", "b", include_zeros=True).set_index("feature_id")
    excluded = engine.differential_features("group", "a", "b", include_zeros=False).set_index("feature_id")
    assert included.loc["100", "n_a"] == 15
    assert excluded.loc["100", "n_a"] == 10
    assert excluded.loc["100", "mean_a"] > included.loc["100", "mean_a"]


def test_extended_reduction_and_trajectory_outputs(tmp_path):
    data = _data()
    engine = PlotEngine(data, tmp_path, copy=False)
    engine.pca(n_components=4)
    for method in ("tsne", "isomap", "lle"):
        kwargs = {"perplexity": 5} if method == "tsne" else {"n_neighbors": 5}
        embedding = engine.reduce_dimension(
            method,
            use_rep="X_pca",
            n_components=2,
            **kwargs,
        )
        assert embedding.shape == (data.n_obs, 2)
        assert f"X_{method}" in data.obsm

    pseudotime = engine.trajectory(
        "diffusion_pseudotime",
        use_rep="X_pca",
        key_added="dpt",
        n_neighbors=5,
        n_components=3,
    )
    assert "dpt" in data.obs
    assert "X_diffmap" in data.obsm
    assert pseudotime.between(0, 1).all()

    engine.cluster_cells("kmeans", key_added="cluster", n_clusters=2)
    for method in ("graph_pseudotime", "pca_pseudotime", "paga_pseudotime"):
        key = method.replace("_pseudotime", "_time")
        result = engine.trajectory(
            method,
            use_rep="X_pca",
            key_added=key,
            n_neighbors=5,
            cluster_key="cluster",
        )
        assert key in data.obs
        assert result.notna().any()
