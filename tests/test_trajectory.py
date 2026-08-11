from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

import scMM.plot._trajectory as trajectory_module
from scMM.plot._trajectory import (
    metabolic_velocity_field,
    metabolite_trends,
    resample_trajectory,
    run_palantir,
    trend_cluster,
)


def make_adata() -> ad.AnnData:
    time = np.arange(6, dtype=float)
    matrix = np.column_stack([2 * time + 1, -time + 5, np.ones_like(time)])
    adata = ad.AnnData(matrix, obs=pd.DataFrame({"time": time}, index=[f"c{i}" for i in range(6)]))
    adata.obsm["X_umap"] = np.column_stack([time, time**2])
    adata.obsm["branch_probs"] = np.column_stack(
        [np.linspace(1.0, 0.0, 6), np.linspace(0.0, 1.0, 6)]
    )
    return adata


def test_resample_trajectory_stores_provenance() -> None:
    adata = make_adata()

    result = resample_trajectory(
        adata,
        window_size=4,
        step_size=3,
        parameterization_key="time",
        branch_prob_key="branch_probs",
        min_cells_per_window=2,
    )

    assert result is adata
    assert adata.uns["trajectory"].shape == (2, 2, 2)
    metadata = adata.uns["trajectory_metadata"]
    np.testing.assert_array_equal(metadata["window_starts"], [0, 2])
    assert metadata["counts"].shape == (2, 2)


def test_metabolic_velocity_recovers_linear_slopes() -> None:
    adata = make_adata()

    metabolic_velocity_field(adata, window_size=6, step_size=2)

    result = adata.uns["metabolic_velocity"]
    np.testing.assert_allclose(result["velocity_field"][0], [2.0, -1.0, 0.0])
    assert result["counts"][0] == 6


def test_metabolite_trends_and_clustering() -> None:
    adata = make_adata()

    metabolite_trends(adata, window_size=2, step_size=1, kernel_stat="mean")
    labels = trend_cluster(
        adata.uns["metabolite_trends"]["pooled"],
        metric="euclidean",
        cluster_method="kmeans",
        n_clusters=2,
        random_state=0,
    )

    assert adata.uns["metabolite_trends"]["pooled"].shape == (5, 3)
    assert labels.shape == (3,)
    assert set(labels) <= {0, 1}


def test_trend_cluster_marks_invalid_features() -> None:
    trends = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 2.0], [3.0, np.nan, 1.0]])

    labels = trend_cluster(trends, metric="euclidean", cluster_method="agglomerative", n_clusters=2)

    assert labels[1] == -1
    assert set(labels[[0, 2]]) == {0, 1}


def test_trajectory_rejects_nonfinite_parameterization() -> None:
    adata = make_adata()
    adata.obs.loc["c0", "time"] = np.nan

    with pytest.raises(ValueError, match="finite"):
        resample_trajectory(adata, parameterization_key="time")


def test_run_palantir_stores_aligned_outputs_and_provenance(monkeypatch) -> None:
    adata = make_adata()

    def fake_diffusion(data, **kwargs):
        return {"EigenVectors": pd.DataFrame(np.ones((6, 3)), index=data.index)}

    def fake_multiscale(diffusion, **kwargs):
        return pd.DataFrame(
            np.arange(12).reshape(6, 2), index=adata.obs_names, columns=["d1", "d2"]
        )

    def fake_palantir(multiscale, **kwargs):
        assert kwargs["early_cell"] == "c1"
        assert kwargs["terminal_states"] == ["c5", "c4"]
        return SimpleNamespace(
            pseudotime=pd.Series(np.linspace(0, 1, 6), index=adata.obs_names),
            entropy=pd.Series(np.linspace(1, 0, 6), index=adata.obs_names),
            branch_probs=pd.DataFrame(
                np.column_stack([np.linspace(1, 0, 6), np.linspace(0, 1, 6)]),
                index=adata.obs_names,
                columns=["left", "right"],
            ),
            terminal_states=["c5", "c4"],
            waypoints=["c0", "c3"],
        )

    monkeypatch.setattr(trajectory_module.palantir.utils, "run_diffusion_maps", fake_diffusion)
    monkeypatch.setattr(
        trajectory_module.palantir.utils,
        "determine_multiscale_space",
        fake_multiscale,
    )
    monkeypatch.setattr(trajectory_module.palantir.core, "run_palantir", fake_palantir)

    result = run_palantir(
        adata,
        start_idx=1,
        terminal_states=[5, "c4"],
        n_diff_components=2,
        store_prefix="pt",
    )

    assert result is adata
    assert adata.uns["pt_branch_names"] == ["left", "right"]
    assert adata.obsm["pt_ms_data"].shape == (6, 2)
    assert adata.uns["pt_params"]["early_cell_internal"] == "c1"
    assert adata.uns["pt_terminal_states_internal"] == ["c5", "c4"]
    assert adata.uns["pt_waypoints_internal"] == ["c0", "c3"]
    assert adata.uns["pt_cell_index_map"]["obs_pos"].tolist() == list(range(6))


def test_metabolite_trends_uses_feature_names_and_sum_pooling() -> None:
    adata = make_adata()
    adata.var["display_name"] = ["rising", "falling", "constant"]

    metabolite_trends(
        adata,
        window_size=2,
        step_size=2,
        kernel_stat="sum",
        feature_name_key="display_name",
    )

    result = adata.uns["metabolite_trends"]
    np.testing.assert_allclose(result["pooled"][0], adata.X[:2].sum(axis=0))
    assert result["feature_names"].tolist() == ["rising", "falling", "constant"]
    assert np.isfinite(result["qval"][:2]).all()


def test_trend_cluster_rejects_ward_with_correlation_distance() -> None:
    trends = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])

    with pytest.raises(ValueError, match="ward linkage"):
        trend_cluster(
            trends,
            metric="correlation",
            cluster_method="agglomerative",
            n_clusters=2,
            linkage="ward",
        )
