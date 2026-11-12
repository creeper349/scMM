import numpy as np
import pandas as pd

from anndata import AnnData

from scMM.file.data import CyESIData


def test_cyesidata_is_anndata_and_roundtrips_h5ad(tmp_path):
    data = CyESIData(
        X=np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"source_file": ["a.mzML", "b.mzML"]},
            index=["cell_0", "cell_1"],
        ),
        var=pd.DataFrame(
            {"mz": [100.0, 200.0]},
            index=["100.0", "200.0"],
        ),
        uns={
            "files": {
                "a.mzML": {"instrument": "test"},
                "b.mzML": {"instrument": "test"},
            }
        },
    )

    assert isinstance(data, AnnData)
    output = data.save(tmp_path / "result")
    restored = CyESIData.read(output)

    assert output.suffix == ".h5ad"
    assert restored.shape == data.shape
    assert restored.obs["source_file"].tolist() == ["a.mzML", "b.mzML"]
    np.testing.assert_allclose(restored.X, data.X)


def test_assign_source_metadata():
    data = CyESIData(
        X=np.ones((3, 1), dtype=np.float32),
        obs=pd.DataFrame(
            {"source_file": ["a.mzML", "a.mzML", "b.mzML"]},
            index=["a0", "a1", "b0"],
        ),
        var=pd.DataFrame({"mz": [100.0]}, index=["100.0"]),
    )

    data.assign_source_metadata(
        {
            "a.mzML": {"group": "control", "batch": 1},
            "b.mzML": {"group": "treated", "batch": 2},
        }
    )

    assert data.obs["group"].tolist() == ["control", "control", "treated"]
    assert data.obs["batch"].tolist() == [1, 1, 2]


def test_reversible_cell_and_feature_filters_use_uint8():
    data = CyESIData(
        X=np.asarray([[1, 0, 2], [0, 3, 0], [4, 5, 0]], dtype=np.float32),
        obs=pd.DataFrame(
            {"source_file": ["a.raw", "a.raw", "b.raw"]},
            index=["c0", "c1", "c2"],
        ),
        var=pd.DataFrame(
            {"mz": [100.0, 200.0, 300.0]},
            index=["f0", "f1", "f2"],
        ),
    )

    assert data.obs["detected_features"].tolist() == [2, 1, 2]
    assert data.obs["raw_total_intensity"].tolist() == [3.0, 3.0, 9.0]
    cell_column = data.set_cell_filter("gate", [1, 0, 1])
    feature_column, rates = data.label_features_by_detection(
        0.75, source_file="a.raw", key="detected_in_a"
    )

    assert data.obs[cell_column].dtype == np.dtype("uint8")
    assert data.var[feature_column].dtype == np.dtype("uint8")
    assert data.cell_keep_mask().tolist() == [True, False, True]
    np.testing.assert_allclose(rates, [1.0, 0.0, 1.0])
    assert data.feature_keep_mask().tolist() == [True, False, True]
    assert data.active_view().shape == (2, 2)

    data.restore_cell_filter(cell_column)
    data.restore_feature_filter(feature_column)
    assert data.cell_keep_mask().all()
    assert data.feature_keep_mask().all()


def test_isolation_forest_labels_without_removing_cells():
    rng = np.random.default_rng(4)
    data = CyESIData(
        X=rng.normal(size=(30, 4)).astype(np.float32),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(30)]),
        var=pd.DataFrame(index=[f"f{i}" for i in range(4)]),
    )
    data.remove_outlier(key="iforest", contamination=0.1, random_state=1)

    assert data.n_obs == 30
    assert "cell_keep_iforest" in data.obs
    assert set(data.obs["cell_keep_iforest"].unique()) <= {0, 1}


def test_processing_layers_preserve_inputs_and_roundtrip(tmp_path):
    original = np.asarray([[1., 0.], [3., 2.]], dtype=np.float32)
    data = CyESIData(X=original.copy(), obs=pd.DataFrame(index=["a", "b"]),
                    var=pd.DataFrame({"mz": [100., 200.]}, index=["f1", "f2"]))
    data.normalize("total", scale=1.)
    assert data.uns["active_layer"] == "normalized_total"
    np.testing.assert_allclose(data.layers["raw"], original)
    normalized = data.X.copy()
    np.testing.assert_allclose(normalized.sum(axis=1), 1)
    data.set_active_layer("raw")
    data.normalize("total", scale=2.)
    assert data.uns["active_layer"] == "normalized_total_2"
    np.testing.assert_allclose(data.layers["normalized_total"], normalized)
    data.set_active_layer("raw")
    data.impute("constant", fill_value=9.)
    np.testing.assert_allclose(data.layers["raw"], original)
    np.testing.assert_allclose(data.X, [[1, 9], [3, 2]])
    assert data.uns["matrix_layers"]["imputed_constant"]["source"] == "raw"
    restored = CyESIData.read(data.save(tmp_path / "layers.h5ad"))
    assert restored.uns["active_layer"] == "imputed_constant"
    np.testing.assert_allclose(restored.layers["raw"], original)
    restored.set_active_layer("normalized_total")
    np.testing.assert_allclose(restored.X, normalized)
    assert restored.obs.raw_total_intensity.tolist() == [1., 5.]


def test_frame_id_is_mandatory_and_summed_spectrum_roundtrips(tmp_path):
    frame_meta = pd.DataFrame(
        {"rt": [0.1, 0.2], "ms_level": [1, 1], "native_id": ["a", "b"]},
        index=pd.Index([3, 7], name="frame"),
    )
    obs = CyESIData._frame_obs(
        frame_meta,
        tmp_path / "sample.mzML",
        {},
        frame_metadata=["rt"],
        metadata_by_file=None,
    )
    assert obs["frame_id"].tolist() == [3, 7]

    data = CyESIData(
        X=np.ones((2, 1), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame({"mz": [100.0]}, index=["100.0"]),
        uns={
            "summed_spectra": {
                "sample.mzML": {
                    "mz": np.asarray([99.9, 100.0, 100.1]),
                    "intensity": np.asarray([0.0, 5.0, 0.0], dtype=np.float32),
                    "candidate_mz": np.asarray([100.0]),
                    "retained_feature_mz": np.asarray([100.0]),
                    "ms_level": 1,
                }
            }
        },
    )
    restored = CyESIData.read(data.save(tmp_path / "summed.h5ad"))
    payload = restored.uns["summed_spectra"]["sample.mzML"]
    np.testing.assert_allclose(payload["intensity"], [0.0, 5.0, 0.0])
    assert int(payload["ms_level"]) == 1
