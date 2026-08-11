from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scMM.file.data import CyESIData


def make_dataset(
    name: str,
    values: list[list[float]],
    columns: list[float],
    *,
    labels: list[str] | None = None,
) -> CyESIData:
    obj = object.__new__(CyESIData)
    index = pd.Index([f"{name}-{idx}" for idx in range(len(values))])
    obj.data = pd.DataFrame(values, index=index, columns=columns)
    obj.peak_meta = pd.DataFrame(
        {"label": labels or [name] * len(values), "rt": np.arange(len(values))},
        index=index,
    )
    obj.feature_meta = pd.DataFrame(
        {"mz": columns, "source": name}, index=pd.Index(columns, name="feature_id")
    )
    obj.file_meta = {"name": name, "ref_mz": columns[0]}
    obj.ref_mz = columns[0]
    return obj


def test_alignwith_union_retains_values_and_metadata() -> None:
    left = make_dataset("left", [[1.0, 2.0]], [100.0, 200.0])
    right = make_dataset("right", [[10.0, 30.0]], [100.0004, 300.0])

    result = left.alignwith(right, ppm_tol=5.0, mz_merge_options="union")

    assert result is left
    assert list(result.data.columns) == [100.0, 200.0, 300.0]
    np.testing.assert_allclose(result.data.to_numpy(), [[1.0, 2.0, 0.0], [10.0, 0.0, 30.0]])
    assert list(result.feature_meta.index) == [100.0, 200.0, 300.0]
    assert result.feature_meta.loc[300.0, "source"] == "right"
    assert result.get_name() == "left+right"
    assert len(result.file_meta["per_file_meta"]) == 2


def test_alignwith_ref_retains_reference_axis() -> None:
    left = make_dataset("left", [[1.0, 2.0]], [100.0, 200.0])
    right = make_dataset("right", [[10.0, 30.0]], [100.0004, 300.0])

    left.alignwith(right, ppm_tol=5.0, mz_merge_options="ref")

    assert list(left.data.columns) == [100.0, 200.0]
    np.testing.assert_allclose(left.data.to_numpy(), [[1.0, 2.0], [10.0, 0.0]])


def test_normalize_and_impute_preserve_index() -> None:
    dataset = make_dataset("sample", [[1.0, 3.0], [0.0, 0.0]], [100.0, 200.0])
    original_index = dataset.data.index.copy()

    dataset.normalize().impute(method="mean", missing_values=0)

    pd.testing.assert_index_equal(dataset.data.index, original_index)


def test_save_load_round_trip(tmp_path: Path) -> None:
    dataset = make_dataset("sample", [[1.0, 2.0]], [100.0, 200.0])

    result_path = dataset.save(tmp_path)
    loaded = CyESIData.load_from_processed(result_path)

    pd.testing.assert_frame_equal(loaded.data, dataset.data)
    pd.testing.assert_frame_equal(loaded.peak_meta, dataset.peak_meta)
    pd.testing.assert_frame_equal(loaded.feature_meta, dataset.feature_meta)
    assert loaded.ref_mz == dataset.ref_mz
    with pytest.raises(FileExistsError):
        dataset.save(tmp_path)
    assert dataset.save(tmp_path, overwrite=True) == result_path


def test_to_anndata_uses_stable_unique_identifiers() -> None:
    dataset = make_dataset("sample", [[1.0, 2.0], [3.0, 4.0]], [100.0, 200.0])

    adata = dataset.to_anndata()

    assert adata.shape == (2, 2)
    assert list(adata.obs_names) == ["cell_0", "cell_1"]
    assert list(adata.var_names) == ["100.0", "200.0"]
    assert "source_index" in adata.obs


def test_deisotope_removes_correlated_isotope() -> None:
    parent = np.array([100.0, 200.0, 300.0, 400.0])
    dataset = make_dataset(
        "sample",
        np.column_stack([parent, parent * 0.05, [5.0, 2.0, 8.0, 1.0]]).tolist(),
        [100.0, 101.003355, 150.0],
    )

    result = dataset.deisotope(ppm_tol=1.0, r_square_threshold=0.99)

    assert result is dataset
    assert list(dataset.data.columns) == [100.0, 150.0]
    assert dataset.file_meta["deisotope"]["n_removed_features"] == 1
    assert dataset.feature_meta.loc[100.0, "n_isotope_children"] == 1


def test_deisotope_audit_does_not_mutate_dataset() -> None:
    parent = np.array([100.0, 200.0, 300.0, 400.0])
    dataset = make_dataset(
        "sample",
        np.column_stack([parent, parent * 0.05, parent * 0.002]).tolist(),
        [100.0, 101.003355, 102.00671],
    )
    original_data = dataset.data.copy()
    original_feature_meta = dataset.feature_meta.copy()
    original_file_meta = dataset.file_meta.copy()

    audit = dataset.deisotope(
        ppm_tol=1.0,
        r_square_threshold=0.99,
        merge_mode="sum",
        remove=False,
        inplace=False,
    )

    pd.testing.assert_frame_equal(dataset.data, original_data)
    pd.testing.assert_frame_equal(dataset.feature_meta, original_feature_meta)
    assert dataset.file_meta == original_file_meta
    assert not hasattr(dataset, "deisotope_result")

    assert audit["isotope_features"] == [101.003355, 102.00671]
    assert audit["final_table"]["isotope_order"].tolist() == [1, 2]
    np.testing.assert_allclose(
        audit["processed_data"][100.0],
        original_data[100.0] + original_data[101.003355] + original_data[102.00671],
    )
    assert audit["feature_meta"].loc[100.0, "n_isotope_children"] == 2
    assert audit["feature_meta"].loc[101.003355, "deisotope_role"] == "isotope"


def test_deisotope_builds_missing_feature_metadata_without_side_effects() -> None:
    dataset = make_dataset("sample", [[1.0, 0.05], [2.0, 0.1]], [100.0, 101.003355])
    del dataset.feature_meta

    audit = dataset.deisotope(ppm_tol=1.0, inplace=False)

    assert not hasattr(dataset, "feature_meta")
    assert audit["feature_meta"]["mz"].tolist() == [100.0]


def test_deisotope_rejects_invalid_missing_mask_shape() -> None:
    dataset = make_dataset("sample", [[1.0, 0.05], [2.0, 0.1]], [100.0, 101.003355])

    with pytest.raises(ValueError, match="same shape"):
        dataset.deisotope(missing_func=lambda values: values[:, 0], inplace=False)
