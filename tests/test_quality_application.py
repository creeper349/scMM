from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scMM.application import build_quality_report, load_quality_report, save_quality_report
from scMM.file.data import CyESIData


def _dataset() -> CyESIData:
    dataset = object.__new__(CyESIData)
    dataset.data = pd.DataFrame(
        [[1.0, 0.0, 3.0], [2.0, 4.0, 0.0], [3.0, 5.0, 6.0], [4.0, 0.0, 8.0]],
        columns=[100.0, 200.0, 300.0],
    )
    dataset.peak_meta = pd.DataFrame({"rt": [1.0, 2.0, 3.0, 4.0], "label": ["a", "a", "b", "b"]})
    dataset.feature_meta = pd.DataFrame(index=dataset.data.columns)
    dataset.file_meta = {"name": "sample", "ref_mz": 100.0}
    dataset.ref_mz = 100.0
    return dataset


def test_quality_report_summarizes_cells_features_and_embeddings() -> None:
    fake_umap = np.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
    with patch("scMM.application.quality._umap_embedding", return_value=fake_umap):
        report = build_quality_report(_dataset())

    assert report.summary.cell_count == 4
    assert report.summary.feature_count == 3
    assert report.summary.zero_fraction == 0.25
    assert report.cells["total_intensity"].tolist() == [4.0, 6.0, 14.0, 12.0]
    assert report.features["detection_rate"].tolist() == [1.0, 0.5, 0.75]
    assert list(report.embedding) == ["cell_index", "PCA1", "PCA2", "UMAP1", "UMAP2"]


def test_quality_report_round_trip(tmp_path: Path) -> None:
    with patch(
        "scMM.application.quality._umap_embedding",
        return_value=np.zeros((4, 2)),
    ):
        report = build_quality_report(_dataset())

    save_quality_report(report, tmp_path)
    loaded = load_quality_report(tmp_path)

    assert loaded.summary == report.summary
    pd.testing.assert_frame_equal(loaded.cells, report.cells)
    pd.testing.assert_frame_equal(loaded.features, report.features)
    assert (tmp_path / "embedding.csv").is_file()


def test_quality_report_handles_tiny_dataset_without_failing() -> None:
    dataset = _dataset()
    dataset.data = dataset.data.iloc[:1, :1]
    dataset.peak_meta = dataset.peak_meta.iloc[:1]

    report = build_quality_report(dataset)

    assert list(report.embedding) == ["cell_index"]
    assert "PCA requires" in report.summary.embedding_warnings[0]


def test_quality_report_handles_constant_and_empty_feature_matrices() -> None:
    dataset = _dataset()
    dataset.data = pd.DataFrame(np.ones((3, 2)), columns=[100.0, 200.0])
    dataset.peak_meta = dataset.peak_meta.iloc[:3]

    constant = build_quality_report(dataset, include_umap=False)
    assert np.allclose(constant.embedding[["PCA1", "PCA2"]], 0)
    assert "constant" in constant.summary.embedding_warnings[0]

    dataset.data = pd.DataFrame(index=range(3))
    empty = build_quality_report(dataset)
    assert empty.summary.feature_count == 0
    assert empty.features.empty


def test_quality_report_caps_embedding_work_deterministically() -> None:
    dataset = _dataset()
    with patch(
        "scMM.application.quality._umap_embedding",
        return_value=np.zeros((3, 2)),
    ):
        report = build_quality_report(
            dataset,
            max_embedding_cells=3,
            max_embedding_features=2,
        )

    assert report.embedding["cell_index"].tolist() == [0, 1, 3]
    assert report.embedding.shape == (3, 5)
    assert any("sample" in warning for warning in report.summary.embedding_warnings)
    assert any("variable features" in warning for warning in report.summary.embedding_warnings)
