import numpy as np
import pandas as pd

from scMM.file.data import CyESIData


def _object(source, mz, values):
    X = np.asarray(values, dtype=np.float32)
    return CyESIData(
        X=X,
        obs=pd.DataFrame(
            {"source_file": [source] * len(X)},
            index=[f"{source}_{i}" for i in range(len(X))],
        ),
        var=pd.DataFrame(
            {"mz": mz}, index=[f"{value:.8f}" for value in mz]
        ),
        layers={"feature_snr": X.copy()},
        uns={"files": {source: {"source_file": source}}},
    )


def test_independent_feature_union_merges_by_ppm_and_keeps_unique_features():
    first = _object("a.mzML", [100.0, 200.0], [[1, 2], [3, 4]])
    second = _object("b.mzML", [100.0003, 300.0], [[5, 6]])
    merged = CyESIData._merge_independent_objects(
        [first, second], ppm_tolerance=5.0
    )

    assert merged.shape == (3, 3)
    np.testing.assert_allclose(merged.var["mz"], [100.00015, 200.0, 300.0])
    assert merged.var["source_file_count"].tolist() == [2, 1, 1]
    assert merged.obs["source_file"].tolist() == ["a.mzML", "a.mzML", "b.mzML"]
