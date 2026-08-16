import numpy as np
import pandas as pd
import pyopenms as oms
import pytest

from scMM.util.peak import filter_spectrum, find_cell_peaks


def make_spectrum(intensity: list[float]) -> oms.MSSpectrum:
    spectrum = oms.MSSpectrum()
    spectrum.setMSLevel(1)
    spectrum.setRT(12.5)
    spectrum.setName("synthetic")
    spectrum.set_peaks(
        (
            np.arange(100.0, 100.0 + len(intensity)),
            np.asarray(intensity, dtype=np.float32),
        )
    )
    return spectrum


def constant_baseline(values: np.ndarray, size, **kwargs) -> np.ndarray:
    return np.ones_like(values)


def test_filter_spectrum_preserves_identity_and_returns_snr() -> None:
    spectrum = make_spectrum([1.0, 1.0, 10.0, 1.0, 1.0])

    filtered, snr = filter_spectrum(
        spectrum,
        baseline_window=3,
        noise_window=3,
        baseline_stride=1,
        snr_threshold=3.0,
        return_snr=True,
    )

    _, intensity = filtered.get_peaks()
    np.testing.assert_allclose(intensity, [0.0, 0.0, 9.0, 0.0, 0.0])
    assert snr[2] > 3.0
    assert filtered.getRT() == 12.5
    assert filtered.getMSLevel() == 1
    assert filtered.getName() == "synthetic"


def test_filter_spectrum_handles_empty_input() -> None:
    spectrum = make_spectrum([])

    filtered, snr = filter_spectrum(spectrum, return_snr=True)

    assert filtered.size() == 0
    assert snr.size == 0
    assert filtered.getRT() == 12.5


def test_find_cell_peaks_extracts_windows_and_filters_sparse_features() -> None:
    data = pd.DataFrame(
        [
            [1.0, 0.0, 0.0],
            [6.0, 4.0, 0.0],
            [7.0, 2.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [8.0, 0.0, 5.0],
            [1.0, 0.0, 0.0],
        ],
        columns=[100.0, 200.0, 300.0],
    )

    result = find_cell_peaks(
        data,
        ref_mz=100.0,
        baseline_filter=constant_baseline,
        baseline_filter_size=3,
        cell_snr=5.0,
        peak_snr=3.0,
        max_zero_frac=0.4,
        n_jobs=1,
    )

    assert result["window_ranges"] == [(1, 2), (5, 5)]
    np.testing.assert_array_equal(result["peak_frames"], [2, 5])
    assert result["cell_df"].columns.tolist() == [100.0]
    np.testing.assert_allclose(result["cell_df"][100.0], [7.0, 8.0])
    assert result["zero_frac"].to_dict() == {100.0: 0.0, 200.0: 0.5, 300.0: 0.5}


def test_find_cell_peaks_returns_stable_empty_schema() -> None:
    data = pd.DataFrame([[1.0, 2.0], [1.0, 3.0]], columns=[100.0, 200.0])

    result = find_cell_peaks(
        data,
        ref_mz=100.0,
        baseline_filter=constant_baseline,
        cell_snr=5.0,
        n_jobs=1,
    )

    assert result["cell_df"].empty
    assert result["cell_df"].columns.tolist() == [100.0, 200.0]
    assert result["peak_frames"].dtype == int
    assert not result["kept_columns"].any()


def test_find_cell_peaks_rejects_unknown_baseline_stat_when_cells_exist() -> None:
    data = pd.DataFrame([[6.0, 2.0], [7.0, 3.0]], columns=[100.0, 200.0])

    with pytest.raises(ValueError, match="baseline_stat"):
        find_cell_peaks(
            data,
            ref_mz=100.0,
            baseline_filter=constant_baseline,
            baseline_stat="minimum",
            n_jobs=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"baseline_quantile": 1.1}, "baseline_quantile"),
        ({"snr_threshold": -1.0}, "snr_threshold"),
    ],
)
def test_filter_spectrum_validates_thresholds(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        filter_spectrum(make_spectrum([1.0, 2.0, 1.0]), **kwargs)


def test_find_cell_peaks_validates_empty_data() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        find_cell_peaks(pd.DataFrame(), ref_mz=100.0)
