import numpy as np
import pytest

from scMM.file.io import _prepare_sorted_unique_peaks, build_orbitrap_grid


def test_prepare_peaks_sorts_filters_and_combines_duplicates() -> None:
    mz, intensity = _prepare_sorted_unique_peaks(
        np.array([200.0, np.nan, 100.0, 200.0]),
        np.array([1.0, 9.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(mz, [100.0, 200.0])
    np.testing.assert_allclose(intensity, [2.0, 4.0])


def test_orbitrap_grid_is_increasing_and_includes_bounds() -> None:
    grid = build_orbitrap_grid((100.0, 101.0), resolution_200=35_000, points_per_fwhm=5)

    assert grid[0] == 100.0
    assert grid[-1] == 101.0
    assert np.all(np.diff(grid) > 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mz_range": (0.0, 100.0)},
        {"resolution_200": 0.0},
        {"points_per_fwhm": 0.0},
    ],
)
def test_orbitrap_grid_validates_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        build_orbitrap_grid(**kwargs)
