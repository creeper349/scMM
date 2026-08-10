import numpy as np
import pyopenms as oms
import pytest

from scMM.file.io import (
    _prepare_sorted_unique_peaks,
    align_frame,
    build_orbitrap_grid,
    extract_peaks,
    load_single_file,
    save_spectra,
    sum_spec,
)


def make_experiment() -> oms.MSExperiment:
    experiment = oms.MSExperiment()
    for retention_time, scale in [(0.0, 1.0), (1.0, 2.0)]:
        spectrum = oms.MSSpectrum()
        spectrum.setMSLevel(1)
        spectrum.setRT(retention_time)
        spectrum.set_peaks(
            (
                np.array([99.9, 100.0, 100.1]),
                np.array([0.0, 10.0 * scale, 0.0], dtype=np.float32),
            )
        )
        experiment.addSpectrum(spectrum)
    return experiment


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


def test_sum_extract_and_align_synthetic_spectra() -> None:
    experiment = make_experiment()

    summed = sum_spec(
        experiment,
        mz_range=(99.8, 100.2),
        resolution_200=1_000,
        points_per_fwhm=2,
    )
    peak_mz, peak_intensity = extract_peaks(summed, resolution_200=1_000, distance=1)
    frame, metadata = align_frame(experiment, [100.0], ppm=20, resolution_200=1_000, distance=1)

    assert len(peak_mz) == 1
    assert peak_mz[0] == pytest.approx(100.0, abs=0.02)
    assert peak_intensity[0] > 0
    np.testing.assert_allclose(frame[100.0], [10.0, 20.0])
    np.testing.assert_allclose(metadata["rt"], [0.0, 1.0])


def test_mzml_save_load_round_trip_uses_timestamp_fallback(tmp_path) -> None:
    output = tmp_path / "synthetic.mzML"

    save_spectra(list(make_experiment()), output)
    experiment, metadata = load_single_file(output, format="auto")

    assert experiment.getNrSpectra() == 2
    assert metadata["name"] == "synthetic"
    assert metadata["timestamp"] == pytest.approx(output.stat().st_mtime, abs=1.0)
