import numpy as np
import pyopenms as oms
import pytest

from scMM.file.io import (
    InvalidMSFileError,
    _prepare_sorted_unique_peaks,
    align_frame,
    build_orbitrap_grid,
    extract_peaks,
    load_single_file,
    save_spectra,
    sum_spec,
)


def make_spectrum(
    mz: list[float],
    intensity: list[float],
    *,
    retention_time: float = 0.0,
    ms_level: int = 1,
) -> oms.MSSpectrum:
    spectrum = oms.MSSpectrum()
    spectrum.setMSLevel(ms_level)
    spectrum.setRT(retention_time)
    spectrum.set_peaks((np.asarray(mz), np.asarray(intensity, dtype=np.float32)))
    return spectrum


def make_experiment() -> oms.MSExperiment:
    experiment = oms.MSExperiment()
    for retention_time, scale in [(0.0, 1.0), (1.0, 2.0)]:
        experiment.addSpectrum(
            make_spectrum(
                [99.9, 100.0, 100.1],
                [0.0, 10.0 * scale, 0.0],
                retention_time=retention_time,
            )
        )
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


def test_mzml_save_load_round_trip_infers_format_and_uses_timestamp_fallback(tmp_path) -> None:
    output = tmp_path / "synthetic.mzML"

    save_spectra(list(make_experiment()), output)
    experiment, metadata = load_single_file(output)

    assert experiment.getNrSpectra() == 2
    assert metadata["name"] == "synthetic"
    assert metadata["timestamp"] == pytest.approx(output.stat().st_mtime, abs=1.0)


def test_load_single_file_infers_mzxml_by_default(tmp_path) -> None:
    output = tmp_path / "synthetic.mzXML"
    oms.MzXMLFile().store(str(output), make_experiment())

    experiment, metadata = load_single_file(output)

    assert experiment.getNrSpectra() == 2
    assert metadata["name"] == "synthetic"


def test_load_single_file_rejects_truncated_mzml_with_actionable_error(tmp_path) -> None:
    output = tmp_path / "truncated.mzML"
    output.write_text(
        '<?xml version="1.0"?><indexedmzML><mzML><spectrumList count="1">'
        '<spectrum id="scan=1"></spectrum></spectrumList></mzML>',
        encoding="utf-8",
    )

    with pytest.raises(InvalidMSFileError, match=r"XML 文档不完整.*重新转换"):
        load_single_file(output)


def test_load_single_file_rejects_mzml_without_spectra(tmp_path) -> None:
    output = tmp_path / "empty.mzML"
    output.write_text(
        '<?xml version="1.0"?><mzML><spectrumList count="0"></spectrumList></mzML>',
        encoding="utf-8",
    )

    with pytest.raises(InvalidMSFileError, match="没有质谱扫描"):
        load_single_file(output)


def test_extract_peaks_supports_centroid_intensity_modes() -> None:
    spectrum = make_spectrum([99.9, 100.0, 100.1], [1.0, 10.0, 2.0])

    apex_mz, apex_intensity = extract_peaks(
        spectrum,
        distance=1,
        centroid_intensity_mode="apex",
    )
    sum_mz, sum_intensity = extract_peaks(
        spectrum,
        distance=1,
        centroid_intensity_mode="sum",
    )

    np.testing.assert_allclose(apex_mz, sum_mz)
    assert apex_mz[0] > 100.0
    assert apex_intensity[0] == pytest.approx(10.0)
    assert sum_intensity[0] == pytest.approx(13.0)


def test_extract_peaks_refines_parabolic_vertex() -> None:
    spectrum = make_spectrum([99.9, 100.0, 100.1], [1.0, 4.0, 3.0])

    peak_mz, peak_intensity = extract_peaks(spectrum, method="parabola", distance=1)

    assert peak_mz[0] == pytest.approx(100.025)
    assert peak_intensity[0] == pytest.approx(4.125)


def test_align_frame_aggregates_multiple_peaks_per_target() -> None:
    experiment = oms.MSExperiment()
    experiment.addSpectrum(
        make_spectrum(
            [99.9996, 99.9998, 100.0, 100.0002, 100.0004],
            [0.0, 2.0, 0.0, 3.0, 0.0],
        )
    )

    summed, _ = align_frame(
        experiment,
        [100.0],
        ppm=5,
        aggregate="sum",
        method="parabola",
        distance=1,
        resolution_200=1e9,
    )
    maximum, _ = align_frame(
        experiment,
        [100.0],
        ppm=5,
        aggregate="max",
        method="parabola",
        distance=1,
        resolution_200=1e9,
    )

    assert summed.iloc[0, 0] == pytest.approx(5.0)
    assert maximum.iloc[0, 0] == pytest.approx(3.0)


def test_align_frame_restores_requested_target_order() -> None:
    frame, _ = align_frame(
        make_experiment(),
        [200.0, 100.0],
        ppm=20,
        resolution_200=1_000,
        distance=1,
    )

    assert frame.columns.tolist() == [200.0, 100.0]
    np.testing.assert_allclose(frame[200.0], 0.0)
    np.testing.assert_allclose(frame[100.0], [10.0, 20.0])
