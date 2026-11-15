import numpy as np
import pyopenms as oms
import pytest

from scMM.file.io import align_frame, extract_peaks


def spectrum(intensity):
    spec = oms.MSSpectrum()
    spec.setMSLevel(1)
    spec.set_peaks((760.0 + np.arange(len(intensity)) * 0.005,
                    np.asarray(intensity, dtype=float)))
    return spec


@pytest.mark.parametrize("mode", ["apex", "sum"])
def test_overlapping_centroid_windows_split_at_valley(mode):
    spec = spectrum([0, 0, 4, 10, 4, 1, 5, 12, 5, 0, 0])
    mz, intensity = spec.get_peaks()
    centers, heights = extract_peaks(spec, centroid_intensity_mode=mode)
    expected = [np.average(mz[:6], weights=intensity[:6]),
                np.average(mz[6:], weights=intensity[6:])]
    np.testing.assert_allclose(centers, expected, rtol=0, atol=1e-12)
    assert centers[0] < mz[5] < centers[1]
    np.testing.assert_allclose(heights, [10, 12] if mode == "apex" else [19, 22])

    exp = oms.MSExperiment()
    exp.addSpectrum(spec)
    aligned, _ = align_frame(exp, centers, ppm=1, resolution_200=35000)
    np.testing.assert_allclose(aligned.iloc[0], [10, 12])


def test_flat_valley_is_split_deterministically_without_double_counting():
    spec = spectrum([0, 10, 2, 2, 2, 12, 0])
    mz, intensity = spec.get_peaks()
    centers, heights = extract_peaks(spec, centroid_intensity_mode="sum")
    np.testing.assert_allclose(centers, [np.average(mz[:4], weights=intensity[:4]),
                                       np.average(mz[4:], weights=intensity[4:])])
    np.testing.assert_allclose(heights, [14, 14])
    assert heights.sum() == intensity.sum()


def test_off_center_plateau_uses_its_own_midpoint():
    spec = spectrum([0, 10, 1, 1, 1, 2, 3, 4, 12, 0])
    _, heights = extract_peaks(spec, centroid_intensity_mode="sum")
    np.testing.assert_allclose(heights, [12, 22])


def test_isolated_peak_preserves_centroid_and_fwhm_limit():
    # At m/z 760, the default half-window is nine points on this grid.
    spec = spectrum(np.r_[np.ones(15), [3, 10, 5], np.ones(15)])
    mz, intensity = spec.get_peaks()
    centers, heights = extract_peaks(spec, centroid_intensity_mode="sum")
    np.testing.assert_allclose(centers, [np.average(mz[7:26], weights=intensity[7:26])])
    np.testing.assert_allclose(heights, [intensity[7:26].sum()])


def test_distant_peaks_keep_fwhm_limits_and_parabola_is_unchanged():
    intensity = np.ones(61)
    intensity[9:12] = [4, 10, 4]
    intensity[49:52] = [5, 12, 5]
    spec = spectrum(intensity)
    mz, _ = spec.get_peaks()
    centers, heights = extract_peaks(spec, centroid_intensity_mode="sum")
    np.testing.assert_allclose(centers, mz[[10, 50]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(heights, [34, 38])
    parabola, apex = extract_peaks(spec, method="parabola")
    np.testing.assert_allclose(parabola, mz[[10, 50]], rtol=0, atol=1e-12)
    np.testing.assert_allclose(apex, [10, 12])
