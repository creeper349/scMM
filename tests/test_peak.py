import numpy as np
import pandas as pd
import pytest

from scMM.util.peak import find_cell_peaks


def _synthetic_frames():
    rng = np.random.default_rng(7)
    X = 10.0 + rng.normal(0.0, 0.25, size=(100, 4))
    X[20:23, 0] += [15.0, 30.0, 15.0]
    X[60:63, 0] += [15.0, 30.0, 15.0]
    X[60:63, 1] += [12.0, 25.0, 12.0]
    X[20:23, 2] += [8.0, 18.0, 8.0]
    X[60:63, 2] += [8.0, 18.0, 8.0]
    return pd.DataFrame(X, columns=[100.0, 200.0, 300.0, 400.0])


def test_multiple_references_union_and_intersection():
    frames = _synthetic_frames()
    common = dict(
        baseline_window=11,
        noise_window=15,
        cell_signal_threshold=2.0,
        feature_snr_threshold=5.0,
        max_zero_fraction=1.0,
        show_progress=False,
    )

    union = find_cell_peaks(
        frames, [100.0, 200.0], reference_mode="union", **common
    )
    intersection = find_cell_peaks(
        frames, [100.0, 200.0], reference_mode="intersection", **common
    )

    assert union["peak_frames"].tolist() == [21, 61]
    assert intersection["peak_frames"].tolist() == [61]


def test_feature_intensity_is_baseline_corrected_and_snr_filtered():
    frames = _synthetic_frames()
    progress = []
    result = find_cell_peaks(
        frames,
        100.0,
        baseline_window=11,
        noise_window=15,
        cell_signal_threshold=2.0,
        feature_snr_threshold=5.0,
        max_zero_fraction=1.0,
        show_progress=False,
        feature_block_size=2,
        progress_callback=lambda value, message: progress.append((value, message)),
    )

    corrected = result["cell_df"][300.0].to_numpy()
    raw_apex = frames.loc[result["peak_frames"], 300.0].to_numpy()
    assert np.all(corrected > 10.0)
    assert np.all(corrected < raw_apex)
    assert result["feature_snr"].shape == result["cell_df"].shape
    assert len(progress) == 2
    assert progress[-1][0] == 1.0


def test_reference_mz_requires_a_feature_inside_ppm_tolerance():
    frames = _synthetic_frames()
    with pytest.raises(ValueError, match="not detected within 10 ppm"):
        find_cell_peaks(
            frames,
            100.01,
            ppm_tolerance=10,
            show_progress=False,
        )
