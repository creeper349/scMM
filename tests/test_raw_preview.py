import numpy as np
import pyopenms as oms

from scMM.analysis.raw_preview import chromatogram_preview, spectrum_preview
from scMM.file.io import align_frame, sum_spec


def _write_mzml(path):
    experiment = oms.MSExperiment()
    for index in range(3):
        spectrum = oms.MSSpectrum()
        spectrum.setMSLevel(1)
        spectrum.setRT(float(index) * 0.5)
        spectrum.set_peaks(
            (
                np.asarray([100.0, 200.0], dtype=float),
                np.asarray([index + 1.0, (index + 1.0) * 10.0], dtype=float),
            )
        )
        experiment.addSpectrum(spectrum)
    oms.MzMLFile().store(str(path), experiment)


def test_interactive_raw_preview_data(tmp_path):
    path = tmp_path / "preview.mzML"
    _write_mzml(path)
    updates = []
    frame, metadata = chromatogram_preview(
        path,
        [100.0],
        ppm=10,
        progress_callback=lambda value, message: updates.append((value, message)),
    )
    assert frame["tic"].tolist() == [11.0, 22.0, 33.0]
    assert frame["eic_100"].tolist() == [1.0, 2.0, 3.0]
    assert frame["frame_id"].tolist() == [0, 1, 2]
    assert metadata["source_file"] == path.name
    assert updates

    spectrum, rt, count, _ = spectrum_preview(path, 1)
    assert count == 3
    assert rt == 0.5
    np.testing.assert_allclose(spectrum["intensity"], [2.0, 20.0])


def test_ms2_frames_are_excluded_and_original_frame_ids_are_preserved(tmp_path):
    path = tmp_path / "mixed.mzML"
    experiment = oms.MSExperiment()
    for frame_id, ms_level in enumerate([1, 2, 1]):
        spectrum = oms.MSSpectrum()
        spectrum.setMSLevel(ms_level)
        spectrum.setRT(float(frame_id))
        spectrum.setNativeID(f"scan={frame_id + 1}")
        spectrum.set_peaks(
            (
                np.asarray([100.0], dtype=float),
                np.asarray([frame_id + 1.0], dtype=float),
            )
        )
        experiment.addSpectrum(spectrum)
    oms.MzMLFile().store(str(path), experiment)

    preview, _ = chromatogram_preview(path, [100.0])
    assert preview["frame_id"].tolist() == [0, 2]
    assert preview["tic"].tolist() == [1.0, 3.0]

    aligned, metadata = align_frame(experiment, [100.0], ppm=10, ms_level=1)
    assert aligned.index.tolist() == [0, 2]
    assert metadata.index.tolist() == [0, 2]
    assert metadata["ms_level"].tolist() == [1, 1]
    assert metadata["native_id"].tolist() == ["scan=1", "scan=3"]

    summed = sum_spec(
        experiment,
        mz_range=(100.0, 100.1),
        resolution_200=1000,
        points_per_fwhm=2,
        ms_level=1,
    )
    _, summed_intensity = summed.get_peaks()
    # The two MS1 frames sum to 4.  The interleaved MS2 intensity must not
    # contribute to the processing spectrum.
    assert float(np.max(summed_intensity)) == 4.0
