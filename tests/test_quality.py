import numpy as np
import pandas as pd

from scMM.analysis import quality_metrics, render_quality_report
from scMM.file.data import CyESIData


def test_quality_metrics_and_pdf(tmp_path):
    data = CyESIData(
        X=np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32),
        obs=pd.DataFrame(
            {
                "source_file": ["a.mzML", "b.mzML"],
                "reference_intensity_100": [10.0, 20.0],
            },
            index=["a", "b"],
        ),
        var=pd.DataFrame(
            {"mz": [100.0, 200.0]}, index=["100.0", "200.0"]
        ),
    )
    metrics = quality_metrics(data)
    assert metrics["cell"]["detected_features"].tolist() == [1, 2]
    output = render_quality_report(data, tmp_path / "quality.pdf")
    assert output.exists()
    assert output.stat().st_size > 0
