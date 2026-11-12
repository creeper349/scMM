import numpy as np
import pandas as pd

from scMM.analysis import SupervisedAnalyzer
from scMM.file.data import CyESIData


def _classification_data():
    rng = np.random.default_rng(13)
    X = np.vstack(
        [rng.normal(0, 0.4, (30, 6)), rng.normal(2, 0.4, (30, 6))]
    ).astype(np.float32)
    return CyESIData(
        X=X,
        obs=pd.DataFrame(
            {"group": ["control"] * 30 + ["treated"] * 30},
            index=[f"cell_{i}" for i in range(60)],
        ),
        var=pd.DataFrame(
            {"mz": np.arange(100, 106, dtype=float)},
            index=[str(value) for value in range(100, 106)],
        ),
    )


def test_logistic_evaluation_and_roc():
    analyzer = SupervisedAnalyzer(_classification_data(), "group")
    result = analyzer.evaluate("logistic", cv=3, test_size=0.25)
    assert result["test_metrics"]["balanced_accuracy"] > 0.9
    assert result["confusion_matrix"].shape == (2, 2)
    curves = analyzer.roc_curves()
    assert set(curves) == {"control", "treated"}
    assert set(analyzer.precision_recall_curves()) == {"control", "treated"}
    assert set(analyzer.calibration_curves()) == {"control", "treated"}
    assert {"r2", "q2", "pr_auc_macro", "brier_macro"} <= set(
        result["model_qc"]
    )


def test_plsda_uses_classifier_interface():
    data = _classification_data()
    analyzer = SupervisedAnalyzer(data, "group")
    result = analyzer.evaluate(
        "plsda", cv=3, model_params={"n_components": 2}
    )
    assert result["test_metrics"]["accuracy"] > 0.9
    key, columns = analyzer.store_latent_scores()
    assert key == "X_plsda"
    assert data.obsm[key].shape == (data.n_obs, 2)
    assert columns == ["PLSDA1", "PLSDA2"]
    assert columns[0] in data.obs


def test_lda_binary_scores_are_plot_ready():
    data = _classification_data()
    analyzer = SupervisedAnalyzer(data, "group")
    analyzer.evaluate("lda", cv=3)
    key, columns = analyzer.store_latent_scores()
    assert key == "X_lda"
    assert columns == ["LDA1"]
    assert data.obsm[key].shape == (data.n_obs, 2)
    assert np.allclose(data.obsm[key][:, 1], 0)


def test_shap_native_plots_and_svg_export(tmp_path):
    import matplotlib.pyplot as plt

    analyzer = SupervisedAnalyzer(_classification_data(), "group")
    analyzer.evaluate("logistic", cv=2, test_size=.25)
    analyzer.explain_shap(max_background=10, max_samples=8)
    assert analyzer.shap_output_names() == ["treated (log-odds)"]
    for kind in ["bar", "beeswarm", "dependence"]:
        fig = analyzer.shap_figure(plot_type=kind, max_display=4, feature="100", color_feature="101")
        assert fig.axes
        plt.close(fig)
        path = analyzer.plot_shap(tmp_path / f"shap_{kind}.svg", plot_type=kind, max_display=4)
        svg = path.read_text(encoding="utf-8")
        assert "<svg" in svg and "<text" in svg


def test_shap_multiclass_output_selection():
    import matplotlib.pyplot as plt

    data = _classification_data()
    data.obs["group"] = ["a"] * 20 + ["b"] * 20 + ["c"] * 20
    analyzer = SupervisedAnalyzer(data, "group")
    analyzer.evaluate("random_forest", cv=2, model_params={"n_estimators": 10})
    explanation = analyzer.explain_shap(max_background=10, max_samples=5)
    assert explanation.values.shape == (5, 6, 3)
    assert analyzer.shap_output_names() == ["a", "b", "c"]
    fig = analyzer.shap_figure(plot_type="bar", class_index=2, max_display=3)
    plt.close(fig)
