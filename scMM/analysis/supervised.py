from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """PLS regression exposed through the sklearn classifier interface."""

    def __init__(self, n_components=2, max_iter=500):
        self.n_components = n_components
        self.max_iter = max_iter

    def fit(self, X, y):
        self.label_encoder_ = LabelEncoder().fit(y)
        encoded = self.label_encoder_.transform(y)
        self.classes_ = self.label_encoder_.classes_
        target = np.eye(len(self.classes_), dtype=float)[encoded]
        components = min(
            int(self.n_components), X.shape[1], X.shape[0] - 1
        )
        self.model_ = PLSRegression(
            n_components=max(1, components), max_iter=self.max_iter
        ).fit(X, target)
        return self

    def predict_proba(self, X):
        values = np.asarray(self.model_.predict(X), dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[1] == 1 and len(self.classes_) == 2:
            values = np.column_stack([-values[:, 0], values[:, 0]])
        return softmax(values, axis=1)

    def predict(self, X):
        indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[indices]

    def transform(self, X):
        """Return the PLS X scores used for PLS-DA visualisation."""
        return np.asarray(self.model_.transform(X), dtype=float)


class SupervisedAnalyzer:
    """Leakage-aware classification, validation, ROC and SHAP analysis."""

    def __init__(self, data, label_key, *, layer=None, random_state=42):
        if label_key not in data.obs:
            raise KeyError(f"{label_key!r} not found in data.obs")
        matrix = data.layers[layer] if layer else data.X
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        labels = data.obs[label_key]
        cell_keep = (
            data.cell_keep_mask()
            if hasattr(data, "cell_keep_mask")
            else np.ones(data.n_obs, dtype=bool)
        )
        feature_keep = (
            data.feature_keep_mask()
            if hasattr(data, "feature_keep_mask")
            else np.ones(data.n_vars, dtype=bool)
        )
        valid = labels.notna().to_numpy() & cell_keep
        self.data = data
        self.input_layer = str(layer or data.uns.get("active_layer", "X"))
        self.label_key = str(label_key)
        self.valid_mask = valid
        self.X = np.asarray(matrix, dtype=float)[valid][:, feature_keep]
        self.y = labels.astype(str).to_numpy()[valid]
        self.obs_names = data.obs_names[valid]
        self.feature_names = data.var_names[feature_keep].astype(str).tolist()
        self.random_state = int(random_state)
        if len(np.unique(self.y)) < 2:
            raise ValueError("supervised analysis requires at least two classes")

    def _estimator(self, model, params):
        params = dict(params)
        if model == "lda":
            return LinearDiscriminantAnalysis(**params), True
        if model in {"logistic", "lr"}:
            params.setdefault("max_iter", 5000)
            params.setdefault("class_weight", "balanced")
            params.setdefault("random_state", self.random_state)
            return LogisticRegression(**params), True
        if model in {"plsda", "pls-da"}:
            return PLSDAClassifier(**params), True
        if model in {"random_forest", "rf"}:
            params.setdefault("n_estimators", 500)
            params.setdefault("class_weight", "balanced")
            params.setdefault("random_state", self.random_state)
            params.setdefault("n_jobs", -1)
            return RandomForestClassifier(**params), False
        raise ValueError("model must be lda, plsda, logistic or random_forest")

    def build_pipeline(self, model="logistic", *, smote=False, model_params=None):
        estimator, needs_scale = self._estimator(model, model_params or {})
        steps = []
        if needs_scale:
            steps.append(("scale", StandardScaler()))
        if smote:
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.pipeline import Pipeline as ImbalancedPipeline
            except ImportError as exc:
                raise ImportError(
                    "SMOTE requires the optional imbalanced-learn dependency"
                ) from exc
            steps.append(("smote", SMOTE(random_state=self.random_state)))
            steps.append(("model", estimator))
            return ImbalancedPipeline(steps)
        steps.append(("model", estimator))
        return Pipeline(steps)

    def evaluate(
        self,
        model="logistic",
        *,
        test_size=0.2,
        cv=5,
        smote=False,
        model_params=None,
        calibration_bins=10,
    ):
        counts = pd.Series(self.y).value_counts()
        if counts.min() < 2:
            raise ValueError("every class needs at least two observations")
        X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
            self.X,
            self.y,
            self.obs_names,
            test_size=test_size,
            random_state=self.random_state,
            stratify=self.y,
        )
        folds = min(int(cv), int(pd.Series(y_train).value_counts().min()))
        if folds < 2:
            raise ValueError("training data cannot support two CV folds")
        pipeline = self.build_pipeline(
            model, smote=smote, model_params=model_params
        )
        splitter = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=self.random_state
        )
        scoring = {
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1_macro": "f1_macro",
            "roc_auc": "roc_auc_ovr_weighted",
        }
        cv_result = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            scoring=scoring,
            return_train_score=False,
            error_score=np.nan,
        )
        cv_probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=splitter,
            method="predict_proba",
            n_jobs=1,
        )
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)
        classes = np.asarray(pipeline.classes_)
        training_probabilities = pipeline.predict_proba(X_train)
        train_indicator = self._class_indicator(y_train, classes)
        test_indicator = self._class_indicator(y_test, classes)
        average_precision = {
            str(class_name): float(
                average_precision_score(
                    test_indicator[:, index], probabilities[:, index]
                )
            )
            for index, class_name in enumerate(classes)
        }
        calibration = {}
        brier_scores = {}
        for index, class_name in enumerate(classes):
            observed, predicted = calibration_curve(
                test_indicator[:, index],
                probabilities[:, index],
                n_bins=max(2, int(calibration_bins)),
                strategy="quantile",
            )
            calibration[str(class_name)] = {
                "mean_predicted_probability": predicted,
                "fraction_positive": observed,
            }
            brier_scores[str(class_name)] = float(
                brier_score_loss(
                    test_indicator[:, index], probabilities[:, index]
                )
            )
        metrics = {
            "accuracy": float(accuracy_score(y_test, prediction)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, prediction)
            ),
            "f1_macro": float(f1_score(y_test, prediction, average="macro")),
            "pr_auc_macro": float(np.mean(list(average_precision.values()))),
        }
        self.model_ = pipeline
        self.model_name_ = model
        self.classes_ = classes
        self.X_train_ = X_train
        self.X_test_ = X_test
        self.y_train_ = y_train
        self.y_test_ = y_test
        self.test_obs_names_ = names_test
        self.prediction_ = prediction
        self.probabilities_ = probabilities
        self.cv_probabilities_ = cv_probabilities
        self.training_probabilities_ = training_probabilities
        self.average_precision_ = average_precision
        self.calibration_ = calibration
        self.result_ = {
            "model": model,
            "smote": bool(smote),
            "test_metrics": metrics,
            "cv_metrics": {
                key.replace("test_", ""): {
                    "mean": float(np.nanmean(value)),
                    "std": float(np.nanstd(value)),
                }
                for key, value in cv_result.items()
                if key.startswith("test_")
            },
            "model_qc": {
                # Classification analogues calculated in one-hot probability
                # space. Q2 uses out-of-fold predictions and therefore measures
                # predictive performance rather than in-sample fit.
                "r2": float(
                    r2_score(
                        train_indicator,
                        training_probabilities,
                        multioutput="variance_weighted",
                    )
                ),
                "q2": float(
                    r2_score(
                        train_indicator,
                        cv_probabilities,
                        multioutput="variance_weighted",
                    )
                ),
                "pr_auc_macro": metrics["pr_auc_macro"],
                "brier_macro": float(np.mean(list(brier_scores.values()))),
                "average_precision_by_class": average_precision,
                "brier_by_class": brier_scores,
                "definition": (
                    "R2/Q2 are computed in one-hot class-probability space; "
                    "Q2 uses out-of-fold training predictions."
                ),
            },
            "confusion_matrix": confusion_matrix(
                y_test, prediction, labels=classes
            ),
            "classification_report": classification_report(
                y_test,
                prediction,
                labels=classes,
                output_dict=True,
                zero_division=0,
            ),
        }
        return self.result_

    @staticmethod
    def _class_indicator(labels, classes):
        binary = label_binarize(labels, classes=classes)
        if len(classes) == 2:
            binary = np.column_stack([1 - binary[:, 0], binary[:, 0]])
        return np.asarray(binary, dtype=float)

    def roc_curves(self):
        self._require_fitted()
        binary = self._class_indicator(self.y_test_, self.classes_)
        curves = {}
        for index, class_name in enumerate(self.classes_):
            fpr, tpr, thresholds = roc_curve(
                binary[:, index], self.probabilities_[:, index]
            )
            curves[str(class_name)] = {
                "fpr": fpr,
                "tpr": tpr,
                "thresholds": thresholds,
                "auc": float(auc(fpr, tpr)),
            }
        return curves

    def precision_recall_curves(self):
        """Return one-vs-rest precision-recall curves for the holdout set."""
        self._require_fitted()
        binary = self._class_indicator(self.y_test_, self.classes_)
        curves = {}
        for index, class_name in enumerate(self.classes_):
            precision, recall, thresholds = precision_recall_curve(
                binary[:, index], self.probabilities_[:, index]
            )
            curves[str(class_name)] = {
                "precision": precision,
                "recall": recall,
                "thresholds": thresholds,
                "average_precision": self.average_precision_[str(class_name)],
            }
        return curves

    def calibration_curves(self):
        """Return one-vs-rest holdout calibration curves."""
        self._require_fitted()
        return self.calibration_

    def latent_scores(self):
        """Calculate LDA or PLS-DA coordinates for all labelled observations."""
        self._require_fitted()
        canonical = str(self.model_name_).lower()
        if canonical not in {"lda", "plsda", "pls-da"}:
            raise ValueError("latent scores are only available for LDA and PLS-DA")
        values = self.X
        for name, step in self.model_.named_steps.items():
            if name in {"model", "smote"}:
                continue
            values = step.transform(values)
        estimator = self.model_.named_steps["model"]
        scores = np.asarray(estimator.transform(values), dtype=float)
        return scores[:, None] if scores.ndim == 1 else scores

    def store_latent_scores(self, key=None):
        """Write LDA/PLS-DA coordinates to AnnData ``obs`` and ``obsm``."""
        scores = self.latent_scores()
        prefix = "LDA" if str(self.model_name_).lower() == "lda" else "PLSDA"
        obsm_key = key or f"X_{prefix.lower()}"
        plot_width = max(2, scores.shape[1])
        stored = np.full((self.data.n_obs, plot_width), np.nan, dtype=float)
        stored[self.valid_mask, : scores.shape[1]] = scores
        if scores.shape[1] == 1:
            stored[self.valid_mask, 1] = 0.0
        self.data.obsm[obsm_key] = stored
        obs_columns = []
        for index in range(scores.shape[1]):
            column = f"{prefix}{index + 1}"
            values = np.full(self.data.n_obs, np.nan, dtype=float)
            values[self.valid_mask] = scores[:, index]
            self.data.obs[column] = values
            obs_columns.append(column)
        self.data.uns[f"{obsm_key}_metadata"] = {
            "method": prefix,
            "label_key": self.label_key,
            "obs_columns": obs_columns,
            "n_components": int(scores.shape[1]),
        }
        return obsm_key, obs_columns

    def plot_roc(self, output_file):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        for class_name, values in self.roc_curves().items():
            ax.plot(
                values["fpr"],
                values["tpr"],
                label=f"{class_name} (AUC={values['auc']:.3f})",
            )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend()
        from scMM.plot.ticks import tidy_numeric_ticks
        tidy_numeric_ticks(fig)
        fig.tight_layout()
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, bbox_inches="tight")
        plt.close(fig)
        return target

    def explain_shap(
        self,
        *,
        max_background=100,
        max_samples=100,
        seed=42,
    ):
        self._require_fitted()
        try:
            import shap
        except ImportError as exc:
            raise ImportError("SHAP requires the optional shap dependency") from exc
        rng = np.random.default_rng(seed)
        background_index = rng.choice(
            len(self.X_train_),
            size=min(max_background, len(self.X_train_)),
            replace=False,
        )
        sample_index = rng.choice(
            len(self.X_test_),
            size=min(max_samples, len(self.X_test_)),
            replace=False,
        )
        background = self.X_train_[background_index]
        samples = self.X_test_[sample_index]
        estimator = self.model_.named_steps["model"]
        transformed_background = background
        transformed_samples = samples
        for name, step in self.model_.named_steps.items():
            if name in {"model", "smote"}:
                continue
            transformed_background = step.transform(transformed_background)
            transformed_samples = step.transform(transformed_samples)

        if isinstance(estimator, RandomForestClassifier):
            explainer = shap.TreeExplainer(
                estimator,
                transformed_background,
                feature_names=self.feature_names,
            )
            explanation = explainer(transformed_samples)
        elif isinstance(estimator, LogisticRegression):
            explainer = shap.LinearExplainer(
                estimator,
                transformed_background,
                feature_names=self.feature_names,
            )
            explanation = explainer(transformed_samples)
        else:
            explainer = shap.Explainer(
                self.model_.predict_proba,
                background,
                feature_names=self.feature_names,
                output_names=self.classes_.astype(str).tolist(),
                seed=seed,
            )
            explanation = explainer(samples)
        self.shap_explanation_ = explanation
        self.shap_obs_names_ = self.test_obs_names_[sample_index]
        return explanation

    def shap_output_names(self):
        explanation = getattr(self, "shap_explanation_", None)
        if explanation is None:
            raise RuntimeError("请先计算 SHAP。")
        if np.asarray(explanation.values).ndim == 3:
            count = explanation.values.shape[2]
            return self.classes_.astype(str).tolist() if count == len(self.classes_) else [f"output {i}" for i in range(count)]
        estimator = self.model_.named_steps["model"]
        if isinstance(estimator, LogisticRegression) and len(self.classes_) == 2:
            return [f"{self.classes_[1]} (log-odds)"]
        return ["model output"]

    def shap_figure(self, *, plot_type="beeswarm", class_index=None, max_display=20,
                    feature=None, color_feature=None, width=9, height=6,
                    point_size=16, alpha=.8, cmap="coolwarm", font_size=10):
        """Draw a native SHAP bar, beeswarm or dependence plot from cached values."""
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import shap

        self.shap_output_names()
        explanation = self.shap_explanation_
        if np.asarray(explanation.values).ndim == 3:
            if class_index is None:
                class_index = 1 if explanation.values.shape[2] == 2 else 0
            explanation = explanation[..., int(class_index)]
        max_display = min(int(max_display), len(self.feature_names))
        with mpl.rc_context({"font.size": font_size, "svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=(width, height))
            try:
                if plot_type == "bar":
                    shap.plots.bar(explanation, max_display=max_display, ax=ax, show=False, show_data=False)
                elif plot_type == "beeswarm":
                    shap.plots.beeswarm(explanation, max_display=max_display, ax=ax, plot_size=None,
                                       s=point_size, alpha=alpha, color=mpl.colormaps[cmap], show=False)
                elif plot_type == "dependence":
                    feature_index = self.feature_names.index(str(feature)) if feature is not None else int(np.argmax(np.mean(np.abs(explanation.values), axis=0)))
                    color_index = self.feature_names.index(str(color_feature)) if color_feature is not None else feature_index
                    shap.plots.scatter(explanation[:, feature_index], color=explanation[:, color_index],
                                       dot_size=point_size, alpha=alpha, cmap=mpl.colormaps[cmap], ax=ax, show=False)
                else:
                    raise ValueError("plot_type must be bar, beeswarm or dependence")
                from scMM.plot.ticks import tidy_numeric_ticks
                tidy_numeric_ticks(fig)
                fig.tight_layout()
                return fig
            except Exception:
                plt.close(fig)
                raise

    def plot_shap(self, output_file, *, class_index=None, max_display=20, plot_type="beeswarm", **kwargs):
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        if not hasattr(self, "shap_explanation_"):
            self.explain_shap()
        fig = self.shap_figure(class_index=class_index, max_display=max_display, plot_type=plot_type, **kwargs)
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with mpl.rc_context({"svg.fonttype": "none"}):
                fig.savefig(target, bbox_inches="tight")
        finally:
            plt.close(fig)
        return target

    def _require_fitted(self):
        if not hasattr(self, "model_"):
            raise RuntimeError("call evaluate before using fitted-model outputs")
