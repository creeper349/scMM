"""Persistent quality summaries for processed scMM datasets."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class QualitySummary:
    """Compact metrics shown before downstream interpretation."""

    cell_count: int
    feature_count: int
    zero_fraction: float
    median_total_intensity: float
    median_detected_features: float
    embedding_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityReport:
    """Summary and tabular quality artifacts derived from one dataset."""

    summary: QualitySummary
    cells: pd.DataFrame
    features: pd.DataFrame
    embedding: pd.DataFrame


def build_quality_report(
    dataset,
    *,
    include_umap: bool = True,
    max_embedding_cells: int = 10_000,
    max_embedding_features: int = 2_000,
) -> QualityReport:
    """Calculate deterministic, read-only quality metrics from a dataset."""
    values = np.asarray(dataset.data, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    cell_count, feature_count = values.shape
    totals = values.sum(axis=1)
    detected = np.count_nonzero(values, axis=1)
    zero_fraction = float(np.mean(values == 0)) if values.size else 0.0

    cells = pd.DataFrame(
        {
            "cell_index": np.arange(cell_count),
            "total_intensity": totals,
            "detected_features": detected,
        }
    )
    for column in ("rt", "time", "label"):
        if column in dataset.peak_meta and len(dataset.peak_meta) == cell_count:
            cells[column] = dataset.peak_meta[column].to_numpy()

    mz = np.asarray(dataset.data.columns, dtype=np.float64)
    features = pd.DataFrame(
        {
            "mz": mz,
            "detection_rate": (
                np.mean(values != 0, axis=0) if cell_count else np.zeros(feature_count)
            ),
            "mean_intensity": (np.mean(values, axis=0) if cell_count else np.zeros(feature_count)),
            "median_nonzero_intensity": _median_nonzero(values),
        }
    )
    embedding, warnings = _embedding_frame(
        values,
        include_umap=include_umap,
        max_cells=max_embedding_cells,
        max_features=max_embedding_features,
    )
    summary = QualitySummary(
        cell_count=cell_count,
        feature_count=feature_count,
        zero_fraction=zero_fraction,
        median_total_intensity=float(np.median(totals)) if cell_count else 0.0,
        median_detected_features=float(np.median(detected)) if cell_count else 0.0,
        embedding_warnings=warnings,
    )
    return QualityReport(summary, cells, features, embedding)


def save_quality_report(report: QualityReport, result_path: str | Path) -> None:
    """Write stable quality artifacts beside a processed dataset."""
    root = Path(result_path)
    with (root / "quality-summary.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(report.summary), handle, ensure_ascii=False, indent=2)
    report.cells.to_csv(root / "cell-quality.csv", index=False)
    report.features.to_csv(root / "feature-quality.csv", index=False)
    report.embedding.to_csv(root / "embedding.csv", index=False)


def load_quality_report(result_path: str | Path) -> QualityReport:
    """Load quality artifacts without recomputing an embedding in Panel."""
    root = Path(result_path)
    with (root / "quality-summary.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["embedding_warnings"] = tuple(payload.get("embedding_warnings", ()))
    return QualityReport(
        QualitySummary(**payload),
        pd.read_csv(root / "cell-quality.csv"),
        pd.read_csv(root / "feature-quality.csv"),
        pd.read_csv(root / "embedding.csv"),
    )


def _median_nonzero(values: np.ndarray) -> np.ndarray:
    medians = []
    for column in values.T:
        nonzero = column[column != 0]
        medians.append(float(np.median(nonzero)) if nonzero.size else 0.0)
    return np.asarray(medians, dtype=np.float64)


def _embedding_frame(
    values: np.ndarray,
    *,
    include_umap: bool,
    max_cells: int,
    max_features: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    cell_count, feature_count = values.shape
    warnings: list[str] = []
    if max_cells < 2 or max_features < 1:
        raise ValueError("Embedding limits must allow at least two cells and one feature")
    cell_indices = np.arange(cell_count)
    if cell_count > max_cells:
        cell_indices = np.linspace(0, cell_count - 1, max_cells, dtype=int)
        warnings.append(f"Embedding uses a deterministic sample of {max_cells} cells")
    embedding = pd.DataFrame({"cell_index": cell_indices})
    if cell_count < 2 or feature_count < 1:
        warnings.append("PCA requires at least two cells and one feature")
        return embedding, tuple(warnings)

    selected = values[cell_indices]
    if feature_count > max_features:
        variances = np.var(np.log1p(np.clip(selected, 0, None)), axis=0)
        feature_indices = np.argpartition(variances, -max_features)[-max_features:]
        selected = selected[:, np.sort(feature_indices)]
        feature_count = max_features
        warnings.append(f"Embedding uses the {max_features} most variable features")
    transformed = np.log1p(np.clip(selected, 0, None))
    transformed = StandardScaler().fit_transform(transformed)
    components = min(2, cell_count, feature_count)
    if np.allclose(np.var(transformed, axis=0), 0):
        coordinates = np.zeros((len(cell_indices), components))
        warnings.append("PCA coordinates are zero because all features are constant")
    else:
        coordinates = PCA(n_components=components, random_state=42).fit_transform(transformed)
    embedding["PCA1"] = coordinates[:, 0]
    if components == 2:
        embedding["PCA2"] = coordinates[:, 1]
    else:
        warnings.append("PCA2 is unavailable because the dataset has fewer than two features")

    if include_umap and len(cell_indices) >= 3 and feature_count >= 2:
        try:
            umap_coordinates = _umap_embedding(transformed)
        except (ImportError, RuntimeError, ValueError) as exc:
            warnings.append(f"UMAP unavailable: {exc}")
        else:
            embedding[["UMAP1", "UMAP2"]] = umap_coordinates
    elif include_umap:
        warnings.append("UMAP requires at least three cells and two features")
    return embedding, tuple(warnings)


def _umap_embedding(values: np.ndarray) -> np.ndarray:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tensorflow not installed; ParametricUMAP will be unavailable",
                category=ImportWarning,
                module="umap",
            )
            from umap import UMAP
    except ImportError as exc:
        raise ImportError("install scMM[plot] to enable UMAP") from exc
    neighbors = max(2, min(15, len(values) - 1))
    return UMAP(
        n_components=2,
        n_neighbors=neighbors,
        min_dist=0.7,
        random_state=42,
        init="random",
        n_jobs=1,
    ).fit_transform(values)


__all__ = [
    "QualityReport",
    "QualitySummary",
    "build_quality_report",
    "load_quality_report",
    "save_quality_report",
]
