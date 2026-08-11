"""Access, persistence, annotation, and AnnData interoperability methods."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from ..util.annotation import SDFMzSearcher

logger = logging.getLogger(__name__)


class DatasetInteropMixin:
    """Provide accessors and external representations for a dataset."""

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, key):
        if self.data.shape[1] == 0:
            raise KeyError("Dataset has no features")
        target = float(key)
        index = np.abs(self.data.columns.values.astype(float) - target).argmin()
        return self.data.iloc[:, index].values

    def get_name(self) -> str:
        """Return the human-readable dataset name."""
        return str(self.file_meta.get("name", "unnamed"))

    def get_labels(self, mapping: dict | None = None) -> np.ndarray:
        """Return per-cell labels, optionally replacing them with ``mapping``."""
        if "label" not in self.peak_meta:
            raise KeyError("peak_meta does not contain a 'label' column")
        labels = self.peak_meta["label"]
        if mapping is not None:
            labels = labels.map(lambda value: mapping.get(value, value))
        return labels.to_numpy()

    def save(self, root_path: str | Path, *, overwrite: bool = False) -> Path:
        """Save the dataset below ``root_path`` and return its directory."""
        result_path = _create_result_directory(root_path, self.get_name(), overwrite)
        logger.info("Saving processed data to %s", result_path)
        with (result_path / ".meta").open("w", encoding="utf-8") as handle:
            json.dump(self.file_meta, handle, ensure_ascii=False, indent=2)
        for name, frame in _dataset_frames(self):
            frame.to_pickle(result_path / f"{name}.pkl")
            frame.to_csv(result_path / f"{name}.csv")
        return result_path

    def to_anndata(self) -> AnnData:
        """Return an AnnData copy with stable cell and feature identifiers."""
        _validate_metadata_dimensions(self)
        observations = self.peak_meta.copy()
        observations.insert(0, "source_index", self.peak_meta.index.astype(str))
        observations.index = pd.Index(
            [f"cell_{index}" for index in range(len(observations))],
            name="cell_id",
        )
        variables = self.feature_meta.reindex(self.data.columns).copy()
        variables.index = pd.Index(self.data.columns.astype(str), name="feature_id")
        adata = AnnData(X=self.data.values, obs=observations, var=variables)
        adata.raw = adata.copy()
        return adata

    def get_annotation(
        self,
        sdf_path: str | Path,
        ppm_tol: float,
        search_mode: Literal["pos", "neg", "both"] = "pos",
        adducts_pos: dict | None = None,
        adducts_neg: dict | None = None,
        **kwargs,
    ):
        """Search an SDF database for the dataset's feature masses."""
        searcher = SDFMzSearcher(
            sdf_path=sdf_path,
            adducts_pos=adducts_pos,
            adducts_neg=adducts_neg,
        )
        return searcher.search(
            mz=self.data.columns.astype(float),
            ppm_tol=ppm_tol,
            mode=search_mode,
            **kwargs,
        )


def _create_result_directory(root_path, dataset_name: str, overwrite: bool) -> Path:
    root = Path(root_path).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    safe_name = Path(dataset_name).name
    if safe_name in {"", ".", ".."}:
        raise ValueError(f"Invalid dataset name: {safe_name!r}")
    result_path = root / safe_name
    result_path.mkdir(exist_ok=overwrite)
    return result_path


def _dataset_frames(dataset):
    return (
        ("data", dataset.data),
        ("peak_meta", dataset.peak_meta),
        ("feature_meta", dataset.feature_meta),
    )


def _validate_metadata_dimensions(dataset) -> None:
    if len(dataset.peak_meta) != len(dataset.data):
        raise ValueError("peak_meta row count must match data row count")
    if len(dataset.feature_meta) != dataset.data.shape[1]:
        raise ValueError("feature_meta row count must match data column count")
