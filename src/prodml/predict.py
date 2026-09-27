"""Load the saved model and predict trip durations.

DurationPredictor is the single interface the rest of the project talks to.
The API, a BentoML runner, or an ONNX model can all sit behind the same methods.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from prodml.config import get_settings
from prodml.decorators import timed
from prodml.features import to_dicts

REQUIRED_FIELDS = ("PULocationID", "DOLocationID", "trip_distance")


class DurationPredictor:
    """Predict NYC green taxi trip duration in minutes."""

    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or get_settings().model_path
        self._model: Pipeline | None = None

    def load(self) -> DurationPredictor:
        """Load the model from disk. Returns self so calls can be chained."""
        with open(self.model_path, "rb") as f:
            self._model = pickle.load(f)
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        """Predict the duration of one trip."""
        return self.predict_batch([features])[0]

    def predict_batch(self, trips: list[dict[str, Any]] | pd.DataFrame) -> list[float]:
        """Predict durations for many trips.

        Each trip needs PULocationID, DOLocationID and trip_distance.
        """
        model = self._require_model()
        df = trips if isinstance(trips, pd.DataFrame) else pd.DataFrame(trips)
        missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        return [float(x) for x in model.predict(to_dicts(df))]

    def _require_model(self) -> Pipeline:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")
        return self._model
