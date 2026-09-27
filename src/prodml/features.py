"""Feature engineering shared by training and prediction."""

from typing import Any, cast

import pandas as pd

CATEGORICAL = ["PU_DO"]
NUMERICAL = ["trip_distance"]
TARGET = "duration"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add PU_DO (pickup_dropoff zone pair) and make trip_distance a float."""
    df = df.copy()
    df["PU_DO"] = df["PULocationID"].astype(str) + "_" + df["DOLocationID"].astype(str)
    df["trip_distance"] = df["trip_distance"].astype(float)
    return df


def to_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn a frame into the list of dicts that DictVectorizer expects."""
    records = add_features(df)[CATEGORICAL + NUMERICAL].to_dict(orient="records")
    return cast(list[dict[str, Any]], records)
