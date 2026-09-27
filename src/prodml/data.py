"""Loading the Parquet file and splitting it into train and validation sets."""

import urllib.request
from pathlib import Path

import pandas as pd

from prodml.config import Settings

PICKUP_COL = "lpep_pickup_datetime"
DROPOFF_COL = "lpep_dropoff_datetime"


def download_month(settings: Settings) -> Path:
    """Download the month's Parquet file if it is not already on disk."""
    path = settings.data_file
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {settings.data_url}")
        urllib.request.urlretrieve(settings.data_url, path)
    return path


def load_trips(path: Path, min_duration: float, max_duration: float) -> pd.DataFrame:
    """Read trips, compute duration in minutes, and drop implausible trips."""
    df = pd.read_parquet(path)
    df["duration"] = (df[DROPOFF_COL] - df[PICKUP_COL]).dt.total_seconds() / 60
    mask = (df["duration"] >= min_duration) & (df["duration"] <= max_duration)
    return df.loc[mask].copy()


def split_by_time(
    df: pd.DataFrame, val_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Use the last `val_days` days of the month as validation."""
    cutoff = df[PICKUP_COL].max().normalize() - pd.Timedelta(days=val_days - 1)
    train = df[df[PICKUP_COL] < cutoff]
    val = df[df[PICKUP_COL] >= cutoff]
    return train, val, cutoff
