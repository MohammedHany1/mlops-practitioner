"""Fit the baseline model, evaluate it, and persist it."""

import logging
import pickle

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline

from prodml.config import Settings, get_settings
from prodml.data import download_month, load_trips, split_by_time
from prodml.features import TARGET, to_dicts

REPORT_START = "<!-- baseline-metrics:start -->"
REPORT_END = "<!-- baseline-metrics:end -->"


def build_pipeline(settings: Settings) -> Pipeline:
    return make_pipeline(
        DictVectorizer(),
        LinearRegression(fit_intercept=settings.fit_intercept),
    )


def save_model(model: Pipeline, settings: Settings) -> None:
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.model_path, "wb") as f:
        pickle.dump(model, f)


def write_report(metrics: dict[str, float], summary: str, settings: Settings) -> None:
    """Put the metrics block at the top of the report, keeping everything else."""
    path = settings.report_path
    path.parent.mkdir(parents=True, exist_ok=True)
    block = (
        f"{REPORT_START}\n"
        "## Baseline metrics\n\n"
        "| Metric | Value (minutes) |\n"
        "|---|---|\n"
        f"| Validation RMSE | {metrics['rmse']:.3f} |\n"
        f"| Validation MAE | {metrics['mae']:.3f} |\n\n"
        f"{summary}\n"
        f"{REPORT_END}"
    )
    existing = path.read_text() if path.exists() else "# Module 1 report\n"
    if REPORT_START in existing:
        existing = existing.split(REPORT_END, 1)[1].lstrip("\n")
    path.write_text(block + "\n\n" + existing)


def run(settings: Settings) -> dict[str, float]:
    path = download_month(settings)
    df = load_trips(path, settings.min_duration_min, settings.max_duration_min)
    train_df, val_df, cutoff = split_by_time(df, settings.val_days)

    y_train = train_df[TARGET].to_numpy()
    y_val = val_df[TARGET].to_numpy()

    model = build_pipeline(settings)
    model.fit(to_dicts(train_df), y_train)
    y_pred = model.predict(to_dicts(val_df))

    metrics = {
        "rmse": float(root_mean_squared_error(y_val, y_pred)),
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "naive_rmse": float(
            root_mean_squared_error(y_val, np.full_like(y_val, y_train.mean()))
        ),
    }

    save_model(model, settings)
    summary = (
        f"Data: green taxi {settings.month}, time split at {cutoff.date()} "
        f"(train {len(train_df):,} / val {len(val_df):,} trips).\n"
        "Model: DictVectorizer + LinearRegression on `PU_DO`, `trip_distance`. "
        f"Naive mean RMSE: {metrics['naive_rmse']:.3f}."
    )
    write_report(metrics, summary, settings)
    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    settings = get_settings()
    metrics = run(settings)
    print(f"Validation RMSE: {metrics['rmse']:.3f} min")
    print(f"Validation MAE:  {metrics['mae']:.3f} min")
    print(f"Naive mean RMSE: {metrics['naive_rmse']:.3f} min")
    print(f"Model saved to {settings.model_path}")
    print(f"Metrics written to {settings.report_path}")


if __name__ == "__main__":
    main()
