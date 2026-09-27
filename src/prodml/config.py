"""All paths, hyperparameters and ports live here.

Every value can be overridden with an environment variable prefixed PRODML_,
for example: PRODML_MONTH=2024-02 prodml-train
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PRODML_", env_file=".env", extra="ignore"
    )

    # Paths (relative to the directory you run commands from, normally the repo root)
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    model_filename: str = "baseline.pkl"
    report_filename: str = "module-1.md"

    # Data
    month: str = "2024-01"
    data_url_template: str = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{month}.parquet"
    )

    # Cleaning and split
    min_duration_min: float = 1.0
    max_duration_min: float = 60.0
    val_days: int = 7

    # Model hyperparameters (LinearRegression)
    fit_intercept: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def data_file(self) -> Path:
        return self.data_dir / f"green_tripdata_{self.month}.parquet"

    @property
    def data_url(self) -> str:
        return self.data_url_template.format(month=self.month)

    @property
    def model_path(self) -> Path:
        return self.models_dir / self.model_filename

    @property
    def report_path(self) -> Path:
        return self.reports_dir / self.report_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
