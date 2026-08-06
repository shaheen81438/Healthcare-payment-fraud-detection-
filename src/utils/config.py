"""Application settings loaded from environment variables / .env.

Single source of configuration for pipeline scripts, the API, and the dashboard.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings with local-development defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Allow the `model_artifacts_dir` field name (pydantic reserves the
        # `model_` prefix by default).
        protected_namespaces=(),
    )

    # Application
    env: str = "dev"
    log_level: str = "INFO"
    random_seed: int = 42

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "pi_user"
    postgres_password: str = "pi_password_change_me"
    postgres_db: str = "payment_integrity"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment: str = "claims-risk"

    # API
    api_port: int = 8000
    batch_predict_limit: int = 1000

    # Artifacts / data
    model_artifacts_dir: str = "models"
    cms_data_year: int = 2023

    @property
    def database_url(self) -> str:
        """SQLAlchemy connection URL for the application database."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    """Return process-wide cached settings."""
    return Settings()
