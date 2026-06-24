from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_raw_dir: Path = Field(default=Path("data/raw"))
    data_processed_dir: Path = Field(default=Path("data/processed"))
    data_features_dir: Path = Field(default=Path("data/features"))
    model_artifacts_dir: Path = Field(default=Path("models/artifacts"))

    # ── Model hyperparameter defaults ─────────────────────────────────────────
    random_seed: int = Field(default=42)
    gmm_max_components: int = Field(default=8)
    lgbm_n_estimators: int = Field(default=300)
    lgbm_learning_rate: float = Field(default=0.05)
    lgbm_num_leaves: int = Field(default=31)

    # ── Financial constants ────────────────────────────────────────────────────
    annual_mgmt_fee: float = Field(default=0.0025)   # 25 bps
    monthly_discount_rate: float = Field(default=0.005)  # 0.5% / month
    risk_free_annual_rate: float = Field(default=0.045)  # 3-month T-bill proxy

    # ── Simulation parameters ─────────────────────────────────────────────────
    n_users: int = Field(default=5000)
    n_days: int = Field(default=730)


settings = Settings()
