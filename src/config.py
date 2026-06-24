from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute project root, regardless of working directory when scripts are invoked
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── Paths ─────────────────────────────────────────────────────────────────
    data_raw_dir: Path = Field(default=_PROJECT_ROOT / "data" / "raw")
    data_processed_dir: Path = Field(default=_PROJECT_ROOT / "data" / "processed")
    data_features_dir: Path = Field(default=_PROJECT_ROOT / "data" / "features")
    model_artifacts_dir: Path = Field(default=_PROJECT_ROOT / "models" / "artifacts")

    # ── Model hyperparameter defaults ─────────────────────────────────────────
    random_seed: int = Field(default=42)
    gmm_max_components: int = Field(default=8)
    lgbm_n_estimators: int = Field(default=300)
    lgbm_learning_rate: float = Field(default=0.05)
    lgbm_num_leaves: int = Field(default=31)

    # ── Financial constants ────────────────────────────────────────────────────
    annual_mgmt_fee: float = Field(default=0.0025)       # 25 bps
    monthly_discount_rate: float = Field(default=0.005)  # 0.5% / month
    risk_free_annual_rate: float = Field(default=0.045)  # 3-month T-bill proxy

    # ── Simulation parameters ─────────────────────────────────────────────────
    n_users: int = Field(default=5000)
    n_days: int = Field(default=730)

    @field_validator("data_raw_dir", "data_processed_dir", "data_features_dir", "model_artifacts_dir", mode="after")
    @classmethod
    def resolve_path(cls, v: Path) -> Path:
        """Make env-supplied relative paths absolute against project root."""
        return v if v.is_absolute() else _PROJECT_ROOT / v

    def mkdirs(self) -> None:
        """Create all configured directories if they don't exist."""
        for p in (self.data_raw_dir, self.data_processed_dir, self.data_features_dir, self.model_artifacts_dir):
            p.mkdir(parents=True, exist_ok=True)


settings = Settings()
