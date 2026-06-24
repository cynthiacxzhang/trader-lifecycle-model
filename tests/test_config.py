import os
from pathlib import Path

import pytest

from src.config import Settings, _PROJECT_ROOT, settings


# ── Default values ────────────────────────────────────────────────────────────

def test_financial_constants():
    assert settings.annual_mgmt_fee == 0.0025
    assert settings.monthly_discount_rate == 0.005
    assert settings.risk_free_annual_rate == 0.045


def test_simulation_params():
    assert settings.n_users == 5000
    assert settings.n_days == 730
    assert settings.random_seed == 42


def test_model_defaults():
    assert settings.gmm_max_components == 8
    assert settings.lgbm_n_estimators == 300
    assert 0 < settings.lgbm_learning_rate < 1
    assert settings.lgbm_num_leaves > 0


# ── Path resolution ───────────────────────────────────────────────────────────

def test_paths_are_absolute():
    assert settings.data_raw_dir.is_absolute()
    assert settings.data_processed_dir.is_absolute()
    assert settings.data_features_dir.is_absolute()
    assert settings.model_artifacts_dir.is_absolute()


def test_paths_anchored_to_project_root():
    assert settings.data_raw_dir == _PROJECT_ROOT / "data" / "raw"
    assert settings.data_processed_dir == _PROJECT_ROOT / "data" / "processed"
    assert settings.data_features_dir == _PROJECT_ROOT / "data" / "features"
    assert settings.model_artifacts_dir == _PROJECT_ROOT / "models" / "artifacts"


def test_mkdirs_creates_directories(tmp_path):
    s = Settings(
        data_raw_dir=tmp_path / "raw",
        data_processed_dir=tmp_path / "processed",
        data_features_dir=tmp_path / "features",
        model_artifacts_dir=tmp_path / "artifacts",
    )
    s.mkdirs()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "processed").is_dir()
    assert (tmp_path / "features").is_dir()
    assert (tmp_path / "artifacts").is_dir()


def test_relative_path_env_override_resolved_to_absolute(monkeypatch):
    """A relative path supplied via env var must be resolved to absolute."""
    monkeypatch.setenv("DATA_RAW_DIR", "some/relative/path")
    s = Settings()
    assert s.data_raw_dir.is_absolute()
    assert s.data_raw_dir == _PROJECT_ROOT / "some" / "relative" / "path"


# ── Env var overrides ─────────────────────────────────────────────────────────

def test_env_override_int(monkeypatch):
    monkeypatch.setenv("N_USERS", "999")
    s = Settings()
    assert s.n_users == 999


def test_env_override_float(monkeypatch):
    monkeypatch.setenv("ANNUAL_MGMT_FEE", "0.005")
    s = Settings()
    assert s.annual_mgmt_fee == 0.005


# ── Imports ───────────────────────────────────────────────────────────────────

def test_all_core_imports():
    """Fail fast if any required package is missing or broken."""
    import fastapi
    import httpx
    import lifelines
    import lightgbm
    import loguru
    import mlflow
    import numba
    import numpy
    import pandas
    import pandera
    import pyarrow
    import pydantic
    import pydantic_settings
    import pytest as _pytest
    import sklearn  # scikit-learn imports as sklearn
    import scipy
    import shap
    import uvicorn
    import yfinance
