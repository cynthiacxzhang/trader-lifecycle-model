"""Tests for feature engineering pipeline (Step 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.portfolio_snapshots import generate as gen_portfolio
from src.features.portfolio_features import compute as compute_portfolio
from src.features.behavioral_features import compute as compute_behavioral
from src.features.risk_features import compute as compute_risk
from src.features.pipeline import run as run_pipeline

N_USERS = 80
N_DAYS = 120  # 4 months — enough for rolling windows to fill


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def daily():
    portfolio, _ = gen_portfolio(seed=7, n_users=N_USERS, n_days=N_DAYS)
    return portfolio


@pytest.fixture(scope="module")
def market():
    path = __import__("src.config", fromlist=["settings"]).settings.data_raw_dir / "market_data.parquet"
    if not path.exists():
        pytest.skip("market_data.parquet not present")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def feat_portfolio(daily):
    return compute_portfolio(daily)


@pytest.fixture(scope="module")
def feat_behavioral(daily):
    return compute_behavioral(daily)


@pytest.fixture(scope="module")
def feat_risk(daily, market):
    return compute_risk(daily, market)


# ── Portfolio features ────────────────────────────────────────────────────────

def test_portfolio_shape(feat_portfolio):
    n_months = 4  # 120 days ÷ 30
    assert len(feat_portfolio) == N_USERS * n_months


def test_portfolio_columns(feat_portfolio):
    expected = {"user_id", "month", "hhi", "enp", "turnover_rate", "cash_drag", "aum_growth_mom", "avg_aum"}
    assert expected.issubset(feat_portfolio.columns)


def test_hhi_in_range(feat_portfolio):
    assert feat_portfolio["hhi"].between(0, 1).all()


def test_enp_ge_one(feat_portfolio):
    assert feat_portfolio["enp"].ge(1).all()


def test_enp_inverse_of_hhi(feat_portfolio):
    computed = 1.0 / feat_portfolio["hhi"].clip(lower=1e-6)
    assert np.allclose(feat_portfolio["enp"], computed.clip(upper=100), atol=1e-4)


def test_turnover_rate_non_negative(feat_portfolio):
    assert feat_portfolio["turnover_rate"].ge(0).all()


def test_cash_drag_in_range(feat_portfolio):
    assert feat_portfolio["cash_drag"].between(0, 1).all()


def test_avg_aum_positive(feat_portfolio):
    assert feat_portfolio["avg_aum"].gt(0).all()


def test_no_nulls_in_portfolio_features(feat_portfolio):
    assert not feat_portfolio.isnull().any().any()


# ── Behavioral features ───────────────────────────────────────────────────────

def test_behavioral_columns(feat_behavioral):
    expected = {"user_id", "month", "trade_frequency", "drs", "deposit_regularity", "weekend_trade_ratio"}
    assert expected.issubset(feat_behavioral.columns)


def test_drs_in_range(feat_behavioral):
    assert feat_behavioral["drs"].dropna().between(-1, 1).all()


def test_trade_frequency_non_negative(feat_behavioral):
    assert feat_behavioral["trade_frequency"].ge(0).all()


def test_weekend_trade_ratio_in_range(feat_behavioral):
    assert feat_behavioral["weekend_trade_ratio"].between(0, 1).all()


def test_deposit_regularity_non_negative(feat_behavioral):
    assert feat_behavioral["deposit_regularity"].dropna().ge(0).all()


# ── Risk features ─────────────────────────────────────────────────────────────

def test_risk_columns(feat_risk):
    expected = {"user_id", "month", "sharpe_ratio", "sortino_ratio", "max_drawdown", "portfolio_beta"}
    assert expected.issubset(feat_risk.columns)


def test_max_drawdown_in_range(feat_risk):
    assert feat_risk["max_drawdown"].between(0, 1).all(skipna=False)


def test_beta_clipped(feat_risk):
    assert feat_risk["portfolio_beta"].dropna().between(-5, 5).all()


def test_sharpe_nan_for_first_month(feat_risk):
    first_month = feat_risk["month"].min()
    sharpe_first = feat_risk[feat_risk["month"] == first_month]["sharpe_ratio"]
    assert sharpe_first.isna().all(), "Sharpe should be NaN in first month (insufficient history)"


def test_sharpe_fills_after_window(feat_risk):
    last_month = feat_risk["month"].max()
    sharpe_last = feat_risk[feat_risk["month"] == last_month]["sharpe_ratio"]
    assert sharpe_last.notna().mean() > 0.5, "Most users should have Sharpe by final month"


# ── Full feature matrix (saved parquet) ──────────────────────────────────────

@pytest.fixture(scope="module")
def feature_matrix():
    path = __import__("src.config", fromlist=["settings"]).settings.data_features_dir / "feature_matrix.parquet"
    if not path.exists():
        pytest.skip("feature_matrix.parquet not present — run src.features.pipeline first")
    return pd.read_parquet(path)


def test_feature_matrix_shape(feature_matrix):
    assert feature_matrix["user_id"].nunique() == 5000
    assert feature_matrix["month"].nunique() == 24


def test_feature_matrix_all_columns(feature_matrix):
    required = {
        "user_id", "month",
        "hhi", "enp", "turnover_rate", "cash_drag", "aum_growth_mom", "avg_aum",
        "trade_frequency", "drs", "deposit_regularity", "weekend_trade_ratio",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "portfolio_beta",
        "archetype", "initial_archetype",
    }
    assert required.issubset(feature_matrix.columns)


def test_feature_matrix_no_nulls_non_rolling(feature_matrix):
    rolling_cols = ["sharpe_ratio", "sortino_ratio", "portfolio_beta", "drs", "deposit_regularity"]
    non_rolling = feature_matrix.drop(columns=rolling_cols)
    assert not non_rolling.isnull().any().any()


def test_feature_matrix_archetype_labels_valid(feature_matrix):
    valid = {"casual", "active", "high_value", "churned"}
    assert set(feature_matrix["archetype"].unique()).issubset(valid)


def test_feature_matrix_hhi_range(feature_matrix):
    assert feature_matrix["hhi"].between(0, 1).all()


def test_feature_matrix_mdd_range(feature_matrix):
    assert feature_matrix["max_drawdown"].between(0, 1).all()
