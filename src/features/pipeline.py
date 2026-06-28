"""
Feature engineering pipeline.

Loads validated raw parquets, computes all feature groups, joins on
(user_id, month), appends archetype ground-truth labels, and writes
data/features/feature_matrix.parquet.

Each row = one user × one calendar month.

Why month as the time unit: coarser than daily (removes noise) but fine
enough to capture archetype transitions, which occur on monthly Markov steps.
"""

from __future__ import annotations

import pandas as pd
from loguru import logger

from src.config import settings
from src.ingestion.schema import load_market_data, load_portfolio_snapshots, load_user_states
from src.features.portfolio_features import compute as compute_portfolio
from src.features.behavioral_features import compute as compute_behavioral
from src.features.risk_features import compute as compute_risk


def run() -> pd.DataFrame:
    """Build and save the feature matrix. Returns the completed DataFrame."""
    logger.info("Loading raw data ...")
    portfolio = load_portfolio_snapshots()
    market = load_market_data()
    states = load_user_states()

    logger.info("Computing portfolio features ...")
    feat_portfolio = compute_portfolio(portfolio)

    logger.info("Computing behavioral features ...")
    feat_behavioral = compute_behavioral(portfolio)

    logger.info("Computing risk features ...")
    feat_risk = compute_risk(portfolio, market)

    logger.info("Joining feature groups ...")
    feature_matrix = (
        feat_portfolio
        .merge(feat_behavioral, on=["user_id", "month"], how="outer")
        .merge(feat_risk, on=["user_id", "month"], how="outer")
    )

    # Attach ground-truth archetype labels for downstream evaluation
    states["month"] = states["month"].dt.to_period("M").dt.to_timestamp()
    feature_matrix = feature_matrix.merge(
        states[["user_id", "month", "archetype", "initial_archetype"]],
        on=["user_id", "month"],
        how="left",
    )

    feature_matrix = feature_matrix.sort_values(["user_id", "month"]).reset_index(drop=True)

    n_rows = len(feature_matrix)
    n_users = feature_matrix["user_id"].nunique()
    n_months = feature_matrix["month"].nunique()

    # Null rate excluding rolling metrics that are NaN by design for early months
    rolling_cols = ["sharpe_ratio", "sortino_ratio", "portfolio_beta"]
    non_rolling = feature_matrix.drop(columns=rolling_cols)
    null_pct = non_rolling.isnull().mean().mean() * 100

    logger.info(f"Feature matrix: {n_rows:,} rows ({n_users:,} users × {n_months} months)")
    logger.info(f"Non-rolling null rate: {null_pct:.1f}%")

    settings.mkdirs()
    out = settings.data_features_dir / "feature_matrix.parquet"
    feature_matrix.to_parquet(out, index=False)
    logger.info(f"Saved → {out}")

    return feature_matrix


if __name__ == "__main__":
    fm = run()

    # Smoke tests
    assert len(fm) > 0, "empty feature matrix"
    required_cols = {
        "user_id", "month",
        "hhi", "enp", "turnover_rate", "cash_drag", "aum_growth_mom", "avg_aum",
        "trade_frequency", "drs", "deposit_regularity", "weekend_trade_ratio",
        "sharpe_ratio", "sortino_ratio", "max_drawdown", "portfolio_beta",
        "archetype", "initial_archetype",
    }
    missing = required_cols - set(fm.columns)
    assert not missing, f"Missing columns: {missing}"

    assert fm["hhi"].between(0, 1).all(), "HHI out of [0,1]"
    assert fm["max_drawdown"].ge(0).all(), "negative max drawdown"
    assert fm["weekend_trade_ratio"].between(0, 1).all(), "weekend ratio out of [0,1]"
    assert fm["drs"].dropna().between(-1, 1).all(), "DRS out of [-1,1]"

    logger.info(f"\nNull counts:\n{fm.isnull().sum()}")
    logger.info(f"\nSample:\n{fm.head(3).T}")
    logger.info("Smoke test passed.")
