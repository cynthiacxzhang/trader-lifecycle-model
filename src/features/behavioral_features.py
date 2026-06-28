"""
Behavioral signal features — monthly aggregation.

DRS (drawdown reaction score): within-month correlation of daily portfolio
return with next-day trade count. Positive = contrarian (trades more after
bad days), negative = panic seller. Computed via the E[XY]-E[X]E[Y] moment
trick so the full groupby is vectorized rather than a per-user apply loop.

Deposit regularity uses the full 2-year history per user since monthly
windows rarely contain enough deposit events to estimate std reliably.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly behavioral features per user.

    Input:  daily portfolio snapshots
    Output: (user_id, month) frame with trade_frequency, drs,
            deposit_regularity, weekend_trade_ratio
    """
    df = daily.sort_values(["user_id", "date"]).copy()

    # Per-user daily return and next-day trade count (shift within user group)
    df["daily_return"] = df.groupby("user_id")["aum"].pct_change()
    df["next_trades"] = df.groupby("user_id")["num_trades_today"].shift(-1)

    # Weekend flag
    df["is_weekend"] = df["date"].dt.dayofweek >= 5
    df["weekend_trades"] = df["num_trades_today"] * df["is_weekend"].astype(int)

    # Cross-product needed for vectorized correlation
    df["return_x_next_trades"] = df["daily_return"] * df["next_trades"]
    df["year_month"] = df["date"].dt.to_period("M")

    monthly = (
        df.groupby(["user_id", "year_month"])
        .agg(
            trade_frequency=("num_trades_today", "mean"),
            total_trades=("num_trades_today", "sum"),
            total_weekend_trades=("weekend_trades", "sum"),
            mean_return=("daily_return", "mean"),
            mean_next_trades=("next_trades", "mean"),
            std_return=("daily_return", "std"),
            std_next_trades=("next_trades", "std"),
            mean_product=("return_x_next_trades", "mean"),
        )
        .reset_index()
    )

    # DRS = corr(return_t, next_trades_t+1)
    # corr = (E[XY] - E[X]E[Y]) / (σ_X * σ_Y)
    cov = monthly["mean_product"] - monthly["mean_return"] * monthly["mean_next_trades"]
    denom = (monthly["std_return"] * monthly["std_next_trades"]).clip(lower=1e-10)
    monthly["drs"] = (cov / denom).clip(-1.0, 1.0)

    monthly["weekend_trade_ratio"] = (
        monthly["total_weekend_trades"] / monthly["total_trades"].clip(lower=1)
    ).clip(0.0, 1.0)

    # Deposit regularity: std of inter-deposit intervals over full user history.
    # Deposit-like event: daily AUM change > 2% (too large to be returns alone).
    df["is_deposit"] = df["daily_return"] > 0.02

    def _interval_std(s: pd.Series) -> float:
        idx = np.where(s.values)[0]
        if len(idx) < 3:
            return np.nan
        return float(np.std(np.diff(idx)))

    deposit_reg = (
        df.groupby("user_id")["is_deposit"]
        .apply(_interval_std)
        .rename("deposit_regularity")
        .reset_index()
    )

    monthly = monthly.merge(deposit_reg, on="user_id", how="left")
    monthly["month"] = monthly["year_month"].dt.to_timestamp()

    cols = ["user_id", "month", "trade_frequency", "drs", "deposit_regularity", "weekend_trade_ratio"]
    return monthly[cols].copy()


if __name__ == "__main__":
    from src.ingestion.schema import load_portfolio_snapshots

    daily = load_portfolio_snapshots()
    sample = daily[daily["user_id"].isin(daily["user_id"].unique()[:50])]
    out = compute(sample)
    logger.info(f"Behavioral features shape: {out.shape}")
    logger.info(f"\n{out.describe()}")
    assert out["drs"].between(-1, 1).all(skipna=True)
    assert out["weekend_trade_ratio"].between(0, 1).all()
    logger.info("Smoke test passed.")
