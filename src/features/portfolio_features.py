"""
Portfolio concentration and size features — monthly aggregation.

Why these features: HHI/ENP are standard industrial-concentration metrics
adapted for portfolio risk. Turnover and cash drag differentiate archetypes
before any model sees the data. AUM growth decomposes to deposit vs return
growth downstream if needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly portfolio features per user.

    Input:  daily portfolio snapshots (user_id, date, aum, num_positions,
            top_holding_pct, cash_pct, num_trades_today, ...)
    Output: (user_id, month) frame with hhi, enp, turnover_rate,
            cash_drag, aum_growth_mom, avg_aum
    """
    df = daily.copy()

    # Approximate HHI from top_holding_pct and num_positions.
    # Remaining (n-1) positions assumed equal-weight — minimum-information estimate.
    # HHI = w_top² + (n-1) * ((1-w_top)/(n-1))²  =  w_top² + (1-w_top)²/(n-1)
    n = df["num_positions"].clip(lower=1).astype(float)
    top = df["top_holding_pct"].clip(0.0, 1.0)
    rest_weight = (1.0 - top) / (n - 1.0).clip(lower=1.0)
    df["hhi"] = np.where(n <= 1, 1.0, top**2 + (n - 1) * rest_weight**2).clip(0.0, 1.0)

    df["year_month"] = df["date"].dt.to_period("M")

    agg = (
        df.groupby(["user_id", "year_month"])
        .agg(
            hhi=("hhi", "mean"),
            cash_drag=("cash_pct", "mean"),
            total_trades=("num_trades_today", "sum"),
            avg_aum=("aum", "mean"),
            aum_start=("aum", "first"),
            aum_end=("aum", "last"),
        )
        .reset_index()
    )

    agg["enp"] = (1.0 / agg["hhi"].clip(lower=1e-6)).clip(upper=100.0)
    agg["turnover_rate"] = (
        agg["total_trades"] / agg["avg_aum"].clip(lower=1.0)
    ).clip(0.0, 100.0)
    agg["aum_growth_mom"] = (
        (agg["aum_end"] - agg["aum_start"]) / agg["aum_start"].clip(lower=1.0)
    ).clip(-1.0, 10.0)

    agg["month"] = agg["year_month"].dt.to_timestamp()

    return agg[["user_id", "month", "hhi", "enp", "turnover_rate", "cash_drag", "aum_growth_mom", "avg_aum"]].copy()


if __name__ == "__main__":
    from src.ingestion.schema import load_portfolio_snapshots

    daily = load_portfolio_snapshots()
    sample = daily[daily["user_id"].isin(daily["user_id"].unique()[:50])]
    out = compute(sample)
    logger.info(f"Portfolio features shape: {out.shape}")
    logger.info(f"\n{out.describe()}")
    assert out["hhi"].between(0, 1).all()
    assert out["enp"].ge(1).all()
    logger.info("Smoke test passed.")
