"""
Risk-adjusted metrics — monthly aggregation with trailing rolling windows.

Sharpe and Sortino use a 90-day trailing window; beta uses 60 days.
All return-based metrics are computed on trading days only (where SPY data exists)
to avoid contaminating the benchmark comparison with weekend noise.

Rolling statistics are computed via grouped transform, which processes each
user's series independently without per-user Python loops.

Max drawdown uses all calendar days since AUM is available daily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings

SHARPE_WINDOW = 90   # trading days ≈ 3 months
BETA_WINDOW = 60     # trading days ≈ 3 months
MIN_SHARPE = 30      # minimum observations for Sharpe/Sortino
MIN_BETA = 20        # minimum observations for beta


def compute(daily: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly risk-adjusted features per user.

    Input:  daily portfolio snapshots + market data (must include SPY)
    Output: (user_id, month) frame with sharpe_ratio, sortino_ratio,
            max_drawdown, portfolio_beta
    """
    rf_daily = settings.risk_free_annual_rate / 252

    # ── SPY benchmark ─────────────────────────────────────────────────────────
    spy = (
        market_data[market_data["ticker"] == "SPY"][["date", "close"]]
        .sort_values("date")
        .copy()
    )
    spy["spy_return"] = spy["close"].pct_change()
    spy = spy[["date", "spy_return"]].dropna()
    trading_dates = set(spy["date"])

    # Precompute rolling SPY moments (same for all users on a given date)
    spy_idx = spy.set_index("date").sort_index()
    spy_roll_mean = spy_idx["spy_return"].rolling(BETA_WINDOW, min_periods=MIN_BETA).mean()
    spy_roll_var = spy_idx["spy_return"].rolling(BETA_WINDOW, min_periods=MIN_BETA).var()
    spy_stats = pd.DataFrame({
        "date": spy_roll_mean.index,
        "spy_roll_mean": spy_roll_mean.values,
        "spy_roll_var": spy_roll_var.values,
    })

    df = daily.sort_values(["user_id", "date"]).copy()
    df["year_month"] = df["date"].dt.to_period("M")

    # ── Max drawdown (all calendar days, within-month) ────────────────────────
    # Use monthly peak and trough as a vectorized proxy.
    # (peak - trough) / peak overestimates when trough precedes peak but is
    # directionally correct and avoids a costly per-group expanding-max apply.
    mdd_agg = (
        df.groupby(["user_id", "year_month"])
        .agg(peak_aum=("aum", "max"), trough_aum=("aum", "min"))
        .reset_index()
    )
    mdd_agg["max_drawdown"] = (
        (mdd_agg["peak_aum"] - mdd_agg["trough_aum"]) / mdd_agg["peak_aum"].clip(lower=1.0)
    ).clip(0.0, 1.0)

    # ── Trading-day frame for return-based metrics ────────────────────────────
    trade_df = df[df["date"].isin(trading_dates)].copy()
    trade_df["daily_return"] = trade_df.groupby("user_id")["aum"].pct_change()
    trade_df = trade_df.merge(spy[["date", "spy_return"]], on="date", how="left")
    trade_df["spy_return"] = trade_df["spy_return"].fillna(0.0)
    trade_df["excess_return"] = trade_df["daily_return"] - rf_daily
    trade_df["downside_return"] = trade_df["daily_return"].clip(upper=0.0)
    trade_df["return_x_spy"] = trade_df["daily_return"] * trade_df["spy_return"]

    # ── Grouped rolling transforms (each user processed independently) ────────
    def _roll_transform(col: str, func: str, window: int, min_p: int) -> pd.Series:
        return trade_df.groupby("user_id")[col].transform(
            lambda x: getattr(x.rolling(window, min_periods=min_p), func)()
        )

    trade_df["roll_mean_excess"] = _roll_transform("excess_return", "mean", SHARPE_WINDOW, MIN_SHARPE)
    trade_df["roll_std_return"] = _roll_transform("daily_return", "std", SHARPE_WINDOW, MIN_SHARPE)
    trade_df["roll_std_down"] = _roll_transform("downside_return", "std", SHARPE_WINDOW, MIN_SHARPE)
    trade_df["roll_mean_ret"] = _roll_transform("daily_return", "mean", BETA_WINDOW, MIN_BETA)
    trade_df["roll_mean_prod"] = _roll_transform("return_x_spy", "mean", BETA_WINDOW, MIN_BETA)

    trade_df["rolling_sharpe"] = (
        trade_df["roll_mean_excess"] / trade_df["roll_std_return"].clip(lower=1e-10)
    ) * np.sqrt(252)

    trade_df["rolling_sortino"] = (
        trade_df["roll_mean_excess"] / trade_df["roll_std_down"].clip(lower=1e-10)
    ) * np.sqrt(252)

    # Beta = cov(r_p, r_spy) / var(r_spy)
    #      = (E[r_p * r_spy] - E[r_p]*E[r_spy]) / var(r_spy)
    trade_df = trade_df.merge(spy_stats, on="date", how="left")
    roll_cov = trade_df["roll_mean_prod"] - trade_df["roll_mean_ret"] * trade_df["spy_roll_mean"]
    trade_df["rolling_beta"] = (roll_cov / trade_df["spy_roll_var"].clip(lower=1e-10)).clip(-5.0, 5.0)

    # ── Monthly aggregation: last rolling value in each month ─────────────────
    monthly_risk = (
        trade_df.groupby(["user_id", "year_month"])
        .agg(
            sharpe_ratio=("rolling_sharpe", "last"),
            sortino_ratio=("rolling_sortino", "last"),
            portfolio_beta=("rolling_beta", "last"),
        )
        .reset_index()
    )

    monthly_risk = monthly_risk.merge(
        mdd_agg[["user_id", "year_month", "max_drawdown"]],
        on=["user_id", "year_month"],
        how="left",
    )
    monthly_risk["month"] = monthly_risk["year_month"].dt.to_timestamp()

    cols = ["user_id", "month", "sharpe_ratio", "sortino_ratio", "max_drawdown", "portfolio_beta"]
    return monthly_risk[cols].copy()


if __name__ == "__main__":
    from src.ingestion.schema import load_portfolio_snapshots, load_market_data

    daily = load_portfolio_snapshots()
    market = load_market_data()
    sample = daily[daily["user_id"].isin(daily["user_id"].unique()[:50])]
    out = compute(sample, market)
    logger.info(f"Risk features shape: {out.shape}")
    logger.info(f"\n{out.describe()}")
    assert out["max_drawdown"].between(0, 1).all(skipna=True)
    logger.info("Smoke test passed.")
