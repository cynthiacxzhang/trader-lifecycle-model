"""
Pandera schemas for all raw parquet files.

Validate on load — raise loudly on violation rather than letting bad data
propagate silently into feature engineering or model training.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check
from loguru import logger


# ── Portfolio snapshots schema ────────────────────────────────────────────────
# Daily rows covering every calendar day, so consecutive gaps must be exactly 1.

PortfolioSchema = DataFrameSchema(
    columns={
        "user_id": Column(str, nullable=False),
        "date": Column("datetime64[ns]", nullable=False),
        "aum": Column(float, Check.ge(0), nullable=False),
        "num_positions": Column(int, Check.ge(1), nullable=False),
        "top_holding_pct": Column(float, [Check.ge(0), Check.le(1)], nullable=False),
        "cash_pct": Column(float, [Check.ge(0), Check.le(1)], nullable=False),
        "realized_pnl": Column(float, nullable=False),
        "unrealized_pnl": Column(float, nullable=False),
        "num_trades_today": Column(int, Check.ge(0), nullable=False),
        "margin_used": Column(float, Check.ge(0), nullable=False),
    },
    checks=[
        Check(
            lambda df: _no_large_date_gaps(df, max_gap_days=1, group_col="user_id"),
            error="date gap > 1 calendar day detected in portfolio snapshots",
        ),
    ],
    coerce=True,
)


# ── User states schema ────────────────────────────────────────────────────────

UserStatesSchema = DataFrameSchema(
    columns={
        "user_id": Column(str, nullable=False),
        "month": Column("datetime64[ns]", nullable=False),
        "archetype": Column(str, Check.isin(["casual", "active", "high_value", "churned"]), nullable=False),
        "initial_archetype": Column(str, Check.isin(["casual", "active", "high_value", "churned"]), nullable=False),
    },
    coerce=True,
)


# ── Market data schema ────────────────────────────────────────────────────────
# Trading-day data only. Regular weekends = 3-day gap (Fri→Mon). US holiday
# Mondays produce 4-day gaps (Fri→Tue). Max allowed = 5 to catch genuine
# pipeline failures without false-alarming on market holidays.

MarketDataSchema = DataFrameSchema(
    columns={
        "date": Column("datetime64[ns]", nullable=False),
        "ticker": Column(str, nullable=False),
        "open": Column(float, Check.gt(0), nullable=False),
        "high": Column(float, Check.gt(0), nullable=False),
        "low": Column(float, Check.gt(0), nullable=False),
        "close": Column(float, Check.gt(0), nullable=False),
        "volume": Column(float, Check.ge(0), nullable=True),
    },
    checks=[
        Check(
            lambda df: _no_large_date_gaps(df, max_gap_days=5, group_col="ticker"),
            error="date gap > 5 calendar days detected in market data",
        ),
        Check(
            lambda df: (df["high"] >= df["low"]).all(),
            error="high < low found in market data",
        ),
    ],
    coerce=True,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _no_large_date_gaps(df: pd.DataFrame, max_gap_days: int, group_col: str) -> bool:
    """
    Return True if no consecutive date gap within any group exceeds max_gap_days.

    Portfolio data (every calendar day) uses max_gap_days=1.
    Market data (trading days only) uses max_gap_days=5 to allow holiday Mondays.
    """
    def max_gap(dates: pd.Series) -> int:
        sorted_dates = dates.sort_values()
        if len(sorted_dates) < 2:
            return 0
        return int(sorted_dates.diff().dropna().dt.days.max())

    worst = df.groupby(group_col)["date"].apply(max_gap)
    return bool((worst <= max_gap_days).all())


# ── Load + validate helpers ───────────────────────────────────────────────────

def load_portfolio_snapshots(path: str | None = None) -> pd.DataFrame:
    from src.config import settings
    p = path or settings.data_raw_dir / "portfolio_snapshots.parquet"
    logger.info(f"Loading portfolio snapshots from {p}")
    df = pd.read_parquet(p)
    return PortfolioSchema.validate(df)


def load_user_states(path: str | None = None) -> pd.DataFrame:
    from src.config import settings
    p = path or settings.data_raw_dir / "user_states.parquet"
    logger.info(f"Loading user states from {p}")
    df = pd.read_parquet(p)
    return UserStatesSchema.validate(df)


def load_market_data(path: str | None = None) -> pd.DataFrame:
    from src.config import settings
    p = path or settings.data_raw_dir / "market_data.parquet"
    logger.info(f"Loading market data from {p}")
    df = pd.read_parquet(p)
    return MarketDataSchema.validate(df)


if __name__ == "__main__":
    portfolio = load_portfolio_snapshots()
    logger.info(f"Portfolio snapshots validated: {portfolio.shape}")

    states = load_user_states()
    logger.info(f"User states validated: {states.shape}")

    market = load_market_data()
    logger.info(f"Market data validated: {market.shape}")

    logger.info("All schemas passed.")
