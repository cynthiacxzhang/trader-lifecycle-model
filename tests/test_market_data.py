"""Tests for market data ingestion and schema validation (Step 3)."""

import numpy as np
import pandas as pd
import pytest
import pandera.pandas as pa

from src.ingestion.market_data import ALL_TICKERS, LARGE_CAP_EQUITIES, SECTOR_ETFS
from src.ingestion.schema import (
    MarketDataSchema,
    PortfolioSchema,
    UserStatesSchema,
    _no_large_date_gaps,
)
from src.config import settings


# ── Ticker basket ─────────────────────────────────────────────────────────────

def test_basket_size():
    assert len(ALL_TICKERS) == 50


def test_basket_no_duplicates():
    assert len(ALL_TICKERS) == len(set(ALL_TICKERS))


def test_spy_in_basket():
    assert "SPY" in ALL_TICKERS


# ── Saved parquet sanity ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def market_df():
    path = settings.data_raw_dir / "market_data.parquet"
    if not path.exists():
        pytest.skip("market_data.parquet not present — run src.ingestion.market_data first")
    return pd.read_parquet(path)


def test_market_tickers_count(market_df):
    # Allow up to 5 failed downloads (transient network issues)
    assert market_df["ticker"].nunique() >= 45


def test_market_columns(market_df):
    assert {"date", "ticker", "open", "high", "low", "close", "volume"}.issubset(market_df.columns)


def test_market_close_positive(market_df):
    assert market_df["close"].gt(0).all()


def test_market_high_ge_low(market_df):
    assert (market_df["high"] >= market_df["low"]).all()


def test_market_rows_per_ticker(market_df):
    counts = market_df.groupby("ticker")["date"].count()
    # 2 years of trading days ≈ 504; allow ±10 for holidays / early listing
    assert counts.between(490, 520).all(), f"Unexpected row counts:\n{counts[~counts.between(490, 520)]}"


def test_market_no_null_ohlc(market_df):
    assert not market_df[["open", "high", "low", "close"]].isnull().any().any()


def test_market_schema_validates(market_df):
    MarketDataSchema.validate(market_df)


# ── Schema validation ─────────────────────────────────────────────────────────

def test_portfolio_schema_validates():
    path = settings.data_raw_dir / "portfolio_snapshots.parquet"
    if not path.exists():
        pytest.skip("portfolio_snapshots.parquet not present")
    df = pd.read_parquet(path)
    PortfolioSchema.validate(df)


def test_user_states_schema_validates():
    path = settings.data_raw_dir / "user_states.parquet"
    if not path.exists():
        pytest.skip("user_states.parquet not present")
    df = pd.read_parquet(path)
    UserStatesSchema.validate(df)


def test_schema_rejects_negative_aum():
    bad = pd.DataFrame({
        "user_id": ["U0"], "date": pd.to_datetime(["2022-01-01"]),
        "aum": [-1.0], "num_positions": [1],
        "top_holding_pct": [0.5], "cash_pct": [0.1],
        "realized_pnl": [0.0], "unrealized_pnl": [0.0],
        "num_trades_today": [0], "margin_used": [0.0],
    })
    with pytest.raises(pa.errors.SchemaError):
        PortfolioSchema.validate(bad)


def test_schema_rejects_top_holding_pct_out_of_range():
    bad = pd.DataFrame({
        "user_id": ["U0"], "date": pd.to_datetime(["2022-01-01"]),
        "aum": [1000.0], "num_positions": [1],
        "top_holding_pct": [1.5], "cash_pct": [0.1],  # > 1
        "realized_pnl": [0.0], "unrealized_pnl": [0.0],
        "num_trades_today": [0], "margin_used": [0.0],
    })
    with pytest.raises(pa.errors.SchemaError):
        PortfolioSchema.validate(bad)


def test_schema_rejects_invalid_archetype():
    bad = pd.DataFrame({
        "user_id": ["U0"],
        "month": pd.to_datetime(["2022-01-01"]),
        "archetype": ["whale"],          # not a valid archetype
        "initial_archetype": ["casual"],
    })
    with pytest.raises(pa.errors.SchemaError):
        UserStatesSchema.validate(bad)


def test_schema_rejects_high_lt_low():
    bad = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03"]),
        "ticker": ["SPY"],
        "open": [400.0], "high": [390.0], "low": [410.0],  # high < low
        "close": [395.0], "volume": [1e6],
    })
    with pytest.raises(pa.errors.SchemaError):
        MarketDataSchema.validate(bad)


# ── Date gap helper ───────────────────────────────────────────────────────────

def test_no_large_date_gaps_passes_for_consecutive():
    df = pd.DataFrame({
        "user_id": ["U0"] * 5,
        "date": pd.date_range("2022-01-01", periods=5, freq="D"),
    })
    assert _no_large_date_gaps(df, max_gap_days=1, group_col="user_id")


def test_no_large_date_gaps_fails_for_large_gap():
    df = pd.DataFrame({
        "user_id": ["U0", "U0"],
        "date": pd.to_datetime(["2022-01-01", "2022-01-10"]),  # 9-day gap
    })
    assert not _no_large_date_gaps(df, max_gap_days=1, group_col="user_id")


def test_no_large_date_gaps_weekend_ok_for_market_data():
    # Friday to Monday = 3-day gap — should pass with max_gap_days=5
    df = pd.DataFrame({
        "ticker": ["SPY", "SPY"],
        "date": pd.to_datetime(["2022-01-07", "2022-01-10"]),  # Fri → Mon
    })
    assert _no_large_date_gaps(df, max_gap_days=5, group_col="ticker")
