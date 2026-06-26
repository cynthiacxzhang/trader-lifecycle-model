"""
Market data ingestion via yfinance.

Pulls 2 years of daily OHLCV for a 50-ticker basket and writes
data/raw/market_data.parquet. The basket covers large-caps, sector ETFs,
fixed income, crypto proxies, and international ETFs so that downstream
feature engineering (Sharpe, beta, PnL simulation) has a realistic
cross-asset universe.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from loguru import logger

from src.config import settings

# ── Ticker basket ─────────────────────────────────────────────────────────────

LARGE_CAP_EQUITIES = [
    "SPY", "AAPL", "MSFT", "AMZN", "GOOGL",
    "META", "NVDA", "TSLA", "BRK-B", "JPM",
    "JNJ", "UNH", "V", "XOM", "PG",
    "HD", "MA", "CVX", "MRK", "ABBV",
]

SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Health Care
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
]

FIXED_INCOME_ETFS = [
    "AGG",   # US Aggregate Bond
    "BND",   # Vanguard Total Bond
    "TLT",   # 20+ Year Treasury
    "IEF",   # 7-10 Year Treasury
    "SHY",   # 1-3 Year Treasury
    "LQD",   # Investment Grade Corp
    "HYG",   # High Yield Corp
    "MUB",   # Muni Bond
    "TIP",   # TIPS (inflation-protected)
    "VCIT",  # Vanguard Intermediate Corp
]

CRYPTO_PROXIES = [
    "BITO",  # Bitcoin futures ETF
    "GBTC",  # Grayscale Bitcoin Trust
    "ETHE",  # Grayscale Ethereum Trust
    "MSTR",  # MicroStrategy (BTC proxy)
    "COIN",  # Coinbase
]

INTERNATIONAL_ETFS = [
    "EFA",   # Developed Markets ex-US
    "EEM",   # Emerging Markets
    "VEU",   # All-World ex-US
    "FXI",   # China Large-Cap
    "EWJ",   # Japan
]

ALL_TICKERS: list[str] = (
    LARGE_CAP_EQUITIES
    + SECTOR_ETFS
    + FIXED_INCOME_ETFS
    + CRYPTO_PROXIES
    + INTERNATIONAL_ETFS
)

assert len(ALL_TICKERS) == 50, f"Expected 50 tickers, got {len(ALL_TICKERS)}"

# Matches the synthetic data window from portfolio_snapshots.py
_START = "2022-01-01"
_END = "2024-01-01"


def _download_batch(tickers: list[str]) -> pd.DataFrame:
    """Download a batch and reshape to long format."""
    raw = yf.download(
        tickers,
        start=_START,
        end=_END,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    raw.columns.names = ["field", "ticker"]
    long = (
        raw.stack(level="ticker", future_stack=True)
        .reset_index()
    )
    long.columns = [c.lower() for c in long.columns]
    # Normalise whatever yfinance calls the date column
    date_col = next((c for c in long.columns if c in ("date", "price")), None)
    if date_col and date_col != "date":
        long = long.rename(columns={date_col: "date"})
    return long


def fetch(tickers: list[str] = ALL_TICKERS, max_retries: int = 2) -> pd.DataFrame:
    """
    Download daily OHLCV for all tickers and return a tidy long-format DataFrame.

    Failed tickers are retried individually up to max_retries times before being
    dropped. A transient yfinance cache lock on a single ticker should not abort
    the whole download.

    Why long format: downstream feature modules join on (ticker, date), which is
    simpler with a tidy frame than a wide multi-index. Pandera schema validation
    also works cleanly on long frames.
    """
    logger.info(f"Downloading OHLCV for {len(tickers)} tickers ({_START} → {_END}) ...")

    long = _download_batch(tickers)

    if long.empty:
        raise RuntimeError("yfinance returned an empty DataFrame for all tickers")

    fetched = set(long["ticker"].unique()) if "ticker" in long.columns else set()
    missing = [t for t in tickers if t not in fetched]

    for attempt in range(1, max_retries + 1):
        if not missing:
            break
        logger.warning(f"Retrying {len(missing)} failed tickers (attempt {attempt}): {missing}")
        retry_df = _download_batch(missing)
        if not retry_df.empty and "ticker" in retry_df.columns:
            long = pd.concat([long, retry_df], ignore_index=True)
            fetched = set(long["ticker"].unique())
            missing = [t for t in missing if t not in fetched]

    if missing:
        logger.warning(f"Could not fetch {len(missing)} tickers after retries: {missing}")

    required = {"date", "ticker", "open", "high", "low", "close", "volume"}
    missing_cols = required - set(long.columns)
    if missing_cols:
        raise ValueError(f"Missing columns after reshape: {missing_cols}")

    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"])
    long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(f"Fetched {long['ticker'].nunique()} tickers, {len(long):,} rows")
    return long[["date", "ticker", "open", "high", "low", "close", "volume"]]


def save(df: pd.DataFrame) -> None:
    settings.mkdirs()
    out = settings.data_raw_dir / "market_data.parquet"
    df.to_parquet(out, index=False)
    logger.info(f"Saved → {out}")


if __name__ == "__main__":
    df = fetch()

    # ── Smoke test ────────────────────────────────────────────────────────────
    n_tickers = df["ticker"].nunique()
    n_dates = df.groupby("ticker")["date"].count()
    logger.info(f"Tickers: {n_tickers}")
    logger.info(f"Rows per ticker — min: {n_dates.min()}, max: {n_dates.max()}")
    assert n_tickers >= 45, f"Expected ≥45 tickers (some may delist), got {n_tickers}"
    assert df["close"].gt(0).all(), "Non-positive close prices found"
    assert df["volume"].ge(0).all(), "Negative volume found"
    assert not df[["date", "ticker", "close"]].isnull().any().any(), "Nulls in key columns"

    save(df)
    logger.info("Smoke test passed.")
