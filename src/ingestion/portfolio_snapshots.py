"""
Synthetic portfolio snapshot generator.

Produces two parquet files:
  - data/raw/portfolio_snapshots.parquet  — daily rows per user
  - data/raw/user_states.parquet          — monthly archetype labels (ground truth)

Why synthetic with explicit transitions: lets us verify that downstream models
(Markov, survival) recover approximately the parameters injected here. Real data
has no such sanity check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings

# ── Archetype definitions ─────────────────────────────────────────────────────

ARCHETYPES = ["casual", "active", "high_value", "churned"]

ARCHETYPE_PREVALENCE = {"casual": 0.60, "active": 0.25, "high_value": 0.10, "churned": 0.05}

# Monthly transition probabilities (row = from, col = to)
# Order: casual, active, high_value, churned
TRANSITION_MATRIX = np.array([
    [0.97, 0.02, 0.00, 0.01],  # casual      → ...
    [0.00, 0.94, 0.05, 0.01],  # active       → ...
    [0.00, 0.00, 0.99, 0.01],  # high_value   → ...
    [0.00, 0.00, 0.00, 1.00],  # churned      → ... (absorbing)
])

# AUM ranges per archetype (uniform draw on log scale for realism)
AUM_RANGES = {
    "casual":     (500,    5_000),
    "active":     (5_000,  50_000),
    "high_value": (50_000, 500_000),
    "churned":    (1_000,  20_000),  # starts somewhere, decays to zero
}

# Approximate num_positions ranges
POSITION_RANGES = {
    "casual":     (1, 3),
    "active":     (5, 15),
    "high_value": (10, 40),
    "churned":    (1, 5),
}

# Daily trade probability
TRADE_PROB = {
    "casual":     0.05,
    "active":     0.30,
    "high_value": 0.15,
    "churned":    0.02,
}

# Typical cash_pct
CASH_PCT_MEAN = {
    "casual":     0.30,
    "active":     0.10,
    "high_value": 0.05,
    "churned":    0.50,
}


def _log_uniform(low: float, high: float, rng: np.random.Generator) -> float:
    return float(np.exp(rng.uniform(np.log(low), np.log(high))))


def _simulate_user(
    user_id: str,
    archetype: str,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate one user's full daily history and monthly state sequence.

    Returns (daily_df, monthly_states_df).

    Why per-user simulation: vectorising across users would require broadcasting
    across heterogeneous archetype parameters, making the transition logic
    opaque. Per-user loops are slower but exactly mirror the generative process
    described in the spec, which matters for ground-truth validity.
    """
    n_days = len(dates)
    n_months = n_days // 30 + 1

    # ── Assign monthly archetypes via Markov chain ────────────────────────────
    state_idx = ARCHETYPES.index(archetype)
    monthly_states: list[str] = []
    for _ in range(n_months):
        monthly_states.append(ARCHETYPES[state_idx])
        state_idx = rng.choice(4, p=TRANSITION_MATRIX[state_idx])

    # ── Initial AUM ───────────────────────────────────────────────────────────
    aum_low, aum_high = AUM_RANGES[archetype]
    aum = _log_uniform(aum_low, aum_high, rng)

    rows: list[dict] = []
    for day_idx, date in enumerate(dates):
        month_idx = min(day_idx // 30, len(monthly_states) - 1)
        current_state = monthly_states[month_idx]

        # AUM dynamics: drift + noise, capped at zero for churned
        if current_state == "churned":
            # Exponential decay toward zero
            decay_rate = rng.uniform(0.005, 0.015)
            aum = max(0.0, aum * (1 - decay_rate) + rng.normal(0, aum * 0.01 + 1))
        else:
            # Random walk with slight upward drift
            drift = {"casual": 0.0002, "active": 0.0005, "high_value": 0.0003}[current_state]
            vol = {"casual": 0.015, "active": 0.025, "high_value": 0.012}[current_state]
            aum = max(1.0, aum * (1 + drift + rng.normal(0, vol)))

        # Clamp AUM to archetype range (soft — allows drift beyond initial range)
        pos_low, pos_high = POSITION_RANGES[current_state]
        num_positions = int(rng.integers(pos_low, pos_high + 1))

        # top_holding_pct: more concentrated for fewer positions
        top_pct = float(np.clip(rng.beta(1, num_positions), 1 / num_positions, 1.0))

        cash_mean = CASH_PCT_MEAN[current_state]
        cash_pct = float(np.clip(rng.beta(2, max(1, int(2 / cash_mean) - 2)), 0.01, 0.95))

        # PnL: proportional to AUM, noisier for active traders
        pnl_scale = {"casual": 0.002, "active": 0.005, "high_value": 0.003, "churned": 0.001}[current_state]
        realized_pnl = float(rng.normal(0, aum * pnl_scale))
        unrealized_pnl = float(rng.normal(0, aum * pnl_scale * 1.5))

        # Trades
        trade_prob = TRADE_PROB[current_state]
        num_trades = int(rng.poisson(trade_prob * 3)) if rng.random() < trade_prob else 0

        # Margin: active traders use more margin
        margin_prob = {"casual": 0.02, "active": 0.15, "high_value": 0.08, "churned": 0.01}[current_state]
        margin_used = float(aum * rng.beta(1, 5)) if rng.random() < margin_prob else 0.0

        rows.append({
            "user_id": user_id,
            "date": date,
            "aum": round(aum, 2),
            "num_positions": num_positions,
            "top_holding_pct": round(top_pct, 4),
            "cash_pct": round(cash_pct, 4),
            "realized_pnl": round(realized_pnl, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "num_trades_today": num_trades,
            "margin_used": round(margin_used, 2),
        })

    daily_df = pd.DataFrame(rows)

    # ── Monthly state records ─────────────────────────────────────────────────
    month_starts = pd.date_range(dates[0], periods=len(monthly_states), freq="MS")
    states_df = pd.DataFrame({
        "user_id": user_id,
        "month": month_starts[: len(monthly_states)],
        "archetype": monthly_states,
        "initial_archetype": archetype,
    })

    return daily_df, states_df


def generate(
    seed: int | None = None,
    n_users: int | None = None,
    n_days: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic portfolio snapshots for all users.

    n_users / n_days override settings for testing; defaults come from config.
    Returns (portfolio_df, user_states_df).
    """
    n_users = n_users if n_users is not None else settings.n_users
    n_days = n_days if n_days is not None else settings.n_days
    rng = np.random.default_rng(seed if seed is not None else settings.random_seed)

    start_date = pd.Timestamp("2022-01-01")
    dates = pd.date_range(start_date, periods=n_days, freq="D")

    archetype_labels = list(ARCHETYPE_PREVALENCE.keys())
    archetype_probs = list(ARCHETYPE_PREVALENCE.values())

    initial_archetypes = rng.choice(archetype_labels, size=n_users, p=archetype_probs)

    all_daily: list[pd.DataFrame] = []
    all_states: list[pd.DataFrame] = []

    logger.info(f"Simulating {n_users} users × {n_days} days ...")
    for i in range(n_users):
        uid = f"U{i:05d}"
        daily, states = _simulate_user(uid, initial_archetypes[i], dates, rng)
        all_daily.append(daily)
        all_states.append(states)
        if (i + 1) % 500 == 0:
            logger.info(f"  {i + 1}/{n_users} users done")

    portfolio_df = pd.concat(all_daily, ignore_index=True)
    user_states_df = pd.concat(all_states, ignore_index=True)

    logger.info(f"Portfolio snapshots: {portfolio_df.shape}")
    logger.info(f"User states:         {user_states_df.shape}")

    return portfolio_df, user_states_df


def save(portfolio_df: pd.DataFrame, user_states_df: pd.DataFrame) -> None:
    settings.mkdirs()
    out_snapshots = settings.data_raw_dir / "portfolio_snapshots.parquet"
    out_states = settings.data_raw_dir / "user_states.parquet"
    portfolio_df.to_parquet(out_snapshots, index=False)
    user_states_df.to_parquet(out_states, index=False)
    logger.info(f"Saved → {out_snapshots}")
    logger.info(f"Saved → {out_states}")


if __name__ == "__main__":
    portfolio_df, user_states_df = generate()
    save(portfolio_df, user_states_df)

    # ── Smoke test ────────────────────────────────────────────────────────────
    sample = portfolio_df[portfolio_df["user_id"] == "U00000"]
    logger.info(f"\nSample user U00000 — first 5 rows:\n{sample.head()}")
    logger.info(f"\nArchetype distribution:\n{user_states_df.groupby('initial_archetype').size()}")
    assert portfolio_df["aum"].ge(0).all(), "negative AUM found"
    assert portfolio_df["top_holding_pct"].between(0, 1).all(), "top_holding_pct out of [0,1]"
    assert portfolio_df["cash_pct"].between(0, 1).all(), "cash_pct out of [0,1]"
    logger.info("Smoke test passed.")
