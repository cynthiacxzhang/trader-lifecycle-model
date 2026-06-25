"""Tests for synthetic data generator (Step 2)."""

import numpy as np
import pandas as pd
import pytest

from src.ingestion.portfolio_snapshots import (
    ARCHETYPE_PREVALENCE,
    ARCHETYPES,
    TRANSITION_MATRIX,
    generate,
)

N_USERS = 200
N_DAYS = 60


@pytest.fixture(scope="module")
def data():
    portfolio, states = generate(seed=0, n_users=N_USERS, n_days=N_DAYS)
    return portfolio, states


# ── Shape ─────────────────────────────────────────────────────────────────────

def test_portfolio_row_count(data):
    portfolio, _ = data
    assert len(portfolio) == N_USERS * N_DAYS


def test_portfolio_columns(data):
    portfolio, _ = data
    expected = {"user_id", "date", "aum", "num_positions", "top_holding_pct",
                "cash_pct", "realized_pnl", "unrealized_pnl", "num_trades_today", "margin_used"}
    assert expected == set(portfolio.columns)


def test_unique_users(data):
    portfolio, _ = data
    assert portfolio["user_id"].nunique() == N_USERS


def test_each_user_has_all_dates(data):
    portfolio, _ = data
    counts = portfolio.groupby("user_id")["date"].count()
    assert (counts == N_DAYS).all()


def test_user_states_columns(data):
    _, states = data
    assert {"user_id", "month", "archetype", "initial_archetype"}.issubset(states.columns)


# ── Domain constraints ────────────────────────────────────────────────────────

def test_aum_non_negative(data):
    portfolio, _ = data
    assert portfolio["aum"].ge(0).all()


def test_top_holding_pct_in_range(data):
    portfolio, _ = data
    assert portfolio["top_holding_pct"].between(0, 1).all()


def test_cash_pct_in_range(data):
    portfolio, _ = data
    assert portfolio["cash_pct"].between(0, 1).all()


def test_num_trades_non_negative(data):
    portfolio, _ = data
    assert portfolio["num_trades_today"].ge(0).all()


def test_margin_non_negative(data):
    portfolio, _ = data
    assert portfolio["margin_used"].ge(0).all()


def test_num_positions_positive(data):
    portfolio, _ = data
    assert portfolio["num_positions"].ge(1).all()


# ── Archetype prevalence (approximate) ───────────────────────────────────────

def test_archetype_prevalence_roughly_correct(data):
    """Initial archetype counts should be within ±10pp of specified prevalence."""
    _, states = data
    first_month = states[states["month"] == states["month"].min()]
    dist = first_month["initial_archetype"].value_counts(normalize=True)
    for archetype, expected_p in ARCHETYPE_PREVALENCE.items():
        actual_p = dist.get(archetype, 0.0)
        assert abs(actual_p - expected_p) < 0.10, (
            f"{archetype}: expected ~{expected_p:.0%}, got {actual_p:.0%}"
        )


# ── Markov structure ──────────────────────────────────────────────────────────

def test_transition_matrix_rows_sum_to_one():
    assert np.allclose(TRANSITION_MATRIX.sum(axis=1), 1.0)


def test_churned_is_absorbing():
    churned_idx = ARCHETYPES.index("churned")
    assert TRANSITION_MATRIX[churned_idx, churned_idx] == 1.0


def test_churned_users_aum_decays(data):
    """Users who start as churned should end with lower AUM than they started."""
    portfolio, states = data
    churned_ids = states[states["initial_archetype"] == "churned"]["user_id"].unique()
    if len(churned_ids) == 0:
        pytest.skip("no churned users in this small sample")
    sub = portfolio[portfolio["user_id"].isin(churned_ids)]
    start_aum = sub.groupby("user_id")["aum"].first()
    end_aum = sub.groupby("user_id")["aum"].last()
    assert (end_aum < start_aum).mean() > 0.5


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_reproducible():
    p1, _ = generate(seed=99, n_users=10, n_days=10)
    p2, _ = generate(seed=99, n_users=10, n_days=10)
    pd.testing.assert_frame_equal(p1, p2)


def test_different_seeds_differ():
    p1, _ = generate(seed=1, n_users=10, n_days=10)
    p2, _ = generate(seed=2, n_users=10, n_days=10)
    assert not p1["aum"].equals(p2["aum"])
