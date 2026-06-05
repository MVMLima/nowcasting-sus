"""Shared fixtures for nowcasting-sus tests."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_dates() -> np.ndarray:
    """Array of 30 consecutive dates as datetime64."""
    start = datetime(2024, 1, 1)
    return np.array(
        [start + timedelta(days=i) for i in range(30)],
        dtype="datetime64",
    )


@pytest.fixture
def synthetic_n_matrix() -> np.ndarray:
    """Small 30 × 10 matrix of onset × delay counts."""
    rng = np.random.default_rng(42)
    nmat = rng.poisson(lam=3, size=(30, 10)).astype(int)
    # Ensure some zeros for realism
    nmat[0, :] = 0
    nmat[15, 5:] = 0
    return nmat


@pytest.fixture
def synthetic_nowcast_median(synthetic_dates) -> np.ndarray:
    """Synthetic nowcast median estimates (already exp'd)."""
    n = len(synthetic_dates)
    # Rising trend: 5 → 20 over 30 days
    trend = np.linspace(5.0, 20.0, n)
    rng = np.random.default_rng(123)
    return trend + rng.uniform(-2, 2, size=n)


@pytest.fixture
def synthetic_nowcast_low(synthetic_nowcast_median) -> np.ndarray:
    """Lower bound ~10% below median."""
    return synthetic_nowcast_median * 0.8


@pytest.fixture
def synthetic_nowcast_high(synthetic_nowcast_median) -> np.ndarray:
    """Upper bound ~15% above median."""
    return synthetic_nowcast_median * 1.15


@pytest.fixture
def synthetic_delay_posterior() -> np.ndarray:
    """Small delay posterior: 100 samples × 10 delay bins."""
    rng = np.random.default_rng(456)
    raw = rng.dirichlet(np.ones(10) * 2, size=100)
    return raw


@pytest.fixture
def synthetic_dow_effect() -> dict:
    """Day-of-week effect dict."""
    return {
        "Seg": 1.20,
        "Ter": 1.10,
        "Qua": 1.05,
        "Qui": 1.00,
        "Sex": 0.95,
        "Sáb": 0.60,
        "Dom": 0.55,
    }


@pytest.fixture
def synthetic_obs_t() -> np.ndarray:
    """Observation time indices (non-zero cells in matrix)."""
    # Simplified: just a subset of time indices
    rng = np.random.default_rng(789)
    t_vals = np.repeat(np.arange(30), 3)  # 3 obs per day
    return t_vals[:80]  # 80 observations


@pytest.fixture
def synthetic_obs_d() -> np.ndarray:
    """Observation delay indices."""
    rng = np.random.default_rng(789)
    return rng.integers(0, 10, size=80)


@pytest.fixture
def synthetic_counts() -> np.ndarray:
    """Counts for each (t, d) pair."""
    rng = np.random.default_rng(789)
    return rng.poisson(lam=3, size=80).astype(int)
