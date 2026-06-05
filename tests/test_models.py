"""Tests for nowcasting_sus.models module.

NOTE: These tests do NOT run PyMC MCMC sampling. They test:
  - _build_prior_delays()          (pure numpy, no PyMC)
  - build()                        (constructs pm.Model, no sampling)
  - get_nowcast_ci / get_nowcast   (error handling before fit)
  - NowcastingModelDOW.build()     (with dow kwarg)
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from nowcasting_sus.models import NowcastingModel, NowcastingModelDOW, _build_prior_delays


# ── _build_prior_delays ─────────────────────────────────────────────────


class TestBuildPriorDelays:
    """Tests for the standalone _build_prior_delays function."""

    def test_shape(self):
        """Output shape should be max_delay + 1."""
        for md in [5, 10, 30, 60]:
            result = _build_prior_delays(md)
            assert result.shape == (md + 1,)

    def test_all_positive(self):
        """All values should be positive."""
        result = _build_prior_delays(30)
        assert np.all(result > 0)

    def test_early_days_have_higher_weight(self):
        """Days 0-7 should have higher values than later days."""
        result = _build_prior_delays(30)
        first_week = result[:8]
        later = result[15:]
        # Each element in the first week should be > corresponding later
        assert np.all(first_week.min() > later.max()), (
            f"Early days too low: first_week min={first_week.min():.2f}, later max={later.max():.2f}"
        )

    def test_default_alpha_scale(self):
        """Default alpha_scale=3.0 should produce consistent values."""
        result = _build_prior_delays(10)
        base = 3.0 / 11
        assert result[0] == pytest.approx(base + 5.0, rel=0.01)

    def test_custom_alpha_scale(self):
        """Custom alpha_scale should change the base value."""
        result = _build_prior_delays(10, alpha_scale=6.0)
        base = 6.0 / 11
        assert result[0] == pytest.approx(base + 5.0, rel=0.01)

    def test_delays_8_to_14_have_middle_weight(self):
        """Days 8-14 should have medium additional weight."""
        result = _build_prior_delays(30)
        base = 3.0 / 31
        assert result[8] == pytest.approx(base + 3.0, rel=0.01)


# ── NowcastingModel ──────────────────────────────────────────────────────


class TestNowcastingModel:
    """Tests for NowcastingModel."""

    def test_build_returns_pm_model(self):
        """build() should return a pymc Model object."""
        T, D = 10, 5
        obs_t = np.array([0, 1, 2, 3], dtype=int)
        obs_d = np.array([0, 0, 1, 2], dtype=int)
        counts = np.array([3, 5, 2, 1], dtype=int)

        model = NowcastingModel()
        result = model.build(obs_t, obs_d, counts, T, D)
        assert isinstance(result, pm.Model)
        # Check build() stores model reference
        assert model.model_ is result

    def test_build_model_has_expected_variables(self):
        """The built model should contain expected PyMC variables."""
        T, D = 15, 7
        obs_t = np.array([0, 1, 2, 3, 5], dtype=int)
        obs_d = np.array([0, 0, 1, 2, 0], dtype=int)
        counts = np.array([4, 3, 5, 1, 2], dtype=int)

        model = NowcastingModel(sigma_rw=0.5, alpha_nb=5.0)
        pm_model = model.build(obs_t, obs_d, counts, T, D)

        # Check model has expected variable names
        var_names = [v.name for v in pm_model.free_RVs]
        assert "sigma_rw" in var_names
        assert "f_t" in var_names
        assert "delay_p" in var_names
        # obs is observed, so it won't be a free RV
        assert "obs" not in var_names  # It's an observed variable

    def test_get_nowcast_raises_before_fit(self):
        """get_nowcast() should raise RuntimeError before fit()."""
        model = NowcastingModel()
        with pytest.raises(RuntimeError, match="não ajustado"):
            model.get_nowcast()

    def test_get_nowcast_ci_raises_before_fit(self):
        """get_nowcast_ci() should raise RuntimeError before fit()."""
        model = NowcastingModel()
        with pytest.raises(RuntimeError, match="não ajustado"):
            model.get_nowcast_ci()

    def test_default_params(self):
        """Default constructor parameters should match expectations."""
        model = NowcastingModel()
        assert model.sigma_rw == 0.1
        assert model.alpha_nb == 10.0
        assert model.alpha_scale == 3.0
        assert model.model_ is None
        assert model.idata_ is None

    def test_custom_params(self):
        """Custom constructor parameters should be stored."""
        model = NowcastingModel(sigma_rw=0.5, alpha_nb=20.0, alpha_scale=5.0)
        assert model.sigma_rw == 0.5
        assert model.alpha_nb == 20.0
        assert model.alpha_scale == 5.0


# ── NowcastingModelDOW ───────────────────────────────────────────────────


class TestNowcastingModelDOW:
    """Tests for NowcastingModelDOW."""

    def test_build_requires_dow_kwarg(self):
        """DOW build() requires the 'dow' keyword argument."""
        T, D = 10, 5
        obs_t = np.array([0, 1, 2], dtype=int)
        obs_d = np.array([0, 0, 1], dtype=int)
        counts = np.array([3, 2, 1], dtype=int)
        model = NowcastingModelDOW()
        # Without dow kwarg should raise TypeError
        with pytest.raises(TypeError):
            model.build(obs_t, obs_d, counts, T, D)

    def test_build_with_dow_returns_pm_model(self):
        """DOW build() with dow should return a pm.Model."""
        T, D = 10, 5
        obs_t = np.array([0, 1, 2, 3], dtype=int)
        obs_d = np.array([0, 0, 1, 2], dtype=int)
        counts = np.array([3, 5, 2, 1], dtype=int)
        dow = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2], dtype=int)

        model = NowcastingModelDOW(sigma_dow=0.5)
        result = model.build(obs_t, obs_d, counts, T, D, dow=dow)
        assert isinstance(result, pm.Model)

    def test_build_dow_has_beta_dow(self):
        """DOW model should have beta_dow variable."""
        T, D = 10, 5
        obs_t = np.array([0, 1, 2], dtype=int)
        obs_d = np.array([0, 0, 1], dtype=int)
        counts = np.array([3, 2, 1], dtype=int)
        dow = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2], dtype=int)

        model = NowcastingModelDOW()
        pm_model = model.build(obs_t, obs_d, counts, T, D, dow=dow)
        var_names = [v.name for v in pm_model.free_RVs]
        assert "beta_dow" in var_names

    def test_dow_default_sigma(self):
        """DOW constructor should have sigma_dow=0.3."""
        model = NowcastingModelDOW()
        assert model.sigma_dow == 0.3
        assert model.sigma_rw == 0.1  # inherited

    def test_get_dow_effect_raises_before_fit(self):
        """get_dow_effect() should raise RuntimeError before fit()."""
        model = NowcastingModelDOW()
        with pytest.raises(RuntimeError, match="não ajustado"):
            model.get_dow_effect()
