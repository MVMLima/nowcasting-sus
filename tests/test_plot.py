"""Tests for nowcasting_sus.plot module.

Uses matplotlib 'Agg' backend (non-interactive) for all tests.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be set before importing pyplot

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nowcasting_sus.plot import plot_dow_effect, plot_nowcasting, plot_panel


# Ensure a clean figure state before each test
def setup_function():
    plt.close("all")


def teardown_function():
    plt.close("all")


class TestPlotNowcasting:
    """Tests for plot_nowcasting()."""

    def test_returns_figure(self, synthetic_n_matrix, synthetic_dates,
                            synthetic_nowcast_median, synthetic_nowcast_low,
                            synthetic_nowcast_high):
        """Should return a matplotlib Figure."""
        fig = plot_nowcasting(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_default_size(self, synthetic_n_matrix, synthetic_dates,
                                  synthetic_nowcast_median, synthetic_nowcast_low,
                                  synthetic_nowcast_high):
        """Default figure size should be (12, 5)."""
        fig = plot_nowcasting(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(12, rel=0.1)
        assert h == pytest.approx(5, rel=0.1)

    def test_custom_figsize(self, synthetic_n_matrix, synthetic_dates,
                            synthetic_nowcast_median, synthetic_nowcast_low,
                            synthetic_nowcast_high):
        """Custom figsize should be applied."""
        fig = plot_nowcasting(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
            figsize=(8, 4),
        )
        w, h = fig.get_size_inches()
        assert w == pytest.approx(8, rel=0.1)
        assert h == pytest.approx(4, rel=0.1)

    def test_custom_title(self, synthetic_n_matrix, synthetic_dates,
                          synthetic_nowcast_median, synthetic_nowcast_low,
                          synthetic_nowcast_high):
        """Custom title should appear in the figure."""
        fig = plot_nowcasting(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
            title="Test Title",
        )
        ax = fig.axes[0]
        assert ax.get_title() == "Test Title"

    def test_two_lines_in_plot(self, synthetic_n_matrix, synthetic_dates,
                               synthetic_nowcast_median, synthetic_nowcast_low,
                               synthetic_nowcast_high):
        """Plot should have 2 line objects (observed + nowcast)."""
        fig = plot_nowcasting(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) == 2

    def test_fill_between_present(self, synthetic_n_matrix, synthetic_dates,
                                   synthetic_nowcast_median, synthetic_nowcast_low,
                                   synthetic_nowcast_high):
        """Plot should have a fill between (CI band)."""
        fig = plot_nowcasting(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        ax = fig.axes[0]
        collections = ax.collections
        assert len(collections) >= 1

    def test_save_to_creates_file(self, synthetic_n_matrix, synthetic_dates,
                                   synthetic_nowcast_median, synthetic_nowcast_low,
                                   synthetic_nowcast_high, tmp_path):
        """save_to parameter should write a file."""
        save_path = str(tmp_path / "test_plot.png")
        plot_nowcasting(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
            save_to=save_path,
        )
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


class TestPlotPanel:
    """Tests for plot_panel()."""

    def test_returns_figure(self, synthetic_n_matrix, synthetic_dates,
                            synthetic_nowcast_median, synthetic_nowcast_low,
                            synthetic_nowcast_high, synthetic_delay_posterior):
        """Should return a matplotlib Figure (2×2 layout)."""
        fig = plot_panel(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_delay_posterior,
            synthetic_dates,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_two_by_two_layout(self, synthetic_n_matrix, synthetic_dates,
                                synthetic_nowcast_median, synthetic_nowcast_low,
                                synthetic_nowcast_high, synthetic_delay_posterior):
        """Without dow_effect should be 2×2 = 4 plot axes (plus colorbar)."""
        fig = plot_panel(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_delay_posterior,
            synthetic_dates,
        )
        axes_no_cbar = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        assert len(axes_no_cbar) == 4

    def test_with_dow_effect_three_columns(self, synthetic_n_matrix, synthetic_dates,
                                            synthetic_nowcast_median,
                                            synthetic_nowcast_low,
                                            synthetic_nowcast_high,
                                            synthetic_delay_posterior,
                                            synthetic_dow_effect):
        """With dow_effect should be 2×3 = 6 plot axes (plus colorbar)."""
        fig = plot_panel(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_delay_posterior,
            synthetic_dates,
            dow_effect=synthetic_dow_effect,
        )
        axes_no_cbar = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        assert len(axes_no_cbar) == 6

    def test_zoom_days_parameter(self, synthetic_n_matrix, synthetic_dates,
                                  synthetic_nowcast_median, synthetic_nowcast_low,
                                  synthetic_nowcast_high, synthetic_delay_posterior):
        """Custom zoom_days should be accepted."""
        fig = plot_panel(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_delay_posterior,
            synthetic_dates,
            zoom_days=10,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_save_to_creates_file(self, synthetic_n_matrix, synthetic_dates,
                                   synthetic_nowcast_median, synthetic_nowcast_low,
                                   synthetic_nowcast_high, synthetic_delay_posterior,
                                   tmp_path):
        """save_to parameter should write a file."""
        save_path = str(tmp_path / "test_panel.png")
        plot_panel(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_delay_posterior,
            synthetic_dates,
            save_to=save_path,
        )
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


class TestPlotDowEffect:
    """Tests for plot_dow_effect()."""

    def test_returns_figure(self, synthetic_dow_effect):
        """Should return a matplotlib Figure."""
        fig = plot_dow_effect(synthetic_dow_effect)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_seven_bars(self, synthetic_dow_effect):
        """Should have 7 bar objects (one per day)."""
        fig = plot_dow_effect(synthetic_dow_effect)
        ax = fig.axes[0]
        bars = ax.patches
        assert len(bars) == 7

    def test_has_hline_at_one(self, synthetic_dow_effect):
        """Should have a horizontal line at y=1."""
        fig = plot_dow_effect(synthetic_dow_effect)
        ax = fig.axes[0]
        lines = ax.lines
        assert len(lines) >= 1
        # One of the lines should be at y=1
        yvals = [l.get_ydata() for l in lines]
        assert any(np.allclose(y, 1.0) for y in yvals), "No line at y=1"

    def test_values_labeled_on_bars(self, synthetic_dow_effect):
        """Bar values should be displayed above bars."""
        fig = plot_dow_effect(synthetic_dow_effect)
        ax = fig.axes[0]
        texts = ax.texts
        assert len(texts) == 7

    def test_save_to_creates_file(self, synthetic_dow_effect, tmp_path):
        """save_to parameter should write a file."""
        save_path = str(tmp_path / "test_dow.png")
        fig = plot_dow_effect(synthetic_dow_effect, save_to=save_path)
        import os
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
