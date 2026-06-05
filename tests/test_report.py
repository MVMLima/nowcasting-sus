"""Tests for nowcasting_sus.report module."""

from __future__ import annotations

import os

import numpy as np
import pytest

from nowcasting_sus.report import generate_report


class TestGenerateReport:
    """Tests for generate_report()."""

    def test_returns_html_string(self, synthetic_n_matrix, synthetic_dates,
                                  synthetic_nowcast_median, synthetic_nowcast_low,
                                  synthetic_nowcast_high):
        """Should return a non-empty HTML string."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert isinstance(html, str)
        assert len(html) > 500

    def test_contains_doctype(self, synthetic_n_matrix, synthetic_dates,
                               synthetic_nowcast_median, synthetic_nowcast_low,
                               synthetic_nowcast_high):
        """HTML should start with DOCTYPE declaration."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert html.startswith("<!DOCTYPE html>")

    def test_contains_title(self, synthetic_n_matrix, synthetic_dates,
                             synthetic_nowcast_median, synthetic_nowcast_low,
                             synthetic_nowcast_high):
        """HTML should contain expected title."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
            agravo="Chikungunya",
            uf="Bahia",
        )
        assert "Boletim Nowcasting" in html
        assert "Chikungunya" in html
        assert "Bahia" in html

    def test_contains_custom_agravo_uf(self, synthetic_n_matrix, synthetic_dates,
                                        synthetic_nowcast_median, synthetic_nowcast_low,
                                        synthetic_nowcast_high):
        """Custom agravo and UF should appear in the HTML."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
            agravo="Dengue",
            uf="São Paulo",
            cid="A90",
        )
        assert "Dengue" in html
        assert "São Paulo" in html
        assert "A90" in html

    def test_contains_resumo_section(self, synthetic_n_matrix, synthetic_dates,
                                      synthetic_nowcast_median, synthetic_nowcast_low,
                                      synthetic_nowcast_high):
        """HTML should have a 'Resumo Executivo' section."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert "Resumo Executivo" in html

    def test_contains_table(self, synthetic_n_matrix, synthetic_dates,
                             synthetic_nowcast_median, synthetic_nowcast_low,
                             synthetic_nowcast_high):
        """HTML should contain a table with headers."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert "<table>" in html
        assert "<th>Data</th>" in html
        assert "<th>Nowcast</th>" in html
        assert "<th>IC 95%</th>" in html

    def test_table_has_14_rows(self, synthetic_n_matrix, synthetic_dates,
                                synthetic_nowcast_median, synthetic_nowcast_low,
                                synthetic_nowcast_high):
        """Table should have up to 14 data rows (last 14 days)."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        # Count <tr> tags that contain data (not header row)
        # Each data row starts with <tr>
        tr_count = html.count("<tr>")
        # 1 header row + up to 14 data rows = 15 max
        assert tr_count <= 15
        assert tr_count >= 2  # at least header + 1 data row

    def test_contains_numbers(self, synthetic_n_matrix, synthetic_dates,
                               synthetic_nowcast_median, synthetic_nowcast_low,
                               synthetic_nowcast_high):
        """HTML should contain numeric summary values."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert "casos" in html
        # Should have some numbers in the page
        assert any(c.isdigit() for c in html)

    def test_contains_nota_tecnica(self, synthetic_n_matrix, synthetic_dates,
                                    synthetic_nowcast_median, synthetic_nowcast_low,
                                    synthetic_nowcast_high):
        """HTML should contain technical note."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert "Nota Técnica" in html

    def test_contains_footer(self, synthetic_n_matrix, synthetic_dates,
                              synthetic_nowcast_median, synthetic_nowcast_low,
                              synthetic_nowcast_high):
        """HTML should contain footer with version."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
        )
        assert "nowcasting-sus" in html
        assert "</footer>" in html

    def test_save_to_creates_file(self, synthetic_n_matrix, synthetic_dates,
                                   synthetic_nowcast_median, synthetic_nowcast_low,
                                   synthetic_nowcast_high, tmp_path):
        """save_to parameter should write an HTML file."""
        save_path = str(tmp_path / "test_report.html")
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
            save_to=save_path,
        )
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
        # File content should match returned HTML
        with open(save_path, "r") as f:
            content = f.read()
        assert content == html

    def test_total_observado_custom(self, synthetic_n_matrix, synthetic_dates,
                                     synthetic_nowcast_median, synthetic_nowcast_low,
                                     synthetic_nowcast_high):
        """Custom total_observado should be reflected in the HTML."""
        html = generate_report(
            synthetic_n_matrix,
            synthetic_nowcast_median,
            synthetic_nowcast_low,
            synthetic_nowcast_high,
            synthetic_dates,
            total_observado=999,
        )
        assert "999" in html

    def test_empty_dates_handling(self):
        """Handle very few dates gracefully."""
        nmat = np.ones((3, 5), dtype=int)
        dates = np.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64")
        median = np.array([2.0, 3.0, 4.0])
        low = np.array([1.0, 2.0, 3.0])
        high = np.array([3.0, 4.0, 5.0])

        html = generate_report(nmat, median, low, high, dates)
        # Should have 3 data rows (min of 14 and len of median)
        tr_count = html.count("<tr>")
        assert tr_count <= 4  # 1 header + 3 data
