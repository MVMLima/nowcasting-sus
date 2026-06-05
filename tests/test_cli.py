"""Tests for nowcasting_sus.cli module.

Only tests argument parsing — does NOT execute the full CLI pipeline
(which would trigger PyMC MCMC sampling).

Note: cli.main() imports load_sinan, prepare_matrix, NowcastingModel, and
generate_report INSIDE the function body, so we patch them at the module
level where they live.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _mock_all_deps():
    """Mock all heavy dependencies used inside cli.main()."""
    # Mock NowcastingModel instance chain
    mock_model_instance = MagicMock()
    mock_model_instance.fit.return_value = None
    mock_model_instance.get_nowcast_ci.return_value = (
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 1.0, 1.5]),
        np.array([1.5, 3.0, 4.5]),
    )

    patches = [
        patch("nowcasting_sus.data.load_sinan"),
        patch("nowcasting_sus.data.prepare_matrix", return_value=(
            np.zeros((5, 10)),
            np.array(["2024-01-01"], dtype="datetime64"),
            np.array([0, 1, 2], dtype=int),
            np.array([0, 0, 1], dtype=int),
            np.array([3, 2, 1], dtype=int),
        )),
        patch("nowcasting_sus.models.NowcastingModel", return_value=mock_model_instance),
        patch("nowcasting_sus.report.generate_report"),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


class TestCliArgs:
    """Tests for CLI argument parsing."""

    def test_parser_requires_csv(self):
        """Parser should fail without csv argument."""
        test_args = ["program"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                from nowcasting_sus.cli import main
                main()

    def test_parser_accepts_defaults(self):
        """With only csv path, parsing should succeed."""
        test_args = ["program", "dados.csv"]
        with patch.object(sys, "argv", test_args):
            from nowcasting_sus.cli import main
            main()

    def test_parser_accepts_all_args(self):
        """All CLI arguments should be parseable."""
        test_args = [
            "program",
            "data.csv",
            "--agravo", "Dengue",
            "--uf", "Acre",
            "--cid", "A90",
            "--max-delay", "20",
            "--output", "./relatorio.html",
        ]
        with patch.object(sys, "argv", test_args):
            from nowcasting_sus.cli import main
            main()

    def test_parser_accepts_short_output(self):
        """Short form -o should work."""
        test_args = [
            "program",
            "data.csv",
            "-o", "./outro.html",
        ]
        with patch.object(sys, "argv", test_args):
            from nowcasting_sus.cli import main
            main()

    def test_parser_max_delay_int(self):
        """--max-delay should accept integer values."""
        test_args = ["program", "data.csv", "--max-delay", "15"]
        with patch.object(sys, "argv", test_args):
            from nowcasting_sus.cli import main
            main()
