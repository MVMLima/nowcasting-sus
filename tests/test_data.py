"""Tests for nowcasting_sus.data module."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from nowcasting_sus.data import load_sinan, prepare_matrix


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_csv(content: str) -> str:
    """Write CSV content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".csv", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# ── load_sinan ───────────────────────────────────────────────────────────


class TestLoadSinan:
    """Tests for load_sinan()."""

    def test_basic_load(self):
        """Load a simple CSV and verify output columns."""
        csv_content = (
            "DT_SIN_PRI,DT_NOTIFIC,OTHER\n"
            "2024-01-01,2024-01-03,X\n"
            "2024-01-02,2024-01-04,Y\n"
            "2024-01-03,2024-01-05,Z\n"
            "2024-01-05,2024-01-06,W\n"
        )
        path = _write_csv(csv_content)
        try:
            df = load_sinan(path, max_delay=30)
            assert isinstance(df, pd.DataFrame)
            assert list(df.columns) == ["onset", "notif", "delay"]
            assert len(df) == 4
            assert df["delay"].iloc[0] == 2  # Jan 3 - Jan 1
            assert df["delay"].iloc[1] == 2
        finally:
            os.remove(path)

    def test_custom_column_names(self):
        """Use custom column names for onset/notif."""
        csv_content = (
            "DATA_INICIO,DATA_NOTIF\n"
            "2024-01-01,2024-01-02\n"
            "2024-01-10,2024-01-12\n"
        )
        path = _write_csv(csv_content)
        try:
            df = load_sinan(path, date_onset="DATA_INICIO", date_notif="DATA_NOTIF")
            assert len(df) == 2
            assert df["delay"].iloc[0] == 1
        finally:
            os.remove(path)

    def test_missing_columns_raises(self):
        """Missing required columns should raise ValueError."""
        csv_content = "COL_A,COL_B\n1,2\n3,4\n"
        path = _write_csv(csv_content)
        try:
            with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
                load_sinan(path)
        finally:
            os.remove(path)

    def test_no_valid_rows_raises(self):
        """No valid records after filtering should raise ValueError."""
        csv_content = (
            "DT_SIN_PRI,DT_NOTIFIC\n"
            "2024-01-01,2024-02-15\n"  # delay = 45 > max_delay=10
        )
        path = _write_csv(csv_content)
        try:
            with pytest.raises(ValueError, match="Nenhum registro válido"):
                load_sinan(path, max_delay=10)
        finally:
            os.remove(path)

    def test_min_year_filter(self):
        """min_year should remove records before that year."""
        csv_content = (
            "DT_SIN_PRI,DT_NOTIFIC\n"
            "2023-12-31,2024-01-02\n"
            "2024-01-01,2024-01-03\n"
            "2024-06-15,2024-06-17\n"
        )
        path = _write_csv(csv_content)
        try:
            df = load_sinan(path, min_year=2024)
            assert len(df) == 2
            assert all(df["onset"].dt.year >= 2024)
        finally:
            os.remove(path)

    def test_negative_delay_filtered(self):
        """Cases with negative delay should be removed."""
        csv_content = (
            "DT_SIN_PRI,DT_NOTIFIC\n"
            "2024-01-10,2024-01-05\n"  # negative delay
            "2024-01-01,2024-01-03\n"  # valid
        )
        path = _write_csv(csv_content)
        try:
            df = load_sinan(path)
            assert len(df) == 1
            assert df["delay"].iloc[0] == 2
        finally:
            os.remove(path)

    def test_invalid_dates_coerced(self):
        """Invalid date strings should be coerced to NaT and removed."""
        csv_content = (
            "DT_SIN_PRI,DT_NOTIFIC\n"
            "NOT_A_DATE,2024-01-03\n"
            "2024-01-01,2024-01-04\n"
        )
        path = _write_csv(csv_content)
        try:
            df = load_sinan(path)
            assert len(df) == 1
            assert df["delay"].iloc[0] == 3
        finally:
            os.remove(path)

    def test_dayfirst_format(self):
        """Dates in DD/MM/YYYY format should be parsed correctly.

        Uses day > 12 in month position to force dayfirst=False to produce
        NaT, triggering the dayfirst=True fallback path.
        """
        csv_content = (
            "DT_SIN_PRI,DT_NOTIFIC\n"
            "13/01/2024,14/01/2024\n"  # delay=1, needs dayfirst=True
            "15/01/2024,17/01/2024\n"  # delay=2, needs dayfirst=True
        )
        path = _write_csv(csv_content)
        try:
            df = load_sinan(path)
            assert len(df) == 2
            # Jan 13 → Jan 14 = 1 day
            assert df["delay"].iloc[0] == 1
            # Jan 15 → Jan 17 = 2 days
            assert df["delay"].iloc[1] == 2
        finally:
            os.remove(path)


# ── prepare_matrix ───────────────────────────────────────────────────────


class TestPrepareMatrix:
    """Tests for prepare_matrix()."""

    def test_output_shapes(self):
        """Check shapes of all returned arrays."""
        df = pd.DataFrame({
            "onset": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-05",
            ]),
            "delay": [0, 1, 2, 0],
        })
        nmat, dates, obs_t, obs_d, counts = prepare_matrix(df, max_delay=5)
        # 5 days (Jan 1-5), 6 delays (0-5)
        assert nmat.shape == (5, 6)
        assert len(dates) == 5
        assert obs_t.dtype == np.int64 or obs_t.dtype == np.int32
        assert obs_d.dtype == np.int64 or obs_d.dtype == np.int32
        assert counts.dtype == np.int64 or counts.dtype == np.int32

    def test_matrix_values(self):
        """Verify correct counts are placed in the matrix."""
        df = pd.DataFrame({
            "onset": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-01-02",
            ]),
            "delay": [0, 1, 2],
        })
        nmat, dates, _, _, _ = prepare_matrix(df, max_delay=5)
        # Jan 1 has 2 cases (delay 0 and 1)
        assert nmat[0, 0] == 1  # delay 0
        assert nmat[0, 1] == 1  # delay 1
        # Jan 2 has 1 case (delay 2)
        assert nmat[1, 2] == 1
        # Delay 5 should exist (6th column)
        assert nmat.shape[1] == 6

    def test_delay_clipped_to_max(self):
        """Delays > max_delay should be clipped."""
        df = pd.DataFrame({
            "onset": pd.to_datetime(["2024-01-01"]),
            "delay": [50],
        })
        nmat, _, _, _, _ = prepare_matrix(df, max_delay=10)
        # delay 50 → clipped to 10
        assert nmat[0, 10] == 1

    def test_sparse_outputs_have_correct_counts(self):
        """obs_t, obs_d, counts should correspond to non-zero cells."""
        df = pd.DataFrame({
            "onset": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02",
            ]),
            "delay": [0, 0, 1, 2],
        })
        nmat, _, obs_t, obs_d, counts = prepare_matrix(df, max_delay=5)
        # Two cases on Jan 1 delay 0 → count should be 2 for that cell
        t0_mask = (obs_t == 0) & (obs_d == 0)
        assert counts[t0_mask][0] == 2

    def test_single_day(self):
        """Handle data with only one unique onset date."""
        df = pd.DataFrame({
            "onset": pd.to_datetime(["2024-06-01", "2024-06-01"]),
            "delay": [0, 3],
        })
        nmat, dates, _, _, _ = prepare_matrix(df, max_delay=10)
        assert nmat.shape == (1, 11)
        assert nmat[0, 0] == 1
        assert nmat[0, 3] == 1
