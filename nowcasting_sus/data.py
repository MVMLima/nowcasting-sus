"""Carga e preparação de dados SINAN para nowcasting."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = ["load_sinan", "prepare_matrix"]


def load_sinan(
    filepath: str,
    date_onset: str = "DT_SIN_PRI",
    date_notif: str = "DT_NOTIFIC",
    max_delay: int = 30,
    min_year: Optional[int] = None,
) -> pd.DataFrame:
    """Carrega e valida dados SINAN para nowcasting.

    Parameters
    ----------
    filepath : str
        Caminho do CSV com dados do SINAN.
    date_onset : str
        Nome da coluna com data de início dos sintomas (default: DT_SIN_PRI).
    date_notif : str
        Nome da coluna com data de notificação (default: DT_NOTIFIC).
    max_delay : int
        Atraso máximo em dias para filtrar (default: 30).
    min_year : int, optional
        Ano mínimo para filtrar (útil para remover registros antigos).

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas: 'onset', 'notif', 'delay'.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    ValueError
        Se colunas obrigatórias estiverem ausentes.
    """
    df = pd.read_csv(filepath, low_memory=False)

    required = [date_onset, date_notif]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    df = df[[date_onset, date_notif]].copy()
    df.columns = ["onset", "notif"]

    # Converter datas — tenta formato automático
    for col in ["onset", "notif"]:
        df[col] = pd.to_datetime(df[col], dayfirst=False, errors="coerce")

    # Se dayfirst=False falhou, tenta dayfirst=True
    if df["onset"].isna().sum() > len(df) * 0.5:
        for col in ["onset", "notif"]:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Remover nulos
    before = len(df)
    df = df.dropna(subset=["onset", "notif"])
    if len(df) < before:
        print(f"[aviso] {before - len(df)} registros removidos por data inválida")

    # Calcular delay em dias
    df["delay"] = (df["notif"] - df["onset"]).dt.days

    # Filtrar delays válidos
    df = df[(df["delay"] >= 0) & (df["delay"] <= max_delay)].copy()
    df = df.sort_values("onset").reset_index(drop=True)

    if min_year:
        df = df[df["onset"].dt.year >= min_year]

    if len(df) == 0:
        raise ValueError("Nenhum registro válido após filtragem")

    print(f"[ok] {len(df)} registros carregados "
          f"({df['onset'].min().date()} a {df['onset'].max().date()})")
    return df


def prepare_matrix(
    df: pd.DataFrame,
    max_delay: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepara matriz onset × delay para o modelo nowcasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas 'onset' e 'delay' (saída de load_sinan).
    max_delay : int
        Atraso máximo (default: 30).

    Returns
    -------
    n_matrix : np.ndarray (T × D)
        Matriz de contagem: n[t, d] = casos com onset no dia t e delay d.
    dates : np.ndarray
        Array de datas correspondentes a cada índice t.
    obs_t : np.ndarray
        Índices t das observações (para o modelo PyMC).
    obs_d : np.ndarray
        Índices d das observações.
    counts : np.ndarray
        Contagens observadas.
    """
    # Criar índice de dias únicos de onset
    date_range = pd.date_range(
        start=df["onset"].min(),
        end=df["onset"].max(),
        freq="D",
    )
    dates = date_range.values
    T = len(dates)
    D = max_delay + 1

    date_to_idx = {d.date(): i for i, d in enumerate(date_range)}

    # Montar índice (t, d) para cada caso
    df = df.copy()
    df["t"] = df["onset"].dt.date.map(date_to_idx)
    df["d"] = df["delay"].clip(0, max_delay)

    # Agregar contagens
    agg = df.groupby(["t", "d"], observed=True).size().reset_index(name="count")

    # Matriz densa T × D
    n_matrix = np.zeros((T, D), dtype=int)
    n_matrix[agg["t"].values, agg["d"].values] = agg["count"].values

    # Vetores esparsos para o modelo
    obs_t = agg["t"].values.astype(int)
    obs_d = agg["d"].values.astype(int)
    counts = agg["count"].values.astype(int)

    total = counts.sum()
    print(f"[ok] Matriz {T} dias × {D} delays ({total:,} casos)")

    return n_matrix, dates, obs_t, obs_d, counts
