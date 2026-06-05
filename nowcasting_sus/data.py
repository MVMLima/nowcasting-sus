"""Carga e preparação de dados SINAN para nowcasting."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd


__all__ = ["load_sinan", "prepare_matrix"]

logger = logging.getLogger("nowcasting_sus.data")


def _validate_types(
    filepath: str,
    date_onset: str,
    date_notif: str,
    max_delay: int,
    min_year: int | None,
) -> None:
    """Valida tipos e ranges dos parâmetros de ``load_sinan``.

    Raises
    ------
    TypeError
        Se algum parâmetro tiver tipo inesperado.
    ValueError
        Se algum parâmetro estiver fora do range válido.
    """
    if not isinstance(filepath, str):
        raise TypeError(
            f"filepath deve ser str, recebeu {type(filepath).__name__}: {filepath!r}"
        )
    if not isinstance(date_onset, str):
        raise TypeError(
            f"date_onset deve ser str, recebeu {type(date_onset).__name__}: {date_onset!r}"
        )
    if not isinstance(date_notif, str):
        raise TypeError(
            f"date_notif deve ser str, recebeu {type(date_notif).__name__}: {date_notif!r}"
        )
    if not isinstance(max_delay, (int, np.integer)):
        raise TypeError(
            f"max_delay deve ser int, recebeu {type(max_delay).__name__}: {max_delay!r}"
        )
    if max_delay < 1 or max_delay > 365:
        raise ValueError(
            f"max_delay deve estar entre 1 e 365, recebeu {max_delay}"
        )
    if min_year is not None:
        if not isinstance(min_year, (int, np.integer)):
            raise TypeError(
                f"min_year deve ser int | None, recebeu {type(min_year).__name__}: {min_year!r}"
            )
        if min_year < 1900 or min_year > 2100:
            raise ValueError(
                f"min_year deve estar entre 1900 e 2100, recebeu {min_year}"
            )


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
        Nome da coluna com data de início dos sintomas (default: ``DT_SIN_PRI``).
    date_notif : str
        Nome da coluna com data de notificação (default: ``DT_NOTIFIC``).
    max_delay : int
        Atraso máximo em dias para filtrar (default: 30). Deve estar entre 1 e 365.
    min_year : int, optional
        Ano mínimo para filtrar (útil para remover registros antigos).
        Deve estar entre 1900 e 2100.

    Returns
    -------
    pd.DataFrame
        DataFrame com colunas ``onset``, ``notif``, ``delay``.

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    ValueError
        Se colunas obrigatórias estiverem ausentes, ou se nenhum registro
        for válido após filtragem.
    TypeError
        Se tipos dos parâmetros estiverem incorretos.
    """
    _validate_types(filepath, date_onset, date_notif, max_delay, min_year)

    logger.info("Carregando dados: %s", filepath)

    try:
        df = pd.read_csv(filepath, low_memory=False, encoding="utf-8")
    except FileNotFoundError:
        raise
    except UnicodeDecodeError:
        logger.warning("UTF-8 falhou, tentando encoding latin1...")
        df = pd.read_csv(filepath, low_memory=False, encoding="latin1")
    except Exception as exc:
        raise ValueError(
            f"Não foi possível ler o arquivo {filepath!r}: {exc}"
        ) from exc

    required = [date_onset, date_notif]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes no arquivo {filepath!r}: {missing}. "
            f"Colunas disponíveis: {list(df.columns[:20])}..."
        )

    df = df[[date_onset, date_notif]].copy()
    df.columns = ["onset", "notif"]

    # Converter datas — tenta formato automático
    for col in ["onset", "notif"]:
        df[col] = pd.to_datetime(df[col], dayfirst=False, errors="coerce")

    # Se dayfirst=False falhou, tenta dayfirst=True
    first_col = "onset"
    if df[first_col].isna().sum() > len(df) * 0.5:
        logger.info("dayfirst=False produziu muitos NaT, tentando dayfirst=True...")
        for col in ["onset", "notif"]:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

    # Verificar se há datas mínimas plausíveis
    valid_dates = df["onset"].dropna()
    if len(valid_dates) > 0:
        min_date = valid_dates.min()
        if min_date < pd.Timestamp("1900-01-01"):
            logger.warning(
                "Datas muito antigas detectadas (mín: %s). Verifique o formato das datas.",
                min_date,
            )
        if min_date > pd.Timestamp("2000-01-01") and len(valid_dates) < len(df) * 0.1:
            logger.warning(
                "Menos de 10%% das datas de onset são posteriores a 2000. "
                "Possível erro de formato ou encoding."
            )

    # Remover nulos
    before = len(df)
    df = df.dropna(subset=["onset", "notif"])
    n_invalid = before - len(df)
    if n_invalid > 0:
        ratio = n_invalid / before * 100
        logger.warning(
            "%d registros removidos por data inválida (%.1f%% do total)",
            n_invalid, ratio,
        )
        if ratio > 50:
            logger.warning(
                "Mais de 50%% dos registros têm datas inválidas. "
                "Verifique o formato das colunas %r e %r.",
                date_onset, date_notif,
            )

    # Calcular delay em dias
    df["delay"] = (df["notif"] - df["onset"]).dt.days

    # Filtrar delays válidos
    n_before_filter = len(df)
    df = df[(df["delay"] >= 0) & (df["delay"] <= max_delay)].copy()

    n_removed = n_before_filter - len(df)
    if n_removed > 0:
        logger.info(
            "%d registros removidos por delay fora do intervalo [0, %d]",
            n_removed, max_delay,
        )

    df = df.sort_values("onset").reset_index(drop=True)

    if min_year is not None:
        before_year = len(df)
        df = df[df["onset"].dt.year >= min_year]
        removed = before_year - len(df)
        if removed > 0:
            logger.info(
                "%d registros removidos por ano < %d", removed, min_year
            )

    if len(df) == 0:
        raise ValueError(
            "Nenhum registro válido após filtragem. "
            f"Verifique: max_delay={max_delay}, min_year={min_year}, "
            f"colunas {date_onset!r}/{date_notif!r}"
        )

    logger.info(
        "%d registros carregados (%s a %s)",
        len(df),
        df["onset"].min().date(),
        df["onset"].max().date(),
    )
    return df


def prepare_matrix(
    df: pd.DataFrame,
    max_delay: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepara matriz onset × delay para o modelo nowcasting.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame com colunas ``onset`` e ``delay`` (saída de :func:`load_sinan`).
    max_delay : int
        Atraso máximo (default: 30). Deve estar entre 1 e 365.

    Returns
    -------
    n_matrix : np.ndarray (T × D)
        Matriz de contagem: ``n[t, d]`` = casos com onset no dia **t** e delay **d**.
    dates : np.ndarray
        Array de datas correspondentes a cada índice **t**.
    obs_t : np.ndarray
        Índices **t** das observações (para o modelo PyMC).
    obs_d : np.ndarray
        Índices **d** das observações.
    counts : np.ndarray
        Contagens observadas.

    Raises
    ------
    TypeError
        Se **df** não for ``pd.DataFrame``.
    ValueError
        Se colunas obrigatórias estiverem ausentes ou dados forem inválidos.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"df deve ser pd.DataFrame, recebeu {type(df).__name__}"
        )
    if not isinstance(max_delay, (int, np.integer)):
        raise TypeError(
            f"max_delay deve ser int, recebeu {type(max_delay).__name__}: {max_delay!r}"
        )
    if max_delay < 1 or max_delay > 365:
        raise ValueError(
            f"max_delay deve estar entre 1 e 365, recebeu {max_delay}"
        )

    required_cols = ["onset", "delay"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes no DataFrame: {missing}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    if len(df) == 0:
        raise ValueError("DataFrame vazio — não é possível preparar a matriz.")

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

    # Garantir que todos os onset dates estão no índice
    unlinked = df["t"].isna().sum()
    if unlinked > 0:
        logger.warning(
            "%d registros não puderam ser mapeados para a grade temporal",
            unlinked,
        )
        df = df.dropna(subset=["t"])

    df["d"] = df["delay"].clip(0, max_delay).astype(int)
    df["t"] = df["t"].astype(int)

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
    logger.info(
        "Matriz %d dias × %d delays (%s casos)",
        T, D, _fmt_num(total),
    )

    return n_matrix, dates, obs_t, obs_d, counts


def _fmt_num(n: int) -> str:
    """Formata número inteiro com separador de milhar (ponto)."""
    return f"{n:,}"
